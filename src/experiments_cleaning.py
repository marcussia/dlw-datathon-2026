"""Imputation variants (a/b/c) + one-at-a-time feature tests.
Protocol: RepeatedStratifiedKFold 3x5 seed 42, per-fold AP, paired deltas.
Robustness: blanks injected into validation rows at the TEST's per-column rates.
Test-like slice: top 20% of train rows by adversarial P(test), OOF per repeat.
"""
import numpy as np, pandas as pd, lightgbm as lgb, json, sys, time
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import average_precision_score as AP, roc_auc_score

tr = pd.read_csv("data/raw/track2/train.csv")
te = pd.read_csv("data/raw/track2/test.csv")
y = tr.fraud
SEED = 42

MERCHANTS = ["cash_transfer", "electronics", "entertainment", "food_delivery", "gaming",
             "grocery", "luxury", "retail", "transport", "travel", "utilities"]
COUNTRIES = ["AU", "GB", "ID", "JP", "MY", "PH", "SG", "TH", "US", "VN"]
CHANNELS  = ["bank_transfer", "card_present", "ecommerce", "mobile_app"]
BASE_FEATURES = ["transaction_amount", "transaction_hour", "merchant_category", "country",
                 "transaction_channel", "transactions_last_24h", "spend_last_24h",
                 "account_age", "new_device", "transactions_last_1h"]
BLANK_COLS = ["account_age", "country", "merchant_category", "new_device",
              "spend_last_24h", "transaction_channel", "transactions_last_1h", "transactions_last_24h"]
TEST_BLANK = te[BLANK_COLS].isna().mean()
ND_RATE = round(float(tr.new_device.mean()), 4)
print("ND_RATE (train new_device prevalence):", ND_RATE)
print("test blank rates:", dict(TEST_BLANK.round(4)))

NUM_FILL = {"transactions_last_24h": 4.0, "spend_last_24h": 217.515, "account_age": 790.0,
            "transactions_last_1h": 1.0}

def make_cleaner(variant):
    unknown = variant in ("b", "c")
    def cleaner(df):
        out = df.copy()
        for c, v in NUM_FILL.items():
            out[c] = out[c].fillna(v)
        if variant == "a":
            out["new_device"] = out["new_device"].fillna(0.0)
            out["merchant_category"] = out["merchant_category"].fillna("grocery")
            out["country"] = out["country"].fillna("SG")
            out["transaction_channel"] = out["transaction_channel"].fillna("card_present")
        else:  # neutral fills
            out["new_device"] = out["new_device"].fillna(ND_RATE)
            out["merchant_category"] = out["merchant_category"].fillna("unknown")
            out["country"] = out["country"].fillna("SG")
            out["transaction_channel"] = out["transaction_channel"].fillna("card_present")
        vocab = {"merchant_category": MERCHANTS + (["unknown"] if unknown else []),
                 "country": COUNTRIES, "transaction_channel": CHANNELS}
        for c, v in vocab.items():
            out[c] = pd.Categorical(out[c], categories=v)
        return out
    return cleaner

def lgbm():
    return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=15,
                              random_state=SEED, verbose=-1)

def mask_fraction(df, rng):
    out = df.copy()
    for c in ("merchant_category", "new_device"):
        out.loc[rng.random(len(out)) < TEST_BLANK[c], c] = np.nan
    return out

# Holed validation copy: inject blanks at per-column TEST rates (fixed seed, built once).
rng = np.random.default_rng(SEED)
holed = tr.copy()
for c in BLANK_COLS:
    holed.loc[rng.random(len(holed)) < TEST_BLANK[c], c] = np.nan

# ---- Adversarial P(test) for every train row (3-fold OOF) + base adversarial AUC ----
def adversarial(extra=None):
    """extra: fn(cleaned_df) -> DataFrame of added columns, applied to both sets."""
    ca = make_cleaner("a")
    parts = []
    for df in (tr, te):
        d = ca(df)[BASE_FEATURES].copy()
        if extra is not None:
            for k, v in extra(ca(df)).items():
                d[k] = v
        parts.append(d)
    adv = pd.concat(parts, ignore_index=True)
    is_test = pd.Series(np.r_[np.zeros(len(tr)), np.ones(len(te))])
    p = np.zeros(len(adv)); aucs = []
    for tri, vai in StratifiedKFold(3, shuffle=True, random_state=SEED).split(adv, is_test):
        m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=15,
                               random_state=SEED, verbose=-1).fit(adv.iloc[tri], is_test.iloc[tri])
        p[vai] = m.predict_proba(adv.iloc[vai])[:, 1]
        aucs.append(roc_auc_score(is_test.iloc[vai], p[vai]))
    return float(np.mean(aucs)), p[:len(tr)]

ADV_AUC_BASE, p_test_like = adversarial()
thr = np.quantile(p_test_like, 0.80)
SLICE = p_test_like >= thr
print(f"\nadversarial AUC (base features): {ADV_AUC_BASE:.3f}")
print(f"test-like slice: {SLICE.sum()} rows, {int(y[SLICE].sum())} frauds ({y[SLICE].mean():.2%})")

RSKF = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED).split(tr, y))

def run(cleaner, mask_train=False, extra=None, feats=None):
    """Returns per-fold APs (clean + holed) and per-repeat slice APs."""
    feats = feats or BASE_FEATURES
    def fx(df):
        c = cleaner(df)
        X = c[BASE_FEATURES].copy()
        if extra is not None:
            for k, v in extra(c).items():
                X[k] = v
        return X[feats] if feats != BASE_FEATURES or extra is None else X
    aps, aps_h = [], []
    oof = np.zeros((3, len(y)))
    for i, (tri, vai) in enumerate(RSKF):
        dtr = tr.iloc[tri]
        if mask_train:
            dtr = mask_fraction(dtr, np.random.default_rng(1000 + i))
        m = lgbm().fit(fx(dtr), y.iloc[tri])
        p = m.predict_proba(fx(tr.iloc[vai]))[:, 1]
        oof[i // 5, vai] = p
        aps.append(AP(y.iloc[vai], p))
        aps_h.append(AP(y.iloc[vai], m.predict_proba(fx(holed.iloc[vai]))[:, 1]))
    slice_aps = [AP(y[SLICE], oof[r][SLICE]) for r in range(3)]
    return np.array(aps), np.array(aps_h), np.array(slice_aps), oof

def fmt(a): return f"{a.mean():.4f}±{a.std():.4f}"

# ================= PART 1: imputation variants =================
print("\n===== imputation variants (3x5 CV, per-fold AP) =====")
variants = {}
for name, (v, mk) in {"a_constants": ("a", False), "b_neutral": ("b", False), "c_neutral+mask": ("c", True)}.items():
    t0 = time.time()
    aps, aps_h, sl, oof = run(make_cleaner(v), mask_train=mk)
    variants[name] = (aps, aps_h, sl, oof)
    print(f"{name:16s} PR-AUC {fmt(aps)} | +blanks {fmt(aps_h)} | test-like {fmt(sl)} ({time.time()-t0:.0f}s)")

json.dump({k: [v[0].tolist(), v[1].tolist(), v[2].tolist()] for k, v in variants.items()},
          open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/variants.json", "w"))

# ================= PART 2: feature tests (one at a time, on variant a) =================
if "--features" in sys.argv:
    CAND = {
        "is_night":          lambda c: {"is_night": c.transaction_hour.isin(range(0, 6)).astype(int)},
        "log_amount":        lambda c: {"log_amount": np.log1p(c.transaction_amount)},
        "amount_cents":      lambda c: {"amount_cents": (c.transaction_amount * 100 % 100).round()},
        "amount_per_spend24":lambda c: {"amount_per_spend24": c.transaction_amount / (c.spend_last_24h + 1)},
        "amount_per_age":    lambda c: {"amount_per_age": c.transaction_amount / (c.account_age + 1)},
        "burst_ratio":       lambda c: {"burst_ratio": c.transactions_last_1h / c.transactions_last_24h},
        "new_device_young":  lambda c: {"new_device_young": (c.new_device.astype(float) * (c.account_age < 180)).astype(float)},
        "spend_per_txn":     lambda c: {"spend_per_txn": c.spend_last_24h / c.transactions_last_24h},
    }
    ca = make_cleaner("a")
    base_aps, base_h, base_sl, _ = run(ca)
    print(f"\nBASE (variant a): PR-AUC {fmt(base_aps)} | +blanks {fmt(base_h)} | test-like {fmt(base_sl)}")
    print(f"{'feature':20s} {'ΔPR-AUC paired':>18s} {'Δtest-like':>12s} {'advAUC':>7s} gate")
    results = {}
    for name, fn in CAND.items():
        aps, aps_h, sl, _ = run(ca, extra=fn)
        adv_auc, _ = adversarial(extra=fn)
        d, ds = aps - base_aps, sl - base_sl
        keep = (d.mean() > d.std()) and (ds.mean() >= -ds.std()) and (adv_auc < ADV_AUC_BASE + 0.01)
        results[name] = dict(d=d.tolist(), dsl=ds.tolist(), adv=adv_auc, keep=bool(keep),
                             dh=(aps_h - base_h).tolist())
        print(f"{name:20s} {d.mean():+.4f} ± {d.std():.4f} {ds.mean():+12.4f} {adv_auc:7.3f} {'KEEP' if keep else 'drop'}")
    json.dump(results, open("/private/tmp/claude-501/-Users-marcusmac-Projects-dlw-datathon-2026/93ec4358-3ecc-4722-891e-6cb2651b4a04/scratchpad/feat_results.json", "w"))
    # combined kept set
    kept = [k for k, r in results.items() if r["keep"]]
    if kept:
        combo = lambda c: {k: v for f in kept for k, v in CAND[f](c).items()}
        aps, aps_h, sl, _ = run(ca, extra=combo)
        adv_auc, _ = adversarial(extra=combo)
        print(f"\nCOMBINED {kept}: PR-AUC {fmt(aps)} (Δ {(aps-base_aps).mean():+.4f} ± {(aps-base_aps).std():.4f}) "
              f"| +blanks {fmt(aps_h)} | test-like {fmt(sl)} (Δ {(sl-base_sl).mean():+.4f}) | advAUC {adv_auc:.3f}")

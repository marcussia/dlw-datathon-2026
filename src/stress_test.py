"""Simulated 'private test sets': perturb the REAL validation rows the way the organisers hinted
("different observations and edge cases") and measure how each model's ranking (PR-AUC) degrades,
and whether anything crashes. This is a stress test, not training data: labels stay real, so the
numbers mean something; nothing synthetic is ever fitted.

Scenarios (applied to raw validation rows before make_features):
  clean            as-is
  blanks_5pct      5% blanks per column (test has ~2.2%; private may have more)
  blanks_10pct     10% blanks per column
  amount_x3        amounts scaled ×3 (bigger purchases, like the public test)
  amount_x10       amounts scaled ×10 (beyond the training cap for many rows)
  old_accounts     account_age ×2 (older customer base)
  young_accounts   account_age ×0.5
  bursty           transactions_last_1h +2, transactions_last_24h +5 (busier customers)
  new_device_x2    new_device flipped on for an extra 10% of rows at random
  unseen_cats      5% of rows get an unseen merchant/country/channel
  garbage          1% of rows get malformed values (negative amount, hour 99, counts 100, strings)
  all_together     blanks 5% + amount ×3 + old accounts + unseen cats + garbage

Usage: .venv-image/bin/python src/stress_test.py"""
import sys, os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, brier_score_loss

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src import models as M  # noqa
warnings.filterwarnings("ignore")

train = pd.read_csv(os.path.join(ROOT, "data/raw/track2/train.csv")); y = train.fraud.values
BLANKABLE = ["merchant_category", "country", "transaction_channel", "transactions_last_24h", "spend_last_24h",
             "account_age", "new_device", "transactions_last_1h"]


def perturb(df, scenario, rng):
    d = df.copy()
    def blanks(rate):
        for c in BLANKABLE:
            d.loc[rng.random(len(d)) < rate, c] = np.nan
    def unseen(rate):
        m = rng.random(len(d)) < rate
        d.loc[m, "merchant_category"] = "crypto"; d.loc[m, "country"] = "FR"; d.loc[m, "transaction_channel"] = "atm"
    def garbage(rate):
        idx = d.index[rng.random(len(d)) < rate]
        k = len(idx) // 5 or 1
        d.loc[idx[:k], "transaction_amount"] = -1.0
        d.loc[idx[k:2*k], "transaction_hour"] = 99
        d.loc[idx[2*k:3*k], "transactions_last_24h"] = 100
        d.loc[idx[3*k:4*k], "account_age"] = 50000
        d["new_device"] = d["new_device"].astype(object); d.loc[idx[4*k:], "new_device"] = "yes"
    if scenario == "clean": pass
    elif scenario == "blanks_5pct": blanks(0.05)
    elif scenario == "blanks_10pct": blanks(0.10)
    elif scenario == "amount_x3": d["transaction_amount"] *= 3
    elif scenario == "amount_x10": d["transaction_amount"] *= 10
    elif scenario == "old_accounts": d["account_age"] *= 2
    elif scenario == "young_accounts": d["account_age"] = (d["account_age"] * 0.5).round()
    elif scenario == "bursty": d["transactions_last_1h"] += 2; d["transactions_last_24h"] += 5
    elif scenario == "new_device_x2":
        d.loc[rng.random(len(d)) < 0.10, "new_device"] = 1.0
    elif scenario == "unseen_cats": unseen(0.05)
    elif scenario == "garbage": garbage(0.01)
    elif scenario == "all_together":
        blanks(0.05); d["transaction_amount"] *= 3; d["account_age"] *= 2; unseen(0.05); garbage(0.01)
    return d


SCENARIOS = ["clean", "blanks_5pct", "blanks_10pct", "amount_x3", "amount_x10", "old_accounts", "young_accounts",
             "bursty", "new_device_x2", "unseen_cats", "garbage", "all_together"]
MODELS = {"LR (shipped)": M.CANDIDATES["logreg"], "LR+LGBM blend": M.CANDIDATES["blend_lr_lgbm"], "LightGBM": M.CANDIDATES["lgbm"]}


def main():
    rng = np.random.default_rng(2026)
    X = M.make_features(train)
    cv = list(StratifiedKFold(5, shuffle=True, random_state=2026).split(X, y))
    fitted = {name: [fn().fit(X.iloc[tr], y[tr]) for tr, _ in cv] for name, fn in MODELS.items()}
    rows = []
    for sc in SCENARIOS:
        pert = perturb(train, sc, np.random.default_rng(7))
        Xp = M.make_features(pert)
        row = {"scenario": sc}
        for name, ms in fitted.items():
            oof = np.zeros(len(y))
            for (tr, va), m in zip(cv, ms):
                oof[va] = m.predict_proba(Xp.iloc[va])[:, 1]
            row[name] = average_precision_score(y, oof)
            if name == "LR (shipped)": row["LR mean p"] = oof.mean()
        rows.append(row)
    df = pd.DataFrame(rows).set_index("scenario")
    base = df.loc["clean"]
    out = df.copy()
    for name in MODELS: out[name + " Δ"] = (df[name] - base[name]).round(4)
    pd.set_option("display.width", 220)
    print(out.round(4).to_string())
    print("\nNo scenario crashed. Δ = change in PR-AUC vs the clean validation rows (same real labels).")
    return out


if __name__ == "__main__":
    main()

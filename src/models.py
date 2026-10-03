"""Model candidates + comparison harness. Owner: Germaine.

Everything the submission notebook (§4 features, §6 model comparison, §9 export) needs from
the modelling side, written so Marcus can inline it verbatim:

  * `make_features(df)`  – works on the raw test csv alone (no id, no train stats at predict
    time); blank handling = the team §3 policy (same constants as src/preprocess.py), then 19
    engineered features on top of the 10 raw columns.
  * `CANDIDATES`         – zero-argument factories; every model must be pickle-loadable under
    `requirements-image.txt` (sklearn 1.5.2, lightgbm 4.5.0, xgboost 2.1.3, NO catboost) with
    no custom classes. The final export is a plain `VotingClassifier(voting="soft")`.
  * `compare(...)`       – repeated stratified CV on identical folds (5×5 = 25 fits per model;
    with 353 positives a single 5-fold is too noisy to rank models), plus the paired
    fold-by-fold test so we only switch model when the gain is real.

Validation notes: scores are mean over 5 repeats. "test-like" PR-AUC reweights validation rows
by how much they resemble test.csv (adversarial classifier, AUC 0.665) – a stand-in for the
shifted private test. "blanks" PR-AUC scores validation rows with extra blanks injected
(test has ~4× more missing values than train).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score,
                             precision_recall_curve, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
N_SPLITS, N_REPEATS = 5, 5

# --------------------------------------------------------------------------- features
# Vocabularies and fill values are frozen from train.csv so make_features is deterministic
# and needs nothing but the test file at prediction time.
CATS = {
    "merchant_category": ["grocery", "retail", "transport", "food_delivery", "cash_transfer", "electronics",
                          "utilities", "entertainment", "gaming", "travel", "luxury"],
    "country": ["SG", "MY", "ID", "AU", "US", "JP", "TH", "PH", "VN", "GB"],
    "transaction_channel": ["card_present", "ecommerce", "mobile_app", "bank_transfer"],
}
# Blank handling = the team's §3 policy (src/preprocess.py, same constants, keep in sync): every blank
# gets a hard-coded TRAIN median/mode, no missing-indicator flags. In train, blank merchant_category /
# new_device rows have 0 frauds (113 / 124 rows), a data-generation artefact; imputing camouflages blank
# rows among typical ones so the model can't learn "missing => safe" (test has ~4x more blanks).
# We tested "neutral" fills (unknown level, new_device = prevalence) for our final model: identical
# PR-AUC (0.2311 vs 0.2306, p = 0.75), so we keep one policy for the whole notebook.
IMPUTE = {"transactions_last_24h": 4.0, "spend_last_24h": 217.515, "account_age": 790.0,
          "new_device": 0.0, "transactions_last_1h": 1.0,
          "merchant_category": "grocery", "country": "SG", "transaction_channel": "card_present"}

NUMS = ["log_amount", "log_age", "log_spend", "transactions_last_1h", "transactions_last_24h", "transaction_hour",
        "amt_share_24h", "amt_vs_prev_avg", "log_amt_per_age", "burst_ratio",
        "log_amt_x_newdev", "log_amt_x_night", "hour_sin", "hour_cos", "spend_per_txn"]
FLAGS = ["new_device", "night", "amt_gt500", "age_lt180", "n1_ge3", "n1_ge4", "n24_ge10", "newdev_young", "big_old",
         "night_newdev", "burst_young"]
FEATURES = list(CATS) + NUMS + FLAGS


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering; must run on the raw test csv with no other inputs (never uses `id`)."""
    d = df.copy()
    for c, v in IMPUTE.items():                       # blanks -> train constants (team §3 policy)
        d[c] = (pd.to_numeric(d[c], errors="coerce") if c not in CATS else d[c]).fillna(v)
    for c in CATS:                                    # unseen category (never in our data) -> "unknown": no crash,
        d[c] = d[c].where(d[c].isin(CATS[c]), "unknown")   # all-zero one-hot, never a KeyError in the sandbox
    a, s, age = d.transaction_amount.astype(float), d.spend_last_24h, d.account_age
    n1, n24, nd, hr = d.transactions_last_1h, d.transactions_last_24h, d.new_device, d.transaction_hour
    X = d[list(CATS)].copy()
    X["log_amount"] = np.log1p(a)
    X["log_age"] = np.log1p(age)
    X["log_spend"] = np.log1p(s)
    X["transactions_last_1h"] = n1
    X["transactions_last_24h"] = n24
    X["transaction_hour"] = hr
    X["amt_share_24h"] = a / (a + s)                                   # this txn vs the day's spend
    X["amt_vs_prev_avg"] = np.log1p(a) - np.log1p(s / np.maximum(n24 - 1, 1))  # vs avg previous txn
    X["log_amt_per_age"] = np.log1p(a) - np.log1p(age)                 # big amount on a young account
    X["burst_ratio"] = n1 / np.maximum(n24, 1)
    X["new_device"] = nd
    X["night"] = (hr <= 5).astype(float)
    X["amt_gt500"] = (a > 500).astype(float)                            # fraud 5–6% above 500 vs ~1% below
    X["age_lt180"] = (age < 180).astype(float)                          # 3.6–4.5% under 180 days
    X["n1_ge3"] = (n1 >= 3).astype(float)                               # 3.3% at 3, 10% at 4, 26% at 5–6
    X["n1_ge4"] = (n1 >= 4).astype(float)
    X["n24_ge10"] = (n24 >= 10).astype(float)                           # 13–24% above 10/day
    X["newdev_young"] = nd * (age < 180)                                # 17% in train
    X["big_old"] = ((a > 500) & (age > 1000)).astype(float)             # loyal big spender: NOT fraud (1.7%)
    X["log_amt_x_newdev"] = X["log_amount"] * nd
    X["log_amt_x_night"] = X["log_amount"] * X["night"]
    X["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    X["spend_per_txn"] = np.log1p(s / np.maximum(n24, 1))
    X["night_newdev"] = X["night"] * nd
    X["burst_young"] = X["n1_ge3"] * (age < 365).astype(float)
    return X[FEATURES]


def to_codes(X: pd.DataFrame) -> pd.DataFrame:
    """Categorical columns -> integer codes (unknown = 0) for tree models that want numerics."""
    X = X.copy()
    for c, cats in CATS.items():
        X[c] = X[c].map({k: i + 1 for i, k in enumerate(cats)}).fillna(0).astype(int)
    return X


# --------------------------------------------------------------------------- candidates
def _lr_pre():
    return ColumnTransformer([
        ("cat", OneHotEncoder(categories=[CATS[c] for c in CATS], handle_unknown="ignore"), list(CATS)),
        ("num", StandardScaler(), NUMS),
        ("flag", "passthrough", FLAGS)])


def logreg(C=0.2):
    """Interpretable reference and, as it turns out, a top model. Assumes additive log-odds (the
    interactions are supplied as explicit features). No class weights: they inflate every
    probability and wreck calibration (Brier 0.023 vs 0.015 in our tests)."""
    return lambda: Pipeline([("pre", _lr_pre()), ("clf", LogisticRegression(C=C, max_iter=5000))])


def _cat_tf():
    """Codes for categoricals inside a sklearn pipeline, so tree models accept the string columns."""
    from sklearn.preprocessing import FunctionTransformer
    return FunctionTransformer(to_codes, feature_names_out="one-to-one")


def lgbm(seed=RANDOM_STATE, **kw):
    """LightGBM: shallow, slow, heavily regularised – 353 positives can't support deep trees.
    Assumes nothing about functional form; prone to overfit, so leaves/depth are kept small."""
    params = dict(n_estimators=600, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=80,
                  reg_lambda=10.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=seed, verbose=-1)
    params.update(kw)
    def make():
        import lightgbm as lgb  # lazy: module still loads where lightgbm/libomp is missing
        # one-hot categoricals: beat integer codes (+0.002) and LightGBM's native categorical
        # splits (+0.004) in our 5x5 CV; with 353 positives the native per-category stats overfit.
        pre = ColumnTransformer([("cat", OneHotEncoder(categories=[CATS[c] for c in CATS], handle_unknown="ignore"), list(CATS))],
                                remainder="passthrough")
        return Pipeline([("pre", pre), ("clf", lgb.LGBMClassifier(**params))])
    return make


def xgb(seed=RANDOM_STATE, **kw):
    params = dict(n_estimators=600, learning_rate=0.03, max_depth=3, min_child_weight=20, reg_lambda=10.0,
                  subsample=0.8, colsample_bytree=0.8, random_state=seed, tree_method="hist", n_jobs=4)
    params.update(kw)
    def make():
        import xgboost as xgb_
        return Pipeline([("codes", _cat_tf()), ("clf", xgb_.XGBClassifier(**params))])
    return make


def hgb(**kw):
    """sklearn's own boosting – the pure-sklearn fallback if lightgbm were unavailable."""
    params = dict(max_depth=2, learning_rate=0.05, max_iter=300, min_samples_leaf=80, l2_regularization=1.0,
                  random_state=RANDOM_STATE)
    params.update(kw)
    return lambda: Pipeline([("codes", _cat_tf()), ("clf", HistGradientBoostingClassifier(**params))])


def blend(*members, weights=None):
    """Soft-voting average of probabilities. Standard sklearn class -> unpickles in the sandbox.
    A fixed 50/50 LR + booster blend beat every tuned/stacked alternative we tried."""
    return lambda: VotingClassifier([(f"m{i}", f()) for i, f in enumerate(members)], voting="soft", weights=weights)


CANDIDATES = {
    "logreg": logreg(),
    "lgbm": lgbm(),
    "xgb": xgb(),
    "hgb": hgb(),
    "blend_lr_lgbm": blend(logreg(), lgbm()),
    "blend_lr_xgb": blend(logreg(), xgb()),
    "blend_lr_hgb": blend(logreg(), hgb()),
}


# --------------------------------------------------------------------------- harness
def oof_predict(model_fn, X, y, X_alt=None, cv=None):
    """Out-of-fold probabilities for each repeat: array (n_repeats, n). If X_alt is given (e.g. a
    blanked copy of X), it is scored by the same fold models and returned as a second array."""
    cv = cv or RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=2026)
    oof = np.zeros((N_REPEATS, len(y))); alt = np.zeros_like(oof)
    for i, (tr, va) in enumerate(cv.split(X, y)):
        r = i // N_SPLITS
        m = model_fn().fit(X.iloc[tr], y.iloc[tr])
        oof[r, va] = m.predict_proba(X.iloc[va])[:, 1]
        if X_alt is not None:
            alt[r, va] = m.predict_proba(X_alt.iloc[va])[:, 1]
    return (oof, alt) if X_alt is not None else oof


def evaluate(y, oof, name="", oof_blanks=None, weights=None, flag_rate=0.02):
    """Mean over repeats. F1/Recall at a fixed 2% flag rate (≈ fraud rate) so models are comparable
    without tuning a threshold on validation data."""
    y = np.asarray(y); rows = []
    for r in range(oof.shape[0]):
        p = oof[r]; lab = (p >= np.quantile(p, 1 - flag_rate)).astype(int)
        row = {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
               "Brier": brier_score_loss(y, p), "F1@2%": f1_score(y, lab), "Recall@2%": recall_score(y, lab)}
        if weights is not None: row["PR-AUC test-like"] = average_precision_score(y, p, sample_weight=weights)
        if oof_blanks is not None: row["PR-AUC blanks"] = average_precision_score(y, oof_blanks[r])
        rows.append(row)
    s = pd.DataFrame(rows).mean().round(4); s["model"] = name
    return s.to_dict()


def paired_test(y, oof_a, oof_b, cv=None):
    """Fold-by-fold PR-AUC difference a−b with the Nadeau–Bengio corrected t-test (CV folds
    overlap, so the naive standard error is too small). Returns (mean diff, wins, p-value)."""
    from scipy import stats
    y = np.asarray(y); cv = cv or RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=2026)
    d = []
    for i, (_, va) in enumerate(cv.split(np.zeros(len(y)), y)):
        r = i // N_SPLITS
        d.append(average_precision_score(y[va], oof_a[r, va]) - average_precision_score(y[va], oof_b[r, va]))
    d = np.array(d); J = len(d)
    se = np.sqrt((1 / J + 1 / (N_SPLITS - 1)) * d.var(ddof=1))
    t = d.mean() / se if se > 0 else 0.0
    return float(d.mean()), int((d > 0).sum()), float(2 * stats.t.sf(abs(t), J - 1))


def compare(X, y, candidates=None, X_blanks=None, weights=None) -> pd.DataFrame:
    """OOF-compare all candidates on identical repeated folds; sorted by PR-AUC."""
    candidates = candidates or CANDIDATES
    rows = []
    for name, fn in candidates.items():
        if X_blanks is not None:
            oof, alt = oof_predict(fn, X, y, X_alt=X_blanks)
        else:
            oof, alt = oof_predict(fn, X, y), None
        rows.append(evaluate(y, oof, name, oof_blanks=alt, weights=weights))
    return pd.DataFrame(rows).set_index("model").sort_values("PR-AUC", ascending=False)


# --------------------------------------------------------------------------- explainability & fairness
_FRIENDLY = {
    "new_device": "new device", "night": "night-time (0–5h)", "amt_gt500": "amount > 500", "age_lt180": "account < 180 days",
    "n1_ge3": "≥3 transactions in the last hour", "n1_ge4": "≥4 transactions in the last hour", "n24_ge10": "≥10 transactions today",
    "newdev_young": "new device on a young account", "big_old": "large purchase on a long-standing account",
    "night_newdev": "new device at night", "burst_young": "burst of activity on a young account",
    "log_amount": "transaction amount", "log_age": "account age", "log_spend": "spend in last 24h",
    "transactions_last_1h": "transactions in last hour", "transactions_last_24h": "transactions today",
    "transaction_hour": "hour of day", "amt_share_24h": "amount vs today's spend", "amt_vs_prev_avg": "amount vs usual transaction",
    "log_amt_per_age": "amount relative to account age", "burst_ratio": "share of today's activity in the last hour",
    "log_amt_x_newdev": "amount × new device", "log_amt_x_night": "amount × night", "hour_sin": "time of day", "hour_cos": "time of day",
    "spend_per_txn": "average transaction today",
}


def _lr_member(model):
    """The fitted logistic-regression pipeline inside a VotingClassifier (or the pipeline itself)."""
    if hasattr(model, "estimators_"):
        for est in model.estimators_:
            if hasattr(est, "named_steps") and "clf" in est.named_steps and isinstance(est.named_steps["clf"], LogisticRegression):
                return est
    return model


def reason_codes(model, X: pd.DataFrame, top=3) -> pd.DataFrame:
    """Top-k reasons per transaction from the logistic-regression member: contribution = coefficient ×
    standardised feature value (log-odds units). Exact for LR, so a flag can always be explained to a
    customer or analyst without SHAP. Returns one row per transaction with p_fraud and reasons."""
    lr = _lr_member(model)
    pre, clf = lr.named_steps["pre"], lr.named_steps["clf"]
    Z = pre.transform(X); Z = Z.toarray() if hasattr(Z, "toarray") else Z
    names = pre.get_feature_names_out()
    contrib = Z * clf.coef_[0]
    p = model.predict_proba(X)[:, 1]
    rows = []
    for i in range(len(X)):
        order = np.argsort(-contrib[i])[:top]
        reasons = []
        for j in order:
            if contrib[i, j] <= 0: break
            raw = names[j].split("__", 1)[1]
            if raw.startswith(tuple(CATS)):           # one-hot column like country_ID
                col, val = raw.rsplit("_", 1) if raw.rsplit("_", 1)[0] in CATS else (raw, "")
                label = f"{col.replace('_', ' ')} = {val}" if val else raw
            else:
                label = _FRIENDLY.get(raw, raw)
            reasons.append(f"{label} (+{contrib[i, j]:.2f})")
        rows.append({"p_fraud": round(float(p[i]), 4), "reasons": "; ".join(reasons)})
    return pd.DataFrame(rows, index=X.index)


def fairness_by_group(y, p, group: pd.Series, threshold: float) -> pd.DataFrame:
    """MAS FEAT-style check: per group, flag rate, false-positive rate (legit customers disrupted),
    recall and actual fraud rate at a given operating threshold. Large FPR gaps are what a bank must
    justify or mitigate (FEAT principles 1–3)."""
    y = np.asarray(y); p = np.asarray(p); flag = p >= threshold
    out = []
    for g, idx in pd.Series(range(len(y))).groupby(group.values):
        idx = idx.values; yy, ff = y[idx], flag[idx]
        legit = yy == 0
        out.append({"group": g, "n": len(idx), "actual fraud %": 100 * yy.mean(), "flag %": 100 * ff.mean(),
                    "FPR % (legit flagged)": 100 * ff[legit].mean() if legit.any() else np.nan,
                    "recall": ff[yy == 1].mean() if (yy == 1).any() else np.nan})
    return pd.DataFrame(out).set_index("group").round(2).sort_values("FPR % (legit flagged)", ascending=False)

"""Model candidates + comparison harness. Owner: Germaine.

Develop candidates here and in notebooks/scratch_germaine.ipynb; the notebook owner
inlines the winners into submission.ipynb §6. Keep every candidate a zero-argument
factory so the same CV folds are reused and numbers stay comparable.

Portal constraint: the saved model's .predict must return P(fraud) — we export the
LightGBM Booster for that. If your winning model is sklearn-based, flag it: we'll
need a different export strategy (and it must unpickle under requirements-image.txt
versions with no custom classes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             precision_recall_curve, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

CAT_COLS = ["merchant_category", "country", "transaction_channel"]
NUM_COLS = ["transaction_amount", "transaction_hour", "transactions_last_24h",
            "spend_last_24h", "account_age", "new_device", "transactions_last_1h"]


def lgbm(**kw):
    """LightGBM factory; assumes NaNs and pandas categoricals are handled natively."""
    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=15,
                  random_state=RANDOM_STATE, verbose=-1)
    params.update(kw)
    return lambda: lgb.LGBMClassifier(**params)


def logreg():
    """Interpretable reference. Assumes roughly linear log-odds; needs impute+one-hot."""
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])
    return lambda: Pipeline([("pre", pre),
                             ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])


CANDIDATES = {
    "lgbm_untuned": lgbm(),
    "lgbm_weighted": lgbm(class_weight="balanced"),
    "logreg_reference": logreg(),
    # TODO(Germaine): xgboost / catboost / tuned lgbm — one line each: why + assumptions.
}


def oof_predict(model_fn, X: pd.DataFrame, y: pd.Series, cv=CV) -> np.ndarray:
    oof = np.zeros(len(y), dtype=float)
    for tr_idx, va_idx in cv.split(X, y):
        m = model_fn()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        oof[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
    return oof


def evaluate(y_true, p, name="") -> dict:
    prec, rec, thr = precision_recall_curve(y_true, p)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-9, None)
    k = int(np.nanargmax(f1s[:-1]))
    return {"model": name,
            "PR-AUC": round(average_precision_score(y_true, p), 4),
            "ROC-AUC": round(roc_auc_score(y_true, p), 4),
            "best_F1": round(float(f1s[k]), 4),
            "recall@bestF1": round(float(rec[k]), 4),
            "threshold@bestF1": round(float(thr[k]), 4),
            "Brier": round(brier_score_loss(y_true, p), 5)}


def compare(X: pd.DataFrame, y: pd.Series, candidates=None) -> pd.DataFrame:
    """OOF-compare all candidates on identical folds; sorted by PR-AUC."""
    candidates = candidates or CANDIDATES
    rows = [evaluate(y, oof_predict(fn, X, y), name) for name, fn in candidates.items()]
    return pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)

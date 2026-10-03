"""Round 5 model experiments (owner: Germaine). Sandbox-compatible candidates only
(sklearn 1.5.2 / lightgbm 4.5.0 / xgboost 2.1.3, standard classes, no custom classes in the pickle).

Protocol = src/models.py harness: 5x5 repeated stratified CV (seed 2026), PR-AUC plain / test-like
(validation rows weighted by adversarial P(test)) / with injected blanks, Brier, paired Nadeau-Bengio
test vs the shipped blend. Pre-registered rule: a challenger replaces the champion only if it wins
clearly more than half of the 25 folds with corrected p < 0.1 and does not lose on test-like or blanks.

Usage:  .venv-image/bin/python src/experiments_models.py <candidate> [<candidate> ...]
        .venv-image/bin/python src/experiments_models.py report
OOF arrays are cached in data/processed/oof_<name>.npz (gitignored)."""
from __future__ import annotations
import sys, os, warnings
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (BaggingClassifier, StackingClassifier, VotingClassifier,
                              HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src import models as M  # noqa: E402
warnings.filterwarnings("ignore")

OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)
train = pd.read_csv(os.path.join(ROOT, "data/raw/track2/train.csv"))
test = pd.read_csv(os.path.join(ROOT, "data/raw/track2/test.csv"))
y = train.fraud
X = M.make_features(train)
X_test = M.make_features(test)

# blanked copy of train at test-like rates (fixed seed), scored by the same fold models
_rng = np.random.default_rng(2026)
_bl = train.copy()
for c in ["merchant_category", "country", "transaction_channel", "transactions_last_24h", "spend_last_24h",
          "account_age", "new_device", "transactions_last_1h"]:
    _bl.loc[_rng.random(len(train)) < max(0.022 - train[c].isna().mean(), 0), c] = np.nan
X_blanks = M.make_features(_bl)


def test_likeness_weights():
    f = os.path.join(OUT, "test_likeness_weights.csv")
    if os.path.exists(f):
        return pd.read_csv(f).weight.values
    both = M.to_codes(pd.concat([X, X_test], ignore_index=True))
    is_test = np.r_[np.zeros(len(X)), np.ones(len(X_test))]
    adv = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=50, random_state=0)
    p = cross_val_predict(adv, both, is_test, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print("adversarial AUC:", round(roc_auc_score(is_test, p), 3))
    w = np.clip(p[:len(X)] / (1 - p[:len(X)]), 0.05, 20); w = w / w.mean()
    pd.DataFrame({"weight": w}).to_csv(f, index=False)
    return w


# ----------------------------------------------------------------------------- building blocks
def _onehot():
    return OneHotEncoder(categories=[M.CATS[c] for c in M.CATS], handle_unknown="ignore")


def lr_pre():
    return ColumnTransformer([("cat", _onehot(), list(M.CATS)), ("num", StandardScaler(), M.NUMS), ("flag", "passthrough", M.FLAGS)])


def lgbm_params(**kw):
    p = dict(n_estimators=600, learning_rate=0.03, num_leaves=7, max_depth=3, min_child_samples=80, reg_lambda=10.0,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
    p.update(kw); return p


def lgbm_pipe(**kw):
    import lightgbm as lgb
    pre = ColumnTransformer([("cat", _onehot(), list(M.CATS))], remainder="passthrough")
    return Pipeline([("pre", pre), ("clf", lgb.LGBMClassifier(**lgbm_params(**kw)))])


# ----------------------------------------------------------------------------- candidates
def lasso_interactions(C):
    """L1 LR on degree-2 products of standardised numerics + flags (+ one-hot cats, linear only)."""
    inter = Pipeline([("sc", StandardScaler()), ("poly", PolynomialFeatures(2, include_bias=False)), ("sc2", StandardScaler())])
    pre = ColumnTransformer([("cat", _onehot(), list(M.CATS)), ("inter", inter, M.NUMS + M.FLAGS)])
    return lambda: Pipeline([("pre", pre), ("clf", LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=5000, tol=1e-3))])


def lasso_interactions_all(C):
    """L1 LR on degree-2 products of EVERYTHING (one-hot cats included): ~700 columns, lasso keeps the few that matter."""
    pre = ColumnTransformer([("cat", _onehot(), list(M.CATS)), ("num", StandardScaler(), M.NUMS), ("flag", "passthrough", M.FLAGS)])
    return lambda: Pipeline([("pre", pre), ("poly", PolynomialFeatures(2, include_bias=False)), ("sc", StandardScaler()),
                             ("clf", LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=3000, tol=1e-3))])


def elasticnet_lr():
    return lambda: Pipeline([("pre", lr_pre()), ("clf", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, C=0.2, solver="saga", max_iter=5000, tol=1e-3))])


def bagged_lr():
    """Random-subspace bagging of LR (50 bags, 70% rows, 80% of the one-hot/scaled columns)."""
    return lambda: Pipeline([("pre", lr_pre()), ("bag", BaggingClassifier(
        LogisticRegression(C=0.2, max_iter=5000), n_estimators=50, max_samples=0.7, max_features=0.8,
        bootstrap=True, random_state=42, n_jobs=4))])


def knn_pipe(k=50):
    return Pipeline([("pre", lr_pre()), ("knn", KNeighborsClassifier(n_neighbors=k, weights="distance"))])


def stack_knn_lgbm():
    """One-hot/scale first, then a StackingClassifier on the numeric matrix: meta-LR sees the v2 features
    (passthrough) + OOF P(fraud) from kNN(50) and LightGBM. All standard sklearn/lightgbm classes."""
    import lightgbm as lgb
    def make():
        stack = StackingClassifier(
            estimators=[("knn", KNeighborsClassifier(n_neighbors=50, weights="distance")),
                        ("lgbm", lgb.LGBMClassifier(**lgbm_params()))],
            final_estimator=Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(C=0.2, max_iter=5000))]),
            cv=StratifiedKFold(5, shuffle=True, random_state=42), stack_method="predict_proba", passthrough=True, n_jobs=1)
        return Pipeline([("pre", lr_pre()), ("stack", stack)])
    return make


def lgbm_negbag():
    return lambda: lgbm_pipe(pos_bagging_fraction=1.0, neg_bagging_fraction=0.3, bagging_freq=1, subsample=1.0, subsample_freq=0)


def lgbm_dart():
    return lambda: lgbm_pipe(boosting_type="dart", n_estimators=400, learning_rate=0.08, drop_rate=0.1, skip_drop=0.5)


def lgbm_rank():
    """LambdaRank on a single query: optimises the ordering of positives directly. Scores are not
    probabilities (reported for PR-AUC only; would need Platt scaling to ship)."""
    import lightgbm as lgb
    class _Wrap:  # experiment-only wrapper (never pickled)
        def __init__(self): self.m = None; self.pre = ColumnTransformer([("cat", _onehot(), list(M.CATS))], remainder="passthrough")
        def fit(self, X, y):
            Z = self.pre.fit_transform(X)
            self.m = lgb.LGBMRanker(objective="lambdarank", n_estimators=400, learning_rate=0.03, num_leaves=7, max_depth=3,
                                    min_child_samples=80, reg_lambda=10.0, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                    random_state=42, verbose=-1, lambdarank_truncation_level=10000)
            n = len(y); g = [n // 2, n - n // 2]           # LightGBM caps a query at 10,000 rows: two arbitrary groups
            self.m.fit(Z, y, group=g); return self
        def predict_proba(self, X):
            s = self.m.predict(self.pre.transform(X)); p = 1 / (1 + np.exp(-s)); return np.c_[1 - p, p]
    return _Wrap


def blend(*fs):
    return lambda: VotingClassifier([(f"m{i}", f()) for i, f in enumerate(fs)], voting="soft")


class PseudoNeg:
    """Experiment-only: adds confident test negatives (champion P(fraud) < thr) to every training fold."""
    def __init__(self, inner, thr=0.003):
        self.inner, self.thr = inner, thr
        champ = M.CANDIDATES["blend_lr_lgbm"]().fit(X, y)
        p = champ.predict_proba(X_test)[:, 1]
        self.X_extra = X_test[p < thr]; print(f"pseudo-negatives added per fold: {len(self.X_extra)} (p < {thr})")
    def __call__(self):
        inner = self.inner(); outer = self
        class _M:
            def fit(self, Xf, yf):
                self.m = inner.fit(pd.concat([Xf, outer.X_extra]), pd.concat([yf, pd.Series(np.zeros(len(outer.X_extra), dtype=int))])); return self
            def predict_proba(self, Xq): return self.m.predict_proba(Xq)
        return _M()


CANDIDATES = {
    "champion": M.CANDIDATES["blend_lr_lgbm"],
    "lasso_int_C0.02": lasso_interactions(0.02),
    "lasso_int_C0.05": lasso_interactions(0.05),
    "lasso_int_C0.1": lasso_interactions(0.1),
    "lasso_int_all_C0.03": lasso_interactions_all(0.03),
    "elasticnet_lr": elasticnet_lr(),
    "bagged_lr": bagged_lr(),
    "knn50": lambda: knn_pipe(50),
    "stack_knn_lgbm": stack_knn_lgbm(),
    "lgbm_negbag": lgbm_negbag(),
    "lgbm_dart": lgbm_dart(),
    "lgbm_rank": lgbm_rank(),
    "blend_lr_negbag": blend(M.logreg(), lgbm_negbag()),
    "blend_lr_dart": blend(M.logreg(), lgbm_dart()),
}
LAZY = {"pseudo_neg_blend": lambda: PseudoNeg(M.CANDIDATES["blend_lr_lgbm"])}


def run(name):
    fn = CANDIDATES[name] if name in CANDIDATES else LAZY[name]()
    oof, alt = M.oof_predict(fn, X, y, X_alt=X_blanks)
    np.savez(os.path.join(OUT, f"oof_{name}.npz"), oof=oof, blanks=alt)
    w = test_likeness_weights()
    print(M.evaluate(y, oof, name, oof_blanks=alt, weights=w), flush=True)


def report():
    w = test_likeness_weights()
    champ = np.load(os.path.join(OUT, "oof_champion.npz"))["oof"]
    rows = []
    for f in sorted(os.listdir(OUT)):
        if not f.startswith("oof_"): continue
        name = f[4:-4]; z = np.load(os.path.join(OUT, f))
        r = M.evaluate(y, z["oof"], name, oof_blanks=z["blanks"], weights=w)
        d, wins, p = M.paired_test(y, z["oof"], champ)
        r.update({"Δ vs champion": round(d, 4), "wins/25": wins, "p": round(p, 3)}); rows.append(r)
    df = pd.DataFrame(rows).set_index("model").sort_values("PR-AUC", ascending=False)
    pd.set_option("display.width", 200); print(df.to_string())
    return df


if __name__ == "__main__":
    for a in sys.argv[1:]:
        report() if a == "report" else run(a)

"""Poster-ready charts (large fonts, plain labels) for the one-page poster. Numbers identical to the notebook.
Usage: .venv-image/bin/python src/poster_charts.py   -> writes docs/poster_charts/*.png at 300 dpi."""
import os, sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from src import models as M  # noqa: E402

plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 12.5})
FRAUD_C, MODEL_C, GREY = "#eb6834", "#1baf7a", "#9a9a97"
OUT = os.path.join(ROOT, "docs", "poster_charts"); os.makedirs(OUT, exist_ok=True)


def tidy(ax):
    [ax.spines[s].set_visible(False) for s in ("top", "right")]


def model_comparison():  # numbers = notebook §6.1 table (3x5 CV)
    res = pd.DataFrame({"model": ["Quick LightGBM (raw columns)", "Small LightGBM (29 features)", "Logistic regression (chosen)", "50/50 mix"],
                        "mean": [0.1952, 0.2231, 0.2341, 0.2351], "std": [0.0448, 0.0439, 0.0365, 0.0380],
                        "lift": ["11.1×", "12.6×", "13.3×", "13.3×"]})
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.barh(res.model, res["mean"], xerr=res["std"], color=[FRAUD_C if "chosen" in m else MODEL_C for m in res.model],
            capsize=5, error_kw=dict(lw=1.4, ecolor="#52514e"))
    ax.axvline(0.0176, ls="--", c="gray", lw=1.2)
    for i, (m, s, l) in enumerate(zip(res["mean"], res["std"], res["lift"])):
        ax.text(m + s + 0.006, i, f"{m:.3f} ± {s:.3f}  ({l})", va="center", fontsize=11.5)
    ax.set_xlim(0, 0.40); ax.set_xlabel("Score (PR-AUC), mean ± spread over 15 practice splits")
    ax.set_title("Four finalists on the same splits — dashed line = guessing (0.018)"); tidy(ax)
    plt.tight_layout(); plt.savefig(f"{OUT}/1_model_comparison.png", dpi=300, bbox_inches="tight"); plt.close()


def money_saved():  # numbers = notebook §7.3 policies table
    sel = pd.Series({"Our model: probability cut-off only": 78109, "Simple rule: flag everything over $500": 80200,
                     "Quick model: probability × amount": 80613, "Our model: probability × amount (chosen)": 84309}).sort_values()
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.barh(sel.index, sel.values, color=[FRAUD_C if "chosen" in n else GREY for n in sel.index])
    for i, v in enumerate(sel.values):
        ax.text(v + 800, i, f"${v:,.0f}", va="center", fontsize=12)
    ax.set_xlim(0, 100000); ax.set_xticks([0, 25000, 50000, 75000, 100000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v / 1000:.0f}k"))
    ax.set_xlabel("Money saved per 10,000 payments, at \\$5 per check"); ax.set_title("Checking by probability × amount saves the most"); tidy(ax)
    plt.tight_layout(); plt.savefig(f"{OUT}/2_money_saved.png", dpi=300, bbox_inches="tight"); plt.close()


def features_on_off(train):  # computed from the feature matrix, no model
    X = M.make_features(train); y = train.fraud
    flags = {"new_device": "New device", "amt_gt500": "Payment over $500", "n1_ge4": "4+ payments in the last hour",
             "n24_ge10": "10+ payments today", "newdev_young": "New device × young account", "burst_young": "Burst × young account"}
    d = pd.DataFrame([(lab, 100 * y[X[c] >= 0.5].mean(), 100 * y[X[c] < 0.5].mean()) for c, lab in flags.items()],
                     columns=["feature", "on", "off"]).sort_values("on")
    fig, ax = plt.subplots(figsize=(7.2, 3.6)); yy = np.arange(len(d)); h = 0.38
    ax.barh(yy - h / 2, d.off, h, color=GREY, label="feature off"); ax.barh(yy + h / 2, d.on, h, color=FRAUD_C, label="feature on")
    for i, v in enumerate(d.on):
        ax.text(v + 0.3, i + h / 2, f"{v:.1f}%", va="center", fontsize=11.5)
    ax.axvline(100 * y.mean(), ls="--", c="gray", lw=1.2); ax.set_yticks(yy); ax.set_yticklabels(d.feature); ax.set_xlim(0, 22)
    ax.set_xlabel("% of payments that are fraud (dashed line = 1.76% overall)")
    ax.set_title("Combined features point to fraud far more than single ones"); ax.legend(frameon=False, loc="lower right"); tidy(ax)
    plt.tight_layout(); plt.savefig(f"{OUT}/3_clues_on_off.png", dpi=300, bbox_inches="tight"); plt.close()


def calibration(train):  # OOF logistic regression, 5-fold, as in notebook §7.1
    X = M.make_features(train); y = train.fraud; p = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        p[b] = M.CANDIDATES["logreg"]().fit(X.iloc[a], y.iloc[a]).predict_proba(X.iloc[b])[:, 1]
    bins = pd.qcut(p, 10, duplicates="drop")
    cal = pd.DataFrame({"Model predicted": pd.Series(p).groupby(bins, observed=True).mean(),
                        "Actually happened": y.groupby(bins, observed=True).mean()}) * 100
    cal.index = [str(i) for i in range(1, len(cal) + 1)]
    ax = cal.plot(kind="bar", figsize=(7.2, 3.4), rot=0, width=0.75, color=[GREY, MODEL_C])
    ax.set_xlabel("Payments sorted into 10 risk groups, lowest (1) to highest (10)"); ax.set_ylabel("Fraud rate (%)")
    ax.set_title("Predicted vs actual fraud rate: matching bars = honest percentages"); ax.legend(frameon=False); tidy(ax)
    plt.tight_layout(); plt.savefig(f"{OUT}/4_calibration.png", dpi=300, bbox_inches="tight"); plt.close()
    print("max calibration gap (pp):", round(float((cal["Model predicted"] - cal["Actually happened"]).abs().max()), 2))


if __name__ == "__main__":
    train = pd.read_csv(os.path.join(ROOT, "data/raw/track2/train.csv"))
    which = sys.argv[1:] or ["all"]
    if "all" in which or "1" in which: model_comparison()
    if "all" in which or "2" in which: money_saved()
    if "all" in which or "3" in which: features_on_off(train)
    if "all" in which or "4" in which: calibration(train)
    print("written:", sorted(os.listdir(OUT)))

# Full source of every changed cell in `notebooks/submission_v2.ipynb`

Paste these into `submission.ipynb` at the same positions (see `final_polish_changes.md`).

---

## v2 cell 0 (markdown)

```markdown
# TrustGuard — Track 2: Financial Fraud Detection
NTU DLW Datathon 2026, Team Top 4 MCDs

**Goal:** estimate the probability that a transaction is fraudulent, and turn it into a cost-based alert policy that catches fraud without blocking honest customers.

> ### Executive summary
> - **Data:** 20,000 training transactions; **353 (1.76%, 1 in 57) are fraud** but carry **4.0% of the dollars**. 12,000 test transactions, label hidden.
> - **Model:** logistic regression on 29 engineered features, **PR-AUC 0.2341 ± 0.0365**, **13.3×** better than guessing (0.0176) and ahead of an untuned LightGBM baseline (0.1952 ± 0.0448). Fancier models did not clearly beat it (a 50/50 blend tied).
> - **Test shift:** a classifier tells training from test rows with **AUC 0.668**; test has 3.5× more blanks. We neutralised a data trap (a blank `merchant_category` or `new_device` was never fraud: 0 of 113, 0 of 124 rows; other blank columns show no such pattern, 0.9–3.3%) and scored every model on a "test-like" slice too.
> - **Value:** alert when *probability × amount* exceeds an assumed \$5 review cost → flags **17.0%** of transactions in our practice tests (about 23% on the test file, whose payments are larger; Section 10), catches **92.1% of fraud dollars**, saves **\$84,309 per 10,000 transactions** vs **\$80,200** for "flag everything over \$500".
> - **Ceiling:** about 1 in 6 frauds shows no observable signal by our definition; no model catches those (Section 9).
> - **Public leaderboard:** the shipped model scored **0.179**; our best upload, **0.184**, was a less-regularised variant we deliberately did not ship (both organiser-scored, logged in `docs/leaderboard_log.md`; Section 6.1). The gap to cross-validation (0.234) is explained by the test-set shift (Section 2.4).

**Pipeline in one line:** raw CSV → clean (fill blanks with training medians/modes) → 29 features → logistic regression → probability → alert if probability × amount > \\$5 → top-3 reason codes.

| Section | Question it answers |
|---|---|
| 1 | Why is this hard, and what does a mistake cost? |
| 2 | What is in the data, what does fraud look like, how is the test set different? |
| 3 | How do we clean the data without teaching the model a false pattern? |
| 4 | Which features do we build, and why? |
| 5 | How good is a baseline, and do hand-made features help a tree? |
| 6 | Which model wins, and which do we ship? |
| 7 | Can the probabilities be trusted, do they survive a shifted test set, what are they worth? |
| 8 | Why does the model flag a transaction, and who gets disrupted? |
| 9 | Limitations and next steps |
| 10 | How the submission files were produced (no analysis, reproducibility only) |

**Terms used in this notebook**

| Term | Meaning |
|---|---|
| **Base rate** | The share of transactions that are fraud knowing nothing else: 1.76%. |
| **PR-AUC** | The competition's score: how well frauds are ranked above legitimate transactions. 0.0176 = guessing, 1.0 = perfect; not fooled by the 98% legitimate majority. |
| **Precision / recall** | Of flagged transactions, the share that are fraud; of all frauds, the share flagged. |
| **False alarm** | A legitimate transaction that gets flagged: a blocked customer plus a review cost. |
| **Cross-validation (CV)** | Train on part of the data, score on the hidden rest, rotate, repeat ("3×5" = five parts, three repeats). We report mean ± spread across parts. |
| **Out-of-fold (OOF) probability** | The score a model gave a training row while that row was hidden from it — an honest stand-in for unseen data. |
| **Calibration / Brier score** | "10% risk" really means 1 in 10. Brier measures it; lower is better (guessing the base rate: 0.0173). |
| **Test-like slice** | The 20% of training rows a classifier finds most similar to the test set; every model is also scored there. |

```

---

## v2 cell 2 (code)

```python
# Setup — single cell, Colab-safe. Everything below uses only these imports.
import os, sys, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from IPython.display import display, Markdown
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (average_precision_score, f1_score, recall_score,
                             precision_recall_curve, brier_score_loss, roc_auc_score)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

warnings.filterwarnings("ignore", category=UserWarning)
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
pd.set_option("display.max_columns", 50)
# One colour system for every chart (meaning never changes):
#   LEGIT_C blue   = legitimate transactions / pushes the fraud score DOWN
#   FRAUD_C orange = fraud / "actually happened" fraud rate / pushes the score UP   (never used for "chosen model")
#   MODEL_C green  = a model's prediction or score;  TRAIN_C grey = training set / neutral;  TEST_C violet = test set
#   The chosen model is marked by a black outline + "shipped" label, not by colour.
#   Orange and green are adjacent slots of a colour-blind-validated palette (deuteranopia dE >= 8), so pairs stay legible.
LEGIT_C, FRAUD_C, MODEL_C, TRAIN_C, TEST_C, SHIP_EDGE = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a97", "#4a3aa7", "#0b0b0b"
print("python", sys.version.split()[0], "| pandas", pd.__version__, "| lightgbm", lgb.__version__)

```

---

## v2 cell 4 (markdown)

```markdown
## 1. Understanding the problem
A bank must stop fraud **without blocking honest customers** — the brief says "minimising unnecessary disruption to legitimate customers". That trade-off, not accuracy, is what this notebook optimises.

**Problem at a glance**

| | |
|---|---|
| Task | Score each transaction for fraud (binary classification) |
| Input | 10 columns per transaction: amount, hour, merchant, country, channel, recent activity counts and spend, account age, new-device flag |
| Output | A probability of fraud, not a yes/no label |
| Official metric | PR-AUC (ranking of frauds above legitimate transactions), plus F1 and recall at a cut-off |
| Two error types | Missed fraud (the bank loses the amount); false alarm (a blocked customer plus a review) |
| Key constraints | Fraud is 1.76% of rows; the test set is shifted from training (Section 2.4); only 353 fraud examples |

```

---

## v2 cell 6 (code)

```python
# 1.1 Rare by count, heavier by dollars - the two numbers behind "accuracy is the wrong yardstick"
LEGIT_C, FRAUD_C = "#2a78d6", "#eb6834"          # one fixed colour per class, used in every chart of this notebook
n_fraud, n_all = int(train[TARGET].sum()), len(train)
rate = n_fraud / n_all
dollars = train.groupby(TARGET)["transaction_amount"].sum()
dollar_share = dollars[1] / dollars.sum()
naive_accuracy = 1 - rate                         # a model that never flags anything

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for ax, shares, title, hero in [(axes[0], (1 - rate, rate), "Share of transactions", f"{rate:.2%}"),
                                (axes[1], (1 - dollar_share, dollar_share), "Share of dollars", f"{dollar_share:.1%}")]:
    ax.pie(shares, colors=[LEGIT_C, FRAUD_C], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.38, edgecolor="white", linewidth=2))
    ax.text(0, 0.08, hero, ha="center", va="center", fontsize=20, fontweight="bold")
    ax.text(0, -0.22, "fraud", ha="center", va="center", fontsize=11, color="#52514e")
    ax.set_title(title, fontsize=12)
fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=LEGIT_C), plt.Rectangle((0, 0), 1, 1, color=FRAUD_C)],
           labels=["legitimate", "fraud"], loc="lower center", ncol=2, frameon=False)
fig.suptitle("Fraud is rare by count but more than twice as heavy by money", fontsize=12, y=1.02)
plt.tight_layout(); plt.show()

display(Markdown(f"""
| | Transactions | Dollars |
|---|---|---|
| **Fraud** | **{n_fraud:,}** of {n_all:,} = **{rate:.2%}** (1 in {1 / rate:.0f}) | **\\${dollars[1]:,.0f}** of \\${dollars.sum():,.0f} = **{dollar_share:.1%}** |
| "Never fraud" model | accuracy **{naive_accuracy:.1%}**, frauds caught **0 of {n_fraud}** | — |
"""))
# Takeaway: 1.76% of transactions but ~4% of the money - the thin orange sliver on the left is what every 'accurate' model ignores.
```

---

## v2 cell 10 (markdown)

```markdown
**What we found**
- Average fraud **\$570.81** vs average legitimate transaction **\$246.95**: **2.3× larger** (medians 224 vs 74).
- The two distributions overlap heavily: amount alone cannot separate them.

**The four outcomes and what each costs**

| | Transaction is fraud | Transaction is legitimate |
|---|---|---|
| **Flagged** | Caught fraud: the amount is saved, minus one review | False alarm: one review cost (an assumption, see Section 7.3) and an annoyed customer |
| **Not flagged** | Missed fraud: the full amount is lost (mean \\$571) | Correct pass: no cost |

**Implications:** we report **fraud dollars caught**, not just cases, and Section 7.3 prices every alert by *probability × amount* — which requires calibrated probabilities (Section 7.1).

```

---

## v2 cell 20 (markdown)

```markdown
**What we found**
- **New device 7.5%** vs 1.2% on a known device; **account under 90 days 4.4%**; **4+ transactions in the last hour 16.1%**.
- Riskiest merchants **luxury 5.3%, cash transfer 4.7%, electronics 3.9%**; grocery and transport under 1%.

**Implications:** the classic account-takeover pattern — unfamiliar device, young account, a burst of activity, resellable goods or movable money — which is what the Section 4 features encode. A fraud showing none of these signs looks ordinary (Section 9).

```

---

## v2 cell 25 (code)

```python
# 2.5 Fraud rate when a column is blank vs when it is filled in
from scipy.stats import binomtest   # Colab-safe: scipy ships with scikit-learn
rows = []
for c in FEAT_COLS:
    miss = train[c].isna()
    if miss.any():
        n_b, k_b = int(miss.sum()), int(train.loc[miss, TARGET].sum())
        rows.append({"Column": c, "Rows where it is blank": n_b,
                     "Frauds among those rows": k_b,
                     "Expected frauds if blanks were random": round(n_b * train[TARGET].mean(), 1),
                     "p-value vs chance (two-sided binomial)": round(binomtest(k_b, n_b, train[TARGET].mean()).pvalue, 3),
                     "Fraud rate when blank (%)": round(100 * train.loc[miss, TARGET].mean(), 2),
                     "Fraud rate when filled in (%)": round(100 * train.loc[~miss, TARGET].mean(), 2)})
blanks = pd.DataFrame(rows).set_index("Column")
display(blanks)

fig, ax = plt.subplots(figsize=(9, 3.8))
y = np.arange(len(blanks)); h = 0.38
ax.barh(y - h / 2, blanks["Fraud rate when filled in (%)"], h, color="#9a9a97", label="value filled in")
ax.barh(y + h / 2, blanks["Fraud rate when blank (%)"], h, color=FRAUD_C, label="value blank")
for yi, (name, row) in zip(y, blanks.iterrows()):
    if row["Frauds among those rows"] == 0:
        ax.text(0.05, yi + h / 2, f"0 frauds in {int(row['Rows where it is blank'])} blank rows", va="center", fontsize=8, color=FRAUD_C)
ax.axvline(100 * train[TARGET].mean(), ls="--", c="gray", lw=1)
ax.set_yticks(y); ax.set_yticklabels(blanks.index); ax.invert_yaxis()
ax.set_xlabel("% of transactions that are fraud"); ax.set_title("When merchant_category or new_device is blank, fraud never happens — in the training data")
ax.legend(frameon=False, loc="upper right"); [ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()

# Is "zero frauds" more than chance? Per column no; pooled, borderline. Either way the safe choice is the same.
pooled = train.merchant_category.isna() | train.new_device.isna()
p_pooled_zero = (1 - train[TARGET].mean()) ** int(pooled.sum())
display(Markdown(f"Each column alone is within chance (smallest p-value {blanks['p-value vs chance (two-sided binomial)'].min():.2f}). "
                 f"Pooling the {int(pooled.sum())} rows where merchant_category or new_device is blank, **0 frauds has about a "
                 f"{100 * p_pooled_zero:.1f}% probability by chance**, and we examined {len(blanks)} columns, so this is suggestive, not proof."))

```

---

## v2 cell 26 (markdown)

```markdown
**What we found**
- A blank `merchant_category` is **never** fraud (**0 of 113**); a blank `new_device` is **never** fraud (**0 of 124**). About 2 frauds per column would be expected if blanks were random.
- Most other blank columns run 2–3% or more (channel 3.3%, 24h count 3.1%, spend 3.0%, 1h count 2.8%, account age 2.2%), while country is 0.9%.
- Column by column, none of this is beyond chance (all p-values above 0.17); pooling the two zero-fraud columns, 0 frauds has about a 1.5% probability, and we examined eight columns.

**Implications:** the "blank means safe" pattern may be an artefact or may be luck; we cannot tell from 235 rows. The safe choice is the same either way: fill blanks with typical values and give the model no "was blank" flag (Section 3.2), so it cannot learn the pattern if it is an artefact and loses nothing if it is luck.

```

---

## v2 cell 33 (code)

```python
# 3.2 Where the fill values sit: a blank becomes a typical value, not an extreme one
num_cols = ["transactions_last_24h", "spend_last_24h", "account_age", "transactions_last_1h", "new_device"]
cat_cols = ["merchant_category", "country", "transaction_channel"]
fig, axes = plt.subplots(2, 4, figsize=(13, 6)); axes = list(axes.flat)
for ax, col in zip(axes[:5], num_cols):
    vals = train[col].dropna(); fill = IMPUTE[col]
    if col == "new_device":
        share = vals.value_counts(normalize=True).sort_index()
        ax.bar(["0 (known)", "1 (new)"], share.values, color=TRAIN_C, edgecolor=[SHIP_EDGE if v == fill else "none" for v in share.index], linewidth=2, width=0.6)
        ax.set_ylabel("share of rows")
    else:
        bins = np.geomspace(max(vals.min(), 1), vals.max(), 35) if col in ("spend_last_24h", "account_age") else np.arange(vals.min(), vals.max() + 2) - 0.5
        ax.hist(vals, bins=bins, weights=np.full(len(vals), 1 / len(vals)), color=TRAIN_C)
        if col in ("spend_last_24h", "account_age"): ax.set_xscale("log")
        ax.axvline(fill, color=SHIP_EDGE, lw=2); ax.set_ylabel("share of rows")
    ax.set_title(f"{col}\nblank → {fill:g}", fontsize=10); [ax.spines[s].set_visible(False) for s in ("top", "right")]
for ax, col in zip(axes[5:8], cat_cols):
    share = train[col].value_counts(normalize=True)
    ax.barh(share.index[::-1], share.values[::-1], color=TRAIN_C, edgecolor=[SHIP_EDGE if k == IMPUTE[col] else "none" for k in share.index[::-1]], linewidth=2)
    ax.set_title(f"{col}\nblank → {IMPUTE[col]}", fontsize=10); ax.set_xlabel("share of rows"); ax.tick_params(axis="y", labelsize=8)
    [ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.suptitle("What a blank value becomes (black line / outline = the fill value, drawn on the training distribution)", y=1.0)
plt.tight_layout(); plt.show()

fills = pd.DataFrame({"Fill value for a blank": [IMPUTE[c] for c in num_cols + cat_cols],
                      "Why this value": ["median"] * 4 + ["most common value (0 = known device)"] + ["most common category"] * 3},
                     index=num_cols + cat_cols); fills.index.name = "Column"
display(fills)
# Takeaway: every fill value sits in the thick middle of its column's distribution, so an imputed row is close to an ordinary one (Section 7.2 measures how close).

```

---

## v2 cell 34 (markdown)

```markdown
**What we found**
- Numeric blanks become the training median (e.g. 4 transactions in 24 hours, \$217.515 spend, 790 days of account age); a blank device flag becomes 0 (known device); blank categories become the most common one (grocery, SG, card present).
- Every fill value sits in the thick middle of its column's distribution (black marks), and the assertion above confirms no blanks remain in either set.

**Implications:** an imputed row is close to an ordinary row, so the model cannot learn "blank means safe". The evidence for this is in Section 7.2: rows that were originally blank score 1.1% on average against 1.8% for all rows, slightly lower but nowhere near zero. The assumption is that blanks are recording gaps, not fraudster behaviour; if the private set's blanks genuinely signalled fraud we would lose that signal, an acceptable loss since it does not exist in training.

```

---

## v2 cell 49 (markdown)

```markdown
## 6. Compare models and choose one
We tried about 65 different models, all tested on exactly the same data splits (identical repeated folds). To tell real differences from luck, we compared them split by split (paired statistical test). The full log is in `docs/model_comparison_results.md`. The three that matter are shown here.

**Model search at a glance** (copied from `docs/model_comparison_results.md`, 5×5 cross-validation in our development environment, not recomputed here; the three finalists are re-run below on the notebook's own 3×5 folds):

| Candidate | PR-AUC | Note |
|---|---|---|
| Logistic regression + CatBoost blend | 0.238 | best overall; CatBoost is not in the organisers' sandbox, so it cannot ship |
| Logistic regression + LightGBM 50/50 (finalist) | 0.231 | |
| Logistic regression, 29 features (finalist, shipped) | 0.229 | |
| CatBoost alone | 0.228 | not shippable |
| Elastic-net logistic regression | 0.227 | |
| Bagged logistic regression (50 bags) | 0.226 | |
| LightGBM, one-hot categories (finalist) | 0.222 | |
| Stacking: kNN + LightGBM into logistic regression | 0.222 | |
| Logistic regression, 10 raw columns only | 0.213 | the engineered features' value |
| XGBoost | 0.212 | |
| Sklearn gradient boosting / random forest / EBM | 0.209 / 0.201 / 0.208 | |
| CatBoost with class weights | 0.204 | probabilities inflated (Brier 0.023) |
| LightGBM DART / LambdaRank / SVM / neural net | 0.190 / 0.191 / 0.180 / 0.177 | |

**The three finalists**
1. **Logistic regression:** a simple model that adds up each factor's contribution to the risk (additive in log-odds), held back from over-fitting (regularisation, C = 0.2). It can't spot combined effects by itself, so we built those in as extra columns in Section 4 (interaction features). It uses the 29 prepared columns (features).
   - Its probabilities tend to be trustworthy without extra fixing (well calibrated by construction).
   - It needs only one library to run (simple deployment).
   - It can show the exact reason behind every flag (per-decision explanation, see Section 8.2).
2. **Small LightGBM:** a tree model that finds patterns without being told their shape (no assumption about functional form). With only 353 positive cases it can't support big, detailed trees, so we kept it very small (depth 3, 7 leaves) and held it back strongly (strong regularisation). Turning each category into its own yes/no column (one-hot encoding) worked better than LightGBM's built-in category handling (native categorical splits).
3. **50/50 mix:** average the two models' probabilities equally (soft-voting blend). This simple even split beat every smarter way of combining them (tuned weights, stacking).

**What we rejected, and why**
- **Giving extra weight to the rare positives** (class weights, resampling): this pushes every probability upwards, and this competition scores how accurate the probabilities are (probability quality). Section 6.2 shows it on our own model: the average predicted chance jumps to about 35% against a true rate of 1.8%.
- **Smarter ways of combining models** (stacking, tuned blend weights): they scored worse than the simple 50/50 mix.
- **CatBoost:** slightly better during development, but it isn't installed in the organisers' environment (sandbox), so it couldn't be used.

```

---

## v2 cell 52 (code)

```python
# 6.1 Mean ± spread of PR-AUC across the 15 folds for each candidate (parsed from the table above; no refit)
res = pd.DataFrame(rows)
res[["mean_3x5", "std_3x5"]] = res["PR-AUC_3x5"].str.split("±", expand=True).astype(float)
res = res.sort_values("mean_3x5")
fig, ax = plt.subplots(figsize=(9, 3.6))
edges = [SHIP_EDGE if "SHIPPED" in n else "none" for n in res["model"]]
ax.barh(res["model"], res["mean_3x5"], xerr=res["std_3x5"], color=MODEL_C, edgecolor=edges, linewidth=2, capsize=4, error_kw=dict(lw=1.2, ecolor="#52514e"))
ax.axvline(BASE_RATE, ls="--", c="gray", lw=1)
for yi, (m, s, lift) in enumerate(zip(res["mean_3x5"], res["std_3x5"], res["lift_vs_base_rate"])):
    ax.text(m + s + 0.004, yi, f"{m:.4f} ± {s:.4f}  ({lift})" + ("  shipped" if "SHIPPED" in res["model"].iloc[yi] else ""), va="center", fontsize=8)
ax.set_xlim(0, res["mean_3x5"].max() + 0.12); ax.set_xlabel("PR-AUC, mean ± spread over 15 folds (3×5 CV)")
ax.set_title(f"Model comparison on identical folds — black outline = shipped, dashed line = guessing ({BASE_RATE:.4f})"); [ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()
# Takeaway: logistic regression and the blend tie; both clear LightGBM alone and the baseline; the spread bars overlap everywhere, so
# the ranking is argued on robustness and explainability, not on a decimal.

```

---

## v2 cell 55 (code)

```python
# 6.2 Same logistic regression, same 3x5 folds, with class_weight="balanced": ranking vs probability quality
def logreg_v2_weighted():
    m = logreg_v2(); m.set_params(clf=LogisticRegression(C=0.2, max_iter=5000, class_weight="balanced")); return m

aps_w, oof_w = fold_scores(logreg_v2_weighted, X, y)
aps_u, oof_u = fold_scores(logreg_v2, X, y)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)   # independent y-axes: the reweighted panel is ~20x taller
for ax, (name, p_) in zip(axes, [("shipped (no reweighting)", oof_u), ("class_weight = balanced", oof_w)]):
    groups = pd.qcut(p_, 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"predicted": pd.Series(p_).groupby(groups).mean(), "actual": y.groupby(groups).mean()})
    xs = np.arange(len(g))
    ax.bar(xs - 0.2, 100 * g["predicted"], 0.4, color=MODEL_C, label="model predicted %")
    ax.bar(xs + 0.2, 100 * g["actual"], 0.4, color=FRAUD_C, label="actually fraud %")
    ax.set_title(name); ax.set_xlabel("risk group (1 = lowest predicted risk, 10 = highest)"); ax.set_xticks(xs); ax.set_xticklabels(xs + 1)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axes[0].set_ylabel("% of transactions"); axes[1].set_ylabel("% of transactions"); axes[0].legend(frameon=False)
plt.suptitle("Reweighting inflates every probability: predicted vs actual fraud rate by risk group"); plt.tight_layout(); plt.show()

cmp = pd.DataFrame({
    "PR-AUC (3x5 mean)": [aps_u.mean(), aps_w.mean()],
    "Brier (lower = better)": [brier_score_loss(y, oof_u), brier_score_loss(y, oof_w)],
    "mean predicted probability": [oof_u.mean(), oof_w.mean()],
}, index=["logistic regression (shipped)", "same, class_weight=balanced"]).round(4)
display(cmp)
print(f"true fraud rate: {y.mean():.4f}")
# Takeaway: on the left the green and orange bars track each other; on the right the green bars tower over the orange ones
# in every group - the reweighted model calls a third of all transactions fraud-likely when fewer than 2% are.

```

---

## v2 cell 58 (markdown)

```markdown
### 7.1 Can we trust the model's percentages, and where should the alarm go?
**Charts (top):** we sort all transactions by the model's fraud estimate and split them into 10 equal groups (deciles), from lowest risk (1) to highest risk (10). Left: for each group, the green bar is the fraud rate the model *predicted* and the orange bar is the fraud rate that *actually happened*. Right: the same ten groups as points on a predicted-vs-actual plot with both axes on a log scale, so groups 1–9 are readable; honest percentages sit on the diagonal.

**Chart (bottom):** the precision–recall curve. Reading left to right, we flag more and more transactions; the curve shows, at each point, the share of flags that are real fraud (precision) against the share of all frauds caught (recall). A curve further up and to the right is better; the area under it is PR-AUC, the competition's score. Guessing is a flat line at the base rate. Two points are marked on the shipped model's curve: its best yes/no cut-off (highest F1), and the operating point of the \\$5 expected-loss rule from Section 7.3.

**Table:** what happens if we raise an alarm at different cut-offs (thresholds), from 2% to 50%.

```

---

## v2 cell 59 (code)

```python
BEST = "logreg_v2 (SHIPPED)"
p_best = oof[BEST]

# 7.1a Calibration: reliability curve + Brier (judging focus).
bins = pd.qcut(p_best, 10, duplicates="drop")
cal = pd.DataFrame({"mean_pred": pd.Series(p_best).groupby(bins, observed=True).mean(),
                    "actual_fraud": y.groupby(bins, observed=True).mean()})
cal.index = [str(i) for i in range(1, len(cal) + 1)]
cal.columns = ["Model predicted", "Actually happened"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
(cal * 100).plot(kind="bar", ax=axes[0], rot=0, width=0.75, color=[MODEL_C, FRAUD_C])
axes[0].set_xlabel("Groups of transactions, from lowest (1) to highest (10) predicted risk"); axes[0].set_ylabel("Fraud rate (%)")
axes[0].set_title("Predicted (green) vs actual (orange) fraud rate by risk group"); axes[0].legend(frameon=False)
lo, hi = 0.2, 15
axes[1].plot([lo, hi], [lo, hi], ls="--", c="gray", lw=1, label="perfect calibration (y = x)")
axes[1].scatter(cal["Model predicted"] * 100, cal["Actually happened"] * 100, color=MODEL_C, edgecolor=SHIP_EDGE, s=45, zorder=3)
for g, (px, ay) in enumerate(zip(cal["Model predicted"] * 100, cal["Actually happened"] * 100), start=1):
    axes[1].annotate(str(g), (px, ay), xytext=(4, 3), textcoords="offset points", fontsize=8)
axes[1].set_xscale("log"); axes[1].set_yscale("log"); axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi)
axes[1].set_xlabel("predicted fraud rate (%), log scale"); axes[1].set_ylabel("actual fraud rate (%), log scale")
axes[1].set_title("Reliability: each group should sit on the diagonal"); axes[1].legend(frameon=False, loc="upper left")
for ax in axes: [ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()

# 7.1b Business threshold: flag rate vs frauds caught vs fraud DOLLARS caught.
rows = []
for t in [0.02, 0.05, 0.10, 0.20, 0.30, 0.50]:
    flag = p_best >= t
    fraud_dollars = train.loc[(y == 1), "transaction_amount"]
    caught_dollars = train.loc[(y == 1) & flag, "transaction_amount"].sum() / fraud_dollars.sum()
    rows.append({"threshold": t, "%flagged": round(100 * flag.mean(), 2),
                 "recall": round(recall_score(y, flag), 3),
                 "F1": round(f1_score(y, flag), 3),
                 "%fraud_$_caught": round(100 * caught_dollars, 1),
                 "false_alarms_per_fraud_caught": round(((flag) & (y == 0)).sum() / max(1, ((flag) & (y == 1)).sum()), 1)})
pd.DataFrame(rows)

```

---

## v2 cell 60 (code)

```python
# 7.1c Precision-recall curves from pooled OOF predictions: guessing, the LightGBM baseline, the shipped LR
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.axhline(BASE_RATE, color=TRAIN_C, ls=":", lw=2, label=f"guessing (base rate): PR-AUC {BASE_RATE:.3f}")   # a random ranking has precision = base rate at every recall
for name, p_, colour, ls in [("LightGBM baseline, raw columns", oof_base, TEST_C, "-"), ("logistic regression (shipped)", p_best, MODEL_C, "-")]:
    pr, rc, _ = precision_recall_curve(y, p_)
    ax.plot(rc, pr, color=colour, ls=ls, lw=2, label=f"{name}: PR-AUC {average_precision_score(y, p_):.3f}")
pr, rc, thr = precision_recall_curve(y, p_best)
f1s = 2 * pr * rc / np.clip(pr + rc, 1e-9, None); k = int(np.nanargmax(f1s[:-1]))
ax.scatter([rc[k]], [pr[k]], color=MODEL_C, edgecolor=SHIP_EDGE, s=70, zorder=4, label=f"best F1 cut-off ({thr[k]:.3f}): precision {pr[k]:.2f}, recall {rc[k]:.2f}")
rule_flag = p_best * train.transaction_amount.to_numpy() > 5.0     # the $5 review cost set as REVIEW_COST in Section 7.3; same value
prec_rule = (rule_flag & (y.to_numpy() == 1)).sum() / rule_flag.sum(); rec_rule = (rule_flag & (y.to_numpy() == 1)).sum() / (y == 1).sum()
ax.scatter([rec_rule], [prec_rule], color=FRAUD_C, edgecolor=SHIP_EDGE, marker="D", s=70, zorder=4,
           label=f"operating point, p x amount > \\$5: precision {prec_rule:.3f}, case recall {rec_rule:.3f}")
ax.set_xlabel("recall (share of all frauds caught)"); ax.set_ylabel("precision (share of flags that are fraud)")
ax.set_title("Precision-recall curves on out-of-fold predictions"); ax.set_ylim(0, 1); ax.legend(frameon=False, fontsize=9)
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()
# Takeaway: the shipped curve sits above the baseline everywhere; the $5 rule deliberately sits far to the right (high recall of
# fraud dollars, low precision), because it is chosen on money, not on F1.

```

---

## v2 cell 61 (markdown)

```markdown
**What we found**
- **The percentages can be trusted.** In every group, predicted and actual fraud rates are close (largest gap 0.33 percentage points). Its estimates are also more accurate than a flat 1.8% guess for everyone (Brier score 0.0151 vs 0.0173).
- **The precision–recall curve** (bottom chart) shows the shipped model above the baseline at every recall; the \\$5 rule's operating point sits far to the right of the best-F1 point because it is chosen on money, not on F1.
- **There's no single "right" alarm level.** It's a trade-off:
  - **Alarm at 10%:** flags 2% of transactions and catches 29% of fraud cases (recall), but 49% of fraud *dollars*. About 3 false alarms for every real fraud caught.
  - **Alarm at 2%:** flags 18% of transactions and catches 59% of fraud cases and 84% of fraud dollars, but with 16 false alarms for every real fraud caught.

**Implications:** we can use the model's numbers as real probabilities. Picking the alarm level is a business decision, so Section 7.3 makes it in dollars.

**A caveat on F1 and recall.** Both need a yes/no cut-off, and we only submit probabilities; we don't know which cut-off the organisers apply. At our best cut-off (about 0.135) the model scores F1 0.30 and recall 0.27; at a plain 0.50 cut-off the table above shows recall 0.12 and F1 0.20. The ranking score (PR-AUC) does not depend on this choice, and it is the main metric.

```

---

## v2 cell 63 (code)

```python
# 7.2 Robustness: inject extra blanks into the RAW data (the private test has ~3.5x more),
# run them through the full clean()+make_features pipeline, and score out-of-fold.
rng = np.random.default_rng(RANDOM_STATE)
test_blank = {c: test[c].isna().mean() for c in test.columns if c != ID_COL and test[c].isna().any()}
raw_holed = train.copy()
for c, rate in test_blank.items():                 # inject at the TEST's own per-column rates
    raw_holed.loc[rng.random(len(raw_holed)) < rate, c] = np.nan
X_holed = make_features(raw_holed)

oof_holed = np.zeros(len(y), dtype=float)
for tr_idx, va_idx in CV.split(X, y):
    m = logreg_v2().fit(X.iloc[tr_idx], y.iloc[tr_idx])       # shipped model, trained on clean folds
    oof_holed[va_idx] = m.predict_proba(X_holed.iloc[va_idx])[:, 1]  # score holed validation
print("PR-AUC out-of-fold — clean:", round(average_precision_score(y, p_best), 4),
      "| with extra blanks (imputed by clean()):", round(average_precision_score(y, oof_holed), 4))

# Blank-leak neutralised? Rows whose merchant_category/new_device was blank should score like anyone else.
was_blank = train.merchant_category.isna() | train.new_device.isna()
print(f"mean OOF probability — originally-blank rows: {p_best[was_blank].mean():.4f} "
      f"| all rows: {p_best.mean():.4f}  <- should be similar, NOT near zero")

big_old = (train.transaction_amount > 2000) & (train.account_age > 1000)
print(f"big+old slice (n={big_old.sum()}, fraud {train.loc[big_old, TARGET].mean():.1%}): "
      f"flagged at t=0.10: {(p_best[big_old] >= 0.10).mean():.1%}  <- keep this LOW (legit VIPs)")

# Test-like slice (top 20% adversarial P(test), defined in 2d): the private set looks more like
# these rows, so a model that only shines on typical train rows would flatter us here.
print(f"test-like slice (n={TEST_LIKE.sum()}, fraud {y[TEST_LIKE].mean():.2%}): "
      f"PR-AUC {average_precision_score(y[TEST_LIKE], p_best[TEST_LIKE]):.4f} "
      f"(lift {average_precision_score(y[TEST_LIKE], p_best[TEST_LIKE]) / y[TEST_LIKE].mean():.1f}x its base rate)")
print(f"for comparison, pooled all-data lift: PR-AUC {average_precision_score(y, p_best):.4f} / base rate {y.mean():.4f} "
      f"= {average_precision_score(y, p_best) / y.mean():.1f}x")

```

---

## v2 cell 64 (markdown)

```markdown
**What we found**
1. **Extra blanks barely hurt.** The score dips from 0.229 to 0.222 (PR-AUC −0.007).
2. **Blanks aren't a giveaway.** Originally blank rows get an average fraud score of 1.1% vs 1.8% overall: a bit lower, but not near zero.
3. **Big spenders on old accounts are mostly left alone.** Of 348 such transactions, only 2.9% are flagged at the 10% alarm level, close to the 2% flagged overall.
4. **It works on the test-like rows too.** It scores 12.6× better than guessing there (lift), close to the 13.0× pooled lift on all data (0.2288 / 0.0176, computed above; like for like, both pooled). (The raw score is higher, 0.43, only because fraud is more common in those rows, 3.4%, so the lift is the fairer comparison.)

**Implications:** when the data gets messier, the model gets slightly worse, not suddenly broken (degrades gracefully). One honest note: the public leaderboard score (0.179, Section 6.1) is lower than our practice score (0.234), so the test-like slice over-estimated how well the model would transfer; we treat it as a robustness check, not a forecast of the test score. A bigger set of 12 stress tests is summarised in Section 9.

```

---

## v2 cell 65 (markdown)

```markdown
### 7.3 How much money does the model save?
**Simple cost model:**
- A **missed fraud** costs the full transaction amount.
- Every **alarm** costs **\$5** to check (review cost). **This is our assumption, not a figure from the data:** the competition gives no review costs. We chose \$5 to stand for a few minutes of a fraud analyst's time plus the hassle for a genuine customer of an extra check, like an OTP or a call. A bank would swap in its own number: it is one line, `REVIEW_COST`, in the next cell.
- Because we can't verify \$5, we re-run the comparison at **\$2** (a mostly automated check) and **\$15** (a manual review, or a good customer annoyed enough to leave), and only claim conclusions that hold across that whole range (sensitivity check).

**Key idea: alarm based on the expected loss, not just the probability.**
- A **\$1,000** transaction with a 20% fraud chance → expected loss **\$200**. Much more than the \$5 check → **alarm.**
- A **\$20** transaction with the same 20% chance → expected loss **\$4**. Less than \$5 → **let it through.**

So the rule is: **alarm when probability × amount is more than \$5** (expected-loss rule).

**Chart:** net money saved per 10,000 transactions for each strategy (fraud dollars caught minus checking costs). Longer bar = better; the outlined green bar is the strategy we chose.

**Tables:** the strategies side by side, then how much we'd catch if the fraud team can only check a limited number of alarms.

**Second chart:** disruption vs benefit. As we flag more transactions (x-axis), how much of the fraud money do we catch (y-axis)? One curve ranks transactions by probability × amount, one by probability alone; the diagonal is random flagging. The three rules are marked where they sit.

```

---

## v2 cell 67 (code)

```python
c = REVIEW_COST

def policy_rows(p, label):
    """The three policies for a given model's OOF probabilities, at review cost c."""
    el = p * amt
    o = np.argsort(-p); kk = np.arange(1, len(p) + 1)
    cost = (total_fraud_dollars - np.cumsum(np.where(is_fraud[o], amt[o], 0.0))) + c * kk
    t_opt = float(p[o][int(np.argmin(cost))])
    def row(name, flag):
        saved = amt[flag & is_fraud].sum() - c * flag.sum()
        return {"model": label, "policy": name, "%flagged": f"{flag.mean():.2%}",
                "fraud_cases": f"{is_fraud[flag].sum() / is_fraud.sum():.1%}",
                "fraud_$": f"{amt[flag & is_fraud].sum() / total_fraud_dollars:.1%}",
                "saved_per_10k": round(saved * 10_000 / len(train))}
    return [row(f"p-threshold optimum (t={t_opt:.3f})", p >= t_opt),
            row(f"p x amount > ${c:.0f}", el > c)]

rule = amt > 500
policies = pd.DataFrame(
    [{"model": "-", "policy": "no model", "%flagged": "0.00%", "fraud_cases": "0.0%",
      "fraud_$": "0.0%", "saved_per_10k": 0},
     {"model": "-", "policy": "rule: amount > $500", "%flagged": f"{rule.mean():.2%}",
      "fraud_cases": f"{is_fraud[rule].sum() / is_fraud.sum():.1%}",
      "fraud_$": f"{amt[rule & is_fraud].sum() / total_fraud_dollars:.1%}",
      "saved_per_10k": round((amt[rule & is_fraud].sum() - c * rule.sum()) * 10_000 / len(train))}]
    + policy_rows(oof_base, "baseline LGBM (10 raw)")
    + policy_rows(p_best, "shipped LR"))
display(policies)

sel = pd.Series({
    "Simple rule: amount > $500": policies.loc[1, "saved_per_10k"],
    "Our model: probability cut-off only": policies.loc[4, "saved_per_10k"],
    "Baseline model: probability × amount": policies.loc[3, "saved_per_10k"],
    "Our model: probability × amount (chosen)": policies.loc[5, "saved_per_10k"],
}).sort_values()
colours = [MODEL_C if "chosen" in name else TRAIN_C for name in sel.index]
ax = sel.plot.barh(figsize=(8.5, 3), color=colours, edgecolor=[SHIP_EDGE if "chosen" in name else "none" for name in sel.index], linewidth=2)
ax.set_xlabel("Money saved per 10,000 transactions, at \\$5 per check")
ax.set_title("Our model + probability × amount saves the most")
ax.bar_label(ax.containers[0], labels=[f"${v:,.0f}" for v in sel], padding=3); ax.margins(x=0.2)
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.savefig("money_saved_by_strategy.png", dpi=150, bbox_inches="tight"); plt.show()

exp_loss = p_best * amt
flag = exp_loss > c
print(f"operating policy (shipped LR): alert when p x amount > ${c:.0f}")
caught = int((flag & is_fraud).sum())
print(f"  precision {caught / flag.sum():.1%}: about 1 in {flag.sum() / caught:.0f} flagged payments is fraud "
      f"({(~is_fraud[flag]).sum() / caught:.1f} false alarms per fraud caught)")
print(f"  flags {flag.mean():.2%} | catches {is_fraud[flag].sum()/is_fraud.sum():.1%} of cases, "
      f"{amt[flag & is_fraud].sum()/total_fraud_dollars:.1%} of fraud dollars "
      f"(dollars > cases: big frauds are caught preferentially, by design)")
for cs in REVIEW_COST_SENSITIVITY:
    fl = exp_loss > cs
    sv = amt[fl & is_fraud].sum() - cs * fl.sum()
    print(f"  review cost ${cs:>4.0f}: flags {fl.mean():6.2%}, saves ${sv * 10_000 / len(train):,.0f}/10k")

# Review-capacity view: if the fraud team can only review the top X%, ranked by expected loss.
order_el = np.argsort(-exp_loss)
cum_d = np.cumsum(np.where(is_fraud[order_el], amt[order_el], 0.0))
cum_n = np.cumsum(is_fraud[order_el])
rows = []
for frac in [0.01, 0.02, 0.05]:
    kk = int(round(frac * len(amt)))
    rows.append({"review_capacity": f"top {frac:.0%} by p x amount", "alerts_per_10k": int(round(kk * 10_000 / len(train))),
                 "fraud_cases_caught": f"{cum_n[kk-1] / is_fraud.sum():.1%}",
                 "fraud_dollars_caught": f"{cum_d[kk-1] / total_fraud_dollars:.1%}"})
pd.DataFrame(rows)

```

---

## v2 cell 69 (code)

```python
# 7.3 Disruption vs benefit: share of fraud dollars caught as the flag rate rises, ranking by p x amount vs by p alone
def gains(score):
    o = np.argsort(-score); d = np.cumsum(np.where(is_fraud[o], amt[o], 0.0)) / total_fraud_dollars
    return np.arange(1, len(score) + 1) / len(score), d
x_el, y_el = gains(p_best * amt); x_p, y_p = gains(p_best)
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(100 * x_el, 100 * y_el, color=MODEL_C, lw=2, label="rank by probability x amount (ours)")
ax.plot(100 * x_p, 100 * y_p, color=TEST_C, lw=2, ls="--", label="rank by probability alone")
ax.plot([0, 100], [0, 100], color=TRAIN_C, lw=1, ls=":", label="random flagging")
marks = [("p x amount > \\$5 (ours)", p_best * amt > c, FRAUD_C, "D"),
         ("amount > \\$500 rule", amt > 500, SHIP_EDGE, "s"),
         ("baseline LGBM, p x amount > \\$5", oof_base * amt > c, TEST_C, "^")]
for name, fl, colour, mk in marks:
    ax.scatter([100 * fl.mean()], [100 * amt[fl & is_fraud].sum() / total_fraud_dollars], color=colour, marker=mk, s=80, zorder=4,
               edgecolor=SHIP_EDGE, label=f"{name}: flags {100 * fl.mean():.1f}%, catches {100 * amt[fl & is_fraud].sum() / total_fraud_dollars:.1f}% of fraud \\$")
ax.set_xlim(0, 40); ax.set_ylim(0, 100); ax.set_xlabel("% of transactions flagged (disruption)"); ax.set_ylabel("% of fraud dollars caught (benefit)")
ax.set_title("Ranking by probability × amount catches more fraud dollars than probability alone, at every flag rate"); ax.legend(frameon=False, fontsize=8, loc="lower right")
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()
# What would ranking by p x amount catch at the $500 rule's flag rate? (honest like-for-like at 10.6%)
k_500 = int((amt > 500).sum()); el_at_500 = y_el[k_500 - 1]
display(Markdown(f"At the \\$500 rule's own flag rate ({100 * (amt > 500).mean():.1f}%), ranking by probability × amount would catch "
                 f"**{100 * el_at_500:.1f}%** of fraud dollars vs the rule's {100 * amt[(amt > 500) & is_fraud].sum() / total_fraud_dollars:.1f}%; "
                 f"our \\$5 rule flags {100 * (p_best * amt > c).mean():.1f}% and catches {100 * amt[(p_best * amt > c) & is_fraud].sum() / total_fraud_dollars:.1f}%, "
                 f"and wins on net dollars at the stated review cost (table above)."))

# The four outcomes per 10,000 transactions under the chosen rule (counts from the notebook's arrays)
fl = p_best * amt > c; scale = 10_000 / len(amt)
tp, fp, fn, tn = (fl & is_fraud).sum(), (fl & ~is_fraud).sum(), (~fl & is_fraud).sum(), (~fl & ~is_fraud).sum()
display(Markdown(f"""
**Per 10,000 transactions under "probability × amount > \\${c:.0f}"** (practice data)

| | Fraud | Legitimate |
|---|---|---|
| **Flagged** | caught fraud: **{tp * scale:,.0f}** (\\${amt[fl & is_fraud].sum() * scale:,.0f} saved before review costs) | false alarms: **{fp * scale:,.0f}** (\\${c * fp * scale:,.0f} of review cost) |
| **Not flagged** | missed fraud: **{fn * scale:,.0f}** (\\${amt[~fl & is_fraud].sum() * scale:,.0f} lost) | correct passes: **{tn * scale:,.0f}** |
"""))

```

---

## v2 cell 70 (code)

```python
# Calibration check for the priced-alert rule: p x amount is an EXPECTED loss, so the policy
# is only as good as the probabilities are honest.
bins = pd.qcut(p_best, 10, duplicates="drop")
cal = pd.DataFrame({"mean_p": pd.Series(p_best).groupby(bins, observed=True).mean(),
                    "actual_fraud": y.groupby(bins, observed=True).mean()})
gap = float((cal.mean_p - cal.actual_fraud).abs().max())
print(f"shipped LR OOF: Brier {brier_score_loss(y, p_best):.5f} (naive {brier_score_loss(y, naive):.5f}) "
      f"| max decile |mean_p - actual| = {gap:.4f}")
# NOTE: if a future candidate uses class weights or resampling, recalibrate on OOF first (Platt/isotonic);
# otherwise p x amount misprices every alert. The shipped LR needs none: no weights, reliability curve ~ diagonal
# (Section 7.1), and isotonic recalibration made Brier worse in 5x5 CV runs, so it is not applied.

```

---

## v2 cell 71 (markdown)

```markdown
**What we found**
We compared strategies by **money saved per 10,000 transactions** (fraud dollars caught minus checking costs):

| Strategy | Flagged | Fraud cases caught | Fraud \$ caught | Saved per 10k |
|---|---|---|---|---|
| No model | 0% | 0% | 0% | \$0 |
| Simple rule: flag everything over \$500 | 10.6% | 34% | 85% | \$80,200 |
| Our model, probability cut-off only | 22.9% | 65% | 89% | \$78,109 |
| **Our model, probability × amount > \$5** | **17.0%** | **56%** | **92%** | **\$84,309** |

- **Probability × amount wins.** It catches 92% of fraud dollars. It catches more dollars than cases because it focuses on big frauds, which is on purpose. About **1 in 17** flagged payments is real fraud (precision 5.8%).
- **The 17% flag rate is a practice-data number.** Test payments are about twice as large, and the rule flags a payment when probability × amount is over \$5, so on the test file the same rule flags about **23%** (computed in Section 10). The \$5 cut-off is the lever a bank would adjust to its review capacity.
- **A probability cut-off alone does worse than the simple \$500 rule**, because it ignores how much money is at stake.
- **The winner doesn't depend on the \$5 guess.** At **\$2** per check our rule saves **\$91,659** vs **\$83,393** for the \$500 rule; at **\$15**, **\$70,450** vs **\$69,555**. It wins at every cost we tried, but the lead shrinks as checks get pricier (only **\$895** at \$15), so a bank with expensive reviews should re-run this with its own figure.
- **Disruption vs benefit (second chart):** at the \\$500 rule's own flag rate (10.6%) it is nearly matched by our ranking (85.7% vs 84.9% of fraud dollars, printed under the chart), so a dollar rule is a strong cheap baseline; our rule flags more (17.0%) and converts that into more fraud dollars caught and more net savings at the \\$5 review cost (figures printed under the chart).
- **If the fraud team is short-staffed** (limited review capacity) and only checks the riskiest alarms: top 1% of transactions → 49.5% of fraud dollars caught; top 5% → 75.2%.
- This rule only works if the percentages are honest, and they are: the biggest gap between predicted and actual fraud in any group is 0.0033 (confirmed in 7.1).

**Implications:** pricing each alarm by its expected loss beats both a simple dollar rule and a plain probability cut-off, by about **\$4,100** per 10,000 transactions over the \$500 rule. It's also harder for fraudsters to dodge: splitting one big fraud into many small purchases would slip under that rule, but it makes the model's "lots of activity in a short time" columns (velocity features) jump.

```

---

## v2 cell 73 (markdown)

```markdown
### 8.1 Which signals matter most?
**Chart:** the 12 signals that move the fraud score the most. Each bar is how strongly the model reacts to a signal, scaled by how much that signal varies between transactions (coefficient × standard deviation), so signals measured in different units can be compared fairly.

**How to read it:** orange bars push the fraud score **up**, blue bars push it **down**. Longer bar = stronger effect. Overlapping features (for example the night-time flag and the hour-of-day curve) split one effect between them, so the second chart shows the time-of-day effect directly: actual vs predicted fraud rate by hour.

```

---

## v2 cell 74 (code)

```python
# 8.1 Global drivers of the shipped LR (effect size on the standardised design matrix)
lr_full = logreg_v2().fit(X, y)
pre_, clf_ = lr_full.named_steps["pre"], lr_full.named_steps["clf"]
Z_ = pre_.transform(X); Z_ = Z_.toarray() if hasattr(Z_, "toarray") else Z_
FRIENDLY = {"transactions_last_24h": "transactions today", "amt_share_24h": "amount vs today's spend",
            "log_amt_x_newdev": "amount x new device", "night": "night-time (0-5h)", "amt_vs_prev_avg": "amount vs usual transaction",
            "amt_gt500": "amount > $500", "n1_ge3": ">=3 transactions in the last hour", "n1_ge4": ">=4 transactions in the last hour",
            "n24_ge10": ">=10 transactions today", "newdev_young": "new device on a young account", "big_old": "large purchase, long-standing account",
            "night_newdev": "new device at night", "burst_young": "burst on a young account", "log_amount": "transaction amount",
            "log_age": "account age", "log_spend": "spend in last 24h", "transactions_last_1h": "transactions in last hour",
            "transaction_hour": "hour of day", "log_amt_per_age": "amount relative to account age", "burst_ratio": "share of today's activity in last hour",
            "log_amt_x_night": "amount x night", "hour_sin": "time of day (sin)", "hour_cos": "time of day (cos)", "spend_per_txn": "average transaction today",
            "new_device": "new device", "age_lt180": "account < 180 days"}
def friendly(raw):
    for c in CAT_VOCAB:
        if raw.startswith(c + "_"): return f"{c.replace('_', ' ')} = {raw[len(c) + 1:]}"
    return FRIENDLY.get(raw, raw)
names_ = [n.split("__", 1)[1] for n in pre_.get_feature_names_out()]
effect = pd.Series(clf_.coef_[0] * Z_.std(axis=0), index=[friendly(n) for n in names_])
top = effect.reindex(effect.abs().sort_values(ascending=False).index)[:12]
ax = top[::-1].plot(kind="barh", figsize=(7, 4), color=np.where(top[::-1] > 0, FRAUD_C, LEGIT_C),
                    title="What pushes the fraud score up (orange) or down (blue)")
ax.set_xlabel("Effect on the fraud score (longer bar = stronger)")
plt.tight_layout(); plt.show()


# 8.1c Time of day: predicted (OOF) vs actual fraud rate by hour, so the night-time story is visible despite the overlapping features
by_hour = pd.DataFrame({"actual": y.groupby(train.transaction_hour).mean() * 100, "predicted (OOF)": pd.Series(p_best).groupby(train.transaction_hour.values).mean() * 100})
ax = by_hour.plot(figsize=(8, 3.2), color=[FRAUD_C, MODEL_C], marker="o", ms=4, title="Fraud rate by hour of day: actual (orange) vs the model's prediction (green)")
ax.set_xlabel("hour of day (0 = midnight)"); ax.set_ylabel("% of transactions that are fraud"); ax.set_xticks(range(0, 24, 2)); ax.legend(frameon=False)
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()
# Takeaway: velocity (transactions today / last hour), amount relative to the day's spend, and amount x new device raise risk most;
# card-present and everyday merchants (retail) lower it. These match the EDA story in Section 2 and the account-takeover pattern.
```

---

## v2 cell 75 (markdown)

```markdown
**What we found**
- **Biggest red flags:**
  - lots of transactions today, or several in the last hour (velocity)
  - a payment that's large compared to what the customer spent earlier today
  - a large payment from a new device
- **Biggest signs of safety:** paying with the physical card present (card-present), and everyday shops like retail.
- **Overlapping features split their effect between them** (night-time flag vs hour-of-day curve; amount vs today's spend vs amount vs usual transaction, which overlap heavily), which is why single bars can point opposite ways. The hour-of-day chart shows the model's predictions tracking the actual late-night rise in fraud.

**Implications:** the model has learned the pattern we saw in Section 2.3, where someone gets into another person's account, often from a new device, and quickly makes many payments (account takeover). It hasn't latched onto a quirk of the data (artefact). A fraud analyst would point to the same warning signs.

```

---

## v2 cell 76 (markdown)

```markdown
### 8.2 Why was this transaction flagged?
For each transaction, we list the 3 signals that pushed its fraud score up the most, and by how much (contribution in log-odds; bigger number = bigger push, compared to a typical transaction). Because logistic regression simply adds these pushes up, the reasons are exact, not an approximation, so they can be shown to a customer or a fraud analyst (reason codes).

Below: the 5 riskiest transactions in the test set, with their raw details, probability and reasons in one table.

```

---

## v2 cell 77 (code)

```python
# 8.2 Reason codes: top-3 contributions (coefficient x standardised value, log-odds units) for the riskiest test transactions
def reason_codes(model, Xq, top=3):
    pre, clf = model.named_steps["pre"], model.named_steps["clf"]
    Zq = pre.transform(Xq); Zq = Zq.toarray() if hasattr(Zq, "toarray") else Zq
    contrib = Zq * clf.coef_[0]
    cols = [friendly(n.split("__", 1)[1]) for n in pre.get_feature_names_out()]
    p = model.predict_proba(Xq)[:, 1]
    out = []
    for r in range(len(Xq)):
        order = np.argsort(-contrib[r])[:top]
        reasons = [f"{cols[j]} (+{contrib[r, j]:.2f})" for j in order if contrib[r, j] > 0]
        out.append({"p_fraud": round(float(p[r]), 3), "reasons": "; ".join(reasons)})
    return pd.DataFrame(out, index=Xq.index)

X_test_all = make_features(test)[FEATURES]
p_test = lr_full.predict_proba(X_test_all)[:, 1]
top_idx = np.argsort(-p_test)[:5]
pd.set_option("display.max_colwidth", None)
flagged = test.iloc[top_idx][["transaction_amount", "account_age", "new_device", "transactions_last_1h", "transactions_last_24h", "country", "merchant_category"]].copy()
flagged = flagged.join(reason_codes(lr_full, X_test_all.iloc[top_idx]))     # one table: raw details + probability + reasons
flagged
# Takeaway: the riskiest test transactions are mostly cash transfers (plus one luxury buy) from a new device on an account a few weeks old,
# with 4-8 transactions in the last hour - exactly the account-takeover pattern, and each flag can be explained to the customer.
```

---

## v2 cell 78 (markdown)

```markdown
**What we found**
- The 5 riskiest test transactions have an **80–85% fraud chance**. They are mostly cash transfers (plus one luxury purchase), all from a **new device**, on accounts only **45–102 days old**, with **4–8 transactions in the last hour**.
- Each one comes with plain-English reasons, for example: *"transactions today (+2.21); amount × new device (+1.41); ≥3 transactions in the last hour (+0.64)"*.

**Implications:** every alarm can be explained in one sentence. A fraud analyst needs this to decide quickly, and a customer needs it to challenge a wrong decision. MAS FEAT principles 10 and 13 expect channels to appeal AI-driven decisions and clear explanations of what data drove them.

```

---

## v2 cell 79 (markdown)

```markdown
### 8.3 Who gets bothered, and is that fair?
Using the 10% alarm level from Section 7, for each country we check:
- what share of **honest customers get wrongly flagged** (false positive rate, FPR)
- what share of **real fraud gets caught** (recall)

**Charts:** two panels in the same country order: honest customers wrongly flagged (left) and fraud caught (right). Singapore and Indonesia are outlined; the dashed line is the figure across all customers.

Then the same table under the rule we actually recommend (probability × amount > \\$5, Section 7.3), and finally we retrain the model **without the country column** to see whether we need it. This follows Singapore's financial regulator's fairness principles (MAS FEAT, principles 1–3).

```

---

## v2 cell 80 (code)

```python
# 8.3 Fairness by country at the operating threshold (MAS FEAT): who gets disrupted, and what `country` is worth
def fairness_by_group(y_true, p, group, threshold):
    y_true, p = np.asarray(y_true), np.asarray(p); flag = p >= threshold; rows = []
    for g in pd.Series(group).unique():
        m = (np.asarray(group) == g); yy, ff = y_true[m], flag[m]
        rows.append({"country": g, "n": int(m.sum()), "actual fraud %": 100 * yy.mean(), "flag %": 100 * ff.mean(),
                     "FPR % (legit flagged)": 100 * ff[yy == 0].mean(), "recall": ff[yy == 1].mean() if (yy == 1).any() else np.nan})
    return pd.DataFrame(rows).set_index("country").round(2).sort_values("FPR % (legit flagged)", ascending=False)

THRESHOLD = 0.10
fair = fairness_by_group(y, p_best, train.country.fillna("unknown"), THRESHOLD)
display(fair)
overall_fpr = 100 * (p_best[y.to_numpy() == 0] >= THRESHOLD).mean()
overall_rec = (p_best[y.to_numpy() == 1] >= THRESHOLD).mean()
fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
edges = [SHIP_EDGE if c_ in ("SG", "ID") else "none" for c_ in fair.index]
axes[0].bar(fair.index, fair["FPR % (legit flagged)"], color=TRAIN_C, edgecolor=edges, linewidth=2)
axes[0].axhline(overall_fpr, ls="--", c="gray", lw=1, label=f"all customers: {overall_fpr:.2f}%"); axes[0].legend(frameon=False, loc="upper right")
axes[0].set_ylabel("Honest customers wrongly flagged (%)"); axes[0].set_title(f"Wrongly flagged at the {THRESHOLD:.0%} alarm level")
axes[1].bar(fair.index, fair["recall"], color=MODEL_C, edgecolor=edges, linewidth=2)
axes[1].axhline(overall_rec, ls="--", c="gray", lw=1, label=f"all customers: {overall_rec:.2f}"); axes[1].legend(frameon=False, loc="upper right")
axes[1].set_ylabel("Share of real fraud caught (recall)"); axes[1].set_title("Fraud caught, same country order")
for ax in axes: ax.set_xlabel(""); [ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.suptitle("Honest customers wrongly flagged, and fraud caught, by country (outlined: Singapore and Indonesia)", y=1.02); plt.tight_layout(); plt.show()

# Same check under the RECOMMENDED rule (p x amount > REVIEW_COST), since that is the policy we propose, not a flat 10% cut-off
rule_flag_ = (p_best * amt > REVIEW_COST)
def fairness_by_rule(flag, group):
    rows_ = []
    for g in pd.Series(group).unique():
        m_ = (np.asarray(group) == g); yy_, ff_ = y.to_numpy()[m_], flag[m_]
        rows_.append({"country": g, "n": int(m_.sum()), "flag %": 100 * ff_.mean(), "FPR % (legit flagged)": 100 * ff_[yy_ == 0].mean(),
                      "recall (cases)": ff_[yy_ == 1].mean() if (yy_ == 1).any() else np.nan,
                      "fraud $ caught %": 100 * amt[m_][ff_ & (yy_ == 1)].sum() / max(amt[m_][yy_ == 1].sum(), 1)})
    return pd.DataFrame(rows_).set_index("country").round(2).sort_values("FPR % (legit flagged)", ascending=False)
fair_rule = fairness_by_rule(rule_flag_, train.country.fillna("unknown"))
display(fair_rule)
gap_rule = fair_rule.loc["ID", "FPR % (legit flagged)"] / fair_rule.loc["SG", "FPR % (legit flagged)"]
display(Markdown(f"Under the recommended rule, honest customers are wrongly flagged **{fair_rule.loc['ID', 'FPR % (legit flagged)']:.1f}% in ID vs "
                 f"{fair_rule.loc['SG', 'FPR % (legit flagged)']:.1f}% in SG** ({gap_rule:.1f}× gap; {fair.loc['ID', 'FPR % (legit flagged)'] / fair.loc['SG', 'FPR % (legit flagged)']:.1f}× at the flat 10% cut-off), "
                 f"and the rule catches {fair_rule.loc['SG', 'recall (cases)']:.0%} of SG fraud cases ({fair_rule.loc['SG', 'fraud $ caught %']:.0f}% of SG fraud dollars)."))
print(f"SG holds {int(y[train.country == 'SG'].sum())} of the {int(y.sum())} fraud cases in the training data "
      f"({y[train.country == 'SG'].sum() / y.sum():.0%}) and {(train.country == 'SG').mean():.0%} of all rows")

# What does `country` buy us? Same model without it, same 3x5 folds.
NUMS_NC, FLAGS_NC = NUMS, FLAGS
def logreg_no_country():
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(categories=[CAT_VOCAB[c] for c in ("merchant_category", "transaction_channel")], handle_unknown="ignore"),
         ["merchant_category", "transaction_channel"]),
        ("num", StandardScaler(), NUMS_NC), ("flag", "passthrough", FLAGS_NC)])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.2, max_iter=5000))])
aps_nc, _ = fold_scores(logreg_no_country, X, y)
aps_lr, _ = fold_scores(logreg_v2, X, y)
print(f"PR-AUC with country: {aps_lr.mean():.4f} +/- {aps_lr.std():.4f} | without country: {aps_nc.mean():.4f} +/- {aps_nc.std():.4f} "
      f"| cost of dropping it: {aps_lr.mean() - aps_nc.mean():+.4f}")
# Takeaway: a legitimate ID customer is ~6x and an AU customer ~5x more likely to be flagged than a SG one, and the model catches only 9% of SG fraud
# (SG fraud looks like normal SG behaviour), even though SG holds over half of the fraud cases in the training data. Dropping `country` costs nothing
# in cross-validation (within ±0.002 PR-AUC) but cost ~0.013 on the public leaderboard, so it is a trade-off, not free. Monitor this table at every
# retrain (FEAT P1-P3).

```

---

## v2 cell 81 (markdown)

```markdown
**What we found**
- **Honest customers in some countries get flagged far more often:** 5.5% in Indonesia (ID) and 4.1% in Australia (AU), vs 0.85% in Singapore (SG), a 6.4× gap. Part of this is expected, because fraud really is more common in those countries (4.1% in ID vs 1.3% in SG). But the flagging gap (6.4×) is about twice as big as the fraud gap (about 3×).
- **The model is weakest in Singapore.** It catches only 9% of Singapore fraud (recall 0.09), vs 38–71% elsewhere. Singapore fraud looks like normal Singapore behaviour, so there's little for the model to spot. This matters a lot: most transactions in the training data are from Singapore, so **182 of the 353 fraud cases** in the training data are Singaporean, more than half.
- **Under the rule we actually recommend** (probability × amount > \$5) the gap narrows but does not vanish: honest customers are wrongly flagged 24.4% in Indonesia vs 13.9% in Singapore, a 1.8× gap (6.4× at the flat cut-off), and Japan is highest at 34.4% (table and line above). Singapore recall rises to 39% of cases and 83% of fraud dollars, because the rule weights every alert by amount.
- **Removing country doesn't hurt in our tests:** score 0.2352 ± 0.0339 without it vs 0.2341 ± 0.0365 with it.

**Implications:** the false alarms fall unevenly across countries, and the country column doesn't improve our cross-validation score. So a version without it (fairness-constrained variant) is a real option. But on the competition's public test, country was worth about 0.013, so dropping it is a trade-off, not free. Whichever version is used, this table should be checked every time the model is retrained (monitoring).

```

---

## v2 cell 82 (markdown)

```markdown
## 9. Limitations and next steps
**Little fraud to learn from.** 353 examples; scores swing ±0.04 between folds, so we report means over repeated cross-validation and never switch models on one number. No customer ID or timestamp, so per-customer baselines — the strongest real-world fraud features — are impossible here.

**Frauds with no signal.** By our definition (known device, account older than 365 days, amount under 300, at most 2 transactions in the last hour) a fraud looks like any other transaction. The cell below counts how many training frauds fit it.

```

---

## v2 cell 84 (markdown)

```markdown
By this definition these frauds are indistinguishable from ordinary transactions on the columns we have, so **no model catches them**; they set a hard ceiling on recall. Most are domestic, which is why Singapore recall is only 0.09 (Section 8.3).

**Imputation can under-score blank rows.** The fill values (grocery, known device, SG, card present) are the most common values but also low-risk ones, so an imputed row can score slightly below its true risk (1.1% vs 1.8%, Section 7.2). This matters more with 3.5× more blanks in the test set; the injected-blank check in Section 7.2 bounds the cost at about 0.007 PR-AUC. We kept the imputation because the alternative, a "was blank" flag, would hand the model the artefact in Section 2.5.

**Fairness.** At a flat 10% cut-off legitimate ID and AU customers are flagged 6.4× more often than SG ones; under the recommended expected-loss rule the same gap narrows to 1.8×, and the widest gap (Japan vs Singapore) is 2.5× (Section 8.3). Dropping `country` costs nothing in cross-validation. A bank should keep the fairness-constrained variant as a live option and monitor Section 8.3 at every retrain.

**The test set is different from the training set.** We measured it (Section 2.4) and the public leaderboard confirms it: every model scores lower there than in cross-validation (ours 0.179 vs 0.234), and the private set may differ again. Our score there could therefore be lower than our own estimates. We planned for this rather than chased it: blanks imputed, numbers clipped to training ranges, every model also scored on a test-like slice and under injected blanks (Section 7.2), a 12-scenario stress test (`src/stress_test.py`, table below) that crashes nothing, and no tuning to the public leaderboard, where we treat ranks within ±0.03 as noise.

**Next:** customer history and timestamps, scheduled retraining, calibration monitoring ("does 30% still mean 30%?"), the fairness-constrained deployment option.

**The 12 stress scenarios** (copied from the stress-test output in `docs/model_comparison_results.md`, not recomputed here; PR-AUC of the shipped model on perturbed validation rows, real labels, 5-fold):

| Scenario | PR-AUC | Change |
|---|---|---|
| clean | 0.2325 | — |
| 5% blanks per column | 0.2290 | −0.004 |
| 10% blanks per column | 0.2246 | −0.008 |
| amounts ×3 | 0.2299 | −0.003 |
| amounts ×10 (far beyond the cap) | 0.2295 | −0.003 |
| account age ×2 | 0.2292 | −0.003 |
| account age ×0.5 | 0.2342 | +0.002 |
| busier customers (+2 txns/1h, +5/24h) | 0.2229 | −0.010 |
| new device forced on for +10% of rows | 0.2098 | −0.023 |
| 5% unseen merchant/country/channel | 0.2335 | +0.001 |
| 1% garbage rows (negative amount, hour 99, counts 100, strings) | 0.2315 | −0.001 |
| all together (blanks 5% + ×3 amounts + older accounts + unseen + garbage) | 0.2227 | −0.010 |


```

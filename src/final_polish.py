"""Final polish pass: submission.ipynb -> submission_v2.ipynb. Wording, labels, colours, added visuals only.
No modelling code, features, constants, seeds, CV or outputs-producing logic is changed."""
import json, re, sys
SRC, DST = "notebooks/submission.ipynb", "notebooks/submission_v2.ipynb"
nb = json.load(open(SRC)); C = nb["cells"]
CHANGED = []  # (cell label, reason)

def md(s): return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n") + "\n"}
def code(s): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s}
def find(prefix, kind=None):
    h = [i for i, c in enumerate(C) if "".join(c["source"]).startswith(prefix) and (kind is None or c["cell_type"] == kind)]
    assert len(h) == 1, (prefix[:50], h); return h[0]
def src(i): return "".join(C[i]["source"])
def setsrc(i, s): C[i]["source"] = s
def rep(i, old, new, reason):
    s = src(i); assert old in s, (i, old[:70]); setsrc(i, s.replace(old, new)); CHANGED.append((i, reason))
def esc(s): return re.sub(r"(?<!\\)\$", r"\\$", s)

# ============================================================ setup: one colour system
i = find("# Setup", "code")
rep(i, 'print("python", sys.version.split()[0]',
    '''# One colour system for every chart (meaning never changes):
#   LEGIT_C blue   = legitimate transactions / pushes the fraud score DOWN
#   FRAUD_C orange = fraud / "actually happened" fraud rate / pushes the score UP   (never used for "chosen model")
#   MODEL_C green  = a model's prediction or score;  TRAIN_C grey = training set / neutral;  TEST_C violet = test set
#   The chosen model is marked by a black outline + "shipped" label, not by colour.
#   Orange and green are adjacent slots of a colour-blind-validated palette (deuteranopia dE >= 8), so pairs stay legible.
LEGIT_C, FRAUD_C, MODEL_C, TRAIN_C, TEST_C, SHIP_EDGE = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a97", "#4a3aa7", "#0b0b0b"
print("python", sys.version.split()[0]''', "colour system defined once, with its meanings (item 10)")

# ============================================================ A8 + A2 + D17: title cell
i = find("# TrustGuard", "markdown")
rep(i, "Fancier models did not beat it.", "Fancier models did not clearly beat it (a 50/50 blend tied).", "exec summary wording (item 8)")
rep(i, "We neutralised a data trap (blank values are *never* fraud in training: 0 of 113, 0 of 124 rows) and scored every model on a \"test-like\" slice too.",
    "We neutralised a data trap (a blank `merchant_category` or `new_device` was never fraud: 0 of 113, 0 of 124 rows; other blank columns show no such pattern, 0.9–3.3%) and scored every model on a \"test-like\" slice too.", "exec summary blank-trap sentence (item 2)")
rep(i, "> - **Ceiling:** about 1 in 6 frauds shows no observable signal; no model catches those (Section 9).",
    "> - **Ceiling:** about 1 in 6 frauds shows no observable signal by our definition; no model catches those (Section 9).\n"
    "> - **Public leaderboard:** the shipped model scored **0.179**; our best upload, **0.184**, was a less-regularised variant we deliberately did not ship (Section 6.3). The gap to cross-validation (0.234) is explained by the test-set shift (Section 2.4).\n\n"
    "**Pipeline in one line:** raw CSV → clean (fill blanks with training medians/modes) → 29 features → logistic regression → probability → alert if probability × amount > \\$5 → top-3 reason codes.",
    "leaderboard bullet + one-line pipeline (items 8, 17)")

# ============================================================ D18: Section 1 glance table; 1.2 outcome grid
i = find("## 1. Understanding the problem", "markdown")
rep(i, "That trade-off, not accuracy, is what this notebook optimises.", esc("""That trade-off, not accuracy, is what this notebook optimises.

**Problem at a glance**

| | |
|---|---|
| Task | Score each transaction for fraud (binary classification) |
| Input | 10 columns per transaction: amount, hour, merchant, country, channel, recent activity counts and spend, account age, new-device flag |
| Output | A probability of fraud, not a yes/no label |
| Official metric | PR-AUC (ranking of frauds above legitimate transactions), plus F1 and recall at a cut-off |
| Two error types | Missed fraud (the bank loses the amount); false alarm (a blocked customer plus a review) |
| Key constraints | Fraud is 1.76% of rows; the test set is shifted from training (Section 2.4); only 353 fraud examples |"""), "problem-at-a-glance table (item 18)")
i = find("**What we found**\n- Average fraud", "markdown")
rep(i, "**Implications:** we report **fraud dollars caught**", esc("""**The four outcomes and what each costs**

| | Transaction is fraud | Transaction is legitimate |
|---|---|---|
| **Flagged** | Caught fraud: the amount is saved, minus one review | False alarm: one review cost (an assumption, see Section 7.3) and an annoyed customer |
| **Not flagged** | Missed fraud: the full amount is lost (mean \\$571) | Correct pass: no cost |

**Implications:** we report **fraud dollars caught**"""), "2x2 outcome grid with costs (item 18)")

# ============================================================ A1: 1.1 comment 1.77 -> 1.76
i = find("# 1.1 Rare by count", "code")
rep(i, "# Takeaway: 1.77% of transactions", "# Takeaway: 1.76% of transactions", "comment said 1.77% (item 8)")

# ============================================================ A2/A3: 2.5 table, p-values, legend, honest wording
i = find("# 2.5 Fraud rate when a column is blank", "code")
rep(i, '''        rows.append({"Column": c, "Rows where it is blank": int(miss.sum()),
                     "Frauds among those rows": int(train.loc[miss, TARGET].sum()),''',
    '''        n_b, k_b = int(miss.sum()), int(train.loc[miss, TARGET].sum())
        rows.append({"Column": c, "Rows where it is blank": n_b,
                     "Frauds among those rows": k_b,
                     "Expected frauds if blanks were random": round(n_b * train[TARGET].mean(), 1),
                     "p-value vs chance (two-sided binomial)": round(binomtest(k_b, n_b, train[TARGET].mean()).pvalue, 3),''',
    "expected frauds + binomial p-value columns (item 3)")
rep(i, "# 2.5 Fraud rate when a column is blank vs when it is filled in\nrows = []",
    "# 2.5 Fraud rate when a column is blank vs when it is filled in\nfrom scipy.stats import binomtest   # Colab-safe: scipy ships with scikit-learn\nrows = []", "scipy binomtest import (item 3)")
rep(i, 'ax.legend(frameon=False, loc="lower right")', 'ax.legend(frameon=False, loc="upper right")', "legend no longer overlaps the last bar (item 3)")
rep(i, "plt.tight_layout(); plt.show()\n", '''plt.tight_layout(); plt.show()

# Is "zero frauds" more than chance? Per column no; pooled, borderline. Either way the safe choice is the same.
pooled = train.merchant_category.isna() | train.new_device.isna()
p_pooled_zero = (1 - train[TARGET].mean()) ** int(pooled.sum())
display(Markdown(f"Each column alone is within chance (smallest p-value {blanks['p-value vs chance (two-sided binomial)'].min():.2f}). "
                 f"Pooling the {int(pooled.sum())} rows where merchant_category or new_device is blank, **0 frauds has about a "
                 f"{100 * p_pooled_zero:.1f}% probability by chance**, and we examined {len(blanks)} columns, so this is suggestive, not proof."))
''', "pooled chance computation printed (item 3)")
i = find("**What we found**\n- A blank `merchant_category`", "markdown")
setsrc(i, md(esc("""
**What we found**
- A blank `merchant_category` is **never** fraud (**0 of 113**); a blank `new_device` is **never** fraud (**0 of 124**). About 2 frauds per column would be expected if blanks were random.
- Most other blank columns run 2–3% or more (channel 3.3%, 24h count 3.1%, spend 3.0%, 1h count 2.8%, account age 2.2%), while country is 0.9%.
- Column by column, none of this is beyond chance (all p-values above 0.17); pooling the two zero-fraud columns, 0 frauds has about a 1.5% probability, and we examined eight columns.

**Implications:** the "blank means safe" pattern may be an artefact or may be luck; we cannot tell from 235 rows. The safe choice is the same either way: fill blanks with typical values and give the model no "was blank" flag (Section 3.2), so it cannot learn the pattern if it is an artefact and loses nothing if it is luck.
"""))["source"]); CHANGED.append((i, "2.5 findings rewritten honestly, country 0.9% fixed (items 2, 3)"))

# ============================================================ A5 + C10: 3.2 chart colours and wording
i = find("# 3.2 Where the fill values sit", "code")
s = src(i)
s = s.replace('color=[TRAIN_C if v != fill else FRAUD_C for v in share.index]', 'color=TRAIN_C, edgecolor=[SHIP_EDGE if v == fill else "none" for v in share.index], linewidth=2')
s = s.replace('ax.axvline(fill, color=FRAUD_C, lw=2)', 'ax.axvline(fill, color=SHIP_EDGE, lw=2)')
s = s.replace('color=[FRAUD_C if k == IMPUTE[col] else TRAIN_C for k in share.index[::-1]]', 'color=TRAIN_C, edgecolor=[SHIP_EDGE if k == IMPUTE[col] else "none" for k in share.index[::-1]], linewidth=2')
s = s.replace("(orange = the fill value, drawn on the training distribution)", "(black line / outline = the fill value, drawn on the training distribution)")
s = s.replace("so an imputed row is indistinguishable from an ordinary one.", "so an imputed row is close to an ordinary one (Section 7.2 measures how close).")
setsrc(i, s); CHANGED.append((i, "fill-value marks black not orange; takeaway softened (items 5, 10)"))
i = find("**What we found**\n- Numeric blanks", "markdown")
s = src(i).replace("(orange marks)", "(black marks)")
s = s.replace("**Implications:** an imputed row is indistinguishable from an ordinary row, so the model cannot learn \"blank means safe\". The proof that this works is in Section 7.2: rows that were originally blank score 0.0109 on average against 0.0177 for all rows — normal, not near zero.",
              "**Implications:** an imputed row is close to an ordinary row, so the model cannot learn \"blank means safe\". The evidence for this is in Section 7.2: rows that were originally blank score 1.1% on average against 1.8% for all rows, slightly lower but nowhere near zero.")
assert "The evidence for this" in s; setsrc(i, s); CHANGED.append((i, "3.2 wording: evidence, close not indistinguishable (item 5)"))

# ============================================================ E19: model search at a glance (Section 6 intro)
i = find("## 6. Compare models and choose one", "markdown")
rep(i, "The three that matter are shown here.", esc("""The three that matter are shown here.

**Model search at a glance** (copied from `docs/model_comparison_results.md`, 5×5 cross-validation in our development environment, not recomputed here; the three finalists are re-run below on the notebook's own 3×5 folds):

| Candidate | PR-AUC | Note |
|---|---|---|
| Logistic regression + CatBoost blend | 0.238 | best overall; CatBoost is not in the organisers' sandbox, so it cannot ship |
| Logistic regression, 29 features (finalist, shipped) | 0.229 | |
| Logistic regression + LightGBM 50/50 (finalist) | 0.231 | |
| CatBoost alone | 0.228 | not shippable |
| Elastic-net logistic regression | 0.227 | |
| Bagged logistic regression (50 bags) | 0.226 | |
| LightGBM, one-hot categories (finalist) | 0.222 | |
| Stacking: kNN + LightGBM into logistic regression | 0.222 | |
| XGBoost | 0.212 | |
| Sklearn gradient boosting / random forest / EBM | 0.209 / 0.201 / 0.208 | |
| Logistic regression, 10 raw columns only | 0.213 | the engineered features' value |
| CatBoost with class weights | 0.204 | probabilities inflated (Brier 0.023) |
| LightGBM DART / LambdaRank / SVM / neural net | 0.190 / 0.191 / 0.180 / 0.177 | |"""), "model search table copied from docs (item 19)")

# ============================================================ B9 + C10: 6.1 chart highlight, split 6.1 findings, new 6.3
i = find("# 6.1 Mean ± spread", "code")
rep(i, 'colours = [FRAUD_C if "SHIPPED" in n else MODEL_C for n in res["model"]]\nax.barh(res["model"], res["mean_3x5"], xerr=res["std_3x5"], color=colours, capsize=4, error_kw=dict(lw=1.2, ecolor="#52514e"))',
    'edges = [SHIP_EDGE if "SHIPPED" in n else "none" for n in res["model"]]\nax.barh(res["model"], res["mean_3x5"], xerr=res["std_3x5"], color=MODEL_C, edgecolor=edges, linewidth=2, capsize=4, error_kw=dict(lw=1.2, ecolor="#52514e"))',
    "chosen model = black outline, not orange (item 10)")
rep(i, 'ax.text(m + s + 0.004, yi, f"{m:.4f} ± {s:.4f}  ({lift})", va="center", fontsize=8)',
    'ax.text(m + s + 0.004, yi, f"{m:.4f} ± {s:.4f}  ({lift})" + ("  shipped" if "SHIPPED" in res["model"].iloc[yi] else ""), va="center", fontsize=8)', "shipped label (item 10)")
rep(i, 'Model comparison on identical folds — orange = shipped, dashed line = guessing', 'Model comparison on identical folds — black outline = shipped, dashed line = guessing', "title colour wording (item 10)")
i = find("**What we found**\n\n| Model | Score (PR-AUC, mean ± std)", "markdown")
setsrc(i, md(esc("""
**What we found**

| Model | Score (PR-AUC, mean ± std) | vs guessing (lift) |
|---|---|---|
| 50/50 mix | 0.235 ± 0.038 | 13.3× |
| Logistic regression | 0.234 ± 0.037 | 13.3× |
| Small LightGBM | 0.223 ± 0.044 | 12.6× |
| Baseline from Section 5 | 0.195 ± 0.045 | 11.1× |

- **Logistic regression and the 50/50 mix are tied.** The gap between them (0.001) is far smaller than the normal variation.
- The variation ranges of all the models overlap (overlapping error bars), so the ranking can't be decided by a decimal place alone.
- Logistic regression does best when used as a yes/no flag (best F1 0.299): at its best cut-off it catches about **1 in 4** real positives (recall 0.27), and about **1 in 3** of its flags are correct (precision 0.34).
- It also has the most accurate probabilities (lowest Brier score, 0.01514).

**Implications:** on our own tests the choice between logistic regression and the mix cannot be made on the score; Section 6.3 adds the one piece of evidence from the test distribution, and Section 6.2 shows why we did not reweight the rare class.
"""))["source"]); CHANGED.append((i, "6.1 findings trimmed to the CV verdict; leaderboard moved to 6.3 (item 9)"))
i62f = find("**What we found**\n1. **Reweighting doesn't help the ranking.**", "markdown")
new63_md = md(esc("""
### 6.3 Public leaderboard check
Our uploads of each model's predictions on the 12,000 test rows, scored by the organisers (logged in `docs/leaderboard_log.md`; the two probe rows are copied from `docs/model_comparison_results.md`, not recomputed here). The chart puts each model's practice score (mean ± spread over the 15 folds) next to its public score.

| Model uploaded | Upload | Public PR-AUC |
|---|---|---|
| Logistic regression, 29 features (shipped) | #364 | **0.179** |
| 50/50 mix | #336 | 0.169 |
| Small LightGBM alone (probe H) | probe, 3 Oct 5:40 PM | 0.167 |
| Baseline LightGBM, 10 raw columns (Section 5) | #362 | 0.160 |
| Logistic regression on the 10 raw columns only (no built features) | #748 | 0.151 |
| Logistic regression with much weaker regularisation (C = 100), a pre-registered probe we did **not** ship | probe F, 3 Oct 5:39 PM | 0.184 |

Every public score sits below its practice score (the test rows are harder: Section 2.4), but the order is the same, and the built features are worth 0.028 on the test set (0.179 vs 0.151).

The team's position on the board (0.184) comes from the last row: the leaderboard shows a team's *best upload*, not its shipped model. We kept the C = 0.2 version because the C = 100 variant scored lower in cross-validation (0.224 vs 0.234) and lost more under the stress tests (−0.025 vs −0.010 PR-AUC; see `docs/model_comparison_results.md`); a +0.005 gain on one public set is inside the noise band (Section 7.2), and picking a model on that would be tuning to the public leaderboard.
"""))
new63_code = code('''# 6.3 Practice score (mean ± spread over the 15 folds, from the 6.1 table) next to the public leaderboard score per uploaded model
public = {   # public PR-AUC per upload: docs/leaderboard_log.md (#364, #336, #362, #748); probes H and F: docs/model_comparison_results.md
    "logreg_v2 (SHIPPED)": 0.17882, "blend_lr_lgbm": 0.16862, "lgbm_v2": 0.16670, "lgbm_baseline (10 raw)": 0.16046}
cv_extra = {"LR, 10 raw columns": (0.2198, None, 0.15107),       # CV 3x5 and public from docs/leaderboard_log.md (#748)
            "LR, C=100 (probe, not shipped)": (0.2244, None, 0.18418)}  # CV 5x5 and public from docs/model_comparison_results.md
lb = res.set_index("model")[["mean_3x5", "std_3x5"]].copy()
lb["public"] = lb.index.map(public)
for k, (m_, s_, p_) in cv_extra.items():
    lb.loc[k] = [m_, s_, p_]
lb = lb.sort_values("public")
labels = {"logreg_v2 (SHIPPED)": "Logistic regression (shipped)", "blend_lr_lgbm": "50/50 mix", "lgbm_v2": "Small LightGBM",
          "lgbm_baseline (10 raw)": "Baseline LightGBM, raw columns"}
fig, ax = plt.subplots(figsize=(9, 3.8)); yy = np.arange(len(lb))
ax.errorbar(lb["mean_3x5"], yy + 0.18, xerr=lb["std_3x5"].fillna(0), fmt="o", color=MODEL_C, ecolor="#9a9a97", capsize=3, label="practice score (CV mean ± spread)")
ax.scatter(lb["public"], yy - 0.18, color=TEST_C, marker="D", zorder=3, label="public leaderboard score")
for yi, (m_, p_) in enumerate(zip(lb["mean_3x5"], lb["public"])):
    ax.plot([p_, m_], [yi - 0.18, yi + 0.18], color="#9a9a97", lw=0.8, zorder=1)
ax.set_yticks(yy); ax.set_yticklabels([labels.get(k, k) for k in lb.index])
ax.set_xlabel("PR-AUC"); ax.set_title("Every model scores lower on the public test rows than in practice; the order holds")
ax.legend(frameon=False, loc="upper left"); [ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()
# Takeaway: the drop from practice to public is similar for every model (the test rows are harder), so the comparison between
# models survives it; the two probe rows are shown for honesty and were not shipped.
''')
new63_found = md(esc("""
**Implications: why logistic regression, not the 50/50 mix**
1. **On our own tests they are tied.** The 0.001 gap is a rounding error next to the ±0.037 spread between folds; re-running with different splits flips the order.
2. **On the public test set, logistic regression scored higher: 0.179 vs 0.169 for the mix** (and 0.167 for LightGBM alone). That set is drawn from the shifted test distribution (Section 2.4), so it is the only evidence from data like what we will be graded on, and it points one way.
3. **Its probabilities are the most accurate** (lowest Brier score, 0.01514); the competition scores probability quality, and Section 7.3 prices every alert as probability × amount.
4. **It is the best yes/no flagger** of the four (best F1 0.299).
5. **Fewer moving parts:** one library, and an exact reason for every flag (Section 8.2). Less to explain, and less to break in the organisers' sandbox.

The 50/50 mix is the documented backup (runner-up).
"""))
C[i62f + 1:i62f + 1] = [new63_md, new63_code, new63_found]; CHANGED.append((i62f + 1, "new 6.3 Public leaderboard check: table, CV-vs-public chart, 5-bullet justification (item 9)"))

# ============================================================ C10/C11: 6.2 independent axes, colours
i = find("# 6.2 Same logistic regression", "code")
rep(i, "fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)", "fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)   # independent y-axes: the reweighted panel is ~20x taller", "6.2 independent y-axes (item 11)")
rep(i, 'axes[0].set_ylabel("% of transactions"); axes[0].legend(frameon=False)', 'axes[0].set_ylabel("% of transactions"); axes[1].set_ylabel("% of transactions"); axes[0].legend(frameon=False)', "6.2 y-label on both panels (item 11)")

# ============================================================ C10/C11/D15: 7.1 calibration panels, PR curve, colours
i = find("### 7.1 Can we trust", "markdown")
setsrc(i, md(esc("""
### 7.1 Can we trust the model's percentages, and where should the alarm go?
**Charts (top):** we sort all transactions by the model's fraud estimate and split them into 10 equal groups (deciles), from lowest risk (1) to highest risk (10). Left: for each group, the green bar is the fraud rate the model *predicted* and the orange bar is the fraud rate that *actually happened*. Right: the same ten groups as points on a predicted-vs-actual plot with both axes on a log scale, so groups 1–9 are readable; honest percentages sit on the diagonal.

**Chart (bottom):** the precision–recall curve. Reading left to right, we flag more and more transactions; the curve shows, at each point, the share of flags that are real fraud (precision) against the share of all frauds caught (recall). A curve further up and to the right is better; the area under it is PR-AUC, the competition's score. Guessing is a flat line at the base rate. Two points are marked on the shipped model's curve: its best yes/no cut-off (highest F1), and the operating point of the \\$5 expected-loss rule from Section 7.3.

**Table:** what happens if we raise an alarm at different cut-offs (thresholds), from 2% to 50%.
"""))["source"]); CHANGED.append((i, "7.1 intro: colours, reliability panel, PR-curve how-to-read (items 10, 11, 15)"))
i = find('BEST = "logreg_v2 (SHIPPED)"', "code")
rep(i, '''ax = (cal * 100).plot(kind="bar", figsize=(8, 3.4), rot=0, width=0.75, color=["#9a9a97", MODEL_C])
ax.set_xlabel("Groups of transactions, from lowest (1) to highest (10) predicted risk")
ax.set_ylabel("Fraud rate (%)")
ax.set_title("Predicted vs actual fraud rate: matching bars = trustworthy percentages")
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()''',
'''fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.5, 1]})
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
plt.tight_layout(); plt.show()''', "7.1 calibration: colour system + log-log reliability panel (items 10, 11)")
new_pr = code('''# 7.1c Precision-recall curves from pooled OOF predictions: guessing, the LightGBM baseline, the shipped LR
fig, ax = plt.subplots(figsize=(8, 4.2))
for name, p_, colour, ls in [("guessing (base rate)", naive, TRAIN_C, ":"), ("LightGBM baseline, raw columns", oof_base, TEST_C, "-"),
                             ("logistic regression (shipped)", p_best, MODEL_C, "-")]:
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
''')
C[i + 1:i + 1] = [new_pr]; CHANGED.append((i + 1, "new PR-curve cell with best-F1 and operating points (item 15)"))
i = find("**What we found**\n- **The percentages can be trusted.**", "markdown")
rep(i, "- **There's no single \"right\" alarm level.**", "- **The precision–recall curve** (bottom chart) shows the shipped model above the baseline at every recall; the \\$5 rule's operating point sits far to the right of the best-F1 point because it is chosen on money, not on F1.\n- **There's no single \"right\" alarm level.**", "7.1 findings: PR-curve bullet (item 15)")

# ============================================================ A6: 7.2 pooled lift
i = find("# 7.2 Robustness: inject extra blanks", "code")
rep(i, '''      f"(lift {average_precision_score(y[TEST_LIKE], p_best[TEST_LIKE]) / y[TEST_LIKE].mean():.1f}x its base rate)")''',
    '''      f"(lift {average_precision_score(y[TEST_LIKE], p_best[TEST_LIKE]) / y[TEST_LIKE].mean():.1f}x its base rate)")
print(f"for comparison, pooled all-data lift: PR-AUC {average_precision_score(y, p_best):.4f} / base rate {y.mean():.4f} "
      f"= {average_precision_score(y, p_best) / y.mean():.1f}x")''', "pooled all-data lift printed for a like-for-like comparison (item 6)")
i = find("**What we found**\n1. **Extra blanks barely hurt.**", "markdown")
rep(i, "4. **It works on the test-like rows too.** It scores 12.6× better than guessing there (lift), similar to 13.3× on all data. (The raw score is higher, 0.43, only because fraud is more common in those rows, 3.4%, so the lift is the fairer comparison.)",
    "4. **It works on the test-like rows too.** It scores 12.6× better than guessing there (lift), close to the 13.0× pooled lift on all data (0.2288 / 0.0176, computed above; like for like, both pooled). (The raw score is higher, 0.43, only because fraud is more common in those rows, 3.4%, so the lift is the fairer comparison.)", "7.2 like-for-like lift (item 6)")
rep(i, "the public leaderboard score (0.179, Section 6.1)", "the public leaderboard score (0.179, Section 6.3)", "cross-reference to 6.3 (item 9)")

# ============================================================ C10/D16/D18: 7.3 colours, gains chart, counts grid
i = find("### 7.3 How much money does the model save?", "markdown")
rep(i, "Longer bar = better; the orange bar is the strategy we chose.", "Longer bar = better; the outlined green bar is the strategy we chose.", "7.3 intro colour wording (item 10)")
rep(i, "**Tables:** the strategies side by side, then how much we'd catch if the fraud team can only check a limited number of alarms.",
    "**Tables:** the strategies side by side, then how much we'd catch if the fraud team can only check a limited number of alarms.\n\n**Second chart:** disruption vs benefit. As we flag more transactions (x-axis), how much of the fraud money do we catch (y-axis)? One curve ranks transactions by probability × amount, one by probability alone; the diagonal is random flagging. The three rules are marked where they sit.", "7.3 intro: gains chart how-to-read (item 16)")
i = find("c = REVIEW_COST", "code")
rep(i, 'colours = [FRAUD_C if "chosen" in name else "#9a9a97" for name in sel.index]\nax = sel.plot.barh(figsize=(8.5, 3), color=colours)',
    'colours = [MODEL_C if "chosen" in name else TRAIN_C for name in sel.index]\nax = sel.plot.barh(figsize=(8.5, 3), color=colours, edgecolor=[SHIP_EDGE if "chosen" in name else "none" for name in sel.index], linewidth=2)',
    "7.3 bar chart: chosen = outlined green, not orange (item 10)")
new_gains = code('''# 7.3 Disruption vs benefit: share of fraud dollars caught as the flag rate rises, ranking by p x amount vs by p alone
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
ax.set_title("Disruption vs benefit: the \\$500 rule sits near the frontier at 10.6%; our rule flags more and nets more"); ax.legend(frameon=False, fontsize=8, loc="lower right")
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()
# What would ranking by p x amount catch at the $500 rule's flag rate? (honest like-for-like at 10.6%)
k_500 = int((amt > 500).sum()); el_at_500 = y_el[k_500 - 1]
display(Markdown(f"At the \\\\$500 rule's own flag rate ({100 * (amt > 500).mean():.1f}%), ranking by probability × amount would catch "
                 f"**{100 * el_at_500:.1f}%** of fraud dollars vs the rule's {100 * amt[(amt > 500) & is_fraud].sum() / total_fraud_dollars:.1f}%; "
                 f"our \\\\$5 rule flags {100 * (p_best * amt > c).mean():.1f}% and catches {100 * amt[(p_best * amt > c) & is_fraud].sum() / total_fraud_dollars:.1f}%, "
                 f"and wins on net dollars at the stated review cost (table above)."))

# The four outcomes per 10,000 transactions under the chosen rule (counts from the notebook's arrays)
fl = p_best * amt > c; scale = 10_000 / len(amt)
tp, fp, fn, tn = (fl & is_fraud).sum(), (fl & ~is_fraud).sum(), (~fl & is_fraud).sum(), (~fl & ~is_fraud).sum()
display(Markdown(f"""
**Per 10,000 transactions under "probability × amount > \\\\${c:.0f}"** (practice data)

| | Fraud | Legitimate |
|---|---|---|
| **Flagged** | caught fraud: **{tp * scale:,.0f}** (\\\\${amt[fl & is_fraud].sum() * scale:,.0f} saved before review costs) | false alarms: **{fp * scale:,.0f}** (\\\\${c * fp * scale:,.0f} of review cost) |
| **Not flagged** | missed fraud: **{fn * scale:,.0f}** (\\\\${amt[~fl & is_fraud].sum() * scale:,.0f} lost) | correct passes: **{tn * scale:,.0f}** |
"""))
''')
j = find("# Calibration check for the priced-alert rule", "code")
C[j:j] = [new_gains]; CHANGED.append((j, "new disruption-vs-benefit chart + per-10k outcome grid (items 16, 18)"))
# A1: neutral NOTE in the calibration-check cell
j = find("# Calibration check for the priced-alert rule", "code")
rep(j, '''# NOTE(Germaine): if any future candidate uses class weights or resampling, recalibrate on OOF
# (Platt/isotonic) BEFORE it feeds this section - otherwise p x amount misprices every alert.
# The shipped LR needs none: no weights, reliability curve ~ diagonal (Section 7.1), and isotonic
# recalibration made Brier worse in your 5x5 runs, so it is correctly NOT applied.''',
'''# NOTE: if a future candidate uses class weights or resampling, recalibrate on OOF first (Platt/isotonic);
# otherwise p x amount misprices every alert. The shipped LR needs none: no weights, reliability curve ~ diagonal
# (Section 7.1), and isotonic recalibration made Brier worse in 5x5 CV runs, so it is not applied.''', "neutral NOTE (item 1)")
i = find("**What we found**\nWe compared strategies by **money saved per 10,000 transactions**", "markdown")
rep(i, "- **If the fraud team is short-staffed**", "- **Disruption vs benefit (second chart):** at the \\$500 rule's own flag rate (10.6%) it sits close to the frontier, so a dollar rule is a strong cheap baseline; our rule flags more (17.0%) and converts that into more fraud dollars caught and more net savings at the \\$5 review cost (figures printed under the chart).\n- **If the fraud team is short-staffed**", "7.3 findings: honest gains-chart caption (item 16)")

# ============================================================ C10/C12: 8.1 colours, family view, hour-of-day chart
i = find("### 8.1 Which signals matter most?", "markdown")
rep(i, "**How to read it:** red bars push the fraud score **up**, blue bars push it **down**. Longer bar = stronger effect.",
    "**How to read it:** orange bars push the fraud score **up**, blue bars push it **down**. Longer bar = stronger effect. Overlapping features (for example the night-time flag and the hour-of-day curve) split one effect between them, so the next two charts regroup the signals into families and show the time-of-day effect directly.", "8.1 intro: colours + family/hour charts (items 10, 12)")
i = find("# 8.1 Global drivers of the shipped LR", "code")
rep(i, 'color=np.where(top[::-1] > 0, "C3", "C0"),\n                    title="What pushes the fraud score up (red) or down (blue)")',
    'color=np.where(top[::-1] > 0, FRAUD_C, LEGIT_C),\n                    title="What pushes the fraud score up (orange) or down (blue)")', "8.1 drivers chart colours (item 10)")
rep(i, "plt.tight_layout(); plt.show()\n# Takeaway: velocity", '''plt.tight_layout(); plt.show()

# 8.1b Family view: per row, sum coefficient x standardised value within each family; show each family's mean absolute contribution
FAMILY = {"velocity": ["transactions_last_24h", "transactions_last_1h", "burst_ratio", "n1_ge3", "n1_ge4", "n24_ge10", "burst_young"],
          "amount vs account history": ["log_amount", "log_spend", "amt_share_24h", "amt_vs_prev_avg", "log_amt_per_age", "spend_per_txn", "amt_gt500", "big_old"],
          "time of day": ["transaction_hour", "hour_sin", "hour_cos", "night", "log_amt_x_night"],
          "device (incl. new-device interactions)": ["new_device", "newdev_young", "log_amt_x_newdev", "night_newdev"], "account age": ["log_age", "age_lt180"]}
contrib_ = Z_ * clf_.coef_[0]
fam_rows = {}
for fam, cols in FAMILY.items():
    idx = [k for k, n in enumerate(names_) if n in cols]
    fam_rows[fam] = np.abs(contrib_[:, idx].sum(axis=1)).mean()
for fam, prefix in [("merchant", "merchant_category_"), ("channel", "transaction_channel_"), ("country", "country_")]:
    idx = [k for k, n in enumerate(names_) if n.startswith(prefix)]
    fam_rows[fam] = np.abs(contrib_[:, idx].sum(axis=1)).mean()
fam = pd.Series(fam_rows).sort_values()
ax = fam.plot(kind="barh", figsize=(7, 3.4), color=MODEL_C, title="Signal families: average size of each family's push on the fraud score")
ax.set_xlabel("mean |contribution| per transaction (log-odds)"); [ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()

# 8.1c Time of day: predicted (OOF) vs actual fraud rate by hour, so the night-time story is visible despite the overlapping features
by_hour = pd.DataFrame({"actual": y.groupby(train.transaction_hour).mean() * 100, "predicted (OOF)": pd.Series(p_best).groupby(train.transaction_hour.values).mean() * 100})
ax = by_hour.plot(figsize=(8, 3.2), color=[FRAUD_C, MODEL_C], marker="o", ms=4, title="Fraud rate by hour of day: actual (orange) vs the model's prediction (green)")
ax.set_xlabel("hour of day (0 = midnight)"); ax.set_ylabel("% of transactions that are fraud"); ax.set_xticks(range(0, 24, 2)); ax.legend(frameon=False)
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()
# Takeaway: velocity''', "8.1 family-level view + hour-of-day predicted vs actual (item 12)")
i = find("**What we found**\n- **Biggest red flags:**", "markdown")
rep(i, "- **Read the time-of-day bars as a pair.** The night-time flag and the \"time of day\" signals describe the same hours, so the model splits the effect between them and the bars partly cancel (overlapping features). Together they still say what Section 2.3 showed: late-night activity raises risk.",
    "- **Overlapping features split their effect between them** (night-time flag vs hour-of-day curve; amount vs today's spend vs amount vs usual transaction, correlated 0.86), which is why single bars can point opposite ways. The family chart adds them up: velocity and amount-vs-history are the largest families, then time of day. The hour-of-day chart shows the model's predictions tracking the actual late-night rise in fraud.", "8.1 findings: family + hour charts (item 12)")

# ============================================================ C14 + A4: 8.2 merged table, FEAT wording
i = find("# 8.2 Reason codes", "code")
rep(i, '''pd.set_option("display.max_colwidth", None)
display(reason_codes(lr_full, X_test_all.iloc[top_idx]))
test.iloc[top_idx][["transaction_amount", "account_age", "new_device", "transactions_last_1h", "transactions_last_24h", "country", "merchant_category"]]''',
'''pd.set_option("display.max_colwidth", None)
flagged = test.iloc[top_idx][["transaction_amount", "account_age", "new_device", "transactions_last_1h", "transactions_last_24h", "country", "merchant_category"]].copy()
flagged = flagged.join(reason_codes(lr_full, X_test_all.iloc[top_idx]))     # one table: raw details + probability + reasons
flagged''', "8.2 one merged table (item 14)")
i = find("### 8.2 Why was this transaction flagged?", "markdown")
rep(i, "Below: the 5 riskiest transactions in the test set, with their reasons and their raw details.", "Below: the 5 riskiest transactions in the test set, with their raw details, probability and reasons in one table.", "8.2 intro matches merged table (item 14)")
i = find("**What we found**\n- The 5 riskiest test transactions", "markdown")
rep(i, "Singapore's financial regulator requires that kind of appeal process in its fairness and ethics principles for AI (MAS FEAT, principle 13).",
    "MAS FEAT principles 10 and 13 expect channels to appeal AI-driven decisions and clear explanations of what data drove them.", "FEAT principles 10 and 13, guidance not requirement (item 4)")

# ============================================================ C13: 8.3 two panels + rule-based table
i = find("### 8.3 Who gets bothered, and is that fair?", "markdown")
setsrc(i, md(esc("""
### 8.3 Who gets bothered, and is that fair?
Using the 10% alarm level from Section 7, for each country we check:
- what share of **honest customers get wrongly flagged** (false positive rate, FPR)
- what share of **real fraud gets caught** (recall)

**Charts:** two panels in the same country order: honest customers wrongly flagged (left) and fraud caught (right). Singapore and Indonesia are outlined; the dashed line is the figure across all customers.

Then the same table under the rule we actually recommend (probability × amount > \\$5, Section 7.3), and finally we retrain the model **without the country column** to see whether we need it. This follows Singapore's financial regulator's fairness principles (MAS FEAT, principles 1–3).
"""))["source"]); CHANGED.append((i, "8.3 intro: two panels + rule-based table (item 13)"))
i = find("# 8.3 Fairness by country", "code")
rep(i, '''overall_fpr = 100 * (p_best[y.to_numpy() == 0] >= THRESHOLD).mean()
ax = fair["FPR % (legit flagged)"].plot(kind="bar", figsize=(7, 3), rot=0, color="#9a9a97",
        title=f"Honest customers wrongly flagged at the {THRESHOLD:.0%} alarm level, by country")
ax.axhline(overall_fpr, ls="--", c="gray", lw=1, label=f"all customers: {overall_fpr:.2f}%")
ax.set_ylabel("Wrongly flagged (%)"); ax.set_xlabel("")
ax.legend(frameon=False)
[ax.spines[s].set_visible(False) for s in ("top", "right")]
plt.tight_layout(); plt.show()''',
'''overall_fpr = 100 * (p_best[y.to_numpy() == 0] >= THRESHOLD).mean()
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
plt.suptitle("Outlined bars: Singapore and Indonesia", y=1.02); plt.tight_layout(); plt.show()

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
                 f"and the rule catches {fair_rule.loc['SG', 'recall (cases)']:.0%} of SG fraud cases ({fair_rule.loc['SG', 'fraud $ caught %']:.0f}% of SG fraud dollars)."))''',
    "8.3 two-panel chart (FPR + recall, SG/ID outlined) and fairness table under the recommended rule (item 13)")
i = find("**What we found**\n- **Honest customers in some countries get flagged far more often:**", "markdown")
rep(i, "- **Removing country doesn't hurt in our tests:**", "- **Under the rule we actually recommend** (probability × amount > \\$5), the by-country gap is recomputed in the table above; the printed line states the ID-vs-SG ratio under that rule so the fairness claim rests on the policy we propose, not only on a flat cut-off.\n- **Removing country doesn't hurt in our tests:**", "8.3 findings: rule-based fairness (item 13)")

# ============================================================ A5/A7/E20: Section 9
i = find("## 9. Limitations and next steps", "markdown")
rep(i, "**Frauds with no signal.** A fraud on a known device, an old account, with an ordinary amount and no burst looks like any other transaction. The cell below counts how many training frauds look like that.",
    "**Frauds with no signal.** By our definition (known device, account older than 365 days, amount under 300, at most 2 transactions in the last hour) a fraud looks like any other transaction. The cell below counts how many training frauds fit it.", "9: definition stated (item 7)")
i = find("These frauds are indistinguishable from ordinary transactions", "markdown")
s = src(i)
s = s.replace("These frauds are indistinguishable from ordinary transactions on the columns we have, so **no model catches them**; they set a hard ceiling on recall.",
              "By this definition these frauds are indistinguishable from ordinary transactions on the columns we have, so **no model catches them**; they set a hard ceiling on recall.")
s = s.replace("and no tuning to the public leaderboard, where ranks within ±0.03 are noise.", "and no tuning to the public leaderboard, where we treat ranks within ±0.03 as noise.")
s = s.replace("**Fairness.** Legitimate ID and AU customers are flagged 6.4× more often than SG ones (Section 8.3); dropping `country` costs nothing in cross-validation.", "**Fairness.** At a flat 10% cut-off legitimate ID and AU customers are flagged 6.4× more often than SG ones; under the recommended expected-loss rule the gap narrows to 1.8× but remains (Section 8.3). Dropping `country` costs nothing in cross-validation.")
s = s.replace("**Fairness.**", "**Imputation can under-score blank rows.** The fill values (grocery, known device, SG, card present) are the most common values but also low-risk ones, so an imputed row can score slightly below its true risk (1.1% vs 1.8%, Section 7.2). This matters more with 3.5× more blanks in the test set; the injected-blank check in Section 7.2 bounds the cost at about 0.007 PR-AUC. We kept the imputation because the alternative, a \"was blank\" flag, would hand the model the artefact in Section 2.5.\n\n**Fairness.**")
s = s.replace("a 12-scenario stress test (`src/stress_test.py`) that crashes nothing,", "a 12-scenario stress test (`src/stress_test.py`, table below) that crashes nothing,")
s = s.rstrip("\n") + esc("""

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
""") + "\n"
setsrc(i, s); CHANGED.append((i, "9: 'by this definition', ±0.03 softened, imputation caveat, stress-test table (items 5, 7, 20)"))

# ============================================================ A1: sweep for names / process notes
bad = []
for k, c in enumerate(C):
    s = src(k)
    for pat in ["Germaine", "Marcus", "Jay", "Pin Wei", "your ", "TODO", "NOTE("]:
        if pat in s: bad.append((k, pat, [l.strip()[:80] for l in s.splitlines() if pat in l][:2]))
print("leftover names / notes:", bad or "none")

# ============================================================ write
nb["cells"] = C
json.dump(nb, open(DST, "w"), indent=1, ensure_ascii=False)
print("cells:", len(C), "| changed entries:", len(CHANGED))
json.dump(CHANGED, open(DST + ".changes.json", "w"))

# Track 2 model comparison — results log (3 Oct 2026)

All scores: 5×5 repeated stratified CV on identical folds (`shared/folds.csv`), averaged over repeats.
Paired comparisons use the Nadeau–Bengio corrected t-test on 25 fold scores.

## Final model
**50/50 logit blend of LogisticRegression(C=0.2) + CatBoost(depth 4, 600 it, 5-seed average), v2 features (29 cols).**
`model.pkl`: `predict()` returns P(fraud). Built only from standard sklearn/scipy/catboost classes (no custom class).

| Metric | Value |
| --- | --- |
| PR-AUC | **0.237** (naive 0.018, round-1 LightGBM baseline 0.197) |
| PR-AUC, test-like weighting | 0.356 |
| PR-AUC, with test-like blanks | 0.236 |
| ROC-AUC | 0.776 |
| Brier | 0.01504 (naive 0.01734); calibration curve ≈ diagonal; isotonic recalibration makes it worse → not applied |
| F1 / Recall at 2% flag rate | 0.271 / 0.289 |
| Mean p on amount>2000 & age>1000 (actual 1.7%) | 2.8% |

## Round 1 (v1 features, 22 cols)
| Model | PR-AUC | test-like | Brier | vs LR (wins/25, p) |
| --- | --- | --- | --- | --- |
| LR C=0.2 | 0.220 | 0.324 | 0.01522 | — |
| CatBoost d4 | 0.224 | 0.340 | 0.01518 | 14/25, 0.49 |
| EBM (± interactions) | 0.208 | 0.301 | 0.01530 | 4/25 |
| HGB d2 | 0.202 | 0.295 | 0.01534 | 5/25 |
| RandomForest | 0.201 | 0.305 | 0.01540 | 5/25 |
| **LR+CatBoost 50/50 logit blend** | **0.231** | 0.349 | 0.01510 | 22/25, 0.068 |
| CatBoost d3 / d5 / l2=30 / slow | 0.218–0.225 | | | flat → settings don't matter |

## Round 2
Feature ablations (same models):
| Change | LR | CatBoost | Verdict |
| --- | --- | --- | --- |
| raw 10 columns only | 0.213 | 0.220 | engineered features help (+0.007/+0.004) |
| drop hand flags | 0.214 | — | flags help (+0.007, 5/25 wins without) |
| drop `country` | 0.219 | 0.221 | costs ~0.001–0.003 → cheap to remove for fairness |
| **v2: + interactions** (night×new_device, burst×young, log_amt×new_device, log_amt×night, hour sin/cos, spend per txn) | **0.229** (20/25, p=0.08) | **0.228** | adopted |

More families (v1 features): LR with degree-2 polynomials 0.203 · CatBoost class-weighted 0.204 (Brier 0.023 — weights wreck probabilities) · ExtraTrees 0.200 · MLP 0.177 · SVM 0.180.
LightGBM/XGBoost not run locally (need libomp); HGB is the sklearn equivalent.

Combining (all nested, i.e. weights fitted inside folds):
| Combination | PR-AUC | vs 50/50 blend_v2 |
| --- | --- | --- |
| **50/50 fixed logit blend lr_v2+cat_v2** | **0.2377** | — |
| same with 5-seed CatBoost | 0.2374 | +0.0002 (noise; kept for variance) |
| nested-tuned blend weight | 0.2355 | 4/25 |
| logistic stacker (2 / 4 / 8 models) | 0.235 / 0.234 / 0.229 | worse, more models = worse |
| 3-way +RF / +EBM | 0.233 / 0.233 | worse |
Simplest combination wins; extra machinery overfits.

Honesty note: blend_v2 vs round-1 champion blend_v1: +0.0054, 18/25 folds, corrected p=0.22 — better on every metric
but not individually "proven"; both components improved with p≈0.08. Total candidates compared: ~30.

## Threshold table (OOF, blend_v2)
| threshold | flagged | precision | recall | F1 | fraud $ caught |
| --- | --- | --- | --- | --- | --- |
| 0.03 | 8.6% | 0.10 | 0.47 | 0.16 | 72% |
| 0.05 | 4.0% | 0.17 | 0.38 | 0.24 | 59% |
| 0.10 | 1.8% | 0.28 | 0.28 | 0.28 | 49% |
| 0.20 | 0.8% | 0.45 | 0.21 | 0.29 | 39% |
| 0.50 | 0.3% | 0.72 | 0.14 | 0.23 | 28% |
Suggested tiers: ≥0.20 hold/review · 0.05–0.20 OTP step-up · <0.05 approve.

## Round 3 (all vs blend_v2 = 0.2377)
Data insights that motivated it: amount only signals fraud on young accounts (>2k: 34% at <180d, 10.9% at 180–1000d, 1.7% at >1000d);
new_device × velocity is multiplicative (new device & 3–4 txns/1h = 24%, 5+ = 38%); amount-relative-to-merchant-median is U-shaped (weak);
0 exact train/test feature matches (no leak); fraud rate flat across number of blanks (missingness carries no signal).

| Idea | Result | Verdict |
| --- | --- | --- |
| v3 features: + amount relative to merchant/country median, big_young, newdev_burst flags | LR 0.226 (−0.002), CatBoost 0.2295 (+0.002), blend **0.2377 (±0.000)** | no gain; models already capture these interactions |
| Importance-weighted training towards test distribution (w^0.5) | blend 0.2353; test-like 0.3527 vs 0.3562 | no gain, even on the test-like metric |
| CatBoost monotone constraints (12 features) | blend 0.2359; test-like 0.3488 | slightly worse |
| Pure-sklearn fallback LR+HGB (if catboost unavailable) | 0.2215, 2/25 folds, p=0.04 | clearly worse → catboost is worth keeping in requirements |

Conclusion: ~40 candidates; every sensible model/feature/weighting variant lands at 0.235–0.238. The dataset's signal ceiling
is reached; remaining differences are fold noise. Champion stays **blend_v2** (pre-registered 50/50, 5-seed CatBoost).

## Round 4: sandbox constraint (portal `requirements-image.txt`: sklearn 1.5.2, lightgbm 4.5.0, xgboost 2.1.3, **no catboost**)
CatBoost cannot ship. Re-ran the only boosters available in the sandbox, trained under the image versions (`.venv-image`), same 5×5 folds.

| Model (v2 features) | PR-AUC | test-like | blanks | Brier |
| --- | --- | --- | --- | --- |
| LR C=0.2 (`lr_v2`) | 0.2286 | 0.3331 | 0.2282 | 0.01513 |
| LightGBM, integer-coded cats (d3, 7 leaves, 600 it, lr .03, λ=10) | 0.2197 | 0.3365 | 0.2173 | 0.01528 |
| LightGBM, one-hot cats | 0.2222 | 0.3395 | 0.2218 | 0.01523 |
| LightGBM, native categoricals | 0.2185 | 0.3289 | 0.2158 | 0.01530 |
| LightGBM 15 leaves (≈ repo baseline params) | 0.2131 | 0.3332 | 0.2107 | 0.01534 |
| LightGBM 3-seed | 0.2215 | 0.3377 | 0.2190 | 0.01526 |
| XGBoost d3 | 0.2116 | 0.3197 | 0.2106 | 0.01534 |
| HGB (sklearn) | 0.2087 | 0.3127 | 0.2059 | 0.01532 |
| **50/50 LR + LightGBM one-hot (`VotingClassifier` soft) — SHIPPED** | **0.2306** | **0.3433** | **0.2306** | **0.01511** |
| LR + LightGBM coded | 0.2300 | 0.3441 | 0.2293 | 0.01513 |
| LR + XGBoost | 0.2261 | 0.3360 | 0.2252 | 0.01516 |
| (reference, not shippable) LR + CatBoost | 0.2377 | 0.3562 | 0.2367 | 0.01505 |

Shipped blend vs LR alone: +0.0020 (18/25 folds, p=0.72); test-like +0.0083 (17/25); blanks +0.0024. Not significant on PR-AUC, but
never worse on any metric and adds a second model family for robustness → shipped, with **LR alone as the documented fallback**.
Shipped blend vs CatBoost blend: −0.0072 (the price of the sandbox's library list).
Export: `VotingClassifier(voting="soft")` of the LR pipeline and the LightGBM pipeline, pickled under the image versions; `predict_proba`
verified on a shuffled, id-less copy of test.csv (what the sandbox feeds `prediction.ipynb`).

## Fairness check (MAS FEAT principles 1–3), OOF predictions, operating threshold 0.10
| Country | n | actual fraud % | flag % | FPR % (legit customers flagged) | recall |
| --- | --- | --- | --- | --- | --- |
| ID | 840 | 4.05 | 6.19 | **5.21** | 0.29 |
| AU | 584 | 2.91 | 7.02 | 5.11 | 0.71 |
| US | 693 | 3.17 | 5.92 | 4.02 | 0.64 |
| VN / PH / JP / TH / GB | 382–644 | 1.7–3.7 | 3.3–5.5 | 2.3–3.5 | 0.38–0.58 |
| MY | 1,225 | 1.47 | 2.04 | 1.41 | 0.44 |
| **SG** | 13,931 | 1.31 | 0.97 | **0.87** | **0.09** |

Two things a bank must confront, both honest and both good Q&A material:
1. **Disruption is unequal:** a legitimate Indonesian or Australian customer is ~6× more likely to be flagged than a Singaporean one.
   Dropping `country` costs only 0.001–0.003 PR-AUC, so a fairness-constrained deployment is cheap; we report both variants.
2. **Recall is unequal the other way:** the model catches 9% of SG frauds vs 50–70% elsewhere at this threshold. SG fraud looks like
   normal SG behaviour (older accounts, smaller amounts, no new device) — it is most of the "1-in-6 frauds with no visible signal".
   This is the real limitation of a transaction-only model: without customer history, domestic fraud is nearly invisible.

Reason codes (exact LR contributions, log-odds) for the three highest-risk test rows, e.g.
`p=0.84: transactions today (+2.19); amount × new device (+1.39); ≥3 transactions in the last hour (+0.69)` — 1,559 cash transfer from
Indonesia on a 102-day-old account, new device, 6 txns in the last hour. `src/models.py: reason_codes()`, `fairness_by_group()`.

## Reconciliation with §3–4 (Marcus's cleaning PR #3), same 5×5 folds, platform versions
| Question | Result | Decision |
| --- | --- | --- |
| Our blend with **§3 imputation constants** (grocery / SG / card_present / new_device 0) vs our "neutral" fills | 0.2311 vs 0.2306, 11/25 folds, p = 0.75 → identical | **Adopt §3 constants** in `make_features` — one cleaning policy for the whole notebook (`src/models.py` asserts it matches `src/preprocess.py`). |
| **§5 baseline** (LightGBM 300 trees / 15 leaves, 10 raw features, §3 `clean()`) vs our LR + LightGBM blend | **0.1919 vs 0.2306: +0.0384 PR-AUC (+20%), blend wins 23/25 folds, p = 0.006**; blanks 0.1902 vs 0.2306; Brier 0.0157 vs 0.0151 | The engineered features were rejected in §4 because they were tested with LightGBM only (trees are invariant to log/ratio transforms). For logistic regression they are the whole gain (raw 0.213 → 0.229), and LR is half of the best shippable model. Recommend §4 adopts `make_features` from `src/models.py` (clean + 19 features) and §6/§9 adopt the blend. |

## Round 5: "out of the box" challengers (sandbox-compatible only), same 5×5 folds, platform versions
Script: `src/experiments_models.py` (`report` prints this table). Champion = shipped LR + LightGBM blend, 0.2311.

| Challenger | Idea | PR-AUC | test-like | Brier | Δ vs champion | wins/25 | p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pseudo_neg_blend | add test rows the champion scores < 0.3% as extra legit examples (domain adaptation) | 0.2300 | 0.3289 | 0.0151 | −0.0008 | 9 | 0.69 |
| elasticnet_lr | L1+L2 logistic regression | 0.2272 | 0.3190 | 0.0151 | −0.0030 | 9 | 0.58 |
| bagged_lr | random-subspace bagging of LR (50 bags) | 0.2258 | 0.3140 | 0.0151 | −0.0038 | 11 | 0.45 |
| lasso_int C=0.1 / 0.02 / 0.05 | L1 LR over all pairwise products of numerics+flags (sparse interaction search) | 0.2255 / 0.2234 / 0.2199 | 0.31–0.32 | 0.0152 | −0.005 to −0.010 | 6–8 | 0.2–0.6 |
| blend_lr_negbag / lgbm_negbag | LightGBM with negative downsampling per tree (all frauds, 30% of legit) | 0.2254 / 0.2157 | 0.326 / 0.318 | 0.0157 / 0.0176 | −0.006 / −0.013 | 4 / 4 | 0.15 / 0.05 |
| stack_knn_lgbm | StackingClassifier: meta-LR on v2 features + OOF P(fraud) from kNN(50) and LightGBM | 0.2222 | 0.3152 | 0.0152 | −0.0079 | 5 | 0.20 |
| blend_lr_dart / lgbm_dart | DART boosting (tree dropout) | 0.2191 / 0.1900 | 0.316 / 0.279 | **0.029 / 0.071** | −0.004 / −0.003 (within-fold) | 6 / 9 | — |
| lgbm_rank | LambdaRank objective (optimise the ordering directly) | 0.1911 | 0.2709 | 0.073 | −0.0326 | 0 | <0.001 |
| knn50 | kNN alone | 0.1888 | 0.2646 | 0.0156 | −0.0353 | 0 | <0.001 |
| lasso_int_all C=0.03 | L1 LR over pairwise products of everything incl. one-hot cats (~700 cols) | (running at time of writing) | | | | | |

Side-finding worth a sentence in the report: DART's *within-fold* ranking is nearly as good as the champion's (Δ −0.003 per fold), but
its pooled OOF PR-AUC collapses to 0.19 because its probability scale drifts from fold to fold (Brier 0.07). Even for a ranking metric,
a model whose scores mean different things on different data is fragile — calibration is not optional for the private test.

**Conclusion after 5 rounds / ~65 candidates:** every sensible model lands within fold noise of 0.23 under the sandbox constraint.
The champion stays. Remaining marks are in the write-up, robustness evidence and the Round-2 demo, not in the model.

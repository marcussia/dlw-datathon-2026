# Final polish: changed cells in `notebooks/submission_v2.ipynb`

Cell numbers refer to `submission_v2.ipynb` (91 cells; 86 in the previous version). Every cell not listed is byte-identical to the previous `submission.ipynb`. Code edits are chart/label/display only; `model.pkl` and `submission.csv` are byte-identical (sha256 28800e81…, d04987a6…).

| v2 cell | Type | What changed |
|---|---|---|
| 0 | markdown | # TrustGuard — Track 2: Financial Fraud Detection — exec summary wording (item 8); exec summary blank-trap sentence (item 2); leaderboard bullet + one-line pipeline (items 8, 17) |
| 2 | code | # Setup — single cell, Colab-safe. Everything below uses only these im — colour system defined once, with its meanings (item 10) |
| 4 | markdown | ## 1. Understanding the problem — problem-at-a-glance table (item 18) |
| 6 | code | # 1.1 Rare by count, heavier by dollars - the two numbers behind "accu — comment said 1.77% (item 8) |
| 10 | markdown | **What we found** — 2x2 outcome grid with costs (item 18) |
| 25 | code | # 2.5 Fraud rate when a column is blank vs when it is filled in — expected frauds + binomial p-value columns (item 3); scipy binomtest import (item 3); legend no longer overlaps the last bar (item 3); pooled chance computation printed (item 3) |
| 26 | markdown | **What we found** — 2.5 findings rewritten honestly, country 0.9% fixed (items 2, 3) |
| 33 | code | # 3.2 Where the fill values sit: a blank becomes a typical value, not  — fill-value marks black not orange; takeaway softened (items 5, 10) |
| 34 | markdown | **What we found** — 3.2 wording: evidence, close not indistinguishable (item 5) |
| 49 | markdown | ## 6. Compare models and choose one — model search table copied from docs (item 19) |
| 52 | code | # 6.1 Mean ± spread of PR-AUC across the 15 folds for each candidate ( — chosen model = black outline, not orange (item 10); shipped label (item 10); title colour wording (item 10) |
| 53 | markdown | **What we found** — 6.1 findings trimmed to the CV verdict; leaderboard moved to 6.3 (item 9) |
| 55 | code | # 6.2 Same logistic regression, same 3x5 folds, with class_weight="bal — 6.2 independent y-axes (item 11); 6.2 y-label on both panels (item 11) |
| 57 | markdown | ### 6.3 Public leaderboard check — new 6.3 Public leaderboard check: table, CV-vs-public chart, 5-bullet justification (item 9) |
| 58 | code | # 6.3 Practice score (mean ± spread over the 15 folds, from the 6.1 ta — new or rewritten cell |
| 59 | markdown | **Implications: why logistic regression, not the 50/50 mix** — new or rewritten cell |
| 61 | markdown | ### 7.1 Can we trust the model's percentages, and where should the ala — 7.1 intro: colours, reliability panel, PR-curve how-to-read (items 10, 11, 15) |
| 62 | code | BEST = "logreg_v2 (SHIPPED)" — 7.1 calibration: colour system + log-log reliability panel (items 10, 11) |
| 63 | code | # 7.1c Precision-recall curves from pooled OOF predictions: guessing,  — new PR-curve cell with best-F1 and operating points (item 15) |
| 64 | markdown | **What we found** — 7.1 findings: PR-curve bullet (item 15) |
| 66 | code | # 7.2 Robustness: inject extra blanks into the RAW data (the private t — pooled all-data lift printed for a like-for-like comparison (item 6) |
| 67 | markdown | **What we found** — 7.2 like-for-like lift (item 6); cross-reference to 6.3 (item 9) |
| 68 | markdown | ### 7.3 How much money does the model save? — 7.3 intro colour wording (item 10); 7.3 intro: gains chart how-to-read (item 16) |
| 70 | code | c = REVIEW_COST — 7.3 bar chart: chosen = outlined green, not orange (item 10) |
| 72 | code | # 7.3 Disruption vs benefit: share of fraud dollars caught as the flag — new disruption-vs-benefit chart + per-10k outcome grid (items 16, 18) |
| 73 | code | # Calibration check for the priced-alert rule: p x amount is an EXPECT — neutral NOTE (item 1) |
| 74 | markdown | **What we found** — 7.3 findings: honest gains-chart caption (item 16) |
| 76 | markdown | ### 8.1 Which signals matter most? — 8.1 intro: colours + family/hour charts (items 10, 12) |
| 77 | code | # 8.1 Global drivers of the shipped LR (effect size on the standardise — 8.1 drivers chart colours (item 10); 8.1 family-level view + hour-of-day predicted vs actual (item 12) |
| 78 | markdown | **What we found** — 8.1 findings: family + hour charts (item 12) |
| 79 | markdown | ### 8.2 Why was this transaction flagged? — 8.2 intro matches merged table (item 14) |
| 80 | code | # 8.2 Reason codes: top-3 contributions (coefficient x standardised va — 8.2 one merged table (item 14) |
| 81 | markdown | **What we found** — FEAT principles 10 and 13, guidance not requirement (item 4) |
| 82 | markdown | ### 8.3 Who gets bothered, and is that fair? — 8.3 intro: two panels + rule-based table (item 13) |
| 83 | code | # 8.3 Fairness by country at the operating threshold (MAS FEAT): who g — 8.3 two-panel chart (FPR + recall, SG/ID outlined) and fairness table under the recommended rule (item 13) |
| 84 | markdown | **What we found** — 8.3 findings: rule-based fairness (item 13) |
| 85 | markdown | ## 9. Limitations and next steps — 9: definition stated (item 7) |
| 87 | markdown | By this definition these frauds are indistinguishable from ordinary tr — 9: 'by this definition', ±0.03 softened, imputation caveat, stress-test table (items 5, 7, 20) |

38 cells differ from the previous version.
# Final polish: changed cells in `notebooks/submission_v2.ipynb`

Cell numbers refer to `submission_v2.ipynb` (88 cells; 86 in `submission.ipynb`). Every cell not listed is byte-identical to `submission.ipynb`. Code edits are chart/label/display only; `notebooks/model.pkl` and `notebooks/submission.csv` are byte-identical to the originals (sha256 28800e81…, d04987a6…).

Review + audit pass (4 Oct): 6.3 leaderboard chart/section dropped and 6.1 findings restored from main; PR-curve guessing line flat at the base rate; 8.1 family chart dropped; 7.3/8.3 wording quotes printed numbers; every number in markdown now matches a printed output (or a rounding of one) or is labelled as copied from a docs file; team name added to the title. Reasons per cell: `git log -p src/final_polish.py`, where each edit carries its reason.

| v2 cell | Type | First line |
|---|---|---|
| 0 | markdown | # TrustGuard — Track 2: Financial Fraud Detection |
| 2 | code | # Setup — single cell, Colab-safe. Everything below uses only these imports. |
| 4 | markdown | ## 1. Understanding the problem |
| 6 | code | # 1.1 Rare by count, heavier by dollars - the two numbers behind "accuracy is th |
| 10 | markdown | **What we found** |
| 20 | markdown | **What we found** |
| 25 | code | # 2.5 Fraud rate when a column is blank vs when it is filled in |
| 26 | markdown | **What we found** |
| 33 | code | # 3.2 Where the fill values sit: a blank becomes a typical value, not an extreme |
| 34 | markdown | **What we found** |
| 49 | markdown | ## 6. Compare models and choose one |
| 52 | code | # 6.1 Mean ± spread of PR-AUC across the 15 folds for each candidate (parsed fro |
| 55 | code | # 6.2 Same logistic regression, same 3x5 folds, with class_weight="balanced": ra |
| 58 | markdown | ### 7.1 Can we trust the model's percentages, and where should the alarm go? |
| 59 | code | BEST = "logreg_v2 (SHIPPED)" |
| 60 | code | # 7.1c Precision-recall curves from pooled OOF predictions: guessing, the LightG |
| 61 | markdown | **What we found** |
| 63 | code | # 7.2 Robustness: inject extra blanks into the RAW data (the private test has ~3 |
| 64 | markdown | **What we found** |
| 65 | markdown | ### 7.3 How much money does the model save? |
| 67 | code | c = REVIEW_COST |
| 69 | code | # 7.3 Disruption vs benefit: share of fraud dollars caught as the flag rate rise |
| 70 | code | # Calibration check for the priced-alert rule: p x amount is an EXPECTED loss, s |
| 71 | markdown | **What we found** |
| 73 | markdown | ### 8.1 Which signals matter most? |
| 74 | code | # 8.1 Global drivers of the shipped LR (effect size on the standardised design m |
| 75 | markdown | **What we found** |
| 76 | markdown | ### 8.2 Why was this transaction flagged? |
| 77 | code | # 8.2 Reason codes: top-3 contributions (coefficient x standardised value, log-o |
| 78 | markdown | **What we found** |
| 79 | markdown | ### 8.3 Who gets bothered, and is that fair? |
| 80 | code | # 8.3 Fairness by country at the operating threshold (MAS FEAT): who gets disrup |
| 81 | markdown | **What we found** |
| 82 | markdown | ## 9. Limitations and next steps |
| 84 | markdown | By this definition these frauds are indistinguishable from ordinary transactions |

35 cells differ from `submission.ipynb`.

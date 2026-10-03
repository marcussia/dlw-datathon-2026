# HANDOFF: final polish of the submission notebook

Written 4 Oct 2026 so a fresh session (or a teammate) can continue without any chat history.

## Status: all 20 items DONE, VERIFY DONE, PR open, not merged

- Work lives in `notebooks/submission_v2.ipynb` (88 cells) on branch `feat/notebook-final-polish`.
  `notebooks/submission.ipynb` (86 cells, main at `6cd6461`) is untouched.
- PR #38: https://github.com/marcussia/dlw-datathon-2026/pull/38 (MERGEABLE / CLEAN vs main at last check).
- Review pass (4 Oct, early morning): dropped the 6.3 leaderboard chart/section (title contradicted its rows; 6.1 findings restored from main), fixed the PR-curve guessing line (flat at base rate), dropped the 8.1 family chart (contradicted the 8.3 country finding), 8.3/7.3 wording now quotes printed numbers. 35 cells differ from main (after the audit pass).
- In progress: nothing. Remaining: Germaine says "merge" (or Marcus pastes the changed cells (docs/final_polish_changes.md) into `submission.ipynb`).

## Items done (spec item -> v2 cells changed; cell numbers below are from before the review pass, see `docs/final_polish_changes.md` for the current ones)

Full per-cell table with reasons: `docs/final_polish_changes.md`. Full source of every changed cell: `docs/final_polish_cells_to_paste.md`.

| Item | Done in cells |
|---|---|
| 1 neutral NOTE in calibration-check cell | 73 |
| 2 blank-trap wording, exec summary + 2.5 | 0, 26 |
| 3 expected frauds + binomial p-value, legend, pooled 235-row test | 25, 26 |
| 4 FEAT principles 10 and 13, guidance not requirement | 81 |
| 5 "evidence" / "close to" wording in 3.2 and 9 | 33, 34, 87 |
| 6 like-for-like pooled lift in 7.2 | 66, 67 |
| 7 fraud definition stated in 9 | 85, 87 |
| 8 exec summary numbers and wording | 0, 6 |
| 9 public-leaderboard check | kept as in main (inside the 6.1 findings); the separate 6.3 chart was reviewed and dropped |
| 10 colour system (LEGIT/FRAUD/MODEL/TRAIN/TEST, black outline = shipped) | see docs/final_polish_changes.md for current cell numbers |
| 11 6.2 independent y-axes; 7.1 log-log reliability panel | 55, 61, 62 |
| 12 8.1 feature-family and hour-of-day charts | 76, 77, 78 |
| 13 8.3 two-panel fairness + table under the p x amount > $5 rule | 82, 83, 84 |
| 14 8.2 one merged reason-code table | 79, 80 |
| 15 7.1c precision-recall curve with best-F1 and operating points | 61, 63, 64 |
| 16 7.3 disruption-vs-benefit (gains) chart and caption | 68, 72, 74 |
| 17 one-line pipeline in title cell | 0 |
| 18 problem-at-a-glance table, 2x2 cost grid, per-10k outcome grid | 4, 10, 72 |
| 19 model-search table copied from docs/model_comparison_results.md | 49 |
| 20 stress-test table copied from docs (12 rows) | 87 |

## VERIFY (already run in the session that made the changes; re-run if anything is touched)

```bash
.venv-image/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.kernel_name=dlw-image --ExecutePreprocessor.timeout=1800 --output /tmp/v2_check.ipynb notebooks/submission_v2.ipynb && shasum -a 256 notebooks/model.pkl notebooks/submission.csv
```

Expected: 35/35 code cells run, 0 errors, no SyntaxWarnings, and
`notebooks/model.pkl` sha256 starts `28800e810092ec5a`, `notebooks/submission.csv` starts `d04987a6c6b6b1ca`
(identical to the untouched `submission.ipynb`; the notebook writes into `notebooks/` because nbconvert runs it there; root-level `model.pkl`/`submission.csv` are stale dev leftovers).

`src/final_polish.py` rebuilds v2 from `submission.ipynb`; it reproduces every changed code cell.

## Hard rules that applied (keep applying)

- No change to modelling code, features, imputation constants, seeds, CV, predictions, `model.pkl`, `submission.csv`.
- Colab-safe: only libraries already imported in the notebook (plus `scipy.stats.binomtest`), no `src/` imports.
- Every number in markdown comes from a printed output, or is labelled "copied from <file>, not recomputed here".
- Escape `$` in matplotlib labels as `\\$` in normal Python strings (a bare `\$` is a SyntaxWarning).
- Edit cells with scripts targeting specific cells; do not read the whole 1 MB notebook into context.

## Still human to-do (not part of this PR)

- Team name "Top 4 MCDs" into `docs/report_draft.md` (TODO_TEAM_NAME) and the notebook title, then `python src/render_report.py`.
- Clean Colab Restart & Run All of whichever notebook is submitted, before Sun 4 Oct 11:30 AM.
- Pitch rehearsal: `docs/judge_qa.md`, `docs/pitch_slides.md`.

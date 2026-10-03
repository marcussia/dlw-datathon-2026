# Public leaderboard log — Track 2

Public scores are **indicative only**: the private test set "may contain different observations
and edge cases". We use the leaderboard to compare a few pre-registered candidates, never to tune.
CSV uploads: 100 allowed. Notebook uploads: 1 per 2h.

| # | Time (3 Oct) | File | Model | CV PR-AUC (3×5) | Public | Rank | Hash |
|---|---|---|---|---|---|---|---|
| 336 | 15:48 | submission.csv | blend LR + LightGBM (shipped in model.pkl) | 0.2351 | 0.16862 | 18 | fbcedfa53e |
| 362 | 15:54 | submission_baseline_lgbm10.csv | §5 baseline LightGBM, 10 raw features | 0.1952 | 0.16046 | 18 | 6010e5afc6 |
| 364 | 15:54 | submission_logreg_v2.csv | logistic regression, v2 features | 0.2341 | **0.17882** | **13** | 2ec43bc56a |
| 748 | 17:44 | ideaB_simple_LR.csv | logistic regression, basic columns only (no engineered features) | 0.2198 | 0.15107 | 15 | 62bacd253a |

## What it tells us
- **Pipeline works end to end** on the portal: format accepted, scored, hash matches our file.
- **The CV→public drop is not uniform.** The baseline kept 82% of its CV score, LR 76%, the blend
  72%. The v2 features' edge over the baseline shrank from about +0.04 in CV to +0.008 (blend)
  and +0.018 (LR) publicly. Most of the feature-engineering gain doesn't carry over to the
  shifted test distribution.
- **LR beat the blend by 0.010 publicly**, but they tied in CV (Germaine's 5×5: blend +0.002,
  p = 0.72), and the two rank the test rows almost identically (rank correlation 0.951). A
  single 0.01 public gap is within noise, so this is weak evidence, though it's the only
  evidence we have from the test distribution.
- Ranking order is still model-with-features > baseline on both CV and public, so nothing is
  broken.

## Decision pending (Germaine + Marcus)
Ship LR alone or keep the blend? They're tied on CV. LR leads on the public set, and it's simpler
with exact per-row reason codes (good for the 40% understanding mark). The blend has CV
test-like-slice support (+0.008, 17/25 folds).

## Notebook uploads (1 per 2h)

| Version | Time (3 Oct) | Contents | Result | Hash |
|---|---|---|---|---|
| 1 | 16:01 | prediction.ipynb + model.pkl (blend LR + LightGBM) | **passed** · reference-CSV comparison: "not applicable" | b651eaab4d |

A working submission is banked. Later model changes only re-upload `model.pkl` (same notebook,
same 29 features).

## Context: the top of the leaderboard (3 Oct, ~4:30 PM)
Rank 1 is **0.19010** (achieved 12:22 PM). Our best (LR, 0.17882) is 0.011 behind, which is
within public-set noise. This fits the conclusion from the model comparison: the dataset's
signal ceiling is around 0.24 on CV, about 0.18–0.19 on the public set, and every team is
hitting it.

## Idea B test (pre-registered rule: >= 0.185 switch, < 0.179 keep)
The simple LR scored **0.15107**, so we keep the shipped LR. The engineered features cost 0.0144 on CV
but **0.0278 publicly** when removed: they carry over to the test distribution and matter more there,
not less. That rules out "a simpler model travels better". Ideas A (blank-field flags, at most
+0.0002 even on train) and C (drop `big_old`, 99.9% same ranking) weren't uploaded because they
couldn't move the score.

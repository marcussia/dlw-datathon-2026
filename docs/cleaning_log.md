# Cleaning & preprocessing log — Track 2

Every transformation, with before/after and why. Protocol for all numbers: repeated
stratified **3×5-fold CV** (seed 42), per-fold PR-AUC mean ± std; "+blanks" = blanks
injected into validation rows at the test set's per-column rates, run through the full
pipeline; "test-like" = PR-AUC on the 20% of train rows most resembling the test set
(top quintile of adversarial P(test); 4,000 rows, 124 frauds, 3.10%).

All numbers reproduce from `src/experiments_cleaning.py` (run with `--features`). The
notebook's §2d/§5b/§7 re-derive the same checks inline; its adversarial model sees raw
NaNs rather than imputed values, so its slice is slightly different (136 frauds, 3.40%,
PR-AUC 0.3809 = 11.2× lift) — same rule, same conclusion.

## Steps

| # | Step | Before → after | Why |
|---|---|---|---|
| 1 | Quality audit | 20,000×12 → unchanged | No duplicate IDs/rows, hours 0–23, amounts > 0, `txns_1h ≤ txns_24h` always. Nothing to repair. |
| 2 | Quirk documented, not fixed | 234 train / 762 test rows | `txns_24h==1` but `spend_24h>0` — inconsistent spend-window definition. Ratio features would use `+1` guards; no row dropped. |
| 3 | Drop `id` | 12 → 11 cols | Fraud rate flat (1.6–2.1%) across ID quintiles; carries no signal. |
| 4 | Impute all 8 blank-able columns with hard-coded train medians/modes | 0.5–0.8% blanks per col → 0 | Kills the "blank ⇒ never fraud" artefact (0/113 blank merchant_category, 0/124 blank new_device) that the test set, with ~3.5× more blanks, would exploit. Constants asserted against train at runtime so they can't drift. |
| 5 | Category vocabularies fixed from train | — | Deterministic; unseen category → NaN (LightGBM-native), never a crash in the organisers' sandbox. |
| 6 | No outlier removal, no resampling | — | In fraud the outliers often *are* the fraud; imbalance is handled at modelling time (class weights, compared in §6). |

## Imputation variant experiment (team decision ①)

Three fills compared under identical folds:

| Variant | PR-AUC (3×5) | +blanks | test-like | paired Δ vs (a) |
|---|---|---|---|---|
| **(a) train median/mode constants — KEPT** | **0.1952 ± 0.0448** | **0.1898 ± 0.0427** | **0.3926 ± 0.0187** | — |
| (b) neutral fills (new_device→0.0906 prevalence, merchant→"unknown" level) | 0.1921 ± 0.0431 | 0.1889 ± 0.0402 | 0.3842 ± 0.0154 | −0.0031 ± 0.0089 (10/15 folds worse) |
| (c) = (b) + masking augmentation (blank merchant/new_device in training folds at test rates) | 0.1874 ± 0.0433 | 0.1858 ± 0.0400 | 0.3769 ± 0.0123 | −0.0078 ± 0.0118 (11/15 worse; test-like −0.0157, all 3 repeats worse) |

**Why (a) wins:** a distinct "unknown" level (b) or a prevalence fill gives the model a
handle to isolate blank rows again — recreating a mild version of the leak — while
masking (c) additionally corrupts real training signal. Mode/median fill fully
camouflages blank rows among typical ones, forcing the model to judge them on their
remaining features.

## Feature experiment (team decision ③) — 8 candidates, 0 kept

Keep-rule: paired ΔPR-AUC > 1 std **and** no hurt to the test-like slice **and** no
adversarial-AUC rise. Base: 0.1952 ± 0.0448, adversarial AUC 0.653.

| feature | ΔPR-AUC ± std (paired) | Δtest-like | adv AUC | keep? |
|---|---|---|---|---|
| is_night | −0.0018 ± 0.0085 | −0.0012 | 0.653 | no |
| log_amount | +0.0000 ± 0.0000 | +0.0000 | 0.653 | no (trees are monotone-invariant) |
| amount_cents | −0.0057 ± 0.0178 | −0.0090 | 0.654 | no |
| amount_per_spend24 | −0.0008 ± 0.0184 | +0.0058 | 0.656 | no |
| amount_per_age | −0.0054 ± 0.0127 | −0.0038 | 0.653 | no |
| burst_ratio | −0.0030 ± 0.0114 | −0.0067 | 0.656 | no |
| new_device_young | −0.0057 ± 0.0131 | −0.0066 | 0.654 | no |
| spend_per_txn | −0.0028 ± 0.0149 | −0.0046 | 0.656 | no |

**Conclusion:** the 10 raw features stay. The ratios/flags are interactions LightGBM
already finds by splitting, and with only 353 positives extra features add variance,
not signal. The organisers' pre-computed velocity columns already encode the burst
pattern. (This is a deliberate, evidenced choice — see notebook §5b for the
reproducible table.)

## What never happens in this pipeline
- Nothing is fitted on test or validation rows (imputation constants, vocabularies: train only, hard-coded, asserted).
- No row removed, no outlier clipped, no SMOTE/resampling.
- No target encoding anywhere.
- `data/raw/` is read-only; no timestamps or entity IDs exist, so no causal/temporal aggregation applies (velocity features come pre-computed by the organisers).

## Headline after cleaning
LightGBM baseline: **PR-AUC 0.1952 ± 0.0448 = 11.0× lift over the 0.0177 base rate**;
+blanks 0.1898 (gap only −0.005); test-like slice 0.3926 (12.7× its 3.10% base rate).

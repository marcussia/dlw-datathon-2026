# Judge Q&A sheet — Track 2 (Round 2 booth)

Twelve questions judges are likely to ask, each with the answer and the number behind it. Every number is in
`notebooks/submission.ipynb` or `docs/model_comparison_results.md`. Say the short answer first, then the number.

**1. Why PR-AUC and not accuracy?**
Fraud is 1.77% of rows (353 of 20,000). "Never fraud" is 98.2% accurate and saves nothing. PR-AUC scores how well frauds are
ranked above legitimate transactions; it is 0.018 for guessing, 0.234 for us (13×).

**2. Why logistic regression? Isn't that too simple?**
We compared ~65 candidates on identical repeated folds with a paired statistical test: boosting, random forests, neural nets,
SVMs, stacking, blends, class weights. Everything sensible lands within noise of 0.23; stacking and tuned blends were *worse*.
LR with our 29 features reached the ceiling, is calibrated by construction, needs only scikit-learn, and gives an exact reason for
every flag. On the public test set it beat LightGBM by 0.012. Simple won on evidence, not on preference.

**3. Did you try gradient boosting / CatBoost?**
Yes. LightGBM 0.222 (one-hot beat native categoricals), XGBoost 0.212, CatBoost blend 0.238 — the best number we saw, but CatBoost
is not in the organisers' sandbox image, so it cannot ship. The LR + LightGBM blend tied LR in CV and scored lower publicly.

**4. What did you find in the data that others might miss?**
A trap: rows with a blank merchant category or device flag are *never* fraud in training (0 of 113, 0 of 124), an artefact of
generation, and the test set has 4× more blanks. A model left alone learns "blank = safe". We impute every blank with a fixed
training value so it can't. Also: the data is organiser-generated synthetic (hard caps, uniform hours, injected blanks); no public
dataset matches, which is why external data can't help and robustness is the right defence.

**5. How do you know you aren't overfitting?**
Repeated stratified CV (25 fits per model), switching models only when a paired test agreed; a pre-registered rule for leaderboard
probes (switch only above +0.02); the probes found variants +0.005 publicly that lose 0.009 in CV and break under unseen
categories, and we did not switch. Calibration on the diagonal is the final tell: an overfit model is miscalibrated somewhere.

**6. The test set is different from training. What did you do about it?**
Adversarial validation (AUC 0.65) showed the shift: bigger amounts, more new devices, 9% of rows above $2,000 mostly from old
accounts. We added a "large purchase on a long-standing account" flag (1.7% fraud, i.e. not fraud), clipped every number to its
training range, and stress-tested 12 simulated shifts: nothing crashes, worst case −0.023.

**7. Which features matter most, and why do they make sense?**
Velocity (transactions today, ≥3 in the last hour), amount relative to today's spend, amount × new device, night-time; card-present
and everyday merchants lower risk. New device on a young account is 17% fraud; new device with 5+ transactions in an hour is 38%.
It is the account-takeover pattern, and MAS's Shared Responsibility Framework mandates a 12-hour cooling-off after a new-device
login for exactly this reason. On the public set our engineered features are worth +0.028 over the raw columns.

**8. Your probabilities — do they mean anything?**
Yes: reliability curve on the diagonal (a predicted 9% is a real 9%), Brier 0.0151 vs 0.0173 for the base rate. We refused class
weights because they inflate every probability (Brier 0.023). Calibration is what makes the priced-alert policy work.

**9. How would a bank use this?**
Alert when probability × amount exceeds the review cost ($5): flags 17% of transactions, catches 56% of frauds and 92% of fraud
dollars, ~$84k saved per 10k transactions vs ~$80k for a plain "amount > $500" rule. Three tiers for the demo: approve < 0.05,
one-time password 0.05–0.20, hold ≥ 0.20. MAS itself notes that "a substantial number of flagged transactions are ultimately legitimate".

**10. Is the model fair?**
Not equally: at threshold 0.10 a legitimate Indonesian or Australian customer is flagged ~6× more often than a Singaporean, and it
catches only 9% of Singapore frauds vs 50–70% abroad, because Singapore fraud looks like normal Singapore behaviour. Dropping
`country` costs nothing in CV but ~0.013 on the public set, so we present the fairness-constrained variant as a bank's choice and
monitor the table at every retrain (MAS FEAT principles 1–3). Every flag comes with its top-3 reasons so a customer can appeal (P13).

**11. What can't it do?**
About 1 in 6 frauds has no observable signal (known device, old account, normal amount, no burst); no model catches those. There is
no customer ID or timestamp, so no per-customer baseline, the strongest real-world feature. Public ranks within ±0.03 are noise.

**12. What would you do next?**
Customer history, scheduled retraining, calibration monitoring ("does 30% still mean 30%?"), and a fairness-constrained
deployment option.

**Numbers never to get wrong:** 1.77% · 353 · 0.234 vs 0.195 · 13× · 0.179 public · 92% of dollars at 17% flagged · 6× / 9%.

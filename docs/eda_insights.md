# EDA insights — Track 2 fraud (for the pitch)

Five findings, each with the number behind it. All reproduced in `notebooks/submission.ipynb` §2.

## 1. Fraud is rare but expensive — accuracy is the wrong yardstick
Fraud is **1.77% of transactions (353 of 20,000) but ~4.0% of dollars ($201,494.76)**.
The average fraud is **$570.81 vs $246.95** for a legit transaction (2.3×). A model that
predicts "never fraud" is 98.2% accurate and saves zero dollars — which is why the
competition scores PR-AUC, and why we also report *% of fraud dollars caught* at each
threshold.

## 2. The fraud here has a classic account-takeover shape
- **New device: 7.5% fraud vs 1.2%** on a known device (6×).
- **Account younger than 90 days: 4.45%** vs 1.77% overall.
- **Activity burst (≥4 transactions in the last hour): 16.1%** fraud (≥5: 21.7%).
- Riskiest merchants: **luxury 5.3%, cash_transfer 4.7%, electronics 3.9%**.
These compound: new device × young account ≈ 17% fraud. This is the story for the demo:
"stolen credentials, logged in from a new phone, racing to move money into resellable goods."

## 3. The test set is NOT a mirror of training — we optimised for robustness
A classifier told train and test rows apart with **AUC 0.668** (0.5 = identical). The test
set has bigger amounts (**mean $524 vs $253**), more new devices (**15.4% vs 9.1%**),
**~3.5× more blanks**, and **9.0% of rows above $2,000 vs 2.3%** in train — mostly from
old accounts, which in train are rarely fraud (1.7%). The private set "may contain
different observations and edge cases", so every choice favours robustness over
leaderboard-fitting.

## 4. We found and neutralised a data trap ("blank ⇒ safe")
In train, a blank `merchant_category` (0 frauds / 113 rows) or blank `new_device`
(0 / 124) is **never** fraud — an artefact of how the data was generated, not a real
pattern. Left alone, the model learns "blank ⇒ safe" and under-scores exactly the rows
the test set has 3.5× more of. We impute every blank with hard-coded train medians/modes
(no missing-indicator flags). Effect: OOF PR-AUC **0.1757 → 0.1864**, and under extra
blanks **0.1676 → 0.1809**; originally-blank rows now score ~normally (0.0111 vs 0.0141).

## 5. There is an honest ceiling on recall
**62 of 353 frauds (~1 in 6) show no observable signal** (known device, old account, no
burst, normal amount). No customer ID or timestamps exist, so per-customer behaviour
baselines — the strongest real-world fraud features — are impossible here. We say this
out loud in Limitations: it's what we'd build first with production data.

---
*Leakage audit: no column is post-hoc (no chargeback/status/refund fields); strongest
single feature is merchant_category at ROC-AUC 0.687 — no leaks. No constant columns,
no duplicate rows. No timestamps and shuffled IDs (fraud rate flat 1.6–2.1% across ID
quintiles) → stratified 5-fold CV, not a time split.*

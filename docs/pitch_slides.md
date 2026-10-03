# Booth deck — 8 slides, one message each (Round 2, Sun 4:30 PM)

Slides are optional per the brief; we present from the notebook. This deck is the spine of the talk: one idea per slide, the
number that proves it, and what to say. Keep each slide to the headline + one visual from the notebook (cell reference given).
Target: 5 minutes, then Q&A (see `docs/judge_qa.md`).

---

## 1. Title
**TrustGuard — catching fraud without blocking customers**
Team TODO · NTU Datathon 2026 · Track 2
*Say:* "One line: we catch 92% of fraud dollars while flagging 17% of transactions, with a model simple enough to explain every decision."

## 2. The problem is a trade-off, not an accuracy contest
- Fraud = 1.77% of transactions (353 of 20,000) but ~4% of the dollars (avg fraud $571 vs $247)
- "Never fraud" is 98.2% accurate and saves $0 → we optimise **PR-AUC** and output a **probability**, not a verdict
- MAS: "a substantial number of flagged transactions are ultimately legitimate" → every false alarm has a cost
*Visual:* §2 fraud-rate bars (new device 7.5% vs 1.2%; 4+ txns/hour 16%).
*Say:* "The bank doesn't want a yes/no, it wants a risk score it can act on in tiers."

## 3. The trap in the data (and the shift)
- Blank merchant / blank device ⇒ **never** fraud in training (0/113, 0/124) — an artefact; test has 4× more blanks
- Fix: impute every blank with a fixed training value so the model can't learn "missing = safe"
- Test set is shifted: amounts 2× bigger, more new devices, 9% of rows > $2,000 mostly from old, safe accounts
*Visual:* §2d train-vs-test profile table or §2e blank-leak chart.
*Say:* "A model left alone would wave through exactly the rows the private test is full of."

## 4. Features that read like a fraud analyst
- 10 raw columns → 29 features: logs, ratios (amount vs today's spend, vs account age), velocity bursts, threshold flags
- Interactions the data shows multiply: new device × young account **17%**; new device + 5+ txns/hour **38%**
- "Big but old" flag: large purchase on a 3-year account = 1.7% fraud → not flagged
- Worth **+0.028 PR-AUC on the public test set** (0.151 raw → 0.179)
*Visual:* §8a top-drivers chart.
*Say:* "This is the account-takeover pattern — stolen credentials, new phone, racing to move money."

## 5. 65 models, one honest winner
- Identical repeated folds (25 fits per model), paired statistical test before any switch
- LR 0.234 ± 0.037 vs baseline LightGBM 0.195 ± 0.045 vs guessing 0.018 (**13×**)
- Boosting, stacking, blends, class weights, neural nets: none better; several worse. CatBoost +0.004 but not in the sandbox
- Public leaderboard agrees: LR 0.179 > blend 0.169 > baseline 0.161; #1 is 0.190 — the field is tied at the data's ceiling
*Visual:* §6 comparison table.
*Say:* "We didn't pick the simple model because we like simple. It won, and we can prove the fancy ones lost."

## 6. Probabilities you can trust
- Reliability curve on the diagonal: a predicted 9% is a real 9%; Brier 0.0151 vs 0.0173 base rate
- Class weights rejected: they inflate every probability (Brier 0.023)
- Calibration is what makes the next slide possible
*Visual:* §7a reliability chart.

## 7. How a bank would use it — price every alert
- Alert when **P(fraud) × amount > $5 review cost**: flags **17%**, catches **56% of frauds, 92% of fraud dollars**, ~$84k saved per 10k txns (vs ~$80k for "amount > $500")
- Three tiers: approve < 0.05 · one-time password 0.05–0.20 · hold ≥ 0.20
- Every flag carries its top-3 reasons ("new device; 6 txns in the last hour; account 102 days old") → customers can appeal
- Fits MAS rules: the Shared Responsibility Framework already mandates a 12-hour cooling-off after a new-device login
*Visual:* §7b policy table / §8b reason codes.

## 8. Honest limits — and what we'd do next
- 1 in 6 frauds has no visible signal; no model catches them
- Fairness (MAS FEAT): legitimate ID/AU customers flagged ~6× more than SG; only 9% of SG frauds caught. Dropping `country` is free in CV, ~0.013 on the public set → a bank's choice, monitored at every retrain
- Stress-tested 12 simulated private sets: nothing crashes, worst case −0.023
- Next: customer history, scheduled retraining, calibration monitoring
*Visual:* §8c fairness bars.
*Say:* "We'd rather tell you where it fails than have you find out."

---
**Closing line:** "Same score as the top teams, because the data has a ceiling. The difference is that we know why, we know who it
harms, and every alert comes with a reason and a price."

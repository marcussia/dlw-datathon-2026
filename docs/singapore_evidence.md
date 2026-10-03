# Singapore evidence for the pitch — what regulators and police say, mapped to our model

Purpose: the 40% "understanding" mark and the Round-2 rubric (real-world applicability, Q&A) reward tying our
signals to how fraud actually happens here. Every number below has a source and a date. The sources were
checked against the live pages on 3 Oct 2026 by a research pass; **click through before quoting in the report**.
Nothing here feeds the model — these are talking points, not features (there is no public transaction-level
fraud data to merge, and MAS/SPF publish aggregates only).

## 1. The scale of the problem (Singapore Police Force)

| Fact | Figure | Source |
| --- | --- | --- |
| Scam cases, 1H 2026 | **16,821** (−14.4% vs 1H 2025) | SPF Mid-Year Scam & Cybercrime Brief 2026 (Aug 2026) |
| Losses, 1H 2026 | **≈ S$410.6M**; median loss per case S$1,350 | same |
| Share of cases that were *self-effected* transfers (victim deceived into paying) | **80.8%** | same |
| Most common scam type by cases | E-commerce: 3,865 cases (+19.3%), ≈ S$8.3M | same |
| Full-year 2025 | **37,308 cases, ≈ S$913.1M** lost; phishing 6,264 cases / ≈ S$39.9M, "predominantly unauthorised card transactions" | SPF Annual Scam & Cybercrime Brief 2025 (Feb 2026) |
| How stolen card details get used | added to Google Pay / Apple Pay, or accounts "logged in from unfamiliar devices" | both briefs |

Links: `police.gov.sg/-/media/SPF/Media-Room/Statistics/Mid-Year-Scam-and-Cybercrime-Brief-2026/...pdf`,
`.../Annual-Scams-and-Cybercrime-Brief-2025/Annual-Scam-and-Cybercrime-Brief-2025.pdf`

**Talking point:** two different frauds live in our data. Card-style *unauthorised* fraud (new device, burst,
resellable goods) is what our model catches well. *Authorised* scam transfers (the victim pays willingly) look like
normal customer behaviour and are a big part of the "1 in 6 frauds with no visible signal". Say both.

## 2. MAS / ABS measures that match our features

| Measure | What it requires | Our feature it validates |
| --- | --- | --- |
| **Shared Responsibility Framework** (MAS/IMDA guidelines, issued 24 Oct 2024, in force 16 Dec 2024) | Banks must impose a **≥12-hour cooling-off after a digital-token activation or a login on a new device**, during which high-risk actions (add payee, raise limits, change contacts) are blocked; real-time alerts for every outgoing transaction; a 24/7 "kill switch"; **real-time fraud surveillance**, e.g. hold a transfer when an account of ≥ S$50k is drained by >50% within 24h | `new_device`, `newdev_young`, `night_newdev` — our strongest signal is the exact risk the regulator legislated against. The 24h drain rule mirrors our `spend_last_24h` / `amt_share_24h` features. |
| MAS–ABS digital-banking measures (19 Jan 2022) | ≥12h before a new soft token activates; default transfer-notification threshold **S$100 or lower**; no clickable links in bank SMS | Supports an "alert" tier below a "block" tier in our threshold policy. |
| Money Lock (SPF mid-year 2026) | ≈ 589,000 customers locked close to S$49B | Context: customers accept friction when it is targeted. |
| MAS written reply to Parliament (3 Feb 2026) | "**A substantial number of flagged transactions are ultimately legitimate**" — the security/convenience trade-off is explicit | Justifies cost-based thresholds and reporting precision alongside recall (brief: "minimise disruption"). |

Caveat to say out loud: the SRF covers account/e-payment scams, **not** cards issued in Singapore (those fall under
the ABS card code). Our dataset mixes channels, so cite it as the regulatory *direction*, not as a rule that binds every row.

Links: `mas.gov.sg/-/media/mas-media-library/regulation/guidelines/pso/guidelines-on-shared-responsibility-framework/...pdf`,
`mas.gov.sg/news/media-releases/2022/mas-and-abs-announce-measures-to-bolster-the-security-of-digital-banking`,
`mas.gov.sg/news/parliamentary-replies/2026/written-reply-to-pqs-on-srf-and-fraud-surveillance`

## 3. MAS FEAT principles — the frame for our fairness check

FEAT = Fairness, Ethics, Accountability, Transparency (MAS, 2018/2019). The principles we address directly:

- **P1** — groups are "not systematically disadvantaged … unless these decisions can be justified" → our fairness table
  (`fairness_by_group`): at threshold 0.10 a legitimate ID/AU customer is flagged ~6× more often than a SG one.
- **P2** — "use of personal attributes as input factors … is justified" → `country` is a risk signal (cross-border
  cash-out), costs only ~0.002 PR-AUC to drop; we report the no-country variant so the bank can choose.
- **P3** — regular validation "to minimize unintentional bias" → the check is a function, re-runnable at every retrain.
- **P10 / P13** — customers can appeal and get "clear explanations on what data is used … and how the data affects the
  decision" → `reason_codes()` gives the top-3 reasons per flag in plain language, exact for the LR member.

Veritas (MAS, 2021–2023) turned FEAT into an open-source toolkit; its case studies include fraud detection.
MAS consulted on AI-risk-management guidelines in Nov 2025 (fairness, explainability, human oversight, monitoring).

Links: `mas.gov.sg/-/media/MAS/News-and-Publications/Monographs-and-Information-Papers/FEAT-Principles-Updated-7-Feb-19.pdf`,
`mas.gov.sg/news/media-releases/2023/toolkit-for-responsible-use-of-ai-in-the-financial-sector`

## 4. Industry evidence on our signals and on false positives

- **Velocity bursts**: Visa's Biannual Threats Report (Fall 2025) describes fraudsters moving "to maximum velocity when
  monetizing" stolen credentials — but also warns that distributed card-testing is "low-velocity at each individual merchant",
  so velocity alone misses it. Our model combines velocity with device, age, amount and country for that reason.
- **Scale of card fraud**: Nilson Report (Jan 2026): global card fraud **$33.41B in 2024**, 6.43¢ per $100.
  UK Finance (Jun 2026): 2025 UK losses £1.28B; **£1.68B of unauthorised fraud prevented** — prevention works.
- **Night-time and young accounts** (Sift, 2015 — old vendor data, context only): fraud peaked ~3 AM; accounts under 3 days
  old were 3× more likely to be fraudulent.
- **Cost of a false positive**: Sapio/ClearSale survey (2020, ~1,000 US shoppers): **33% would never shop again** with a
  merchant that wrongly declined them. Pair with MAS's "substantial number … legitimate" line above.
- **Not found / not verified** (do not cite): Mastercard's "$13 lost to false declines per $1 of fraud"; any official
  Singapore ranking of merchant categories by fraud risk (none exists). Closest SPF detail: collectibles were 15.7% of
  e-commerce scams in 1H 2026 and gift-card scams mostly used gaming (Razer Gold) cards.

## How to use this in the notebook and report

- §1 Problem: SPF scale numbers + the two-frauds framing.
- §4 Features / §8 Explainability: SRF cooling-off ↔ `new_device` features; FEAT P13 ↔ reason codes.
- §7 Thresholds: MAS's own admission that many flags are legitimate → cost-based tiers (and Marcus's p × amount rule).
- §11 Limitations: fairness table under FEAT P1–P3; authorised-scam transfers as the unlearnable remainder.
- Report §5 (Limitations & Risks): one sentence each on fairness, calibration drift, and no customer history.

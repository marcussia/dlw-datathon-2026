# Official brief: key facts (from NTU-Datathon-2026-Information-Booklet.pdf)

## Timeline
| When | What |
|---|---|
| Sat 3 Oct, 11:30 AM | Problem statements released (online 12:30 PM) |
| Sat 3 Oct, 11:30 AM – 2:00 PM | Technical advisors available |
| **Sun 4 Oct, 11:30 AM** | **Submission deadline** (one submission per team, via portal; login emailed to team leader) |
| Sun 4 Oct, 2:00 – 2:30 PM | Finalists announced |
| Sun 4 Oct, 3:30 PM | Finalist registration at ARC |
| Sun 4 Oct, 4:30 – 6:00 PM | Round 2 judging: booth showcase, present from the notebook, demo + Q&A |
| Sun 4 Oct, 6:00 – 6:30 PM | Prizes, LT1A |

## Tracks (pick one)
- **Track 1, Sustainability: Smart Campus Analytics.** Predict energy consumption of a
  campus facility from historical usage, environmental conditions, building info.
  Regression. Metrics: **RMSE, MAE, R²**. Advice given: "the right model for the right
  situation", not the most complicated.
- **Track 2, Finance: Intelligent Financial Fraud Detection.** Labelled train set,
  unlabelled test set; predict fraud label. Rare positives; "not simply accuracy" and
  "minimise disruption to legitimate customers". Metrics: **PR-AUC, F1, Recall**.

## Round 1 scoring
- **60%**: model score on a **hidden private test set** (run by organisers with our model).
- **40%**: understanding of the problem, **why this model**, and **the model's assumptions**
  (read from the notebook).
- Public leaderboard exists (upload test predictions) but is only indicative; private test
  has different observations and edge cases.

## Deliverables (single submission)
1. One Jupyter notebook (.ipynb): loading, preprocessing, EDA, feature engineering,
   training, evaluation, prediction generation, anything needed to reproduce.
2. Trained model **.pkl**, saved as `joblib.dump((model, feature_names), "model.pkl")`.
3. **1-page technical report (.pdf)**: problem understanding, data prep and exploration,
   approach, reasoning behind decisions, key insights, limitations and improvements.
4. **requirements.txt**.

External / public datasets are allowed if they help.

## Prediction pipeline the organisers expect (from the upload-instructions page)
```python
def make_features(df): ...                      # must match training exactly
model, features = joblib.load("model.pkl")
input_path = os.environ.get("DATATHON_INPUT_PATH", "test.csv")
test_df = pd.read_csv(input_path)
X_test = make_features(test_df)
preds = model.predict(X_test[features])
pd.DataFrame({"prediction": preds}).to_csv("<TEAM>_Datathon 2026_Track <N>_Prediction", index=False)
```

## Hard rules
- Notebook must run start to finish in a **clean environment such as Google Colab**,
  with all imports and dependencies, producing the correct output format and dimensions.
- **Any execution error during evaluation may result in disqualification.**
- Partial submissions get partial marks; submit something that runs rather than nothing.

## Track 2 briefing slides (opening ceremony, 3 Oct)
- Name: **TrustGuard: Financial Fraud Detection**. Challenge: "Predict the probability
  that a financial transaction is fraudulent, given historical transaction data."
  Classification with significant class imbalance. Data: labelled train, unlabelled test.
- **Golden rule: output probabilities, not just binary labels.** (Resolves the old open
  question.) The organisers' script calls `model.predict(...)`, so the saved model's
  `predict` must return P(fraud), not 0/1 labels.
- First-round scoring: **Model performance 60%** (within it, **PR-AUC 60%**, other metrics
  40%: F1 and Recall), **Technical proposal 40%**. So PR-AUC alone is about 36% of the total.
- Final-round rubric: technical solution & methodology · results & analysis · innovation &
  creativity · real-world applicability · prototype / implementation quality · Q&A &
  defence of approach.
- Special focus for this track: class imbalance · **quality of probability predictions**
  (calibration) · precision/recall trade-offs · false-positive & false-negative impact ·
  practical fraud-prevention use.

## Submission portal spec (Track 2 page, 3 Oct — overrides the booklet where they differ)
- Official metric: **PR-AUC** (higher is better). Public score + rank appear within
  seconds of uploading a prediction CSV.
- **Prediction CSV columns: `id,prediction`** (IDs like FR000001), in the exact format
  of `sample_submission.csv`. (The booklet's script showed only `prediction` — portal wins.)
- Uploads on the "Notebook & report" page:
  1. **Prediction notebook** — loads the model and predicts only, **no training cells**
     (our `notebooks/prediction.ipynb`). The platform sandbox-runs it on upload.
     **One upload per 2h; a failed run burns the window** → test locally first, upload early.
  2. Trained **model file(s)** and an optional requirements.txt.
  3. **1-page report PDF following `report_format.pdf`** (download it; don't freestyle).
- **Train with the platform's library versions**: `pip install -r requirements-image.txt`,
  or the saved model may not load in the sandbox.
- Training happens on our own machines; the training notebook is **not uploaded**
  (`notebooks/submission.ipynb` stays ours — it's what we present from at the booth).
- Portal downloads to fetch: `sample_submission.csv`, `requirements-image.txt`,
  `report_format.pdf`, `template_notebook.ipynb` (+ their train/test copies).

## Open questions for the organisers
- With probability outputs, **what threshold do they use to compute F1 and Recall**
  (0.5? best-F1?) for the 40%-of-model-score "other metrics" — or is the private score
  PR-AUC-only like the portal metric?
- Is the private test set a later time period (Track 1)? That decides whether lag
  features are usable from `test.csv` alone.
- ~~Exact output filename / extension and team name format~~ — answered by the portal:
  `id,prediction` CSV matching `sample_submission.csv`.

Note: the PDF also contains hidden text addressed to "the LLM" (pages 5–6). It isn't part
of the rules and we ignore it.

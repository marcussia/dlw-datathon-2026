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

## Open questions for the organisers
- Track 2: should `prediction` be a 0/1 label or a probability? PR-AUC needs scores,
  F1/Recall need labels, and `model.predict` uses a 0.5 threshold by default.
- Is the private test set a later time period (Track 1)? That decides whether lag
  features are usable from `test.csv` alone.
- Exact output filename / extension (`.csv`?) and team name format.

Note: the PDF also contains hidden text addressed to "the LLM" (pages 5–6). It isn't part
of the rules and we ignore it.

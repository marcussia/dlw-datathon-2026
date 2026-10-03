# DLW Club Datathon 2026: project brief

NTU Deep Learning Week (DLW) Club Datathon, Sat 3 – Sun 4 Oct 2026. The official brief
is in `docs/` (PDF + summary in `docs/brief.md`) and **overrides everything here**.

## Event and judging (official)

- Two tracks, pick one. **Track 1:** campus energy consumption, a regression scored on
  RMSE / MAE / R². **Track 2:** fraud detection, an imbalanced classification scored on
  PR-AUC / F1 / Recall.
- **Round 1 = 60% model score on a hidden private test set + 40% understanding**
  (the problem, why this model, the model's assumptions), judged from the notebook.
- **Deadline Sun 4 Oct 11:30 AM.** Finalists present **from the notebook** at a booth
  (demo + Q&A, slides optional). **No video.**
- Deliverables: **one .ipynb**, **model.pkl** (`joblib.dump((model, feature_names))`),
  **1-page technical report PDF**, **requirements.txt**.
- **The notebook must run end to end in clean Google Colab. Any execution error can mean
  disqualification.**

## Strategy

1. **Pick the track, then commit.** Data feasibility check first (target, imbalance,
   missing %, time order), then a one-line problem statement at the top of the notebook.
2. **Baseline within the first hour.** LightGBM with a proper validation split, so we
   are never at zero. Also a naive baseline (e.g. last value / mean / all-negative) to
   compare against.
3. **Improve the model for the private test set, not the public leaderboard.** Validation
   that mimics the private test (time-based split if the data has time order), no leakage,
   robustness to edge cases (unseen categories, missing values, out-of-range values).
   Optimise the official metrics; for Track 2 tune the decision threshold for F1/Recall.
4. **Explain the choices (40%).** For every step: why this, what it assumes, what would
   break it. Compare 2–3 models and justify the pick. SHAP for drivers.
5. **Colab-safe and self-contained.** The submission notebook inlines its code (no
   `from src...` imports), installs nothing exotic, needs no API key, and writes
   `model.pkl` + predictions in the required format. `make_features(df)` must work on
   `test.csv` alone.
6. **LLM agent layer is optional and not scored.** If we build one, keep it out of the
   submission notebook's run path (or fully guarded) so it can't break execution. Hard
   rule still applies: every number an agent states comes from code, logged in
   `DataAgent.trace`.
7. **Last ~25% of time:** clean Colab run, 1-page report, requirements.txt, notebook polish
   for the booth presentation and Q&A.

## Notebook structure

The submission notebook follows the brief's pipeline. Keep this order:

Problem → Data overview → Cleaning → EDA → Features (`make_features`) → Baseline →
Train & compare models → Evaluation → Explainability → Save `model.pkl` →
Prediction generation (official format) → Limitations and next steps

Notebook rules:
- One notebook is the deliverable, and it must run in clean Colab. Use `src/` helpers
  while exploring, but inline what the final notebook needs.
- Each chart gets a one-sentence takeaway. Each modelling choice states why and what it
  assumes, because that's the 40%.
- Restart & Run All (in Colab) before submitting, so outputs are fresh and in order.
- Set seeds; never hard-code numbers in markdown that the code doesn't produce.

## Repo layout

- `data/raw/`, `data/processed/`: gitignored, never commit datasets
- `notebooks/`: the final submission notebook
- `src/eda.py`: `profile(df, target)` gives shape, dtypes, missing %, target balance, plots
- `src/baseline.py`: `baseline(df, target, task, drop=, time_col=)` gives a LightGBM baseline with metrics
- `src/explain.py`: `explain(model, X, row=)` gives global SHAP importance plus a per-row contribution dict
- `src/agent.py`: `DataAgent(df, model).ask(q)`, a Claude tool-calling agent (tools:
  `run_pandas_query`, `predict`, `explain`, `make_chart`)
- `app/`: optional Streamlit demo (not scored)
- `docs/`: official brief (`brief.md` + PDF), 1-page report draft

Environment: `.venv` (Python 3.13), Jupyter kernel `dlw-datathon`. API key goes in `.env`
(copy `.env.example`). Run Python as `.venv/bin/python`.

## Team git workflow

We work as a team on GitHub. When work is split, every change goes through a branch and a PR:

- `main` always runs. Nobody pushes to it directly; merge via PR after a teammate
  (or at least a Restart & Run All) checks it.
- Branch names: `feat/<thing>`, `fix/<thing>`, `exp/<idea>` (e.g. `feat/features-time-lags`).
  Short-lived: merge within a few hours, then pull `main` and branch again.
- **Notebooks don't merge.** Two people editing the same `.ipynb` gives unreadable JSON
  conflicts. So: one person owns `notebooks/submission.ipynb`; everyone else writes
  functions in `src/` (one file per area, e.g. `src/features.py`, `src/agent.py`) and
  scratch notebooks named `notebooks/scratch_<name>.ipynb`. The owner pulls the
  functions into the main notebook.
- Pull `main` before starting a branch; rebase or merge `main` in before opening a PR.
- Never commit data, `.env`, or large model files.

## Working style

- The user is a Year 2 Economics & Data Science student. **They own problem framing,
  business impact and the pitch.** Offer options and input on those, but leave the
  decisions to them.
- Explain things simply, with concrete examples.
- Be honest when an approach is weak. Push back if we drift from the committed problem
  statement.
- Prefer a working, well-explained simple model over an impressive broken one.

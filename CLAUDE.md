# DLW Club Datathon 2026: project brief

NTU Deep Learning Week (DLW) Club Datathon, weekend of Sat 3 – Sun 4 Oct 2026.
Problem statements and datasets are released at kickoff. Until the brief is in
`docs/`, nothing here is solution-specific.

## Event and judging

Assume last year's rubric unless the official brief says otherwise:

| Criterion | Weight |
|---|---|
| Product Value / Functionality | 30% |
| Technical Accomplishment | 30% |
| Completeness | 20% |
| Creativity / Innovation | 10% |
| Presentation | 10% |

Last year's submission was an **8-minute YouTube video** plus a **GitHub repo with a SINGLE
Jupyter notebook**, and **only the notebook was marked**. So the notebook must be clean and
well-structured, with markdown explanations and outputs left visible. It has to read well
on its own, without the video.

Likely sponsors: Micron, TikTok, Jane Street, OCBC, AI Singapore. Frame the problem in terms
the sponsor behind it would care about.

When the official brief arrives, save it to `docs/` and update this section if the rules,
rubric or submission format differ. **The official brief overrides everything here.**

## Strategy

1. **Problem first, data second.** Read the brief, then answer: who is the user, what
   decision do they make, how is success measured? Run a quick data feasibility check
   (is the target present? class imbalance? missing %?), sharpen the problem to fit the
   data, then **commit** to a one-line problem statement at the top of the notebook.
2. **Baseline within the first hour.** LightGBM with a proper validation split
   (`src/baseline.py`), so we are never at zero.
3. **Improve the model.** Feature engineering, sound validation (time-based split if the
   data has time order), no leakage, a metric that matches the business cost.
4. **Agent layer on top of the model, never instead of it.**
   - Explainer Agent: SHAP values into plain English (core).
   - Optional: Action Agent (RAG over documents into recommendations).
   - Optional: chat Q&A tool over the data.
5. **Hard rule: every number the agent states must come from running code or calling the
   model, never from the LLM's own head.** Tool calls are logged in `DataAgent.trace` so
   the notebook can show this. It is part of the pitch.
6. **"So what?" section** with business impact in $ / time / risk and the assumptions
   stated.
7. **The last ~25% of time** goes on notebook polish, the Streamlit demo (`app/`) and the
   video.

## Notebook structure

`notebooks/submission_template.ipynb` follows this order. Keep it:

Problem → Data overview → Cleaning → EDA → Features → Model → Evaluation →
Explainability → Agent layer → So what? → Next steps

Notebook rules:
- One notebook is the deliverable. Helpers in `src/` are fine, but the notebook must
  show outputs and explain every step in markdown.
- Each chart gets a one-sentence takeaway.
- Restart & Run All before submitting, so outputs are fresh and in order.
- Set seeds; never hard-code numbers in markdown that the code doesn't produce.
- If only the notebook is marked, consider inlining critical `src/` code (or printing
  it) so graders can see it.

## Repo layout

- `data/raw/`, `data/processed/`: gitignored, never commit datasets
- `notebooks/`: the final submission notebook
- `src/eda.py`: `profile(df, target)` gives shape, dtypes, missing %, target balance, plots
- `src/baseline.py`: `baseline(df, target, task, drop=, time_col=)` gives a LightGBM baseline with metrics
- `src/explain.py`: `explain(model, X, row=)` gives global SHAP importance plus a per-row contribution dict
- `src/agent.py`: `DataAgent(df, model).ask(q)`, a Claude tool-calling agent (tools:
  `run_pandas_query`, `predict`, `explain`, `make_chart`)
- `app/`: Streamlit demo
- `docs/`: problem brief, pitch notes, video script

Environment: `.venv` (Python 3.13), Jupyter kernel `dlw-datathon`. API key goes in `.env`
(copy `.env.example`). Run Python as `.venv/bin/python`.

## Working style

- The user is a Year 2 Economics & Data Science student. **They own problem framing,
  business impact and the pitch.** Offer options and input on those, but leave the
  decisions to them.
- Explain things simply, with concrete examples.
- Be honest when an approach is weak. Push back if we drift from the committed problem
  statement.
- Prefer a working, well-explained simple model over an impressive broken one.

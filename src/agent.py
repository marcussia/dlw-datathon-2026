"""Minimal tool-calling data agent on top of a trained model.

Hard rule: every number the agent states must come from a tool result
(pandas query, model prediction, SHAP), never from the LLM's own head.
The system prompt enforces this, and `trace` records every tool call so
the notebook can show where each number came from.
"""

from __future__ import annotations

import json
from pathlib import Path

import anthropic
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.baseline import prepare_X
from src.explain import explain as shap_explain

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You are a data analyst assistant for a datathon project.
You answer questions about a dataset and a trained machine-learning model using tools.

Rules:
- Every number you state must come from a tool result in this conversation. Never estimate,
  recall or invent numbers. If no tool can produce a number, say so.
- Prefer one precise pandas query over several vague ones.
- When explaining a prediction, use the `explain` tool and describe the top drivers in plain
  English for a non-technical business user.
- Keep answers short and decision-focused."""

TOOLS = [
    {
        "name": "run_pandas_query",
        "description": (
            "Evaluate a single pandas expression against the dataset and return the result. "
            "The DataFrame is `df`; `pd` and `np` are available. Must be an expression, "
            "e.g. df.groupby('region')['sales'].mean().sort_values().tail(5)"
        ),
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
    {
        "name": "predict",
        "description": (
            "Run the trained model on one row. Give either `row_id` (an index label in the "
            "dataset) or `features` (a dict of feature name -> value; missing features become NaN)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row_id": {"type": ["string", "integer"]},
                "features": {"type": "object"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "explain",
        "description": (
            "SHAP explanation. With `row_id`, returns the top features pushing that row's "
            "prediction up or down. Without it, returns global feature importance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row_id": {"type": ["string", "integer"]},
                "top_n": {"type": "integer"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "make_chart",
        "description": "Plot columns of the dataset and save a PNG. Returns the file path.",
        "input_schema": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "enum": ["bar", "line", "hist", "scatter", "box"]},
                "x": {"type": "string"},
                "y": {"type": "string"},
                "agg": {"type": "string", "enum": ["mean", "sum", "count", "median"]},
                "title": {"type": "string"},
            },
            "required": ["kind", "x"],
            "additionalProperties": False,
        },
    },
]


SAFE_BUILTINS = {f.__name__: f for f in (len, sum, min, max, abs, round, sorted, list, dict, set,
                                         tuple, range, zip, enumerate, int, float, str, bool, any, all)}


def _to_jsonable(obj, max_rows: int = 50):
    if isinstance(obj, pd.DataFrame):
        return obj.head(max_rows).to_dict(orient="split")
    if isinstance(obj, pd.Series):
        return obj.head(max_rows).to_dict()
    if isinstance(obj, np.ndarray):
        return obj[:max_rows].tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


class DataAgent:
    """Wraps a DataFrame + fitted model and lets Claude answer questions via tools.

    df: the full dataset (for queries and charts).
    model: fitted sklearn-API model (e.g. LightGBM from src.baseline).
    feature_cols: columns the model was trained on, in training order
                  (defaults to model.feature_cols_, set by src.baseline).
    """

    def __init__(self, df: pd.DataFrame, model=None, feature_cols: list[str] | None = None,
                 chart_dir: str = "outputs/charts", effort: str = "medium"):
        self.df = df
        self.model = model
        self.feature_cols = feature_cols or getattr(model, "feature_cols_", None)
        self.chart_dir = Path(chart_dir)
        self.effort = effort
        self.client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
        self.trace: list[dict] = []
        # Model-ready features, built once so every tool sees the same dtypes/categories.
        self.X = prepare_X(df[self.feature_cols]) if self.feature_cols else None

    # ---- tools -----------------------------------------------------------
    def run_pandas_query(self, expression: str):
        # NOTE: eval on model-written code. Fine for a local demo on our own data;
        # never expose this to untrusted users.
        result = eval(expression, {"__builtins__": SAFE_BUILTINS}, {"df": self.df, "pd": pd, "np": np})
        return _to_jsonable(result)

    def _key(self, row_id):
        return row_id if row_id in self.X.index else type(self.X.index[0])(row_id)

    def _row_frame(self, row_id=None, features=None) -> pd.DataFrame:
        if row_id is not None:
            return self.X.loc[[self._key(row_id)]]
        # Hypothetical row: match training dtypes, or LightGBM rejects the categoricals.
        features = features or {}
        cols = {}
        for c in self.feature_cols:
            v = features.get(c, np.nan)
            if isinstance(self.X[c].dtype, pd.CategoricalDtype):
                cols[c] = pd.Categorical([v], categories=self.X[c].cat.categories)
            else:
                cols[c] = pd.to_numeric(pd.Series([v]), errors="coerce").astype(float)
        return pd.DataFrame(cols)

    def predict(self, row_id=None, features=None):
        X = self._row_frame(row_id, features)
        out = {"prediction": _to_jsonable(self.model.predict(X)[0])}
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            out["probabilities"] = {str(c): float(p) for c, p in zip(self.model.classes_, proba)}
        return out

    def explain(self, row_id=None, top_n: int = 8):
        if row_id is not None:  # SHAP for one row only: fast on any dataset size
            key = self._key(row_id)
            return shap_explain(self.model, self.X.loc[[key]], row=key, top_n=top_n, plot=False)["row"]
        X = self.X.sample(min(len(self.X), 5000), random_state=0)  # global importance on a sample
        res = shap_explain(self.model, X, top_n=top_n, plot=False)
        return res["global_importance"].head(top_n).to_dict(orient="records")

    def make_chart(self, kind: str, x: str, y: str | None = None, agg: str = "mean", title: str | None = None):
        self.chart_dir.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        if kind == "hist":
            self.df[x].dropna().plot.hist(ax=ax, bins=30)
        elif kind == "scatter":
            self.df.plot.scatter(x=x, y=y, ax=ax, alpha=0.5)
        elif kind == "box":
            self.df.boxplot(column=y, by=x, ax=ax)
        else:
            data = self.df.groupby(x)[y].agg(agg) if y else self.df[x].value_counts()
            getattr(data.plot, kind)(ax=ax)
        ax.set_title(title or f"{kind}: {x}" + (f" vs {y}" if y else ""))
        fig.tight_layout()
        path = self.chart_dir / f"chart_{len(list(self.chart_dir.glob('*.png'))) + 1}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return {"path": str(path)}

    # ---- loop ------------------------------------------------------------
    def _run_tool(self, name: str, args: dict):
        fn = {"run_pandas_query": self.run_pandas_query, "predict": self.predict,
              "explain": self.explain, "make_chart": self.make_chart}[name]
        return fn(**args)

    def ask(self, question: str, max_turns: int = 10, verbose: bool = True) -> str:
        """Answer a question, calling tools as needed. Returns the final text."""
        messages = [{"role": "user", "content": question}]
        for _ in range(max_turns):
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                output_config={"effort": self.effort},
                messages=messages,
            )
            if response.stop_reason == "refusal":
                return "[model declined this request]"
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return "".join(b.text for b in response.content if b.type == "text")

            results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                try:
                    output, is_error = self._run_tool(block.name, dict(block.input)), False
                except Exception as e:  # report tool errors back to Claude so it can retry
                    output, is_error = f"{type(e).__name__}: {e}", True
                self.trace.append({"tool": block.name, "input": block.input, "output": output, "error": is_error})
                if verbose:
                    print(f"[tool] {block.name}({json.dumps(block.input, default=str)[:200]})")
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(output, default=str)[:20000],
                    "is_error": is_error,
                })
            messages.append({"role": "user", "content": results})
        return "[stopped: max_turns reached]"


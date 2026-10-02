"""SHAP explanations for tree models: global importance + per-row contributions."""

from __future__ import annotations

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def _shap_matrix(model, X: pd.DataFrame, class_index: int | None):
    """Return (n_rows, n_features) SHAP values and the matching base value."""
    explainer = shap.TreeExplainer(model)
    with warnings.catch_warnings():  # binary LightGBM output-format notice; handled below
        warnings.simplefilter("ignore", UserWarning)
        values = explainer.shap_values(X)
    base = np.atleast_1d(explainer.expected_value)

    if isinstance(values, list):  # older shap: one array per class
        values = np.stack(values, axis=-1)
    values = np.asarray(values)
    if values.ndim == 3:  # (rows, features, classes)
        k = values.shape[2] - 1 if class_index is None else class_index
        return values[:, :, k], float(base[k] if len(base) > 1 else base[0])
    return values, float(base[-1])


def explain(
    model,
    X: pd.DataFrame,
    row=None,
    top_n: int = 10,
    class_index: int | None = None,
    plot: bool = True,
) -> dict:
    """Explain a fitted tree model (LightGBM / XGBoost / sklearn trees).

    Returns:
      global_importance: DataFrame of mean |SHAP| per feature, sorted.
      row: if `row` (an index label in X) is given, a dict with the base value,
           the model output for that row (in SHAP's raw units, e.g. log-odds
           for classifiers), and {feature: {"value": x, "contribution": shap}}
           for the top_n features by |contribution|.
    For multiclass, class_index picks the class (default: last class).
    """
    sv, base = _shap_matrix(model, X, class_index)

    importance = (
        pd.DataFrame({"feature": X.columns, "mean_abs_shap": np.abs(sv).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    out = {"global_importance": importance, "row": None}

    if plot:
        top = importance.head(top_n).iloc[::-1]
        plt.figure(figsize=(6, 0.35 * len(top) + 1))
        plt.barh(top["feature"], top["mean_abs_shap"])
        plt.xlabel("mean |SHAP value|")
        plt.title("Global feature importance")
        plt.tight_layout()
        plt.show()

    if row is not None:
        i = X.index.get_loc(row)
        contrib = pd.Series(sv[i], index=X.columns)
        top_feats = contrib.abs().sort_values(ascending=False).head(top_n).index
        out["row"] = {
            "row": _py(row),
            "base_value": base,
            "model_output": float(base + contrib.sum()),
            "contributions": {
                f: {"value": _py(X.iloc[i][f]), "contribution": float(contrib[f])} for f in top_feats
            },
        }
    return out


def _py(v):
    """Convert numpy scalars to plain Python so the dict is JSON-serialisable."""
    if isinstance(v, np.generic):
        return v.item()
    if pd.isna(v):
        return None
    return v if isinstance(v, (int, float, str, bool)) else str(v)

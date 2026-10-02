"""One-call LightGBM baseline with a proper hold-out validation split."""

from __future__ import annotations

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split


def prepare_X(df: pd.DataFrame) -> pd.DataFrame:
    """Make a frame LightGBM can eat: object/bool columns -> pandas category."""
    X = df.copy()
    for col in X.columns:
        if X[col].dtype == object or str(X[col].dtype) == "bool":
            X[col] = X[col].astype("category")
    return X


def baseline(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    test_size: float = 0.2,
    random_state: int = 42,
    drop: list[str] | None = None,
    time_col: str | None = None,
    params: dict | None = None,
) -> dict:
    """Train a LightGBM baseline and print validation metrics.

    task: "classification" or "regression".
    drop: ID / leakage columns to exclude from features.
    time_col: if given, split by time (last `test_size` fraction = validation)
              instead of randomly. Use this whenever the data has a time order.
    15% of the training rows (the latest 15% for a time split) are held back for
    early stopping, so validation metrics stay honest.
    Returns dict with model, X_train, X_val, y_train, y_val, val_pred, metrics.
    """
    if task not in ("classification", "regression"):
        raise ValueError("task must be 'classification' or 'regression'")

    data = df.dropna(subset=[target])
    drop_cols = [target, *(drop or [])]
    if time_col:
        data = data.sort_values(time_col)
        drop_cols.append(time_col)
    X = prepare_X(data.drop(columns=drop_cols))
    y = data[target]

    if time_col:
        cut = int(len(data) * (1 - test_size))
        X_train, X_val, y_train, y_val = X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    else:
        stratify = y if task == "classification" else None
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify
        )

    # Early stopping uses its own slice of the training data, so the validation
    # metrics below come from rows the model never saw during fitting.
    if time_col:
        es_cut = int(len(X_train) * 0.85)
        X_fit, X_es, y_fit, y_es = X_train.iloc[:es_cut], X_train.iloc[es_cut:], y_train.iloc[:es_cut], y_train.iloc[es_cut:]
    else:
        stratify = y_train if task == "classification" else None
        X_fit, X_es, y_fit, y_es = train_test_split(
            X_train, y_train, test_size=0.15, random_state=random_state, stratify=stratify
        )

    base_params = {"n_estimators": 500, "learning_rate": 0.05, "random_state": random_state, "verbose": -1}
    base_params.update(params or {})
    model = (lgb.LGBMClassifier if task == "classification" else lgb.LGBMRegressor)(**base_params)
    with warnings.catch_warnings():
        # eval_set is deprecated in LightGBM 4.7, but its replacement (eval_X/eval_y)
        # does not label-encode string targets like "yes"/"no" and crashes.
        warnings.simplefilter("ignore", (FutureWarning, DeprecationWarning, UserWarning))
        model.fit(
            X_fit, y_fit,
            eval_set=[(X_es, y_es)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
    model.feature_cols_ = list(X.columns)  # LightGBM renames columns internally (spaces -> _)

    if task == "classification":
        scores, val_pred = _classification_metrics(model, X_val, y_val)
    else:
        val_pred = model.predict(X_val)
        scores = {
            "rmse": float(np.sqrt(metrics.mean_squared_error(y_val, val_pred))),
            "mae": float(metrics.mean_absolute_error(y_val, val_pred)),
            "r2": float(metrics.r2_score(y_val, val_pred)),
            "naive_mae (predict train mean)": float(metrics.mean_absolute_error(y_val, np.full(len(y_val), y_train.mean()))),
        }

    split = "time-based" if time_col else ("stratified random" if task == "classification" else "random")
    print(f"LightGBM {task} baseline | train {len(X_fit):,} + early-stop {len(X_es):,} / val {len(X_val):,} ({split} split)")
    print(f"Best iteration: {model.best_iteration_} | features: {X.shape[1]}")
    for k, v in scores.items():
        if k != "confusion_matrix":
            print(f"  {k:32s} {v:.4f}")
    if "confusion_matrix" in scores:
        print("Confusion matrix (rows = actual, cols = predicted):")
        print(scores["confusion_matrix"])

    return {
        "model": model, "feature_cols": list(X.columns), "X_train": X_train, "X_val": X_val,
        "y_train": y_train, "y_val": y_val, "val_pred": val_pred, "metrics": scores,
    }


def _classification_metrics(model, X_val, y_val):
    proba = model.predict_proba(X_val)
    pred = model.predict(X_val)
    scores = {
        "accuracy": float(metrics.accuracy_score(y_val, pred)),
        "f1_macro": float(metrics.f1_score(y_val, pred, average="macro")),
    }
    if proba.shape[1] == 2:
        pos = proba[:, 1]
        scores["roc_auc"] = float(metrics.roc_auc_score(y_val, pos))
        scores["pr_auc"] = float(metrics.average_precision_score(y_val, pos, pos_label=model.classes_[1]))
        scores["positive_rate (PR-AUC floor)"] = float((y_val == model.classes_[1]).mean())
        val_pred = pos
    else:
        scores["roc_auc_ovr"] = float(metrics.roc_auc_score(y_val, proba, multi_class="ovr"))
        val_pred = proba
    scores["confusion_matrix"] = metrics.confusion_matrix(y_val, pred)
    return scores, val_pred

"""Quick first-look profiling for any tabular dataset."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd


def profile(df: pd.DataFrame, target: str | None = None, plots: bool = True, max_plots: int = 12) -> dict:
    """Print a feasibility snapshot of `df` and return the key facts as a dict.

    Covers shape, dtypes, missing %, duplicates, target balance and quick plots.
    """
    n_rows, n_cols = df.shape
    print(f"Shape: {n_rows:,} rows x {n_cols} columns")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    print(f"Memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB\n")

    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_%": (df.isna().mean() * 100).round(2),
        "n_unique": df.nunique(),
        "example": df.iloc[0] if n_rows else None,
    }).sort_values("missing_%", ascending=False)
    print(summary.to_string(), "\n")

    result = {"shape": df.shape, "summary": summary, "target_balance": None}

    if target is not None:
        y = df[target]
        print(f"Target '{target}': {y.isna().sum():,} missing")
        if y.nunique() <= 20:
            balance = y.value_counts(normalize=True, dropna=False).mul(100).round(2)
            print("Class balance (%):")
            print(balance.to_string(), "\n")
            if balance.min() < 10:
                print("WARNING: minority class < 10%. Use stratified splits and PR-AUC / F1, not accuracy.\n")
        else:
            balance = y.describe()
            print(balance.to_string(), "\n")
        result["target_balance"] = balance

    if plots:
        _quick_plots(df, target, max_plots)
    return result


def _quick_plots(df: pd.DataFrame, target: str | None, max_plots: int) -> None:
    num_cols = [c for c in df.select_dtypes("number").columns if c != target][:max_plots]
    if num_cols:
        n = len(num_cols)
        ncols = min(4, n)
        nrows = -(-n // ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)
        for ax, col in zip(axes.flat, num_cols):
            df[col].dropna().hist(ax=ax, bins=30)
            ax.set_title(col, fontsize=9)
        for ax in axes.flat[n:]:
            ax.axis("off")
        fig.suptitle("Numeric distributions")
        fig.tight_layout()
        plt.show()

    missing = df.isna().mean().mul(100)
    missing = missing[missing > 0].sort_values()
    if len(missing):
        missing.tail(max_plots * 2).plot.barh(figsize=(6, 0.3 * min(len(missing), max_plots * 2) + 1))
        plt.title("Missing %")
        plt.tight_layout()
        plt.show()

    if target is not None and df[target].nunique() <= 20:
        df[target].value_counts().plot.bar(figsize=(5, 3), title=f"Target: {target}")
        plt.tight_layout()
        plt.show()

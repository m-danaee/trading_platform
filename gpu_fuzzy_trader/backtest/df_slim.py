"""
df_slim.py — Reduce DataFrame memory for backtest engines.

Keeps only columns required for simulation and downcasts numeric dtypes.
"""

from __future__ import annotations

import pandas as pd

from gpu_fuzzy_trader.config import (
    CONTEXT_COLUMNS,
    LABEL_COLUMNS,
    META_COLUMNS,
)


def downcast_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast label and feature columns to float32; symbol to category (inplace)."""
    out = df
    for col in LABEL_COLUMNS:
        if col in out.columns and pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype("float32")
    if "_symbol_bar_index" in out.columns:
        out["_symbol_bar_index"] = out["_symbol_bar_index"].astype("int32")
    if "symbol" in out.columns and out["symbol"].dtype == object:
        out["symbol"] = out["symbol"].astype("category")
    non_meta = [
        c
        for c in out.columns
        if c not in META_COLUMNS
        and c not in LABEL_COLUMNS
        and c != "_symbol_bar_index"
        and pd.api.types.is_numeric_dtype(out[c])
    ]
    for col in non_meta:
        if str(col).startswith("_barrier_") and str(col).endswith("_offset"):
            out[col] = out[col].fillna(-1).astype("int16")
            continue
        out[col] = out[col].astype("float32")
    return out


def slim_backtest_df(
    df: pd.DataFrame,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Return a copy with only meta, label, and requested feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Source frame (may contain many unused feature columns).
    feature_names : list[str] | None
        Feature columns to retain. When None, keeps all non-meta/label columns.
    """
    if feature_names is None:
        exclude = set(META_COLUMNS) | set(LABEL_COLUMNS)
        exclude.update(c for c in df.columns if str(c).startswith("_"))
        feature_names = [c for c in df.columns if c not in exclude]

    cols: list[str] = []
    # Internal columns carry execution metadata (bar index and exact barrier
    # outcomes).  They are never feature candidates, but must survive every
    # slim/prune step so the backtest engine can honour the production trade
    # contract after Phase 1 removes unused indicators.
    internal_columns = [c for c in df.columns if str(c).startswith("_")]
    # Mandatory trend-context columns are fixed execution conditions (not
    # ordinary features) and must survive every slimmed train/validation/
    # evaluation frame so the direction + LWC-trigger masks keep working.
    context_columns = [c for c in CONTEXT_COLUMNS if c in df.columns]
    for c in META_COLUMNS + list(LABEL_COLUMNS) + internal_columns + context_columns:
        if c in df.columns and c not in cols:
            cols.append(c)
    for name in feature_names:
        if name in df.columns and name not in cols:
            cols.append(name)

    slim = df[cols].copy()
    return downcast_numeric_df(slim)


def prune_train_columns(
    train_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:
    """Drop unused feature columns from the full training split after Phase 1."""
    return slim_backtest_df(train_df, feature_names=feature_names)

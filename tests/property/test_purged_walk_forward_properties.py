"""
Property-based tests for gpu_fuzzy_trader.data.purged_walk_forward.Purged_Walk_Forward

**Validates: Purged walk-forward invariants**

Property A: Embargo enforcement
  For any valid multi-symbol dataset and valid PWF configuration, every
  fold must satisfy: max(train._symbol_bar_index) < (min(val._symbol_bar_index)
  - purge_bars) for every symbol present in both train and val.

Property B: No overlap within fold
  Train and validation rows must be disjoint within each fold.

Property C: Chronological within fold
  For each symbol, train rows must precede validation rows in index order.
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from gpu_fuzzy_trader.data.purged_walk_forward import Purged_Walk_Forward


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

SYMBOL_POOL = ["SYM_A", "SYM_B", "SYM_C"]
MIN_ROWS = 50
MAX_ROWS = 300


@st.composite
def multi_symbol_df(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a sorted multi-symbol DataFrame with bar indices."""
    num_symbols = draw(st.integers(min_value=1, max_value=3))
    symbols = SYMBOL_POOL[:num_symbols]

    dfs = []
    for sym in symbols:
        n = draw(st.integers(min_value=MIN_ROWS, max_value=MAX_ROWS))
        base = pd.Timestamp("2020-01-01")
        rows = []
        for i in range(n):
            rows.append({
                "datetime": base + pd.Timedelta(minutes=5 * i),
                "symbol": sym,
                "feature_a": float(i),
                "_symbol_bar_index": i,
                "label_open_next": 1.0,
                "label_max_288": 1.01,
                "label_min_288": 0.99,
                "label_close_288": 1.0,
                "label_max_before_min": 1.0,
            })
        dfs.append(pd.DataFrame(rows))

    return pd.concat(dfs, ignore_index=True)


@st.composite
def pwf_config(draw: st.DrawFn) -> tuple[int, int, int]:
    """Generate valid PWF config parameters."""
    n_splits = draw(st.integers(min_value=2, max_value=6))
    purge_bars = draw(st.integers(min_value=0, max_value=100))
    min_train_folds = draw(st.integers(min_value=1, max_value=min(3, n_splits - 1)))
    return n_splits, purge_bars, min_train_folds


# ---------------------------------------------------------------------------
# Property A: Embargo enforcement
# ---------------------------------------------------------------------------


@given(df=multi_symbol_df(), cfg=pwf_config())
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_embargo_enforcement(
    df: pd.DataFrame,
    cfg: tuple[int, int, int],
) -> None:
    """
    **Property A: Embargo enforcement**

    For every fold, for every symbol present in both train and val,
    max(train._symbol_bar_index) < min(val._symbol_bar_index) - purge_bars.
    """
    n_splits, purge_bars, min_train_folds = cfg
    pwf = Purged_Walk_Forward(
        n_splits=n_splits, purge_bars=purge_bars, min_train_folds=min_train_folds)
    folds = pwf.split(df)

    for fi, (train_df, val_df) in enumerate(folds):
        for sym in train_df["symbol"].unique():
            t_sym = train_df[train_df["symbol"] == sym]
            v_sym = val_df[val_df["symbol"] == sym]
            if len(t_sym) == 0 or len(v_sym) == 0:
                continue
            t_max = t_sym["_symbol_bar_index"].max()
            v_min = v_sym["_symbol_bar_index"].min()
            assert t_max < (v_min - purge_bars), (
                f"Fold {fi}, symbol '{sym}': embargo violation — "
                f"train_max={t_max}, val_min={v_min}, purge_bars={purge_bars}, "
                f"required gap={v_min - purge_bars}"
            )


# ---------------------------------------------------------------------------
# Property B: No overlap within fold
# ---------------------------------------------------------------------------


@given(df=multi_symbol_df(), cfg=pwf_config())
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_no_overlap(
    df: pd.DataFrame,
    cfg: tuple[int, int, int],
) -> None:
    """
    **Property B: No overlap within fold**

    Train and validation rows must be disjoint.
    """
    n_splits, purge_bars, min_train_folds = cfg
    pwf = Purged_Walk_Forward(
        n_splits=n_splits, purge_bars=purge_bars, min_train_folds=min_train_folds)
    folds = pwf.split(df)

    for fi, (train_df, val_df) in enumerate(folds):
        t_keys = set(
            zip(train_df["symbol"], train_df["_symbol_bar_index"]))
        v_keys = set(
            zip(val_df["symbol"], val_df["_symbol_bar_index"]))
        overlap = t_keys & v_keys
        assert not overlap, (
            f"Fold {fi}: found {len(overlap)} overlapping rows"
        )


# ---------------------------------------------------------------------------
# Property C: Chronological within fold
# ---------------------------------------------------------------------------


@given(df=multi_symbol_df(), cfg=pwf_config())
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_chronological_order(
    df: pd.DataFrame,
    cfg: tuple[int, int, int],
) -> None:
    """
    **Property C: Chronological within fold**

    For each symbol, train rows must precede validation rows in index order.
    """
    n_splits, purge_bars, min_train_folds = cfg
    pwf = Purged_Walk_Forward(
        n_splits=n_splits, purge_bars=purge_bars, min_train_folds=min_train_folds)
    folds = pwf.split(df)

    for fi, (train_df, val_df) in enumerate(folds):
        for sym in train_df["symbol"].unique():
            t_sym = train_df[train_df["symbol"] == sym]
            v_sym = val_df[val_df["symbol"] == sym]
            if len(t_sym) == 0 or len(v_sym) == 0:
                continue
            t_max = t_sym["_symbol_bar_index"].max()
            v_min = v_sym["_symbol_bar_index"].min()
            assert t_max < v_min, (
                f"Fold {fi}, symbol '{sym}': train rows not before val rows — "
                f"train_max={t_max}, val_min={v_min}"
            )

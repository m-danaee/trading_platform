"""
Property-based tests for gpu_fuzzy_trader.data.splitter.Data_Splitter

**Validates: Requirements 2.6, 2.7**

Property 5: Per-Symbol Chronological Split with Embargo
  For any valid dataset with multiple symbols, after calling
  Data_Splitter().split_and_persist():

  1. Each symbol has exactly floor(N * HOLDOUT_TRAIN_FRACTION) rows in train.
  2. After train, HOLDOUT_EMBARGO_CANDLES bars are dropped (embargo).
  3. Remaining bars (after embargo_end) go to validation.
  4. No row appears in both train and validation sets (no overlap).
  5. Train rows are chronologically before validation rows for each symbol.
  6. The split is per-symbol (independent for each symbol).
"""

from __future__ import annotations

import math
import os
import tempfile

import pandas as pd
import pytest
from hypothesis import given, HealthCheck
from hypothesis import strategies as st

from tests.property.hypothesis_config import prop_settings

import gpu_fuzzy_trader.data.splitter as splitter_mod
import gpu_fuzzy_trader.config as config_mod
from gpu_fuzzy_trader.data.splitter import Data_Splitter


# ---------------------------------------------------------------------------
# Strategies / generators
# ---------------------------------------------------------------------------

SYMBOL_POOL = ["SYM_A", "SYM_B", "SYM_C", "SYM_D"]

# Keep row counts large enough to exercise the holdout+embargo split.
# With HOLDOUT_TRAIN_FRACTION=0.65 and HOLDOUT_EMBARGO_CANDLES=288, a symbol
# needs at least floor(N * 0.65) + 288 + 1 rows to have any validation rows.
# For a 1000-row symbol: train=650, embargo=288, val=62.
MIN_ROWS_PER_SYMBOL = 1000
MAX_ROWS_PER_SYMBOL = 2000


def _make_timestamps(n: int, base_ts: pd.Timestamp, freq_minutes: int = 5) -> list[pd.Timestamp]:
    """Return n strictly increasing timestamps starting from base_ts."""
    return [base_ts + pd.Timedelta(minutes=freq_minutes * i) for i in range(n)]


@st.composite
def valid_multi_symbol_dataframe(draw: st.DrawFn) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Hypothesis composite strategy that generates a DataFrame together with a
    dict mapping each symbol to its row count.

    Generates:
      - 2 to 4 distinct symbols
      - Each symbol has between MIN_ROWS_PER_SYMBOL and MAX_ROWS_PER_SYMBOL rows
      - All required label columns present and non-NaN
      - One feature column
      - Rows are already in chronological order per symbol (as Data_Loader
        would produce — the splitter trusts the incoming order)
    """
    num_symbols = draw(st.integers(min_value=2, max_value=len(SYMBOL_POOL)))
    symbols = SYMBOL_POOL[:num_symbols]

    rows: list[dict] = []
    symbol_counts: dict[str, int] = {}

    for sym in symbols:
        n_rows = draw(st.integers(min_value=MIN_ROWS_PER_SYMBOL, max_value=MAX_ROWS_PER_SYMBOL))
        symbol_counts[sym] = n_rows

        # Offset timestamps per symbol so they are disjoint across symbols
        sym_offset = SYMBOL_POOL.index(sym) * 100_000  # minutes
        base_ts = pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(minutes=sym_offset)
        timestamps = _make_timestamps(n_rows, base_ts)

        for i, ts in enumerate(timestamps):
            rows.append(
                {
                    "datetime": ts,
                    "symbol": sym,
                    "label_open_next": 1.0,
                    "label_close_288": 1.0,
                    "label_min_288": 0.99,
                    "label_max_288": 1.01,
                    "label_max_before_min": 1.0,
                    "feature_a": float(i),
                    "_symbol_bar_index": i,
                }
            )

    df = pd.DataFrame(rows).sort_values(["symbol", "datetime"]).reset_index(drop=True)
    return df, symbol_counts


# ---------------------------------------------------------------------------
# Helper: patch paths and run split
# ---------------------------------------------------------------------------

def _run_split(
    df: pd.DataFrame,
    tmp_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patch TRAIN_70_PATH / VALIDATION_30_PATH to tmp_path and run split."""
    original_train = config_mod.TRAIN_70_PATH
    original_val = config_mod.VALIDATION_30_PATH
    original_mode = config_mod.SPLIT_MODE

    train_path = os.path.join(tmp_path, "train_70.parquet")
    val_path = os.path.join(tmp_path, "validation_30.parquet")

    splitter_mod.TRAIN_70_PATH = train_path
    splitter_mod.VALIDATION_30_PATH = val_path
    config_mod.SPLIT_MODE = "holdout"

    try:
        train_df, val_df, _ = Data_Splitter().split_and_persist(df)
    finally:
        splitter_mod.TRAIN_70_PATH = original_train
        splitter_mod.VALIDATION_30_PATH = original_val
        config_mod.SPLIT_MODE = original_mode

    return train_df, val_df


# ---------------------------------------------------------------------------
# Property 5: Per-Symbol Chronological Split with Embargo
# Validates: Requirements 2.6, 2.7
# ---------------------------------------------------------------------------

@given(data=valid_multi_symbol_dataframe())
@prop_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_5_per_symbol_split_ratio_and_no_overlap(
    data: tuple[pd.DataFrame, dict[str, int]],
) -> None:
    """
    **Property 5: Per-Symbol Chronological Split with Embargo**
    **Validates: Requirements 2.6, 2.7**

    For any valid dataset with multiple symbols, after calling
    Data_Splitter().split_and_persist():

    1. Train = floor(N * HOLDOUT_TRAIN_FRACTION) rows per symbol.
    2. Embargo = HOLDOUT_EMBARGO_CANDLES dropped after train.
    3. Validation = rows after embargo_end (if any).
    4. No row appears in both train and validation sets (no overlap).
    5. Train rows are chronologically before validation rows for each symbol.
    6. The split is per-symbol (independent for each symbol).
    """
    train_frac = float(config_mod.HOLDOUT_TRAIN_FRACTION)
    embargo = int(config_mod.HOLDOUT_EMBARGO_CANDLES)

    df, symbol_counts = data
    with tempfile.TemporaryDirectory() as tmp_dir:
        train_df, val_df = _run_split(df, tmp_dir)

    # --- 1 & 2: Per-symbol row counts with embargo ---
    for sym, n in symbol_counts.items():
        expected_train = math.floor(n * train_frac)
        embargo_end = min(expected_train + embargo, n)
        expected_val = n - embargo_end

        actual_train = len(train_df[train_df["symbol"] == sym])
        actual_val = len(val_df[val_df["symbol"] == sym])

        assert actual_train == expected_train, (
            f"Symbol '{sym}' (N={n}): expected {expected_train} train rows "
            f"(floor({n} * {train_frac})), got {actual_train}."
        )
        assert actual_val == expected_val, (
            f"Symbol '{sym}' (N={n}): expected {expected_val} validation rows "
            f"(N - min(train_end + embargo, N) = {n} - {embargo_end}), got {actual_val}."
        )

    # --- 3: No overlap — no (symbol, datetime) pair appears in both sets ---
    train_keys = set(zip(train_df["symbol"], train_df["datetime"]))
    val_keys = set(zip(val_df["symbol"], val_df["datetime"]))

    overlap = train_keys & val_keys
    assert not overlap, (
        f"Found {len(overlap)} row(s) present in both train and validation sets. "
        f"Sample overlap keys: {list(overlap)[:5]}"
    )

    # --- 4: Train rows are chronologically before validation rows per symbol ---
    for sym in symbol_counts:
        sym_train = train_df[train_df["symbol"] == sym]
        sym_val = val_df[val_df["symbol"] == sym]

        # Only check ordering when both splits are non-empty
        if len(sym_train) == 0 or len(sym_val) == 0:
            continue

        train_max_dt = sym_train["datetime"].max()
        val_min_dt = sym_val["datetime"].min()

        assert train_max_dt < val_min_dt, (
            f"Symbol '{sym}': latest train datetime ({train_max_dt}) is not "
            f"strictly before earliest validation datetime ({val_min_dt}). "
            f"Train rows must be chronologically before validation rows."
        )

    # --- 5 & 6: Split is per-symbol (total counts sum of per-symbol splits) ---
    total_expected_train = sum(
        math.floor(n * train_frac) for n in symbol_counts.values()
    )
    total_expected_val = sum(
        n - min(math.floor(n * train_frac) + embargo, n)
        for n in symbol_counts.values()
    )

    assert len(train_df) == total_expected_train, (
        f"Total train rows: expected {total_expected_train} "
        f"(sum of per-symbol floor(N * {train_frac})), got {len(train_df)}. "
        f"This indicates the split is not being done per-symbol independently."
    )
    assert len(val_df) == total_expected_val, (
        f"Total validation rows: expected {total_expected_val} "
        f"(sum of per-symbol val after embargo), got {len(val_df)}."
    )

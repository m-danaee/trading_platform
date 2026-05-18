"""
Property-based tests for gpu_fuzzy_trader.data.loader.Data_Loader

**Validates: Requirements 2.2**

Property 1: Per-Symbol Chronological Sort
  For any valid dataset with multiple symbols and rows (each symbol having
  more than TAIL_DROP_ROWS rows), after calling Data_Loader().load_dataset(),
  the `datetime` column must be monotonically non-decreasing within each
  symbol group — regardless of the input row order.
"""

from __future__ import annotations

import os
import random
import tempfile

import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from gpu_fuzzy_trader.config import (
    INTERNAL_COLUMNS,
    LABEL_COLUMNS,
    META_COLUMNS,
    TAIL_DROP_ROWS,
)
from gpu_fuzzy_trader.data.loader import Data_Loader


# ---------------------------------------------------------------------------
# Strategies / generators
# ---------------------------------------------------------------------------

# Symbols are short strings; use a small fixed pool to keep generated CSVs
# manageable while still exercising multi-symbol behaviour.
SYMBOL_POOL = ["SYM_A", "SYM_B", "SYM_C", "SYM_D"]

# Minimum rows per symbol so that at least 1 row survives the tail drop.
MIN_ROWS_PER_SYMBOL = TAIL_DROP_ROWS + 1

# Maximum rows per symbol kept small to keep tests fast.
MAX_ROWS_PER_SYMBOL = TAIL_DROP_ROWS + 30


def _make_datetime_series(n: int, base_ts: pd.Timestamp, freq_minutes: int = 5) -> list[pd.Timestamp]:
    """Return n strictly increasing timestamps starting from base_ts."""
    return [base_ts + pd.Timedelta(minutes=freq_minutes * i) for i in range(n)]


@st.composite
def valid_multi_symbol_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """
    Hypothesis composite strategy that generates a shuffled DataFrame with:
      - 2 to 4 distinct symbols
      - Each symbol has between MIN_ROWS_PER_SYMBOL and MAX_ROWS_PER_SYMBOL rows
      - All required label columns present and non-NaN
      - At least one feature column
      - Rows are shuffled (random order) so the loader must sort them
    """
    # Choose how many symbols to include (at least 2 for multi-symbol test)
    num_symbols = draw(st.integers(min_value=2, max_value=len(SYMBOL_POOL)))
    symbols = SYMBOL_POOL[:num_symbols]

    rows: list[dict] = []

    for sym in symbols:
        n_rows = draw(st.integers(min_value=MIN_ROWS_PER_SYMBOL, max_value=MAX_ROWS_PER_SYMBOL))

        # Use a fixed base timestamp per symbol (offset by symbol index to avoid
        # any accidental cross-symbol datetime collisions)
        sym_offset = SYMBOL_POOL.index(sym) * 10_000  # minutes
        base_ts = pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(minutes=sym_offset)
        timestamps = _make_datetime_series(n_rows, base_ts)

        for ts in timestamps:
            row: dict = {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                # All label columns must be non-NaN so no rows are dropped by step 5
                "label_open_next": 1.0,
                "label_close_288": 1.0,
                "label_min_288": 0.99,
                "label_max_288": 1.01,
                "label_max_before_min": 1.0,
                # One feature column
                "feature_x": draw(st.floats(min_value=-100.0, max_value=100.0,
                                             allow_nan=False, allow_infinity=False)),
            }
            rows.append(row)

    df = pd.DataFrame(rows)

    # Shuffle rows so the loader is forced to sort them
    df = df.sample(frac=1, random_state=draw(st.integers(min_value=0, max_value=2**31 - 1)))
    df = df.reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Helper: write DataFrame to a temp CSV and load via Data_Loader
# ---------------------------------------------------------------------------

def _load_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """Persist df to a temporary CSV file and load it through Data_Loader."""
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, newline=""
    ) as f:
        df.to_csv(f, index=False)
        tmp_path = f.name
    try:
        return Data_Loader().load_dataset(tmp_path)
    finally:
        os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Property 1: Per-Symbol Chronological Sort
# Validates: Requirements 2.2
# ---------------------------------------------------------------------------

@given(raw_df=valid_multi_symbol_dataframe())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_1_per_symbol_chronological_sort(raw_df: pd.DataFrame) -> None:
    """
    **Property 1: Per-Symbol Chronological Sort**
    **Validates: Requirements 2.2**

    For any valid dataset with multiple symbols and rows (each symbol having
    more than TAIL_DROP_ROWS rows), after calling Data_Loader().load_dataset(),
    the `datetime` column must be monotonically non-decreasing within each
    symbol group — regardless of the input row order.
    """
    loaded_df = _load_from_df(raw_df)

    # The loaded DataFrame must be non-empty (each symbol has > TAIL_DROP_ROWS rows)
    assert len(loaded_df) > 0, (
        "Expected at least one row to survive after tail drop, "
        f"but got an empty DataFrame. Input had {len(raw_df)} rows."
    )

    # For every symbol present in the output, datetimes must be non-decreasing
    for symbol, group in loaded_df.groupby("symbol"):
        datetimes = group["datetime"].reset_index(drop=True)
        assert datetimes.is_monotonic_increasing, (
            f"Symbol '{symbol}': datetime column is not monotonically increasing "
            f"after load_dataset(). "
            f"First few datetimes: {datetimes.head(5).tolist()}"
        )


# ---------------------------------------------------------------------------
# Property 2: Last-288-Row Drop
# Validates: Requirements 2.3
# ---------------------------------------------------------------------------

@st.composite
def valid_multi_symbol_dataframe_with_counts(draw: st.DrawFn):
    """
    Composite strategy that generates a shuffled DataFrame together with a
    dict mapping each symbol to its original row count.  This lets the
    property test verify the exact post-drop count without re-deriving it.

    Each symbol gets between TAIL_DROP_ROWS + 1 and TAIL_DROP_ROWS + 50 rows
    so that at least 1 row survives the tail drop and the test stays fast.
    """
    num_symbols = draw(st.integers(min_value=1, max_value=len(SYMBOL_POOL)))
    symbols = SYMBOL_POOL[:num_symbols]

    rows: list[dict] = []
    original_counts: dict[str, int] = {}

    for sym in symbols:
        n_rows = draw(
            st.integers(
                min_value=TAIL_DROP_ROWS + 1,
                max_value=TAIL_DROP_ROWS + 50,
            )
        )
        original_counts[sym] = n_rows

        sym_offset = SYMBOL_POOL.index(sym) * 10_000  # minutes — keeps timestamps disjoint
        base_ts = pd.Timestamp("2021-01-01 00:00:00") + pd.Timedelta(minutes=sym_offset)
        timestamps = _make_datetime_series(n_rows, base_ts)

        for ts in timestamps:
            row: dict = {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                # All label columns non-NaN so step 5 (NaN-label drop) removes nothing
                "label_open_next": 1.0,
                "label_close_288": 1.0,
                "label_min_288": 0.99,
                "label_max_288": 1.01,
                "label_max_before_min": 1.0,
                "feature_x": draw(
                    st.floats(
                        min_value=-100.0,
                        max_value=100.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                ),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    # Shuffle so the loader must sort before dropping
    df = df.sample(
        frac=1,
        random_state=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    ).reset_index(drop=True)

    return df, original_counts


@given(data=valid_multi_symbol_dataframe_with_counts())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_2_last_288_row_drop(data) -> None:
    """
    **Property 2: Last-288-Row Drop**
    **Validates: Requirements 2.3**

    For any valid dataset where each symbol has N rows (N > TAIL_DROP_ROWS),
    after calling Data_Loader().load_dataset():

    1. Each symbol has exactly N - TAIL_DROP_ROWS rows remaining.
    2. The rows that remain are the EARLIEST rows chronologically — i.e. the
       maximum datetime in the output for a symbol equals the
       (N - TAIL_DROP_ROWS - 1)-th timestamp (0-indexed) from the sorted
       input for that symbol.
    """
    raw_df, original_counts = data

    loaded_df = _load_from_df(raw_df)

    for symbol, original_n in original_counts.items():
        expected_kept = original_n - TAIL_DROP_ROWS

        # --- 1. Row count check ---
        sym_group = loaded_df[loaded_df["symbol"] == symbol]
        actual_kept = len(sym_group)

        assert actual_kept == expected_kept, (
            f"Symbol '{symbol}': expected {expected_kept} rows after dropping "
            f"last {TAIL_DROP_ROWS} rows (original={original_n}), "
            f"but got {actual_kept} rows."
        )

        # --- 2. Earliest-rows check ---
        # Reconstruct the sorted input timestamps for this symbol so we can
        # identify the expected maximum datetime in the output.
        sym_input = raw_df[raw_df["symbol"] == symbol].copy()
        sym_input["datetime"] = pd.to_datetime(sym_input["datetime"])
        sym_input = sym_input.sort_values("datetime").reset_index(drop=True)

        # The (expected_kept - 1)-th row (0-indexed) is the last kept row.
        expected_max_dt = sym_input.loc[expected_kept - 1, "datetime"]

        actual_max_dt = sym_group["datetime"].max()

        assert actual_max_dt == expected_max_dt, (
            f"Symbol '{symbol}': the maximum datetime in the output "
            f"({actual_max_dt}) does not match the expected last-kept "
            f"timestamp ({expected_max_dt}). "
            f"This means the loader is not keeping the EARLIEST rows."
        )


# ---------------------------------------------------------------------------
# Property 3: No NaN Labels After Loading
# Validates: Requirements 2.4
# ---------------------------------------------------------------------------

@st.composite
def dataframe_with_nan_labels(draw: st.DrawFn) -> pd.DataFrame:
    """
    Composite strategy that generates a DataFrame where some rows have NaN
    values in one or more LABEL_COLUMNS.  Each symbol still has more than
    TAIL_DROP_ROWS rows so that at least one row survives the tail drop.

    The strategy deliberately injects NaN into label columns so that the
    loader's step 5 (drop NaN label rows) has real work to do.
    """
    num_symbols = draw(st.integers(min_value=1, max_value=len(SYMBOL_POOL)))
    symbols = SYMBOL_POOL[:num_symbols]

    # All five label columns that the loader must clean
    label_col_names = [
        "label_open_next",
        "label_close_288",
        "label_min_288",
        "label_max_288",
        "label_max_before_min",
    ]

    rows: list[dict] = []

    for sym in symbols:
        # Ensure enough rows survive: TAIL_DROP_ROWS dropped + at least 1 kept
        # + at least 1 NaN-label row that will be dropped by step 5.
        n_rows = draw(
            st.integers(
                min_value=TAIL_DROP_ROWS + 2,   # 1 kept + 1 NaN-label row
                max_value=TAIL_DROP_ROWS + 40,
            )
        )

        sym_offset = SYMBOL_POOL.index(sym) * 10_000
        base_ts = pd.Timestamp("2022-01-01 00:00:00") + pd.Timedelta(minutes=sym_offset)
        timestamps = _make_datetime_series(n_rows, base_ts)

        # Decide which rows (by position within this symbol) will have NaN
        # labels.  We inject NaN only into the "kept" portion (before the
        # tail drop) so that step 5 actually has to remove them.
        kept_count = n_rows - TAIL_DROP_ROWS
        # At least 1 NaN row, at most half of the kept rows
        max_nan_rows = max(1, kept_count // 2)
        nan_row_count = draw(st.integers(min_value=1, max_value=max_nan_rows))
        # Pick which positions (0-indexed within the kept portion) get NaN
        nan_positions = set(
            draw(
                st.lists(
                    st.integers(min_value=0, max_value=kept_count - 1),
                    min_size=nan_row_count,
                    max_size=nan_row_count,
                    unique=True,
                )
            )
        )

        for i, ts in enumerate(timestamps):
            # Determine if this row is in the "kept" portion and should be NaN
            in_kept_portion = i < kept_count
            make_nan = in_kept_portion and (i in nan_positions)

            # Choose which label column(s) to set to NaN for this row
            if make_nan:
                nan_col = draw(st.sampled_from(label_col_names))
            else:
                nan_col = None

            row: dict = {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                "label_open_next": float("nan") if nan_col == "label_open_next" else 1.0,
                "label_close_288": float("nan") if nan_col == "label_close_288" else 1.0,
                "label_min_288": float("nan") if nan_col == "label_min_288" else 0.99,
                "label_max_288": float("nan") if nan_col == "label_max_288" else 1.01,
                "label_max_before_min": float("nan") if nan_col == "label_max_before_min" else 1.0,
                "feature_x": draw(
                    st.floats(
                        min_value=-100.0,
                        max_value=100.0,
                        allow_nan=False,
                        allow_infinity=False,
                    )
                ),
            }
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(
        frac=1,
        random_state=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    ).reset_index(drop=True)

    return df


@given(raw_df=dataframe_with_nan_labels())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_3_no_nan_labels_after_loading(raw_df: pd.DataFrame) -> None:
    """
    **Property 3: No NaN Labels After Loading**
    **Validates: Requirements 2.4**

    For any valid dataset (including rows that have NaN in one or more label
    columns), after calling Data_Loader().load_dataset(), the resulting
    DataFrame must have NO NaN values in any of the LABEL_COLUMNS.

    This validates that step 5 of the loader (drop rows where any label
    column is NaN) works correctly regardless of which label column contains
    NaN and regardless of how many such rows exist.
    """
    loaded_df = _load_from_df(raw_df)

    # The loaded DataFrame must be non-empty
    assert len(loaded_df) > 0, (
        "Expected at least one row to survive after tail drop and NaN-label "
        f"removal, but got an empty DataFrame. Input had {len(raw_df)} rows."
    )

    # Check every label column that is present in the output
    label_cols_present = [c for c in LABEL_COLUMNS if c in loaded_df.columns]
    assert label_cols_present, (
        f"None of the expected LABEL_COLUMNS {LABEL_COLUMNS} were found in "
        f"the loaded DataFrame (columns: {list(loaded_df.columns)})."
    )

    for col in label_cols_present:
        nan_count = loaded_df[col].isna().sum()
        assert nan_count == 0, (
            f"Label column '{col}' still contains {nan_count} NaN value(s) "
            f"after load_dataset(). The loader must drop all rows where any "
            f"label column is NaN (Requirement 2.4)."
        )


# ---------------------------------------------------------------------------
# Property 4: No NaN Features After Loading
# Validates: Requirements 2.5
# ---------------------------------------------------------------------------

@st.composite
def dataframe_with_nan_features(draw: st.DrawFn) -> pd.DataFrame:
    """
    Composite strategy that generates a DataFrame where some feature columns
    randomly contain NaN values.  Each symbol still has more than
    TAIL_DROP_ROWS rows so that at least one row survives the tail drop.
    All label columns are non-NaN so that step 5 (NaN-label drop) does not
    remove any rows — this isolates the feature-NaN-fill behaviour (step 6).

    Multiple feature columns are generated so the test exercises the case
    where NaN values appear in different columns and in different rows.
    """
    num_symbols = draw(st.integers(min_value=1, max_value=len(SYMBOL_POOL)))
    symbols = SYMBOL_POOL[:num_symbols]

    # Use between 1 and 4 feature columns so we can inject NaN into a
    # random subset of them.
    num_feature_cols = draw(st.integers(min_value=1, max_value=4))
    feature_col_names = [f"feature_{chr(ord('a') + i)}" for i in range(num_feature_cols)]

    rows: list[dict] = []

    for sym in symbols:
        n_rows = draw(
            st.integers(
                min_value=TAIL_DROP_ROWS + 1,
                max_value=TAIL_DROP_ROWS + 30,
            )
        )

        sym_offset = SYMBOL_POOL.index(sym) * 10_000
        base_ts = pd.Timestamp("2023-01-01 00:00:00") + pd.Timedelta(minutes=sym_offset)
        timestamps = _make_datetime_series(n_rows, base_ts)

        # Decide which (row_index, feature_col) pairs will be NaN.
        # We allow NaN in any row (including the tail that will be dropped)
        # to ensure the loader handles NaN in all positions.
        nan_pairs: set[tuple[int, str]] = set()
        if num_feature_cols > 0:
            max_nan_cells = max(1, (n_rows * num_feature_cols) // 3)
            nan_count = draw(st.integers(min_value=1, max_value=max_nan_cells))
            for _ in range(nan_count):
                row_idx = draw(st.integers(min_value=0, max_value=n_rows - 1))
                col_name = draw(st.sampled_from(feature_col_names))
                nan_pairs.add((row_idx, col_name))

        for i, ts in enumerate(timestamps):
            row: dict = {
                "datetime": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": sym,
                # All label columns non-NaN — isolates feature NaN handling
                "label_open_next": 1.0,
                "label_close_288": 1.0,
                "label_min_288": 0.99,
                "label_max_288": 1.01,
                "label_max_before_min": 1.0,
            }
            for col in feature_col_names:
                if (i, col) in nan_pairs:
                    row[col] = float("nan")
                else:
                    row[col] = draw(
                        st.floats(
                            min_value=-100.0,
                            max_value=100.0,
                            allow_nan=False,
                            allow_infinity=False,
                        )
                    )
            rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sample(
        frac=1,
        random_state=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    ).reset_index(drop=True)

    return df


@given(raw_df=dataframe_with_nan_features())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.data_too_large],
)
def test_property_4_no_nan_features_after_loading(raw_df: pd.DataFrame) -> None:
    """
    **Property 4: No NaN Features After Loading**
    **Validates: Requirements 2.5**

    For any valid dataset (with some feature columns having NaN values),
    after calling Data_Loader().load_dataset(), the resulting DataFrame must
    have NO NaN values in any feature column.

    Feature columns are all columns that are NOT in LABEL_COLUMNS or
    META_COLUMNS.  This validates that step 6 of the loader (fill NaN in
    feature columns with 0) works correctly regardless of which feature
    column contains NaN, which rows are affected, and how many NaN values
    exist.
    """
    loaded_df = _load_from_df(raw_df)

    # The loaded DataFrame must be non-empty
    assert len(loaded_df) > 0, (
        "Expected at least one row to survive after tail drop, "
        f"but got an empty DataFrame. Input had {len(raw_df)} rows."
    )

    # Identify feature columns: everything that is not a label or meta column
    non_feature = set(LABEL_COLUMNS) | set(
        META_COLUMNS) | set(INTERNAL_COLUMNS)
    feature_cols_present = [c for c in loaded_df.columns if c not in non_feature]

    assert feature_cols_present, (
        "No feature columns found in the loaded DataFrame. "
        f"Columns present: {list(loaded_df.columns)}"
    )

    for col in feature_cols_present:
        nan_count = loaded_df[col].isna().sum()
        assert nan_count == 0, (
            f"Feature column '{col}' still contains {nan_count} NaN value(s) "
            f"after load_dataset(). The loader must fill all feature NaN "
            f"values with 0 (Requirement 2.5)."
        )

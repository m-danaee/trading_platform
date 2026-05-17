"""
Property-based tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator

**Validates: Requirements 11.2**

Property 27: Test Data Preparation Consistency
  For any valid test CSV, the output of OOS_Evaluator.prepare_test_data()
  must be identical to the output of Data_Loader.load_dataset() on the same
  file.  This guarantees that the OOS evaluation uses exactly the same data
  preparation pipeline as training (sort by symbol+datetime, drop last 288
  rows per symbol, drop NaN label rows, fill feature NaN with 0, compute
  _symbol_bar_index).
"""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from gpu_fuzzy_trader.config import LABEL_COLUMNS, TAIL_DROP_ROWS
from gpu_fuzzy_trader.data.loader import Data_Loader
from gpu_fuzzy_trader.phases.phase5_oos import OOS_Evaluator


# ---------------------------------------------------------------------------
# Strategies / generators
# ---------------------------------------------------------------------------

SYMBOL_POOL = ["SYM_A", "SYM_B", "SYM_C", "SYM_D"]

# Each symbol must have more than TAIL_DROP_ROWS rows so at least one row
# survives the tail drop.  Keep the upper bound small for test speed.
MIN_ROWS_PER_SYMBOL = TAIL_DROP_ROWS + 1
MAX_ROWS_PER_SYMBOL = TAIL_DROP_ROWS + 20


def _make_timestamps(n: int, base: pd.Timestamp, freq_minutes: int = 5) -> list[str]:
    """Return n strictly increasing timestamp strings starting from *base*."""
    return [
        (base + pd.Timedelta(minutes=freq_minutes * i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(n)
    ]


@st.composite
def valid_test_csv_dataframe(draw: st.DrawFn) -> pd.DataFrame:
    """
    Hypothesis composite strategy that generates a shuffled DataFrame
    suitable for writing to a test CSV file.

    Characteristics:
    - 1 to 4 distinct symbols
    - Each symbol has between MIN_ROWS_PER_SYMBOL and MAX_ROWS_PER_SYMBOL rows
    - All required label columns present; some rows may have NaN labels
      (to exercise the NaN-label-drop step)
    - One or more feature columns; some cells may be NaN
      (to exercise the feature-NaN-fill step)
    - Rows are shuffled so the loader must sort them
    """
    num_symbols = draw(st.integers(min_value=1, max_value=len(SYMBOL_POOL)))
    symbols = SYMBOL_POOL[:num_symbols]

    label_col_names = [
        "label_open_next",
        "label_close_288",
        "label_min_288",
        "label_max_288",
        "label_max_before_min",
    ]

    # Use 1–3 feature columns
    num_feature_cols = draw(st.integers(min_value=1, max_value=3))
    feature_col_names = [f"feature_{chr(ord('a') + i)}" for i in range(num_feature_cols)]

    rows: list[dict] = []

    for sym in symbols:
        n_rows = draw(
            st.integers(min_value=MIN_ROWS_PER_SYMBOL, max_value=MAX_ROWS_PER_SYMBOL)
        )

        # Offset timestamps per symbol to avoid cross-symbol collisions
        sym_offset = SYMBOL_POOL.index(sym) * 10_000  # minutes
        base_ts = pd.Timestamp("2020-01-01 00:00:00") + pd.Timedelta(minutes=sym_offset)
        timestamps = _make_timestamps(n_rows, base_ts)

        # Decide which rows in the *kept* portion (before tail drop) will have
        # a NaN label — exercises step 5 of the loader.
        kept_count = n_rows - TAIL_DROP_ROWS
        nan_label_positions: set[int] = set()
        if kept_count >= 2:
            max_nan = max(1, kept_count // 3)
            nan_count = draw(st.integers(min_value=0, max_value=max_nan))
            if nan_count > 0:
                nan_label_positions = set(
                    draw(
                        st.lists(
                            st.integers(min_value=0, max_value=kept_count - 1),
                            min_size=nan_count,
                            max_size=nan_count,
                            unique=True,
                        )
                    )
                )

        # Decide which (row_index, feature_col) pairs will be NaN — exercises
        # step 6 of the loader.
        nan_feature_pairs: set[tuple[int, str]] = set()
        if num_feature_cols > 0:
            max_nan_cells = max(1, (n_rows * num_feature_cols) // 4)
            nan_feat_count = draw(st.integers(min_value=0, max_value=max_nan_cells))
            for _ in range(nan_feat_count):
                ri = draw(st.integers(min_value=0, max_value=n_rows - 1))
                col = draw(st.sampled_from(feature_col_names))
                nan_feature_pairs.add((ri, col))

        for i, ts in enumerate(timestamps):
            in_kept = i < kept_count
            make_label_nan = in_kept and (i in nan_label_positions)

            # Pick one label column to NaN (if this row is a NaN-label row)
            nan_label_col = None
            if make_label_nan:
                nan_label_col = draw(st.sampled_from(label_col_names))

            row: dict = {
                "datetime": ts,
                "symbol": sym,
                "label_open_next": (
                    float("nan") if nan_label_col == "label_open_next" else 1.0
                ),
                "label_close_288": (
                    float("nan") if nan_label_col == "label_close_288" else 1.0
                ),
                "label_min_288": (
                    float("nan") if nan_label_col == "label_min_288" else 0.99
                ),
                "label_max_288": (
                    float("nan") if nan_label_col == "label_max_288" else 1.01
                ),
                "label_max_before_min": (
                    float("nan") if nan_label_col == "label_max_before_min" else 1.0
                ),
            }

            for col in feature_col_names:
                if (i, col) in nan_feature_pairs:
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

    # Shuffle rows so both loaders must sort them
    df = df.sample(
        frac=1,
        random_state=draw(st.integers(min_value=0, max_value=2**31 - 1)),
    ).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Property 27: Test Data Preparation Consistency
# Validates: Requirements 11.2
# ---------------------------------------------------------------------------


@given(raw_df=valid_test_csv_dataframe())
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_27_test_data_preparation_consistency(raw_df: pd.DataFrame) -> None:
    """
    **Property 27: Test Data Preparation Consistency**
    **Validates: Requirements 11.2**

    For any valid test CSV, the output of OOS_Evaluator.prepare_test_data()
    must be identical to the output of Data_Loader.load_dataset() on the
    same file.

    This validates that Phase 5 applies exactly the same preparation steps
    as the training pipeline:
      1. Sort by (symbol, datetime)
      2. Drop last TAIL_DROP_ROWS (288) rows per symbol
      3. Drop NaN label rows
      4. Fill feature NaN with 0
      5. Compute _symbol_bar_index
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "test_data.csv")
        raw_df.to_csv(csv_path, index=False)

        # Load via both paths
        loader_df = Data_Loader().load_dataset(csv_path)
        oos_df = OOS_Evaluator.prepare_test_data(csv_path)

    # --- Shape must match ---
    assert loader_df.shape == oos_df.shape, (
        f"Shape mismatch: Data_Loader produced {loader_df.shape}, "
        f"OOS_Evaluator produced {oos_df.shape}. "
        "Both must apply identical preparation steps."
    )

    # --- Columns must match (same set, same order) ---
    assert list(loader_df.columns) == list(oos_df.columns), (
        f"Column mismatch:\n"
        f"  Data_Loader columns : {list(loader_df.columns)}\n"
        f"  OOS_Evaluator columns: {list(oos_df.columns)}"
    )

    # --- Index must match ---
    assert list(loader_df.index) == list(oos_df.index), (
        "Index mismatch between Data_Loader and OOS_Evaluator outputs."
    )

    # --- Values must match column by column ---
    for col in loader_df.columns:
        loader_col = loader_df[col].reset_index(drop=True)
        oos_col = oos_df[col].reset_index(drop=True)

        if pd.api.types.is_numeric_dtype(loader_col):
            # Use pandas testing for numeric columns (handles NaN equality)
            try:
                pd.testing.assert_series_equal(
                    loader_col,
                    oos_col,
                    check_names=False,
                    check_exact=True,
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"Numeric column '{col}' differs between Data_Loader and "
                    f"OOS_Evaluator outputs.\n{exc}"
                ) from exc
        else:
            # Non-numeric columns (e.g. symbol, datetime)
            try:
                pd.testing.assert_series_equal(
                    loader_col,
                    oos_col,
                    check_names=False,
                )
            except AssertionError as exc:
                raise AssertionError(
                    f"Column '{col}' differs between Data_Loader and "
                    f"OOS_Evaluator outputs.\n{exc}"
                ) from exc

    # --- Spot-check: no NaN labels in either output ---
    label_cols_present = [c for c in LABEL_COLUMNS if c in oos_df.columns]
    for col in label_cols_present:
        nan_count = oos_df[col].isna().sum()
        assert nan_count == 0, (
            f"OOS_Evaluator output still has {nan_count} NaN(s) in label "
            f"column '{col}'. The preparation pipeline must drop NaN label rows."
        )

    # --- Spot-check: _symbol_bar_index starts at 0 per symbol ---
    if "_symbol_bar_index" in oos_df.columns:
        for symbol, group in oos_df.groupby("symbol"):
            min_idx = group["_symbol_bar_index"].min()
            assert min_idx == 0, (
                f"Symbol '{symbol}': _symbol_bar_index minimum is {min_idx}, "
                f"expected 0. The bar index must be reset per symbol after all drops."
            )

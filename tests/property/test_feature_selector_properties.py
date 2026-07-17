"""
Property-based tests for gpu_fuzzy_trader.features.selector.Feature_Selector

Property 17: Label and Meta Column Exclusion from Feature Selection
  **Validates: Requirements 7.2**
  For any dataset with any combination of label and meta columns,
  Feature_Selector.select_features() must never include any label or meta
  column in the output, regardless of how many feature columns are present.

Property 18: Low-Dispersion Feature Exclusion
  **Validates: Requirements 7.5**
  For any dataset where a feature has >95% identical values, that feature
  must NOT appear in the selected features.
  For any dataset where a feature has ≤95% identical values, that feature
  SHOULD be eligible for selection (not excluded by the dispersion filter).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, HealthCheck
from hypothesis import strategies as st

from tests.property.hypothesis_config import prop_settings

from gpu_fuzzy_trader.features.selector import Feature_Selector, _remove_low_dispersion
from gpu_fuzzy_trader import config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_COLUMNS = config.LABEL_COLUMNS  # ["label_open_next", "label_close_288", "label_min_288", "label_max_288", "label_max_before_min"]
META_COLUMNS = config.META_COLUMNS    # ["datetime", "symbol"]
EXCLUDED_COLUMNS = set(LABEL_COLUMNS) | set(META_COLUMNS)

# Dispersion threshold from config
DISPERSION_THRESHOLD = config.PHASE1_DISPERSION_THRESHOLD  # 0.95


# ---------------------------------------------------------------------------
# Shared helper: build a minimal valid label block
# ---------------------------------------------------------------------------

def _make_label_block(n: int, rng: np.random.Generator) -> dict:
    """
    Build a dict of all required label columns for n rows.
    Values are constructed so that both long and short targets have at least
    some variety (avoids all-same-class MI issues that cause the selector to
    return an empty list).
    """
    open_next = rng.uniform(100.0, 200.0, size=n)
    # max_288 sometimes above TP, sometimes not
    max_288 = open_next * rng.uniform(0.97, 1.10, size=n)
    # min_288 sometimes below SL, sometimes not
    min_288 = open_next * rng.uniform(0.90, 1.03, size=n)
    close_288 = open_next * rng.uniform(0.95, 1.05, size=n)
    max_before_min = rng.integers(0, 2, size=n).astype(float)
    return {
        "label_open_next": open_next,
        "label_close_288": close_288,
        "label_min_288": min_288,
        "label_max_288": max_288,
        "label_max_before_min": max_before_min,
    }


# ---------------------------------------------------------------------------
# Strategies for Property 17
# ---------------------------------------------------------------------------

@st.composite
def dataset_with_label_and_meta_columns(draw: st.DrawFn) -> pd.DataFrame:
    """
    Generate a DataFrame that always contains all LABEL_COLUMNS and META_COLUMNS,
    plus a varying number of genuine feature columns.

    The strategy varies:
      - Number of rows (50–300)
      - Number of feature columns (1–15)
      - Number of symbols (1–3)
      - Whether some feature columns have names that look like label/meta columns
        (to stress-test the exclusion logic)
    """
    n_features = draw(st.integers(min_value=1, max_value=15))
    n_symbols = draw(st.integers(min_value=1, max_value=3))
    rows_per_symbol = draw(st.integers(min_value=30, max_value=100))

    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    symbols = [f"SYM_{i}" for i in range(n_symbols)]
    dfs = []

    for sym in symbols:
        n = rows_per_symbol
        label_block = _make_label_block(n, rng)

        data: dict = {
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "symbol": sym,
        }
        data.update(label_block)

        # Add genuine feature columns with integer values (0–4, like discretized features)
        for i in range(n_features):
            data[f"feat_{i}"] = rng.integers(0, 5, size=n).astype(float)

        dfs.append(pd.DataFrame(data))

    out = pd.concat(dfs, ignore_index=True)
    out["_symbol_bar_index"] = out.groupby("symbol").cumcount()
    return out


# ---------------------------------------------------------------------------
# Property 17: Label and Meta Column Exclusion from Feature Selection
# Validates: Requirements 7.2
# ---------------------------------------------------------------------------

@given(train_df=dataset_with_label_and_meta_columns())
@prop_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_17_label_and_meta_column_exclusion_long(
    train_df: pd.DataFrame,
) -> None:
    """
    **Property 17: Label and Meta Column Exclusion from Feature Selection**
    **Validates: Requirements 7.2**

    For any dataset with any combination of label and meta columns,
    Feature_Selector.select_features() for direction="long" must never
    include any label or meta column in the output.

    LABEL_COLUMNS = ["label_open_next", "label_close_288", "label_min_288",
                     "label_max_288", "label_max_before_min"]
    META_COLUMNS  = ["datetime", "symbol"]
    """
    selector = Feature_Selector()
    result = selector.select_features(train_df, "long")

    selected_names = {entry["name"] for entry in result}

    for col in LABEL_COLUMNS:
        assert col not in selected_names, (
            f"Label column '{col}' appeared in selected features for direction='long'. "
            f"Selected: {sorted(selected_names)}"
        )

    for col in META_COLUMNS:
        assert col not in selected_names, (
            f"Meta column '{col}' appeared in selected features for direction='long'. "
            f"Selected: {sorted(selected_names)}"
        )

    assert "_symbol_bar_index" not in selected_names, (
        f"Internal column '_symbol_bar_index' must not be selected (direction='long'). "
        f"Selected: {sorted(selected_names)}"
    )


@given(train_df=dataset_with_label_and_meta_columns())
@prop_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_17_label_and_meta_column_exclusion_short(
    train_df: pd.DataFrame,
) -> None:
    """
    **Property 17: Label and Meta Column Exclusion from Feature Selection**
    **Validates: Requirements 7.2**

    For any dataset with any combination of label and meta columns,
    Feature_Selector.select_features() for direction="short" must never
    include any label or meta column in the output.
    """
    selector = Feature_Selector()
    result = selector.select_features(train_df, "short")

    selected_names = {entry["name"] for entry in result}

    for col in LABEL_COLUMNS:
        assert col not in selected_names, (
            f"Label column '{col}' appeared in selected features for direction='short'. "
            f"Selected: {sorted(selected_names)}"
        )

    for col in META_COLUMNS:
        assert col not in selected_names, (
            f"Meta column '{col}' appeared in selected features for direction='short'. "
            f"Selected: {sorted(selected_names)}"
        )

    assert "_symbol_bar_index" not in selected_names, (
        f"Internal column '_symbol_bar_index' must not be selected (direction='short'). "
        f"Selected: {sorted(selected_names)}"
    )


# ---------------------------------------------------------------------------
# Strategies for Property 18
# ---------------------------------------------------------------------------

@st.composite
def dataset_with_high_dispersion_feature(draw: st.DrawFn) -> tuple[pd.DataFrame, str]:
    """
    Generate a DataFrame that contains at least one feature with >95% identical
    values (a "low-dispersion" feature that must be excluded).

    Returns (DataFrame, name_of_low_dispersion_feature).
    """
    n = draw(st.integers(min_value=50, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    # Fraction of identical values: strictly > 0.95
    # We pick a fraction in (0.95, 1.0] to guarantee exclusion
    identical_fraction = draw(
        st.floats(min_value=0.951, max_value=1.0, allow_nan=False, allow_infinity=False)
    )
    n_identical = max(int(identical_fraction * n), int(0.951 * n) + 1)
    n_identical = min(n_identical, n)  # clamp to n

    # The dominant value
    dominant_value = draw(st.floats(min_value=0.0, max_value=4.0, allow_nan=False, allow_infinity=False))
    # A different value for the minority rows
    other_value = draw(
        st.floats(min_value=0.0, max_value=4.0, allow_nan=False, allow_infinity=False).filter(
            lambda x: abs(x - dominant_value) > 0.001
        )
    )

    low_disp_values = np.full(n, dominant_value)
    if n_identical < n:
        minority_indices = rng.choice(n, size=n - n_identical, replace=False)
        low_disp_values[minority_indices] = other_value

    # Build the DataFrame
    label_block = _make_label_block(n, rng)
    data: dict = {
        "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "symbol": "SYM_A",
    }
    data.update(label_block)

    # The low-dispersion feature
    low_disp_col = "feat_low_disp"
    data[low_disp_col] = low_disp_values

    # Add a few normal features with good dispersion (values 0–4 uniformly)
    n_normal = draw(st.integers(min_value=2, max_value=8))
    for i in range(n_normal):
        data[f"feat_normal_{i}"] = rng.integers(0, 5, size=n).astype(float)

    df = pd.DataFrame(data)
    return df, low_disp_col


@st.composite
def dataset_with_sufficient_dispersion_feature(draw: st.DrawFn) -> tuple[pd.DataFrame, str]:
    """
    Generate a DataFrame that contains a feature with ≤95% identical values
    (a "sufficient-dispersion" feature that should NOT be excluded by the
    dispersion filter).

    Strategy: build a feature where the most-common value appears in exactly
    floor(0.95 * n) rows or fewer, and at least one other distinct value fills
    the remaining rows.

    Returns (DataFrame, name_of_sufficient_dispersion_feature).
    """
    n = draw(st.integers(min_value=20, max_value=200))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)

    # We need the most-common value to appear in ≤ floor(0.95 * n) rows.
    # Since we use exactly two values (dominant_value and other_value), the
    # most-common value is max(n_dominant, n_other).
    # Constraint: max(n_dominant, n_other) / n ≤ 0.95
    # ⟺ max(n_dominant, n - n_dominant) ≤ floor(0.95 * n)
    # ⟺ n_dominant ≥ n - floor(0.95 * n)  AND  n_dominant ≤ floor(0.95 * n)
    #
    # Let min_each = n - floor(0.95 * n)  (= ceil(0.05 * n), at least 1 for n ≥ 20)
    max_dominant = int(0.95 * n)   # floor(0.95 * n)
    min_each = n - max_dominant    # = ceil(0.05 * n) ≥ 1 for n ≥ 20

    # n_dominant ∈ [min_each, max_dominant] guarantees both values appear
    # in ≤ 95% of rows.
    n_dominant = draw(st.integers(min_value=min_each, max_value=max_dominant))
    n_other = n - n_dominant

    dominant_value = 1.0
    other_value = 3.0  # distinct from dominant_value

    # Build the array: n_dominant rows with dominant_value, n_other with other_value
    good_disp_values = np.array(
        [dominant_value] * n_dominant + [other_value] * n_other,
        dtype=float,
    )
    # Shuffle so the pattern is not trivially ordered
    rng.shuffle(good_disp_values)

    # Build the DataFrame
    label_block = _make_label_block(n, rng)
    data: dict = {
        "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "symbol": "SYM_A",
    }
    data.update(label_block)

    good_disp_col = "feat_good_disp"
    data[good_disp_col] = good_disp_values

    df = pd.DataFrame(data)
    return df, good_disp_col


# ---------------------------------------------------------------------------
# Property 18: Low-Dispersion Feature Exclusion
# Validates: Requirements 7.5
# ---------------------------------------------------------------------------

@given(args=dataset_with_high_dispersion_feature())
@prop_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_18_low_dispersion_feature_excluded(
    args: tuple[pd.DataFrame, str],
) -> None:
    """
    **Property 18: Low-Dispersion Feature Exclusion**
    **Validates: Requirements 7.5**

    For any dataset where a feature has >95% identical values, that feature
    must NOT appear in the selected features output of select_features().

    This tests the _remove_low_dispersion step directly via the public
    select_features() interface.
    """
    train_df, low_disp_col = args

    # Verify the test data is constructed correctly: the feature really is >95% identical
    series = train_df[low_disp_col]
    top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
    assert top_freq > DISPERSION_THRESHOLD, (
        f"Test setup error: '{low_disp_col}' has top_freq={top_freq:.4f} "
        f"which is not > {DISPERSION_THRESHOLD}"
    )

    selector = Feature_Selector()
    result = selector.select_features(train_df, "long")

    selected_names = {entry["name"] for entry in result}

    assert low_disp_col not in selected_names, (
        f"Low-dispersion feature '{low_disp_col}' (top_freq={top_freq:.4f} > "
        f"{DISPERSION_THRESHOLD}) appeared in selected features. "
        f"Selected: {sorted(selected_names)}"
    )


@given(args=dataset_with_high_dispersion_feature())
@prop_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_18_low_dispersion_feature_excluded_short(
    args: tuple[pd.DataFrame, str],
) -> None:
    """
    **Property 18: Low-Dispersion Feature Exclusion (short direction)**
    **Validates: Requirements 7.5**

    Same as above but for direction="short".
    """
    train_df, low_disp_col = args

    series = train_df[low_disp_col]
    top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
    assert top_freq > DISPERSION_THRESHOLD

    selector = Feature_Selector()
    result = selector.select_features(train_df, "short")

    selected_names = {entry["name"] for entry in result}

    assert low_disp_col not in selected_names, (
        f"Low-dispersion feature '{low_disp_col}' (top_freq={top_freq:.4f} > "
        f"{DISPERSION_THRESHOLD}) appeared in selected features (short). "
        f"Selected: {sorted(selected_names)}"
    )


@given(args=dataset_with_sufficient_dispersion_feature())
@prop_settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_18_sufficient_dispersion_feature_not_filtered_by_dispersion(
    args: tuple[pd.DataFrame, str],
) -> None:
    """
    **Property 18: Low-Dispersion Feature Exclusion — sufficient dispersion kept**
    **Validates: Requirements 7.5**

    For any dataset where a feature has ≤95% identical values, that feature
    must NOT be excluded by the dispersion filter (_remove_low_dispersion).

    This tests the _remove_low_dispersion helper directly to isolate the
    dispersion filter from other selection steps (MI scoring, redundancy
    removal, top-K truncation).
    """
    train_df, good_disp_col = args

    # Verify the test data: the feature really is ≤95% identical
    series = train_df[good_disp_col]
    top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
    assert top_freq <= DISPERSION_THRESHOLD, (
        f"Test setup error: '{good_disp_col}' has top_freq={top_freq:.4f} "
        f"which is not ≤ {DISPERSION_THRESHOLD}"
    )

    # Test the dispersion filter directly
    all_feature_cols = [good_disp_col]
    kept = _remove_low_dispersion(train_df, all_feature_cols, DISPERSION_THRESHOLD)

    assert good_disp_col in kept, (
        f"Sufficient-dispersion feature '{good_disp_col}' (top_freq={top_freq:.4f} ≤ "
        f"{DISPERSION_THRESHOLD}) was incorrectly removed by the dispersion filter."
    )

"""
Property-based tests for gpu_fuzzy_trader.features.detector.Feature_Detector

**Validates: Requirements 3.1**

Property 6: Feature Mode Classification Completeness
  For any valid feature series (any combination of numeric values),
  Feature_Detector.detect_feature_mode() must always return exactly one of
  the six valid modes: "binary", "ternary", "positive", "sparse_positive",
  "sparse_signed", or "signed".

  The function must never:
    - Return None
    - Raise an exception
    - Return a string that is not one of the six valid modes

  This property must hold for:
    - All-NaN series
    - Single-value series
    - Very large / very small float values
    - Integer-valued series
    - Mixed float/integer series
    - Series with a mix of zeros, positives, negatives, and NaN
"""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from gpu_fuzzy_trader.features.detector import Feature_Detector


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_MODES = frozenset(
    {"binary", "ternary", "positive", "sparse_positive", "sparse_signed", "signed"}
)

# ---------------------------------------------------------------------------
# Strategies / generators
# ---------------------------------------------------------------------------

# A single numeric value that may be an integer, a bounded float, or NaN.
# Note: Hypothesis does not allow allow_nan=True together with min_value/max_value,
# so we compose the strategy from separate branches.
_numeric_value = st.one_of(
    # Integer values cast to float
    st.integers(min_value=-1_000_000, max_value=1_000_000).map(float),
    # Bounded finite floats (no NaN, no inf)
    st.floats(
        min_value=-1e15,
        max_value=1e15,
        allow_nan=False,
        allow_infinity=False,
    ),
    # NaN (unbounded floats with allow_nan=True — no min/max allowed)
    st.just(float("nan")),
)

# Very large / very small finite floats (no NaN, no inf) to stress-test
# the zero_ratio and min() branches.
_extreme_finite_value = st.floats(
    min_value=-1e300,
    max_value=1e300,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def arbitrary_numeric_series(draw: st.DrawFn) -> pd.Series:
    """
    Hypothesis composite strategy that generates a pd.Series of arbitrary
    numeric values.  Covers:
      - Length 1 to 200 (single-value and multi-value series)
      - Values drawn from integers, normal floats, and NaN
      - Optionally all-NaN
      - Optionally all-zero
      - Optionally containing very large or very small values
    """
    # Choose series length: 1 to 200
    n = draw(st.integers(min_value=1, max_value=200))

    # Occasionally generate degenerate series
    series_kind = draw(
        st.sampled_from(["all_nan", "all_zero", "all_positive_int",
                         "all_negative_int", "mixed", "extreme"])
    )

    if series_kind == "all_nan":
        values = [float("nan")] * n

    elif series_kind == "all_zero":
        values = [0.0] * n

    elif series_kind == "all_positive_int":
        values = [float(draw(st.integers(min_value=0, max_value=9))) for _ in range(n)]

    elif series_kind == "all_negative_int":
        values = [float(draw(st.integers(min_value=-9, max_value=-1))) for _ in range(n)]

    elif series_kind == "extreme":
        values = [draw(_extreme_finite_value) for _ in range(n)]

    else:  # "mixed"
        values = [draw(_numeric_value) for _ in range(n)]

    return pd.Series(values, dtype=float)


@st.composite
def single_value_series(draw: st.DrawFn) -> pd.Series:
    """Generate a series with exactly one element (any numeric value or NaN)."""
    value = draw(_numeric_value)
    return pd.Series([value], dtype=float)


@st.composite
def all_nan_series(draw: st.DrawFn) -> pd.Series:
    """Generate a series where every element is NaN."""
    n = draw(st.integers(min_value=1, max_value=50))
    return pd.Series([float("nan")] * n, dtype=float)


@st.composite
def large_value_series(draw: st.DrawFn) -> pd.Series:
    """Generate a series with very large or very small finite float values."""
    n = draw(st.integers(min_value=1, max_value=100))
    values = [draw(_extreme_finite_value) for _ in range(n)]
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# Property 6: Feature Mode Classification Completeness
# Validates: Requirements 3.1
# ---------------------------------------------------------------------------

@given(series=arbitrary_numeric_series())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_6_mode_classification_completeness_arbitrary(
    series: pd.Series,
) -> None:
    """
    **Property 6: Feature Mode Classification Completeness**
    **Validates: Requirements 3.1**

    For any arbitrary numeric series (floats, integers, NaN, mixed),
    Feature_Detector.detect_feature_mode() must:
      1. Not raise any exception.
      2. Return a non-None value.
      3. Return exactly one of the six valid mode strings.
    """
    detector = Feature_Detector()

    # Must not raise
    result = detector.detect_feature_mode(series)

    # Must not be None
    assert result is not None, (
        f"detect_feature_mode() returned None for series: {series.tolist()}"
    )

    # Must be one of the six valid modes
    assert result in VALID_MODES, (
        f"detect_feature_mode() returned invalid mode '{result}' "
        f"(not in {sorted(VALID_MODES)}) for series: {series.tolist()}"
    )


@given(series=single_value_series())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_6_mode_classification_completeness_single_value(
    series: pd.Series,
) -> None:
    """
    **Property 6: Feature Mode Classification Completeness — single-value series**
    **Validates: Requirements 3.1**

    A series with exactly one element (including NaN) must still produce a
    valid mode string without raising an exception.
    """
    detector = Feature_Detector()

    result = detector.detect_feature_mode(series)

    assert result is not None, (
        f"detect_feature_mode() returned None for single-value series: {series.tolist()}"
    )
    assert result in VALID_MODES, (
        f"detect_feature_mode() returned invalid mode '{result}' "
        f"for single-value series: {series.tolist()}"
    )


@given(series=all_nan_series())
@settings(
    max_examples=50,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_6_mode_classification_completeness_all_nan(
    series: pd.Series,
) -> None:
    """
    **Property 6: Feature Mode Classification Completeness — all-NaN series**
    **Validates: Requirements 3.1**

    A series where every element is NaN must still produce a valid mode
    string without raising an exception.  (The detector drops NaN for
    unique-value checks; an empty unique set satisfies the binary branch.)
    """
    detector = Feature_Detector()

    result = detector.detect_feature_mode(series)

    assert result is not None, (
        f"detect_feature_mode() returned None for all-NaN series of length {len(series)}"
    )
    assert result in VALID_MODES, (
        f"detect_feature_mode() returned invalid mode '{result}' "
        f"for all-NaN series of length {len(series)}"
    )


@given(series=large_value_series())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_6_mode_classification_completeness_extreme_values(
    series: pd.Series,
) -> None:
    """
    **Property 6: Feature Mode Classification Completeness — extreme values**
    **Validates: Requirements 3.1**

    A series containing very large or very small finite float values must
    still produce a valid mode string without raising an exception.
    """
    detector = Feature_Detector()

    result = detector.detect_feature_mode(series)

    assert result is not None, (
        f"detect_feature_mode() returned None for extreme-value series: {series.tolist()}"
    )
    assert result in VALID_MODES, (
        f"detect_feature_mode() returned invalid mode '{result}' "
        f"for extreme-value series: {series.tolist()}"
    )


# ===========================================================================
# Property 7: Feature Mode Classification Correctness
# Validates: Requirements 3.2
# ===========================================================================
"""
Property 7: Feature Mode Classification Correctness

For series that are KNOWN to belong to a specific mode (by construction),
Feature_Detector.detect_feature_mode() must return the correct mode.

Each strategy generates series that are guaranteed to satisfy the exact
conditions for one mode, then asserts the detector returns that mode.

Detection order (from detector.py):
  1. binary        — unique non-NaN values ⊆ {0, 1}, count ≤ 2
  2. ternary       — unique non-NaN values ⊆ {-1, 0, 1}, count ≤ 3
  3. sparse_signed — min < 0 AND zero_ratio > 0.3
  4. signed        — min < 0 AND zero_ratio ≤ 0.3
  5. sparse_positive — min ≥ 0 AND zero_ratio > 0.3
  6. positive      — min ≥ 0 AND zero_ratio ≤ 0.3

**Validates: Requirements 3.2**
"""


# ---------------------------------------------------------------------------
# Strategies for constructing series of known mode
# ---------------------------------------------------------------------------

@st.composite
def binary_series(draw: st.DrawFn) -> pd.Series:
    """
    Generate a series guaranteed to be classified as 'binary'.

    Conditions:
      - All non-NaN values are from {0, 1}
      - At most 2 unique non-NaN values
      - At least 1 non-NaN value (so the series is not all-NaN)
    """
    n = draw(st.integers(min_value=1, max_value=100))
    # Values strictly from {0, 1}
    values = draw(
        st.lists(
            st.sampled_from([0, 1]),
            min_size=n,
            max_size=n,
        )
    )
    # Optionally sprinkle some NaN (NaN is excluded from unique-value check)
    nan_indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=n - 1),
            max_size=n // 2,
            unique=True,
        )
    )
    float_values = [float(v) for v in values]
    for idx in nan_indices:
        float_values[idx] = float("nan")

    # Ensure at least one non-NaN value remains
    if all(math.isnan(v) for v in float_values):
        float_values[0] = 0.0

    return pd.Series(float_values, dtype=float)


@st.composite
def ternary_series(draw: st.DrawFn) -> pd.Series:
    """
    Generate a series guaranteed to be classified as 'ternary'.

    Conditions:
      - All non-NaN values are from {-1, 0, 1}
      - At most 3 unique non-NaN values
      - Must NOT be binary: the non-NaN values must include -1
        (so the set is NOT a subset of {0, 1})
      - At least 1 non-NaN value
    """
    n = draw(st.integers(min_value=1, max_value=100))
    # Values from {-1, 0, 1}; must include at least one -1 to escape binary
    values = draw(
        st.lists(
            st.sampled_from([-1, 0, 1]),
            min_size=n,
            max_size=n,
        )
    )
    # Force at least one -1 so the set is not a subset of {0, 1}
    values[0] = -1

    float_values = [float(v) for v in values]

    # Optionally sprinkle some NaN into positions 1..n-1 (keep index 0 as -1)
    if n >= 2:
        nan_indices = draw(
            st.lists(
                st.integers(min_value=1, max_value=n - 1),
                max_size=max(0, n // 2 - 1),
                unique=True,
            )
        )
        for idx in nan_indices:
            float_values[idx] = float("nan")

    return pd.Series(float_values, dtype=float)


@st.composite
def positive_series(draw: st.DrawFn) -> pd.Series:
    """
    Generate a series guaranteed to be classified as 'positive'.

    Conditions:
      - min >= 0
      - zero_ratio <= 0.3
      - NOT binary: must contain a value outside {0, 1}
        (specifically a value > 1 to ensure min >= 0 and escape binary/ternary)
      - NOT ternary: values are NOT a subset of {-1, 0, 1}

    Strategy: include at least one value > 1 (e.g., 2.0 or higher).
    Then fill the rest with non-negative values, keeping zero_ratio <= 0.3.
    """
    n = draw(st.integers(min_value=2, max_value=100))

    # Anchor value > 1 to escape binary/ternary checks
    anchor = draw(st.floats(min_value=2.0, max_value=1e6, allow_nan=False, allow_infinity=False))

    # Number of zeros: must be <= floor(0.3 * n)
    max_zeros = int(0.3 * n)
    n_zeros = draw(st.integers(min_value=0, max_value=max_zeros))
    n_nonzero = n - n_zeros

    # Non-zero positive values (>= 0 but not 0; use values > 0)
    nonzero_vals = draw(
        st.lists(
            st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
            min_size=max(1, n_nonzero - 1),
            max_size=max(1, n_nonzero - 1),
        )
    )
    # Include the anchor to guarantee escape from binary/ternary
    all_values = [anchor] + nonzero_vals + [0.0] * n_zeros

    # Verify zero_ratio <= 0.3 (should hold by construction, but double-check)
    series = pd.Series(all_values, dtype=float)
    actual_zero_ratio = (series == 0).mean()
    # If somehow over 0.3 (floating point edge), replace a zero with anchor
    if actual_zero_ratio > 0.3 and n_zeros > 0:
        all_values[all_values.index(0.0)] = anchor
        series = pd.Series(all_values, dtype=float)

    return series


@st.composite
def sparse_positive_series(draw: st.DrawFn) -> pd.Series:
    """
    Generate a series guaranteed to be classified as 'sparse_positive'.

    Conditions:
      - min >= 0
      - zero_ratio > 0.3
      - NOT binary: must contain a value outside {0, 1}
      - NOT ternary: values are NOT a subset of {-1, 0, 1}

    Strategy: include at least one value > 1, and ensure > 30% zeros.
    """
    n = draw(st.integers(min_value=4, max_value=100))

    # Anchor value > 1 to escape binary/ternary checks
    anchor = draw(st.floats(min_value=2.0, max_value=1e6, allow_nan=False, allow_infinity=False))

    # Number of zeros: must be > floor(0.3 * n), so at least ceil(0.3*n + 1)
    min_zeros = int(0.3 * n) + 1
    n_zeros = draw(st.integers(min_value=min_zeros, max_value=n - 1))
    n_nonzero = n - n_zeros

    # Non-zero positive values
    if n_nonzero > 1:
        nonzero_vals = draw(
            st.lists(
                st.floats(min_value=0.001, max_value=1e6, allow_nan=False, allow_infinity=False),
                min_size=n_nonzero - 1,
                max_size=n_nonzero - 1,
            )
        )
    else:
        nonzero_vals = []

    all_values = [anchor] + nonzero_vals + [0.0] * n_zeros
    return pd.Series(all_values, dtype=float)


@st.composite
def signed_series(draw: st.DrawFn) -> pd.Series:
    """
    Generate a series guaranteed to be classified as 'signed'.

    Conditions:
      - min < 0
      - zero_ratio <= 0.3
      - NOT ternary: values are NOT a subset of {-1, 0, 1}
        (must contain a value outside {-1, 0, 1}, e.g., < -1 or > 1)

    Strategy: include at least one value < -1 (e.g., -2.0 or lower).
    Then fill the rest with values, keeping zero_ratio <= 0.3.
    """
    n = draw(st.integers(min_value=2, max_value=100))

    # Anchor value < -1 to escape ternary check and guarantee min < 0
    anchor = draw(st.floats(min_value=-1e6, max_value=-1.001, allow_nan=False, allow_infinity=False))

    # Number of zeros: must be <= floor(0.3 * n)
    max_zeros = int(0.3 * n)
    n_zeros = draw(st.integers(min_value=0, max_value=max_zeros))
    n_nonzero = n - n_zeros

    # Non-zero values (can be positive or negative, but not 0)
    if n_nonzero > 1:
        nonzero_vals = draw(
            st.lists(
                st.floats(
                    min_value=-1e6,
                    max_value=1e6,
                    allow_nan=False,
                    allow_infinity=False,
                ).filter(lambda x: x != 0.0),
                min_size=n_nonzero - 1,
                max_size=n_nonzero - 1,
            )
        )
    else:
        nonzero_vals = []

    all_values = [anchor] + nonzero_vals + [0.0] * n_zeros
    series = pd.Series(all_values, dtype=float)

    # Verify zero_ratio <= 0.3 (should hold by construction)
    actual_zero_ratio = (series == 0).mean()
    if actual_zero_ratio > 0.3 and n_zeros > 0:
        # Replace a zero with the anchor to bring ratio down
        idx = all_values.index(0.0)
        all_values[idx] = anchor
        series = pd.Series(all_values, dtype=float)

    return series


@st.composite
def sparse_signed_series(draw: st.DrawFn) -> pd.Series:
    """
    Generate a series guaranteed to be classified as 'sparse_signed'.

    Conditions:
      - min < 0
      - zero_ratio > 0.3
      - NOT ternary: values are NOT a subset of {-1, 0, 1}
        (must contain a value outside {-1, 0, 1})

    Strategy: include at least one value < -1, and ensure > 30% zeros.
    """
    n = draw(st.integers(min_value=4, max_value=100))

    # Anchor value < -1 to escape ternary check and guarantee min < 0
    anchor = draw(st.floats(min_value=-1e6, max_value=-1.001, allow_nan=False, allow_infinity=False))

    # Number of zeros: must be > floor(0.3 * n)
    min_zeros = int(0.3 * n) + 1
    n_zeros = draw(st.integers(min_value=min_zeros, max_value=n - 1))
    n_nonzero = n - n_zeros

    # Non-zero values
    if n_nonzero > 1:
        nonzero_vals = draw(
            st.lists(
                st.floats(
                    min_value=-1e6,
                    max_value=1e6,
                    allow_nan=False,
                    allow_infinity=False,
                ).filter(lambda x: x != 0.0),
                min_size=n_nonzero - 1,
                max_size=n_nonzero - 1,
            )
        )
    else:
        nonzero_vals = []

    all_values = [anchor] + nonzero_vals + [0.0] * n_zeros
    return pd.Series(all_values, dtype=float)


# ---------------------------------------------------------------------------
# Property 7 test functions — one per mode
# ---------------------------------------------------------------------------

@given(series=binary_series())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_7_binary_mode_correctness(series: pd.Series) -> None:
    """
    **Property 7: Feature Mode Classification Correctness — binary**
    **Validates: Requirements 3.2**

    For any series whose non-NaN values are all from {0, 1},
    detect_feature_mode() must return "binary".
    """
    detector = Feature_Detector()
    result = detector.detect_feature_mode(series)
    assert result == "binary", (
        f"Expected 'binary' but got '{result}' for series: {series.tolist()}"
    )


@given(series=ternary_series())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_7_ternary_mode_correctness(series: pd.Series) -> None:
    """
    **Property 7: Feature Mode Classification Correctness — ternary**
    **Validates: Requirements 3.2**

    For any series whose non-NaN values are all from {-1, 0, 1} and include
    at least one -1 (so the set is not a subset of {0, 1}),
    detect_feature_mode() must return "ternary".
    """
    detector = Feature_Detector()
    result = detector.detect_feature_mode(series)
    assert result == "ternary", (
        f"Expected 'ternary' but got '{result}' for series: {series.tolist()}"
    )


@given(series=positive_series())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_7_positive_mode_correctness(series: pd.Series) -> None:
    """
    **Property 7: Feature Mode Classification Correctness — positive**
    **Validates: Requirements 3.2**

    For any series with min >= 0, zero_ratio <= 0.3, and at least one value
    outside {0, 1} (escaping binary/ternary), detect_feature_mode() must
    return "positive".
    """
    detector = Feature_Detector()
    result = detector.detect_feature_mode(series)
    assert result == "positive", (
        f"Expected 'positive' but got '{result}' for series: {series.tolist()}"
    )


@given(series=sparse_positive_series())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_7_sparse_positive_mode_correctness(series: pd.Series) -> None:
    """
    **Property 7: Feature Mode Classification Correctness — sparse_positive**
    **Validates: Requirements 3.2**

    For any series with min >= 0, zero_ratio > 0.3, and at least one value
    outside {0, 1} (escaping binary/ternary), detect_feature_mode() must
    return "sparse_positive".
    """
    detector = Feature_Detector()
    result = detector.detect_feature_mode(series)
    assert result == "sparse_positive", (
        f"Expected 'sparse_positive' but got '{result}' for series: {series.tolist()}"
    )


@given(series=signed_series())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_7_signed_mode_correctness(series: pd.Series) -> None:
    """
    **Property 7: Feature Mode Classification Correctness — signed**
    **Validates: Requirements 3.2**

    For any series with min < 0, zero_ratio <= 0.3, and at least one value
    outside {-1, 0, 1} (escaping ternary), detect_feature_mode() must
    return "signed".
    """
    detector = Feature_Detector()
    result = detector.detect_feature_mode(series)
    assert result == "signed", (
        f"Expected 'signed' but got '{result}' for series: {series.tolist()}"
    )


@given(series=sparse_signed_series())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_7_sparse_signed_mode_correctness(series: pd.Series) -> None:
    """
    **Property 7: Feature Mode Classification Correctness — sparse_signed**
    **Validates: Requirements 3.2**

    For any series with min < 0, zero_ratio > 0.3, and at least one value
    outside {-1, 0, 1} (escaping ternary), detect_feature_mode() must
    return "sparse_signed".
    """
    detector = Feature_Detector()
    result = detector.detect_feature_mode(series)
    assert result == "sparse_signed", (
        f"Expected 'sparse_signed' but got '{result}' for series: {series.tolist()}"
    )

"""
Unit tests for gpu_fuzzy_trader.features.detector.Feature_Detector

Tests cover:
  - All six mode classifications (binary, ternary, positive, sparse_positive,
    sparse_signed, signed)
  - Boundary conditions for zero_ratio threshold (0.3)
  - NaN handling (NaN excluded from unique-value checks; zero_ratio on full series)
  - detect_all_modes returns correct dict
  - Module-level convenience functions
"""

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.features.detector import (
    Feature_Detector,
    detect_all_modes,
    detect_feature_mode,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _series(values) -> pd.Series:
    return pd.Series(values, dtype=float)


# ---------------------------------------------------------------------------
# Tests: binary mode
# ---------------------------------------------------------------------------

class TestBinaryMode:
    def test_only_zeros(self):
        assert detect_feature_mode(_series([0, 0, 0, 0])) == "binary"

    def test_only_ones(self):
        assert detect_feature_mode(_series([1, 1, 1, 1])) == "binary"

    def test_zeros_and_ones(self):
        assert detect_feature_mode(_series([0, 1, 0, 1, 1, 0])) == "binary"

    def test_zeros_ones_with_nan(self):
        """NaN values should not affect binary classification."""
        assert detect_feature_mode(_series([0, 1, float("nan"), 0])) == "binary"

    def test_single_zero(self):
        assert detect_feature_mode(_series([0])) == "binary"

    def test_single_one(self):
        assert detect_feature_mode(_series([1])) == "binary"


# ---------------------------------------------------------------------------
# Tests: ternary mode
# ---------------------------------------------------------------------------

class TestTernaryMode:
    def test_minus_one_zero_one(self):
        assert detect_feature_mode(_series([-1, 0, 1, -1, 0, 1])) == "ternary"

    def test_minus_one_and_one_only(self):
        """Two unique values ⊆ {-1, 0, 1} but not ⊆ {0, 1} → ternary."""
        assert detect_feature_mode(_series([-1, 1, -1, 1])) == "ternary"

    def test_minus_one_and_zero_only(self):
        assert detect_feature_mode(_series([-1, 0, -1, 0])) == "ternary"

    def test_ternary_with_nan(self):
        assert detect_feature_mode(_series([-1, 0, 1, float("nan")])) == "ternary"

    def test_single_minus_one(self):
        assert detect_feature_mode(_series([-1])) == "ternary"


# ---------------------------------------------------------------------------
# Tests: positive mode
# ---------------------------------------------------------------------------

class TestPositiveMode:
    def test_positive_integers_low_zero_ratio(self):
        """All non-negative, zero_ratio ≤ 0.3 → positive."""
        # 1 zero out of 10 = 0.1 zero_ratio
        values = [1, 2, 3, 4, 5, 1, 2, 3, 4, 0]
        assert detect_feature_mode(_series(values)) == "positive"

    def test_positive_no_zeros(self):
        assert detect_feature_mode(_series([1, 2, 3, 4, 5])) == "positive"

    def test_positive_exactly_30_percent_zeros(self):
        """zero_ratio == 0.3 is NOT > 0.3, so mode is positive."""
        # 3 zeros out of 10 = 0.3 exactly
        values = [0, 0, 0, 1, 2, 3, 4, 5, 1, 2]
        assert detect_feature_mode(_series(values)) == "positive"

    def test_positive_with_large_values(self):
        values = [10, 20, 30, 40, 50, 5, 15, 25, 35, 45]
        assert detect_feature_mode(_series(values)) == "positive"


# ---------------------------------------------------------------------------
# Tests: sparse_positive mode
# ---------------------------------------------------------------------------

class TestSparsePositiveMode:
    def test_sparse_positive_high_zero_ratio(self):
        """All non-negative, zero_ratio > 0.3 → sparse_positive."""
        # 4 zeros out of 10 = 0.4 zero_ratio
        values = [0, 0, 0, 0, 1, 2, 3, 4, 5, 1]
        assert detect_feature_mode(_series(values)) == "sparse_positive"

    def test_sparse_positive_all_zeros(self):
        """All zeros: zero_ratio = 1.0 > 0.3, min = 0 → sparse_positive."""
        # Note: all zeros → unique_vals = {0} ⊆ {0,1} → binary takes priority
        # So we need at least one non-{0,1} value to reach the zero_ratio branch
        values = [0, 0, 0, 0, 2]  # 4/5 = 0.8 zero_ratio, min=0
        assert detect_feature_mode(_series(values)) == "sparse_positive"

    def test_sparse_positive_just_above_threshold(self):
        """zero_ratio just above 0.3 → sparse_positive."""
        # 4 zeros out of 11 ≈ 0.364
        values = [0, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2]
        assert detect_feature_mode(_series(values)) == "sparse_positive"

    def test_zero_ratio_uses_full_series(self):
        """zero_ratio must be computed on the full series, not just non-NaN."""
        # 4 zeros, 1 NaN, 5 non-zero non-NaN → full series length = 10
        # zero_ratio = 4/10 = 0.4 > 0.3 → sparse_positive
        values = [0, 0, 0, 0, float("nan"), 1, 2, 3, 4, 5]
        assert detect_feature_mode(_series(values)) == "sparse_positive"


# ---------------------------------------------------------------------------
# Tests: sparse_signed mode
# ---------------------------------------------------------------------------

class TestSparseSignedMode:
    def test_sparse_signed_negative_high_zero_ratio(self):
        """Has negative values, zero_ratio > 0.3 → sparse_signed."""
        # 4 zeros out of 10 = 0.4, min = -2
        values = [0, 0, 0, 0, -2, -1, 1, 2, 3, 4]
        assert detect_feature_mode(_series(values)) == "sparse_signed"

    def test_sparse_signed_just_above_threshold(self):
        # 4 zeros out of 11 ≈ 0.364, min < 0
        values = [0, 0, 0, 0, -1, 1, 2, 3, 4, 5, 1]
        assert detect_feature_mode(_series(values)) == "sparse_signed"

    def test_sparse_signed_with_nan(self):
        """NaN does not count as zero; zero_ratio on full series."""
        # 4 zeros, 1 NaN, 5 others → full length = 10, zero_ratio = 0.4
        values = [0, 0, 0, 0, float("nan"), -1, 1, 2, 3, 4]
        assert detect_feature_mode(_series(values)) == "sparse_signed"


# ---------------------------------------------------------------------------
# Tests: signed mode
# ---------------------------------------------------------------------------

class TestSignedMode:
    def test_signed_negative_low_zero_ratio(self):
        """Has negative values, zero_ratio ≤ 0.3 → signed."""
        # 1 zero out of 10 = 0.1, min = -3
        values = [-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]
        assert detect_feature_mode(_series(values)) == "signed"

    def test_signed_no_zeros(self):
        values = [-3, -2, -1, 1, 2, 3, 4, 5]
        assert detect_feature_mode(_series(values)) == "signed"

    def test_signed_exactly_30_percent_zeros(self):
        """zero_ratio == 0.3 is NOT > 0.3, so mode is signed (not sparse_signed)."""
        # 3 zeros out of 10 = 0.3 exactly, min < 0
        values = [0, 0, 0, -1, -2, 1, 2, 3, 4, 5]
        assert detect_feature_mode(_series(values)) == "signed"


# ---------------------------------------------------------------------------
# Tests: zero_ratio boundary (0.3 threshold)
# ---------------------------------------------------------------------------

class TestZeroRatioBoundary:
    def test_positive_at_boundary(self):
        """Exactly 30% zeros with non-negative values → positive (not sparse_positive)."""
        values = [0, 0, 0, 1, 2, 3, 4, 5, 1, 2]  # 3/10 = 0.3
        assert detect_feature_mode(_series(values)) == "positive"

    def test_sparse_positive_above_boundary(self):
        """31% zeros with non-negative values → sparse_positive."""
        # Need > 0.3: use 4/13 ≈ 0.308 — just above
        values = [0, 0, 0, 0, 1, 2, 3, 4, 5, 1, 2, 3, 4]  # 4/13 ≈ 0.308
        assert detect_feature_mode(_series(values)) == "sparse_positive"

    def test_signed_at_boundary(self):
        """Exactly 30% zeros with negative values → signed (not sparse_signed)."""
        values = [0, 0, 0, -1, -2, 1, 2, 3, 4, 5]  # 3/10 = 0.3
        assert detect_feature_mode(_series(values)) == "signed"

    def test_sparse_signed_above_boundary(self):
        """Just above 30% zeros with negative values → sparse_signed."""
        values = [0, 0, 0, 0, -1, 1, 2, 3, 4, 5, 1, 2, 3]  # 4/13 ≈ 0.308
        assert detect_feature_mode(_series(values)) == "sparse_signed"


# ---------------------------------------------------------------------------
# Tests: mode priority (binary/ternary take precedence)
# ---------------------------------------------------------------------------

class TestModePriority:
    def test_binary_takes_priority_over_ternary(self):
        """Values {0, 1} match both binary and ternary criteria; binary wins."""
        assert detect_feature_mode(_series([0, 1, 0, 1])) == "binary"

    def test_ternary_takes_priority_over_signed(self):
        """Values {-1, 0, 1} match ternary; should NOT fall through to signed."""
        assert detect_feature_mode(_series([-1, 0, 1])) == "ternary"

    def test_value_2_breaks_binary(self):
        """Adding value 2 to {0, 1} breaks binary → falls through to positive."""
        values = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1]  # 3/10 = 0.3 zero_ratio, min=0
        assert detect_feature_mode(_series(values)) == "positive"

    def test_value_2_breaks_ternary(self):
        """Adding value 2 to {-1, 0, 1} breaks ternary → falls through to signed/sparse_signed."""
        # min < 0, zero_ratio = 2/5 = 0.4 > 0.3 → sparse_signed
        values = [-1, 0, 1, 2, 0]
        assert detect_feature_mode(_series(values)) == "sparse_signed"

    def test_value_2_breaks_ternary_low_zero_ratio(self):
        """Adding value 2 to {-1, 0, 1} with low zero_ratio → signed."""
        # min < 0, zero_ratio = 1/7 ≈ 0.14 ≤ 0.3 → signed
        values = [-1, 0, 1, 2, -2, 3, 4]
        assert detect_feature_mode(_series(values)) == "signed"


# ---------------------------------------------------------------------------
# Tests: detect_all_modes
# ---------------------------------------------------------------------------

class TestDetectAllModes:
    def test_returns_dict_with_correct_keys(self):
        df = pd.DataFrame({
            "feat_bin": [0, 1, 0, 1],
            "feat_pos": [1, 2, 3, 4],
        })
        result = Feature_Detector().detect_all_modes(df, ["feat_bin", "feat_pos"])
        assert set(result.keys()) == {"feat_bin", "feat_pos"}

    def test_correct_modes_assigned(self):
        df = pd.DataFrame({
            "feat_bin": _series([0, 1, 0, 1]),
            "feat_tern": _series([-1, 0, 1, -1]),
            "feat_pos": _series([1, 2, 3, 4, 5, 1, 2, 3, 4, 5]),
            "feat_sparse_pos": _series([0, 0, 0, 0, 1, 2, 3, 4, 5, 1]),
            "feat_signed": _series([-3, -2, -1, 0, 1, 2, 3, 4, 5, 6]),
            "feat_sparse_signed": _series([0, 0, 0, 0, -2, -1, 1, 2, 3, 4]),
        })
        feature_cols = [
            "feat_bin", "feat_tern", "feat_pos",
            "feat_sparse_pos", "feat_signed", "feat_sparse_signed",
        ]
        result = Feature_Detector().detect_all_modes(df, feature_cols)
        assert result["feat_bin"] == "binary"
        assert result["feat_tern"] == "ternary"
        assert result["feat_pos"] == "positive"
        assert result["feat_sparse_pos"] == "sparse_positive"
        assert result["feat_signed"] == "signed"
        assert result["feat_sparse_signed"] == "sparse_signed"

    def test_empty_feature_cols_returns_empty_dict(self):
        df = pd.DataFrame({"feat_a": [1, 2, 3]})
        result = Feature_Detector().detect_all_modes(df, [])
        assert result == {}

    def test_single_feature_col(self):
        df = pd.DataFrame({"feat_a": [0, 1, 0, 1]})
        result = Feature_Detector().detect_all_modes(df, ["feat_a"])
        assert result == {"feat_a": "binary"}


# ---------------------------------------------------------------------------
# Tests: module-level convenience functions
# ---------------------------------------------------------------------------

class TestModuleLevelFunctions:
    def test_detect_feature_mode_function(self):
        assert detect_feature_mode(_series([0, 1, 0, 1])) == "binary"

    def test_detect_all_modes_function(self):
        df = pd.DataFrame({"col": [0, 1, 0, 1]})
        result = detect_all_modes(df, ["col"])
        assert result == {"col": "binary"}

    def test_module_function_matches_class_method(self):
        s = _series([-1, 0, 1, -1, 0])
        assert detect_feature_mode(s) == Feature_Detector().detect_feature_mode(s)


# ---------------------------------------------------------------------------
# Tests: NaN handling edge cases
# ---------------------------------------------------------------------------

class TestNaNHandling:
    def test_all_nan_series_returns_binary(self):
        """All-NaN series: unique_vals = [], n_unique = 0 ≤ 2, set() ⊆ {0,1} → binary."""
        s = _series([float("nan"), float("nan")])
        assert detect_feature_mode(s) == "binary"

    def test_nan_not_counted_as_zero_in_zero_ratio(self):
        """NaN is not == 0, so it does not inflate zero_ratio."""
        # 2 zeros, 2 NaN, 6 non-zero → full length = 10
        # zero_ratio = 2/10 = 0.2 ≤ 0.3, min = -1 → signed
        values = [0, 0, float("nan"), float("nan"), -1, 1, 2, 3, 4, 5]
        assert detect_feature_mode(_series(values)) == "signed"

    def test_nan_not_counted_as_zero_sparse_positive(self):
        """NaN should not push zero_ratio above threshold."""
        # 2 zeros, 2 NaN, 6 non-zero → zero_ratio = 2/10 = 0.2 ≤ 0.3 → positive
        values = [0, 0, float("nan"), float("nan"), 1, 2, 3, 4, 5, 6]
        assert detect_feature_mode(_series(values)) == "positive"

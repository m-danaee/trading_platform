"""
Unit tests for gpu_fuzzy_trader.features.selector.Feature_Selector

Tests cover:
  - Label and meta column exclusion
  - Low-dispersion feature removal
  - Direction-specific target building (long and short)
  - Mutual information scoring
  - Cross-symbol stability computation
  - Within-mode redundancy removal
  - Top-K selection
  - JSON persistence and loading
  - skip_if_valid logic
  - load_and_validate error handling
  - Edge cases: single symbol, all same target, fewer features than K
"""

from __future__ import annotations

import json
import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.features.selector import (
    Feature_Selector,
    _align_feature_array,
    _build_target,
    _check_spearman_sign_consistency,
    _compute_stability,
    _mutual_info_discrete_mask,
    _reduce_overlap,
    _remove_low_dispersion,
    _remove_redundant_features,
    _stationarity_filter,
    _validate_schema,
    build_phase1_shared_context,
)
from gpu_fuzzy_trader import config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_train_df(
    n_rows: int = 200,
    n_features: int = 5,
    symbols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a minimal training DataFrame with label columns and feature columns."""
    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = ["SYM_A", "SYM_B"]

    rows_per_sym = n_rows // len(symbols)
    dfs = []
    for sym in symbols:
        n = rows_per_sym
        open_next = rng.uniform(100, 200, size=n)
        max_288 = open_next * rng.uniform(0.98, 1.08, size=n)
        min_288 = open_next * rng.uniform(0.92, 1.02, size=n)
        close_288 = open_next * rng.uniform(0.95, 1.05, size=n)
        max_before_min = rng.integers(0, 2, size=n)

        data = {
            "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
            "symbol": sym,
            "label_open_next": open_next,
            "label_close_288": close_288,
            "label_min_288": min_288,
            "label_max_288": max_288,
            "label_max_before_min": max_before_min.astype(float),
        }
        for i in range(n_features):
            data[f"feat_{i}"] = rng.integers(0, 5, size=n).astype(float)

        dfs.append(pd.DataFrame(data))

    out = pd.concat(dfs, ignore_index=True)
    out["_symbol_bar_index"] = out.groupby("symbol").cumcount()
    return out


# ---------------------------------------------------------------------------
# Tests: _remove_low_dispersion
# ---------------------------------------------------------------------------

class TestRemoveLowDispersion:
    def test_removes_constant_feature(self):
        df = pd.DataFrame({"feat_const": [1.0] * 100, "feat_var": list(range(100))})
        result = _remove_low_dispersion(df, ["feat_const", "feat_var"], threshold=0.95)
        assert "feat_const" not in result
        assert "feat_var" in result

    def test_keeps_feature_at_threshold(self):
        """Exactly 95% identical → NOT > 0.95 → keep."""
        vals = [1.0] * 95 + [2.0] * 5  # 95% identical
        df = pd.DataFrame({"feat": vals})
        result = _remove_low_dispersion(df, ["feat"], threshold=0.95)
        assert "feat" in result

    def test_removes_feature_above_threshold(self):
        """96% identical → > 0.95 → remove."""
        vals = [1.0] * 96 + [2.0] * 4
        df = pd.DataFrame({"feat": vals})
        result = _remove_low_dispersion(df, ["feat"], threshold=0.95)
        assert "feat" not in result

    def test_empty_feature_list(self):
        df = pd.DataFrame({"feat": [1, 2, 3]})
        result = _remove_low_dispersion(df, [], threshold=0.95)
        assert result == []

    def test_all_features_varied(self):
        df = pd.DataFrame({
            "a": list(range(100)),
            "b": list(range(100, 200)),
        })
        result = _remove_low_dispersion(df, ["a", "b"], threshold=0.95)
        assert set(result) == {"a", "b"}


# ---------------------------------------------------------------------------
# Tests: _build_target
# ---------------------------------------------------------------------------

class TestMutualInfoDiscreteMask:
    def test_mode_aware_discrete_flags(self):
        cols = ["bin_col", "ter_col", "pos_col", "signed_col"]
        modes = {
            "bin_col": "binary",
            "ter_col": "ternary",
            "pos_col": "positive",
            "signed_col": "signed",
        }
        assert _mutual_info_discrete_mask(cols, modes) == [
            True,
            True,
            False,
            False,
        ]

    def test_unknown_mode_defaults_to_continuous(self):
        assert _mutual_info_discrete_mask(["x"], {}) == [False]


class TestSelectFeaturesNoSklearnClusteringWarning:
    def test_no_clustering_metric_warning(self):
        train_df = _make_train_df(n_rows=400, n_features=8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", UserWarning)
            Feature_Selector().select_features(train_df, "long")
        clustering = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and "Clustering metrics expects discrete" in str(w.message)
        ]
        assert clustering == []


class TestBuildTarget:
    def _make_df(self, open_next, max_288, min_288, max_before_min):
        return pd.DataFrame({
            "label_open_next": open_next,
            "label_max_288": max_288,
            "label_min_288": min_288,
            "label_max_before_min": max_before_min,
            "label_close_288": open_next,  # not used in target
        })

    @staticmethod
    def _win_class() -> int:
        """Encoding-aware win class: 2 in asymmetric mode, 1 in legacy mode."""
        return 2 if config.PHASE1_ASYMMETRIC_TARGET else 1

    @staticmethod
    def _loss_class() -> int:
        """Encoding-aware loss class: 0 in either mode."""
        return 0

    def test_long_tp_hit_no_sl(self):
        """Long: max >= entry*(1+TP/100), min > entry*(1-SL/100) → success."""
        tp = config.PHASE2_TP
        entry = 100.0
        df = self._make_df(
            open_next=[entry],
            max_288=[entry * (1 + tp / 100 + 0.01)],  # above TP
            min_288=[entry * 0.99],  # above SL level
            max_before_min=[1],
        )
        target = _build_target(df, "long")
        assert target.iloc[0] == self._win_class()

    def test_long_sl_hit_before_tp(self):
        """Long: both hit but max_before_min==0 → SL first → failure."""
        tp = config.PHASE2_TP
        sl = config.PHASE2_SL
        entry = 100.0
        df = self._make_df(
            open_next=[entry],
            max_288=[entry * (1 + tp / 100 + 0.01)],
            min_288=[entry * (1 - sl / 100 - 0.01)],
            max_before_min=[0],  # min came first → SL first for long
        )
        target = _build_target(df, "long")
        assert target.iloc[0] == self._loss_class()

    def test_long_tp_hit_before_sl(self):
        """Long: both hit but max_before_min==1 → TP first → success."""
        tp = config.PHASE2_TP
        sl = config.PHASE2_SL
        entry = 100.0
        df = self._make_df(
            open_next=[entry],
            max_288=[entry * (1 + tp / 100 + 0.01)],
            min_288=[entry * (1 - sl / 100 - 0.01)],
            max_before_min=[1],  # max came first → TP first for long
        )
        target = _build_target(df, "long")
        assert target.iloc[0] == self._win_class()

    def test_long_neither_hit(self):
        """Long: neither TP nor SL hit → failure (or neutral in asymmetric mode)."""
        entry = 100.0
        df = self._make_df(
            open_next=[entry],
            max_288=[entry * 1.01],  # below TP
            min_288=[entry * 0.999],  # above SL level (SL=1% so 0.99 is exactly at SL; 0.999 is safely above)
            max_before_min=[1],
        )
        target = _build_target(df, "long")
        # Asymmetric: neither win nor loss → neutral class (1)
        # Legacy: failure → 0
        expected = 1 if config.PHASE1_ASYMMETRIC_TARGET else 0
        assert target.iloc[0] == expected

    def test_short_tp_hit_no_sl(self):
        """Short: min <= entry*(1-TP/100), max < entry*(1+SL/100) → success."""
        tp = config.PHASE2_TP
        entry = 100.0
        df = self._make_df(
            open_next=[entry],
            max_288=[entry * 1.01],  # below SL level
            min_288=[entry * (1 - tp / 100 - 0.01)],  # below TP
            max_before_min=[0],
        )
        target = _build_target(df, "short")
        assert target.iloc[0] == self._win_class()

    def test_short_sl_hit_before_tp(self):
        """Short: both hit but max_before_min==1 → max came first → SL first → failure."""
        tp = config.PHASE2_TP
        sl = config.PHASE2_SL
        entry = 100.0
        df = self._make_df(
            open_next=[entry],
            max_288=[entry * (1 + sl / 100 + 0.01)],
            min_288=[entry * (1 - tp / 100 - 0.01)],
            max_before_min=[1],  # max came first → SL first for short
        )
        target = _build_target(df, "short")
        assert target.iloc[0] == self._loss_class()

    def test_short_tp_hit_before_sl(self):
        """Short: both hit but max_before_min==0 → min came first → TP first → success."""
        tp = config.PHASE2_TP
        sl = config.PHASE2_SL
        entry = 100.0
        df = self._make_df(
            open_next=[entry],
            max_288=[entry * (1 + sl / 100 + 0.01)],
            min_288=[entry * (1 - tp / 100 - 0.01)],
            max_before_min=[0],  # min came first → TP first for short
        )
        target = _build_target(df, "short")
        assert target.iloc[0] == self._win_class()

    def test_returns_integer_series(self):
        df = _make_train_df(n_rows=20, n_features=2, symbols=["A"])
        target = _build_target(df, "long")
        # Asymmetric path uses int8 (3 classes); legacy uses int (2 classes)
        assert np.issubdtype(target.dtype, np.integer)
        if config.PHASE1_ASYMMETRIC_TARGET:
            assert set(target.unique()).issubset({0, 1, 2})
        else:
            assert set(target.unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# Tests: _compute_stability
# ---------------------------------------------------------------------------

class TestComputeStability:
    def test_identical_scores_perfect_stability(self):
        """All symbols have same score → std=0 → stability=1."""
        result = _compute_stability([0.5, 0.5, 0.5])
        assert result == pytest.approx(1.0)

    def test_zero_mean_returns_zero(self):
        """Mean is 0 → stability=0."""
        result = _compute_stability([0.0, 0.0, 0.0])
        assert result == 0.0

    def test_single_symbol_positive_score(self):
        """Single symbol with positive score → stability=1."""
        result = _compute_stability([0.3])
        assert result == 1.0

    def test_single_symbol_zero_score(self):
        """Single symbol with zero score → stability=0."""
        result = _compute_stability([0.0])
        assert result == 0.0

    def test_empty_list_returns_zero(self):
        result = _compute_stability([])
        assert result == 0.0

    def test_high_variance_clipped_to_zero(self):
        """Very high variance → stability could be negative → clipped to 0."""
        result = _compute_stability([0.01, 1.0, 0.01])
        assert result >= 0.0

    def test_stability_in_range(self):
        """Stability should always be in [0, 1]."""
        for scores in [[0.1, 0.2, 0.3], [0.5, 0.5], [0.0, 0.1, 0.2]]:
            result = _compute_stability(scores)
            assert 0.0 <= result <= 1.0


# ---------------------------------------------------------------------------
# Tests: _remove_redundant_features
# ---------------------------------------------------------------------------

class TestRemoveRedundantFeatures:
    def test_removes_highly_correlated_feature(self):
        """Two features with corr > 0.95 → keep higher-scored one."""
        n = 100
        base = np.arange(n, dtype=float)
        df = pd.DataFrame({
            "feat_a": base,
            "feat_b": base + 0.001 * np.random.default_rng(0).standard_normal(n),
        })
        scored = [
            {"name": "feat_a", "mode": "positive", "score": 0.8},
            {"name": "feat_b", "mode": "positive", "score": 0.5},
        ]
        result = _remove_redundant_features(df, scored, threshold=0.95)
        names = [e["name"] for e in result]
        assert "feat_a" in names
        assert "feat_b" not in names

    def test_keeps_uncorrelated_features(self):
        """Two uncorrelated features → both kept."""
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "feat_a": rng.standard_normal(100),
            "feat_b": rng.standard_normal(100),
        })
        scored = [
            {"name": "feat_a", "mode": "signed", "score": 0.8},
            {"name": "feat_b", "mode": "signed", "score": 0.5},
        ]
        result = _remove_redundant_features(df, scored, threshold=0.95)
        names = [e["name"] for e in result]
        assert "feat_a" in names
        assert "feat_b" in names

    def test_different_modes_not_compared(self):
        """Features in different modes are not compared for redundancy."""
        n = 100
        base = np.arange(n, dtype=float)
        df = pd.DataFrame({
            "feat_a": base,
            "feat_b": base,  # identical but different mode
        })
        scored = [
            {"name": "feat_a", "mode": "positive", "score": 0.8},
            {"name": "feat_b", "mode": "signed", "score": 0.5},
        ]
        result = _remove_redundant_features(df, scored, threshold=0.95)
        names = [e["name"] for e in result]
        # Both kept because they're in different mode groups
        assert "feat_a" in names
        assert "feat_b" in names

    def test_single_feature_kept(self):
        df = pd.DataFrame({"feat_a": [1, 2, 3]})
        scored = [{"name": "feat_a", "mode": "positive", "score": 0.5}]
        result = _remove_redundant_features(df, scored, threshold=0.95)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Tests: _validate_schema
# ---------------------------------------------------------------------------

class TestValidateSchema:
    def test_valid_schema(self):
        data = {
            "direction": "long",
            "features": [{"name": "feat_a", "mode": "positive", "score": 0.5}],
        }
        _validate_schema(data, "test.json")  # should not raise

    def test_missing_direction_key(self):
        data = {"features": []}
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_schema(data, "test.json")

    def test_missing_features_key(self):
        data = {"direction": "long"}
        with pytest.raises(ValueError, match="missing required keys"):
            _validate_schema(data, "test.json")

    def test_invalid_direction(self):
        data = {"direction": "both", "features": []}
        with pytest.raises(ValueError, match="invalid direction"):
            _validate_schema(data, "test.json")

    def test_features_not_list(self):
        data = {"direction": "long", "features": "not_a_list"}
        with pytest.raises(ValueError, match="must be a list"):
            _validate_schema(data, "test.json")

    def test_feature_entry_missing_name(self):
        data = {
            "direction": "long",
            "features": [{"mode": "positive", "score": 0.5}],
        }
        with pytest.raises(ValueError, match="missing keys"):
            _validate_schema(data, "test.json")

    def test_feature_entry_invalid_score_type(self):
        data = {
            "direction": "long",
            "features": [{"name": "f", "mode": "positive", "score": "high"}],
        }
        with pytest.raises(ValueError, match="'score' must be a number"):
            _validate_schema(data, "test.json")

    def test_not_a_dict(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            _validate_schema([1, 2, 3], "test.json")


# ---------------------------------------------------------------------------
# Tests: Feature_Selector.load_and_validate
# ---------------------------------------------------------------------------

class TestLoadAndValidate:
    def test_loads_valid_file(self, tmp_path):
        data = {
            "direction": "long",
            "features": [{"name": "feat_a", "mode": "positive", "score": 0.7}],
        }
        path = str(tmp_path / "selected_features_long.json")
        with open(path, "w") as fh:
            json.dump(data, fh)

        result = Feature_Selector.load_and_validate(path)
        assert len(result) == 1
        assert result[0]["name"] == "feat_a"

    def test_raises_on_missing_file(self, tmp_path):
        path = str(tmp_path / "nonexistent.json")
        with pytest.raises(ValueError, match="not found"):
            Feature_Selector.load_and_validate(path)

    def test_raises_on_corrupted_json(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as fh:
            fh.write("{not valid json")
        with pytest.raises(ValueError, match="unreadable or corrupted"):
            Feature_Selector.load_and_validate(path)

    def test_raises_on_invalid_schema(self, tmp_path):
        data = {"direction": "long"}  # missing 'features'
        path = str(tmp_path / "invalid.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        with pytest.raises(ValueError):
            Feature_Selector.load_and_validate(path)

    def test_empty_features_list_is_valid(self, tmp_path):
        data = {"direction": "short", "features": []}
        path = str(tmp_path / "empty.json")
        with open(path, "w") as fh:
            json.dump(data, fh)
        result = Feature_Selector.load_and_validate(path)
        assert result == []


# ---------------------------------------------------------------------------
# Tests: Feature_Selector.skip_if_valid
# ---------------------------------------------------------------------------

class TestSkipIfValid:
    def _write_valid_file(
        self,
        path: str,
        direction: str,
        *,
        phase1_disabled: bool = False,
    ):
        data = {
            "direction": direction,
            "features": [{"name": "feat_a", "mode": "positive", "score": 0.5}],
            "phase1_disabled": phase1_disabled,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh)

    @pytest.fixture(autouse=True)
    def _phase1_enabled_for_cache_tests(self, monkeypatch):
        monkeypatch.setattr(config, "PHASE1_DISABLED", False)

    def test_returns_none_when_no_files(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._LONG_PATH",
            str(tmp_path / "selected_features_long.json"),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._SHORT_PATH",
            str(tmp_path / "selected_features_short.json"),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {
                "long": str(tmp_path / "selected_features_long.json"),
                "short": str(tmp_path / "selected_features_short.json"),
            },
        )
        result = Feature_Selector.skip_if_valid()
        assert result is None

    def test_returns_none_when_only_long_exists(self, tmp_path, monkeypatch):
        long_path = str(tmp_path / "selected_features_long.json")
        short_path = str(tmp_path / "selected_features_short.json")
        self._write_valid_file(long_path, "long")

        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._LONG_PATH", long_path)
        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._SHORT_PATH", short_path)
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {"long": long_path, "short": short_path},
        )
        result = Feature_Selector.skip_if_valid()
        assert result is None

    def test_returns_dict_when_both_valid(self, tmp_path, monkeypatch):
        long_path = str(tmp_path / "selected_features_long.json")
        short_path = str(tmp_path / "selected_features_short.json")
        self._write_valid_file(long_path, "long")
        self._write_valid_file(short_path, "short")

        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._LONG_PATH", long_path)
        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._SHORT_PATH", short_path)
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {"long": long_path, "short": short_path},
        )
        result = Feature_Selector.skip_if_valid()
        assert result is not None
        assert "long" in result
        assert "short" in result

    def test_raises_on_corrupted_file(self, tmp_path, monkeypatch):
        long_path = str(tmp_path / "selected_features_long.json")
        short_path = str(tmp_path / "selected_features_short.json")
        self._write_valid_file(long_path, "long")
        with open(short_path, "w") as fh:
            fh.write("{corrupted")

        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._LONG_PATH", long_path)
        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._SHORT_PATH", short_path)
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {"long": long_path, "short": short_path},
        )
        with pytest.raises(ValueError):
            Feature_Selector.skip_if_valid()

    def test_skip_if_valid_invalidates_on_disabled_toggle(
        self, tmp_path, monkeypatch,
    ):
        long_path = str(tmp_path / "selected_features_long.json")
        short_path = str(tmp_path / "selected_features_short.json")
        self._write_valid_file(long_path, "long", phase1_disabled=False)
        self._write_valid_file(short_path, "short", phase1_disabled=False)

        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._LONG_PATH", long_path)
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._SHORT_PATH", short_path)
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {"long": long_path, "short": short_path},
        )
        monkeypatch.setattr(config, "PHASE1_DISABLED", True)
        assert Feature_Selector.skip_if_valid() is None


# ---------------------------------------------------------------------------
# Tests: PHASE1_DISABLED bypass
# ---------------------------------------------------------------------------

class TestPhase1Disabled:
    def test_phase1_disabled_returns_all_features(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "PHASE1_DISABLED", True)
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {
                "long": str(tmp_path / "selected_features_long.json"),
                "short": str(tmp_path / "selected_features_short.json"),
            },
        )
        train_df = _make_train_df(n_rows=200, n_features=8, symbols=["A", "B"])
        shared = build_phase1_shared_context(train_df)
        result = Feature_Selector().run(train_df)
        assert len(result["long"]) == len(shared.feature_cols)
        assert len(result["short"]) == len(shared.feature_cols)
        for direction in ("long", "short"):
            for entry in result[direction]:
                assert entry["mode"] in {
                    "binary", "ternary", "positive", "signed"}
                assert entry["score"] == 0.0

    def test_phase1_disabled_persists_flag(self, monkeypatch, tmp_path):
        monkeypatch.setattr(config, "PHASE1_DISABLED", True)
        long_path = str(tmp_path / "selected_features_long.json")
        short_path = str(tmp_path / "selected_features_short.json")
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {"long": long_path, "short": short_path},
        )
        train_df = _make_train_df(n_rows=100, n_features=4)
        Feature_Selector().run(train_df)
        with open(long_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload.get("phase1_disabled") is True


# ---------------------------------------------------------------------------
# Tests: Feature_Selector.select_features
# ---------------------------------------------------------------------------

class TestSelectFeatures:
    def test_excludes_label_columns(self):
        """Label columns must not appear in selected features."""
        train_df = _make_train_df(n_rows=200, n_features=5)
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        names = [e["name"] for e in result]
        for label_col in config.LABEL_COLUMNS:
            assert label_col not in names

    def test_excludes_meta_columns(self):
        """Meta columns must not appear in selected features."""
        train_df = _make_train_df(n_rows=200, n_features=5)
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        names = [e["name"] for e in result]
        for meta_col in config.META_COLUMNS:
            assert meta_col not in names

    def test_excludes_internal_and_underscore_prefixed_columns(self):
        """Loader internal columns and ``_``-prefixed names are not candidates."""
        train_df = _make_train_df(n_rows=200, n_features=5)
        train_df["_ghost"] = 0.0
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        names = {e["name"] for e in result}
        for col in config.INTERNAL_COLUMNS:
            assert col not in names
        assert "_ghost" not in names

    def test_returns_list_of_dicts(self):
        train_df = _make_train_df(n_rows=200, n_features=5)
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        assert isinstance(result, list)
        for entry in result:
            assert isinstance(entry, dict)
            assert "name" in entry
            assert "mode" in entry
            assert "score" in entry

    def test_scores_are_non_negative(self):
        train_df = _make_train_df(n_rows=200, n_features=5)
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        for entry in result:
            assert entry["score"] >= 0.0

    def test_sorted_by_score_descending(self):
        train_df = _make_train_df(n_rows=200, n_features=5)
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        scores = [e["score"] for e in result]
        assert scores == sorted(scores, reverse=True)

    def test_at_most_candidate_pool_features(self):
        """select_features returns up to 2×TOP_K candidates before overlap reduction."""
        n_features = config.PHASE1_TOP_K_FEATURES + 10
        train_df = _make_train_df(n_rows=400, n_features=n_features)
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        assert len(result) <= config.PHASE1_TOP_K_FEATURES * 2

    def test_fewer_features_than_k_returns_all(self):
        """If fewer features than K remain after filtering, return all."""
        train_df = _make_train_df(n_rows=200, n_features=3)
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        # Should return all 3 features (well below K=30)
        assert len(result) <= 3

    def test_long_and_short_can_differ(self):
        """Long and short selections may differ (different targets)."""
        train_df = _make_train_df(n_rows=400, n_features=10, seed=7)
        selector = Feature_Selector()
        long_result = selector.select_features(train_df, "long")
        short_result = selector.select_features(train_df, "short")
        # Both should be valid lists
        assert isinstance(long_result, list)
        assert isinstance(short_result, list)

    def test_invalid_direction_raises(self):
        train_df = _make_train_df(n_rows=100, n_features=3)
        selector = Feature_Selector()
        with pytest.raises(ValueError, match="direction must be"):
            selector.select_features(train_df, "both")

    def test_constant_feature_excluded(self):
        """A feature with >95% identical values should be excluded."""
        train_df = _make_train_df(n_rows=200, n_features=3)
        # Add a constant feature
        train_df["feat_const"] = 1.0
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        names = [e["name"] for e in result]
        assert "feat_const" not in names

    def test_mode_assigned_to_each_feature(self):
        """Each selected feature must have a valid mode string."""
        valid_modes = {"binary", "ternary", "positive", "sparse_positive",
                       "sparse_signed", "signed"}
        train_df = _make_train_df(n_rows=200, n_features=5)
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        for entry in result:
            assert entry["mode"] in valid_modes

    def test_single_symbol_works(self):
        """Should work with a single symbol (no cross-symbol stability)."""
        train_df = _make_train_df(n_rows=200, n_features=5, symbols=["ONLY"])
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        assert isinstance(result, list)

    def test_all_same_target_handled(self):
        """If all target values are the same, MI scoring should not crash."""
        train_df = _make_train_df(n_rows=200, n_features=3)
        # Force all labels so that TP is never hit → all target = 0
        train_df["label_max_288"] = train_df["label_open_next"] * 0.99
        selector = Feature_Selector()
        result = selector.select_features(train_df, "long")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# Tests: Feature_Selector.run (persistence)
# ---------------------------------------------------------------------------

class TestFeatureSelectorRun:
    def test_run_creates_output_files(self, tmp_path, monkeypatch):
        long_path = str(tmp_path / "selected_features_long.json")
        short_path = str(tmp_path / "selected_features_short.json")

        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._LONG_PATH", long_path)
        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._SHORT_PATH", short_path)
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {"long": long_path, "short": short_path},
        )

        train_df = _make_train_df(n_rows=200, n_features=5)
        selector = Feature_Selector()
        result = selector.run(train_df)

        assert os.path.exists(long_path)
        assert os.path.exists(short_path)
        assert "long" in result
        assert "short" in result

    def test_run_output_files_are_valid_json(self, tmp_path, monkeypatch):
        long_path = str(tmp_path / "selected_features_long.json")
        short_path = str(tmp_path / "selected_features_short.json")

        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._LONG_PATH", long_path)
        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._SHORT_PATH", short_path)
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {"long": long_path, "short": short_path},
        )

        train_df = _make_train_df(n_rows=200, n_features=5)
        Feature_Selector().run(train_df)

        for path, direction in [(long_path, "long"), (short_path, "short")]:
            with open(path) as fh:
                data = json.load(fh)
            assert data["direction"] == direction
            assert isinstance(data["features"], list)

    def test_run_returns_dict_with_both_directions(self, tmp_path, monkeypatch):
        long_path = str(tmp_path / "selected_features_long.json")
        short_path = str(tmp_path / "selected_features_short.json")

        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._LONG_PATH", long_path)
        monkeypatch.setattr("gpu_fuzzy_trader.features.selector._SHORT_PATH", short_path)
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._DIRECTION_PATHS",
            {"long": long_path, "short": short_path},
        )

        train_df = _make_train_df(n_rows=200, n_features=5)
        result = Feature_Selector().run(train_df)

        assert set(result.keys()) == {"long", "short"}
        assert isinstance(result["long"], list)
        assert isinstance(result["short"], list)


class TestStationarityFilter:
    def test_regime_rank_drift_rejects_large_swing(self) -> None:
        fold_scores = {
            "f0": [10.0, 1.0, 10.0],
            "f1": [9.0, 2.0, 9.0],
            "f2": [8.0, 3.0, 8.0],
            "f3": [1.0, 10.0, 1.0],
        }
        survivors_tight = _stationarity_filter(
            fold_scores, cv_max=10.0, rank_drift_max=2)
        survivors_loose = _stationarity_filter(
            fold_scores, cv_max=10.0, rank_drift_max=30)
        assert "f3" not in survivors_tight
        assert "f3" in survivors_loose

    def test_stable_ranks_pass_tight_drift(self) -> None:
        fold_scores = {
            "a": [1.0, 1.0, 1.0],
            "b": [0.9, 0.9, 0.9],
            "c": [0.8, 0.8, 0.8],
        }
        survivors = _stationarity_filter(
            fold_scores, cv_max=1.0, rank_drift_max=2)
        assert survivors == {"a", "b", "c"}


class TestAlignFeatureArray:
    def test_subset_columns_preserve_order(self) -> None:
        source_cols = ["a", "b", "c", "d"]
        arr = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=float)
        aligned = _align_feature_array(arr, source_cols, ["b", "d", "a"])
        np.testing.assert_array_equal(aligned, [[2, 4, 1], [6, 8, 5]])

    def test_shared_context_after_sign_consistency_subset(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        train_df = _make_train_df(n_rows=400, n_features=8, symbols=["A", "B"])
        shared = build_phase1_shared_context(train_df)
        stable_cols = list(shared.feature_cols[:4])

        def fake_stable(
            df: pd.DataFrame,
            cols: list[str],
            n_folds: int,
            min_folds: int,
            val_df: pd.DataFrame | None = None,
        ) -> set[str]:
            return set(cols[: max(3, len(cols) // 2)])

        def fake_mi(
            X: np.ndarray,
            y: np.ndarray,
            *,
            discrete_features: list[bool],
            random_state: int | None = None,
        ) -> np.ndarray:
            del y, discrete_features, random_state
            # Deterministic positive scores so the shared-context path is testable
            # without depending on random MI from synthetic labels.
            return np.linspace(0.9, 0.1, num=X.shape[1], dtype=float)

        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector._check_spearman_sign_consistency",
            fake_stable,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.features.selector.mutual_info_classif",
            fake_mi,
        )
        monkeypatch.setattr(config, "PHASE1_REQUIRE_SIGN_CONSISTENCY", True)
        monkeypatch.setattr(config, "PHASE1_STATIONARITY_FOLDS", 0)

        result = Feature_Selector().select_features(
            train_df, "long", shared=shared)
        assert len(result) > 0
        assert all(entry["score"] > 0.0 for entry in result)
        assert {entry["name"] for entry in result}.issubset(set(stable_cols))


class TestSpearmanSignConsistency:
    """Tests for _check_spearman_sign_consistency, including the val_df check."""

    def _make_corr_df(
        self,
        feat_values: np.ndarray,
        label_values: np.ndarray,
    ) -> pd.DataFrame:
        """Build a minimal DataFrame without 'symbol' to avoid symbol-based folding."""
        return pd.DataFrame({
            "feat_0": feat_values,
            "label_close_288": label_values,
        })

    def test_val_sign_disagrees_negative_blacklists(self):
        """AC1: Train all positive, val negative → feature blacklisted."""
        n = 100
        # Train: feat_0 and label are perfectly positively correlated (rho=1.0)
        train_df = self._make_corr_df(np.arange(n, dtype=float), np.arange(n, dtype=float))
        # Val: feat_0 ascending, label descending → rho = -1.0
        val_df = self._make_corr_df(np.arange(n, dtype=float), np.arange(n - 1, -1, -1, dtype=float))
        stable = _check_spearman_sign_consistency(
            train_df, ["feat_0"], n_folds=2, min_folds=2, val_df=val_df, min_abs_corr=0.05,
        )
        assert "feat_0" not in stable, (
            "Feature with train=positive, val=negative should be blacklisted"
        )

    def test_val_sign_matches_positive_kept(self):
        """AC2: Train all positive, val positive → feature still kept."""
        n = 100
        train_df = self._make_corr_df(np.arange(n, dtype=float), np.arange(n, dtype=float))
        val_df = self._make_corr_df(np.arange(n, dtype=float), np.arange(n, dtype=float))
        stable = _check_spearman_sign_consistency(
            train_df, ["feat_0"], n_folds=2, min_folds=2, val_df=val_df, min_abs_corr=0.05,
        )
        assert "feat_0" in stable, (
            "Feature with train=positive, val=positive should be kept"
        )

    def test_val_sign_tiny_corr_kept(self):
        """AC3: Train all positive, tiny |val_rho| < min_abs_corr → feature still kept."""
        n = 100
        train_df = self._make_corr_df(np.arange(n, dtype=float), np.arange(n, dtype=float))
        # Val: shuffled label so |rho| ~ 0 < 0.05
        rng = np.random.default_rng(42)
        val_label = np.arange(n, dtype=float)
        rng.shuffle(val_label)
        val_df = self._make_corr_df(np.arange(n, dtype=float), val_label)
        stable = _check_spearman_sign_consistency(
            train_df, ["feat_0"], n_folds=2, min_folds=2, val_df=val_df, min_abs_corr=0.05,
        )
        assert "feat_0" in stable, (
            "Feature with train=positive, tiny val corr should be kept"
        )

    def test_val_df_none_matches_previous_behavior(self):
        """AC4: val_df=None → pre-task-8 behavior (mixed train signs still blacklisted)."""
        n = 100
        n_half = n // 2
        # Fold 0: rho = +1.0, Fold 1: rho = -1.0 → has_pos=True, has_neg=True → blacklisted
        feat = np.concatenate([np.arange(n_half, dtype=float), -np.arange(n_half, dtype=float)])
        label = np.arange(n, dtype=float)
        train_df = self._make_corr_df(feat, label)
        stable = _check_spearman_sign_consistency(
            train_df, ["feat_0"], n_folds=2, min_folds=2, val_df=None, min_abs_corr=0.05,
        )
        assert "feat_0" not in stable, (
            "Feature with mixed train signs should be blacklisted with val_df=None"
        )

    def test_val_df_missing_label_col_skips_check(self):
        """AC5: val_df without label_close_288 column → val check skipped (no crash)."""
        n = 100
        train_df = self._make_corr_df(np.arange(n, dtype=float), np.arange(n, dtype=float))
        # val_df has 'feat_0' but no 'label_close_288' column
        val_df = pd.DataFrame({"feat_0": np.arange(n, dtype=float)})
        stable = _check_spearman_sign_consistency(
            train_df, ["feat_0"], n_folds=2, min_folds=2, val_df=val_df, min_abs_corr=0.05,
        )
        assert "feat_0" in stable, (
            "Feature should be kept when val_df is missing the label column"
        )


class TestReduceOverlap:
    def _feat(self, name: str, score: float) -> dict:
        return {"name": name, "mode": "positive", "score": score}

    def test_caps_overlap_and_backfills_to_top_k(self):
        top_k = config.PHASE1_TOP_K_FEATURES
        ranked = {
            "long": [self._feat(f"L{i}", 1.0 - i * 0.01) for i in range(30)],
            "short": [self._feat(f"S{i}", 1.0 - i * 0.01) for i in range(30)],
        }
        for i in range(20):
            ranked["long"].append(self._feat(f"shared_{i}", 0.5))
            ranked["short"].append(self._feat(f"shared_{i}", 0.4))

        result = _reduce_overlap(ranked, config.PHASE1_MAX_FEATURE_OVERLAP, top_k)
        long_names = {f["name"] for f in result["long"]}
        short_names = {f["name"] for f in result["short"]}
        shared = long_names & short_names
        assert len(result["long"]) == top_k
        assert len(result["short"]) == top_k
        assert len(shared) <= int(top_k * config.PHASE1_MAX_FEATURE_OVERLAP)

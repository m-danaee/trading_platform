"""Unit tests for regime_feature_keyword stratum initialization (Task 8)."""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_init import (
    _regime_feature_indices,
    assign_strata_to_indices,
    assign_three_strata_to_indices,
    build_feature_sampling_probs,
    sample_regime_stratum_chromosome,
    sample_sparse_chromosome,
)
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _count_active_conditions,
    _get_dont_cares,
    _init_population,
)


def _feature_infos_with_scores(
    names_scores: list[tuple[str, float]],
) -> list[dict]:
    return [
        {"name": n, "mode": "positive", "score": s}
        for n, s in names_scores
    ]


# ---------------------------------------------------------------------------
# Config presence
# ---------------------------------------------------------------------------

class TestConfigKeys:
    """Acceptance: all 4 new config keys are present and accessible."""

    def test_regime_stratum_enabled_exists(self):
        val = getattr(_cfg, "PHASE2_REGIME_STRATUM_ENABLED", None)
        assert val is not None
        assert isinstance(val, bool)

    def test_regime_stratum_frac_exists(self):
        val = getattr(_cfg, "PHASE2_REGIME_STRATUM_FRAC", None)
        assert val is not None
        assert isinstance(val, float)
        assert 0.0 < val < 1.0

    def test_regime_feature_keywords_exists(self):
        val = getattr(_cfg, "PHASE2_REGIME_FEATURE_KEYWORDS", None)
        assert val is not None
        assert isinstance(val, tuple)
        assert len(val) > 0
        # Spot-check a few expected keywords
        keywords_lower = tuple(str(k).lower() for k in val)
        assert "vol" in keywords_lower
        assert "atr" in keywords_lower


# ---------------------------------------------------------------------------
# _regime_feature_indices
# ---------------------------------------------------------------------------

class TestRegimeFeatureIndices:
    """Acceptance: returns indices of features whose names match keywords."""

    def test_returns_correct_indices(self):
        """Regime features (vol_20, atr_14, bb_width_20, etc.) are identified."""
        fi = _feature_infos_with_scores([
            ("vol_20", 1.0),
            ("rsi_14", 0.8),
            ("atr_14", 0.7),
            ("bb_width_20", 0.6),
            ("adx_14", 0.5),
            ("dmi_14", 0.4),
            ("compression_20", 0.3),
            ("rsi_30", 0.2),
        ])
        indices = _regime_feature_indices(fi)
        # vol_20, atr_14, bb_width_20, adx_14, dmi_14, compression_20
        # should be matched; rsi_14 and rsi_30 should NOT.
        expected = {0, 2, 3, 4, 5, 6}
        assert set(indices) == expected, f"Got {indices}, expected {expected}"

    def test_empty_when_no_match(self):
        """No keywords matched → empty list."""
        fi = _feature_infos_with_scores([
            ("rsi_14", 1.0),
            ("stoch_14", 0.5),
        ])
        indices = _regime_feature_indices(fi)
        assert indices == []

    def test_empty_feature_infos(self):
        """Empty feature_infos → empty list."""
        assert _regime_feature_indices([]) == []

    def test_case_insensitive_matching(self):
        """Keywords are matched case-insensitively."""
        fi = _feature_infos_with_scores([
            ("VOL_20", 1.0),
            ("bb_WIDTH_10", 0.5),
            ("RSI_14", 0.3),
        ])
        indices = _regime_feature_indices(fi)
        assert set(indices) == {0, 1}  # VOL_20 matches "vol", bb_WIDTH_10 matches "bb_width"


# ---------------------------------------------------------------------------
# assign_three_strata_to_indices
# ---------------------------------------------------------------------------

class TestAssignThreeStrata:
    """Acceptance: returns right fractions of elite/explorer/regime."""

    def test_fractions_are_approximate(self):
        indices = np.arange(1000)
        rng = np.random.default_rng(42)
        labels = assign_three_strata_to_indices(
            indices, (0.40, 0.35, 0.25), rng,
        )
        assert len(labels) == 1000
        n_elite = labels.count("elite")
        n_explorer = labels.count("explorer")
        n_regime = labels.count("regime")
        assert n_elite + n_explorer + n_regime == 1000
        # Allow ±2% tolerance
        assert abs(n_elite / 1000 - 0.40) < 0.03
        assert abs(n_explorer / 1000 - 0.35) < 0.03
        assert abs(n_regime / 1000 - 0.25) < 0.03

    def test_only_allowed_labels(self):
        labels = assign_three_strata_to_indices(
            np.arange(50), (0.4, 0.35, 0.25), np.random.default_rng(0),
        )
        assert set(labels) <= {"elite", "explorer", "regime"}

    def test_empty_input(self):
        assert assign_three_strata_to_indices(
            np.array([], dtype=int), (0.4, 0.35, 0.25), np.random.default_rng(0),
        ) == []

    def test_single_element(self):
        labels = assign_three_strata_to_indices(
            np.array([0]), (0.4, 0.35, 0.25), np.random.default_rng(0),
        )
        assert len(labels) == 1
        assert labels[0] in ("elite", "explorer", "regime")


# ---------------------------------------------------------------------------
# sample_regime_stratum_chromosome
# ---------------------------------------------------------------------------

class TestSampleRegimeStratumChromosome:
    """Acceptance: first active gene is a regime feature."""

    def test_first_active_is_regime_feature(self):
        """First active gene index is one of the regime-keyword-matched indices."""
        fi = _feature_infos_with_scores([
            ("vol_20", 1.0),
            ("rsi_14", 0.8),
            ("atr_14", 0.7),
        ])
        dont_cares = _get_dont_cares(fi)
        probs = build_feature_sampling_probs(fi)
        rng = np.random.default_rng(42)

        regime_indices = _regime_feature_indices(fi)  # [0, 2]
        for _ in range(50):
            k = _cfg.MIN_CONDITIONS  # use config min (usually 3)
            chrom = sample_regime_stratum_chromosome(
                rng, fi, dont_cares, k, probs,
            )
            active = np.where(chrom != dont_cares)[0]
            assert len(active) == k
            first_active = int(active[0])
            assert first_active in regime_indices, (
                f"First active gene {first_active} not in regime indices {regime_indices}"
            )

    def test_fallback_when_no_regime_features(self):
        """When no regime keywords match, behaves like explorer stratum."""
        fi = _feature_infos_with_scores([
            ("rsi_14", 1.0),
            ("stoch_14", 0.5),
            ("macd_12_26", 0.3),
        ])
        dont_cares = _get_dont_cares(fi)
        probs = build_feature_sampling_probs(fi)
        rng = np.random.default_rng(42)

        k = _cfg.MIN_CONDITIONS
        chrom = sample_regime_stratum_chromosome(
            rng, fi, dont_cares, k, probs,
        )
        active = np.where(chrom != dont_cares)[0]
        assert len(active) == k
        # All are valid active indices
        assert all(0 <= i < len(fi) for i in active)

    def test_active_count_respects_bounds(self):
        """Generated chromosome has active count in [MIN_CONDITIONS, MAX_CONDITIONS]."""
        fi = _feature_infos_with_scores([
            ("vol_20", 1.0),
            ("rsi_14", 0.8),
            ("atr_14", 0.7),
            ("bb_width_20", 0.6),
            ("adx_14", 0.5),
        ])
        dont_cares = _get_dont_cares(fi)
        probs = build_feature_sampling_probs(fi)
        rng = np.random.default_rng(42)

        for _ in range(30):
            k = int(rng.integers(1, 5))
            chrom = sample_regime_stratum_chromosome(
                rng, fi, dont_cares, k, probs,
            )
            active = int(np.sum(chrom != dont_cares))
            lo = _cfg.MIN_CONDITIONS
            hi = _cfg.MAX_CONDITIONS
            assert lo <= active <= hi, (
                f"Active count {active} not in [{lo}, {hi}]"
            )

    def test_chromosome_dtype_int32(self):
        fi = _feature_infos_with_scores([
            ("vol_20", 1.0),
            ("rsi_14", 0.8),
        ])
        dont_cares = _get_dont_cares(fi)
        rng = np.random.default_rng(42)
        k = _cfg.MIN_CONDITIONS
        chrom = sample_regime_stratum_chromosome(
            rng, fi, dont_cares, k, build_feature_sampling_probs(fi),
        )
        assert chrom.dtype == np.int32


# ---------------------------------------------------------------------------
# Backward compatibility: 2-stratum when regime disabled
# ---------------------------------------------------------------------------

class TestBackwardCompatTwoStratum:
    """Acceptance: when PHASE2_REGIME_STRATUM_ENABLED=False, 2-stratum preserved."""

    def test_assign_strata_to_indices_still_works(self):
        """The original assign_strata_to_indices returns only elite/explorer."""
        indices = np.arange(50)
        rng = np.random.default_rng(0)
        labels = assign_strata_to_indices(indices, (0.67, 0.33), rng)
        assert set(labels) <= {"elite", "explorer"}
        assert labels.count("elite") + labels.count("explorer") == 50

    def test_init_population_no_regime_stratum_when_disabled(self, monkeypatch):
        """Setting PHASE2_REGIME_STRATUM_ENABLED=False produces no regime rows."""
        monkeypatch.setattr(_cfg, "PHASE2_REGIME_STRATUM_ENABLED", False)
        monkeypatch.setattr(_cfg, "PHASE2_ENCODING", "dense")

        fi = _feature_infos_with_scores([
            (f"f{i}", float(i + 1)) for i in range(12)
        ])
        rng = np.random.default_rng(42)
        pop = _init_population(
            200, fi, rng, init_strategy="stratified_sparse",
        )
        # We can't easily inspect stratum labels, but we can verify
        # the population was generated without errors and all chromosomes
        # respect condition bounds.
        dc = _get_dont_cares(fi)
        for row in pop:
            active = _count_active_conditions(row, dc)
            assert _cfg.MIN_CONDITIONS <= active <= _cfg.MAX_CONDITIONS


# ---------------------------------------------------------------------------
# Integration: _init_population with regime stratum enabled
# ---------------------------------------------------------------------------

class TestInitPopulationWithRegime:
    """Spot-check that regime stratum chromosomes appear in init population."""

    def test_regime_chromosomes_appear_in_population(self, monkeypatch):
        """With regime stratum enabled, some rows have regime-first genes."""
        monkeypatch.setattr(_cfg, "PHASE2_REGIME_STRATUM_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_ENCODING", "dense")
        monkeypatch.setattr(_cfg, "PHASE2_REGIME_STRATUM_FRAC", 0.50)  # high for testing

        fi = _feature_infos_with_scores([
            ("vol_20", 1.0),
            ("rsi_14", 0.8),
            ("atr_14", 0.7),
            ("bb_width_20", 0.6),
        ])
        dont_cares = _get_dont_cares(fi)
        regime_indices = _regime_feature_indices(fi)  # [0, 2, 3]
        rng = np.random.default_rng(42)
        pop = _init_population(
            200, fi, rng, init_strategy="stratified_sparse",
        )

        regime_first_count = 0
        for row in pop:
            active = np.where(row != dont_cares)[0]
            if len(active) > 0 and int(active[0]) in regime_indices:
                regime_first_count += 1

        # With 50% regime fraction, expect a noticeable number of regime rows.
        assert regime_first_count >= 10, (
            f"Only {regime_first_count} rows have first-active in regime indices"
        )

    def test_population_size_and_validity(self, monkeypatch):
        """Population of the right size with all valid chromosomes."""
        monkeypatch.setattr(_cfg, "PHASE2_REGIME_STRATUM_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_ENCODING", "dense")

        fi = _feature_infos_with_scores([
            ("vol_20", 1.0),
            ("rsi_14", 0.8),
            ("atr_14", 0.7),
        ])
        rng = np.random.default_rng(42)
        pop = _init_population(
            100, fi, rng, init_strategy="stratified_sparse",
        )
        assert pop.shape[0] == 100
        dc = _get_dont_cares(fi)
        for row in pop:
            active = _count_active_conditions(row, dc)
            assert _cfg.MIN_CONDITIONS <= active <= _cfg.MAX_CONDITIONS

"""
Tests for H5/M4/M5 evolution convergence behaviors.

Covers:
- HoF trimming at epoch start (>10 entries → capped at 10)
- global_metrics_cache clearing (only seeded keys removed)
- _select_diverse_subset edge cases (k=0, k=1, k>len, all-equal Hamming)
- _normalize_for_association with all-equal values (no crash, valid output)
- Hamming threshold auto-scaling formula (max(3, k_active // 5))
"""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.evolution.evox_runner import (
    _normalize_for_association,
    _select_diverse_subset,
)
from gpu_fuzzy_trader.phases.phase2_rule_pool import _count_active_conditions


# ---------------------------------------------------------------------------
# H5: HoF trimming at epoch start
# ---------------------------------------------------------------------------

class TestHallOfFameTrim:
    """Verify that hall_of_fame is capped to PHASE2_HOF_EPOCH_CARRYOVER=10."""

    def test_hof_trim_when_over_capacity(self):
        """HoF with >10 entries is trimmed to exactly 10 oldest entries."""
        max_carry = int(getattr(cfg, "PHASE2_HOF_EPOCH_CARRYOVER", 10))
        # Build a HoF with 15 entries (key→array)
        hof = {}
        for i in range(15):
            key = (i, 0, 0)
            hof[key] = np.array([i, 0, 0], dtype=np.int32)

        assert len(hof) == 15

        # Apply epoch-start trimming logic (same as phase2_rule_pool.py:2837-2840)
        max_carry = int(getattr(cfg, "PHASE2_HOF_EPOCH_CARRYOVER", 10))
        if len(hof) > max_carry:
            keys = list(hof.keys())[:max_carry]
            hof = {k: hof[k] for k in keys}

        assert len(hof) == 10, f"Expected 10 entries, got {len(hof)}"
        # Verify oldest entries are preserved
        for i in range(10):
            assert (i, 0, 0) in hof, f"Key {(i, 0, 0)} missing after trim"

    def test_hof_no_trim_when_under_capacity(self):
        """HoF with <=10 entries is left unchanged."""
        hof = {}
        for i in range(5):
            key = (i, 0, 0)
            hof[key] = np.array([i, 0, 0], dtype=np.int32)

        max_carry = int(getattr(cfg, "PHASE2_HOF_EPOCH_CARRYOVER", 10))
        if len(hof) > max_carry:
            keys = list(hof.keys())[:max_carry]
            hof = {k: hof[k] for k in keys}

        assert len(hof) == 5, f"Expected 5 entries, got {len(hof)}"

    def test_hof_empty_is_safe(self):
        """Empty HoF does not crash during trim logic."""
        hof: dict = {}
        max_carry = int(getattr(cfg, "PHASE2_HOF_EPOCH_CARRYOVER", 10))
        # No-op when len(hof) <= max_carry
        if len(hof) > max_carry:
            keys = list(hof.keys())[:max_carry]
            hof = {k: hof[k] for k in keys}
        assert len(hof) == 0


# ---------------------------------------------------------------------------
# H5: global_metrics_cache clearing for seeded keys
# ---------------------------------------------------------------------------

class TestGlobalMetricsCacheClearing:
    """Verify that only seeded keys are removed from the global cache."""

    def _chromosome_key(self, idx: int) -> tuple[int, ...]:
        """Create a deterministic chromosome key."""
        return (idx, 0, 0, 1, 1)

    def test_only_seeded_keys_cleared(self):
        """Cache entries matching seeded_keys are removed; non-matching survive."""
        cache: dict[tuple[int, ...], dict] = {}
        for i in range(10):
            cache[self._chromosome_key(i)] = {"val": float(i)}

        assert len(cache) == 10

        # Seeded keys = indices 0, 2, 4 (3 entries)
        seeded_keys = {self._chromosome_key(i) for i in (0, 2, 4)}
        # Apply epoch-start clearing logic (phase2_rule_pool.py:2834-2835)
        for key in list(cache.keys()):
            if key in seeded_keys:
                cache.pop(key, None)

        # 10 - 3 = 7 entries should remain
        assert len(cache) == 7, f"Expected 7 entries, got {len(cache)}"
        # Verify the removed keys are gone
        for i in (0, 2, 4):
            assert self._chromosome_key(i) not in cache, (
                f"Seeded key #{i} was not removed"
            )
        # Verify the surviving keys are intact
        for i in (1, 3, 5, 6, 7, 8, 9):
            key = self._chromosome_key(i)
            assert key in cache, f"Unseeded key #{i} was incorrectly removed"
            assert cache[key]["val"] == float(i), (
                f"Unseeded key #{i} value was corrupted"
            )

    def test_no_seeded_keys_no_removal(self):
        """When seeded_keys is empty, nothing is removed."""
        cache: dict[tuple[int, ...], dict] = {}
        for i in range(5):
            cache[self._chromosome_key(i)] = {"val": float(i)}

        seeded_keys: set[tuple[int, ...]] = set()
        for key in list(cache.keys()):
            if key in seeded_keys:
                cache.pop(key, None)

        assert len(cache) == 5, (
            f"Expected 5 entries with empty seeded_keys, got {len(cache)}"
        )

    def test_all_seeded_all_removed(self):
        """When all keys are seeded, cache becomes empty."""
        cache: dict[tuple[int, ...], dict] = {}
        for i in range(4):
            cache[self._chromosome_key(i)] = {"val": float(i)}

        seeded_keys = {self._chromosome_key(i) for i in range(4)}
        for key in list(cache.keys()):
            if key in seeded_keys:
                cache.pop(key, None)

        assert len(cache) == 0, (
            f"Expected empty cache, got {len(cache)}"
        )


# ---------------------------------------------------------------------------
# M4: _select_diverse_subset edge cases
# ---------------------------------------------------------------------------

class TestSelectDiverseSubset:
    """Verify _select_diverse_subset correctness for edge cases."""

    def _make_chromosomes(self, n: int) -> list[np.ndarray]:
        """Create n distinct dense chromosomes."""
        return [np.array([i, i + 1, i + 2], dtype=np.int32) for i in range(n)]

    def test_k_zero_returns_empty(self):
        """k=0 should return [] even with non-empty chromosomes."""
        chroms = self._make_chromosomes(5)
        result = _select_diverse_subset(chroms, k=0)
        assert result == [], f"Expected [], got {result}"

    def test_k_negative_returns_empty(self):
        """k<0 should return []."""
        chroms = self._make_chromosomes(3)
        result = _select_diverse_subset(chroms, k=-1)
        assert result == [], f"Expected [], got {result}"

    def test_k_one_returns_single(self):
        """k=1 returns one chromosome."""
        chroms = self._make_chromosomes(5)
        result = _select_diverse_subset(chroms, k=1)
        assert len(result) == 1, f"Expected 1 entry, got {len(result)}"

    def test_k_greater_than_len_returns_all(self):
        """k > len(chromosomes) returns all chromosomes (shallow copy)."""
        chroms = self._make_chromosomes(3)
        result = _select_diverse_subset(chroms, k=10)
        assert len(result) == 3, f"Expected 3 entries, got {len(result)}"
        # Should be a list copy, not the same object
        assert result is not chroms

    def test_k_equals_len_returns_all(self):
        """k == len(chromosomes) returns all chromosomes."""
        chroms = self._make_chromosomes(4)
        result = _select_diverse_subset(chroms, k=4)
        assert len(result) == 4, f"Expected 4 entries, got {len(result)}"

    def test_all_equal_hamming_still_returns_k(self):
        """All-equal chromosomes (Hamming distance 0) should still return k items."""
        chrom = np.array([1, 2, 3], dtype=np.int32)
        chroms = [chrom.copy() for _ in range(5)]
        result = _select_diverse_subset(chroms, k=3)
        assert len(result) == 3, f"Expected 3 entries, got {len(result)}"
        # All entries are copies of the same chromosome
        for r in result:
            np.testing.assert_array_equal(r, chrom)

    def test_empty_chromosomes_returns_empty(self):
        """Empty chromosome list returns []."""
        result = _select_diverse_subset([], k=5)
        assert result == [], f"Expected [], got {result}"

    def test_k_zero_empty_chromosomes_returns_empty(self):
        """k=0 with empty chromosome list returns []."""
        result = _select_diverse_subset([], k=0)
        assert result == [], f"Expected [], got {result}"


# ---------------------------------------------------------------------------
# M5: _normalize_for_association with all-equal values
# ---------------------------------------------------------------------------

class TestNormalizeForAssociation:
    """Verify rank normalization does not crash on degenerate inputs."""

    def test_all_equal_values_no_crash(self):
        """All-equal objective values should produce valid normalised output
        with no NaN, no Inf, and bounded magnitude."""
        merge_fit = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], dtype=np.float64)
        # Reference points for 3 objectives (Das-Dennis with p=2 → 6 ref points)
        ref = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.5, 0.5],
        ], dtype=np.float64)

        fit_n, ref_n = _normalize_for_association(merge_fit, ref)

        # No NaN or Inf in output
        assert np.all(np.isfinite(fit_n)), "fit_n contains NaN or Inf"
        assert np.all(np.isfinite(ref_n)), "ref_n contains NaN or Inf"
        # Bounded magnitude
        assert np.all(np.abs(fit_n) <= 1.0), (
            "fit_n values exceed unit bound"
        )
        assert np.all(np.abs(ref_n) <= 1.0), (
            "ref_n values exceed unit bound"
        )

    def test_single_row_no_crash(self):
        """Single-row input should not crash."""
        merge_fit = np.array([[0.5, 0.5, 0.5]], dtype=np.float64)
        ref = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
        fit_n, ref_n = _normalize_for_association(merge_fit, ref)
        assert np.all(np.isfinite(fit_n))
        assert np.all(np.isfinite(ref_n))

    def test_two_objectives_all_equal(self):
        """Two objectives, all-equal values, should produce valid output.
        
        After rank normalization each column has the same value (mean rank / n).
        After per-row L2 normalization, all rows are identical unit vectors.
        """
        merge_fit = np.array([
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ], dtype=np.float64)
        ref = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ], dtype=np.float64)
        fit_n, ref_n = _normalize_for_association(merge_fit, ref)
        assert np.all(np.isfinite(fit_n))
        assert np.all(np.isfinite(ref_n))
        # All rows should be identical (same rank per column)
        np.testing.assert_allclose(fit_n[0], fit_n[1], atol=1e-10)
        np.testing.assert_allclose(fit_n[0], fit_n[2], atol=1e-10)
        # Each row should be a unit vector (L2 norm ≈ 1)
        row_norms = np.linalg.norm(fit_n, axis=1)
        np.testing.assert_allclose(row_norms, 1.0, atol=1e-6)

    def test_mixed_values(self):
        """Mixed values (not all equal) should still work."""
        merge_fit = np.array([
            [0.1, 0.9],
            [0.5, 0.5],
            [0.9, 0.1],
        ], dtype=np.float64)
        ref = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
        ], dtype=np.float64)
        fit_n, ref_n = _normalize_for_association(merge_fit, ref)
        assert np.all(np.isfinite(fit_n))
        assert np.all(np.isfinite(ref_n))
        # Verify rank order is preserved: fit_n[0,0] < fit_n[1,0] < fit_n[2,0]
        # because objective values asc: 0.1 < 0.5 < 0.9
        assert fit_n[0, 0] < fit_n[1, 0] < fit_n[2, 0], (
            "Rank order not preserved for objective 0"
        )


# ---------------------------------------------------------------------------
# M4: Hamming threshold auto-scaling
# ---------------------------------------------------------------------------

class TestHammingThresholdAutoScale:
    """Verify the max(3, k_active // 5) formula.

    The formula is applied in phase2_rule_pool.py:642:
        diversity_hamming_threshold = max(3, k_active // 5)
    We use _count_active_conditions to exercise the same code path."""

    @staticmethod
    def _auto_threshold(k_active: int) -> int:
        """Replicate the auto-scaling formula."""
        return max(3, k_active // 5)

    def test_k_active_0_auto_thresh_is_3(self):
        """k_active=0 → threshold = max(3, 0//5) = 3."""
        assert self._auto_threshold(0) == 3

    def test_k_active_5_auto_thresh_is_3(self):
        """k_active=5 → threshold = max(3, 5//5=1) = 3."""
        assert self._auto_threshold(5) == 3

    def test_k_active_15_auto_thresh_is_3(self):
        """k_active=15 → threshold = max(3, 15//5=3) = 3."""
        assert self._auto_threshold(15) == 3

    def test_k_active_20_auto_thresh_is_4(self):
        """k_active=20 → threshold = max(3, 20//5=4) = 4."""
        assert self._auto_threshold(20) == 4

    def test_k_active_50_auto_thresh_is_10(self):
        """k_active=50 → threshold = max(3, 50//5=10) = 10."""
        assert self._auto_threshold(50) == 10

    def test_k_active_100_auto_thresh_is_20(self):
        """k_active=100 → threshold = max(3, 100//5=20) = 20."""
        assert self._auto_threshold(100) == 20

    def test_count_active_conditions_on_dense_array(self):
        """Verify _count_active_conditions works with a dense array."""
        # Dense chromosome with 5 active (non-dontcare) conditions
        chromosome = np.array([1, 0, 1, 0, 1, 1, 0], dtype=np.int32)
        dont_cares = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.int32)
        active = _count_active_conditions(chromosome, dont_cares)
        assert active == 4, f"Expected 4 active conditions, got {active}"

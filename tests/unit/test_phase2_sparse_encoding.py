"""Unit tests for Phase 2 sparse slot chromosome encoding."""

from __future__ import annotations

import numpy as np

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _count_active_conditions,
    _get_dont_cares,
    _hamming_distance,
    _init_population,
)
from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
    canonicalize_slots,
    dense_to_sparse,
    max_slots,
    sparse_hamming,
    sparse_to_dense,
    use_sparse_slots,
)


def _feature_infos(n: int = 8) -> list[dict]:
    return [
        {"name": f"f{i}", "mode": "positive", "score": float(i + 1)}
        for i in range(n)
    ]


def test_use_sparse_slots_enabled():
    assert use_sparse_slots()


def test_dense_sparse_roundtrip():
    fi = _feature_infos(6)
    dc = _get_dont_cares(fi)
    dense = np.array([5, 5, 2, 5, 0, 5], dtype=np.int32)
    slots = dense_to_sparse(dense, dc)
    assert _count_active_conditions(slots, dc) == 2
    back = sparse_to_dense(slots, dc)
    assert np.array_equal(back, dense)


def test_dynamic_min_max_repair():
    fi = _feature_infos(10)
    dc = _get_dont_cares(fi)
    rng = np.random.default_rng(0)
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
        repair_sparse_slots,
        sample_sparse_slots_chromosome,
    )

    orig_min, orig_max = _cfg.MIN_CONDITIONS, _cfg.MAX_CONDITIONS
    try:
        _cfg.MIN_CONDITIONS = 2
        _cfg.MAX_CONDITIONS = 5
        for k in (2, 3, 5):
            chrom = sample_sparse_slots_chromosome(
                rng, fi, dc, k, "explorer", np.ones(len(fi)) / len(fi),
            )
            assert _count_active_conditions(chrom, dc) == k
            repaired = repair_sparse_slots(chrom, fi, dc, rng)
            active = _count_active_conditions(repaired, dc)
            assert _cfg.MIN_CONDITIONS <= active <= _cfg.MAX_CONDITIONS
    finally:
        _cfg.MIN_CONDITIONS = orig_min
        _cfg.MAX_CONDITIONS = orig_max


def test_init_population_sparse_shape():
    fi = _feature_infos(12)
    dc = _get_dont_cares(fi)
    pop = _init_population(50, fi, np.random.default_rng(1))
    assert pop.shape == (50, max_slots(), 2)
    for row in pop:
        active = _count_active_conditions(row, dc)
        assert _cfg.MIN_CONDITIONS <= active <= _cfg.MAX_CONDITIONS


def test_sparse_hamming_ignores_inactive_slots():
    fi = _feature_infos(6)
    dc = _get_dont_cares(fi)
    a = dense_to_sparse(np.array([5, 1, 5, 5, 5, 5], dtype=np.int32), dc)
    b = dense_to_sparse(np.array([5, 2, 5, 5, 5, 5], dtype=np.int32), dc)
    assert sparse_hamming(a, b) == 1
    assert _hamming_distance(a, b) == 1


def test_canonicalize_sorts_and_dedupes():
    slots = np.array([[3, 1], [1, 2], [-1, 0], [-1, 0]], dtype=np.int32)
    out = canonicalize_slots(slots)
    assert out[0, 0] == 1
    assert out[1, 0] == 3
    assert out[2, 0] == -1

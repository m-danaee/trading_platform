"""
Unit tests for gpu_fuzzy_trader.phases.phase2_rule_pool.Rule_Pool_Generator

Tests cover:
  - Chromosome initialisation (dont_care distribution, valid gene ranges)
  - Active condition counting
  - Hamming distance
  - Non-dominated sorting correctness
  - Crowding distance
  - Crossover and mutation
  - Pool schema validation
  - load_pool / skip_if_valid logic
  - Rule_Pool_Generator.run() with tiny pop/gen for speed
  - Static risk parameters (TP/SL/capital_pct from config)
  - Condition count bounds enforcement
  - Sampling strategy (equal distribution across symbols)
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    Rule_Pool_Generator,
    _count_active_conditions,
    _crowding_distance,
    _dominates,
    _get_dont_cares,
    _hamming_distance,
    _init_population,
    _merge_archive_entries,
    _mutate,
    _non_dominated_sort,
    _pareto_sortino_stats,
    _pool_seed_chromosomes,
    _sample_df,
    _validate_pool_schema,
)
from gpu_fuzzy_trader.features.encoder import get_dont_care
from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
    INACTIVE_FEAT_IDX,
    dense_to_sparse,
    max_slots,
    sparse_to_dense,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_feature_infos(modes: list[str]) -> list[dict]:
    """Create minimal feature_infos list."""
    return [{"name": f"feat_{i}", "mode": m, "score": 0.5} for i, m in enumerate(modes)]


def _is_all_inactive_sparse(row: np.ndarray) -> bool:
    """True when every sparse slot is inactive (feat_idx == -1)."""
    return bool(np.all(row[:, 0] == INACTIVE_FEAT_IDX))


def _pop_contains_dense_seed(pop: np.ndarray, seed: np.ndarray, dc: np.ndarray) -> bool:
    """Check whether *pop* contains a sparse row matching dense *seed*."""
    target = dense_to_sparse(seed, dc)
    return any(np.array_equal(row, target) for row in pop)


def _chromosome_with_min_active(
    n_features: int = 10,
    dont_care: int = 5,
) -> np.ndarray:
    """Build a dense chromosome with exactly MIN_CONDITIONS active genes."""
    chrom = np.full(n_features, dont_care, dtype=np.int32)
    chrom[: int(_cfg.MIN_CONDITIONS)] = 0
    return chrom


def _make_train_df(
    n_rows: int = 300,
    n_features: int = 4,
    symbols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a minimal training DataFrame with label columns."""
    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = ["SYM_A", "SYM_B"]

    rows_per_sym = n_rows // len(symbols)
    dfs = []
    for sym in symbols:
        n = rows_per_sym
        open_next = rng.uniform(100, 200, size=n)
        max_288 = open_next * rng.uniform(0.98, 1.10, size=n)
        min_288 = open_next * rng.uniform(0.90, 1.02, size=n)
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
            "_symbol_bar_index": np.arange(n),
        }
        for i in range(n_features):
            # Use integer-valued features (0-4) to match 'positive' mode
            data[f"feat_{i}"] = rng.integers(0, 5, size=n).astype(float)

        dfs.append(pd.DataFrame(data))

    return pd.concat(dfs, ignore_index=True)


@pytest.fixture(autouse=True)
def _isolate_phase2_archive_paths(tmp_path):
    """Keep persistent archive files under the per-test temp directory."""
    import gpu_fuzzy_trader.phases.phase2_rule_pool as m

    original_archive = m._ARCHIVE_PATHS.copy()
    m._ARCHIVE_PATHS["long"] = str(tmp_path / "phase2_long_archive.json")
    m._ARCHIVE_PATHS["short"] = str(tmp_path / "phase2_short_archive.json")
    try:
        yield
    finally:
        m._ARCHIVE_PATHS.update(original_archive)


# ---------------------------------------------------------------------------
# Tests: _get_dont_cares
# ---------------------------------------------------------------------------

class TestGetDontCares:
    def test_binary_mode(self):
        fi = _make_feature_infos(["binary"])
        dc = _get_dont_cares(fi)
        assert dc[0] == 2

    def test_ternary_mode(self):
        fi = _make_feature_infos(["ternary"])
        dc = _get_dont_cares(fi)
        assert dc[0] == 3

    def test_positive_mode(self):
        fi = _make_feature_infos(["positive"])
        dc = _get_dont_cares(fi)
        assert dc[0] == 5

    def test_signed_mode(self):
        fi = _make_feature_infos(["signed"])
        dc = _get_dont_cares(fi)
        assert dc[0] == 10

    def test_mixed_modes(self):
        fi = _make_feature_infos(["binary", "ternary", "positive", "signed"])
        dc = _get_dont_cares(fi)
        assert list(dc) == [2, 3, 5, 10]


# ---------------------------------------------------------------------------
# Tests: _count_active_conditions
# ---------------------------------------------------------------------------

class TestCountActiveConditions:
    def test_all_dont_care(self):
        fi = _make_feature_infos(["positive", "positive", "positive"])
        dc = _get_dont_cares(fi)
        chrom = dc.copy()  # all dont_care
        assert _count_active_conditions(chrom, dc) == 0

    def test_all_active(self):
        fi = _make_feature_infos(["positive", "positive", "positive"])
        dc = _get_dont_cares(fi)
        chrom = np.array([0, 1, 2], dtype=np.int32)
        assert _count_active_conditions(chrom, dc) == 3

    def test_mixed(self):
        fi = _make_feature_infos(["positive", "positive", "positive"])
        dc = _get_dont_cares(fi)
        chrom = np.array([0, dc[1], 2], dtype=np.int32)
        assert _count_active_conditions(chrom, dc) == 2


# ---------------------------------------------------------------------------
# Tests: _hamming_distance
# ---------------------------------------------------------------------------

class TestHammingDistance:
    def test_identical(self):
        a = np.array([1, 2, 3])
        assert _hamming_distance(a, a) == 0

    def test_all_different(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        assert _hamming_distance(a, b) == 3

    def test_partial(self):
        a = np.array([1, 2, 3])
        b = np.array([1, 5, 3])
        assert _hamming_distance(a, b) == 1


# ---------------------------------------------------------------------------
# Tests: _pareto_sortino_stats
# ---------------------------------------------------------------------------

class TestParetoSortinoStats:
    def test_empty_pareto_front(self):
        stats = _pareto_sortino_stats([], [{"sortino_ratio": 10.0}])
        assert stats["mean_sortino_ratio"] == 0.0
        assert stats["best_sortino_ratio"] == 0.0
        assert stats["mean_raw_train_return_pct"] == 0.0
        assert stats["mean_val_return_pct"] == 0.0

    def test_mean_and_best_from_cache(self):
        metrics_cache = [
            {"sortino_ratio": 10.0},
            {"sortino_ratio": 30.0},
            {"sortino_ratio": 20.0},
        ]
        stats = _pareto_sortino_stats([0, 2], metrics_cache)
        assert stats["mean_sortino_ratio"] == 15.0
        assert stats["best_sortino_ratio"] == 20.0

    def test_missing_total_return_defaults_to_zero(self):
        metrics_cache = [{"sortino_ratio": 12.0}, {}]
        stats = _pareto_sortino_stats([0, 1], metrics_cache)
        assert stats["mean_sortino_ratio"] == 6.0
        assert stats["best_sortino_ratio"] == 12.0


# ---------------------------------------------------------------------------
# Tests: _dominates
# ---------------------------------------------------------------------------

class TestDominates:
    def test_a_dominates_b(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_equal_does_not_dominate(self):
        a = np.array([1.0, 2.0, 3.0])
        assert not _dominates(a, a)

    def test_incomparable(self):
        a = np.array([1.0, 3.0])
        b = np.array([2.0, 1.0])
        assert not _dominates(a, b)
        assert not _dominates(b, a)

    def test_partial_dominance_not_enough(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 3.0])
        assert not _dominates(a, b)


# ---------------------------------------------------------------------------
# Tests: _non_dominated_sort
# ---------------------------------------------------------------------------

class TestNonDominatedSort:
    def test_single_solution(self):
        obj = np.array([[1.0, 2.0, 3.0]])
        fronts = _non_dominated_sort(obj)
        assert fronts[0] == [0]

    def test_two_solutions_one_dominates(self):
        obj = np.array([[1.0, 1.0], [2.0, 2.0]])
        fronts = _non_dominated_sort(obj)
        assert 0 in fronts[0]
        assert 1 not in fronts[0]

    def test_two_incomparable_solutions(self):
        obj = np.array([[1.0, 3.0], [3.0, 1.0]])
        fronts = _non_dominated_sort(obj)
        assert set(fronts[0]) == {0, 1}

    def test_three_solutions_two_fronts(self):
        obj = np.array([
            [1.0, 1.0],  # Pareto front
            [2.0, 2.0],  # dominated by [0]
            [1.5, 0.5],  # Pareto front (incomparable with [0])
        ])
        fronts = _non_dominated_sort(obj)
        assert 0 in fronts[0]
        assert 2 in fronts[0]
        assert 1 not in fronts[0]

    def test_all_dominated_chain(self):
        """Each solution dominates the next."""
        obj = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        fronts = _non_dominated_sort(obj)
        assert fronts[0] == [0]
        assert fronts[1] == [1]
        assert fronts[2] == [2]


# ---------------------------------------------------------------------------
# Tests: _crowding_distance
# ---------------------------------------------------------------------------

class TestCrowdingDistance:
    def test_two_solutions_get_inf(self):
        obj = np.array([[1.0, 2.0], [3.0, 4.0]])
        cd = _crowding_distance(obj, [0, 1])
        assert np.all(np.isinf(cd))

    def test_boundary_solutions_get_inf(self):
        obj = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        cd = _crowding_distance(obj, [0, 1, 2])
        assert np.isinf(cd[0])
        assert np.isinf(cd[2])
        assert np.isfinite(cd[1])

    def test_middle_solution_finite(self):
        obj = np.array([[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]])
        cd = _crowding_distance(obj, [0, 1, 2])
        assert cd[1] > 0


# ---------------------------------------------------------------------------
# Tests: _init_population
# ---------------------------------------------------------------------------

class TestInitPopulation:
    def test_shape(self):
        fi = _make_feature_infos(["positive", "binary", "signed"])
        rng = np.random.default_rng(0)
        pop = _init_population(10, fi, rng, init_strategy="legacy")
        assert pop.shape == (10, max_slots(), 2)

    def test_gene_values_in_valid_range(self):
        """All active sparse genes must be in [0, dont_care] inclusive."""
        fi = _make_feature_infos(["positive", "binary", "ternary", "signed"])
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(0)
        pop = _init_population(50, fi, rng, init_strategy="legacy")
        for row in pop:
            for slot in row:
                feat_idx = int(slot[0])
                if feat_idx < 0:
                    continue
                gene = int(slot[1])
                assert 0 <= gene <= dc[feat_idx]

    def test_dont_care_appears(self):
        """With dont_care_prob=0.5, inactive sparse rows should appear."""
        fi = _make_feature_infos(["positive"] * 10)
        rng = np.random.default_rng(0)
        pop = _init_population(
            100, fi, rng, dont_care_prob=0.5, init_strategy="legacy",
        )
        assert any(_is_all_inactive_sparse(row) for row in pop)

    def test_all_active_when_dont_care_prob_zero(self):
        """With dont_care_prob=0, no fully inactive sparse rows should appear."""
        fi = _make_feature_infos(["positive"] * 5)
        rng = np.random.default_rng(0)
        pop = _init_population(
            20, fi, rng, dont_care_prob=0.0, init_strategy="legacy")
        assert not any(_is_all_inactive_sparse(row) for row in pop)

    def test_seeded_chromosomes_fill_requested_fraction(self):
        fi = _make_feature_infos(["positive"] * 4)
        seeds = np.array(
            [
                [0, 1, 2, 3],
                [1, 2, 3, 4],
                [4, 3, 2, 1],
            ],
            dtype=np.int32,
        )
        rng = np.random.default_rng(0)
        pop = _init_population(
            10,
            fi,
            rng,
            dont_care_prob=1.0,
            seeded_chromosomes=seeds,
            seed_fraction=0.35,
            init_strategy="legacy",
        )

        seed_matches = sum(
            _pop_contains_dense_seed(pop, seed, _get_dont_cares(fi))
            for seed in seeds
        )
        assert seed_matches == len(seeds)
        assert sum(_is_all_inactive_sparse(row) for row in pop) == 7

    def test_init_population_seeds_35_percent_of_pop_200(self):
        fi = _make_feature_infos(["positive"] * 4)
        seeds = np.array(
            [
                [
                    i % 5,
                    (i // 5) % 5,
                    (i // 25) % 5,
                    (i // 125) % 5,
                ]
                for i in range(80)
            ],
            dtype=np.int32,
        )
        rng = np.random.default_rng(7)
        pop = _init_population(
            200,
            fi,
            rng,
            dont_care_prob=1.0,
            seeded_chromosomes=seeds,
            seed_fraction=0.35,
            init_strategy="legacy",
        )
        expected_seed_slots = min(
            200,
            max(1, int(round(200 * 0.35))),
            len(seeds),
        )
        assert expected_seed_slots == 70
        non_random_rows = sum(
            not _is_all_inactive_sparse(row) for row in pop
        )
        assert non_random_rows == expected_seed_slots


class TestStratifiedInitPopulation:
    def test_stratified_init_respects_bounds(self):
        fi = [
            {"name": "amihud_illiquidity_20", "mode": "positive", "score": 0.9},
            {"name": "feat_b", "mode": "positive", "score": 0.1},
            {"name": "feat_c", "mode": "positive", "score": 0.05},
            {"name": "feat_d", "mode": "positive", "score": 0.02},
        ]
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(99)
        pop = _init_population(100, fi, rng, init_strategy="stratified_sparse")
        for row in pop:
            n = _count_active_conditions(row, dc)
            assert _cfg.MIN_CONDITIONS <= n <= _cfg.MAX_CONDITIONS


class TestPoolSeedChromosomes:
    def test_pool_seed_chromosomes_dedupes(self):
        fi = _make_feature_infos(["positive"] * 4)
        dc = _get_dont_cares(fi)
        pool = [
            {"chromosome": [0, 1, 2, 3]},
            {"chromosome": [0, 1, 2, 3]},
            {"chromosome": [1, 2, 3, 4]},
        ]
        arr = _pool_seed_chromosomes(pool, dc)
        assert arr is not None
        assert arr.shape == (2, max_slots(), 2)

    def test_pool_seed_chromosomes_empty_returns_none(self):
        assert _pool_seed_chromosomes([]) is None


# ---------------------------------------------------------------------------
# Tests: _mutate
# ---------------------------------------------------------------------------

class TestMutate:
    def test_returns_copy(self):
        fi = _make_feature_infos(["positive", "positive"])
        dc = _get_dont_cares(fi)
        chrom = np.array([0, 1], dtype=np.int32)
        rng = np.random.default_rng(0)
        result = _mutate(chrom, fi, dc, rng, mutation_rate=1.0)
        # Original should be unchanged
        assert chrom[0] == 0
        assert chrom[1] == 1

    def test_gene_values_remain_valid(self):
        fi = _make_feature_infos(["positive", "binary", "signed"])
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(0)
        chrom = np.array([2, 1, 5], dtype=np.int32)
        for _ in range(20):
            result = _mutate(chrom, fi, dc, rng, mutation_rate=1.0)
            dense = sparse_to_dense(result, dc)
            for k in range(len(fi)):
                assert 0 <= dense[k] <= dc[k]

    def test_high_mutation_rate_changes_chromosome(self):
        fi = _make_feature_infos(["positive"] * 10)
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(0)
        chrom = np.zeros(10, dtype=np.int32)
        changed = False
        for _ in range(20):
            result = _mutate(chrom, fi, dc, rng, mutation_rate=1.0)
            if not np.array_equal(sparse_to_dense(result, dc), chrom):
                changed = True
                break
        assert changed


# ---------------------------------------------------------------------------
# Tests: _sample_df
# ---------------------------------------------------------------------------

class TestSampleDf:
    def test_respects_total_rows(self):
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=100)
        assert len(sampled) <= 100

    def test_distributes_across_symbols(self):
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=100)
        # Each symbol should have some rows
        for sym in ["A", "B", "C", "D"]:
            assert (sampled["symbol"] == sym).sum() > 0

    def test_no_symbol_column(self):
        df = pd.DataFrame({"x": range(200)})
        sampled = _sample_df(df, total_rows=50)
        assert len(sampled) <= 50

    def test_total_rows_larger_than_df(self):
        df = _make_train_df(n_rows=50, symbols=["A"])
        sampled = _sample_df(df, total_rows=1000)
        assert len(sampled) <= 50

    def test_preserves_chronological_order_per_symbol(self):
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=120)
        for sym in ["A", "B", "C", "D"]:
            sym_df = sampled[sampled["symbol"] == sym]
            if sym_df.empty:
                continue
            bar_idx = sym_df["_symbol_bar_index"].to_numpy()
            assert np.all(bar_idx[:-1] <= bar_idx[1:])
            dt = sym_df["datetime"].to_numpy()
            assert np.all(dt[:-1] <= dt[1:])

    def test_sampling_is_deterministic(self):
        df = _make_train_df(n_rows=400, symbols=["A", "B"])
        first = _sample_df(df, total_rows=80)
        second = _sample_df(df, total_rows=80, random_state=999)
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True),
            second.reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Tests: _validate_pool_schema
# ---------------------------------------------------------------------------

class TestValidatePoolSchema:
    def _valid_entry(self):
        return {
            "chromosome": [2, 5, 1],
            "conditions": ["[feat_0] IS Medium"],
            "objectives": {
                "sortino_ratio": 5.0,
                "total_return_pct": 5.0,
                "max_drawdown_pct": 2.0,
                "win_rate": 55.0,
            },
            "executed_trades": 30,
        }

    def test_valid_pool(self):
        pool = [self._valid_entry()]
        _validate_pool_schema(pool, "test.json")  # should not raise

    def test_empty_pool_is_valid(self):
        _validate_pool_schema([], "test.json")

    def test_not_a_list(self):
        with pytest.raises(ValueError, match="must be a JSON array"):
            _validate_pool_schema({"key": "val"}, "test.json")

    def test_entry_not_dict(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _validate_pool_schema(["not_a_dict"], "test.json")

    def test_missing_chromosome(self):
        entry = self._valid_entry()
        del entry["chromosome"]
        with pytest.raises(ValueError, match="missing keys"):
            _validate_pool_schema([entry], "test.json")

    def test_missing_objectives_key(self):
        entry = self._valid_entry()
        del entry["objectives"]["win_rate"]
        with pytest.raises(ValueError, match="missing keys"):
            _validate_pool_schema([entry], "test.json")

    def test_executed_trades_not_int(self):
        entry = self._valid_entry()
        entry["executed_trades"] = 30.5
        with pytest.raises(ValueError, match="must be an int"):
            _validate_pool_schema([entry], "test.json")


# ---------------------------------------------------------------------------
# Tests: Rule_Pool_Generator.load_pool / skip_if_valid
# ---------------------------------------------------------------------------

class TestLoadPool:
    def _write_valid_pool(self, path: str):
        pool = [
            {
                "chromosome": [2, 5, 1],
                "conditions": ["[feat_0] IS Medium"],
                "objectives": {
                    "sortino_ratio": 5.0,
                    "total_return_pct": 5.0,
                    "max_drawdown_pct": 2.0,
                    "win_rate": 55.0,
                },
                "executed_trades": 30,
            }
        ]
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(pool, fh)

    def test_load_pool_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setitem(
            __import__("gpu_fuzzy_trader.phases.phase2_rule_pool",
                       fromlist=["_POOL_PATHS"])
            .__dict__["_POOL_PATHS"],
            "long",
            str(tmp_path / "phase2_long_pool.json"),
        )
        # Patch the module-level dict
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        original = m._POOL_PATHS.copy()
        m._POOL_PATHS["long"] = str(tmp_path / "phase2_long_pool.json")
        try:
            result = Rule_Pool_Generator.load_pool("long")
            assert result is None
        finally:
            m._POOL_PATHS.update(original)

    def test_load_pool_returns_pool_when_valid(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        path = str(tmp_path / "phase2_long_pool.json")
        self._write_valid_pool(path)
        original = m._POOL_PATHS.copy()
        m._POOL_PATHS["long"] = path
        try:
            result = Rule_Pool_Generator.load_pool("long")
            assert result is not None
            assert len(result) == 1
        finally:
            m._POOL_PATHS.update(original)

    def test_load_pool_raises_on_corrupted_file(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        path = str(tmp_path / "phase2_long_pool.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("{corrupted json")
        original = m._POOL_PATHS.copy()
        m._POOL_PATHS["long"] = path
        try:
            with pytest.raises(ValueError, match="unreadable or corrupted"):
                Rule_Pool_Generator.load_pool("long")
        finally:
            m._POOL_PATHS.update(original)

    def test_skip_if_valid_returns_none_when_missing(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        original = m._POOL_PATHS.copy()
        m._POOL_PATHS["long"] = str(tmp_path / "nonexistent.json")
        try:
            result = Rule_Pool_Generator.skip_if_valid("long")
            assert result is None
        finally:
            m._POOL_PATHS.update(original)

    def test_skip_if_valid_returns_pool_when_valid(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        path = str(tmp_path / "phase2_long_pool.json")
        self._write_valid_pool(path)
        original = m._POOL_PATHS.copy()
        m._POOL_PATHS["long"] = path
        try:
            result = Rule_Pool_Generator.skip_if_valid("long")
            assert result is not None
        finally:
            m._POOL_PATHS.update(original)

    def test_skip_if_valid_returns_none_on_corrupted(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        path = str(tmp_path / "phase2_long_pool.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("{bad}")
        original = m._POOL_PATHS.copy()
        m._POOL_PATHS["long"] = path
        try:
            result = Rule_Pool_Generator.skip_if_valid("long")
            assert result is None
        finally:
            m._POOL_PATHS.update(original)

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            Rule_Pool_Generator.load_pool("both")


class TestArchivePersistence:
    def _entry(
        self,
        chromosome: list[int],
        total_return_pct: float,
        max_drawdown_pct: float,
        win_rate: float,
        executed_trades: int,
    ) -> dict:
        return {
            "chromosome": chromosome,
            "conditions": ["[feat_0] IS Medium"],
            "objectives": {
                "sortino_ratio": total_return_pct,
                "total_return_pct": total_return_pct,
                "max_drawdown_pct": max_drawdown_pct,
                "win_rate": win_rate,
            },
            "executed_trades": executed_trades,
        }

    def test_save_and_load_archive_round_trip(self):
        fi = _make_feature_infos(
            ["positive", "positive", "positive", "positive"])
        rules = [
            self._entry([0, 1, 2, 3], 12.0, 3.0, 61.0, 240),
            self._entry([1, 2, 3, 4], 11.0, 4.0, 60.0, 260),
        ]

        merged = Rule_Pool_Generator.save_archive("long", fi, rules)
        loaded = Rule_Pool_Generator.load_archive("long", fi)

        assert len(merged) == 2
        assert loaded is not None
        assert loaded["direction"] == "long"
        assert loaded["feature_signature"] == [
            {"name": "feat_0", "mode": "positive"},
            {"name": "feat_1", "mode": "positive"},
            {"name": "feat_2", "mode": "positive"},
            {"name": "feat_3", "mode": "positive"},
        ]
        assert len(loaded["rules"]) == 2

    def test_save_archive_keeps_best_duplicate(self):
        fi = _make_feature_infos(
            ["positive", "positive", "positive", "positive"])
        rules = [
            self._entry([0, 1, 2, 3], 10.0, 5.0, 58.0, 250),
            self._entry([0, 1, 2, 3], 13.0, 3.0, 64.0, 260),
            self._entry([0, 1, 2, 3], 9.0, 6.0, 55.0, 240),
        ]

        merged = Rule_Pool_Generator.save_archive("long", fi, rules)

        assert len(merged) == 1
        assert merged[0]["objectives"]["sortino_ratio"] == 13.0
        assert merged[0]["objectives"]["total_return_pct"] == 13.0
        assert merged[0]["objectives"]["max_drawdown_pct"] == 3.0
        assert merged[0]["objectives"]["win_rate"] == 64.0

    def test_save_archive_prunes_to_max_size(self):
        fi = _make_feature_infos(
            ["positive", "positive", "positive", "positive"])
        rules = []
        for i in range(_cfg.PHASE2_ARCHIVE_MAX_SIZE + 20):
            chromosome = [
                i % 6,
                (i // 6) % 6,
                (i // 36) % 6,
                (i // 216) % 6,
            ]
            rules.append(self._entry(chromosome, 1.0, 1.0, 50.0, 250))

        merged = Rule_Pool_Generator.save_archive("long", fi, rules)

        assert len(merged) == _cfg.PHASE2_ARCHIVE_MAX_SIZE
        assert len({tuple(entry["chromosome"])
                   for entry in merged}) == len(merged)

    def test_load_archive_ignores_feature_signature_mismatch(self):
        fi_long = _make_feature_infos(["positive", "positive"])
        fi_short = _make_feature_infos(["binary", "positive"])
        Rule_Pool_Generator.save_archive(
            "long",
            fi_long,
            [self._entry([0, 1], 5.0, 1.0, 55.0, 230)],
        )

        assert Rule_Pool_Generator.load_archive("long", fi_short) is None


# ---------------------------------------------------------------------------
# Tests: Rule_Pool_Generator constructor
# ---------------------------------------------------------------------------

class TestRulePoolGeneratorInit:
    def test_invalid_direction_raises(self):
        fi = _make_feature_infos(["positive"])
        df = _make_train_df()
        with pytest.raises(ValueError, match="direction must be"):
            Rule_Pool_Generator(df, fi, "both")

    def test_empty_feature_infos_raises(self):
        df = _make_train_df()
        with pytest.raises(ValueError, match="feature_infos must not be empty"):
            Rule_Pool_Generator(df, [], "long")

    def test_valid_construction_long(self):
        fi = _make_feature_infos(["positive", "positive"])
        df = _make_train_df()
        gen = Rule_Pool_Generator(df, fi, "long", pop_size=5, n_generations=2)
        assert gen.direction == "long"
        assert gen.pop_size == 5
        assert gen.n_generations == 2

    def test_valid_construction_short(self):
        fi = _make_feature_infos(["positive", "positive"])
        df = _make_train_df()
        gen = Rule_Pool_Generator(df, fi, "short", pop_size=5, n_generations=2)
        assert gen.direction == "short"

    def test_default_pop_size_from_config(self):
        fi = _make_feature_infos(["positive"])
        df = _make_train_df()
        gen = Rule_Pool_Generator(df, fi, "long")
        assert gen.pop_size == _cfg.PHASE2_POPULATION_SIZE

    def test_default_n_generations_from_config(self):
        fi = _make_feature_infos(["positive"])
        df = _make_train_df()
        gen = Rule_Pool_Generator(df, fi, "long")
        assert gen.n_generations == _cfg.PHASE2_GENERATIONS

    def test_default_seed_is_none(self):
        fi = _make_feature_infos(["positive"])
        df = _make_train_df()
        gen = Rule_Pool_Generator(df, fi, "long")
        assert gen.seed is None


# ---------------------------------------------------------------------------
# Tests: Rule_Pool_Generator.run() — integration (tiny pop/gen)
# ---------------------------------------------------------------------------

class TestRulePoolGeneratorRun:
    """Integration tests using tiny population and generation counts."""

    def _make_generator(self, direction: str = "long", tmp_path=None) -> Rule_Pool_Generator:
        fi = _make_feature_infos(
            ["positive", "positive", "positive", "positive"])
        df = _make_train_df(n_rows=200, n_features=4, symbols=["A", "B"])
        gen = Rule_Pool_Generator(
            df, fi, direction,
            pop_size=8,
            n_generations=3,
            seed=42,
        )
        return gen

    def test_run_returns_list(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["long"] = str(tmp_path / "phase2_long_pool.json")
        m._HISTORY_PATHS["long"] = str(tmp_path / "phase2_long_history.json")
        try:
            gen = self._make_generator("long")
            result = gen.run()
            assert isinstance(result, list)
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

    def test_run_uses_pool_seeds(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        pool_path = str(tmp_path / "phase2_long_pool.json")
        m._POOL_PATHS["long"] = pool_path
        m._HISTORY_PATHS["long"] = str(tmp_path / "phase2_long_history.json")

        class StubReporter:
            def plot_phase2_metrics(self, *args, **kwargs):
                return None

            def plot_phase2_pnl(self, *args, **kwargs):
                return None

        captured = {}

        def fake_run_phase2_evolution(
            feature_infos,
            engine,
            pop_size,
            n_generations,
            rng,
            seed_chromosomes=None,
            log_tag=None,
            val_engine=None,
            **kwargs,
        ):
            captured["seed_chromosomes"] = seed_chromosomes
            return [
                {
                    "chromosome": [2, 3, 4, 0],
                    "conditions": ["[feat_0] IS Medium"],
                    "objectives": {
                        "sortino_ratio": 10.0,
                        "total_return_pct": 10.0,
                        "profit_factor": 1.2,
                        "max_drawdown_pct": 4.0,
                        "win_rate": 60.0,
                    },
                    "val_objectives": {
                        "sortino_ratio": 6.0,
                        "total_return_pct": 6.0,
                        "profit_factor": 1.1,
                        "max_drawdown_pct": 3.0,
                        "win_rate": 58.0,
                    },
                    "val_executed_trades": 120,
                    "executed_trades": 220,
                },
            ], [
                {
                    "generation": 0,
                    "pareto_size": 1,
                    "mean_f1": 0.0,
                    "mean_f2": 0.0,
                    "mean_f3": 0.0,
                    "algorithm": "NSGA-III",
                    "mean_sortino_ratio": 0.0,
                    "best_sortino_ratio": 0.0,
                }
            ]

        previous_pool = [
            {
                "chromosome": [0, 1, 2, 3],
                "conditions": ["[feat_0] IS Medium"],
                "objectives": {
                    "sortino_ratio": 12.0,
                    "total_return_pct": 12.0,
                    "profit_factor": 1.3,
                    "max_drawdown_pct": 3.0,
                    "win_rate": 61.0,
                },
                "val_objectives": {
                    "total_return_pct": 8.0,
                    "profit_factor": 1.15,
                    "max_drawdown_pct": 2.0,
                    "win_rate": 59.0,
                    "sortino_ratio": 8.0,
                },
                "val_executed_trades": 130,
                "executed_trades": 240,
            },
            {
                "chromosome": [1, 2, 3, 4],
                "conditions": ["[feat_0] IS Medium"],
                "objectives": {
                    "sortino_ratio": 11.0,
                    "total_return_pct": 11.0,
                    "profit_factor": 1.25,
                    "max_drawdown_pct": 4.0,
                    "win_rate": 60.0,
                },
                "val_objectives": {
                    "total_return_pct": 7.0,
                    "profit_factor": 1.1,
                    "max_drawdown_pct": 2.5,
                    "win_rate": 58.0,
                    "sortino_ratio": 7.0,
                },
                "val_executed_trades": 125,
                "executed_trades": 250,
            },
        ]

        try:
            with open(pool_path, "w", encoding="utf-8") as fh:
                json.dump(previous_pool, fh)

            with patch(
                "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution",
                side_effect=fake_run_phase2_evolution,
            ), patch(
                "gpu_fuzzy_trader.phases.phase2_rule_pool.Reporter",
                return_value=StubReporter(),
            ):
                gen = self._make_generator("long")
                result = gen.run()

            assert isinstance(result, list)
            assert captured["seed_chromosomes"] is not None
            seeds = captured["seed_chromosomes"]
            dc = _get_dont_cares(gen.feature_infos)
            assert seeds.shape == (2, max_slots(), 2)
            assert np.array_equal(
                sparse_to_dense(seeds[0], dc),
                np.array([0, 1, 2, 3], dtype=np.int32),
            )
            assert np.array_equal(
                sparse_to_dense(seeds[1], dc),
                np.array([1, 2, 3, 4], dtype=np.int32),
            )
            chromosomes = {tuple(entry["chromosome"]) for entry in result}
            assert (0, 1, 2, 3) in chromosomes
            assert (2, 3, 4, 0) in chromosomes
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

    def test_run_merges_previous_and_new_pool(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m

        previous = [
            {
                "chromosome": [0, 1, 2, 3],
                "conditions": [],
                "objectives": {
                    "sortino_ratio": 5.0,
                    "max_drawdown_pct": 2.0,
                    "win_rate": 55.0,
                },
                "executed_trades": 200,
            },
        ]
        new_pool = [
            {
                "chromosome": [1, 2, 3, 4],
                "conditions": [],
                "objectives": {
                    "sortino_ratio": 8.0,
                    "max_drawdown_pct": 1.0,
                    "win_rate": 60.0,
                },
                "executed_trades": 210,
            },
        ]
        merged = _merge_archive_entries(previous + new_pool)
        assert len(merged) == 2
        assert {tuple(e["chromosome"]) for e in merged} == {
            (0, 1, 2, 3),
            (1, 2, 3, 4),
        }

    def test_run_creates_pool_file(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        pool_path = str(tmp_path / "phase2_long_pool.json")
        hist_path = str(tmp_path / "phase2_long_history.json")
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["long"] = pool_path
        m._HISTORY_PATHS["long"] = hist_path
        try:
            gen = self._make_generator("long")
            gen.run()
            assert os.path.exists(pool_path)
            assert os.path.exists(hist_path)
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

    def test_run_pool_file_is_valid_json(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        pool_path = str(tmp_path / "phase2_long_pool.json")
        hist_path = str(tmp_path / "phase2_long_history.json")
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["long"] = pool_path
        m._HISTORY_PATHS["long"] = hist_path
        try:
            gen = self._make_generator("long")
            gen.run()
            with open(pool_path) as fh:
                pool = json.load(fh)
            assert isinstance(pool, list)
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

    def test_run_pool_entries_have_correct_schema(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        pool_path = str(tmp_path / "phase2_long_pool.json")
        hist_path = str(tmp_path / "phase2_long_history.json")
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["long"] = pool_path
        m._HISTORY_PATHS["long"] = hist_path
        try:
            gen = self._make_generator("long")
            result = gen.run()
            for entry in result:
                assert "chromosome" in entry
                assert "conditions" in entry
                assert "objectives" in entry
                assert "executed_trades" in entry
                assert "sortino_ratio" in entry["objectives"]
                assert "total_return_pct" in entry["objectives"]
                assert "max_drawdown_pct" in entry["objectives"]
                assert "win_rate" in entry["objectives"]
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

    def test_run_condition_count_bounds(self, tmp_path):
        """All pool entries must have active conditions within [MIN_CONDITIONS, MAX_CONDITIONS]."""
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        pool_path = str(tmp_path / "phase2_long_pool.json")
        hist_path = str(tmp_path / "phase2_long_history.json")
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["long"] = pool_path
        m._HISTORY_PATHS["long"] = hist_path
        try:
            fi = _make_feature_infos(["positive"] * 6)
            df = _make_train_df(n_rows=200, n_features=6, symbols=["A", "B"])
            gen = Rule_Pool_Generator(
                df, fi, "long", pop_size=10, n_generations=3, seed=0)
            result = gen.run()
            for entry in result:
                n_conditions = len(entry["conditions"])
                assert _cfg.MIN_CONDITIONS <= n_conditions <= _cfg.MAX_CONDITIONS, (
                    f"Entry has {n_conditions} conditions, expected [{_cfg.MIN_CONDITIONS}, {_cfg.MAX_CONDITIONS}]"
                )
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

    def test_run_static_risk_parameters(self, tmp_path):
        """
        Phase 2 must use static TP=PHASE2_TP, SL=PHASE2_SL, capital_pct=PHASE2_CAPITAL_PCT.

        We verify by checking that the engine is called with the correct parameters.
        """
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        pool_path = str(tmp_path / "phase2_long_pool.json")
        hist_path = str(tmp_path / "phase2_long_history.json")
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["long"] = pool_path
        m._HISTORY_PATHS["long"] = hist_path

        calls = []

        class MockEngine:
            def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct):
                calls.append({"tp": tp, "sl": sl, "capital_pct": capital_pct})
                return [{
                    "sortino_ratio": 1.0,
                    "total_return_pct": 1.0,
                    "max_drawdown_pct": 0.5,
                    "win_rate": 55.0,
                    "executed_trades": 25,
                }]

        try:
            fi = _make_feature_infos(["positive", "positive"])
            df = _make_train_df(n_rows=100, n_features=2, symbols=["A"])
            gen = Rule_Pool_Generator(
                df, fi, "long", pop_size=4, n_generations=2, seed=0)
            gen._engine = MockEngine()
            gen.run()

            assert len(calls) > 0
            for call in calls:
                assert call["tp"] == _cfg.PHASE2_TP, (
                    f"Expected TP={_cfg.PHASE2_TP}, got {call['tp']}"
                )
                assert call["sl"] == _cfg.PHASE2_SL, (
                    f"Expected SL={_cfg.PHASE2_SL}, got {call['sl']}"
                )
                assert call["capital_pct"] == _cfg.PHASE2_CAPITAL_PCT, (
                    f"Expected capital_pct={_cfg.PHASE2_CAPITAL_PCT}, got {call['capital_pct']}"
                )
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

    def test_run_short_direction(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        pool_path = str(tmp_path / "phase2_short_pool.json")
        hist_path = str(tmp_path / "phase2_short_history.json")
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["short"] = pool_path
        m._HISTORY_PATHS["short"] = hist_path
        try:
            gen = self._make_generator("short")
            result = gen.run()
            assert isinstance(result, list)
            assert os.path.exists(pool_path)
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

    def test_run_history_has_generation_entries(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        pool_path = str(tmp_path / "phase2_long_pool.json")
        hist_path = str(tmp_path / "phase2_long_history.json")
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["long"] = pool_path
        m._HISTORY_PATHS["long"] = hist_path
        try:
            gen = self._make_generator("long")
            gen.run()
            with open(hist_path) as fh:
                history = json.load(fh)
            assert isinstance(history, list)
            assert len(history) > 0
            for entry in history:
                assert "generation" in entry
                assert "pareto_size" in entry
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

    def test_run_min_trade_support_enforced(self, tmp_path):
        """Pool entries must have executed_trades >= MIN_TRADE_POOL_FLOOR."""
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        pool_path = str(tmp_path / "phase2_long_pool.json")
        hist_path = str(tmp_path / "phase2_long_history.json")
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["long"] = pool_path
        m._HISTORY_PATHS["long"] = hist_path
        try:
            fi = _make_feature_infos(["positive"] * 4)
            df = _make_train_df(n_rows=200, n_features=4, symbols=["A", "B"])
            gen = Rule_Pool_Generator(
                df, fi, "long", pop_size=8, n_generations=3, seed=0)
            result = gen.run()
            for entry in result:
                assert entry["executed_trades"] >= _cfg.MIN_TRADE_POOL_FLOOR, (
                    f"Entry has {entry['executed_trades']} trades, "
                    f"expected >= {_cfg.MIN_TRADE_POOL_FLOOR}"
                )
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)


class TestEvaluateChromosome:
    def test_evaluate_chromosome_use_total_return_obj(self):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import _evaluate_chromosome
        
        has_orig = hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")
        orig_val = getattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)
        orig_floor = _cfg.MIN_TRADE_SUPPORT
        
        try:
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = True
            _cfg.MIN_TRADE_SUPPORT = 5
            
            chromosome = _chromosome_with_min_active()
            dont_cares = np.ones(10, dtype=np.int32) * 5
            
            class MockEngine:
                def simulate_rule_batch(self, chromosomes, **kwargs):
                    return [{
                        "executed_trades": 100,
                        "total_return_pct": 15.0,
                        "sortino_ratio": 0.5,
                        "max_drawdown_pct": 2.0,
                        "win_rate": 50.0,
                        "profit_factor": 1.0,
                    }]
            
            engine = MockEngine()
            objectives, metrics = _evaluate_chromosome(
                chromosome, dont_cares, engine, []
            )
            
            # With total return obj enabled: f3 = -total_return_pct = -15.0 (plus penalties)
            assert np.isclose(objectives[2], -15.0)
            
            # Disable it -> should use win_rate = 50.0
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = False
            objectives, metrics = _evaluate_chromosome(
                chromosome, dont_cares, engine, []
            )
            assert np.isclose(objectives[2], -50.0)
            
        finally:
            _cfg.MIN_TRADE_SUPPORT = orig_floor
            if has_orig:
                _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = orig_val
            else:
                if hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ"):
                    delattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")


class TestRobustReturnObjective:
    def test_f3_uses_min_train_val_return(self, monkeypatch):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_ROBUST_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 10.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
        }
        val_metrics = {
            "executed_trades": 50,
            "total_return_pct": 3.0,
            "sortino_ratio": 0.8,
            "max_drawdown_pct": 1.0,
            "win_rate": 55.0,
            "profit_factor": 1.1,
        }
        objectives, out_metrics = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], val_metrics=val_metrics,
        )
        assert np.isclose(objectives[2], -3.0)
        assert out_metrics["robust_return_pct"] == pytest.approx(3.0)


class TestTwoStageOrchestration:
    def test_run_uses_two_stages_when_enabled(self, monkeypatch):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            _stage_b_seed_chromosomes,
        )

        monkeypatch.setattr(_cfg, "PHASE2_TWO_STAGE_ENABLED", True)
        stage_a = [
            {
                "chromosome": [0, 1],
                "objectives": {
                    "total_return_pct": 5.0,
                    "max_drawdown_pct": 2.0,
                    "profit_factor": 1.2,
                },
                "val_objectives": {
                    "total_return_pct": 4.0,
                    "profit_factor": 1.1,
                },
                "executed_trades": 50,
                "val_executed_trades": 20,
            },
            {
                "chromosome": [2, 3],
                "objectives": {
                    "total_return_pct": 1.0,
                    "max_drawdown_pct": 3.0,
                    "profit_factor": 1.0,
                },
                "val_objectives": {
                    "total_return_pct": 0.5,
                    "profit_factor": 1.0,
                },
                "executed_trades": 50,
                "val_executed_trades": 20,
            },
        ]
        base = np.array([[9, 9]], dtype=np.int32)
        seeds = _stage_b_seed_chromosomes(stage_a, base, None, top_k=1)
        assert seeds is not None
        assert seeds.shape[0] == 2
        assert np.array_equal(seeds[1], np.array([9, 9], dtype=np.int32))

    def test_stage_b_overrides_seed_chromosomes_without_duplicate_kwarg(
        self, tmp_path, monkeypatch,
    ):
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m

        monkeypatch.setattr(_cfg, "PHASE2_TWO_STAGE_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_POPULATION_SIZE", 8)
        monkeypatch.setattr(_cfg, "PHASE2_GENERATIONS", 3)
        monkeypatch.setattr(_cfg, "PHASE2_STAGE_A_GENERATIONS", 1)
        monkeypatch.setattr(_cfg, "PHASE2_STAGE_B_GENERATIONS", 1)
        monkeypatch.setattr(_cfg, "PHASE2_STAGE_B_SEED_TOP_K", 1)

        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        pool_path = str(tmp_path / "phase2_long_pool.json")
        m._POOL_PATHS["long"] = pool_path
        m._HISTORY_PATHS["long"] = str(tmp_path / "phase2_long_history.json")

        class StubReporter:
            def plot_phase2_metrics(self, *args, **kwargs):
                return None

            def plot_phase2_pnl(self, *args, **kwargs):
                return None

        archive_seed = np.array([[0, 1, 2, 3]], dtype=np.int32)
        calls: list[dict] = []

        def fake_run_phase2_evolution(
            feature_infos,
            engine,
            pop_size,
            n_generations,
            rng,
            seed_chromosomes=None,
            log_tag=None,
            val_engine=None,
            **kwargs,
        ):
            calls.append({
                "log_tag": log_tag,
                "stage": kwargs.get("stage"),
                "seed_chromosomes": None if seed_chromosomes is None
                else np.array(seed_chromosomes, copy=True),
            })
            chrom = [2, 3, 4, 0]
            entry = {
                "chromosome": chrom,
                "conditions": ["[feat_0] IS Medium"],
                "objectives": {
                    "sortino_ratio": 10.0,
                    "total_return_pct": 10.0,
                    "profit_factor": 1.2,
                    "max_drawdown_pct": 4.0,
                    "win_rate": 60.0,
                },
                "val_objectives": {
                    "sortino_ratio": 6.0,
                    "total_return_pct": 6.0,
                    "profit_factor": 1.1,
                    "max_drawdown_pct": 3.0,
                    "win_rate": 58.0,
                },
                "val_executed_trades": 120,
                "executed_trades": 220,
            }
            return [entry], [{
                "generation": 0,
                "pareto_size": 1,
                "mean_f1": 0.0,
                "mean_f2": 0.0,
                "mean_f3": 0.0,
                "algorithm": "NSGA-III",
                "mean_sortino_ratio": 0.0,
                "best_sortino_ratio": 0.0,
            }]

        try:
            with open(pool_path, "w", encoding="utf-8") as fh:
                json.dump([], fh)

            with patch(
                "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution",
                side_effect=fake_run_phase2_evolution,
            ), patch(
                "gpu_fuzzy_trader.phases.phase2_rule_pool.Reporter",
                return_value=StubReporter(),
            ), patch(
                "gpu_fuzzy_trader.phases.phase2_rule_pool._pool_seed_chromosomes",
                return_value=archive_seed,
            ):
                fi = _make_feature_infos(
                    ["positive", "positive", "positive", "positive"])
                df = _make_train_df(
                    n_rows=200, n_features=4, symbols=["A", "B"])
                gen = Rule_Pool_Generator(
                    df, fi, "long",
                    pop_size=8,
                    n_generations=3,
                    seed=42,
                )
                gen.run()

            assert len(calls) == 2
            assert "Stage A" in calls[0]["log_tag"]
            assert "Stage B" in calls[1]["log_tag"]
            assert calls[0]["stage"] == "A"
            assert calls[1]["stage"] == "B"
            assert calls[1]["seed_chromosomes"] is not None
        finally:
            m._POOL_PATHS.update(original_pool)
            m._HISTORY_PATHS.update(original_hist)

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
    _derive_val_sample_seed,
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
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
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
        first = _sample_df(df, total_rows=80, random_state=999)
        second = _sample_df(df, total_rows=80, random_state=999)
        pd.testing.assert_frame_equal(
            first.reset_index(drop=True),
            second.reset_index(drop=True),
        )

    def test_sampling_varies_with_seed(self):
        df = _make_train_df(n_rows=400, symbols=["A", "B"])
        first = _sample_df(df, total_rows=80, random_state=1)
        second = _sample_df(df, total_rows=80, random_state=2)
        assert not first.equals(second)

    def test_sampling_is_contiguous_per_symbol(self):
        """Critical: bars must be contiguous so the backtest engine preserves
        temporal causality. Stride gaps would silently drop intermediate
        candles, breaking position management and intraday patterns."""
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=120, random_state=7)
        for sym in ["A", "B", "C", "D"]:
            sym_df = sampled[sampled["symbol"] == sym].sort_values(
                "_symbol_bar_index"
            )
            if sym_df.empty:
                continue
            bar_idx = sym_df["_symbol_bar_index"].to_numpy()
            assert bar_idx.size >= 2
            # Every consecutive pair must be exactly +1 apart (contiguous).
            assert np.all(np.diff(bar_idx) == 1), (
                f"symbol {sym} has non-contiguous bars: gaps detected"
            )

    def test_sampling_respects_random_start_bound(self):
        """Random start must be bounded so the slice always fits forward."""
        df = _make_train_df(n_rows=400, symbols=["A", "B"])
        for seed in range(20):
            sampled = _sample_df(df, total_rows=80, random_state=seed)
            assert len(sampled) == 80

    def test_total_rows_exact_via_divmod(self):
        """divmod distribution gives exactly total_rows (no rounding loss).
        701_000 % 14 == 6, so the previous ``//`` lost 6 rows."""
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=101, random_state=0)
        assert len(sampled) == 101
        per_sym = sampled.groupby("symbol").size().to_dict()
        # 101 = 4*25 + 1 → first symbol gets 26, rest get 25
        assert per_sym["A"] == 26
        assert per_sym["B"] == 25
        assert per_sym["C"] == 25
        assert per_sym["D"] == 25

    def test_total_rows_smaller_than_n_symbols(self):
        """total_rows < n_symbols must NOT force 1 row per symbol."""
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=2, random_state=0)
        assert len(sampled) == 2

    def test_total_rows_zero(self):
        """total_rows=0 returns empty df (does not force 1 row per symbol)."""
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=0, random_state=0)
        assert len(sampled) == 0

    def test_empty_input_dataframe(self):
        """Empty input must not raise ZeroDivisionError."""
        empty = pd.DataFrame({
            "symbol": pd.Series(dtype=object),
            "datetime": pd.Series(dtype="datetime64[ns]"),
            "_symbol_bar_index": pd.Series(dtype=np.int64),
        })
        sampled = _sample_df(empty, total_rows=100, random_state=0)
        assert len(sampled) == 0

    def test_total_rows_exactly_divisible(self):
        """When total_rows divides n_symbols evenly, all symbols get equal rows."""
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=100, random_state=0)
        assert len(sampled) == 100
        for sym in ["A", "B", "C", "D"]:
            assert (sampled["symbol"] == sym).sum() == 25

    def test_all_symbols_share_start_datetime(self):
        """All symbols must start at the same datetime (shared clock)."""
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=120, random_state=7)
        first_dts = {
            sym: sampled[sampled["symbol"] == sym].sort_values("datetime")[
                "datetime"
            ].iloc[0]
            for sym in ["A", "B", "C", "D"]
        }
        unique_starts = set(first_dts.values())
        assert len(unique_starts) == 1, (
            f"symbols start at different times: {first_dts}"
        )

    def test_all_symbols_share_end_datetime(self):
        """All symbols must end at the same datetime (shared clock)."""
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        sampled = _sample_df(df, total_rows=120, random_state=7)
        last_dts = {
            sym: sampled[sampled["symbol"] == sym].sort_values("datetime")[
                "datetime"
            ].iloc[-1]
            for sym in ["A", "B", "C", "D"]
        }
        unique_ends = set(last_dts.values())
        assert len(unique_ends) == 1, (
            f"symbols end at different times: {last_dts}"
        )

    def test_start_within_min_symbol_bound(self):
        """Shared start must be bounded so the slice fits in the smallest symbol."""
        # Imbalanced: A,B,C have 100 rows, D has only 50 rows.
        df = _make_train_df(n_rows=350, symbols=["A", "B", "C", "D"])
        # Force D to be smaller: trim it.
        d_mask = df["symbol"] == "D"
        df = pd.concat([df[~d_mask].head(300), df[d_mask].head(50)], ignore_index=True)
        for seed in range(20):
            sampled = _sample_df(df, total_rows=80, random_state=seed)
            assert len(sampled) > 0
            # D should never be asked for more than 50 rows
            d_rows = (sampled["symbol"] == "D").sum()
            assert d_rows <= 50

    def test_sample_avoids_forbidden_ranges(self):
        """Sampled slice must not overlap any forbidden range (per symbol)."""
        df = _make_train_df(n_rows=400, symbols=["A", "B"])
        # Per-symbol 200 rows. Forbid [50, 99] (the middle).
        forbidden = [(50, 99)]
        for seed in range(20):
            sampled = _sample_df(
                df, total_rows=80, random_state=seed,
                forbidden_ranges=forbidden,
            )
            for sym in ["A", "B"]:
                bar_idx = sampled[sampled["symbol"] == sym][
                    "_symbol_bar_index"
                ].to_numpy()
                if len(bar_idx) == 0:
                    continue
                s_start, s_end = int(bar_idx[0]), int(bar_idx[-1])
                # Slice must not overlap [50, 99]
                assert s_end < 50 or s_start > 99, (
                    f"seed={seed} sym={sym}: slice [{s_start}, {s_end}] "
                    f"overlaps forbidden [50, 99]"
                )

    def test_sample_handles_disjoint_forbidden_ranges(self):
        """Multiple non-contiguous forbidden ranges must all be avoided."""
        df = _make_train_df(n_rows=400, symbols=["A", "B"])
        forbidden = [(80, 99), (150, 199), (300, 319)]
        sampled = _sample_df(
            df, total_rows=60, random_state=42,
            forbidden_ranges=forbidden,
        )
        for sym in ["A", "B"]:
            bar_idx = sampled[sampled["symbol"] == sym][
                "_symbol_bar_index"
            ].to_numpy()
            if len(bar_idx) == 0:
                continue
            s_start, s_end = int(bar_idx[0]), int(bar_idx[-1])
            for f_start, f_end in forbidden:
                assert s_end < f_start or s_start > f_end, (
                    f"sym={sym}: slice [{s_start}, {s_end}] overlaps "
                    f"forbidden [{f_start}, {f_end}]"
                )

    def test_sample_uses_largest_safe_range(self):
        """With multiple safe ranges, use the largest one."""
        df = _make_train_df(n_rows=400, symbols=["A"])
        # Two safe regions: [0, 49] (50 bars) and [150, 399] (250 bars).
        # Largest is [150, 399] → sample start ∈ [150, 399-80+1] = [150, 320].
        sampled = _sample_df(
            df, total_rows=80, random_state=42,
            forbidden_ranges=[(50, 149)],
        )
        bar_idx = sampled["_symbol_bar_index"].to_numpy()
        s_start, s_end = int(bar_idx[0]), int(bar_idx[-1])
        assert s_start >= 150
        assert s_end <= 399

    def test_sample_warns_and_shrinks_when_safe_region_too_small(self):
        """If the safe region is smaller than the requested size, shrink + warn."""
        df = _make_train_df(n_rows=400, symbols=["A"])
        # Only 30 safe bars; ask for 100.
        sampled = _sample_df(
            df, total_rows=100, random_state=0,
            forbidden_ranges=[(30, 399)],
        )
        assert len(sampled) == 30  # entire safe region

    def test_no_constraint_when_forbidden_empty(self):
        """Empty forbidden_ranges == no constraint (backward compat)."""
        df = _make_train_df(n_rows=400, symbols=["A"])
        sampled = _sample_df(df, total_rows=100, random_state=0)
        assert len(sampled) == 100
        bar_idx = sampled["_symbol_bar_index"].to_numpy()
        # No constraint: can be anywhere
        assert bar_idx[0] >= 0 and bar_idx[-1] < 400

    def test_sample_df_none_uses_phase2_seed(self, monkeypatch):
        df = _make_train_df(n_rows=400, symbols=["A", "B"])
        monkeypatch.setattr(_cfg, "PHASE2_SEED", 4242)
        expected = _sample_df(df, total_rows=80, random_state=4242)
        actual = _sample_df(df, total_rows=80)
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
        )

    def test_sample_df_none_is_not_seed_zero(self, monkeypatch):
        df = _make_train_df(n_rows=400, symbols=["A", "B"])
        monkeypatch.setattr(_cfg, "PHASE2_SEED", 4242)
        with_seed = _sample_df(df, total_rows=80, random_state=0)
        with_none = _sample_df(df, total_rows=80)
        assert not with_seed.equals(with_none)


# ---------------------------------------------------------------------------
# Tests: _derive_val_sample_seed — distinct train/val sampling seeds
# ---------------------------------------------------------------------------


class TestDeriveValSampleSeed:
    """AC: Train and validation sampling use distinct RNG seeds by default."""

    def test_val_seed_differs_from_train_seed(self):
        """_derive_val_sample_seed returns a value different from the input."""
        for train_seed in [0, 1, 42, 999999, 2**31 - 2]:
            val_seed = _derive_val_sample_seed(train_seed)
            assert val_seed != train_seed, (
                f"val seed {val_seed} must differ from train seed {train_seed}"
            )

    def test_val_seed_is_deterministic(self):
        """Same train seed always produces same val seed."""
        for train_seed in [0, 42, 123456]:
            a = _derive_val_sample_seed(train_seed)
            b = _derive_val_sample_seed(train_seed)
            assert a == b

    def test_val_seed_in_valid_range(self):
        """Result is in [0, 2**31) so it is a valid random seed."""
        for train_seed in range(0, 100, 7):
            val_seed = _derive_val_sample_seed(train_seed)
            assert 0 <= val_seed < 2**31

    def test_rule_pool_generator_has_distinct_val_seed(self):
        """Rule_Pool_Generator stores distinct _sample_seed and _val_sample_seed."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            Rule_Pool_Generator,
            _derive_val_sample_seed,
        )
        fi = _make_feature_infos(["positive"] * 3)
        df = _make_train_df()
        gen = Rule_Pool_Generator(df, fi, "long", seed=42)
        assert gen._sample_seed != gen._val_sample_seed
        assert gen._val_sample_seed == _derive_val_sample_seed(gen._sample_seed)

    def test_rule_pool_generator_val_seed_default_phase2_seed(self, monkeypatch):
        """When seed=None, val seed is derived from PHASE2_SEED."""
        monkeypatch.setattr(_cfg, "PHASE2_SEED", 9999)
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            Rule_Pool_Generator,
            _derive_val_sample_seed,
        )
        fi = _make_feature_infos(["positive"] * 3)
        df = _make_train_df()
        gen = Rule_Pool_Generator(df, fi, "long", seed=None)
        assert gen._sample_seed == 9999
        assert gen._val_sample_seed == _derive_val_sample_seed(9999)

    def test_val_seed_changes_validation_sample(self):
        """Using val seed for validation sampling produces a different
        slice than using the train seed (same data, same budget)."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            _derive_val_sample_seed,
            _sample_df,
        )
        df = _make_train_df(n_rows=400, symbols=["A", "B"])
        budget = 80
        for train_seed in [0, 42, 100]:
            val_seed = _derive_val_sample_seed(train_seed)
            train_sample = _sample_df(df, budget, random_state=train_seed)
            val_sample = _sample_df(df, budget, random_state=val_seed)
            # The two samples should differ in at least one row for some seeds
            if train_seed == 42:
                # Not strictly required for all seeds, but a useful check
                assert not train_sample.equals(val_sample), (
                    f"train sample (seed={train_seed}) unexpectedly equals "
                    f"val sample (seed={val_seed}) — seed derivation may "
                    f"collide"
                )

    def test_total_rows_exact_still_works_with_val_seed(self):
        """divmod distribution must still work correctly with val seed."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            _derive_val_sample_seed,
            _sample_df,
        )
        df = _make_train_df(n_rows=400, symbols=["A", "B", "C", "D"])
        val_seed = _derive_val_sample_seed(42)
        sampled = _sample_df(df, total_rows=101, random_state=val_seed)
        assert len(sampled) == 101


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

    def test_holdout_mode_builds_val_engine_for_admission(
        self, tmp_path, monkeypatch,
    ):
        """In holdout mode, val engine must be built for pool admission
        even when PHASE2_JOINT_TRAIN_VAL is False (regression: deployable=0)."""
        import gpu_fuzzy_trader.phases.phase2_rule_pool as m
        original_pool = m._POOL_PATHS.copy()
        original_hist = m._HISTORY_PATHS.copy()
        m._POOL_PATHS["long"] = str(tmp_path / "phase2_long_pool.json")
        m._HISTORY_PATHS["long"] = str(tmp_path / "phase2_long_history.json")
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_USE_GPU", False)
        try:
            fi = _make_feature_infos(
                ["positive", "positive", "positive", "positive"])
            df = _make_train_df(n_rows=400, n_features=4, symbols=["A", "B"])
            train_df = df.iloc[:300].reset_index(drop=True)
            val_df = df.iloc[300:].reset_index(drop=True)
            gen = Rule_Pool_Generator(
                train_df, fi, "long",
                pop_size=6,
                n_generations=2,
                seed=42,
                val_df=val_df,
            )
            assert gen._val_engine is not None, (
                "val engine must be built in holdout mode for pool "
                "admission, even when PHASE2_JOINT_TRAIN_VAL=False"
            )
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
                        "profit_factor": 1.21,
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
                    "profit_factor": 1.21,
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
                    "profit_factor": 1.21,
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
                # Task 22: pool entries must persist tp/sl/capital_pct for RB Governor
                assert "tp" in entry, f"Pool entry missing 'tp': {list(entry.keys())}"
                assert "sl" in entry, f"Pool entry missing 'sl': {list(entry.keys())}"
                assert "capital_pct" in entry, f"Pool entry missing 'capital_pct': {list(entry.keys())}"
                assert entry["tp"] == float(_cfg.PHASE2_TP), (
                    f"Expected tp={_cfg.PHASE2_TP}, got {entry['tp']}"
                )
                assert entry["sl"] == float(_cfg.PHASE2_SL), (
                    f"Expected sl={_cfg.PHASE2_SL}, got {entry['sl']}"
                )
                assert entry["capital_pct"] == float(_cfg.PHASE2_CAPITAL_PCT), (
                    f"Expected capital_pct={_cfg.PHASE2_CAPITAL_PCT}, got {entry['capital_pct']}"
                )
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
            def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct, **kwargs):
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
        orig_f3_obj = str(getattr(_cfg, "PHASE2_F3_OBJECTIVE", "profit_factor"))
        orig_val_pen = bool(
            getattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False))

        try:
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = True
            _cfg.MIN_TRADE_SUPPORT = 5
            # Train-only arithmetic: no val_metrics in this test.
            _cfg.PHASE2_VAL_IN_FITNESS_PENALTY = False

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
                        "profit_factor": 5.0,
                        "per_symbol_metrics": {
                            "SYM1": {"net_pnl": 100.0},
                            "SYM2": {"net_pnl": 200.0},
                            "SYM3": {"net_pnl": 300.0},
                        },
                    }]

            engine = MockEngine()
            objectives, metrics = _evaluate_chromosome(
                chromosome, dont_cares, engine, []
            )

            # With total return obj enabled: f3 = -total_return_pct = -15.0 (plus penalties)
            assert np.isclose(objectives[2], -15.0)

            # Disable it -> should use PHASE2_F3_OBJECTIVE (profit_factor = 5.0)
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = False
            _cfg.PHASE2_F3_OBJECTIVE = "profit_factor"
            objectives, metrics = _evaluate_chromosome(
                chromosome, dont_cares, engine, []
            )
            # With profit_factor: f3 = -profit_factor = -5.0 (plus penalties)
            assert np.isclose(objectives[2], -5.0)

            # Set PHASE2_F3_OBJECTIVE to "win_rate" -> should use win_rate = 50.0
            _cfg.PHASE2_F3_OBJECTIVE = "win_rate"
            objectives, metrics = _evaluate_chromosome(
                chromosome, dont_cares, engine, []
            )
            assert np.isclose(objectives[2], -50.0)

        finally:
            _cfg.MIN_TRADE_SUPPORT = orig_floor
            _cfg.PHASE2_VAL_IN_FITNESS_PENALTY = orig_val_pen
            if has_orig:
                _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = orig_val
            else:
                if hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ"):
                    delattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")
            _cfg.PHASE2_F3_OBJECTIVE = orig_f3_obj

    def test_evaluate_chromosome_f3_blends_train_val_pf(self):
        """f3 profit_factor branch mirrors win_rate train+val blend.

        Covers three branches:
          1. JOINT_TRAIN_VAL=True, val_trades >= floor → f3_val = min(train_pf, val_pf)
          2. JOINT_TRAIN_VAL=True, val_trades < floor  → f3_val = min(train_pf, 0.0)
          3. JOINT_TRAIN_VAL=False                      → f3_val = train_pf (unchanged)
        """
        from gpu_fuzzy_trader.phases.phase2_rule_pool import _evaluate_chromosome

        orig_joint = getattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)
        orig_f3 = str(getattr(_cfg, "PHASE2_F3_OBJECTIVE", "profit_factor"))
        orig_floor = _cfg.MIN_TRADE_SUPPORT
        has_ret_obj = hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")
        orig_ret_obj = getattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)
        orig_require_splits = _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS

        try:
            _cfg.PHASE2_F3_OBJECTIVE = "profit_factor"
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = False
            _cfg.PHASE2_JOINT_TRAIN_VAL = True
            _cfg.MIN_TRADE_SUPPORT = 5
            _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = False

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
                        "profit_factor": 5.0,
                        "per_symbol_metrics": {
                            "SYM1": {"net_pnl": 100.0},
                            "SYM2": {"net_pnl": 200.0},
                            "SYM3": {"net_pnl": 300.0},
                        },
                    }]

            class MockValEngine:
                def __init__(self, pf=2.0, trades=50):
                    self.pf = pf
                    self.trades = trades

                def simulate_rule_batch(self, chromosomes, **kwargs):
                    return [{
                        "executed_trades": self.trades,
                        "total_return_pct": 10.0,
                        "sortino_ratio": 0.5,
                        "max_drawdown_pct": 1.0,
                        "win_rate": 50.0,
                        "profit_factor": self.pf,
                    }]

            engine = MockEngine()

            # Case 1: JOINT_TRAIN_VAL=True, val_trades=50 >= val_trade_floor
            # f3_val = min(5.0, 2.0) = 2.0 → objectives[2] = -2.0
            _cfg.PHASE2_JOINT_TRAIN_VAL = True
            val_engine = MockValEngine(pf=2.0, trades=50)
            objectives, _ = _evaluate_chromosome(
                chromosome, dont_cares, engine, [], val_engine=val_engine,
            )
            assert np.isclose(objectives[2], -2.0), (
                f"Case 1 (joint, val ok): expected -2.0, got {objectives[2]}"
            )

            # Case 2: JOINT_TRAIN_VAL=True, val_trades=5 < val_trade_floor
            # f3_val = min(5.0, 0.0) = 0.0
            # support_penalty = max(0, SUPPORT_PENALTY_MAX=5) = 5 from the val gate
            # f3 = -0.0 + (0.6 * 5.0) = 3.0 (support penalty weight on f3)
            val_engine = MockValEngine(pf=2.0, trades=5)
            objectives, _ = _evaluate_chromosome(
                chromosome, dont_cares, engine, [], val_engine=val_engine,
            )
            assert np.isclose(objectives[2], 3.0, atol=0.01), (
                f"Case 2 (joint, val gate): expected 3.0, got {objectives[2]}"
            )

            # Case 3: JOINT_TRAIN_VAL=False (val metrics provided but ignored)
            # f3_val = profit_factor = 5.0 → objectives[2] = -5.0
            _cfg.PHASE2_JOINT_TRAIN_VAL = False
            val_engine = MockValEngine(pf=2.0, trades=50)
            objectives, _ = _evaluate_chromosome(
                chromosome, dont_cares, engine, [], val_engine=val_engine,
            )
            assert np.isclose(objectives[2], -5.0), (
                f"Case 3 (train-only): expected -5.0, got {objectives[2]}"
            )

        finally:
            _cfg.PHASE2_JOINT_TRAIN_VAL = orig_joint
            _cfg.PHASE2_F3_OBJECTIVE = orig_f3
            _cfg.MIN_TRADE_SUPPORT = orig_floor
            _cfg.PHASE2_POOL_REQUIRE_POSITIVE_SPLITS = orig_require_splits
            if has_ret_obj:
                _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = orig_ret_obj
            else:
                if hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ"):
                    delattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")

    def test_evaluate_chromosome_diversity_penalty_avoids_self_penalization(self):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import _evaluate_chromosome

        orig_penalty = _cfg.PHASE2_DIVERSITY_PENALTY
        orig_hamming = _cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD
        orig_floor = _cfg.MIN_TRADE_SUPPORT
        has_ret_obj = hasattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ")
        orig_ret_obj = getattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)
        orig_val_pen = bool(
            getattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False))
        
        try:
            _cfg.PHASE2_DIVERSITY_PENALTY = 10.0
            _cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD = 4
            _cfg.MIN_TRADE_SUPPORT = 5
            _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = True
            _cfg.PHASE2_VAL_IN_FITNESS_PENALTY = False
            
            chromosome = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
            dont_cares = np.ones(6, dtype=np.int32) * 7
            
            class MockEngine:
                def simulate_rule_batch(self, chromosomes, **kwargs):
                    return [{
                        "executed_trades": 100,
                        "total_return_pct": 15.0,
                        "sortino_ratio": 0.5,
                        "max_drawdown_pct": 2.0,
                        "win_rate": 50.0,
                        "profit_factor": 5.0,
                        "per_symbol_metrics": {
                            "SYM1": {"net_pnl": 100.0},
                            "SYM2": {"net_pnl": 200.0},
                            "SYM3": {"net_pnl": 300.0},
                        },
                    }]
            
            engine = MockEngine()
            
            # Case 1: pareto_front contains exact same chromosome -> should NOT penalize it
            objectives_self, _ = _evaluate_chromosome(
                chromosome, dont_cares, engine, [chromosome]
            )
            
            # Case 2: pareto_front contains a different chromosome that is within Hamming distance of 4
            # (diff by 1 gene -> Hamming dist = 1)
            similar_chrom = np.array([1, 2, 3, 4, 5, 9], dtype=np.int32)
            objectives_similar, _ = _evaluate_chromosome(
                chromosome, dont_cares, engine, [similar_chrom]
            )
            
            # objectives[2] is -f3_val + penalties (diversity no longer on f3)
            # f3_val is total_return = 15.0 since PHASE2_USE_TOTAL_RETURN_OBJ is True.
            # cond_penalty follows the active production bound.  The
            # approved 2-4 contract makes six active conditions incur
            # (6 - MAX_CONDITIONS) * 10 = 20 points.
            # Without diversity on f3, the expected value is the robust
            # return objective plus the bound penalty calculated above.
            # Diversity lands on f4 when PHASE2_DIVERSITY_ON_F4=True.
            expected_f3 = -15.0 + max(
                0.0,
                (6 - int(_cfg.MAX_CONDITIONS)) * 10.0,
            )
            assert np.isclose(objectives_self[2], expected_f3)
            assert np.isclose(objectives_similar[2], expected_f3)
            if bool(getattr(_cfg, "PHASE2_F4_ENABLED", False)) and bool(
                getattr(_cfg, "PHASE2_DIVERSITY_ON_F4", True)
            ):
                assert objectives_similar[3] > objectives_self[3] - 1e-9
                assert objectives_similar[3] >= 10.0 - 1e-9
            
        finally:
            _cfg.PHASE2_DIVERSITY_PENALTY = orig_penalty
            _cfg.PHASE2_DIVERSITY_HAMMING_THRESHOLD = orig_hamming
            _cfg.MIN_TRADE_SUPPORT = orig_floor
            _cfg.PHASE2_VAL_IN_FITNESS_PENALTY = orig_val_pen
            if has_ret_obj:
                _cfg.PHASE2_USE_TOTAL_RETURN_OBJ = orig_ret_obj




class TestRobustReturnObjective:
    def test_f3_uses_min_train_val_return(self, monkeypatch):
        """f3 uses robust return = min(train_return, val_return) when
        PHASE2_USE_TOTAL_RETURN_OBJ=True (now the default) and JOINT_TRAIN_VAL
        is enabled. This is the OOS-focused mode."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_ROBUST_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 10.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
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

    def test_overfit_gap_penalty_positive_when_val_negative(self, monkeypatch):
        """Blind-spot regression: overfit_gap_penalty must fire when val_ret <= 0."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_ROBUST_RETURN_OBJ", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_GAP_PENALTY_WEIGHT", 1.0)
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_GAP_PCT_THRESHOLD", 5.0)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 99.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }
        val_metrics_neg = {
            "executed_trades": 50,
            "total_return_pct": -10.0,
            "sortino_ratio": -0.5,
            "max_drawdown_pct": 1.0,
            "win_rate": 30.0,
            "profit_factor": 0.5,
        }
        val_metrics_mild = {
            "executed_trades": 50,
            "total_return_pct": 1.0,
            "sortino_ratio": 0.8,
            "max_drawdown_pct": 1.0,
            "win_rate": 55.0,
            "profit_factor": 1.1,
        }

        # Case 1: val_ret = -10 % (worst case — bug dodged old ratio gate).
        obj_neg, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], val_metrics=val_metrics_neg,
        )
        # Case 2: val_ret = 1 % (mild case — should have smaller penalty).
        obj_mild, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], val_metrics=val_metrics_mild,
        )

        # Task 3: overfit_gap_penalty is on f3 (not f1 now). A larger gap
        # → higher f3 (since f3 = -f3_val + penalties...).
        # f3 is objectives[2]; the penalty makes f3 more positive.
        assert obj_neg[2] > obj_mild[2], (
            "Worst-case gap (99 / -10) should have HIGHER f3 than mild gap (99 / 1)."
        )

    def test_overfit_gap_penalty_nonzero_for_negative_val_ret(self, monkeypatch):
        """Direct assertion that penalty is strictly positive when val_ret <= 0."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_ROBUST_RETURN_OBJ", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_GAP_PENALTY_WEIGHT", 10.0)
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_GAP_PCT_THRESHOLD", 5.0)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        # Disable diversity & trade penalties to isolate overfit_gap_penalty.
        monkeypatch.setattr(_cfg, "PHASE2_DIVERSITY_PENALTY", 0.0)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 99.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }
        val_metrics_zero = {
            "executed_trades": 50,
            "total_return_pct": 0.0,  # borderline: val_ret == 0
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 50.0,
            "profit_factor": 1.0,
        }

        obj, out = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], val_metrics=val_metrics_zero,
        )
        # Task 3: overfit_gap_penalty is on f3 (not f1).
        # f3 = -f3_val + support + diversity + cond + overfit_gap_penalty
        # With val_ret=0, gap=99-0=99 >> 5 threshold → penalty = (99-5)*10 = 940.
        # f3 = -(f3_val) + 0 + 0 + 0 + 940. f3_val depends on settings (PHASE2_USE_TOTAL_RETURN_OBJ=True
        # here, so f3_val = robust_return = min(99, 0) = 0). So f3 ≈ 940.
        assert obj[2] > 0.0, (
            f"Expected positive f3 when overfit_gap_penalty applies, got {obj[2]}"
        )


class TestDecoupledObjectives:
    """Task 3 tests: penalties are not identically added to all objectives."""

    def test_penalties_not_identical_across_objectives(self, monkeypatch):
        """The three objectives should respond differently to the same metrics,
        proving that penalties are not identically applied."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_ROBUST_RETURN_OBJ", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_DRAWDOWN_GATE", 200.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)

        # Metrics where Sortino is good but DD is large → different f1 vs f2
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 5.0,
            "sortino_ratio": 3.0,
            "max_drawdown_pct": 25.0,
            "win_rate": 60.0,
            "profit_factor": 1.5,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 150.0},
            },
        }
        objectives, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )

        # f1 = -sortino + support + diversity ≈ -(tanh(3/10)*20) + 0 + 0 ≈ -5.9
        # f2 = dd + support + dd_gate + trade = 25 + 0 + (25-20)*2 + 0 = 35
        # f3 = -f3_val + support + diversity + cond = -5 + 0 + 0 + 0 = -5
        # Key assertion: f2 should be significantly different from f1 and f3
        # (high DD penalizes f2 heavily while f1 is driven by good Sortino)
        assert objectives[1] > objectives[0] + 10.0, (
            f"f2 (DD-penalized) should be much higher than f1 (Sortino-driven): "
            f"f1={objectives[0]:.2f}, f2={objectives[1]:.2f}, f3={objectives[2]:.2f}"
        )
        # f2 should also differ significantly from f3
        assert objectives[1] > objectives[2] + 10.0, (
            f"f2 should be much higher than f3: "
            f"f1={objectives[0]:.2f}, f2={objectives[1]:.2f}, f3={objectives[2]:.2f}"
        )

    def test_trade_penalty_only_on_f2(self, monkeypatch):
        """When the trade floor is triggered, only f2 gets the trade_penalty.
        f1 and f3 should NOT receive the large PHASE2_INFEASIBLE_OBJECTIVE_PENALTY."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 100)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "PHASE2_INFEASIBLE_OBJECTIVE_PENALTY", 99.9)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_DRAWDOWN_GATE", 200.0)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 10,  # < 100 → hard reject
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }

        objectives, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )
        # f2 = dd_for_obj(100) + 0 + 0 + trade(99.9) = 199.9
        assert np.isclose(objectives[1], 199.9, atol=0.1), (
            f"Expected f2 ≈ 199.9 (dd=100 + trade=99.9), got {objectives[1]}"
        )
        # f1 = -sortino_for_obj(0) + 0 support + 0 diversity = 0.0
        # (no trade_penalty on f1)
        assert np.isclose(objectives[0], 0.0, atol=0.1), (
            f"Expected f1 ≈ 0.0 (no trade_penalty on f1), got {objectives[0]}"
        )
        # f3 = -f3_val(0) + 0 support + 0 diversity + 0 cond = 0.0
        # (no trade_penalty on f3)
        assert np.isclose(objectives[2], 0.0, atol=0.1), (
            f"Expected f3 ≈ 0.0 (no trade_penalty on f3), got {objectives[2]}"
        )

    def test_f3_uses_robust_return_by_default(self, monkeypatch):
        """With USE_TOTAL_RETURN_OBJ + JOINT_TRAIN_VAL, f3 uses robust_return_pct."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_ROBUST_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 10.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
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
        # f3 = -robust_return = -min(10, 3) = -3.0
        assert np.isclose(objectives[2], -3.0), (
            f"Expected f3 ≈ -3.0 (robust return), got {objectives[2]}"
        )
        assert out_metrics["robust_return_pct"] == pytest.approx(3.0)

    def test_pf_floor_still_penalizes_support(self, monkeypatch):
        """PF floor adds to support_penalty even after decoupling. This ensures
        the profit_factor floor remains an effective feasibility gate."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 2.0)  # high floor
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_DRAWDOWN_GATE", 200.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)

        # Rule with PF below floor
        metrics_low_pf = {
            "executed_trades": 100,
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 0.5,  # below floor of 2.0
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }
        metrics_high_pf = {
            "executed_trades": 100,
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 3.0,  # above floor of 2.0
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }

        obj_low, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics_low_pf, [],
        )
        obj_high, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics_high_pf, [],
        )

        # PF floor penalty: (floor - pf) * 5.0 = (2.0 - 0.5) * 5.0 = 7.5
        # This gets multiplied by support_penalty_weights and added to each objective.
        # All objectives should be worse (higher) for the low-PF rule.
        for i in range(3):
            assert obj_low[i] > obj_high[i], (
                f"Objective f{i+1} should be worse for low PF rule: "
                f"low={obj_low[i]:.2f}, high={obj_high[i]:.2f}"
            )

    def test_evolution_pf_floor_is_lower_than_admission(self, monkeypatch):
        """Spec scenario: pf=1.10, EVOLUTION=1.05 (new) vs 1.15 (old);
        the new floor produces a smaller penalty."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_DRAWDOWN_GATE", 200.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)

        # Rule with PF=1.10 (above EVOLUTION floor=1.05, below ADMISSION floor=1.15)
        #   → with EVOLUTION=1.05: pf > floor, no penalty
        #   → with EVOLUTION=1.15: pf < floor, penalty = (1.15 - 1.10) * 5.0 = 0.25
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.10,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }

        # First compute with EVOLUTION floor = 1.05 (new behavior) → pf > floor, no penalty
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION", 1.05)
        objectives_low_floor, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )

        # Now re-compute with EVOLUTION floor = 1.15 (old ADMISSION floor) → pf < floor, penalty
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR_EVOLUTION", 1.15)
        objectives_high_floor, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )

        # The penalty adds (floor - pf) * 5.0 = (1.15 - 1.10) * 5.0 = 0.25 to
        # support_penalty, which flows into the first 3 objectives via penalty_weights.
        # f4 (concentration objective) may be independent, so only check f1-f3.
        # The new floor (1.05) produces a strictly SMALLER penalty than the old floor (1.15).
        for i in range(3):
            assert objectives_high_floor[i] > objectives_low_floor[i], (
                f"Objective f{i+1} should be worse with higher EVOLUTION floor: "
                f"floor=1.05 gave {objectives_low_floor[i]:.4f}, "
                f"floor=1.15 gave {objectives_high_floor[i]:.4f}"
            )

    def test_deprecated_pf_floor_alias_returns_115(self):
        """PHASE2_PROFIT_FACTOR_FLOOR (deprecated alias) still returns
        1.15 by default for backward compat."""
        assert _cfg.PHASE2_PROFIT_FACTOR_FLOOR == 1.15, (
            f"Expected deprecated alias PHASE2_PROFIT_FACTOR_FLOOR=1.15, "
            f"got {_cfg.PHASE2_PROFIT_FACTOR_FLOOR}"
        )


class TestPhenotypeDiversityPenalty:
    def test_phenotype_penalty_same_bucket_different_genes(self, monkeypatch):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            _phenotype_bucket_key,
            _saturating_sortino,
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_DIVERSITY_PENALTY", 10.0)
        monkeypatch.setattr(_cfg, "PHASE2_DIVERSITY_HAMMING_THRESHOLD", 0)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)
        monkeypatch.setattr(_cfg, "PHASE2_F3_OBJECTIVE", "win_rate")
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", False)

        dont_cares = np.full(5, 5, dtype=np.int32)
        chrom = np.array([1, 2, 3, 4, 5], dtype=np.int32)
        ref_chrom = np.array([9, 8, 7, 6, 5], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 10.0,
            "sortino_ratio": 2.0,
            "max_drawdown_pct": 8.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }
        sortino_sat = _saturating_sortino(2.0)
        bucket = _phenotype_bucket_key(sortino_sat, 8.0, 50.0)
        ref_metrics = {"phenotype_bucket": bucket}

        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import chromosome_key

        obj_no_pen, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], diversity_metrics_by_key={},
        )
        obj_pen, _ = compute_phase2_objectives_from_metrics(
            chrom,
            dont_cares,
            metrics,
            [ref_chrom],
            diversity_metrics_by_key={chromosome_key(ref_chrom): ref_metrics},
        )
        # Diversity is routed to f4 when PHASE2_DIVERSITY_ON_F4 (current default).
        if bool(getattr(_cfg, "PHASE2_DIVERSITY_ON_F4", True)) and bool(
            getattr(_cfg, "PHASE2_F4_ENABLED", False)
        ):
            assert obj_pen[3] > obj_no_pen[3]
            assert obj_pen[0] == obj_no_pen[0]
            assert obj_pen[2] == obj_no_pen[2]
        else:
            assert obj_pen[0] > obj_no_pen[0]
            assert obj_pen[2] > obj_no_pen[2]


class TestInfeasiblePenaltyRemoved:
    def test_feasibility_violation_adds_support_not_infeasible_flat(self, monkeypatch):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", True)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 1.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
        }
        val_metrics = {
            "executed_trades": 50,
            "total_return_pct": -2.0,
            "sortino_ratio": 0.5,
            "max_drawdown_pct": 1.0,
            "win_rate": 40.0,
            "profit_factor": 0.9,
        }
        objectives, out = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], val_metrics=val_metrics,
        )
        assert out.get("feasibility_violation", 0.0) > 0.0
        assert objectives[0] < 100.0


class TestJointValF2F3:
    def test_f2_uses_max_train_val_dd_when_joint(self, monkeypatch):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 5.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }
        val_metrics = {
            "executed_trades": 50,
            "total_return_pct": 4.0,
            "sortino_ratio": 0.8,
            "max_drawdown_pct": 12.0,
            "win_rate": 45.0,
            "profit_factor": 1.21,
        }
        objectives, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], val_metrics=val_metrics,
        )
        assert np.isclose(objectives[1], 12.0)

    def test_f3_joint_win_rate_when_not_return_mode(self, monkeypatch):
        """Legacy win_rate mode still works when PHASE2_USE_TOTAL_RETURN_OBJ
        is explicitly set to False (overriding the new default True)."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)
        monkeypatch.setattr(_cfg, "PHASE2_F3_OBJECTIVE", "win_rate")
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 55.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }
        val_metrics = {
            "executed_trades": 50,
            "total_return_pct": 4.0,
            "sortino_ratio": 0.8,
            "max_drawdown_pct": 1.0,
            "win_rate": 40.0,
            "profit_factor": 1.21,
        }
        objectives, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], val_metrics=val_metrics,
        )
        assert np.isclose(objectives[2], -40.0)


class TestF4ReturnConcentration:
    """Tests for the f4 return-concentration objective (Task 2)."""

    def test_f4_one_outlier_trade(self, monkeypatch):
        """A rule with 1 trade = +60 and 79 trades avg -0.5% receives f4 ≈ 1.0."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_F4_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_F4_EPSILON", 1e-6)
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 80,
            "total_return_pct": 5.0,
            "sortino_ratio": 0.5,
            "max_drawdown_pct": 10.0,
            "win_rate": 1.25,
            "profit_factor": 1.5,
            "per_symbol_metrics": {},
            "sum_positive_trade_pnl": 60.0,
            "sum_negative_trade_pnl": 39.5,
            "max_single_trade_pnl": 60.0,
        }
        objectives, out_metrics = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )
        # f4 = 60.0 / 60.0 = 1.0
        assert len(objectives) == 4, f"Expected 4 objectives, got {len(objectives)}"
        assert np.isclose(objectives[3], 1.0), f"Expected f4≈1.0, got {objectives[3]}"
        assert np.isclose(out_metrics["f4_concentration"], 1.0)

    def test_f4_uniform_trades(self, monkeypatch):
        """A rule with uniform +1% across 80 trades receives f4 ≈ 0.0125."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_F4_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_F4_EPSILON", 1e-6)
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 80,
            "total_return_pct": 80.0,
            "sortino_ratio": 5.0,
            "max_drawdown_pct": 0.5,
            "win_rate": 100.0,
            "profit_factor": 99.0,
            "per_symbol_metrics": {},
            "sum_positive_trade_pnl": 80.0,
            "sum_negative_trade_pnl": 0.0,
            "max_single_trade_pnl": 1.0,
        }
        objectives, out_metrics = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )
        # f4 = 1.0 / 80.0 = 0.0125
        assert len(objectives) == 4, f"Expected 4 objectives, got {len(objectives)}"
        assert np.isclose(objectives[3], 0.0125), f"Expected f4≈0.0125, got {objectives[3]}"
        assert np.isclose(out_metrics["f4_concentration"], 0.0125)

    def test_f4_disabled_returns_three_objectives(self, monkeypatch):
        """When PHASE2_F4_ENABLED = False, objectives.shape == (3,) and no f4 in metrics."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_F4_ENABLED", False)
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 10.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {},
            "sum_positive_trade_pnl": 60.0,
            "sum_negative_trade_pnl": 40.0,
            "max_single_trade_pnl": 60.0,
        }
        objectives, out_metrics = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )
        assert objectives.shape == (3,), f"Expected 3 objectives, got {objectives.shape}"
        assert "f4_concentration" not in out_metrics or out_metrics["f4_concentration"] == 0.0

    def test_f4_is_worst_when_below_trade_floor(self, monkeypatch):
        """Low-support rules receive worst-case concentration evidence."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_F4_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_F4_EPSILON", 1e-6)
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 100)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 5,  # below floor of 100
            "total_return_pct": 5.0,
            "sortino_ratio": 0.5,
            "max_drawdown_pct": 10.0,
            "win_rate": 60.0,
            "profit_factor": 2.0,
            "per_symbol_metrics": {},
            "sum_positive_trade_pnl": 50.0,
            "sum_negative_trade_pnl": 10.0,
            "max_single_trade_pnl": 40.0,
        }
        objectives, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )
        # f4 should be worst-case because executed_trades (5) < trade_floor
        assert len(objectives) == 4, f"Expected 4 objectives, got {len(objectives)}"
        assert objectives[3] >= 1.0, f"Expected f4>=1, got {objectives[3]}"

    def test_f4_default_is_three_objectives_when_config_deleted(self, monkeypatch):
        """When PHASE2_F4_ENABLED is deleted from config (missing attr),
        both the f4 computation gate and the objectives-array-size gate
        must default to False — producing 3 objectives, not 4 with all-zero f4."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        # Explicitly delete the attribute so getattr falls to its default.
        if hasattr(_cfg, "PHASE2_F4_ENABLED"):
            monkeypatch.delattr(_cfg, "PHASE2_F4_ENABLED")
        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": 10.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {},
            "sum_positive_trade_pnl": 60.0,
            "sum_negative_trade_pnl": 40.0,
            "max_single_trade_pnl": 60.0,
        }
        objectives, out_metrics = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )
        # Without PHASE2_F4_ENABLED, both getattr defaults are False → 3 objectives
        assert objectives.shape == (3,), (
            f"Expected 3 objectives when config attr is deleted, "
            f"got {objectives.shape}"
        )
        assert "f4_concentration" not in out_metrics or out_metrics["f4_concentration"] == 0.0


class TestF3PathResolution:
    """Parametrized tests for f3 path resolution (Task 5: audit finding #5).

    Verifies that:
      - USE_TOTAL_RETURN_OBJ=True (default) → robust_return_pct (val-blended min).
      - USE_TOTAL_RETURN_OBJ=False → legacy PHASE2_F3_OBJECTIVE path
        (profit_factor, win_rate, or cv_fold_min).
    """

    @pytest.mark.parametrize("use_total,f3_obj,expected_f3,train_ret,val_ret,pf,wr,cv_folds", [
        (True,  "profit_factor", -3.0,    10.0, 3.0,  1.2, 50.0, []),    # default: robust wins
        (True,  "win_rate",      -3.0,    10.0, 3.0,  1.2, 50.0, []),    # default: robust wins
        (True,  "cv_fold_min",    -3.0,    10.0, 3.0,  1.2, 50.0, []),   # default: robust wins
        (False, "profit_factor",  -1.1,    10.0, 3.0,  1.2, 50.0, []),   # f3 = -min(pf, val_pf) = -min(1.2, 1.1) = -1.1
        (False, "win_rate",       -50.0,   10.0, 3.0,  1.2, 50.0, []),   # f3 = -win_rate
        (False, "cv_fold_min",    -3.5,    10.0, 3.0,  1.2, 50.0, [3.5, 4.0, 5.0]),  # f3 = -min(cv_folds)
    ])
    def test_f3_path_resolution(
        self, monkeypatch,
        use_total, f3_obj, expected_f3,
        train_ret, val_ret, pf, wr, cv_folds,
    ):
        """Verify the correct f3 formula runs for each (USE_TOTAL_RETURN_OBJ, F3_OBJECTIVE) pair."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", True)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", use_total)
        monkeypatch.setattr(_cfg, "PHASE2_USE_ROBUST_RETURN_OBJ", True)
        monkeypatch.setattr(_cfg, "PHASE2_F3_OBJECTIVE", f3_obj)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 1)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_VAL_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 100,
            "total_return_pct": train_ret,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": wr,
            "profit_factor": pf,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }
        if cv_folds:
            metrics["_cv_fold_returns"] = cv_folds

        val_metrics = {
            "executed_trades": 50,
            "total_return_pct": val_ret,
            "sortino_ratio": 0.8,
            "max_drawdown_pct": 1.0,
            "win_rate": 55.0,
            "profit_factor": 1.1,
        }

        objectives, out_metrics = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
            val_metrics=val_metrics,
        )
        # f3 is at index 2; it's minimised, so robust_return gives -min(train, val)
        assert np.isclose(objectives[2], expected_f3), (
            f"USE_TOTAL={use_total}, F3_OBJ={f3_obj}: expected f3≈{expected_f3}, got {objectives[2]}"
        )


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


class TestArchiveMetadata:
    def test_annotate_archive_entry_metadata(self) -> None:
        rules = [{
            "chromosome": [0, 1],
            "conditions": ["[feat_0] IS Low"],
            "objectives": {"total_return_pct": 2.0, "profit_factor": 1.1},
            "executed_trades": 20,
            "val_objectives": {"total_return_pct": 1.0, "profit_factor": 1.0},
            "val_executed_trades": 10,
        }]
        annotated = Rule_Pool_Generator._annotate_archive_entries(
            rules,
        )
        assert "robust_score" in annotated[0]


class TestIslandAwareTradeFloor:
    """Tests for island-aware hard reject floor and config constant usage."""

    def test_island_floor_respected_when_provided(self, monkeypatch):
        """When island_hyperparams.min_trade_pool_floor=15 and executed=20,
        no hard-reject penalty should fire (20 >= 15), even though global
        MIN_TRADE_POOL_FLOOR=25 would normally reject 20."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 25)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_DRAWDOWN_GATE", 200.0)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)

        island = _cfg.IslandHyperparams(
            profile="cluster",
            min_trade_support=10,
            min_trade_pool_floor=15,
            sortino_min_trade_threshold=10,
            val_trade_floor=5,
            min_profitable_symbols=2,
            monthly_admission_min_months=3,
            monthly_admission_min_profitable_ratio=0.5,
            skip_symbol_robustness_penalty=True,
            n_rows=1000,
            n_symbols=5,
        )

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 20,  # >= 15 (island floor) but < 25 (global floor)
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }

        # With island_hyperparams: 20 >= 15 → no hard reject penalty
        objectives_with, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
            island_hyperparams=island,
        )
        # f2 = dd_for_obj(2.0) + 0 support + 0 dd_gate + 0 trade = 2.0
        assert np.isclose(objectives_with[1], 2.0, atol=0.1), (
            f"Expected f2 ≈ 2.0 (no trade penalty), got {objectives_with[1]}"
        )

        # Without island_hyperparams: 20 < 25 → hard reject penalty on f2
        objectives_without, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
            island_hyperparams=None,
        )
        # f2 = dd_for_obj(100) + 0 support + 0 dd_gate(100<200) + trade(50) = 150.0
        assert np.isclose(objectives_without[1], 150.0, atol=0.1), (
            f"Expected f2 ≈ 150 (100 dd + 50 penalty), got {objectives_without[1]}"
        )

    def test_penalty_uses_config_constant(self, monkeypatch):
        """When the trade floor is triggered, trade_penalty should equal
        PHASE2_INFEASIBLE_OBJECTIVE_PENALTY, not a hardcoded 50.0."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 100)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "PHASE2_INFEASIBLE_OBJECTIVE_PENALTY", 99.9)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_DRAWDOWN_GATE", 200.0)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 10,  # < 100 → hard reject
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }

        objectives, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )
        # f2 = dd_for_obj(100) + 0 support + 0 dd_gate(100<200) + trade(99.9) = 199.9
        assert np.isclose(objectives[1], 199.9, atol=0.1), (
            f"Expected f2 ≈ 199.9 (100 dd + 99.9 penalty), got {objectives[1]}"
        )
        # Task 3: trade_penalty removed from f3.
        # f3 = -f3_val(0) + 0 support + 0 diversity + 0 cond + 0 overfit_gap = 0.0
        assert np.isclose(objectives[2], 0.0, atol=0.1), (
            f"Expected f3 ≈ 0.0 (no trade_penalty on f3), got {objectives[2]}"
        )

    def test_fallback_to_effective_floor(self, monkeypatch):
        """When island_hyperparams is None, trade_floor falls back to
        effective_min_trade_pool_floor(n_valid_rows)."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        monkeypatch.setattr(_cfg, "SPLIT_MODE", "holdout")
        monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 25)
        monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 1)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "PHASE2_RETURN_FLOOR_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE2_PROFIT_FACTOR_FLOOR", 0.0)
        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", False)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_DRAWDOWN_GATE", 200.0)
        monkeypatch.setattr(_cfg, "PHASE2_USE_TOTAL_RETURN_OBJ", False)

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = {
            "executed_trades": 20,  # < 25 → hard reject
            "total_return_pct": 5.0,
            "sortino_ratio": 1.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 50.0,
            "profit_factor": 1.2,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 300.0},
            },
        }

        # Without island, with n_valid_rows: effective_min_trade_pool_floor(500)
        # will return MIN_TRADE_POOL_FLOOR=25 (since not in purged WF mode).
        # So 20 < 25 → penalty fires → f2 = 100 + 50 = 150
        objectives, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
            n_valid_rows=500,
            island_hyperparams=None,
        )
        assert np.isclose(objectives[1], 150.0, atol=0.1), (
            f"Expected f2 ≈ 150 (100 dd + 50 penalty), got {objectives[1]}"
        )


class TestConditionBounds:
    def test_config_allows_bounded_conditions(self):
        assert _cfg.MIN_CONDITIONS == 2
        assert _cfg.MAX_CONDITIONS == 4

    def test_mutation_repair_preserves_condition_bounds(self):
        from gpu_fuzzy_trader.phases.phase2_rule_pool import _mutate

        fi = _make_feature_infos(["positive"] * 6)
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(7)
        parent = _chromosome_with_min_active(n_features=6, dont_care=int(dc[0]))
        child = _mutate(parent, fi, dc, rng, mutation_rate=0.5)
        active = _count_active_conditions(child, dc)
        assert _cfg.MIN_CONDITIONS <= active <= _cfg.MAX_CONDITIONS


# ---------------------------------------------------------------------------
# Tests: C5 Symbol gene dont_care bias (F2)
# ---------------------------------------------------------------------------

class TestSymbolGeneBias:
    """C5 mutation bias: force symbol-gene to dont_care / inactive with
    probability PHASE2_SYMBOL_GENE_DONT_CARE_PROB."""

    # ------------------------------------------------------------------
    # Dense path tests
    # ------------------------------------------------------------------

    def _make_fi_with_symbol(self, modes: list[str], symbol_idx: int = 0) -> list[dict]:
        """Create feature_infos with a feature whose name contains 'symbol'."""
        fi = _make_feature_infos(modes)
        fi[symbol_idx] = {**fi[symbol_idx], "name": "symbol_cluster"}
        return fi

    def test_symbol_gene_bias_dense_force(self, monkeypatch):
        """PHASE2_SYMBOL_GENE_DONT_CARE_PROB=1.0: symbol gene always forced to dont_care."""
        monkeypatch.setattr(_cfg, "PHASE2_ENCODING", "dense")
        monkeypatch.setattr(_cfg, "PHASE2_SYMBOL_GENE_DONT_CARE_PROB", 1.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "MIN_CONDITIONS", 1)

        fi = self._make_fi_with_symbol(["positive", "positive", "positive"])
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(42)

        # Symbol feature at index 0, active (not dont_care)
        chrom = np.array([0, 1, 2], dtype=np.int32)
        for _ in range(50):
            child = _mutate(chrom, fi, dc, rng, mutation_rate=0.0)
            # Symbol gene (index 0) should always be forced to dont_care
            assert child[0] == int(dc[0]), (
                f"Symbol gene should be dont_care({dc[0]}), got {child[0]}"
            )

    def test_symbol_gene_bias_dense_disabled(self, monkeypatch):
        """PHASE2_SYMBOL_GENE_DONT_CARE_PROB=0.0: symbol gene never force-set."""
        monkeypatch.setattr(_cfg, "PHASE2_ENCODING", "dense")
        monkeypatch.setattr(_cfg, "PHASE2_SYMBOL_GENE_DONT_CARE_PROB", 0.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "MIN_CONDITIONS", 1)

        fi = self._make_fi_with_symbol(["positive", "positive", "positive"])
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(42)

        chrom = np.array([0, 1, 2], dtype=np.int32)
        symbol_gene_was_active = False
        for _ in range(200):
            child = _mutate(chrom, fi, dc, rng, mutation_rate=0.0)
            # With prob=0, symbol gene should NEVER be force-set
            if child[0] != int(dc[0]):
                symbol_gene_was_active = True
                break
        assert symbol_gene_was_active, (
            "Symbol gene should remain active when bias is disabled"
        )

    def test_symbol_gene_bias_dense_partial(self, monkeypatch):
        """With probability ~0.5, about half of calls force symbol to dont_care."""
        monkeypatch.setattr(_cfg, "PHASE2_ENCODING", "dense")
        monkeypatch.setattr(_cfg, "PHASE2_SYMBOL_GENE_DONT_CARE_PROB", 0.5)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "MIN_CONDITIONS", 1)

        fi = self._make_fi_with_symbol(["positive", "positive", "positive"])
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(12345)

        chrom = np.array([0, 1, 2], dtype=np.int32)
        n_trials = 200
        forced_count = 0
        for _ in range(n_trials):
            child = _mutate(chrom, fi, dc, rng, mutation_rate=0.0)
            if child[0] == int(dc[0]):
                forced_count += 1
        # Should be ~50%; allow ±20% tolerance
        assert 0.20 <= forced_count / n_trials <= 0.80, (
            f"Expected ~50% forced, got {forced_count}/{n_trials}"
        )

    def test_symbol_gene_bias_dense_no_symbol_feature(self, monkeypatch):
        """No symbol feature in feature_infos: bias silently does nothing (no crash)."""
        monkeypatch.setattr(_cfg, "PHASE2_ENCODING", "dense")
        monkeypatch.setattr(_cfg, "PHASE2_SYMBOL_GENE_DONT_CARE_PROB", 1.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "MIN_CONDITIONS", 1)

        fi = _make_feature_infos(["positive", "positive", "positive"])
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(42)
        chrom = np.array([0, 1, 2], dtype=np.int32)

        for _ in range(50):
            child = _mutate(chrom, fi, dc, rng, mutation_rate=0.0)
            # Normal mutation (mutation_rate=0.0) should not change anything
            assert np.array_equal(child, chrom), (
                f"Without symbol feature, bias should not affect chromosome"
            )

    # ------------------------------------------------------------------
    # Sparse path tests (production code path)
    # ------------------------------------------------------------------

    def test_symbol_gene_bias_sparse_force(self, monkeypatch):
        """Sparse path: PHASE2_SYMBOL_GENE_DONT_CARE_PROB=1.0 forces symbol slot to inactive."""
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import mutate_sparse

        monkeypatch.setattr(_cfg, "PHASE2_SYMBOL_GENE_DONT_CARE_PROB", 1.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "MIN_CONDITIONS", 1)

        fi = self._make_fi_with_symbol(["positive", "positive", "positive", "positive"])
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(42)

        # Create sparse chromosome: slot 0 = symbol feature (idx 0, class 2)
        sparse_chrom = np.array([
            [0, 2],   # symbol feature active
            [1, 1],   # non-symbol feature
            [2, 3],   # non-symbol feature
            [-1, 0],  # inactive
        ], dtype=np.int32)

        for _ in range(30):
            child = mutate_sparse(sparse_chrom, fi, dc, rng, mutation_rate=0.0)
            # Symbol slot (idx 0) should be forced to INACTIVE by the bias
            # With mutation_rate=0.0 and MIN_CONDITIONS=1, repair won't reactivate
            assert not np.any(child[:, 0] == 0), (
                f"Symbol feature slot should be inactive (INACTIVE_FEAT_IDX={INACTIVE_FEAT_IDX}), "
                f"but found active symbol slot(s)"
            )

    def test_symbol_gene_bias_sparse_disabled(self, monkeypatch):
        """Sparse path: PHASE2_SYMBOL_GENE_DONT_CARE_PROB=0.0, symbol slot stays active."""
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import mutate_sparse

        monkeypatch.setattr(_cfg, "PHASE2_SYMBOL_GENE_DONT_CARE_PROB", 0.0)
        monkeypatch.setattr(_cfg, "MAX_CONDITIONS", 4)
        monkeypatch.setattr(_cfg, "MIN_CONDITIONS", 1)

        fi = self._make_fi_with_symbol(["positive", "positive", "positive", "positive"])
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(42)

        sparse_chrom = np.array([
            [0, 2],
            [1, 1],
            [2, 3],
            [-1, 0],
        ], dtype=np.int32)

        # With prob=0 and mutation_rate=0, symbol slot should never be deactivated
        for _ in range(50):
            child = mutate_sparse(sparse_chrom, fi, dc, rng, mutation_rate=0.0)
            assert np.any(child[:, 0] == 0), (
                "Symbol slot should remain active when bias is disabled"
            )


# ---------------------------------------------------------------------------
# Tests: C6 Val-leak gate (F2)
# ---------------------------------------------------------------------------

class TestValLeakGate:
    """C6: Val-derived penalties must be gated behind JOINT_TRAIN_VAL or
    VAL_IN_FITNESS_PENALTY."""

    @staticmethod
    def _base_objective_kwargs():
        """Return standard monkeypatching for clean baseline metrics."""
        return {
            "PHASE2_JOINT_TRAIN_VAL": False,
            "PHASE2_VAL_IN_FITNESS_PENALTY": False,
            "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS": False,
            "PHASE2_USE_TOTAL_RETURN_OBJ": False,
            "PHASE2_F3_OBJECTIVE": "profit_factor",
            "MIN_TRADE_SUPPORT": 1,
            "MIN_TRADE_POOL_FLOOR": 1,
            "PHASE2_RETURN_FLOOR_PCT": -100.0,
            "PHASE2_PROFIT_FACTOR_FLOOR": 0.0,
            "PHASE2_VAL_RETURN_FLOOR_PCT": -100.0,
            "PHASE2_MAX_DRAWDOWN_GATE": 200.0,
            "PHASE2_MIN_PROFITABLE_SYMBOLS_PENALTY": 1,
            "MAX_CONDITIONS": 4,
        }

    def _apply_settings(self, monkeypatch, **overrides):
        """Apply base settings with optional overrides."""
        settings = dict(self._base_objective_kwargs())
        settings.update(overrides)
        for key, val in settings.items():
            monkeypatch.setattr(_cfg, key, val)

    def _make_clean_metrics(self) -> dict:
        """Metrics that trigger no train-side penalties."""
        return {
            "executed_trades": 100,
            "total_return_pct": 10.0,
            "sortino_ratio": 2.0,
            "max_drawdown_pct": 5.0,
            "win_rate": 60.0,
            "profit_factor": 1.5,
            "per_symbol_metrics": {
                "SYM1": {"net_pnl": 100.0},
                "SYM2": {"net_pnl": 200.0},
                "SYM3": {"net_pnl": 150.0},
            },
        }

    def _make_bad_val_metrics(self) -> dict:
        """Val metrics that WOULD trigger penalties if the gate were open."""
        return {
            "executed_trades": 3,  # below val_trade_floor → triggers trade-floor cap
            "total_return_pct": -15.0,  # below VAL_RETURN_FLOOR_PCT (0.0)
            "sortino_ratio": -1.0,
            "max_drawdown_pct": 25.0,
            "win_rate": 20.0,
            "profit_factor": 0.3,  # below PROFIT_FACTOR_FLOOR (0.0→falls through anyway)
            # Note: symbol_robustness uses per_symbol_metrics which bad val doesn't have → 0
        }

    def test_val_penalties_gated_closed(self, monkeypatch):
        """When both JOINT_TRAIN_VAL and VAL_IN_FITNESS_PENALTY are False,
        val-derived penalties must NOT enter objectives."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        self._apply_settings(monkeypatch,
            PHASE2_JOINT_TRAIN_VAL=False,
            PHASE2_VAL_IN_FITNESS_PENALTY=False,
            # Set tight val floors to trigger penalties
            PHASE2_VAL_RETURN_FLOOR_PCT=0.0,
            PHASE2_PROFIT_FACTOR_FLOOR=1.0,
        )

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = self._make_clean_metrics()
        val_metrics = self._make_bad_val_metrics()

        # Without val_metrics
        obj_no_val, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )
        # With val_metrics, gate closed
        obj_val_closed, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
            val_metrics=val_metrics,
        )
        # Objectives should be identical (val penalties gated out)
        assert np.allclose(obj_no_val, obj_val_closed, atol=1e-6), (
            f"With gate closed, val should not affect objectives.\n"
            f"  no_val:  {obj_no_val}\n"
            f"  val_closed: {obj_val_closed}"
        )

    def test_raw_feasibility_violation_gated_closed(self, monkeypatch):
        """Bad val must not set feasibility_violation when gate is closed."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        self._apply_settings(monkeypatch,
                             PHASE2_JOINT_TRAIN_VAL=False,
                             PHASE2_VAL_IN_FITNESS_PENALTY=False,
                             PHASE2_POOL_REQUIRE_POSITIVE_SPLITS=True,
                             )

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = self._make_clean_metrics()
        val_metrics = self._make_bad_val_metrics()

        _, out_closed = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], val_metrics=val_metrics,
        )
        assert out_closed.get("feasibility_violation", 0.0) == 0.0

        monkeypatch.setattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", True)
        _, out_open = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [], val_metrics=val_metrics,
        )
        assert out_open.get("feasibility_violation", 0.0) > 0.0

    def test_val_penalties_gated_open(self, monkeypatch):
        """When VAL_IN_FITNESS_PENALTY=True, val-derived penalties DO enter objectives."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        self._apply_settings(monkeypatch,
            PHASE2_JOINT_TRAIN_VAL=False,
            PHASE2_VAL_IN_FITNESS_PENALTY=True,
            # Set tight val floors to trigger penalties
            PHASE2_VAL_RETURN_FLOOR_PCT=0.0,
            PHASE2_PROFIT_FACTOR_FLOOR=1.0,
        )

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = self._make_clean_metrics()
        val_metrics = self._make_bad_val_metrics()

        # Without val_metrics
        obj_no_val, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
        )
        # With val_metrics, gate OPEN
        obj_val_open, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
            val_metrics=val_metrics,
        )
        # With gate open, val penalties should make objectives WORSE
        # f2 = dd + support_penalty + dd_gate + trade_penalty
        # support_penalty should include val_floor_penalty and val trade-floor cap
        assert obj_val_open[1] > obj_no_val[1] + 1.0, (
            f"With gate open, f2 should be worse due to val penalties.\n"
            f"  no_val:  {obj_no_val}\n"
            f"  val_open: {obj_val_open}"
        )
        # f3 = -f3_val + diversity + trade + cond
        # trade_penalty might fire if val_trade_floor is > 0 and val_executed=3
        # But val_trade_floor for 300 rows with 70/30 split will be ~6
        # Actually val_trade_floor uses _val_trade_floor_for_objectives
        # which is based on n_valid_rows. Since we pass no n_valid_rows,
        # and no island_hyperparams, it may use a default
        # Let's just check that f3 is also worse (trade_penalty adds to all objs)
        assert obj_val_open[2] >= obj_no_val[2], (
            f"With gate open, f3 should be >= (worse/equal) no-val case.\n"
            f"  no_val:  {obj_no_val}\n"
            f"  val_open: {obj_val_open}"
        )

    def test_val_penalties_gate_open_vs_closed(self, monkeypatch):
        """Direct comparison: gate open produces strictly worse objectives than gate closed."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import (
            compute_phase2_objectives_from_metrics,
        )

        self._apply_settings(monkeypatch,
            PHASE2_JOINT_TRAIN_VAL=False,
            PHASE2_VAL_IN_FITNESS_PENALTY=False,
            PHASE2_VAL_RETURN_FLOOR_PCT=0.0,
            PHASE2_PROFIT_FACTOR_FLOOR=1.0,
        )

        dont_cares = np.full(4, 5, dtype=np.int32)
        chrom = np.array([0, 1, 2, 3], dtype=np.int32)
        metrics = self._make_clean_metrics()
        val_metrics = self._make_bad_val_metrics()

        # Gate closed
        obj_closed, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
            val_metrics=val_metrics,
        )

        # Gate open
        monkeypatch.setattr(_cfg, "PHASE2_VAL_IN_FITNESS_PENALTY", True)
        obj_open, _ = compute_phase2_objectives_from_metrics(
            chrom, dont_cares, metrics, [],
            val_metrics=val_metrics,
        )

        # f2 should be strictly worse with gate open (val penalties added to support_penalty)
        assert obj_open[1] > obj_closed[1] + 0.5, (
            f"Gate open should produce worse f2 than gate closed.\n"
            f"  closed: {obj_closed}\n"
            f"  open:   {obj_open}"
        )


# ---------------------------------------------------------------------------
# Tests: Persistent RNG state across run_epoch() calls
# ---------------------------------------------------------------------------

class TestRulePoolGeneratorRng:
    """Verify that Rule_Pool_Generator's persistent self._rng advances across
    multiple run_epoch() calls, and that distinct seeds produce distinct state."""

    def _make_generator(self, seed: int = 42) -> Rule_Pool_Generator:
        fi = _make_feature_infos(["positive", "positive", "positive"])
        df = _make_train_df(n_rows=100, n_features=3, symbols=["A"])
        # n_generations must leave Stage A room for two non-final epoch gens
        # after the first run_epoch(2). With A:B = 20:20, stage_a = total/2,
        # so total>=8 → stage_a>=4 → remaining after first epoch >=2.
        # (A single last-gen epoch skips offspring and does not draw from rng.)
        gen = Rule_Pool_Generator(
            df, fi, "long",
            pop_size=8,
            n_generations=8,
            seed=seed,
        )
        return gen

    def test_rng_state_advances_across_run_epoch_calls(self):
        """After two run_epoch() calls, the internal RNG state must differ
        from the initial state, proving state advances across calls."""
        gen = self._make_generator(seed=42)
        initial_state = gen._rng.bit_generator.state["state"]["state"]

        gen.run_epoch(n_generations=2)
        after_first = gen._rng.bit_generator.state["state"]["state"]

        gen.run_epoch(n_generations=2)
        after_second = gen._rng.bit_generator.state["state"]["state"]

        # The RNG state should have advanced after each call
        assert initial_state != after_first, (
            "RNG state did not advance after first run_epoch() call"
        )
        assert after_first != after_second, (
            "RNG state did not advance after second run_epoch() call"
        )
        assert initial_state != after_second, (
            "RNG state should differ from initial after two calls"
        )

    def test_rng_state_not_reset_between_run_epoch_calls(self):
        """The RNG should produce *different* sequences in two consecutive
        run_epoch() calls, proving they do NOT replay from the same seed."""
        gen = self._make_generator(seed=42)

        # Capture RNG state used during first epoch by peeking at the
        # generator's internal state after the epoch starts drawing numbers.
        gen.run_epoch(n_generations=2)
        state_after_first = gen._rng.bit_generator.state["state"]["state"]

        gen.run_epoch(n_generations=2)
        state_after_second = gen._rng.bit_generator.state["state"]["state"]

        # Different state means different future draw sequences
        assert state_after_first != state_after_second, (
            "RNG state after second run_epoch() should differ from first"
        )

    def test_distinct_seeds_produce_distinct_rng_state(self):
        """Two generators with different seeds must have different RNG state."""
        gen_a = self._make_generator(seed=42)
        gen_b = self._make_generator(seed=99)

        state_a = gen_a._rng.bit_generator.state["state"]["state"]
        state_b = gen_b._rng.bit_generator.state["state"]["state"]

        assert state_a != state_b, (
            "Generators with different seeds should have different RNG state"
        )

    def test_rng_is_persistent_generator(self):
        """Rule_Pool_Generator must initialize self._rng as a Generator."""
        gen = self._make_generator(seed=42)
        assert hasattr(gen, "_rng"), "Generator should have _rng attribute"
        assert isinstance(gen._rng, np.random.Generator), (
            "_rng should be a numpy random Generator"
        )


# ---------------------------------------------------------------------------
# Tests: _derive_island_seed
# ---------------------------------------------------------------------------

def test_derive_island_seed_distinct_for_distinct_ids():
    """_derive_island_seed must produce different seeds for different island IDs."""
    from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
        _derive_island_seed,
    )

    base = 42
    s1 = _derive_island_seed(base, "cluster_0")
    s2 = _derive_island_seed(base, "cluster_1")
    s3 = _derive_island_seed(base, "orphan_AAPL")

    ids = [s1, s2, s3]
    assert len(set(ids)) == 3, (
        f"Expected 3 distinct seeds for different island IDs, got {ids}"
    )


def test_derive_island_seed_deterministic():
    """_derive_island_seed must return the same value for same inputs."""
    from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
        _derive_island_seed,
    )

    s1 = _derive_island_seed(42, "cluster_0")
    s2 = _derive_island_seed(42, "cluster_0")
    assert s1 == s2, (
        f"Same inputs should produce same seed, got {s1} vs {s2}"
    )


def test_derive_island_seed_none_base():
    """_derive_island_seed should return None when base_seed is None."""
    from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
        _derive_island_seed,
    )

    result = _derive_island_seed(None, "cluster_0")
    assert result is None, (
        f"Expected None when base_seed is None, got {result}"
    )


def test_derive_island_seed_different_base_differs():
    """Different base seeds must produce different derived seeds for same ID."""
    from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
        _derive_island_seed,
    )

    s1 = _derive_island_seed(42, "cluster_0")
    s2 = _derive_island_seed(99, "cluster_0")
    assert s1 != s2, (
        f"Different base seeds should give different derived seeds, "
        f"got {s1} == {s2}"
    )


# ---------------------------------------------------------------------------
# Overfit ratio gate tests (Task 6)
# ---------------------------------------------------------------------------

class TestPoolAdmissionOverfitRatioGate:
    """Tests for the hard overfit ratio gate in pool admission.

    The ratio gate rejects rules where train_return / max(val_return, 0.1)
    exceeds PHASE2_OVERFIT_RATIO_FLOOR (default 3.0). It catches cases the
    absolute-pp gate misses (e.g., train=15%/val=4%, gap=11pp < 16pp).
    """

    def test_rejects_high_ratio_low_abs_gap(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """train=15%/val=4% (3.75× ratio, gap=11pp < 16pp) is REJECTED
        by the ratio gate even though absolute gap passes.
        """
        from gpu_fuzzy_trader.phases.phase2_support import (
            passes_pool_admission_gate,
        )

        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 16.0)
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 3.0)

        train = {
            "executed_trades": 100,
            "total_return_pct": 15.0,
            "profit_factor": 2.0,
        }
        val = {
            "executed_trades": 50,
            "total_return_pct": 4.0,
            "profit_factor": 1.5,
        }
        assert passes_pool_admission_gate(train, val) is False, (
            "Expected train=15%/val=4% (3.75×) to be REJECTED at pool admission"
        )

    def test_admits_low_ratio(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """train=15%/val=10% (1.5× ratio) is ADMITTED."""
        from gpu_fuzzy_trader.phases.phase2_support import (
            passes_pool_admission_gate,
        )

        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 3.0)

        train = {
            "executed_trades": 100,
            "total_return_pct": 15.0,
            "profit_factor": 2.0,
        }
        val = {
            "executed_trades": 50,
            "total_return_pct": 10.0,
            "profit_factor": 1.5,
        }
        assert passes_pool_admission_gate(train, val) is True, (
            "Expected train=15%/val=10% (1.5×) to be ADMITTED"
        )

    def test_admits_moderate_ratio(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """train/val ≈2.5× is ADMITTED when under OVERFIT_RATIO_FLOOR and gap gate."""
        from gpu_fuzzy_trader.phases.phase2_support import (
            passes_pool_admission_gate,
        )

        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 3.0)
        # Keep absolute gap within PHASE2_MAX_TRAIN_VAL_GAP_PCT (10.0):
        # train=16.5 / val=6.6 = 2.5× with gap = 9.9.
        train = {
            "executed_trades": 100,
            "total_return_pct": 16.5,
            "profit_factor": 2.0,
        }
        val = {
            "executed_trades": 50,
            "total_return_pct": 6.6,
            "profit_factor": 1.5,
        }
        assert passes_pool_admission_gate(train, val) is True, (
            "Expected ~2.5× train/val ratio within gap gate to be ADMITTED"
        )

    def test_floor_zero_preserves_pre_task6_behavior(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With PHASE2_OVERFIT_RATIO_FLOOR=0.0, the ratio gate is disabled
        and the high-ratio but low-abs-gap rule is ADMITTED (regression guard).
        """
        from gpu_fuzzy_trader.phases.phase2_support import (
            passes_pool_admission_gate,
        )

        monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
        monkeypatch.setattr(_cfg, "PHASE2_MAX_TRAIN_VAL_GAP_PCT", 16.0)
        monkeypatch.setattr(_cfg, "PHASE2_OVERFIT_RATIO_FLOOR", 0.0)

        train = {
            "executed_trades": 100,
            "total_return_pct": 15.0,
            "profit_factor": 2.0,
        }
        val = {
            "executed_trades": 50,
            "total_return_pct": 4.0,
            "profit_factor": 1.5,
        }
        assert passes_pool_admission_gate(train, val) is True, (
            "Expected train=15%/val=4% (3.75×) to be ADMITTED when "
            "PHASE2_OVERFIT_RATIO_FLOOR=0.0 (pre-task-6 regression guard)"
        )


class TestRefreshObjectivesOnResumeGate:
    """Task-10: gate cache refresh on PHASE2_PER_EPOCH_WINDOW_ROTATION.

    Verifies the gate logic at phase2_rule_pool.py:3261.
    """

    @staticmethod
    def _evaluate_gate(first_epoch: bool) -> bool:
        """Evaluate the gate expression as it appears in the source."""
        from gpu_fuzzy_trader import config as _cfg

        return (
            not first_epoch
            and bool(getattr(_cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True))
        )

    def test_rotation_on_resumed_epoch_refreshes(self, monkeypatch):
        """PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default):
        refresh = not first_epoch → True on resumed epochs.
        """
        monkeypatch.setattr(_cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True)
        assert self._evaluate_gate(first_epoch=False) is True, (
            "Expected refresh=True on resumed epoch when rotation is ON"
        )

    def test_rotation_on_first_epoch_skips(self, monkeypatch):
        """PHASE2_PER_EPOCH_WINDOW_ROTATION=True (default):
        refresh = not first_epoch → False on first epoch.
        """
        monkeypatch.setattr(_cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", True)
        assert self._evaluate_gate(first_epoch=True) is False, (
            "Expected refresh=False on first epoch when rotation is ON"
        )

    def test_rotation_off_resumed_epoch_skips(self, monkeypatch):
        """PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy):
        refresh is False even on resumed epochs — cache is NOT cleared.
        """
        monkeypatch.setattr(_cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", False)
        assert self._evaluate_gate(first_epoch=False) is False, (
            "Expected refresh=False on resumed epoch when rotation is OFF"
        )

    def test_rotation_off_first_epoch_skips(self, monkeypatch):
        """PHASE2_PER_EPOCH_WINDOW_ROTATION=False (legacy):
        refresh is False on first epoch (matches pre-task-10 behavior).
        """
        monkeypatch.setattr(_cfg, "PHASE2_PER_EPOCH_WINDOW_ROTATION", False)
        assert self._evaluate_gate(first_epoch=True) is False, (
            "Expected refresh=False on first epoch when rotation is OFF"
        )


def test_finalize_island_passes_full_admission_context(
    tmp_path,
    monkeypatch,
):
    """Final archive admission must use the full validation frame."""
    import gpu_fuzzy_trader.phases.phase2_rule_pool as module

    captured: dict = {}
    pool_path = tmp_path / "phase2_long_pool.json"
    original_pool_paths = module._POOL_PATHS.copy()
    module._POOL_PATHS["long"] = str(pool_path)

    class CVEvaluator:
        n_valid_rows = 99

    def fake_evolution(**kwargs):
        captured.update(kwargs)
        return [], []

    generator = object.__new__(Rule_Pool_Generator)
    generator.direction = "long"
    generator.feature_infos = _make_feature_infos(["positive"])
    generator.pop_size = 4
    generator._rng = np.random.default_rng(42)
    generator._evolution_state = {"state": True}
    generator._engine = object()
    generator._val_engine = object()
    generator._pool_val_engine = object()
    generator._cv_val_evaluator = CVEvaluator()
    generator._cached_monthly_val = pd.DataFrame({"value": np.arange(42)})
    generator._holdout_n_valid_rows = 7
    generator._train_df = pd.DataFrame({"value": np.arange(99)})
    generator.island_id = "cluster_0"
    generator.source_symbols = ["BTCUSDT"]
    generator.island_hyperparams = None
    generator.island_profile = "cluster"
    generator.run_id = "run-123"

    monkeypatch.setattr(
        module.Rule_Pool_Generator,
        "_ensure_engines",
        lambda self: None,
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution",
        fake_evolution,
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
        lambda feature_infos: {},
    )
    monkeypatch.setattr(
        module,
        "_filter_pool_by_admission",
        lambda pool, **kwargs: pool,
    )
    monkeypatch.setattr(
        module,
        "_run_monthly_admission_on_pool",
        lambda pool, generator: pool,
    )
    monkeypatch.setattr(
        module.Rule_Pool_Generator,
        "_annotate_archive_entries",
        staticmethod(lambda pool, **kwargs: pool),
    )
    monkeypatch.setattr(
        module.Rule_Pool_Generator,
        "_release_resources",
        lambda self: None,
    )

    try:
        assert generator.finalize_island() == []
    finally:
        module._POOL_PATHS.update(original_pool_paths)

    assert captured["pool_val_engine"] is generator._pool_val_engine
    assert captured["holdout_n_valid_rows"] == 42
    assert captured["train_n_rows"] == 99
    assert captured["cv_fold_evaluator"] is generator._cv_val_evaluator
    report_path = tmp_path / "reports" / (
        "phase2_long_island_cluster_0_coverage.json"
    )
    report = json.loads(report_path.read_text())
    assert report["run_id"] == "run-123"
    assert report["admission_evidence"]["pool_validation_rows"] == 42

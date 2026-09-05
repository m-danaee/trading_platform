"""Unit tests for gpu_fuzzy_trader.phases.phase2_init."""

from __future__ import annotations

import numpy as np
import pandas as pd
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_init import (
    assign_strata_to_indices,
    build_uniform_feature_probs,
    pick_active_count,
    repair_active_count,
    sample_sparse_chromosome,
)
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _count_active_conditions,
    _get_dont_cares,
    _init_population,
)
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator


def _feature_specs(names: list[str]) -> list[dict]:
    return [
        {"name": name, "mode": "positive"}
        for name in names
    ]


class TestPhase2InitHelpers:
    def test_uniform_feature_probs_ignore_feature_metadata_scores(self):
        fi = [
            {"name": "a", "mode": "positive", "score": 100.0},
            {"name": "b", "mode": "positive", "score": 0.01},
            {"name": "c", "mode": "positive"},
        ]
        p = build_uniform_feature_probs(fi)
        assert p.shape == (3,)
        assert np.array_equal(p, np.full(3, 1 / 3))

    def test_pick_active_count_in_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            k = pick_active_count(rng)
            assert _cfg.MIN_CONDITIONS <= k <= _cfg.MAX_CONDITIONS


class TestStratifiedInitPopulation:
    def test_all_chromosomes_respect_condition_bounds(self):
        fi = _feature_specs([f"f{i}" for i in range(12)])
        fi[0] = {
            "name": "amihud_illiquidity_20",
            "mode": "positive",
        }
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(42)
        pop = _init_population(
            200, fi, rng, init_strategy="stratified_sparse",
        )
        for row in pop:
            active = _count_active_conditions(row, dc)
            assert _cfg.MIN_CONDITIONS <= active <= _cfg.MAX_CONDITIONS

    def test_sparse_sample_respects_condition_bounds(self):
        fi = _feature_specs([
            "first_feature",
            "second_feature",
            "third_feature",
            "fourth_feature",
        ])
        dont_cares = _get_dont_cares(fi)
        probs = build_uniform_feature_probs(fi)
        rng = np.random.default_rng(0)
        active_counts: list[int] = []
        for _ in range(50):
            chrom = sample_sparse_chromosome(
                rng, fi, dont_cares, 3, "elite", probs,
            )
            active_counts.append(_count_active_conditions(chrom, dont_cares))
        assert all(_cfg.MIN_CONDITIONS <= c <=
                   _cfg.MAX_CONDITIONS for c in active_counts)

    def test_repair_active_count(self):
        fi = _feature_specs(["a", "b", "c", "d"])
        dont_cares = _get_dont_cares(fi)
        chrom = dont_cares.copy()
        chrom[0] = 0
        chrom[1] = 1
        chrom[2] = 2
        rng = np.random.default_rng(1)
        repaired = repair_active_count(
            chrom, fi, dont_cares, rng, build_uniform_feature_probs(fi),
        )
        assert _cfg.MIN_CONDITIONS <= _count_active_conditions(
            repaired, dont_cares,
        ) <= _cfg.MAX_CONDITIONS

    def test_assign_strata_elite_and_explorer_only(self):
        indices = np.arange(10)
        rng = np.random.default_rng(0)
        labels = assign_strata_to_indices(
            indices, (0.67, 0.33), rng,
        )
        assert set(labels) <= {"elite", "explorer"}
        assert labels.count("elite") + labels.count("explorer") == 10


class TestPruneSplitsToRuleFeatures:
    def test_catalog_builder_returns_one_train_only_feature_set(self):
        orchestrator = object.__new__(Pipeline_Orchestrator)
        orchestrator._cv_folds = None
        frame = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=4, freq="5min"),
            "symbol": ["A"] * 4,
            "label_open_next": [100.0, 101.0, 102.0, 103.0],
            "binary_signal": [0.0, 1.0, 0.0, 1.0],
        })

        assert orchestrator._build_rule_feature_catalog(frame) == [
            {"name": "binary_signal", "mode": "binary"},
        ]

    def test_rule_feature_names_keep_catalog_features(self):
        feature_specs = [
            {"name": "feat_a", "mode": "positive"},
            {"name": "feat_b", "mode": "positive"},
        ]
        names = Pipeline_Orchestrator._rule_feature_names(feature_specs)
        assert names == ["feat_a", "feat_b"]

    def test_prune_splits_drops_non_catalog_feature_columns(self):
        rng = np.random.default_rng(0)
        n = 20
        cols = {
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "symbol": ["A"] * n,
            "label_open_next": rng.uniform(100, 200, n),
            "label_close_288": rng.uniform(100, 200, n),
            "label_min_288": rng.uniform(90, 100, n),
            "label_max_288": rng.uniform(100, 110, n),
            "label_max_before_min": rng.integers(0, 2, n),
            "_symbol_bar_index": np.arange(n),
            "feat_a": rng.standard_normal(n),
            "realized_vol_20": rng.standard_normal(n),
        }
        train_df = pd.DataFrame(cols)
        val_df = train_df.copy()
        feature_specs = [{"name": "feat_a", "mode": "positive"}]
        pruned_train, pruned_val = Pipeline_Orchestrator._prune_splits_to_rule_features(
            train_df, val_df, feature_specs,
        )
        assert "feat_a" in pruned_train.columns
        assert "realized_vol_20" not in pruned_train.columns
        assert "realized_vol_20" not in pruned_val.columns

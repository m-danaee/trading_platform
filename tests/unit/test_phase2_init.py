"""Unit tests for gpu_fuzzy_trader.phases.phase2_init."""

from __future__ import annotations

import numpy as np
import pandas as pd
from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_init import (
    assign_strata_to_indices,
    build_feature_sampling_probs,
    pick_active_count,
    regime_gene_indices,
    repair_active_count,
    sample_sparse_chromosome,
)
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _count_active_conditions,
    _get_dont_cares,
    _init_population,
)
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator


def _feature_infos_with_scores(names_scores: list[tuple[str, float]]) -> list[dict]:
    return [
        {"name": n, "mode": "positive", "score": s}
        for n, s in names_scores
    ]


class TestPhase2InitHelpers:
    def test_build_feature_sampling_probs_sums_to_one(self):
        fi = _feature_infos_with_scores([("a", 1.0), ("b", 0.01), ("c", 0.0)])
        p = build_feature_sampling_probs(fi)
        assert p.shape == (3,)
        assert abs(p.sum() - 1.0) < 1e-9
        assert p[0] > p[1] > p[2]

    def test_regime_gene_indices_overlap(self):
        fi = _feature_infos_with_scores([
            ("amihud_illiquidity_20", 0.2),
            ("rsi_centered_14", 0.1),
        ])
        assert regime_gene_indices(fi) == [0]

    def test_pick_active_count_in_bounds(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            k = pick_active_count(rng)
            assert _cfg.MIN_CONDITIONS <= k <= _cfg.MAX_CONDITIONS


class TestStratifiedInitPopulation:
    def test_all_chromosomes_respect_condition_bounds(self):
        fi = _feature_infos_with_scores([
            (f"f{i}", float(i + 1)) for i in range(12)
        ])
        fi[0] = {
            "name": "amihud_illiquidity_20",
            "mode": "positive",
            "score": 0.5,
        }
        dc = _get_dont_cares(fi)
        rng = np.random.default_rng(42)
        pop = _init_population(
            200, fi, rng, init_strategy="stratified_sparse",
        )
        for row in pop:
            active = _count_active_conditions(row, dc)
            assert _cfg.MIN_CONDITIONS <= active <= _cfg.MAX_CONDITIONS

    def test_regime_stratum_extreme_classes(self):
        fi = [
            {
                "name": "amihud_illiquidity_20",
                "mode": "positive",
                "score": 0.5,
            },
            {"name": "rsi_centered_14", "mode": "positive", "score": 0.1},
            {"name": "bb_width_rel_20", "mode": "positive", "score": 0.05},
            {"name": "channel_pos_20", "mode": "positive", "score": 0.02},
        ]
        dont_cares = _get_dont_cares(fi)
        probs = build_feature_sampling_probs(fi)
        regime_idx = regime_gene_indices(fi)
        rng = np.random.default_rng(0)
        found_extreme = False
        for _ in range(100):
            chrom = sample_sparse_chromosome(
                rng, fi, dont_cares, 4, "regime", probs, regime_idx,
            )
            for idx in regime_idx:
                if chrom[idx] in (0, int(dont_cares[idx]) - 1):
                    found_extreme = True
        assert found_extreme

    def test_repair_active_count(self):
        fi = _feature_infos_with_scores([("a", 1.0), ("b", 0.5), ("c", 0.1)])
        dont_cares = _get_dont_cares(fi)
        chrom = dont_cares.copy()
        chrom[0] = 0
        chrom[1] = 1
        chrom[2] = 2
        rng = np.random.default_rng(1)
        repaired = repair_active_count(
            chrom, fi, dont_cares, rng, build_feature_sampling_probs(fi),
        )
        assert _cfg.MIN_CONDITIONS <= _count_active_conditions(
            repaired, dont_cares,
        ) <= _cfg.MAX_CONDITIONS

    def test_assign_strata_no_regime_merges_to_elite(self):
        fi = _feature_infos_with_scores([("a", 1.0), ("b", 0.5)])
        indices = np.arange(10)
        rng = np.random.default_rng(0)
        labels = assign_strata_to_indices(
            indices, (0.5, 0.3, 0.2), [], rng,
        )
        assert "regime" not in labels
        assert labels.count("elite") + labels.count("explorer") == 10


class TestPruneSplitsRegimeColumns:
    def test_phase1_keep_feature_names_includes_regime_features(self):
        phase1 = {
            "long": [{"name": "feat_a", "mode": "positive", "score": 1.0}],
            "short": [],
        }
        names = Pipeline_Orchestrator._phase1_keep_feature_names(phase1)
        assert "feat_a" in names
        for col in _cfg.PHASE1_REGIME_FEATURES:
            assert col in names

    def test_prune_splits_retains_regime_column(self):
        rng = np.random.default_rng(0)
        n = 20
        cols = {
            "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
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
        phase1 = {
            "long": [{"name": "feat_a", "mode": "positive", "score": 1.0}],
            "short": [],
        }
        pruned_train, pruned_val = Pipeline_Orchestrator._prune_splits_after_phase1(
            train_df, val_df, phase1,
        )
        assert "feat_a" in pruned_train.columns
        assert "realized_vol_20" in pruned_train.columns
        assert "realized_vol_20" in pruned_val.columns

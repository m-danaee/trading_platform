"""Unit tests for cluster island scheduler budget math."""

from __future__ import annotations

from gpu_fuzzy_trader import config as cfg


def test_gens_per_cluster_split():
    total = int(cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
    k = int(cfg.PHASE2_N_CLUSTERS)
    gens_per = max(1, total // max(1, k))
    assert gens_per * k <= total + k


def test_epoch_rounds_cover_budget():
    total = int(cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
    k = int(cfg.PHASE2_N_CLUSTERS)
    epoch = int(cfg.PHASE2_ISLAND_EPOCH_GENERATIONS)
    gens_per = max(1, total // max(1, k))
    rounds = (gens_per + epoch - 1) // epoch
    assert rounds * epoch >= gens_per

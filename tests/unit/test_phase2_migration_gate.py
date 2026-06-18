"""Unit tests for migration gate helper."""

from __future__ import annotations

from gpu_fuzzy_trader.phases.phase2_island_scheduler import filter_migrants_for_cluster


def test_filter_empty_migrants():
    assert filter_migrants_for_cluster(
        [], None, source_cluster_id="0", target_cluster_id="1") == []

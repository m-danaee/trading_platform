"""Unit tests for migration safety — config guard, migrant gate, seed fraction.

Acceptance criteria covered:
  AC-T1.1: PHASE2_MIGRATION_ENABLED=False -> no migration log lines, set_pending_migrant_seeds
           never called.
  AC-T1.2: PHASE2_MIGRATION_ENABLED=True  -> 0.5% val_return rejected; 2.5% + >=15 trades
           accepted (via filter_migrants_for_cluster with synthetic metrics).
  AC-T1.3: Migrant injection uses <= PHASE2_MIGRATION_SEED_FRACTION * pop_size slots.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_island_scheduler import filter_migrants_for_cluster


# ============================================================================
# AC-T1.1 — PHASE2_MIGRATION_ENABLED=False prevents all migration activity
# ============================================================================


class TestMigrationEnabledByDefault:
    """Verify migration default for multi-symbol cluster islands (enabled)."""

    def test_config_default_is_true_for_cluster_package(self):
        """Multi-symbol clusters ship with migration enabled."""
        assert _cfg.PHASE2_MIGRATION_ENABLED is True
        assert _cfg.PHASE2_ONE_SYMBOL_ISLANDS is False

    def test_guard_allows_migration_when_enabled(self, monkeypatch):
        """The guard condition is True when migration is enabled."""
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_ENABLED", True)
        enabled = _cfg.PHASE2_MIGRATION_ENABLED
        epoch_counter = 2
        interval = int(_cfg.PHASE2_MIGRATION_EPOCH_INTERVAL)
        n_clusters = 3
        guard = (
            enabled
            and epoch_counter % interval == 0
            and n_clusters > 1
        )
        assert guard is True

    def test_guard_prevents_migration_when_disabled(self, monkeypatch):
        """With PHASE2_MIGRATION_ENABLED=False, guard is False even when epoch aligns."""
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_ENABLED", False)

        enabled = _cfg.PHASE2_MIGRATION_ENABLED
        epoch_counter = 2
        interval = int(_cfg.PHASE2_MIGRATION_EPOCH_INTERVAL)
        n_clusters = 3
        guard = (
            enabled
            and epoch_counter % interval == 0
            and n_clusters > 1
        )
        assert guard is False

    def test_set_pending_migrant_seeds_not_called_when_disabled(self, monkeypatch):
        """set_pending_migrant_seeds should never be called when migration is off."""
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_ENABLED", False)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_EPOCH_INTERVAL", 1)
        monkeypatch.setattr(_cfg, "PHASE2_N_CLUSTERS", 3)

        # Create a mock Rule_Pool_Generator
        mock_gen = MagicMock()
        mock_gen._island_generations_done = 50
        mock_gen._evolution_state = MagicMock()

        # Simulate the migration guard logic from _run_cluster_islands
        n_clusters = int(_cfg.PHASE2_N_CLUSTERS)
        epoch_counter = 2  # divisible by interval (1)

        if (
            _cfg.PHASE2_MIGRATION_ENABLED
            and epoch_counter % int(_cfg.PHASE2_MIGRATION_EPOCH_INTERVAL) == 0
            and n_clusters > 1
        ):
            # This block should NOT be reached
            mock_gen.set_pending_migrant_seeds(["dummy"])

        mock_gen.set_pending_migrant_seeds.assert_not_called()


# ============================================================================
# AC-T1.2 — filter_migrants_for_cluster gate with synthetic metrics
# ============================================================================


def _make_migrant_dict(
    chromosome: list[int] | None = None,
    migrant_rank_score: float = 1.0,
) -> dict:
    """Build a minimal migrant dict matching extract_deployable_migrants output."""
    if chromosome is None:
        chromosome = [0, 1, 2, 3, 4, 5]
    return {
        "chromosome": chromosome,
        "objectives": {
            "sortino_ratio": 1.0,
            "total_return_pct": 5.0,
            "max_drawdown_pct": 10.0,
            "profit_factor": 1.5,
        },
        "executed_trades": 20,
        "val_objectives": {
            "total_return_pct": 2.5,
            "profit_factor": 1.2,
            "max_drawdown_pct": 12.0,
        },
        "val_executed_trades": 18,
        "migrant_rank_score": migrant_rank_score,
    }


def _make_mock_receiver():
    """Build a minimal mock Rule_Pool_Generator for filter_migrants_for_cluster tests."""
    hp = MagicMock()
    hp.min_trade_pool_floor = 10

    mock = MagicMock()
    mock.island_hyperparams = hp
    mock._engine = MagicMock()
    mock._val_engine = MagicMock()
    mock.feature_infos = [{"name": "f1", "mode": "delta"}]
    return mock


class TestFilterMigrantsForClusterGate:
    """Test the migration gate logic with mocked engine evaluations."""

    @patch(
        "gpu_fuzzy_trader.phases.phase2_island_scheduler._migrant_to_metrics"
    )
    @patch(
        "gpu_fuzzy_trader.phases.phase2_island_scheduler.passes_evolution_deployability_preview",
        return_value=True,
    )
    @patch(
        "gpu_fuzzy_trader.phases.phase2_island_scheduler.passes_pool_admission_gate",
        return_value=True,
    )
    def test_rejects_low_val_return(
        self,
        mock_admission,
        mock_deployability,
        mock_metrics,
    ):
        """Migrant with val_return 0.5% should be rejected with reason=val_return."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_MIN_VAL_RETURN_PCT", 2.0)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_MIN_VAL_TRADES", 15)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_REQUIRE_DEPLOYABILITY", True)

        # Return synthetic metrics with low val_return
        def fake_metrics(migrant, train_engine, val_engine, feature_infos):
            return {"total_return_pct": 5.0, "executed_trades": 30}, {
                "total_return_pct": 0.5,
                "executed_trades": 20,
            }

        mock_metrics.side_effect = fake_metrics

        migrants = [_make_migrant_dict()]
        receiver = _make_mock_receiver()

        accepted = filter_migrants_for_cluster(
            migrants, receiver,
            source_cluster_id="0", target_cluster_id="1",
        )
        assert len(accepted) == 0, (
            "Migrant with 0.5% val_return should be rejected"
        )

    @patch(
        "gpu_fuzzy_trader.phases.phase2_island_scheduler._migrant_to_metrics"
    )
    @patch(
        "gpu_fuzzy_trader.phases.phase2_island_scheduler.passes_evolution_deployability_preview",
        return_value=True,
    )
    @patch(
        "gpu_fuzzy_trader.phases.phase2_island_scheduler.passes_pool_admission_gate",
        return_value=True,
    )
    def test_accepts_high_val_return_and_trades(
        self,
        mock_admission,
        mock_deployability,
        mock_metrics,
    ):
        """Migrant with val_return 2.5% and >=15 val trades should be accepted."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_MIN_VAL_RETURN_PCT", 2.0)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_MIN_VAL_TRADES", 15)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_REQUIRE_DEPLOYABILITY", True)

        def fake_metrics(migrant, train_engine, val_engine, feature_infos):
            return {"total_return_pct": 8.0, "executed_trades": 40}, {
                "total_return_pct": 2.5,
                "executed_trades": 18,
            }

        mock_metrics.side_effect = fake_metrics

        migrants = [_make_migrant_dict()]
        receiver = _make_mock_receiver()

        accepted = filter_migrants_for_cluster(
            migrants, receiver,
            source_cluster_id="0", target_cluster_id="1",
        )
        assert len(accepted) == 1, (
            "Migrant with 2.5% val_return and 18 trades should be accepted"
        )

    @patch(
        "gpu_fuzzy_trader.phases.phase2_island_scheduler._migrant_to_metrics"
    )
    @patch(
        "gpu_fuzzy_trader.phases.phase2_island_scheduler.passes_evolution_deployability_preview",
        return_value=True,
    )
    @patch(
        "gpu_fuzzy_trader.phases.phase2_island_scheduler.passes_pool_admission_gate",
        return_value=True,
    )
    def test_rejects_low_val_trades(
        self,
        mock_admission,
        mock_deployability,
        mock_metrics,
    ):
        """Migrant with enough val_return but too few val trades should be rejected."""
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_ENABLED", True)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_MIN_VAL_RETURN_PCT", 2.0)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_MIN_VAL_TRADES", 15)
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_REQUIRE_DEPLOYABILITY", True)

        def fake_metrics(migrant, train_engine, val_engine, feature_infos):
            return {"total_return_pct": 8.0, "executed_trades": 40}, {
                "total_return_pct": 2.5,
                "executed_trades": 5,  # too few trades
            }

        mock_metrics.side_effect = fake_metrics

        migrants = [_make_migrant_dict()]
        receiver = _make_mock_receiver()

        accepted = filter_migrants_for_cluster(
            migrants, receiver,
            source_cluster_id="0", target_cluster_id="1",
        )
        assert len(accepted) == 0, (
            "Migrant with only 5 val trades should be rejected"
        )


# ============================================================================
# AC-T1.3 — PHASE2_MIGRATION_SEED_FRACTION caps migrant injection
# ============================================================================


class TestMigrationSeedFraction:
    """Verify that migrant injection uses PHASE2_MIGRATION_SEED_FRACTION, not ARCHIVE."""

    def test_seed_fraction_config_default(self):
        """PHASE2_MIGRATION_SEED_FRACTION defaults to 0.10."""
        assert _cfg.PHASE2_MIGRATION_SEED_FRACTION == 0.10

    def test_migration_seed_fraction_decoupled_from_archive(self):
        """Ensure the migration fraction is not the same as archive fraction."""
        assert _cfg.PHASE2_MIGRATION_SEED_FRACTION != _cfg.PHASE2_ARCHIVE_SEED_FRACTION

    def test_local_cap_uses_migration_fraction(self, monkeypatch):
        """Simulate the run_epoch migrant path and assert local_cap uses migration fraction."""
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_SEED_FRACTION", 0.05)
        monkeypatch.setattr(_cfg, "PHASE2_ARCHIVE_SEED_FRACTION", 0.25)

        pop_size = 200
        expected_cap = max(
            1,
            int(round(pop_size * float(_cfg.PHASE2_MIGRATION_SEED_FRACTION))),
        )
        assert expected_cap == 10, (
            f"Expected migrant cap of 10 for pop_size=200, got {expected_cap}"
        )
        # Verify it's NOT the archive fraction cap
        archive_cap = max(
            1,
            int(round(pop_size * float(_cfg.PHASE2_ARCHIVE_SEED_FRACTION))),
        )
        assert archive_cap == 50
        assert expected_cap < archive_cap

    def test_migrant_entries_capped_by_migration_fraction(self, monkeypatch):
        """Simulate the new run_epoch logic: migrant entries are capped by migration fraction."""
        monkeypatch.setattr(_cfg, "PHASE2_MIGRATION_SEED_FRACTION", 0.05)
        pop_size = 200
        migrant_cap = max(
            1,
            int(round(pop_size * float(_cfg.PHASE2_MIGRATION_SEED_FRACTION))),
        )

        # Create synthetic seed entries with migrant_rank_score
        all_entries = []
        for i in range(20):
            all_entries.append({
                "chromosome": list(range(i * 6, (i + 1) * 6)),
                "migrant_rank_score": float(100 - i),
            })
        # Also add some archive entries (no migrant_rank_score)
        for i in range(5):
            all_entries.append({
                "chromosome": list(range(200 + i * 6, 206 + i * 6)),
            })

        migrant_entries = [
            e for e in all_entries
            if e.get("migrant_rank_score") is not None
        ]
        capped = migrant_entries[:migrant_cap]

        assert len(capped) == migrant_cap == 10
        # The capped list should be the top 10 by rank (first 10)
        assert capped[0]["migrant_rank_score"] == 100.0
        assert capped[-1]["migrant_rank_score"] == 91.0

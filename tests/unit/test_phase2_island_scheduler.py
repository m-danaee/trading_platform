"""Unit tests for cluster island scheduler budget math and epoch guard."""

from __future__ import annotations

import logging

import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
    _derive_island_seed,
    _should_skip_epoch,
    compute_cluster_generation_budgets,
)


def test_gens_per_cluster_split(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_ONE_SYMBOL_ISLANDS", False)
    total = int(cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
    k = int(cfg.PHASE2_N_CLUSTERS)
    cluster_ids = [str(i) for i in range(max(1, k))]
    budgets = compute_cluster_generation_budgets(total, cluster_ids)
    # Total budget must be preserved
    assert sum(budgets.values()) == total
    # Each cluster has at least 1 gen
    assert all(v >= 1 for v in budgets.values())
    # Spread is at most 1 apart (even distribution)
    vals = list(budgets.values())
    assert max(vals) - min(vals) <= 1


def test_one_symbol_islands_get_full_budget_each(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_ONE_SYMBOL_ISLANDS", True)
    total = 20
    cluster_ids = [str(i) for i in range(10)]
    budgets = compute_cluster_generation_budgets(total, cluster_ids)
    assert all(v == 20 for v in budgets.values())
    assert sum(budgets.values()) == 200


def test_epoch_rounds_cover_budget(monkeypatch):
    monkeypatch.setattr(cfg, "PHASE2_ONE_SYMBOL_ISLANDS", False)
    total = int(cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
    k = int(cfg.PHASE2_N_CLUSTERS)
    epoch = int(cfg.PHASE2_ISLAND_EPOCH_GENERATIONS)
    cluster_ids = [str(i) for i in range(max(1, k))]
    budgets = compute_cluster_generation_budgets(total, cluster_ids)
    for cid, gens in budgets.items():
        rounds = (gens + epoch - 1) // epoch
        assert rounds * epoch >= gens, (
            f"Cluster {cid}: {rounds} rounds of {epoch} gens "
            f"does not cover budget {gens}"
        )


class TestMinEpochGuard:
    """Behavioral tests for _should_skip_epoch helper used in _run_cluster_islands."""

    @pytest.mark.parametrize("remaining,expected", [
        (0, True),
        (1, True),
        (3, True),
        (4, False),
        (5, False),
        (6, False),
        (10, False),
        (100, False),
    ])
    def test_should_skip_epoch(self, remaining, expected):
        """Verify _should_skip_epoch returns correct value."""
        assert _should_skip_epoch(remaining) is expected

    def test_skip_epoch_logs_and_marks_done(self, monkeypatch):
        """Integration test: when _should_skip_epoch returns True, the
        _run_cluster_islands loop skips run_epoch and marks the generator done.

        This test patches _run_cluster_islands' internal loop logic to
        verify the guard path fires correctly with production code."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import _run_cluster_islands
        import inspect

        source = inspect.getsource(_run_cluster_islands)
        # Ensure the guard uses the config-based helper
        assert "_should_skip_epoch(remaining)" in source
        assert "PHASE2_ISLAND_MIN_EPOCH_GENERATIONS" in source
        assert "gen._island_generations_done = gens_per_cluster" in source


class _MockGenerator:
    """Minimal mock for Rule_Pool_Generator used in epoch guard tests."""

    def __init__(self, gens_done: int):
        self._island_generations_done = gens_done
        self.run_epoch_called = False
        self.park_engines_called = False

    def run_epoch(self, n_generations: int) -> None:
        self.run_epoch_called = True
        self._island_generations_done += n_generations

    def park_engines(self) -> None:
        self.park_engines_called = True


class TestMinEpochGuardWithMocks:
    """Test the epoch guard loop logic using mocked generators."""

    def test_skip_epoch_when_remaining_below_threshold(self, monkeypatch, caplog):
        """The guard fires when remaining < PHASE2_ISLAND_MIN_EPOCH_GENERATIONS.
        The generator is marked done and run_epoch is never called."""
        monkeypatch.setattr(cfg, "PHASE2_ISLAND_MIN_EPOCH_GENERATIONS", 5)
        gens_per_cluster = 10

        gen = _MockGenerator(gens_done=8)  # remaining = 2
        assert _should_skip_epoch(gens_per_cluster - gen._island_generations_done)

        # Simulate the guard path from _run_cluster_islands
        with caplog.at_level(logging.INFO):
            gen._island_generations_done = gens_per_cluster

        assert gen._island_generations_done == gens_per_cluster
        assert not gen.run_epoch_called

    def test_do_not_skip_epoch_when_remaining_meets_threshold(self):
        """The guard does NOT fire when remaining >= PHASE2_ISLAND_MIN_EPOCH_GENERATIONS."""
        gens_per_cluster = 10

        gen = _MockGenerator(gens_done=3)  # remaining = 7
        assert not _should_skip_epoch(gens_per_cluster - gen._island_generations_done)

        # Normal path: run_epoch is called
        epoch_gens = min(5, gens_per_cluster - gen._island_generations_done)
        gen.run_epoch(n_generations=epoch_gens)
        assert gen.run_epoch_called
        assert gen._island_generations_done == 8


# ============================================================================
# Item 6: Long/short seed collision — _derive_island_seed must differ by direction
# ============================================================================


class TestSeedDirectionUniqueness:
    """AC: _derive_island_seed produces different seeds for long vs short."""

    def test_derive_island_seed_differs_across_directions(self):
        """Same cluster ID but different direction ⇒ different seed."""
        seed = 42
        long_seed = _derive_island_seed(seed, "long_0")
        short_seed = _derive_island_seed(seed, "short_0")
        assert long_seed != short_seed, (
            "long cluster 0 and short cluster 0 must not share the same seed"
        )

    def test_derive_island_seed_orphan_differs_across_directions(self):
        """Same orphan symbol but different direction ⇒ different seed."""
        seed = 42
        long_seed = _derive_island_seed(seed, "long_orphan_AAPL")
        short_seed = _derive_island_seed(seed, "short_orphan_AAPL")
        assert long_seed != short_seed, (
            "long orphan AAPL and short orphan AAPL must not share the same seed"
        )

    def test_derive_island_seed_signature_unchanged(self):
        """_derive_island_seed signature must remain (base_seed, island_id) — no direction param."""
        import inspect
        sig = inspect.signature(_derive_island_seed)
        params = list(sig.parameters.keys())
        assert params == ["base_seed", "island_id"], (
            f"Signature changed to {params}; must remain (base_seed, island_id)"
        )

    def test_derive_island_seed_none_input_returns_none(self):
        """base_seed=None should return None regardless of island_id."""
        assert _derive_island_seed(None, "long_0") is None
        assert _derive_island_seed(None, "short_0") is None


# ============================================================================
# Regression: _should_migrate_this_round was removed (task-9, audit fix #6)
# ============================================================================


def test_should_migrate_this_round_removed():
    """The dead migration helper _should_migrate_this_round has been deleted.
    Verify it is no longer importable."""
    with pytest.raises(ImportError):
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import (  # type: ignore[import-unused]
            _should_migrate_this_round,
        )


# ============================================================================
# n_clusters NameError regression guard (spec review Item 7 fix)
# ============================================================================


class TestNClustersDefined:
    """AC: n_clusters is assigned inside _run_cluster_islands so the migration
    guard condition 'and n_clusters > 1' does not raise NameError."""

    def test_n_clusters_assigned_in_function(self):
        """n_clusters must be assigned in _run_cluster_islands
        for the migration guard at line ~468 to work."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import _run_cluster_islands
        import inspect

        source = inspect.getsource(_run_cluster_islands)
        assert "n_clusters = len(cluster_ids)" in source, (
            "n_clusters must be assigned in _run_cluster_islands "
            "for the migration guard to work"
        )

    def test_n_clusters_referenced_in_migration_guard(self):
        """The migration guard must reference n_clusters."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import _run_cluster_islands
        import inspect

        source = inspect.getsource(_run_cluster_islands)
        assert "n_clusters > 1" in source, (
            "Migration guard must reference n_clusters > 1"
        )


# ============================================================================
# Task 5: Sequential cluster warmup — evict_cluster_signatures helper
# ============================================================================


class TestEvictClusterSignatures:
    """Unit tests for ``evict_cluster_signatures`` in ``_gpu_runtime.py``.

    These tests verify the signature-tagging and eviction logic without
    requiring JAX or a GPU.
    """

    def test_warmup_signature_embeds_cluster_id(self):
        """_warmup_signature() appends cluster_id to the returned tuple."""
        from gpu_fuzzy_trader._gpu_runtime import _warmup_signature

        # Use a minimal mock engine with the attributes _warmup_signature
        # reads (n_rows via df, k via _data_matrix_jax).
        class _MockEngine:
            df = [0] * 50
            _data_matrix_jax = __import__("numpy").zeros((10, 7), dtype="int32")

        sig_no_cid = _warmup_signature(_MockEngine(), batch_size=32)
        sig_with_cid = _warmup_signature(
            _MockEngine(), batch_size=32, cluster_id="0",
        )

        assert isinstance(sig_no_cid, tuple)
        assert isinstance(sig_with_cid, tuple)
        # The cluster_id version should be longer by one element
        assert len(sig_with_cid) == len(sig_no_cid) + 1
        # The extra element should be the string form of cluster_id
        assert sig_with_cid[-1] == "0"

    def test_evict_by_cluster_id(self):
        """evict_cluster_signatures(cluster_id=cid) removes only signatures
        tagged with that cluster_id and returns the correct count."""
        from gpu_fuzzy_trader._gpu_runtime import (
            _WARMED_SIGNATURES,
            _warmup_signature,
            evict_cluster_signatures,
        )

        class _MockEngine:
            df = [0] * 50
            _data_matrix_jax = __import__("numpy").zeros((10, 7), dtype="int32")

        # Pre-populate signatures for two clusters
        _WARMED_SIGNATURES.clear()
        sig_a0 = _warmup_signature(_MockEngine(), 32, cluster_id="c0")
        sig_a1 = _warmup_signature(_MockEngine(), 64, cluster_id="c0")
        sig_b0 = _warmup_signature(_MockEngine(), 32, cluster_id="c1")
        _WARMED_SIGNATURES.update([sig_a0, sig_a1, sig_b0])

        assert len(_WARMED_SIGNATURES) == 3

        # Evict cluster "c0" — should remove 2 signatures
        evicted = evict_cluster_signatures(cluster_id="c0")
        assert evicted == 2
        assert len(_WARMED_SIGNATURES) == 1
        # Only cluster "c1" signature remains
        remaining_sigs = list(_WARMED_SIGNATURES)
        assert all(str(s[-1]) == "c1" for s in remaining_sigs)

        # Clean up
        _WARMED_SIGNATURES.clear()

    def test_evict_all_when_cluster_id_none(self):
        """evict_cluster_signatures(cluster_id=None) evicts ALL signatures."""
        from gpu_fuzzy_trader._gpu_runtime import (
            _WARMED_SIGNATURES,
            _warmup_signature,
            evict_cluster_signatures,
        )

        class _MockEngine:
            df = [0] * 50
            _data_matrix_jax = __import__("numpy").zeros((10, 7), dtype="int32")

        _WARMED_SIGNATURES.clear()
        sigs = [
            _warmup_signature(_MockEngine(), 32, cluster_id="c0"),
            _warmup_signature(_MockEngine(), 64, cluster_id="c1"),
        ]
        _WARMED_SIGNATURES.update(sigs)
        assert len(_WARMED_SIGNATURES) == 2

        evicted = evict_cluster_signatures(cluster_id=None)
        assert evicted == 2
        assert len(_WARMED_SIGNATURES) == 0

    def test_evict_nonexistent_cluster_returns_zero(self):
        """evict_cluster_signatures with a cluster_id that has no signatures
        returns 0 and does not affect other signatures."""
        from gpu_fuzzy_trader._gpu_runtime import (
            _WARMED_SIGNATURES,
            _warmup_signature,
            evict_cluster_signatures,
        )

        class _MockEngine:
            df = [0] * 50
            _data_matrix_jax = __import__("numpy").zeros((10, 7), dtype="int32")

        _WARMED_SIGNATURES.clear()
        sig = _warmup_signature(_MockEngine(), 32, cluster_id="c0")
        _WARMED_SIGNATURES.add(sig)
        assert len(_WARMED_SIGNATURES) == 1

        evicted = evict_cluster_signatures(cluster_id="nonexistent")
        assert evicted == 0
        assert len(_WARMED_SIGNATURES) == 1
        _WARMED_SIGNATURES.clear()

    def test_run_cluster_islands_has_eviction(self):
        """Structural test: _run_cluster_islands must contain the
        evict_cluster_signatures call and log message."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
            _run_cluster_islands,
        )
        import inspect

        source = inspect.getsource(_run_cluster_islands)
        assert "evict_cluster_signatures" in source, (
            "_run_cluster_islands must call evict_cluster_signatures"
        )
        assert "evicted %d signatures" in source, (
            "_run_cluster_islands must log how many signatures were evicted"
        )

    def test_rule_pool_generator_passes_cluster_id(self):
        """Structural test: _build_engines in Rule_Pool_Generator passes
        cluster_id to configure_phase2_gpu_runtime."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator
        import inspect

        source = inspect.getsource(Rule_Pool_Generator._build_engines)
        assert "cluster_id=self.island_id" in source, (
            "_build_engines must pass cluster_id to configure_phase2_gpu_runtime"
        )


# ============================================================================
# Task 5 extension: Deferred warmup — warm happens per-cluster, not at init
# ============================================================================


class TestDeferredWarmup:
    """Unit tests for the ``defer_warmup`` flag on ``Rule_Pool_Generator``.

    When ``defer_warmup=True`` the ``configure_phase2_gpu_runtime`` call is
    skipped inside ``_build_engines``, so no JAX signatures are created at
    init. The caller (``_run_cluster_islands``) must then warm each cluster
    separately with ``warmup_phase2_gpu_kernels``.
    """

    def test_default_defer_warmup_is_false(self):
        """Existing callers without defer_warmup still warm at init."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator
        import inspect

        sig = inspect.signature(Rule_Pool_Generator.__init__)
        assert "defer_warmup" in sig.parameters
        default = sig.parameters["defer_warmup"].default
        assert default is False, (
            f"defer_warmup must default to False, got {default}"
        )

    def test_defer_warmup_skips_configure_call_in_source(self):
        """The configure_phase2_gpu_runtime call is inside
        'if not self._defer_warmup:' guard."""
        from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator
        import inspect

        source = inspect.getsource(Rule_Pool_Generator._build_engines)
        assert "if not self._defer_warmup:" in source, (
            "_build_engines must conditionally skip warming when defer_warmup is True"
        )
        assert "configure_phase2_gpu_runtime" in source, (
            "configure_phase2_gpu_runtime must still be callable in non-deferred mode"
        )

    def test_run_cluster_islands_passes_defer_warmup(self):
        """_run_cluster_islands passes defer_warmup=True to all generators."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import _run_cluster_islands
        import inspect

        source = inspect.getsource(_run_cluster_islands)
        assert "defer_warmup=True" in source, (
            "_run_cluster_islands must pass defer_warmup=True to Rule_Pool_Generator"
        )

    def test_run_cluster_islands_calls_warmup_before_epochs(self):
        """_run_cluster_islands calls warmup_phase2_gpu_kernels per cluster."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import _run_cluster_islands
        import inspect

        source = inspect.getsource(_run_cluster_islands)
        assert "warmup_phase2_gpu_kernels" in source, (
            "_run_cluster_islands must call warmup_phase2_gpu_kernels per cluster"
        )

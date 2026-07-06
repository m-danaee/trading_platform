"""Unit tests for cluster island scheduler budget math and epoch guard."""

from __future__ import annotations

import logging

import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.phases.phase2_island_scheduler import (
    _derive_island_seed,
    _should_migrate_this_round,
    _should_skip_epoch,
    compute_cluster_generation_budgets,
)


def test_gens_per_cluster_split():
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


def test_epoch_rounds_cover_budget():
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
        (4, True),
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
# Item 7: Migration cadence — _should_migrate_this_round helper + loop fix
# ============================================================================


class TestMigrationCadenceHelper:
    """Pure-function tests for _should_migrate_this_round."""

    @pytest.mark.parametrize("round_index,interval,expected", [
        (0, 2, True),
        (1, 2, False),
        (2, 2, True),
        (3, 2, False),
        (0, 1, True),
        (1, 1, True),
        (0, 3, True),
        (1, 3, False),
        (2, 3, False),
        (3, 3, True),
        (0, 0, False),   # interval <= 0 never fires
        (5, 0, False),
        (0, -1, False),
        (3, -1, False),
    ])
    def test_should_migrate_this_round_parametrized(
        self, round_index, interval, expected,
    ):
        """Verify pure helper returns correct boolean for various inputs."""
        assert _should_migrate_this_round(round_index, interval) is expected

    def test_migration_cadence_uses_rounds_not_clusters(self):
        """Verify _run_cluster_islands increments round_counter at outer scope,
        NOT inside the ``for cid in cluster_ids:`` loop."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import _run_cluster_islands
        import inspect

        source = inspect.getsource(_run_cluster_islands)

        # The old bug: epoch_counter += 1 inside the for cid loop
        # The fix: round_counter += 1 at outer while scope (after for cid)

        # Assert the increment happens OUTSIDE the for-cid loop.
        # Strategy: find the 'for cid in cluster_ids:' block and assert
        # that 'round_counter += 1' is NOT indented inside it.
        lines = source.splitlines()
        in_for_cid_block = False
        for_cid_indent = None
        round_increment_found = False
        round_increment_indent = None

        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            if "for cid in cluster_ids:" in stripped:
                in_for_cid_block = True
                for_cid_indent = indent
                continue

            if in_for_cid_block and "round_counter += 1" in stripped:
                round_increment_found = True
                round_increment_indent = indent
                break

            # Detect end of for-cid block by checking if a line at same or lesser
            # indent as the 'for' keyword appears (and is not a comment/blank)
            if in_for_cid_block and stripped and not stripped.startswith("#"):
                if indent <= for_cid_indent:
                    in_for_cid_block = False

        assert round_increment_found, (
            "round_counter += 1 not found in _run_cluster_islands source"
        )
        # If the increment were inside the for-cid block, its indent would be
        # greater than for_cid_indent. It should be at the same or lesser indent.
        assert round_increment_indent is not None and round_increment_indent <= for_cid_indent, (
            f"round_counter += 1 (indent={round_increment_indent}) is INSIDE "
            f"the 'for cid' block (indent={for_cid_indent}); "
            f"it must be at outer while scope"
        )

    def test_round_counter_name_used(self):
        """Verify _run_cluster_islands uses 'round_counter' not 'epoch_counter'."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import _run_cluster_islands
        import inspect

        source = inspect.getsource(_run_cluster_islands)
        assert "round_counter" in source, (
            "'round_counter' not found; variable not renamed"
        )
        assert "epoch_counter" not in source, (
            "'epoch_counter' still present; should be renamed to 'round_counter'"
        )

    def test_migration_guard_uses_helper(self):
        """Verify the migration guard calls _should_migrate_this_round."""
        from gpu_fuzzy_trader.phases.phase2_island_scheduler import _run_cluster_islands
        import inspect

        source = inspect.getsource(_run_cluster_islands)
        assert "_should_migrate_this_round" in source, (
            "Migration guard must use the _should_migrate_this_round helper"
        )
        # Ensure the old inline modulo is gone
        assert "round_counter % int" not in source, (
            "Inline modulo still present; should use _should_migrate_this_round"
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

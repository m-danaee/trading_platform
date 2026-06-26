"""Regression tests for plateau-state leak fixes (Fixes A + B).

Fix A: reset_plateau is always True (per-epoch plateau reset).
Fix B: _island_generations_done charges len(epoch_history), not epoch_gens.

These tests use object.__new__() to avoid GPU/engine dependencies, and
monkeypatch the internal run_phase2_evolution_epoch to capture call args.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.phases.phase2_rule_pool import Rule_Pool_Generator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_minimal_gen() -> Rule_Pool_Generator:
    """Build a Rule_Pool_Generator with minimal attributes, no engine init."""
    gen = object.__new__(Rule_Pool_Generator)
    gen.direction = "long"
    gen.feature_infos = [
        {"name": "feat_0", "mode": "binary", "score": 0.5},
    ]
    gen.pop_size = 10
    gen.n_generations = 100
    gen.seed = 42
    gen._feature_signature = "test_sig"
    gen._evolution_state = None
    gen._island_history = []
    gen._island_generations_done = 0
    gen.island_id = "test_island"
    gen.source_symbols = []
    gen.island_hyperparams = None
    gen.island_profile = "cluster_0"
    gen.reference_rows = None
    gen._pending_migrant_seeds = []
    gen._feature_names = ["feat_0"]
    gen._feature_modes = {"feat_0": "discrete"}
    gen._engine = None
    gen._val_engine = None
    gen._ensure_engines = lambda: None  # no-op: no GPU needed
    return gen


def _mock_evolution_state(
    gen: Rule_Pool_Generator,
    plateau_streak: int = 5,
    plateau_best_progress: float = 0.5,
    n_history_gens: int = 0,
) -> object:
    """Build a minimal Phase2EvolutionState-like object for testing."""
    from gpu_fuzzy_trader.evolution.evox_runner import Phase2EvolutionState

    state = Phase2EvolutionState.__new__(Phase2EvolutionState)
    state.plateau_streak = plateau_streak
    state.plateau_best_progress = plateau_best_progress
    state.population = np.zeros((gen.pop_size, 2), dtype=np.int32)
    state.objectives = np.full((gen.pop_size, 3), np.inf)
    state.metrics_cache = [{} for _ in range(gen.pop_size)]
    state.pareto_archive = []
    state.hall_of_fame = {}
    state.deployable_archive = {}
    state.global_metrics_cache = {}
    state.history = [{"gen": i} for i in range(n_history_gens)]
    state.ref_vec = None
    state.generation_offset = 0
    state.mutation_rate = 0.5
    state.weighted_activate_prob = 0.5
    state.stage = None
    return state


def _mock_stage_plan(
    remaining_in_stage: int = 100,
    entering_stage_b: bool = False,
    two_stage_active: bool = False,
    stage: str | None = None,
) -> object:
    """Build a minimal IslandStagePlan-like object."""
    return type(
        "StagePlan",
        (),
        {
            "remaining_in_stage": remaining_in_stage,
            "entering_stage_b": entering_stage_b,
            "two_stage_active": two_stage_active,
            "stage": stage,
        },
    )()


# ---------------------------------------------------------------------------
# Fix A - reset_plateau is always True
# ---------------------------------------------------------------------------


class TestResetPlateau:
    """AC-1, AC-3: plateau_streak resets per epoch; reset_plateau=True always."""

    def test_reset_plateau_is_true_for_first_epoch(self, monkeypatch):
        """reset_plateau=True is passed on the very first epoch call."""
        gen = _make_minimal_gen()
        captured = {}

        def mock_evolution(*args, **kwargs):
            captured["reset_plateau"] = kwargs.get("reset_plateau")
            state = _mock_evolution_state(gen)
            return state, [{"gen": i} for i in range(3)]

        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution_epoch",
            mock_evolution,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_stage.resolve_island_stage",
            lambda *a, **kw: _mock_stage_plan(),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
            lambda *a, **kw: np.array([0.5, 0.5]),
        )

        gen.run_epoch(n_generations=10)
        assert captured.get("reset_plateau") is True, (
            f"Expected reset_plateau=True, got {captured.get('reset_plateau')}"
        )

    def test_reset_plateau_is_true_for_second_epoch(self, monkeypatch):
        """reset_plateau=True is also passed on epoch 2 (regression for leak)."""
        gen = _make_minimal_gen()
        call_count = [0]
        captured = {}

        def mock_evolution(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                captured["reset_plateau"] = kwargs.get("reset_plateau")
            state = _mock_evolution_state(gen)
            return state, [{"gen": i} for i in range(3)]

        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution_epoch",
            mock_evolution,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_stage.resolve_island_stage",
            lambda *a, **kw: _mock_stage_plan(remaining_in_stage=90),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
            lambda *a, **kw: np.array([0.5, 0.5]),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.extract_deployable_migrants",
            lambda *a, **kw: [],
        )

        gen.run_epoch(n_generations=10)
        gen.run_epoch(n_generations=10)

        assert captured.get("reset_plateau") is True, (
            f"Epoch 2: expected reset_plateau=True, got "
            f"{captured.get('reset_plateau')}"
        )

    def test_reset_plateau_true_overrides_two_stage_disabled(self, monkeypatch):
        """Even when PHASE2_ISLAND_TWO_STAGE_ENABLED=False, reset_plateau=True."""
        monkeypatch.setattr(cfg, "PHASE2_ISLAND_TWO_STAGE_ENABLED", False)
        gen = _make_minimal_gen()
        captured = {}

        def mock_evolution(*args, **kwargs):
            captured["reset_plateau"] = kwargs.get("reset_plateau")
            state = _mock_evolution_state(gen)
            return state, [{"gen": i} for i in range(3)]

        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution_epoch",
            mock_evolution,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_stage.resolve_island_stage",
            lambda *a, **kw: _mock_stage_plan(),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
            lambda *a, **kw: np.array([0.5, 0.5]),
        )

        gen.run_epoch(n_generations=10)
        assert captured.get("reset_plateau") is True, (
            f"Expected reset_plateau=True even with two-stage disabled, got "
            f"{captured.get('reset_plateau')}"
        )


# ---------------------------------------------------------------------------
# Fix B - _island_generations_done charges actual generations
# ---------------------------------------------------------------------------


class TestIslandGenerationsDone:
    """AC-2: _island_generations_done increments by len(epoch_history)."""

    def test_charges_actual_gens_when_epoch_fully_used(self, monkeypatch):
        """When all requested gens execute, len(epoch_history) == epoch_gens."""
        gen = _make_minimal_gen()

        def mock_evolution(*args, **kwargs):
            state = _mock_evolution_state(gen, plateau_streak=0, plateau_best_progress=-np.inf)
            return state, [{"gen": i} for i in range(10)]

        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution_epoch",
            mock_evolution,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_stage.resolve_island_stage",
            lambda *a, **kw: _mock_stage_plan(),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
            lambda *a, **kw: np.array([0.5, 0.5]),
        )

        gen.run_epoch(n_generations=10)
        assert gen._island_generations_done == 10, (
            f"Expected 10 gens done (full epoch), got "
            f"{gen._island_generations_done}"
        )

    def test_charges_actual_gens_when_early_stop(self, monkeypatch):
        """Early-stop: 3 actual gens run out of 10 requested, budget += 3."""
        gen = _make_minimal_gen()

        def mock_evolution(*args, **kwargs):
            state = _mock_evolution_state(gen, n_history_gens=3)
            return state, [{"gen": i} for i in range(3)]

        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution_epoch",
            mock_evolution,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_stage.resolve_island_stage",
            lambda *a, **kw: _mock_stage_plan(),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
            lambda *a, **kw: np.array([0.5, 0.5]),
        )

        gen.run_epoch(n_generations=10)
        assert gen._island_generations_done == 3, (
            f"Expected 3 gens done (early-stop at gen 3), got "
            f"{gen._island_generations_done}"
        )

    def test_accumulates_across_epochs_with_early_stop(self, monkeypatch):
        """Two epochs with early-stop: 3 + 5 = 8 gens done total."""
        gen = _make_minimal_gen()
        call_count = [0]

        def mock_evolution(*args, **kwargs):
            call_count[0] += 1
            n_gens = 3 if call_count[0] == 1 else 5
            state = _mock_evolution_state(gen, n_history_gens=n_gens)
            return state, [{"gen": i} for i in range(n_gens)]

        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution_epoch",
            mock_evolution,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_stage.resolve_island_stage",
            lambda *a, **kw: _mock_stage_plan(remaining_in_stage=90),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
            lambda *a, **kw: np.array([0.5, 0.5]),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.extract_deployable_migrants",
            lambda *a, **kw: [],
        )

        gen.run_epoch(n_generations=10)
        gen.run_epoch(n_generations=10)
        assert gen._island_generations_done == 8, (
            f"Expected 8 gens done across two early-stop epochs (3+5), got "
            f"{gen._island_generations_done}"
        )


# ---------------------------------------------------------------------------
# AC-4: Stage B seeding / migration unchanged
# ---------------------------------------------------------------------------


def test_stage_b_seeding_still_uses_entering_stage_b(monkeypatch):
    """The entering_stage_b variable still controls seed logic (not reset_plateau).

    Verifies that reset_plateau=True does NOT break the existing
    entering_stage_b seeding path.
    """
    gen = _make_minimal_gen()
    # Need a non-None _evolution_state for entering_stage_b seed logic
    gen._evolution_state = _mock_evolution_state(gen)

    monkeypatch.setattr(
        "gpu_fuzzy_trader.phases.phase2_stage.resolve_island_stage",
        lambda *a, **kw: _mock_stage_plan(
            remaining_in_stage=50,
            entering_stage_b=True,
            two_stage_active=True,
            stage="B",
        ),
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
        lambda *a, **kw: np.array([0.5, 0.5]),
    )

    # Dont-care mask with all 0s so seeds pass through
    monkeypatch.setattr(
        "gpu_fuzzy_trader.phases.phase2_rule_pool._get_dont_cares",
        lambda *a, **kw: np.array([0, 0], dtype=np.int32),
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.phases.phase2_rule_pool._deployable_archive_pool_entries",
        lambda *a, **kw: [],
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.phases.phase2_rule_pool._stage_b_seed_chromosomes",
        lambda *a, **kw: np.array([[0, 0], [1, 1]], dtype=np.int32),
    )

    def mock_evolution(*args, **kwargs):
        state = _mock_evolution_state(gen, plateau_streak=0, plateau_best_progress=-np.inf)
        return state, [{"gen": i} for i in range(3)]

    monkeypatch.setattr(
        "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution_epoch",
        mock_evolution,
    )
    monkeypatch.setattr(
        "gpu_fuzzy_trader.evolution.evox_runner.extract_deployable_migrants",
        lambda *a, **kw: [],
    )

    try:
        gen.run_epoch(n_generations=10)
    except Exception as e:
        pytest.fail(f"run_epoch raised unexpectedly with Stage B: {e}")


# ---------------------------------------------------------------------------
# Source-level assertion: reset_plateau is a constant True (AC-3)
# ---------------------------------------------------------------------------


def test_reset_plateau_is_literal_true():
    """The source file has 'reset_plateau = True' (not a variable)."""
    import ast
    import inspect
    import textwrap

    source = inspect.getsource(Rule_Pool_Generator.run_epoch)
    dedented = textwrap.dedent(source)
    tree = ast.parse(dedented)

    class ResetPlateauFinder(ast.NodeVisitor):
        def __init__(self):
            self.found_literal_true = False

        def visit_Assign(self, node):
            if (
                len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "reset_plateau"
            ):
                if isinstance(node.value, ast.Constant) and node.value.value is True:
                    self.found_literal_true = True
            self.generic_visit(node)

    finder = ResetPlateauFinder()
    finder.visit(tree)
    assert finder.found_literal_true, (
        "reset_plateau must be assigned the literal True - "
        "found variable or expression instead"
    )

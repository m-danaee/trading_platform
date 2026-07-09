"""Unit tests for the Phase 2 monthly-window shadow-test gate (Task 13).

Tests ``_apply_monthly_admission_gate`` directly by monkeypatching
``_evaluate_rule_on_window`` to return deterministic values,
so no real backtest engine or data is needed.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    _apply_monthly_admission_gate,
    Rule_Pool_Generator,
)


# ---------------------------------------------------------------------------
# Deterministic evaluator monkeypatch helper
# ---------------------------------------------------------------------------


class _DeterministicEvaluator:
    """Maps (rule_index, window_index) → ``total_return_pct``.

    Usage in tests::

        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns_by_rule),
        )

    Each call advances an internal counter; *returns_by_rule[r][w]* is returned
    for the *r*-th rule on the *w*-th window (rules evaluated sequentially,
    all windows per rule before moving to the next rule).
    """

    def __init__(self, returns_by_rule: list[list[float]]):
        self.returns_by_rule = returns_by_rule
        self.call_count = 0

    def __call__(
        self, pool_entry: dict, window_df: object, direction: str,
    ) -> float:
        n_windows = len(self.returns_by_rule[0]) if self.returns_by_rule else 1
        rule_idx = self.call_count // n_windows
        window_idx = self.call_count % n_windows
        self.call_count += 1
        if rule_idx >= len(self.returns_by_rule):
            return 0.0
        return float(self.returns_by_rule[rule_idx][window_idx])


# ---------------------------------------------------------------------------
# Shared fixture: a pool of 3 rules
# ---------------------------------------------------------------------------

POOL_THREE = [
    {
        "conditions": ["[feat_a] IS high", "symbol is 1"],
        "chromosome": [0, 1, 1],
        "objectives": {"total_return_pct": 5.0, "profit_factor": 1.5},
        "executed_trades": 50,
    },
    {
        "conditions": ["[feat_b] IS low", "symbol is 2"],
        "chromosome": [1, 0, 1],
        "objectives": {"total_return_pct": 2.0, "profit_factor": 1.2},
        "executed_trades": 40,
    },
    {
        "conditions": ["[feat_c] IS medium", "symbol is 3"],
        "chromosome": [2, 2, 2],
        "objectives": {"total_return_pct": -3.0, "profit_factor": 0.8},
        "executed_trades": 30,
    },
]


# ===========================================================================
# Tests
# ===========================================================================


class TestMonthlyAdmissionGate:
    """Verify the gate keeps / rejects rules based on profitable_ratio."""

    @pytest.fixture(autouse=True)
    def _gate_thresholds(self, monkeypatch):
        """Pin thresholds so tests assert gate logic, not current config churn."""
        monkeypatch.setattr(_cfg, "PHASE2_MONTHLY_ADMISSION_MIN_RATIO", 0.5)
        monkeypatch.setattr(_cfg, "PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT", 0.0)

    # ------------------------------------------------------------------
    # Test 1: basic profitability filtering
    # ------------------------------------------------------------------

    def test_keeps_profitable_and_half_profitable(self, monkeypatch):
        """Rule 0 profitable on all 6 months (ratio=1.0) passes at >=0.5;
        rule 1 on half (ratio=0.5) also passes at >=0.5; rule 2 (ratio=0.0)
        dropped."""
        returns = [
            [1.0, 0.5, 2.0, 1.5, 0.8, 1.2],    # rule 0: 6/6 = 1.0
            [1.0, -0.5, 2.0, -1.0, 0.5, -0.3],  # rule 1: 3/6 = 0.5
            [-1.0, -2.0, -0.5, -1.5, -3.0, -0.8],  # rule 2: 0/6 = 0.0
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        # Rule 0 (1.0) passes >=0.5; rule 1 (0.5) passes >=0.5; rule 2 (0.0) dropped.
        assert len(result) == 2
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]
        assert result[1]["conditions"] == POOL_THREE[1]["conditions"]

    # ------------------------------------------------------------------
    # Test 2: all rules just below threshold are rejected,
    #         triggering graceful degradation
    # ------------------------------------------------------------------

    def test_rejects_below_threshold_graceful_degradation(self, monkeypatch):
        """All rules have ratio < 0.5 → gate empties pool → graceful
        degradation keeps original pool."""
        returns = [
            [-1.0, -0.5, -2.0, -1.5, -0.8, -1.2],  # rule 0: 0/6 = 0.0
            [-1.0, -0.5, -2.0, -1.0, -0.5, -0.3],  # rule 1: 0/6 = 0.0
            [-1.0, -2.0, -0.5, -1.5, -3.0, -0.8],  # rule 2: 0/6 = 0.0
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        # Graceful degradation: original pool returned (3 rules)
        assert len(result) == 3
        assert result == POOL_THREE

    # ------------------------------------------------------------------
    # Test 3: boundary — ratio exactly at 0.5 threshold
    # ------------------------------------------------------------------

    def test_boundary_exact_half_profitable(self, monkeypatch):
        """Rule with exactly 3/6 = 0.5 passes at >=0.5 threshold."""
        returns = [
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],  # rule 0: 3/6 = 0.5 (passes)
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],  # rule 1: 0/6
            [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],  # rule 2: 0/6
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        # Rule 0 (0.5) passes at >=0.5; rules 1+2 rejected.
        assert len(result) == 1
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]

    # ------------------------------------------------------------------
    # Test 4: boundary — ratio just below 0.5 threshold
    # ------------------------------------------------------------------

    def test_boundary_just_below_threshold(self, monkeypatch):
        """Rule with 2/6 ≈ 0.33 is below 0.5 threshold; rule with 3/6 = 0.5 passes."""
        returns = [
            [1.0, -1.0, 1.0, -1.0, -1.0, -1.0],  # rule 0: 2/6 ≈ 0.33 (rejected)
            [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],   # rule 1: 3/6 = 0.5 (passes)
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],  # rule 2: 0/6
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        # Rule 1 (0.5) passes; rule 0 rejected; rule 2 rejected.
        assert len(result) == 1
        assert result[0]["conditions"] == POOL_THREE[1]["conditions"]

    def test_zero_threshold_strict_profit_excludes_flat_months(
        self, monkeypatch,
    ) -> None:
        """Phase 2 at min=0 uses strict >0; flat months do not count."""
        returns = [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],  # rule 0: 1/6 strict profit
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )
        monkeypatch.setattr(_cfg, "PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT", 0.0)

        result = _apply_monthly_admission_gate(
            POOL_THREE, list(range(6)), "long")
        assert len(result) == 3
        assert result == POOL_THREE

    def test_positive_threshold_requires_min_return(self, monkeypatch) -> None:
        """With PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT=2, months need return >= 2%.
        Rule 0 has 5/6 >= 2% (ratio=0.833) which passes 0.667; rule 1,2 fail."""
        returns = [
            [3.0, 2.0, 2.0, 2.5, 2.0, 0.0],  # rule 0: 5/6 >= 2% (0.833)
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )
        monkeypatch.setattr(_cfg, "PHASE2_MONTHLY_GOOD_RETURN_MIN_PCT", 2.0)

        result = _apply_monthly_admission_gate(
            POOL_THREE, list(range(6)), "long")
        assert len(result) == 1
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]

    def test_multiple_rules_pass_gate(self, monkeypatch):
        """Rules with >= 0.5 ratio are kept; rule below 0.5 threshold dropped."""
        returns = [
            [1.0, 0.5, 2.0, 1.5, 0.8, 1.2],    # rule 0: 6/6 = 1.0 (passes)
            [1.0, -0.5, 2.0, -1.0, 0.5, -1.0],  # rule 1: 3/6 = 0.5 (passes)
            [1.0, -0.5, -2.0, -1.0, -0.5, -0.3],  # rule 2: 1/6 ≈ 0.17 (rejected)
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        result = _apply_monthly_admission_gate(POOL_THREE, list(range(6)), "long")
        assert len(result) == 2
        assert result[0]["conditions"] == POOL_THREE[0]["conditions"]
        assert result[1]["conditions"] == POOL_THREE[1]["conditions"]


# ===========================================================================
# Tests: gate DataFrame source — verifies the fix for audit finding #4
# (monthly gate used train, making it a near no-op; now uses val).
# ===========================================================================


class TestMonthlyGateDataSource:
    """Verify the gate sources its DataFrame from ``_cached_slim_val`` (val,
    not train) and handles the None-case gracefully."""

    # ------------------------------------------------------------------
    # Test A: gate uses cached_slim_val (val DataFrame)
    # ------------------------------------------------------------------

    def _make_train_df(self, n_rows: int = 200) -> pd.DataFrame:
        """Create a minimal training DataFrame similar to test_phase2_rule_pool."""
        rng = np.random.default_rng(42)
        data = {
            "datetime": pd.date_range("2020-01-01", periods=n_rows, freq="D"),
            "symbol": "A",
            "label_open_next": 100.0 + np.arange(n_rows, dtype=float),
            "label_close_288": 100.0 + np.arange(n_rows, dtype=float),
            "label_min_288": 99.0 + np.arange(n_rows, dtype=float),
            "label_max_288": 101.0 + np.arange(n_rows, dtype=float),
            "label_max_before_min": np.ones(n_rows, dtype=float),
            "_symbol_bar_index": np.arange(n_rows),
            "feat_0": rng.integers(0, 5, size=n_rows).astype(float),
        }
        return pd.DataFrame(data)

    def _make_feature_infos(self, modes: list[str]) -> list[dict]:
        return [{"name": f"feat_{i}", "mode": m, "score": 0.5}
                for i, m in enumerate(modes)]

    def test_gate_uses_cached_monthly_val(self, monkeypatch):
        """``build_monthly_windows`` receives unsampled monthly val
        (``_cached_monthly_val``), not the train DataFrame.

        Uses monkeypatch to record the DataFrame passed to
        ``build_monthly_windows`` and asserts it is the slimmed val,
        distinguishable from the slimmed train by having a different
        ``datetime`` range.
        """
        # Create train and val data with distinct datetime ranges
        train_df = self._make_train_df(n_rows=200)
        train_df["datetime"] = pd.date_range("2020-01-01", periods=200, freq="D")

        val_df = self._make_train_df(n_rows=200)
        val_df["datetime"] = pd.date_range("2022-01-01", periods=200, freq="D")

        fi = self._make_feature_infos(["positive"])

        # Record what build_monthly_windows is called with
        recorded_dfs: list[pd.DataFrame] = []

        def tracking_build_monthly_windows(df, *args, **kwargs):
            recorded_dfs.append(df)
            # Return enough dummy windows so the gate doesn't hit
            # the "too few windows" warning
            return [pd.DataFrame() for _ in range(3)]

        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool.build_monthly_windows",
            tracking_build_monthly_windows,
        )

        # Patch _evaluate_rule_on_window so _apply_monthly_admission_gate
        # does not crash on empty dummy windows
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            lambda pool_entry, window_df, direction: 1.0,
        )

        # Create generator with val_df — after init, _cached_monthly_val
        # holds unsampled slim val, and _scoped_val_df is freed
        gen = Rule_Pool_Generator(
            train_df, fi, "long",
            pop_size=4,
            n_generations=1,
            seed=42,
            val_df=val_df,
            # Skip GPU warmup during engine init
            defer_warmup=True,
        )

        # Sanity-check the attrs after init
        assert gen._cached_slim_val is not None
        assert gen._cached_monthly_val is not None
        assert gen._scoped_val_df is None

        # --- Trip the gate via finalize_island ---
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool.Rule_Pool_Generator._ensure_engines",
            lambda self: None,
        )
        # run_phase2_evolution and build_feature_sampling_probs are imported
        # locally inside finalize_island — patch at their source modules
        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution",
            lambda **kw: ([POOL_THREE], {"metrics": {}}),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
            lambda feature_infos: {},
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._filter_pool_by_admission",
            lambda pool: pool,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool.Rule_Pool_Generator._annotate_archive_entries",  # noqa: E501
            staticmethod(lambda pool, **kw: pool),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool.Rule_Pool_Generator._release_resources",
            lambda self: None,
        )

        gen.island_id = "test_island"
        gen._evolution_state = {"dummy": "state"}
        gen.finalize_island()

        # --- Assertions ---
        assert len(recorded_dfs) == 1, (
            f"build_monthly_windows was called {len(recorded_dfs)} times"
        )
        used_df = recorded_dfs[0]

        # The recorded DataFrame must NOT be the slimmed train
        assert used_df is not gen._train_df, (
            "build_monthly_windows received the TRAIN DataFrame!"
        )
        # Prefer unsampled monthly val over sampled slim val
        assert used_df is gen._cached_monthly_val, (
            "build_monthly_windows should receive _cached_monthly_val"
        )
        # Verify the datetime range matches the slimmed val (year 2022+)
        dt_min = used_df["datetime"].min()
        assert dt_min >= pd.Timestamp("2022-01-01"), (
            f"Expected val datetime >= 2022-01-01, got {dt_min}"
        )

    # ------------------------------------------------------------------
    # Test B: gate is skipped when _cached_slim_val is None
    # ------------------------------------------------------------------

    def test_gate_skipped_when_cached_slim_val_none(self, monkeypatch, caplog):
        """When no val_df was provided, ``_cached_slim_val`` is None and the
        gate is skipped with a warning (no crash)."""
        train_df = self._make_train_df(n_rows=200)
        fi = self._make_feature_infos(["positive"])

        # Ensure the flag is on so the gate would run if it could
        monkeypatch.setattr(_cfg, "PHASE2_MONTHLY_ADMISSION_ENABLED", True)

        # Watch for the warning log
        caplog.set_level(logging.WARNING)

        # Spy on build_monthly_windows — it should NOT be called
        call_count = 0

        def spy_build(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return [pd.DataFrame() for _ in range(3)]

        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool.build_monthly_windows",
            spy_build,
        )

        # Create generator WITHOUT val_df → _cached_slim_val stays None
        # We must also prevent _build_engines from crashing (it will try
        # to build real engines). Skip it.
        monkeypatch.setattr(Rule_Pool_Generator, '_build_engines',
                            lambda self: None)

        gen = object.__new__(Rule_Pool_Generator)
        # Minimal attribute setup (normally done in __init__)
        gen.direction = "long"
        gen.pop_size = 4
        gen._cached_slim_val = None
        gen._cached_monthly_val = None
        gen._train_df = train_df
        gen._engine = None
        gen._val_engine = None
        gen.island_id = "test_island"
        gen._evolution_state = {"dummy": "state"}
        gen.island_hyperparams = None
        gen.island_profile = "test"
        gen._feature_signature = []
        gen.feature_infos = fi
        gen._feature_modes = {}
        gen._feature_names = []
        gen._rng = np.random.default_rng(42)
        gen.source_symbols = []
        gen._pending_migrant_seeds = []
        gen._island_history = []
        gen._island_generations_done = 0
        gen._cv_folds = None
        gen._cv_val_evaluator = None

        # Patch remaining dependencies for finalize_island
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool.Rule_Pool_Generator._ensure_engines",
            lambda self: None,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.evolution.evox_runner.run_phase2_evolution",
            lambda **kw: ([POOL_THREE], {"metrics": {}}),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_init.build_feature_sampling_probs",
            lambda feature_infos: {},
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._filter_pool_by_admission",
            lambda pool: pool,
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool.Rule_Pool_Generator._annotate_archive_entries",  # noqa: E501
            staticmethod(lambda pool, **kw: pool),
        )
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool.Rule_Pool_Generator._release_resources",
            lambda self: None,
        )

        with caplog.at_level(logging.WARNING):
            gen.finalize_island()

        # build_monthly_windows should NOT have been called
        assert call_count == 0, (
            "build_monthly_windows was called even though "
            "_cached_slim_val is None"
        )
        # The warning about missing val DataFrame should be logged
        assert any(
            "no val DataFrame available" in rec.message
            for rec in caplog.records
        ), (
            "Expected warning 'no val DataFrame available' in logs, "
            f"got: {[r.message for r in caplog.records]}"
        )

    # ------------------------------------------------------------------
    # Test C: gate shows non-trivial filtering on val-like windows
    # ------------------------------------------------------------------

    def test_non_trivial_filtering_on_val(self, monkeypatch):
        """With a val-like data pattern (many unprofitable months), the gate
        rejects a non-trivial number of rules (more than the 0-8/75 seen on
        train).  We simulate 4 rules with mixed performance across 6 months.
        """
        # 4 rules, 6 months each:
        #   rule 0: profitable on 5/6 (ratio=0.83) → kept
        #   rule 1: profitable on 0/6 (ratio=0.00) → rejected
        #   rule 2: profitable on 4/6 (ratio=0.67) → kept
        #   rule 3: profitable on 1/6 (ratio=0.17) → rejected
        returns = [
            [2.0, 1.5, -0.5, 3.0, 1.0, 0.5],   # rule 0: 5/6
            [-3.0, -2.0, -1.0, -4.0, -0.5, -2.5],  # rule 1: 0/6
            [1.0, -1.0, 2.0, 1.5, -0.5, 3.0],  # rule 2: 4/6
            [-1.0, 2.0, -0.5, -1.5, -2.0, -1.0],  # rule 3: 1/6
        ]
        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase2_rule_pool._evaluate_rule_on_window",
            _DeterministicEvaluator(returns),
        )

        # A pool of 4 rules
        pool_four = POOL_THREE + [{
            "conditions": ["[feat_d] IS low", "symbol is 4"],
            "chromosome": [3, 3, 3],
            "objectives": {"total_return_pct": 1.0, "profit_factor": 1.1},
            "executed_trades": 20,
        }]

        result = _apply_monthly_admission_gate(
            pool_four, list(range(6)), "long",
        )

        # 2 of 4 rules kept → non-trivial filtering (50% rejection)
        assert len(result) == 2, (
            f"Expected 2 kept / 4 rules (50% rejection), got {len(result)}"
        )
        assert result[0]["conditions"] == pool_four[0]["conditions"]
        assert result[1]["conditions"] == pool_four[2]["conditions"]

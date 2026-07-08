"""Unit tests for Pareto-collapse warning gate (audit finding #13).

AC: The warning at evox_runner.py:2736-2745 is gated on
    len(pareto_indices) >= PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution


class _FakeEngine:
    """Fake engine that returns metrics producing a tradeoff between
    f1 (-sortino) and f2 (dd_for_obj) so that *all* individuals in the
    population are Pareto-optimal — pareto front size == pop_size.
    """

    def simulate_rule_batch(self, chromosomes, tp, sl, capital_pct, **kwargs):
        B = chromosomes.shape[0]
        # sortino_ratio = i, max_drawdown_pct = i → f1 ≈ -sat(i), f2 = i.
        # f1 improves with i, f2 worsens with i → every individual is
        # non-dominated, so pareto_size == pop_size.
        return [
            {
                "sortino_ratio": float(i),
                "total_return_pct": float(B - i),
                "max_drawdown_pct": float(i),
                "win_rate": 50.0,
                "executed_trades": 25,
            }
            for i in range(B)
        ]


class TestParetoCollapseWarningGate:
    """AC 1–5: warning gated on len(pareto_indices) >= config threshold."""

    def _run_and_count_warnings(
        self, monkeypatch, caplog,
        pop_size: int, min_pareto_size: int, threshold: float,
    ) -> int:
        """Run 2-gen evolution and return count of 'Pareto collapse risk' warnings."""
        monkeypatch.setattr(
            _cfg, "PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE", min_pareto_size,
        )
        monkeypatch.setattr(
            _cfg, "PHASE2_OBJECTIVE_CORR_WARN_THRESHOLD", threshold,
        )

        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
            {"name": "feat_1", "mode": "binary", "score": 0.5},
        ]
        rng = np.random.default_rng(42)

        with caplog.at_level(logging.WARNING, logger="gpu_fuzzy_trader.evolution.evox_runner"):
            run_phase2_evolution(
                feature_infos=feature_infos,
                engine=_FakeEngine(),
                pop_size=pop_size,
                n_generations=2,
                rng=rng,
            )

        return len([
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "Pareto collapse risk" in r.getMessage()
        ])

    # --- AC 2: pareto size < 5 → no warning ---
    def test_no_warning_below_min_pareto(self, monkeypatch, caplog):
        """AC 2: pareto_size=4 < min_pareto_size=5 → no warning fires."""
        n = self._run_and_count_warnings(
            monkeypatch, caplog,
            pop_size=4, min_pareto_size=5, threshold=0.0,
        )
        assert n == 0, (
            f"Expected 0 warnings for pareto_size=4 < min=5, got {n}."
        )

    # --- AC 3: pareto size >= 5 with |corr| ≥ 0.9 → warning fires ---
    def test_warning_fires_above_min_pareto(self, monkeypatch, caplog):
        """AC 3: pareto_size=7 >= min_pareto_size=5 → warning fires."""
        n = self._run_and_count_warnings(
            monkeypatch, caplog,
            pop_size=7, min_pareto_size=5, threshold=0.0,
        )
        assert n >= 1, (
            f"Expected ≥1 warning for pareto_size=7 >= min=5, got {n}."
        )

    # --- AC 4: PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE = 5 is the default ---
    def test_config_default_is_five(self):
        """AC 4: The default value of the config flag is 5."""
        assert _cfg.PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE == 5, (
            f"Expected default PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE=5, "
            f"got {_cfg.PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE}."
        )

    # --- AC 5: min_pareto_size = 0 → warning fires regardless of pareto size
    #           (regression guard for pre-task-12 behavior) ---
    def test_regression_min_pareto_zero(self, monkeypatch, caplog):
        """AC 5: min_pareto_size=0 → warning fires even with small pareto (size=4)."""
        n = self._run_and_count_warnings(
            monkeypatch, caplog,
            pop_size=4, min_pareto_size=0, threshold=0.0,
        )
        assert n >= 1, (
            f"Expected ≥1 warning for min_pareto_size=0 (regression guard), "
            f"got {n}."
        )

    # --- bonus: verify the pareto_size=N suffix appears in the message ---
    def test_warning_message_contains_pareto_size(self, monkeypatch, caplog):
        """The log message includes 'pareto_size=N' suffix."""
        monkeypatch.setattr(
            _cfg, "PHASE2_OBJECTIVE_CORR_MIN_PARETO_SIZE", 0,
        )
        monkeypatch.setattr(
            _cfg, "PHASE2_OBJECTIVE_CORR_WARN_THRESHOLD", 0.0,
        )

        feature_infos = [
            {"name": "feat_0", "mode": "binary", "score": 0.5},
            {"name": "feat_1", "mode": "binary", "score": 0.5},
        ]
        rng = np.random.default_rng(42)

        with caplog.at_level(logging.WARNING, logger="gpu_fuzzy_trader.evolution.evox_runner"):
            run_phase2_evolution(
                feature_infos=feature_infos,
                engine=_FakeEngine(),
                pop_size=5,
                n_generations=2,
                rng=rng,
            )

        for record in caplog.records:
            if record.levelno == logging.WARNING and "Pareto collapse risk" in record.getMessage():
                msg = record.getMessage()
                assert "pareto_size=" in msg, (
                    f"Warning message missing 'pareto_size=N' suffix: {msg}"
                )
                return  # found at least one with the suffix

        pytest.fail("No 'Pareto collapse risk' warning found to verify suffix.")

"""Unit tests for gpu_fuzzy_trader.log_progress."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock

import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.log_progress import (
    generation_log_interval,
    iteration_log_interval,
    log_generation,
    maybe_log_generation,
    should_log_step,
)


class TestGenerationLogInterval:
    def test_zero_generations(self):
        assert generation_log_interval(0) == 1

    def test_one_generation(self):
        assert generation_log_interval(1) == 1

    def test_short_run_logs_every_step(self):
        assert generation_log_interval(15) == 1

    def test_long_run_throttled(self):
        assert generation_log_interval(500) == 25

    def test_config_override(self, monkeypatch):
        monkeypatch.setattr(_cfg, "LOG_GENERATION_INTERVAL", 10)
        assert generation_log_interval(500) == 10


class TestShouldLogStep:
    def test_empty_total(self):
        assert should_log_step(0, 0, 5) is False

    def test_first_and_last_always(self):
        assert should_log_step(0, 10, 25) is True
        assert should_log_step(9, 10, 25) is True

    def test_middle_on_interval(self):
        assert should_log_step(5, 10, 5) is True
        assert should_log_step(3, 10, 5) is False

    def test_single_step_total(self):
        assert should_log_step(0, 1, 5) is True


class TestIterationLogInterval:
    def test_small_total(self):
        assert iteration_log_interval(5) == 1

    def test_large_total(self):
        assert iteration_log_interval(5000, target_logs=20) == 250


class TestLogGeneration:
    def test_emits_info(self):
        log = MagicMock(spec=logging.Logger)
        log_generation(log, "Phase 2 [long] NSGA-III",
                       24, 500, 38, 2.14, elapsed_s=45.2)
        log.info.assert_called_once()
        msg = log.info.call_args[0][0]
        assert "gen 25/500" in msg
        assert "pareto=38" in msg
        assert "2.14%" in msg
        assert "45.2s" in msg

    def test_emits_info_with_extra_metrics(self):
        log = MagicMock(spec=logging.Logger)
        log_generation(
            log, "Phase 2 [long] NSGA-III",
            24, 500, 38, 2.14,
            max_return_pct=15.5,
            max_sortino=2.5,
            valid_count=12,
            elapsed_s=45.2
        )
        log.info.assert_called_once()
        msg = log.info.call_args[0][0]
        assert "gen 25/500" in msg
        assert "pareto=38" in msg
        assert "mean_return=2.14%" in msg
        assert "max_return=15.50%" in msg
        assert "max_sortino=2.50" in msg
        assert "valid_rules=12" in msg
        assert "elapsed=45.2s" in msg

    def test_emits_unique_chrom_and_cache_hit_rate(self):
        log = MagicMock(spec=logging.Logger)
        log_generation(
            log, "Phase 2 [long] NSGA-III",
            36, 80, 300, 9.69,
            unique_chromosome_ratio=1.0,
            pop_unique_chromosome_ratio=0.05,
            cache_hit_rate=0.82,
        )
        msg = log.info.call_args[0][0]
        assert "pareto_unique=1.00" in msg
        assert "pop_unique=0.05" in msg
        assert "cache_hit_rate=0.82" in msg

    def test_emits_deployable_pop_viable_and_plateau_streak(self):
        log = MagicMock(spec=logging.Logger)
        log_generation(
            log, "Phase 2 [long] NSGA-III",
            19, 80, 12, 4.5,
            deployable_count=3,
            pop_viable_count=18,
            plateau_streak=2,
        )
        msg = log.info.call_args[0][0]
        assert "deployable=3" in msg
        assert "pop_viable=18" in msg
        assert "plateau_streak=2" in msg


class TestMaybeLogGeneration:
    def test_skips_when_not_on_interval(self):
        log = MagicMock(spec=logging.Logger)
        maybe_log_generation(
            log, "tag", 3, 500, 10, -1.0, interval=25,
        )
        log.info.assert_not_called()

    def test_logs_on_interval(self):
        log = MagicMock(spec=logging.Logger)
        maybe_log_generation(
            log, "tag", 25, 500, 10, -2.5, interval=25,
        )
        log.info.assert_called_once()

    def test_maybe_logs_extra_metrics(self):
        log = MagicMock(spec=logging.Logger)
        maybe_log_generation(
            log, "tag", 25, 500, 10, -2.5,
            max_return_pct=12.3,
            max_sortino=1.8,
            valid_count=5,
            interval=25,
        )
        log.info.assert_called_once()
        msg = log.info.call_args[0][0]
        assert "max_return=12.30%" in msg
        assert "max_sortino=1.80" in msg
        assert "valid_rules=5" in msg

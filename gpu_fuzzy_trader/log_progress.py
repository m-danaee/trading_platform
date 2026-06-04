"""
log_progress.py — Throttled progress logging for long pipeline loops.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from gpu_fuzzy_trader import config as _cfg

if TYPE_CHECKING:
    pass


def generation_log_interval(n_generations: int) -> int:
    """
    Return how often to log generation progress.

    Uses LOG_GENERATION_INTERVAL from config when positive; otherwise auto-throttles
    so long runs (~500 gens) log ~every 25 steps and short runs log every step.
    """
    if n_generations <= 0:
        return 1
    override = _cfg.LOG_GENERATION_INTERVAL
    if override > 0:
        return max(1, override)
    return max(1, min(25, n_generations // 20))


def should_log_step(step: int, total: int, interval: int) -> bool:
    """Log first step, last step, and every *interval* steps in between."""
    if total <= 0:
        return False
    if step == 0 or step == total - 1:
        return True
    if interval <= 0:
        return True
    return step % interval == 0


def iteration_log_interval(total: int, target_logs: int = 20) -> int:
    """
    Interval for generic loops (RL samples, SB3 windows).

    Aims for about *target_logs* INFO lines over the full run.
    """
    if total <= 0:
        return 1
    return max(1, total // target_logs)


def log_generation(
    logger: logging.Logger,
    tag: str,
    gen: int,
    n_generations: int,
    pareto_size: int,
    mean_return_pct: float,
    *,
    max_return_pct: float | None = None,
    median_return_pct: float | None = None,
    max_sortino: float | None = None,
    valid_count: int | None = None,
    elapsed_s: float | None = None,
) -> None:
    """Emit a consistent INFO line for one evolutionary generation."""
    msg = (
        "%s gen %d/%d: pareto=%d mean_return=%.2f%%"
        % (tag, gen + 1, n_generations, pareto_size, mean_return_pct)
    )
    if median_return_pct is not None:
        msg += " median_return=%.2f%%" % median_return_pct
    if max_return_pct is not None:
        msg += " max_return=%.2f%%" % max_return_pct
    if max_sortino is not None:
        msg += " max_sortino=%.2f" % max_sortino
    if valid_count is not None:
        msg += " valid_rules=%d" % valid_count
    if elapsed_s is not None:
        msg += " elapsed=%.1fs" % elapsed_s
    logger.info(msg)


def log_phase3_generation(
    logger: logging.Logger,
    tag: str,
    gen: int,
    n_generations: int,
    pareto_size: int,
    mean_train_return_pct: float,
    mean_val_return_pct: float,
    *,
    max_val_return_pct: float | None = None,
    elapsed_s: float | None = None,
) -> None:
    """Phase 3 refine progress (actual train/val returns, not objective f1)."""
    msg = (
        "%s gen %d/%d: pareto=%d mean_train_return=%.2f%% "
        "mean_val_return=%.2f%%"
        % (
            tag,
            gen + 1,
            n_generations,
            pareto_size,
            mean_train_return_pct,
            mean_val_return_pct,
        )
    )
    if max_val_return_pct is not None:
        msg += " max_val_return=%.2f%%" % max_val_return_pct
    if elapsed_s is not None:
        msg += " elapsed=%.1fs" % elapsed_s
    logger.info(msg)


def maybe_log_generation(
    logger: logging.Logger,
    tag: str,
    gen: int,
    n_generations: int,
    pareto_size: int,
    mean_return_pct: float,
    *,
    max_return_pct: float | None = None,
    median_return_pct: float | None = None,
    max_sortino: float | None = None,
    valid_count: int | None = None,
    loop_start: float | None = None,
    interval: int | None = None,
) -> None:
    """Log generation progress when the step matches the throttle interval."""
    import time

    iv = interval if interval is not None else generation_log_interval(
        n_generations)
    if not should_log_step(gen, n_generations, iv):
        return
    elapsed = None
    if loop_start is not None:
        elapsed = time.monotonic() - loop_start
    log_generation(
        logger,
        tag,
        gen,
        n_generations,
        pareto_size,
        float(mean_return_pct),
        max_return_pct=max_return_pct,
        median_return_pct=median_return_pct,
        max_sortino=max_sortino,
        valid_count=valid_count,
        elapsed_s=elapsed,
    )


def maybe_log_phase3_generation(
    logger: logging.Logger,
    tag: str,
    gen: int,
    n_generations: int,
    pareto_size: int,
    mean_train_return_pct: float,
    mean_val_return_pct: float,
    *,
    max_val_return_pct: float | None = None,
    loop_start: float | None = None,
    interval: int | None = None,
) -> None:
    """Throttled Phase 3 refine logging with real split returns."""
    import time

    iv = interval if interval is not None else generation_log_interval(
        n_generations)
    if not should_log_step(gen, n_generations, iv):
        return
    elapsed = None
    if loop_start is not None:
        elapsed = time.monotonic() - loop_start
    log_phase3_generation(
        logger,
        tag,
        gen,
        n_generations,
        pareto_size,
        mean_train_return_pct,
        mean_val_return_pct,
        max_val_return_pct=max_val_return_pct,
        elapsed_s=elapsed,
    )

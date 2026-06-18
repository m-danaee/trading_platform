"""
phase4_wf_optimizer.py — WalkForwardRiskOptimizer (Phase 4)

Fine-tunes TP, SL, and capital_pct for each rule in the selected rule set using
deterministic grid search. Rule conditions are frozen from Phase 3. Each rule is
optimized on its assigned symbol slice (``symbol is X`` conditions). Deployment
gate metrics still use the full ruleset on the full universe.

Skip logic:
  If outputs/{direction}.json exists, TP/SL/capital_pct are within bounds,
  and risk_optimized is true, Phase 4 is skipped.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df
from gpu_fuzzy_trader.backtest.symbol_conditions import (
    get_normalized_symbol_array,
    split_feature_and_symbol_conditions,
)
from gpu_fuzzy_trader.output.writer import (
    _maybe_write_evaluator_clean,
    write_evaluator_clean,
)
from gpu_fuzzy_trader.phases.phase3_cache import (
    Phase3EvalCache,
    build_rules_signal_cache,
    build_single_split_signal_cache,
)
from gpu_fuzzy_trader.phases.phase3_rule_set import _monthly_feature_names
from gpu_fuzzy_trader.reporting.reporter import Reporter
from gpu_fuzzy_trader.validation.monthly_windows import (
    build_monthly_windows,
    evaluate_rule_set_monthly,
    monthly_penalty,
)

logger = logging.getLogger(__name__)


class Phase4NoFeasibleTrialError(RuntimeError):
    """Raised when no Optuna trial passes walk-forward feasibility gates."""


_OUTPUT_PATHS = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "long.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "short.json"),
}


def split_validation_walk_forward(
    val_df: pd.DataFrame,
    k: int,
) -> list[pd.DataFrame]:
    """
    Split validation data into K walk-forward windows (per symbol, chronological).

    For each symbol, rows are sorted by datetime and split into K contiguous
    chunks. Window i is the concatenation of all symbols' chunk i.

    Parameters
    ----------
    val_df : pd.DataFrame
        Validation split (must contain ``symbol`` and ``datetime``).
    k : int
        Number of windows (must be >= 1).

    Returns
    -------
    list[pd.DataFrame]
        K DataFrames, one per window.

    Raises
    ------
    ValueError
        If k < 1, val_df is empty, or any symbol has fewer than k rows.
    """
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if len(val_df) == 0:
        raise ValueError("val_df is empty; cannot build walk-forward splits")

    if "symbol" not in val_df.columns:
        raise ValueError(
            "val_df must contain a 'symbol' column for walk-forward splits")

    sort_col = "datetime" if "datetime" in val_df.columns else None
    symbol_chunks: dict[str, list[pd.DataFrame]] = {}

    for symbol, group in val_df.groupby("symbol", sort=True):
        g = group.sort_values(sort_col) if sort_col else group
        n = len(g)
        if n < k:
            raise ValueError(
                f"Symbol {symbol!r} has {n} validation rows; need at least {k} "
                f"for PHASE4_WF_SPLITS={k}"
            )
        indices = np.array_split(np.arange(n), k)
        symbol_chunks[str(symbol)] = [g.iloc[idx].copy() for idx in indices]

    windows: list[pd.DataFrame] = []
    symbols = sorted(symbol_chunks.keys())
    for i in range(k):
        parts = [symbol_chunks[sym][i] for sym in symbols]
        windows.append(pd.concat(parts, ignore_index=True))

    return windows


def build_tail_holdout_split(
    val_df: pd.DataFrame,
    fraction: float,
) -> pd.DataFrame:
    """
    Last *fraction* of each symbol's validation rows (chronological tail).

    Used as an extra strict walk-forward window to reduce val-period overfitting.
    """
    if fraction <= 0.0 or fraction >= 1.0:
        raise ValueError(f"fraction must be in (0, 1), got {fraction}")
    if len(val_df) == 0:
        raise ValueError("val_df is empty")

    sort_col = "datetime" if "datetime" in val_df.columns else None
    parts: list[pd.DataFrame] = []
    for _, group in val_df.groupby("symbol", sort=True):
        g = group.sort_values(sort_col) if sort_col else group
        n = len(g)
        start = max(0, int(np.floor(n * (1.0 - fraction))))
        if start < n:
            parts.append(g.iloc[start:].copy())
    if not parts:
        raise ValueError("tail holdout produced no rows")
    return pd.concat(parts, ignore_index=True)


def build_phase4_walk_forward_splits(
    val_df: pd.DataFrame,
    k: int,
) -> list[pd.DataFrame]:
    """Standard K WF windows plus optional chronological tail holdout."""
    splits = split_validation_walk_forward(val_df, k)
    if _cfg.PHASE4_INCLUDE_TAIL_HOLDOUT:
        tail = build_tail_holdout_split(
            val_df, float(_cfg.PHASE4_TAIL_HOLDOUT_FRACTION))
        splits.append(tail)
    return splits


def _build_candidate_rule_set(
    rules: list[dict],
    params_list: list[dict],
) -> list[dict]:
    """Attach optimized risk params; keep conditions frozen."""
    return [
        {
            "conditions": list(rules[i]["conditions"]),
            "tp": float(params_list[i]["tp"]),
            "sl": float(params_list[i]["sl"]),
            "capital_pct": float(params_list[i]["capital_pct"]),
        }
        for i in range(len(rules))
    ]


class Phase4WalkForwardEvaluator:
    """
    Pre-built per-window backtest engines for walk-forward evaluation.

    Engines are created once per walk-forward split (not per Optuna trial).
    """

    __slots__ = ("_engines",)

    def __init__(
        self,
        val_df: pd.DataFrame,
        direction: str,
        k: int,
    ) -> None:
        splits = build_phase4_walk_forward_splits(val_df, k)
        self._engines = [
            CPUBacktestEngine(split_df, {}, direction)
            for split_df in splits
        ]

    def evaluate_worst_case(
        self,
        candidate_rule_set: list[dict],
    ) -> tuple[float, float, float, float]:
        """Run backtest on each pre-built window engine."""
        split_returns: list[float] = []
        split_drawdowns: list[float] = []
        split_turnover: list[float] = []
        split_pf: list[float] = []

        for engine in self._engines:
            try:
                metrics = engine.simulate_rule_set(candidate_rule_set)
            except Exception:
                split_returns.append(-100.0)
                split_drawdowns.append(100.0)
                split_turnover.append(0.0)
                split_pf.append(0.0)
                continue

            split_returns.append(float(metrics.get("total_return_pct", 0.0)))
            split_drawdowns.append(float(metrics.get("max_drawdown_pct", 0.0)))
            split_turnover.append(float(metrics.get("executed_trades", 0.0)))
            split_pf.append(float(metrics.get("profit_factor", 0.0)))

        worst_return = min(split_returns) if split_returns else -100.0
        worst_drawdown = max(split_drawdowns) if split_drawdowns else 100.0
        worst_turnover = min(split_turnover) if split_turnover else 0.0
        worst_pf = min(split_pf) if split_pf else 0.0
        return worst_return, worst_drawdown, worst_turnover, worst_pf


def _normalize_capital_pct(rules_set: list[dict]) -> list[dict]:
    """Scale capital_pct so sum <= MAX_TOTAL_EXPOSURE_PCT when enabled."""
    if not _cfg.PHASE4_HARD_CAP_NORMALIZE or not rules_set:
        return rules_set

    total_cap = sum(float(p.get("capital_pct", 0.0)) for p in rules_set)
    limit = float(_cfg.MAX_TOTAL_EXPOSURE_PCT)
    if total_cap > limit and total_cap > 0:
        scale = limit / total_cap
        for p in rules_set:
            p["capital_pct"] = float(p.get("capital_pct", 0.0)) * scale
    return rules_set


@dataclass
class _Phase4MonthlyContext:
    """Cached monthly windows and per-window mask caches for Phase 4 grid scoring."""

    combined_df: pd.DataFrame
    monthly_windows: list[pd.DataFrame]
    feature_names: list[str]
    direction: str
    window_engines: list[CPUBacktestEngine] = field(default_factory=list)
    window_caches: list[Phase3EvalCache] = field(default_factory=list)


@dataclass
class _RuleOptContext:
    """Per-rule train/val engines scoped to the rule's assigned symbol(s)."""

    symbols: list[str]
    train_engine: CPUBacktestEngine
    val_engine: CPUBacktestEngine
    eval_cache: Phase3EvalCache | None = None
    monthly_ctx: _Phase4MonthlyContext | None = None


def _symbols_for_rule(rule: dict, rule_index: int) -> list[str]:
    """Return normalized symbol filters for a rule (empty = all symbols)."""
    _, symbols = split_feature_and_symbol_conditions(
        list(rule.get("conditions", [])),
        rule_number=rule_index + 1,
    )
    return symbols


def _filter_df_by_symbols(
    df: pd.DataFrame,
    symbols: list[str],
) -> pd.DataFrame:
    """Keep rows whose normalized symbol is in *symbols* (no-op when empty)."""
    if not symbols or len(df) == 0:
        return df
    if "symbol" not in df.columns:
        raise ValueError(
            "Rule has symbol filters but dataframe has no 'symbol' column"
        )
    allowed = {str(s) for s in symbols}
    norm = get_normalized_symbol_array(df)
    mask = np.isin(norm, list(allowed))
    if not mask.any():
        return df.iloc[0:0].copy()
    return df.iloc[np.where(mask)[0]].copy()


def _build_rule_opt_contexts(
    rules: list[dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    direction: str,
) -> list[_RuleOptContext]:
    """Build per-rule engines on each rule's assigned symbol slice."""
    contexts: list[_RuleOptContext] = []
    for idx, rule in enumerate(rules):
        symbols = _symbols_for_rule(rule, idx)
        rule_train = _filter_df_by_symbols(train_df, symbols)
        rule_val = _filter_df_by_symbols(val_df, symbols)
        train_engine = CPUBacktestEngine(rule_train, {}, direction)
        val_engine = CPUBacktestEngine(rule_val, {}, direction)
        eval_cache = (
            build_rules_signal_cache([rule], rule_train, rule_val)
            if len(rule_train) > 0 and len(rule_val) > 0
            else None
        )
        monthly_ctx = _build_phase4_monthly_context(
            rule_train, rule_val, direction, [rule],
        )
        contexts.append(_RuleOptContext(
            symbols=symbols,
            train_engine=train_engine,
            val_engine=val_engine,
            eval_cache=eval_cache,
            monthly_ctx=monthly_ctx,
        ))
    return contexts


def _phase4_scaled_monthly_penalty(raw_penalty: float) -> float:
    """Convert raw monthly_penalty points into a grid-score drag."""
    weight = float(getattr(_cfg, "PHASE4_MONTHLY_SCORE_WEIGHT", 0.70))
    scale = float(getattr(_cfg, "PHASE4_MONTHLY_PENALTY_SCALE", 10.0))
    if scale <= 0.0:
        raise ValueError(
            f"PHASE4_MONTHLY_PENALTY_SCALE must be > 0, got {scale!r}"
        )
    return float(raw_penalty) * weight / scale


def _build_phase4_monthly_context(
    train_df: pd.DataFrame | None,
    val_df: pd.DataFrame,
    direction: str,
    rules: list[dict],
) -> _Phase4MonthlyContext | None:
    """Build train+val monthly windows and per-window mask caches once per grid run."""
    if not bool(getattr(_cfg, "MONTHLY_VALIDATION_ENABLED", False)):
        return None
    if not bool(getattr(_cfg, "PHASE4_MONTHLY_EVAL_EVERY_TRIAL", True)):
        return None

    parts: list[pd.DataFrame] = []
    if train_df is not None and len(train_df) > 0:
        parts.append(train_df)
    if val_df is not None and len(val_df) > 0:
        parts.append(val_df)
    if not parts:
        return None

    combined = pd.concat(parts, ignore_index=True)
    feature_names = _monthly_feature_names(combined)
    monthly_windows = build_monthly_windows(combined)

    window_engines: list[CPUBacktestEngine] = []
    window_caches: list[Phase3EvalCache] = []
    for part in monthly_windows:
        slim = (
            slim_backtest_df(part, feature_names)
            if feature_names
            else part
        )
        window_caches.append(build_single_split_signal_cache(rules, slim))
        window_engines.append(CPUBacktestEngine(slim, {}, direction))

    return _Phase4MonthlyContext(
        combined_df=combined,
        monthly_windows=monthly_windows,
        feature_names=feature_names,
        direction=direction,
        window_engines=window_engines,
        window_caches=window_caches,
    )


def _monthly_drag_for_rules(
    rules: list[dict],
    monthly_ctx: _Phase4MonthlyContext,
) -> float:
    """Monthly penalty drag for one grid trial's rule set."""
    try:
        if monthly_ctx.window_engines and monthly_ctx.window_caches:
            metrics: list[dict] = []
            for eng, win_cache in zip(
                monthly_ctx.window_engines,
                monthly_ctx.window_caches,
                strict=True,
            ):
                metrics.append(
                    eng.simulate_rule_set_from_cache(rules, win_cache, "train"))
            summary = summarize_monthly_metrics(metrics)
        else:
            summary, _ = evaluate_rule_set_monthly(
                monthly_ctx.combined_df,
                rules,
                monthly_ctx.direction,
                feature_names=monthly_ctx.feature_names,
                windows=monthly_ctx.monthly_windows,
            )
        if summary.windows <= 0:
            raw_pen = float(
                getattr(_cfg, "PHASE4_MONTHLY_FALLBACK_PENALTY", 5.0))
        else:
            raw_pen = monthly_penalty(summary)
    except Exception as exc:
        logger.debug("Phase 4 monthly eval failed for grid trial: %s", exc)
        raw_pen = float(getattr(_cfg, "PHASE4_MONTHLY_FALLBACK_PENALTY", 5.0))
    return _phase4_scaled_monthly_penalty(raw_pen)


def _score_metrics(train_m: dict, valid_m: dict) -> float:
    """Composite score for grid search (return / DD / PF weighted).

    When ``PHASE4_USE_ROBUST_SCORE`` is True, return terms use
    ``min(train, val)`` so Phase 4 does not chase validation-only spikes.
    """
    train_ret = float(train_m.get("total_return_pct", 0.0))
    train_dd = float(train_m.get("max_drawdown_pct", 100.0))
    train_pf = float(train_m.get("profit_factor", 0.0))
    train_wr = float(train_m.get("win_rate", 0.0))

    valid_ret = float(valid_m.get("total_return_pct", 0.0))
    valid_dd = float(valid_m.get("max_drawdown_pct", 100.0))
    valid_pf = float(valid_m.get("profit_factor", 0.0))
    valid_wr = float(valid_m.get("win_rate", 0.0))

    use_robust = bool(getattr(_cfg, "PHASE4_USE_ROBUST_SCORE", True))
    score_ret = min(train_ret, valid_ret) if use_robust else valid_ret
    aux_ret = valid_ret if use_robust else train_ret
    score_dd = max(train_dd, valid_dd) if use_robust else valid_dd
    aux_dd = train_dd if use_robust else valid_dd
    score_pf = min(train_pf, valid_pf) if use_robust else valid_pf
    aux_pf = train_pf if use_robust else valid_pf

    dd_floor = 0.50
    score_ratio = score_ret / max(score_dd, dd_floor)
    aux_ratio = aux_ret / max(aux_dd, dd_floor)

    def _pf_term(pf: float, cap: float = 5.0) -> float:
        return min(pf, cap)

    score = (
        120.0 * score_ratio
        + 45.0 * aux_ratio
        + 4.5 * score_ret
        + 1.2 * aux_ret
        + 14.0 * _pf_term(score_pf)
        + 5.0 * _pf_term(aux_pf)
        + 0.06 * valid_wr
        + 0.025 * train_wr
        - 0.25 * score_dd
        - 0.08 * aux_dd
    )

    # Penalise negative returns and low PF heavily
    if train_ret <= 0.0:
        score -= 500.0 + abs(train_ret) * 20.0
    if valid_ret <= 0.0:
        score -= 1000.0 + abs(valid_ret) * 30.0
    if train_pf < 1.0:
        score -= (1.0 - train_pf) * 60.0
    if valid_pf < 1.0:
        score -= (1.0 - valid_pf) * 120.0

    val_train_gap = valid_ret - train_ret
    gap_max = float(getattr(_cfg, "PHASE4_MAX_VAL_TRAIN_GAP_PCT", 12.0))
    if val_train_gap > gap_max:
        score -= (val_train_gap - gap_max) * 35.0

    return float(score)


def _evaluate_ruleset(
    train_engine: CPUBacktestEngine,
    val_engine: CPUBacktestEngine,
    rules: list[dict],
    *,
    monthly_ctx: _Phase4MonthlyContext | None = None,
    eval_cache: Phase3EvalCache | None = None,
) -> tuple[dict, dict, float]:
    """Evaluate a full rule set on train and val, returning metrics + score."""
    if eval_cache is not None:
        train_m = train_engine.simulate_rule_set_from_cache(
            rules, eval_cache, "train")
        val_m = val_engine.simulate_rule_set_from_cache(
            rules, eval_cache, "val")
    else:
        train_m = train_engine.simulate_rule_set(rules)
        val_m = val_engine.simulate_rule_set(rules)
    score = _score_metrics(train_m, val_m)
    if monthly_ctx is not None:
        score -= _monthly_drag_for_rules(rules, monthly_ctx)
    return train_m, val_m, score


def _evaluate_single_rule(
    rule: dict,
    rule_ctx: _RuleOptContext,
) -> tuple[dict, dict, float]:
    """Evaluate one rule on its assigned symbol slice."""
    rules = [rule]
    return _evaluate_ruleset(
        rule_ctx.train_engine,
        rule_ctx.val_engine,
        rules,
        monthly_ctx=rule_ctx.monthly_ctx,
        eval_cache=rule_ctx.eval_cache,
    )


def _optimize_risk_grid(
    rules: list[dict],
    train_engine: CPUBacktestEngine,
    val_engine: CPUBacktestEngine,
    *,
    monthly_ctx: _Phase4MonthlyContext | None = None,
    eval_cache: Phase3EvalCache | None = None,
    rule_contexts: list[_RuleOptContext] | None = None,
    min_improvement: float = 0.02,
) -> tuple[list[dict], dict, dict, float, list[dict]]:
    """Per-rule round-robin grid search over TP, SL, capital_pct.

    For each rule, every (TP, SL, capital_pct) combination from the configured
    grid is tested.  Combinations that violate ``sum(capital_pct) <=
    PHASE4_GRID_MAX_TOTAL_CAPITAL`` or fail ``gate_positive_good`` are skipped.
    Two passes of round-robin per-rule tuning are performed.

    When *rule_contexts* is provided, each rule is scored in isolation on its
    assigned symbol slice (from ``symbol is X`` conditions).

    Parameters
    ----------
    rules : list[dict]
        Input rule set with ``tp``, ``sl``, ``capital_pct``.
    train_engine : CPUBacktestEngine
        Engine for training split evaluation (full universe).
    val_engine : CPUBacktestEngine
        Engine for validation split evaluation (full universe).
    rule_contexts : list[_RuleOptContext] | None
        Per-rule symbol-scoped engines; when set, grid trials score each rule
        only on its symbols.
    min_improvement : float
        Minimum score improvement to accept a new combination (default 0.02).

    Returns
    -------
    tuple[list[dict], dict, dict, float, list[dict]]
        (optimized_rules, train_metrics, val_metrics, score, history)
    """
    from gpu_fuzzy_trader.phases.phase3_rule_set import gate_positive_good

    per_rule_symbols = rule_contexts is not None and len(
        rule_contexts) == len(rules)

    best_rules = [dict(r) for r in rules]

    # Initial evaluation (full ruleset on full universe for deployment metrics)
    cur_train, cur_val, cur_score = _evaluate_ruleset(
        train_engine, val_engine, best_rules,
        monthly_ctx=monthly_ctx, eval_cache=eval_cache)

    hist: list[dict] = [{
        "pass": 0,
        "rule_index": -1,
        "score": cur_score,
        "train_return_pct": float(cur_train.get("total_return_pct", 0.0)),
        "valid_return_pct": float(cur_val.get("total_return_pct", 0.0)),
        "train_pf": float(cur_train.get("profit_factor", 0.0)),
        "valid_pf": float(cur_val.get("profit_factor", 0.0)),
    }]

    tp_grid = tuple(float(x) for x in getattr(
        _cfg, "PHASE4_GRID_TP_VALUES",
        (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)))
    sl_grid = tuple(float(x) for x in getattr(
        _cfg, "PHASE4_GRID_SL_VALUES",
        (1.0, 1.2, 1.5, 2.0, 2.5, 3.0)))
    cap_grid = tuple(float(x) for x in getattr(
        _cfg, "PHASE4_GRID_CAPITAL_VALUES",
        (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 35.0, 50.0)))
    max_total_cap = float(getattr(_cfg, "PHASE4_GRID_MAX_TOTAL_CAPITAL", 95.0))
    passes = int(getattr(_cfg, "PHASE4_GRID_PASSES", 2))

    n_rules = len(best_rules)
    total_trials = n_rules * passes * len(tp_grid) * len(sl_grid) * len(cap_grid)
    trial_count = 0
    _best_score_sofar = cur_score

    for p in range(1, passes + 1):
        improved = False
        for idx in range(n_rules):
            rule_ctx = rule_contexts[idx] if per_rule_symbols else None
            if per_rule_symbols and rule_ctx is not None:
                _, _, cur_rule_score = _evaluate_single_rule(
                    best_rules[idx], rule_ctx)
            else:
                cur_rule_score = cur_score

            local_best: tuple[float, list[dict], dict, dict] | None = None
            for tp in tp_grid:
                for sl in sl_grid:
                    for cap in cap_grid:
                        trial_count += 1

                        # --- progress report every 100 combos ---
                        if trial_count % 100 == 0:
                            logger.info(
                                "Phase 4 grid progress: trial %d/%d "
                                "(pass=%d, rule=%d/%d), best_score=%.2f",
                                trial_count, total_trials,
                                p, idx + 1, n_rules, _best_score_sofar,
                            )

                        trial = [dict(r) for r in best_rules]
                        trial[idx]["tp"] = tp
                        trial[idx]["sl"] = sl
                        trial[idx]["capital_pct"] = cap

                        # Skip if total capital exceeds the cap
                        total_cap = sum(
                            float(r.get("capital_pct", 0.0)) for r in trial)
                        if total_cap > max_total_cap:
                            continue

                        try:
                            if per_rule_symbols and rule_ctx is not None:
                                train_m, val_m, score = _evaluate_single_rule(
                                    trial[idx], rule_ctx)
                            else:
                                train_m, val_m, score = _evaluate_ruleset(
                                    train_engine, val_engine, trial,
                                    monthly_ctx=monthly_ctx,
                                    eval_cache=eval_cache)
                        except (ValueError, KeyError, TypeError) as exc:
                            logger.debug(
                                "grid trial failed (tp=%.2f, sl=%.2f, cap=%.2f): %s",
                                tp, sl, cap, exc)
                            continue

                        # Must pass gate_positive_good
                        if not gate_positive_good(train_m, val_m):
                            continue

                        train_ret = float(train_m.get("total_return_pct", 0.0))
                        val_ret = float(val_m.get("total_return_pct", 0.0))
                        gap_max = float(
                            getattr(_cfg, "PHASE4_MAX_VAL_TRAIN_GAP_PCT", 12.0))
                        if val_ret - train_ret > gap_max:
                            continue

                        if local_best is None or score > local_best[0]:
                            local_best = (score, trial, train_m, val_m)
                            if score > _best_score_sofar:
                                _best_score_sofar = score

            accept_threshold = (
                cur_rule_score + min_improvement
                if per_rule_symbols
                else cur_score + min_improvement
            )
            if local_best is not None and local_best[0] > accept_threshold:
                if per_rule_symbols:
                    cur_rule_score = local_best[0]
                    best_rules[idx] = dict(local_best[1][idx])
                else:
                    cur_score, best_rules, cur_train, cur_val = local_best
                improved = True
                cur_train, cur_val, cur_score = _evaluate_ruleset(
                    train_engine, val_engine, best_rules,
                    monthly_ctx=monthly_ctx, eval_cache=eval_cache)
                hist.append({
                    "pass": p,
                    "rule_index": idx + 1,
                    "score": cur_score,
                    "rule_score": (
                        float(local_best[0]) if per_rule_symbols else None
                    ),
                    "symbols": (
                        list(rule_ctx.symbols)
                        if per_rule_symbols and rule_ctx is not None
                        else None
                    ),
                    "train_return_pct": float(cur_train.get("total_return_pct", 0.0)),
                    "valid_return_pct": float(cur_val.get("total_return_pct", 0.0)),
                    "train_pf": float(cur_train.get("profit_factor", 0.0)),
                    "valid_pf": float(cur_val.get("profit_factor", 0.0)),
                    "train_dd": float(cur_train.get("max_drawdown_pct", 0.0)),
                    "valid_dd": float(cur_val.get("max_drawdown_pct", 0.0)),
                    "tp": best_rules[idx]["tp"],
                    "sl": best_rules[idx]["sl"],
                    "capital_pct": best_rules[idx]["capital_pct"],
                })
                logger.info(
                    "Phase 4 grid [%s]: improve pass=%d rule=%d score=%.2f "
                    "rule_score=%.2f symbols=%s train=%.2f%% val=%.2f%%",
                    getattr(train_engine, "direction", "?"),
                    p, idx + 1, cur_score,
                    float(local_best[0]) if per_rule_symbols else cur_score,
                    (
                        rule_ctx.symbols
                        if per_rule_symbols and rule_ctx is not None
                        else "all"
                    ),
                    float(cur_train.get("total_return_pct", 0.0)),
                    float(cur_val.get("total_return_pct", 0.0)),
                )

        if not improved:
            logger.info(
                "Phase 4 grid: no improvement in pass %d; stopping early.",
                p)
            break

    # Final evaluation with optimized rules
    final_train, final_val, final_score = _evaluate_ruleset(
        train_engine, val_engine, best_rules,
        monthly_ctx=monthly_ctx, eval_cache=eval_cache)
    return best_rules, final_train, final_val, final_score, hist


def _load_rule_set(path: str) -> Optional[dict]:
    """Load and return rule set JSON, or None if missing/invalid."""
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None
    return data


def _tp_sl_ratio_valid(tp: float, sl: float) -> bool:
    """Return True when TP exceeds SL by at least PHASE4_MIN_TP_SL_RATIO."""
    if sl <= 0.0:
        return False
    return tp > sl and (tp / sl) >= float(_cfg.PHASE4_MIN_TP_SL_RATIO)


def _params_within_bounds(rule_set: dict) -> bool:
    """Return True if all TP/SL/capital_pct values are within Phase 4 bounds."""
    if not isinstance(rule_set, dict):
        return False

    rules = rule_set.get("rules_set", [])
    if not isinstance(rules, list) or not rules:
        return False
    for rule in rules:
        if not isinstance(rule, dict):
            return False
        try:
            tp = float(rule.get("tp", 0.0))
            sl = float(rule.get("sl", 0.0))
            cap = float(rule.get("capital_pct", 0.0))
        except (TypeError, ValueError):
            return False

        if not (_cfg.PHASE4_TP_MIN <= tp <= _cfg.PHASE4_TP_MAX):
            return False
        if not (_cfg.PHASE4_SL_MIN <= sl <= _cfg.PHASE4_SL_MAX):
            return False
        if not _tp_sl_ratio_valid(tp, sl):
            return False
        if not (0.0 < cap <= _cfg.PHASE4_CAPITAL_PCT_MAX):
            return False
    return True


def _is_risk_optimized(rule_set: dict) -> bool:
    return rule_set.get("risk_optimized") is True


class WalkForwardRiskOptimizer:
    """
    Phase 4: Optuna walk-forward risk optimizer.

    Parameters
    ----------
    val_df : pd.DataFrame
        Persisted validation split (validation_25) for deployment gate metrics.
    rule_set : dict
        Rule set dict from Phase 3 (evaluator_v3.ipynb format).
    direction : str
        ``"long"`` or ``"short"``.
    train_df : pd.DataFrame | None
        Training split dataframe for grid-search train engine (separate from
        ``val_df``).  When ``None`` (backward compat), the grid search falls
        back to ``val_df`` for both train and val engines.
    n_trials : int | None
        Override ``PHASE4_N_TRIALS`` (useful for testing).
    n_splits : int | None
        Override ``PHASE4_WF_SPLITS``.
    seed : int | None
        Override ``PHASE4_SEED``.
  """

    def __init__(
        self,
        val_df: pd.DataFrame,
        rule_set: dict,
        direction: str,
        train_df: pd.DataFrame | None = None,
        n_trials: int | None = None,
        n_splits: int | None = None,
        seed: int | None = None,
    ) -> None:
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        rules = rule_set.get("rules_set", [])
        if not rules:
            raise ValueError(
                "rule_set must contain at least one rule in 'rules_set'")

        self.val_df = val_df
        self.train_df = train_df
        self.rule_set = rule_set
        self.direction = direction
        self.n_trials = 1
        self.n_splits = (
            n_splits if n_splits is not None else _cfg.effective_phase4_wf_splits()
        )
        self.seed = seed if seed is not None else 42
        self._selected_trial: Any = None
        self._study: Any = None

    def _write_non_optimized_output(self, study: Any) -> dict:
        """Persist Phase 3 rules unchanged when no feasible WF trial exists."""
        rules = self.rule_set.get("rules_set", [])
        val_engine = CPUBacktestEngine(self.val_df, {}, self.direction)
        val_metrics = val_engine.simulate_rule_set(rules)
        val_ret = float(val_metrics.get("total_return_pct", 0.0))
        val_pf = float(val_metrics.get("profit_factor", 0.0))
        ret_gate = float(_cfg.PHASE5_VALIDATION_RETURN_GATE_PCT)
        pf_gate = float(_cfg.PHASE5_VALIDATION_PROFIT_FACTOR_GATE)

        output_dict = {
            "direction": self.direction,
            "risk_optimized": False,
            "deployment_accepted": False,
            "validation_gate": {
                "return_pct": val_ret,
                "profit_factor": val_pf,
                "required_return_pct": ret_gate,
                "required_profit_factor": pf_gate,
            },
            "rules_set": [
                {
                    "conditions": list(r["conditions"]),
                    "tp": float(r.get("tp", _cfg.PHASE2_TP)),
                    "sl": float(r.get("sl", _cfg.PHASE2_SL)),
                    "capital_pct": float(r.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
                }
                for r in rules
            ],
        }

        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        output_path = _OUTPUT_PATHS[self.direction]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2)
        _maybe_write_evaluator_clean(output_dict, output_path, self.direction)

        try:
            Reporter().plot_phase4_pareto(study.trials, None, self.direction)
        except Exception as exc:
            logger.warning(
                "Reporter.plot_phase4_pareto failed (non-fatal): %s", exc
            )

        return output_dict

    def train(self) -> dict:
        """
        Run deterministic grid search for TP/SL/capital instead of Optuna.

        Uses ``_optimize_risk_grid`` internally and writes the optimized
        rule set to ``outputs/{direction}.json``.  The existing
        ``train()`` (Optuna) path is preserved behind the
        ``PHASE4_GRID_ENABLED`` flag.

        Returns
        -------
        dict
            Optimized rule set (evaluator-compatible format).
        """
        rules = self.rule_set.get("rules_set", [])
        if not rules:
            raise ValueError(
                "Phase 4 grid: empty rules_set; nothing to optimize")

        # Use separate train/val dataframes when train_df is provided (Task 7 code review).
        # This restores the purpose of gate_positive_good, which checks that rules
        # perform well on BOTH train and validation data.
        # When train_df is None (backward compat), fall back to val_df only.
        _train_df = self.train_df if self.train_df is not None else self.val_df
        train_engine = CPUBacktestEngine(
            _train_df, {}, self.direction)
        val_engine = CPUBacktestEngine(
            self.val_df, {}, self.direction)

        eval_cache = build_rules_signal_cache(
            rules, _train_df, self.val_df)

        monthly_ctx = _build_phase4_monthly_context(
            _train_df, self.val_df, self.direction, rules)

        per_rule_symbols = bool(
            getattr(_cfg, "PHASE4_OPTIMIZE_PER_RULE_SYMBOL", True))
        rule_contexts = (
            _build_rule_opt_contexts(
                rules, _train_df, self.val_df, self.direction,
            )
            if per_rule_symbols
            else None
        )

        min_improvement = float(
            getattr(_cfg, "PHASE4_GRID_MIN_IMPROVEMENT", 0.02))

        # Log grid dimensions for progress visibility
        tp_grid = tuple(float(x) for x in getattr(
            _cfg, "PHASE4_GRID_TP_VALUES",
            (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)))
        sl_grid = tuple(float(x) for x in getattr(
            _cfg, "PHASE4_GRID_SL_VALUES",
            (1.0, 1.2, 1.5, 2.0, 2.5, 3.0)))
        cap_grid = tuple(float(x) for x in getattr(
            _cfg, "PHASE4_GRID_CAPITAL_VALUES",
            (5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 35.0, 50.0)))
        max_total_cap = float(getattr(_cfg, "PHASE4_GRID_MAX_TOTAL_CAPITAL", 95.0))
        passes = int(getattr(_cfg, "PHASE4_GRID_PASSES", 2))

        total_combos = len(tp_grid) * len(sl_grid) * len(cap_grid)
        n_rules = len(rules)
        n_backtests_per_trial = 2  # train + val
        expected_trials = total_combos * n_rules * passes * n_backtests_per_trial

        logger.info(
            "Phase 4 grid [%s]: grid_size=%d (tp=%d × sl=%d × capital=%d), "
            "rules=%d, passes=%d, expected_backtests≈%d, "
            "min_improvement=%.2f, max_total_capital=%.1f%%, monthly=%s, "
            "per_rule_symbol=%s",
            self.direction, total_combos, len(tp_grid), len(sl_grid), len(cap_grid),
            n_rules, passes, expected_trials, min_improvement, max_total_cap,
            "on" if monthly_ctx is not None else "off",
            "on" if rule_contexts is not None else "off",
        )

        optimized_rules, train_metrics, val_metrics, score, history = (
            _optimize_risk_grid(
                rules,
                train_engine,
                val_engine,
                monthly_ctx=monthly_ctx,
                eval_cache=eval_cache,
                rule_contexts=rule_contexts,
                min_improvement=min_improvement,
            )
        )

        logger.info(
            "Phase 4 grid [%s]: score=%.2f train_ret=%.2f%% "
            "val_ret=%.2f%% rules=%d passes_seen=%d",
            self.direction,
            score,
            float(train_metrics.get("total_return_pct", 0.0)),
            float(val_metrics.get("total_return_pct", 0.0)),
            len(optimized_rules),
            len([h for h in history if h.get("pass", 0) > 0]),
        )

        # Normalize capital pct
        optimized_rules = _normalize_capital_pct(optimized_rules)
        cap = float(_cfg.PHASE3_MAX_CAPITAL_PCT_PER_RULE)
        for rule in optimized_rules:
            rule["capital_pct"] = min(
                float(rule.get("capital_pct", 0.0)), cap)
        optimized_rules = _normalize_capital_pct(optimized_rules)

        val_metrics_final = val_engine.simulate_rule_set_from_cache(
            optimized_rules, eval_cache, "val")
        val_ret = float(val_metrics_final.get("total_return_pct", 0.0))
        val_pf = float(val_metrics_final.get("profit_factor", 0.0))
        ret_gate = float(_cfg.PHASE5_VALIDATION_RETURN_GATE_PCT)
        pf_gate = float(_cfg.PHASE5_VALIDATION_PROFIT_FACTOR_GATE)
        deployable = (
            val_ret >= (ret_gate - 1e-9)
            and val_pf >= (pf_gate - 1e-9)
        )

        output_dict = {
            "direction": self.direction,
            "risk_optimized": bool(deployable),
            "deployment_accepted": bool(deployable),
            "validation_gate": {
                "return_pct": val_ret,
                "profit_factor": val_pf,
                "required_return_pct": ret_gate,
                "required_profit_factor": pf_gate,
            },
            "rules_set": optimized_rules,
        }

        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        output_path = _OUTPUT_PATHS[self.direction]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2)
        _maybe_write_evaluator_clean(output_dict, output_path, self.direction)

        logger.info(
            "Phase 4 grid [%s]: score=%.2f val_ret=%.2f%% val_pf=%.3f "
            "deployable=%s -> %s",
            self.direction, score, val_ret, val_pf, deployable, output_path,
        )

        return output_dict

    @staticmethod
    def skip_if_valid(direction: str) -> Optional[dict]:
        """
        Return loaded rule set if risk params were already optimized by Phase 4.

        Parameters
        ----------
        direction : str
            ``"long"`` or ``"short"``.

        Returns
        -------
        dict | None
            Loaded rule set if valid, else None.
        """
        if direction not in _OUTPUT_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        path = _OUTPUT_PATHS[direction]
        data = _load_rule_set(path)
        if data is None:
            return None

        if not _params_within_bounds(data):
            logger.info(
                "Phase 4 [%s]: existing file has out-of-bounds TP/SL/capital_pct; "
                "will re-run.",
                direction,
            )
            return None

        if not _is_risk_optimized(data):
            logger.info(
                "Phase 4 [%s]: existing file has not been risk-optimized; "
                "will re-run.",
                direction,
            )
            return None

        logger.info(
            "Phase 4 [%s]: existing file is risk-optimized and within bounds; "
            "skipping.",
            direction,
        )
        return data


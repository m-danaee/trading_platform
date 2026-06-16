"""
phase4_wf_optimizer.py — WalkForwardRiskOptimizer (Phase 4)

Fine-tunes TP, SL, and capital_pct for each rule in the selected rule set using
Optuna multi-objective walk-forward optimization.

When purged rolling CV folds are available, walk-forward windows are built on
every fold's validation block (not only validation_25). Rule conditions are
frozen from Phase 3. Each trial is scored on the worst return, drawdown, and
turnover across all windows.

Skip logic:
  If outputs/{direction}.json exists, TP/SL/capital_pct are within bounds,
  and risk_optimized is true, Phase 4 is skipped.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.output.writer import (
    _maybe_write_evaluator_clean,
    write_evaluator_clean,
)
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


def _walk_forward_splits_for_val_block(
    val_df: pd.DataFrame,
    k: int,
) -> list[pd.DataFrame]:
    """
    WF windows for one validation block; fall back to the whole block if too short.
    """
    if len(val_df) == 0:
        return []
    try:
        return build_phase4_walk_forward_splits(val_df, k)
    except ValueError:
        return [val_df.copy()]


def build_phase4_multi_cv_fold_splits(
    cv_folds: list[Any],
    k: int,
) -> list[pd.DataFrame]:
    """
    Walk-forward windows from each CV fold's validation block.

    Aggregates windows across all folds so inactive rules in one regime (e.g. high
    illiquidity on the last fold) still contribute to Optuna scoring.
    """
    all_splits: list[pd.DataFrame] = []
    for fold in cv_folds:
        val_df = getattr(fold, "val_df", None)
        if val_df is None and isinstance(fold, dict):
            val_df = fold.get("val_df")
        if val_df is None or len(val_df) == 0:
            continue
        all_splits.extend(_walk_forward_splits_for_val_block(val_df, k))
    if not all_splits:
        raise ValueError(
            "cv_folds produced no walk-forward windows for Phase 4"
        )
    return all_splits


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
        cv_folds: list[Any] | None = None,
    ) -> None:
        if cv_folds:
            splits = build_phase4_multi_cv_fold_splits(cv_folds, k)
        else:
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


def _score_metrics(train_m: dict, valid_m: dict) -> float:
    """Composite score for grid search (return / DD / PF weighted).

    This is a simplified port of the friend's ``_score_metrics`` from
    ``rb_governor.py``.  It emphasises validation return-to-drawdown ratio,
    profit factor, and penalises negative returns.
    """
    train_ret = float(train_m.get("total_return_pct", 0.0))
    train_dd = float(train_m.get("max_drawdown_pct", 100.0))
    train_pf = float(train_m.get("profit_factor", 0.0))
    train_wr = float(train_m.get("win_rate", 0.0))

    valid_ret = float(valid_m.get("total_return_pct", 0.0))
    valid_dd = float(valid_m.get("max_drawdown_pct", 100.0))
    valid_pf = float(valid_m.get("profit_factor", 0.0))
    valid_wr = float(valid_m.get("win_rate", 0.0))

    dd_floor = 0.50
    train_ratio = train_ret / max(train_dd, dd_floor)
    valid_ratio = valid_ret / max(valid_dd, dd_floor)

    def _pf_term(pf: float, cap: float = 5.0) -> float:
        return min(pf, cap)

    score = (
        120.0 * valid_ratio
        + 45.0 * train_ratio
        + 4.5 * valid_ret
        + 1.2 * train_ret
        + 14.0 * _pf_term(valid_pf)
        + 5.0 * _pf_term(train_pf)
        + 0.06 * valid_wr
        + 0.025 * train_wr
        - 0.25 * valid_dd
        - 0.08 * train_dd
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

    return float(score)


def _evaluate_ruleset(
    train_engine: CPUBacktestEngine,
    val_engine: CPUBacktestEngine,
    rules: list[dict],
) -> tuple[dict, dict, float]:
    """Evaluate a full rule set on train and val, returning metrics + score."""
    train_m = train_engine.simulate_rule_set(rules)
    val_m = val_engine.simulate_rule_set(rules)
    score = _score_metrics(train_m, val_m)
    return train_m, val_m, score


def _optimize_risk_grid(
    rules: list[dict],
    train_engine: CPUBacktestEngine,
    val_engine: CPUBacktestEngine,
    *,
    min_improvement: float = 0.02,
) -> tuple[list[dict], dict, dict, float, list[dict]]:
    """Per-rule round-robin grid search over TP, SL, capital_pct.

    For each rule, every (TP, SL, capital_pct) combination from the configured
    grid is tested.  Combinations that violate ``sum(capital_pct) <=
    PHASE4_GRID_MAX_TOTAL_CAPITAL`` or fail ``gate_positive_good`` are skipped.
    Two passes of round-robin per-rule tuning are performed.

    Parameters
    ----------
    rules : list[dict]
        Input rule set with ``tp``, ``sl``, ``capital_pct``.
    train_engine : CPUBacktestEngine
        Engine for training split evaluation.
    val_engine : CPUBacktestEngine
        Engine for validation split evaluation.
    min_improvement : float
        Minimum score improvement to accept a new combination (default 0.02).

    Returns
    -------
    tuple[list[dict], dict, dict, float, list[dict]]
        (optimized_rules, train_metrics, val_metrics, score, history)
    """
    from gpu_fuzzy_trader.phases.phase3_rule_set import gate_positive_good

    best_rules = [dict(r) for r in rules]

    # Initial evaluation
    cur_train, cur_val, cur_score = _evaluate_ruleset(
        train_engine, val_engine, best_rules)

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
                            train_m, val_m, score = _evaluate_ruleset(
                                train_engine, val_engine, trial)
                        except (ValueError, KeyError, TypeError) as exc:
                            logger.debug(
                                "grid trial failed (tp=%.2f, sl=%.2f, cap=%.2f): %s",
                                tp, sl, cap, exc)
                            continue

                        # Must pass gate_positive_good
                        if not gate_positive_good(train_m, val_m):
                            continue

                        if local_best is None or score > local_best[0]:
                            local_best = (score, trial, train_m, val_m)
                            if score > _best_score_sofar:
                                _best_score_sofar = score

            if local_best is not None and local_best[0] > cur_score + min_improvement:
                cur_score, best_rules, cur_train, cur_val = local_best
                improved = True
                hist.append({
                    "pass": p,
                    "rule_index": idx + 1,
                    "score": cur_score,
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
                    "train=%.2f%% val=%.2f%%",
                    getattr(train_engine, "direction", "?"),
                    p, idx + 1, cur_score,
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
        train_engine, val_engine, best_rules)
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
    cv_folds : list | None
        Purged rolling CV folds; when set, WF optimization uses every fold's val
        block instead of only *val_df*.
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
        cv_folds: list[Any] | None = None,
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
        self.cv_folds = cv_folds or []
        self.n_trials = 1
        self.n_splits = n_splits if n_splits is not None else _cfg.PHASE4_WF_SPLITS
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
            "min_improvement=%.2f, max_total_capital=%.1f%%",
            self.direction, total_combos, len(tp_grid), len(sl_grid), len(cap_grid),
            n_rules, passes, expected_trials, min_improvement, max_total_cap,
        )

        optimized_rules, train_metrics, val_metrics, score, history = (
            _optimize_risk_grid(
                rules,
                train_engine,
                val_engine,
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

        val_metrics_final = val_engine.simulate_rule_set(optimized_rules)
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


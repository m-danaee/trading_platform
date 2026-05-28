"""
phase4_wf_optimizer.py — WalkForwardRiskOptimizer (Phase 4)

Fine-tunes TP, SL, and capital_pct for each rule in the selected rule set using
Optuna multi-objective walk-forward optimization on the validation split.

Rule conditions are frozen from Phase 3. Each trial is scored on the worst
Sortino and worst max drawdown across K chronological validation windows.

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
from gpu_fuzzy_trader.reporting.reporter import Reporter

logger = logging.getLogger(__name__)

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


def _overalloc_penalty(params_list: list[dict]) -> float:
    total_cap = sum(float(p.get("capital_pct", 0.0)) for p in params_list)
    return (
        max(0.0, total_cap - 100.0) / 100.0 * _cfg.PHASE4_TOTAL_CAP_PENALTY
    )


def _evaluate_params_worst_case(
    val_splits: list[pd.DataFrame],
    direction: str,
    candidate_rule_set: list[dict],
    params_list: list[dict],
) -> tuple[float, float, float]:
    """
    Run backtest on each walk-forward window; return worst Sortino and worst DD.

    Each split uses a new CPUBacktestEngine instance (parallel-safe).
    """
    split_returns: list[float] = []
    split_drawdowns: list[float] = []
    split_turnover: list[float] = []

    for split_df in val_splits:
        engine = CPUBacktestEngine(split_df, {}, direction)
        try:
            metrics = engine.simulate_rule_set(candidate_rule_set)
        except Exception:
            split_returns.append(-100.0)
            split_drawdowns.append(100.0)
            split_turnover.append(0.0)
            continue

        split_returns.append(float(metrics.get("total_return_pct", 0.0)))
        split_drawdowns.append(float(metrics.get("max_drawdown_pct", 0.0)))
        split_turnover.append(float(metrics.get("executed_trades", 0.0)))

    penalty = _overalloc_penalty(params_list)
    worst_return = min(split_returns) - penalty
    worst_drawdown = max(split_drawdowns) + penalty
    worst_turnover = min(split_turnover)
    return worst_return, worst_drawdown, worst_turnover


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


def _create_sampler(seed: int):
    import optuna

    name = str(_cfg.PHASE4_SAMPLER).strip().lower()
    if name == "tpe":
        return optuna.samplers.TPESampler(multivariate=True, seed=seed)
    if name in ("nsga2", "nsga-ii", "nsgaii"):
        return optuna.samplers.NSGAIISampler(seed=seed)
    raise ValueError(
        f"PHASE4_SAMPLER must be 'nsga2' or 'tpe', got {_cfg.PHASE4_SAMPLER!r}"
    )


def _select_pareto_trial(study: Any, max_worst_dd_pct: float) -> Any:
    """
    Pick trial from Pareto front: filter by DD and min trades, then max worst return.

    Fallback if filter is empty: minimum drawdown objective, then max Sortino.
    """
    completed = [
        t for t in study.trials
        if t.state.name == "COMPLETE" and t.values is not None
    ]
    if not completed:
        raise RuntimeError("Phase 4: no completed Optuna trials")

    pareto = list(study.best_trials) if study.best_trials else completed
    candidates = []
    for t in pareto:
        values = t.values or ()
        if len(values) < 2:
            continue
        dd_ok = values[1] <= max_worst_dd_pct
        trades_ok = True
        if len(values) >= 3:
            trades_ok = values[2] >= _cfg.PHASE4_MIN_WORST_TRADES
        if dd_ok and trades_ok:
            candidates.append(t)
    pool = candidates if candidates else completed

    if candidates:
        return max(pool, key=lambda t: t.values[0])

    min_dd = min(t.values[1] for t in pool)
    tied = [t for t in pool if t.values[1] == min_dd]
    return max(tied, key=lambda t: t.values[0])


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
        if not (_cfg.PHASE4_CAPITAL_PCT_MIN <= cap <= _cfg.PHASE4_CAPITAL_PCT_MAX):
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
        Validation split DataFrame (already prepared).
    rule_set : dict
        Rule set dict from Phase 3 (evaluator_v3.ipynb format).
    direction : str
        ``"long"`` or ``"short"``.
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
        self.rule_set = rule_set
        self.direction = direction
        self.n_trials = n_trials if n_trials is not None else _cfg.PHASE4_N_TRIALS
        self.n_splits = n_splits if n_splits is not None else _cfg.PHASE4_WF_SPLITS
        self.seed = seed if seed is not None else _cfg.PHASE4_SEED
        self._selected_trial: Any = None
        self._study: Any = None

    def train(self) -> dict:
        """
        Run Optuna optimization and write optimized rule set to outputs JSON.

        Returns
        -------
        dict
            Optimized rule set (evaluator_v3.ipynb compatible format).
        """
        try:
            import optuna
        except ImportError as exc:
            raise ImportError(
                "Phase 4 requires Optuna. Install with: pip install optuna>=3.5.0"
            ) from exc

        optuna.logging.set_verbosity(optuna.logging.WARNING)

        rules = self.rule_set.get("rules_set", [])
        n_rules = len(rules)
        val_splits = split_validation_walk_forward(self.val_df, self.n_splits)
        n_jobs = int(_cfg.PHASE4_N_JOBS)

        logger.info(
            "Phase 4 [%s]: walk-forward Optuna (trials=%d, splits=%d, "
            "sampler=%s, n_jobs=%d, rules=%d)",
            self.direction,
            self.n_trials,
            self.n_splits,
            _cfg.PHASE4_SAMPLER,
            n_jobs,
            n_rules,
        )
        if n_jobs > 1:
            logger.info(
                "Phase 4 [%s]: n_jobs=%d — each trial uses isolated "
                "CPUBacktestEngine per split (read-only val_splits)",
                self.direction,
                n_jobs,
            )

        def objective(trial: "optuna.Trial") -> tuple[float, float, float]:
            params_list: list[dict] = []
            for i in range(n_rules):
                tp = trial.suggest_float(
                    f"tp_{i}",
                    _cfg.PHASE4_TP_MIN,
                    _cfg.PHASE4_TP_MAX,
                    step=_cfg.PHASE4_TP_STEP,
                )
                sl = trial.suggest_float(
                    f"sl_{i}",
                    _cfg.PHASE4_SL_MIN,
                    _cfg.PHASE4_SL_MAX,
                    step=_cfg.PHASE4_SL_STEP,
                )
                cap = trial.suggest_float(
                    f"capital_pct_{i}",
                    _cfg.PHASE4_CAPITAL_PCT_MIN,
                    _cfg.PHASE4_CAPITAL_PCT_MAX,
                    step=_cfg.PHASE4_CAPITAL_STEP,
                )
                params_list.append({"tp": tp, "sl": sl, "capital_pct": cap})

            candidate_rule_set = _build_candidate_rule_set(rules, params_list)
            worst_return, worst_drawdown, worst_turnover = _evaluate_params_worst_case(
                val_splits,
                self.direction,
                candidate_rule_set,
                params_list,
            )
            trial.set_user_attr("rule_set", candidate_rule_set)
            trial.set_user_attr("params_list", params_list)
            score_return = worst_return * float(_cfg.PHASE4_WORST_RETURN_WEIGHT)
            score_dd = worst_drawdown * float(_cfg.PHASE4_WORST_DRAWDOWN_WEIGHT)
            score_turnover = worst_turnover * float(_cfg.PHASE4_WORST_TURNOVER_WEIGHT)
            return score_return, score_dd, score_turnover

        sampler = _create_sampler(self.seed)
        study = optuna.create_study(
            directions=["maximize", "minimize", "maximize"],
            sampler=sampler,
        )
        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=n_jobs,
            show_progress_bar=False,
        )

        self._study = study
        selected = _select_pareto_trial(
            study, float(_cfg.PHASE4_MAX_WORST_DRAWDOWN_PCT)
        )
        self._selected_trial = selected

        optimized_params = list(selected.user_attrs.get("rule_set", []))
        if not optimized_params:
            raise RuntimeError(
                f"Phase 4 [{self.direction}]: selected trial has no rule_set"
            )

        optimized_params = _normalize_capital_pct(optimized_params)

        val_engine = CPUBacktestEngine(self.val_df, {}, self.direction)
        val_metrics = val_engine.simulate_rule_set(optimized_params)
        val_ret = float(val_metrics.get("total_return_pct", 0.0))
        val_pf = float(val_metrics.get("profit_factor", 0.0))
        ret_gate = float(_cfg.PHASE5_VALIDATION_RETURN_GATE_PCT)
        pf_gate = float(_cfg.PHASE5_VALIDATION_PROFIT_FACTOR_GATE)
        deployable = (
            val_ret >= (ret_gate - 1e-9)
            and val_pf >= (pf_gate - 1e-9)
        )
        if not deployable:
            logger.warning(
                "Phase 4 [%s]: validation gate failed (return=%.2f%%, profit_factor=%.3f). "
                "Marking strategy as non-deployable.",
                self.direction,
                val_ret,
                val_pf,
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
            "rules_set": optimized_params,
        }

        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        output_path = _OUTPUT_PATHS[self.direction]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2)

        logger.info(
            "Phase 4 [%s]: selected trial #%d (worst_return=%.4f%%, "
            "worst_drawdown=%.2f%%, worst_turnover=%.1f). Saved to %s",
            self.direction,
            selected.number,
            selected.values[0],
            selected.values[1],
            selected.values[2],
            output_path,
        )

        try:
            Reporter().plot_phase4_pareto(
                study.trials,
                selected,
                self.direction,
            )
        except Exception as exc:
            logger.warning(
                "Reporter.plot_phase4_pareto failed (non-fatal): %s", exc
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

    @property
    def study(self) -> Any:
        """Optuna study after train(); None before."""
        return self._study

    @property
    def selected_trial(self) -> Any:
        """Selected Pareto trial after train(); None before."""
        return self._selected_trial

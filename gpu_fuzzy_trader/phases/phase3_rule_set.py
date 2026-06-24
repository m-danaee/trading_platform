"""
phase3_rule_set.py — Rule_Set_Selector (Phase 3)

Per-symbol greedy rule selection from the Phase 2 pool.

For each symbol in the universe, selects 0–3 rules from the pool that perform
best on that symbol's validation data.  Rules selected for multiple symbols are
merged into a single output rule with "symbol is X" conditions.

Output:
    outputs/long.json  and  outputs/short.json
    (evaluator_v5.ipynb compatible format with "symbol is X" conditions)

Skip logic:
    If both files exist and pass schema validation → skip Phase 3.
    If only one exists → skip and proceed with available file.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.output.writer import (
    _maybe_write_evaluator_clean,
    write_evaluator_clean,
)
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.phases.phase3_cache import (
    Phase3EvalCache,
    build_phase3_eval_cache,
)
from gpu_fuzzy_trader.phases.phase3_objectives import (
    conditions_key as _conditions_key,
    count_symbols_with_trades as _count_symbols_with_trades,
    symbols_with_trades as _symbols_with_trades,
)
from gpu_fuzzy_trader.reporting.reporter import Reporter
from gpu_fuzzy_trader.validation.monthly_windows import (
    build_monthly_windows,
    evaluate_rule_set_monthly,
    monthly_penalty,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Positive-good gate (pure function, importable from other modules)
# ---------------------------------------------------------------------------


def gate_positive_good(
    train_metrics: dict,
    val_metrics: dict,
    *,
    min_train_return: float = 0.0,
    min_val_return: float = 0.0,
    min_train_pf: float = 1.0,
    min_val_pf: float = 1.0,
    min_train_trades: int = 25,
    min_val_trades: int = 15,
    require_execution_health: bool = False,
) -> bool:
    """Return ``True`` iff the rule is positive on both train and val.

    A rule passes this gate when *all* of the following hold:

    * ``total_return_pct > min_train_return`` on train **and** on val
    * ``profit_factor >= min_train_pf`` on train **and** ``>= min_val_pf`` on val
    * ``executed_trades >= min_train_trades`` on train **and** ``>= min_val_trades`` on val
    * If ``require_execution_health=True``, ``execution_ok(train_metrics)`` **and**
      ``execution_ok(val_metrics)`` must also pass (evaluator skip-rate gate).

    Missing or absent keys (``total_return_pct``, ``profit_factor``,
    ``executed_trades``, ``raw_signal_count`` when execution-health is on)
    are treated as hard failures and return ``False``.

    This is a **pure function** — no side effects, no engine calls, no IO.
    """
    def _safe_get(m: dict, key: str, default: float = 0.0) -> float:
        """Read a metric, returning *default* for missing, None, NaN, or Inf values."""
        val = m.get(key)
        if val is None:
            return float(default)
        try:
            f = float(val)
            return f if math.isfinite(f) else float(default)
        except (TypeError, ValueError):
            return float(default)

    def _safe_get_int(m: dict, key: str, default: int = 0) -> int:
        """Read an integer metric, returning *default* for missing, None, NaN, or Inf values."""
        f = _safe_get(m, key, default=float(default))
        return int(f) if math.isfinite(f) else int(default)

    # --- train checks ---
    train_ret = _safe_get(train_metrics, "total_return_pct")
    if train_ret <= min_train_return:
        return False

    train_pf = _safe_get(train_metrics, "profit_factor")
    if train_pf < min_train_pf:
        return False

    train_trades = _safe_get_int(train_metrics, "executed_trades")
    if train_trades < min_train_trades:
        return False

    # --- val checks ---
    val_ret = _safe_get(val_metrics, "total_return_pct")
    if val_ret <= min_val_return:
        return False

    val_pf = _safe_get(val_metrics, "profit_factor")
    if val_pf < min_val_pf:
        return False

    val_trades = _safe_get_int(val_metrics, "executed_trades")
    if val_trades < min_val_trades:
        return False

    # --- Optional evaluator execution-health gate (Task 4) ---
    if require_execution_health:
        from gpu_fuzzy_trader.scoring.evaluator_health import execution_ok

        if not execution_ok(train_metrics):
            return False
        if not execution_ok(val_metrics):
            return False

    return True


# ---------------------------------------------------------------------------
# Symbol condition helpers (multi-symbol combinations, Task 6)
# ---------------------------------------------------------------------------


def _is_symbol_condition(condition: str) -> bool:
    """Return ``True`` if *condition* is a ``symbol is X`` filter."""
    text = str(condition).strip().lower()
    return text.startswith("symbol is ") or text.startswith("[symbol] is ")


def _strip_symbol_conditions(conditions: list[str]) -> list[str]:
    """Return *conditions* without any ``symbol is X`` entries."""
    return [str(c) for c in conditions if not _is_symbol_condition(str(c))]


def _build_symbol_specialized_variants(
    rule: dict,
    train_engine: object,
    val_engine: object,
    symbols: list[str],
    *,
    eligible_symbols: list[str] | None = None,
) -> list[dict]:
    """Build top-K 1-, 2-, and 3-symbol variants of *rule*, scored and filtered.

    For each single-symbol combination of the *rule* on *symbols*, evaluate
    on train+val.  Keep those passing ``gate_positive_good`` with
    ``SYMBOL_SPECIALIZATION_MIN_TRAIN_TRADES`` /
    ``SYMBOL_SPECIALIZATION_MIN_VAL_TRADES``, and rank by
    ``min(train_return, val_return)``.

    When ``SYMBOL_SPECIALIZATION_USE_COMBINATIONS=True``, also generate all
    2- and 3-symbol combinations of every single-symbol variant that passes
    the trade floors, and score them similarly.

    Returns at most ``MAX_VARIANTS_PER_RULE`` variants, sorted descending by
    score.  Each variant has the *rule*'s original conditions (without any
    previous symbol filters) plus the new ``symbol is X`` conditions.

    Parameters
    ----------
    rule : dict
        Pool rule dict with ``conditions``, ``tp``, ``sl``, ``capital_pct``.
    train_engine : object
        Engine with ``simulate_rule_set`` method (train split).
    val_engine : object
        Engine with ``simulate_rule_set`` method (validation split).
    symbols : list[str]
        All available symbol labels in the universe.
    eligible_symbols : list[str] | None
        When set, only these symbols are considered for specialization
        (typically symbols that greedily selected this pool rule in Phase 3).

    Returns
    -------
    list[dict]
        Up to ``MAX_VARIANTS_PER_RULE`` variants, each a full rule dict with
        ``conditions``, ``tp``, ``sl``, ``capital_pct``.
    """
    use_combinations = bool(
        getattr(_cfg, "SYMBOL_SPECIALIZATION_USE_COMBINATIONS", True))
    max_symbols = max(
        1, int(getattr(_cfg, "SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE", 3)))
    max_variants = max(
        1, int(getattr(_cfg, "SYMBOL_SPECIALIZATION_MAX_VARIANTS_PER_RULE", 10)))
    min_train_trades = int(
        getattr(_cfg, "SYMBOL_SPECIALIZATION_MIN_TRAIN_TRADES", 10))
    min_val_trades = int(
        getattr(_cfg, "SYMBOL_SPECIALIZATION_MIN_VAL_TRADES", 6))

    # --- Early-exit cases ---

    # If the rule already has a symbol condition, return as-is (no re-specialization).
    if any(_is_symbol_condition(str(c)) for c in rule.get("conditions", [])):
        return [dict(rule)]

    # Base conditions without any existing symbol filters (belt-and-suspenders:
    # if we hit this condition above, base_conditions will be the same).
    base_conditions = _strip_symbol_conditions(
        list(rule.get("conditions", [])))

    if not symbols:
        return [dict(rule)]

    if eligible_symbols:
        eligible_set = {str(s) for s in eligible_symbols}
        symbols = [str(s) for s in symbols if str(s) in eligible_set]
        if not symbols:
            return [dict(rule)]

    # When combinations are disabled, only single-symbol variants are generated.
    # This is equivalent to the legacy Phase 3 behaviour.
    if not use_combinations:
        max_symbols = 1

    # ---- 1. Score single-symbol variants ----
    scored_singles: list[tuple[float, str]] = []
    for sym in symbols:
        variant = dict(rule)
        variant["conditions"] = base_conditions + [f"symbol is {sym}"]
        fmt = _rule_set_to_engine_format([variant])
        try:
            train_m = train_engine.simulate_rule_set(fmt)
            val_m = val_engine.simulate_rule_set(fmt)
        except Exception:
            continue

        if not gate_positive_good(
            train_m, val_m,
            min_train_trades=min_train_trades,
            min_val_trades=min_val_trades,
        ):
            continue

        score = min(
            float(train_m.get("total_return_pct", 0.0)),
            float(val_m.get("total_return_pct", 0.0)),
        )
        scored_singles.append((score, sym))

    scored_singles.sort(key=lambda x: x[0], reverse=True)
    eligible_syms = [sym for _, sym in scored_singles]

    # ---- 2. Build candidate symbol sets ----
    candidate_sets: list[tuple[str, ...]] = [(sym,) for sym in eligible_syms]

    if use_combinations and len(eligible_syms) >= 2:
        from itertools import combinations

        for k in range(2, min(max_symbols, len(eligible_syms)) + 1):
            for combo in combinations(eligible_syms, k):
                candidate_sets.append(tuple(combo))

    if not candidate_sets:
        # Fallback: return a single variant with one symbol or the original rule.
        if eligible_syms:
            variant = dict(rule)
            variant["conditions"] = base_conditions + [
                f"symbol is {eligible_syms[0]}"]
            return [variant]
        return [dict(rule)]

    # ---- 3. Score all candidates, gate-filter, sort, keep top-K ----
    scored_variants: list[tuple[float, dict]] = []
    seen_sets: set[tuple[str, ...]] = set()

    for sym_set in candidate_sets:
        # Deduplicate while preserving order.
        sym_set = tuple(dict.fromkeys(str(s) for s in sym_set))
        if not sym_set or sym_set in seen_sets:
            continue
        seen_sets.add(sym_set)

        variant = dict(rule)
        variant["conditions"] = (
            base_conditions + [f"symbol is {s}" for s in sym_set]
        )
        fmt = _rule_set_to_engine_format([variant])
        try:
            train_m = train_engine.simulate_rule_set(fmt)
            val_m = val_engine.simulate_rule_set(fmt)
        except Exception:
            continue

        if not gate_positive_good(
            train_m, val_m,
            min_train_trades=min_train_trades,
            min_val_trades=min_val_trades,
        ):
            continue

        score = min(
            float(train_m.get("total_return_pct", 0.0)),
            float(val_m.get("total_return_pct", 0.0)),
        )
        scored_variants.append((score, variant))

    scored_variants.sort(key=lambda x: x[0], reverse=True)
    return [v for _, v in scored_variants[:max_variants]]


def _build_per_symbol_assigned_rules(
    symbol_assignments: dict[str, list[int]],
    pool: list[dict],
) -> list[dict]:
    """One rule per greedy (symbol, pool_idx) pair — preserves Phase 3 selections."""
    merged: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for sym in sorted(symbol_assignments.keys(), key=str):
        for pool_idx in symbol_assignments[sym]:
            key = (int(pool_idx), str(sym))
            if key in seen:
                continue
            seen.add(key)
            rule = dict(pool[int(pool_idx)])
            base_conditions = _strip_symbol_conditions(
                list(rule.get("conditions", [])))
            rule["conditions"] = base_conditions + [f"symbol is {sym}"]
            rule["_pool_idx"] = int(pool_idx)
            merged.append(rule)
    return merged


def _build_multi_symbol_merged_rules(
    symbol_assignments: dict[str, list[int]],
    pool: list[dict],
    train_engine: object,
    val_engine: object,
    symbols: list[str],
) -> list[dict]:
    """Build merged rules using multi-symbol combinations.

    For each unique pool rule selected by at least one symbol, call
    ``_build_symbol_specialized_variants`` to find the best symbol
    combination.  The best variant replaces the naive "append all symbols"
    merge.

    Parameters
    ----------
    symbol_assignments : dict[str, list[int]]
        Mapping of symbol -> list of pool rule indices selected for that symbol.
    pool : list[dict]
        The enriched pool of candidate rules.
    train_engine : object
        Engine for train-split simulation.
    val_engine : object
        Engine for validation-split simulation.
    symbols : list[str]
        Full symbol universe.

    Returns
    -------
    list[dict]
        Merged rules ready for sorting and final output.
    """
    selected_indices: set[int] = set()
    pool_idx_to_symbols: dict[int, list[str]] = {}
    for sym, indices in symbol_assignments.items():
        for idx in indices:
            selected_indices.add(idx)
            pool_idx_to_symbols.setdefault(int(idx), []).append(str(sym))

    merged: list[dict] = []
    for pool_idx in sorted(selected_indices):
        rule = dict(pool[pool_idx])
        rule["_pool_idx"] = int(pool_idx)

        eligible = pool_idx_to_symbols.get(int(pool_idx), [])
        variants = _build_symbol_specialized_variants(
            rule, train_engine, val_engine, symbols,
            eligible_symbols=eligible or None,
        )
        if variants:
            # variants are sorted by score descending; pick the best.
            best = variants[0]
            best["_pool_idx"] = int(pool_idx)
            merged.append(best)
        elif eligible:
            # Never emit an unscoped rule: bind to the first greedy symbol.
            fallback = dict(rule)
            base_conditions = _strip_symbol_conditions(
                list(rule.get("conditions", [])))
            fallback["conditions"] = base_conditions + [
                f"symbol is {eligible[0]}"]
            fallback["_pool_idx"] = int(pool_idx)
            merged.append(fallback)
        else:
            merged.append(rule)

    return merged


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_OUTPUT_PATHS = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "long.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "short.json"),
}


# ---------------------------------------------------------------------------
# Output JSON schema validation
# ---------------------------------------------------------------------------


def _validate_rule_set_schema(data: object, path: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(
            f"Rule set must be a JSON object, got {type(data).__name__}: {path}"
        )
    required_top = {"direction", "rules_set"}
    missing = required_top - set(data.keys())
    if missing:
        raise ValueError(
            f"Rule set missing top-level keys {missing}: {path}"
        )
    if data["direction"] not in ("long", "short"):
        raise ValueError(
            f"Rule set 'direction' must be 'long' or 'short': {path}"
        )
    rules_set = data["rules_set"]
    if not isinstance(rules_set, list):
        raise ValueError(
            f"Rule set 'rules_set' must be a list: {path}"
        )
    min_rules = int(_cfg.PHASE3_GLOBAL_MIN_RULES)
    max_rules = int(_cfg.PHASE3_GLOBAL_MAX_RULES)
    if not (min_rules <= len(rules_set) <= max_rules):
        selection_accepted = data.get("selection_accepted")
        if selection_accepted is False and len(rules_set) == 0:
            pass
        else:
            raise ValueError(
                f"Rule set 'rules_set' must have {min_rules}–{max_rules} rules, "
                f"got {len(rules_set)}: {path}"
            )
    for i, rule in enumerate(rules_set):
        if not isinstance(rule, dict):
            raise ValueError(
                f"Rule set entry {i} must be a dict: {path}"
            )
        required_rule_keys = {"tp", "sl", "capital_pct", "conditions"}
        missing_rule = required_rule_keys - set(rule.keys())
        if missing_rule:
            raise ValueError(
                f"Rule set entry {i} missing keys {missing_rule}: {path}"
            )
        if not isinstance(rule["conditions"], list) or len(rule["conditions"]) == 0:
            raise ValueError(
                f"Rule set entry {i} 'conditions' must be a non-empty list: {path}"
            )
    if "risk_optimized" in data and not isinstance(data["risk_optimized"], bool):
        raise ValueError(
            f"Rule set 'risk_optimized' must be a bool if present: {path}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_team(
    rule_set: list[dict],
    train_engine,
    val_engine,
    cache: Phase3EvalCache | None,
) -> tuple[dict, dict]:
    if cache is not None and hasattr(train_engine, "simulate_rule_set_from_cache"):
        try:
            train_metrics = train_engine.simulate_rule_set_from_cache(
                rule_set, cache, "train")
            val_metrics = val_engine.simulate_rule_set_from_cache(
                rule_set, cache, "val")
            return train_metrics, val_metrics
        except Exception as exc:
            logger.debug("cached simulate failed, falling back: %s", exc)

    try:
        train_metrics = train_engine.simulate_rule_set(rule_set)
    except Exception as exc:
        logger.debug("train simulate_rule_set failed: %s", exc)
        train_metrics = _empty_metrics()
    try:
        val_metrics = val_engine.simulate_rule_set(rule_set)
    except Exception as exc:
        logger.debug("val simulate_rule_set failed: %s", exc)
        val_metrics = _empty_metrics()
    return train_metrics, val_metrics


def _empty_metrics() -> dict:
    return {
        "sortino_ratio": 0.0,
        "total_return_pct": 0.0,
        "max_drawdown_pct": 100.0,
        "win_rate": 0.0,
        "executed_trades": 0,
        "per_symbol_metrics": {},
    }


def _build_phase3_engines(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    direction: str,
    pool: list[dict],
) -> tuple[object, object, Phase3EvalCache]:
    feature_modes: dict[str, str] = {}
    use_jax = bool(_cfg.PHASE3_USE_GPU)

    val_engine: object
    train_engine: object

    if use_jax:
        from gpu_fuzzy_trader.backtest.jax_compat import get_gpu_backtest_engine_class

        GPUBacktestEngine = get_gpu_backtest_engine_class()
        if GPUBacktestEngine is not None:
            val_engine = GPUBacktestEngine(
                val_df, feature_modes, direction, fee_pct=_cfg.FEE_PCT,
            )
            train_engine = GPUBacktestEngine(
                train_df, feature_modes, direction, fee_pct=_cfg.FEE_PCT,
            )
            logger.info("Phase 3 using GPUBacktestEngine (JAX mask + batch eval)")
        else:
            logger.warning("PHASE3_USE_GPU=True but JAX/GPU unavailable; using CPU.")
            use_jax = False
            val_engine = CPUBacktestEngine(
                val_df, feature_modes, direction, fee_pct=_cfg.FEE_PCT,
            )
            train_engine = CPUBacktestEngine(
                train_df, feature_modes, direction, fee_pct=_cfg.FEE_PCT,
            )
    else:
        val_engine = CPUBacktestEngine(
            val_df, feature_modes, direction, fee_pct=_cfg.FEE_PCT,
        )
        train_engine = CPUBacktestEngine(
            train_df, feature_modes, direction, fee_pct=_cfg.FEE_PCT,
        )

    cache = build_phase3_eval_cache(pool, train_df, val_df, val_engine)
    return val_engine, train_engine, cache


def _pool_rule_val_score(rule: dict) -> float:
    val_obj = rule.get("val_objectives") or rule.get("objectives") or {}
    train_obj = rule.get("objectives") or {}
    val_ret = float(val_obj.get("total_return_pct", 0.0))
    train_ret = float(train_obj.get("total_return_pct", 0.0))
    return min(val_ret, train_ret)


def _pool_rule_passes_gap_gate(rule: dict) -> bool:
    """Return False when stored train/val returns show classic overfit (train >> val)."""
    val_obj = rule.get("val_objectives") or rule.get("objectives") or {}
    train_obj = rule.get("objectives") or {}
    val_ret = float(val_obj.get("total_return_pct", 0.0))
    train_ret = float(train_obj.get("total_return_pct", 0.0))
    max_gap = float(getattr(_cfg, "PHASE3_MAX_TRAIN_VAL_GAP_PCT", 40.0))
    return train_ret - val_ret <= max_gap


def _try_global_pool_fallback(
    pool: list[dict],
    train_engine,
    val_engine,
    eval_cache: Phase3EvalCache | None,
    direction: str,
    global_min: int,
) -> list[dict] | None:
    """Pick top pool rules when per-symbol greedy finds too few."""
    fallback = _best_rules_from_pool_fallback(pool, global_min)
    engine_fmt = _rule_set_to_engine_format(fallback)
    train_m, val_m = _simulate_team(
        engine_fmt, train_engine, val_engine, eval_cache,
    )

    # Positive-good gate on the fallback team (Task 3).
    if bool(getattr(_cfg, "PHASE3_REQUIRE_POSITIVE_GOOD", True)):
        if not gate_positive_good(
            train_m,
            val_m,
            min_train_return=float(
                getattr(_cfg, "PHASE3_MIN_TRAIN_RETURN", 0.0)),
            min_val_return=float(
                getattr(_cfg, "PHASE3_MIN_VAL_RETURN", 0.0)),
            min_train_pf=float(
                getattr(_cfg, "PHASE3_MIN_TRAIN_PF", 1.0)),
            min_val_pf=float(
                getattr(_cfg, "PHASE3_MIN_VAL_PF", 1.0)),
            min_train_trades=int(
                getattr(_cfg, "PHASE3_MIN_TRAIN_TRADES", 25)),
            min_val_trades=int(
                getattr(_cfg, "PHASE3_MIN_VAL_TRADES", 15)),
            require_execution_health=bool(
                getattr(_cfg, "PHASE3_GATE_EXECUTION_HEALTH", True)),
        ):
            logger.warning(
                "Phase 3 [%s]: fallback failed positive-good gate",
                direction,
            )
            return None

    val_floor = float(_cfg.effective_phase3_val_return_floor_pct())
    robust_ret = min(
        float(train_m.get("total_return_pct", -999.0)),
        float(val_m.get("total_return_pct", -999.0)),
    )
    if robust_ret > val_floor:
        logger.info(
            "Phase 3 [%s]: fallback produced %d global rules (min(train,val)=%.2f%%)",
            direction, len(fallback), robust_ret,
        )
        return fallback
    logger.warning(
        "Phase 3 [%s]: fallback failed robust return floor (min(train,val)=%.2f%% <= %.2f%%)",
        direction,
        robust_ret,
        val_floor,
    )
    return None


def _try_lean_fallback(
    pool: list[dict],
    direction: str,
    global_min: int = 2,
) -> list[dict] | None:
    """Pick top-*global_min* pool rules — relaxed but with val_floor sanity check.

    This is a deliberate relaxation for when both the per-symbol greedy
    selection **and** the strict ``_try_global_pool_fallback`` have failed.
    It applies only the ``val_floor`` (from
    ``effective_phase3_val_return_floor_pct``) as a sanity check, then picks
    the top-*global_min* surviving rules by ``_pool_rule_val_score``
    (``min(train_ret, val_ret)``) **without** calling:

    * ``gate_positive_good`` (PF ≥ 1.0 / min-trades gate)
    * ``_simulate_team`` (full-team backtest)

    Why this is OK
    --------------
    * The ``val_floor`` sanity check (default 2.5%) prevents clearly
      negative-return teams from leaking through.
    * Phase 4 risk optimization and Phase 5 evaluator scoring will still
      filter out truly broken rules.
    * The pool itself has already passed Phase 2's quality filters (CV,
      spec consistency, etc.).
    * Even a 2-rule team with mediocre PF gives the hidden-test pipeline
      *something* to grade instead of skipping OOS entirely.

    Parameters
    ----------
    pool : list[dict]
        Phase 2 pool (enriched with tp/sl/capital_pct defaults).
    direction : str
        ``"long"`` or ``"short"`` — used only for logging.
    global_min : int
        Minimum number of rules to return.  Default 2.

    Returns
    -------
    list[dict] | None
        Top-*global_min* rules sorted by ``_pool_rule_val_score`` descending,
        or ``None`` if fewer than ``global_min`` rules survive the
        ``val_floor`` filter.
    """
    n_rules = max(1, int(global_min))

    if len(pool) < n_rules:
        logger.warning(
            "Phase 3 [%s]: _try_lean_fallback — pool has %d rule(s), "
            "need at least %d.  Cannot pick lean fallback.",
            direction, len(pool), n_rules,
        )
        return None

    val_floor = float(_cfg.effective_phase3_val_return_floor_pct())

    # Filter by val_floor and train-val gap (overfit rejection).
    above_floor = [
        r for r in pool
        if _pool_rule_val_score(r) > val_floor and _pool_rule_passes_gap_gate(r)
    ]

    if len(above_floor) < n_rules:
        logger.warning(
            "Phase 3 [%s]: _try_lean_fallback — only %d rule(s) above "
            "val_floor=%.2f%% and within train-val gap, need at least %d.  "
            "Cannot pick lean fallback.",
            direction, len(above_floor), val_floor, n_rules,
        )
        return None

    logger.warning(
        "Phase 3 [%s]: _try_lean_fallback — applying relaxed fallback. "
        "Per-symbol greedy and _try_global_pool_fallback both failed; "
        "picking top-%d pool rules by min(train_ret, val_ret) with "
        "val_floor=%.2f%% sanity check (positive-good gate, team backtest "
        "NOT applied).  Phase 4 risk optimization and Phase 5 evaluator "
        "will still filter out weak rules.",
        direction, n_rules, val_floor,
    )

    ranked = sorted(above_floor, key=_pool_rule_val_score, reverse=True)
    selected = ranked[:n_rules]
    logger.warning(
        "Phase 3 [%s]: lean fallback — returning top %d rules "
        "(min(train,val)=%.2f%%, val_floor=%.2f%%).",
        direction, len(selected),
        min(_pool_rule_val_score(r) for r in selected),
        val_floor,
    )
    return selected


def _best_rules_from_pool_fallback(
    pool: list[dict],
    n_rules: int,
) -> list[dict]:
    ranked = sorted(pool, key=_pool_rule_val_score, reverse=True)
    out: list[dict] = []
    seen: set[frozenset] = set()
    for rule in ranked:
        key = _conditions_key(rule.get("conditions", []))
        if key in seen:
            continue
        seen.add(key)
        out.append(rule)
        if len(out) >= n_rules:
            break
    return out


def _rule_set_to_engine_format(rule_set: list[dict]) -> list[dict]:
    return [
        {
            "conditions": rule["conditions"],
            "tp": float(rule.get("tp", _cfg.PHASE2_TP)),
            "sl": float(rule.get("sl", _cfg.PHASE2_SL)),
            "capital_pct": float(rule.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
        }
        for rule in rule_set
    ]


def _monthly_feature_names(df: pd.DataFrame) -> list[str]:
    """Column whitelist for ``evaluate_rule_set_monthly`` / ``slim_backtest_df``."""
    return [
        c for c in df.columns
        if c not in set(_cfg.LABEL_COLUMNS)
        | set(_cfg.META_COLUMNS)
        | set(_cfg.INTERNAL_COLUMNS)
        and not str(c).startswith("_")
    ]


def _build_per_symbol_monthly_context(
    train_symbol_df: pd.DataFrame | None,
    val_symbol_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, list[pd.DataFrame] | None]:
    """Build train+val combined slice and sequential monthly windows for one symbol."""
    parts: list[pd.DataFrame] = []
    if train_symbol_df is not None and len(train_symbol_df) > 0:
        parts.append(train_symbol_df)
    if val_symbol_df is not None and len(val_symbol_df) > 0:
        parts.append(val_symbol_df)
    if not parts:
        return None, None
    combined = pd.concat(parts, ignore_index=True)
    return combined, build_monthly_windows(combined)


# ---------------------------------------------------------------------------
# Per-symbol greedy selection
# ---------------------------------------------------------------------------


def _cv_worst_val_for_rule(
    rule: dict,
    direction: str,
    cv_folds: list,
    *,
    symbol: str | None = None,
) -> tuple[float, dict, int]:
    """Worst-fold validation return for a rule (optionally per symbol)."""
    from gpu_fuzzy_trader.validation.rolling_cv import cv_folds_only

    fmt = _rule_set_to_engine_format([rule])
    fold_returns: list[float] = []
    fold_metrics: list[dict] = []
    sym_row_counts: list[int] = []

    for fold in cv_folds_only(cv_folds):
        if symbol is not None and "symbol" in fold.valid_df.columns:
            sym_df = fold.valid_df[
                fold.valid_df["symbol"].astype(str) == str(symbol)
            ].reset_index(drop=True)
        else:
            sym_df = fold.valid_df

        if len(sym_df) == 0:
            fold_returns.append(-999.0)
            fold_metrics.append({})
            sym_row_counts.append(0)
            continue

        sym_row_counts.append(len(sym_df))
        engine = CPUBacktestEngine(sym_df, {}, direction, fee_pct=_cfg.FEE_PCT)
        try:
            metrics = engine.simulate_rule_set(fmt)
            fold_returns.append(float(metrics.get("total_return_pct", -999.0)))
            fold_metrics.append(metrics)
        except Exception:
            fold_returns.append(-999.0)
            fold_metrics.append({})

    if not fold_returns:
        return -999.0, {}, 0

    worst_idx = int(np.argmin(fold_returns))
    min_sym_rows = min((r for r in sym_row_counts if r > 0), default=0)
    return fold_returns[worst_idx], fold_metrics[worst_idx], min_sym_rows


def _score_pool_rule_on_symbol(
    rule: dict,
    symbol_df: pd.DataFrame,
    direction: str,
    train_symbol_df: pd.DataFrame | None = None,
    val_engine: object | None = None,
    train_engine: object | None = None,
    cv_folds: list | None = None,
    symbol: str | None = None,
) -> dict:
    """Score a pool rule on a single symbol.

    If *train_symbol_df* is provided the score is ``min(train_return,
    val_return)``, preventing rules that are lucky only on validation from
    bubbling to the top.  Rules where ``train_return - val_return`` exceeds
    ``PHASE3_MAX_TRAIN_VAL_GAP_PCT`` are hard-rejected (return -999).
    """
    fmt = _rule_set_to_engine_format([rule])

    # --- val metrics ---
    min_sym_val_rows: int | None = None
    if cv_folds:
        val_return, val_metrics, min_sym_val_rows = _cv_worst_val_for_rule(
            rule, direction, cv_folds, symbol=symbol,
        )
        val_trades = int(val_metrics.get("executed_trades", 0))
    elif val_engine is not None:
        engine = val_engine
        try:
            val_metrics = engine.simulate_rule_set(fmt)
        except Exception:
            return {"return_pct": -999.0, "trades": 0}
        val_return = float(val_metrics.get("total_return_pct", -999.0))
        val_trades = int(val_metrics.get("executed_trades", 0))
        min_sym_val_rows = len(symbol_df)
    else:
        engine = CPUBacktestEngine(
            symbol_df, {}, direction, fee_pct=_cfg.FEE_PCT,
        )
        try:
            val_metrics = engine.simulate_rule_set(fmt)
        except Exception:
            return {"return_pct": -999.0, "trades": 0}
        val_return = float(val_metrics.get("total_return_pct", -999.0))
        val_trades = int(val_metrics.get("executed_trades", 0))
        min_sym_val_rows = len(symbol_df)

    scaled_min_val_trades = int(
        _cfg.effective_phase3_min_val_trades(min_sym_val_rows)
    )

    # --- train metrics (when available) ---
    if train_symbol_df is not None and len(train_symbol_df) > 0:
        if train_engine is not None:
            t_engine = train_engine
        else:
            t_engine = CPUBacktestEngine(
                train_symbol_df, {}, direction, fee_pct=_cfg.FEE_PCT,
            )
        try:
            train_metrics = t_engine.simulate_rule_set(fmt)
        except Exception:
            train_metrics = {}
        train_return = float(train_metrics.get("total_return_pct", -999.0))

        # Positive-good gate (Task 3 — additional floor on top of gap check).
        if bool(getattr(_cfg, "PHASE3_REQUIRE_POSITIVE_GOOD", True)):
            if not gate_positive_good(
                train_metrics,
                val_metrics,
                min_train_return=float(
                    getattr(_cfg, "PHASE3_MIN_TRAIN_RETURN", 0.0)),
                min_val_return=float(
                    getattr(_cfg, "PHASE3_MIN_VAL_RETURN", 0.0)),
                min_train_pf=float(
                    getattr(_cfg, "PHASE3_MIN_TRAIN_PF", 1.0)),
                min_val_pf=float(
                    getattr(_cfg, "PHASE3_MIN_VAL_PF", 1.0)),
                min_train_trades=int(
                    getattr(_cfg, "PHASE3_MIN_TRAIN_TRADES", 25)),
                min_val_trades=scaled_min_val_trades,
                require_execution_health=bool(
                    getattr(_cfg, "PHASE3_GATE_EXECUTION_HEALTH", True)),
            ):
                return {"return_pct": -999.0, "trades": val_trades}

        max_gap = float(getattr(_cfg, "PHASE3_MAX_TRAIN_VAL_GAP_PCT", 40.0))
        if train_return - val_return > max_gap:
            return {"return_pct": -999.0, "trades": val_trades}

        robust_return = min(train_return, val_return)
        return {"return_pct": robust_return, "trades": val_trades}

    return {"return_pct": val_return, "trades": val_trades}


def _phase3_scaled_monthly_penalty(raw_penalty: float) -> float:
    """Convert raw monthly_penalty points into a return-% drag for greedy scoring."""
    weight = float(getattr(_cfg, "PHASE3_MONTHLY_PENALTY_WEIGHT", 1.0))
    scale = float(getattr(_cfg, "PHASE3_MONTHLY_PENALTY_SCALE", 1.0))
    if scale <= 0.0:
        raise ValueError(
            f"PHASE3_MONTHLY_PENALTY_SCALE must be > 0, got {scale!r}"
        )
    return float(raw_penalty) * weight / scale


def _per_symbol_greedy(
    symbol: str,
    symbol_df: pd.DataFrame,
    pool: list[dict],
    direction: str,
    train_symbol_df: pd.DataFrame | None = None,
    combined_df: pd.DataFrame | None = None,
    feature_names: list[str] | None = None,
    monthly_windows: list[pd.DataFrame] | None = None,
    cv_folds: list | None = None,
) -> list[int]:
    """Greedy rule selection for a single symbol.

    All three greedy rounds use ``min(train_return, val_return)`` as the
    combo score when *train_symbol_df* is provided, preventing val-only
    overfit at the combination level (not just the per-rule level).

    When *combined_df* and *monthly_windows* are provided and
    ``MONTHLY_VALIDATION_ENABLED`` is True, a monthly-window penalty is
    subtracted from the combo score so that rule-sets with poor monthly
    performance on **this symbol's** train+val history score lower.
    Effective drag is ``monthly_penalty * PHASE3_MONTHLY_PENALTY_WEIGHT
    / PHASE3_MONTHLY_PENALTY_SCALE`` (subtracted from return %).
    Callers must pass per-symbol slices (not the full universe).
    Pre-built *monthly_windows* avoids redundant window construction inside
    the greedy loop.
    """
    min_trades = int(_cfg.effective_phase3_per_symbol_min_trades())
    min_return = float(_cfg.effective_phase3_per_symbol_min_return())
    max_rules = int(_cfg.PHASE3_PER_SYMBOL_MAX_RULES)
    top_k = int(_cfg.PHASE3_PER_SYMBOL_GREEDY_TOP_K)

    val_engine = CPUBacktestEngine(
        symbol_df, {}, direction, fee_pct=_cfg.FEE_PCT,
    )
    train_engine = None
    if train_symbol_df is not None and len(train_symbol_df) > 0:
        train_engine = CPUBacktestEngine(
            train_symbol_df, {}, direction, fee_pct=_cfg.FEE_PCT,
        )

    monthly_enabled = bool(getattr(_cfg, "MONTHLY_VALIDATION_ENABLED", False))

    scored: list[tuple[int, float, int]] = []
    for idx, rule in enumerate(pool):
        result = _score_pool_rule_on_symbol(
            rule, symbol_df, direction,
            train_symbol_df=train_symbol_df,
            val_engine=val_engine,
            train_engine=train_engine,
            cv_folds=cv_folds,
            symbol=symbol,
        )
        if result["trades"] >= min_trades and result["return_pct"] >= min_return:
            scored.append((idx, result["return_pct"], result["trades"]))

    scored.sort(key=lambda x: x[1], reverse=True)

    if not scored:
        return []

    selected: list[int] = []

    # Round 1: pick best rule
    best_idx, best_ret, _ = scored[0]
    selected.append(best_idx)

    if max_rules <= 1:
        return selected

    def _robust_combo_return(combo_indices: list[int]) -> float:
        """Evaluate a combination with min(train, val) return and monthly penalty."""
        combo_fmt = _rule_set_to_engine_format([pool[i] for i in combo_indices])
        if cv_folds:
            from gpu_fuzzy_trader.validation.rolling_cv import cv_folds_only

            v_rets: list[float] = []
            val_m: dict = {}
            for fold in cv_folds_only(cv_folds):
                if "symbol" in fold.valid_df.columns:
                    sym_df = fold.valid_df[
                        fold.valid_df["symbol"].astype(str) == str(symbol)
                    ].reset_index(drop=True)
                else:
                    sym_df = fold.valid_df
                if len(sym_df) == 0:
                    v_rets.append(-999.0)
                    continue
                val_engine_local = CPUBacktestEngine(
                    sym_df, {}, direction, fee_pct=_cfg.FEE_PCT,
                )
                try:
                    val_m = val_engine_local.simulate_rule_set(combo_fmt)
                    v_rets.append(float(val_m.get("total_return_pct", -999.0)))
                except Exception:
                    v_rets.append(-999.0)
            v_ret = min(v_rets) if v_rets else -999.0
        else:
            # val leg
            val_engine_local = CPUBacktestEngine(
                symbol_df, {}, direction, fee_pct=_cfg.FEE_PCT,
            )
            try:
                val_m = val_engine_local.simulate_rule_set(combo_fmt)
                v_ret = float(val_m.get("total_return_pct", -999.0))
            except Exception:
                v_ret = -999.0
                val_m = {}
        # train leg (when available)
        if train_symbol_df is not None and len(train_symbol_df) > 0:
            train_engine_local = CPUBacktestEngine(
                train_symbol_df, {}, direction, fee_pct=_cfg.FEE_PCT,
            )
            try:
                train_m = train_engine_local.simulate_rule_set(combo_fmt)
                t_ret = float(train_m.get("total_return_pct", -999.0))
            except Exception:
                t_ret = -999.0

            # Positive-good gate on combo (Task 3).
            if bool(getattr(_cfg, "PHASE3_REQUIRE_POSITIVE_GOOD", True)):
                if not gate_positive_good(
                    train_m,
                    val_m,
                    min_train_return=float(
                        getattr(_cfg, "PHASE3_MIN_TRAIN_RETURN", 0.0)),
                    min_val_return=float(
                        getattr(_cfg, "PHASE3_MIN_VAL_RETURN", 0.0)),
                    min_train_pf=float(
                        getattr(_cfg, "PHASE3_MIN_TRAIN_PF", 1.0)),
                    min_val_pf=float(
                        getattr(_cfg, "PHASE3_MIN_VAL_PF", 1.0)),
                    min_train_trades=int(
                        getattr(_cfg, "PHASE3_MIN_TRAIN_TRADES", 25)),
                    min_val_trades=int(
                        getattr(_cfg, "PHASE3_MIN_VAL_TRADES", 15)),
                    require_execution_health=bool(
                        getattr(_cfg, "PHASE3_GATE_EXECUTION_HEALTH", True)),
                ):
                    return -999.0

            base_ret = min(t_ret, v_ret)
        else:
            base_ret = v_ret

        # Evaluator-failure-mode penalty (Task 4).
        eval_health_weight = float(
            getattr(_cfg, "PHASE3_EVAL_HEALTH_WEIGHT", 1.0))
        if eval_health_weight > 0.0 and train_symbol_df is not None and len(train_symbol_df) > 0:
            from gpu_fuzzy_trader.scoring.evaluator_health import (
                evaluator_health_penalty,
            )

            try:
                train_penalty = evaluator_health_penalty(
                    train_m, role="train")
                val_penalty = evaluator_health_penalty(val_m, role="valid")
                base_ret -= (train_penalty + val_penalty) * eval_health_weight
            except Exception as exc:
                logger.debug(
                    "evaluator_health_penalty failed for combo: %s", exc)

        # Monthly-window penalty (per-symbol train+val slice, same scope as base_ret).
        if monthly_enabled and monthly_windows is not None:
            try:
                monthly_summary, _ = evaluate_rule_set_monthly(
                    combined_df, combo_fmt, direction,
                    feature_names=feature_names,
                    windows=monthly_windows,
                )
                if monthly_summary.windows <= 0:
                    monthly_pen = _phase3_scaled_monthly_penalty(float(getattr(
                        _cfg, "PHASE3_MONTHLY_FALLBACK_PENALTY", 5.0)))
                else:
                    monthly_pen = _phase3_scaled_monthly_penalty(
                        monthly_penalty(monthly_summary))
            except Exception as exc:
                logger.debug("monthly eval failed for combo, using fallback: %s", exc)
                monthly_pen = _phase3_scaled_monthly_penalty(float(getattr(
                    _cfg, "PHASE3_MONTHLY_FALLBACK_PENALTY", 5.0)))
            return base_ret - monthly_pen
        return base_ret

    # Round 2: test top-K extensions using min(train, val) for combo return
    best_combo_ret = best_ret
    candidates = [s for s in scored if s[0] not in selected][:top_k]
    if candidates:
        best_combo = selected[:]
        for cand_idx, cand_ret, _ in candidates:
            combo = selected + [cand_idx]
            combo_ret = _robust_combo_return(combo)
            if combo_ret > best_combo_ret:
                best_combo = combo[:]
                best_combo_ret = combo_ret
        selected = best_combo[:]

    if max_rules <= 2 or len(selected) < 2:
        return selected

    # Round 3: test top-K extensions on best_2 using min(train, val)
    candidates = [s for s in scored if s[0] not in selected][:top_k]
    if candidates:
        best_combo = selected[:]
        best_combo_ret = best_combo_ret
        for cand_idx, cand_ret, _ in candidates:
            combo = selected + [cand_idx]
            combo_ret = _robust_combo_return(combo)
            if combo_ret > best_combo_ret:
                best_combo = combo[:]
                best_combo_ret = combo_ret
        selected = best_combo[:]

    return selected


def _merge_per_symbol_rules(
    symbol_assignments: dict[str, list[int]],
    pool: list[dict],
) -> list[dict]:
    reverse: dict[int, list[str]] = {}
    for sym, indices in symbol_assignments.items():
        for idx in indices:
            if idx not in reverse:
                reverse[idx] = []
            reverse[idx].append(sym)

    merged: list[dict] = []
    for pool_idx, symbols in reverse.items():
        rule = dict(pool[pool_idx])
        sym_conditions = [f"symbol is {sym}" for sym in sorted(symbols)]
        rule["conditions"] = list(rule.get("conditions", [])) + sym_conditions
        rule["_pool_idx"] = int(pool_idx)
        merged.append(rule)

    return merged


def _score_merged_rule_on_splits(
    rule: dict,
    train_engine: object,
    val_engine: object,
) -> float:
    """Robust min(train, val) return for ordering rules_set under v5 allocation."""
    fmt = _rule_set_to_engine_format([rule])
    try:
        val_metrics = val_engine.simulate_rule_set(fmt)
        val_return = float(val_metrics.get("total_return_pct", -999.0))
    except Exception:
        return -999.0

    try:
        train_metrics = train_engine.simulate_rule_set(fmt)
        train_return = float(train_metrics.get("total_return_pct", -999.0))
    except Exception:
        train_return = -999.0

    # Positive-good gate on merged rule (Task 3).
    if bool(getattr(_cfg, "PHASE3_REQUIRE_POSITIVE_GOOD", True)):
        if not gate_positive_good(
            train_metrics,
            val_metrics,
            min_train_return=float(
                getattr(_cfg, "PHASE3_MIN_TRAIN_RETURN", 0.0)),
            min_val_return=float(
                getattr(_cfg, "PHASE3_MIN_VAL_RETURN", 0.0)),
            min_train_pf=float(
                getattr(_cfg, "PHASE3_MIN_TRAIN_PF", 1.0)),
            min_val_pf=float(
                getattr(_cfg, "PHASE3_MIN_VAL_PF", 1.0)),
            min_train_trades=int(
                getattr(_cfg, "PHASE3_MIN_TRAIN_TRADES", 25)),
            min_val_trades=int(
                getattr(_cfg, "PHASE3_MIN_VAL_TRADES", 15)),
            require_execution_health=bool(
                getattr(_cfg, "PHASE3_GATE_EXECUTION_HEALTH", True)),
        ):
            return -999.0

    max_gap = float(getattr(_cfg, "PHASE3_MAX_TRAIN_VAL_GAP_PCT", 40.0))
    if train_return - val_return > max_gap:
        return -999.0
    return min(train_return, val_return)


def _sort_merged_rules_by_score(
    merged_rules: list[dict],
    train_engine: object,
    val_engine: object,
) -> list[dict]:
    """
    Order rules_set so stronger rules appear first in JSON (v5 capital priority).
    """
    scored: list[tuple[float, int, dict]] = []
    for rule in merged_rules:
        pool_idx = int(rule.pop("_pool_idx", 0))
        score = _score_merged_rule_on_splits(rule, train_engine, val_engine)
        scored.append((score, pool_idx, rule))

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [rule for _, _, rule in scored]


# ---------------------------------------------------------------------------
# Output serialisation
# ---------------------------------------------------------------------------


def _cap_capital_per_rule(rule_set: list[dict]) -> list[dict]:
    cap = float(_cfg.PHASE3_MAX_CAPITAL_PCT_PER_RULE)
    out: list[dict] = []
    for rule in rule_set:
        r = dict(rule)
        r["capital_pct"] = min(
            float(r.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)), cap)
        out.append(r)
    return out


def _build_output_dict(rule_set: list[dict], direction: str) -> dict:
    rule_set = _cap_capital_per_rule(rule_set)
    rules_list = []
    for rule in rule_set:
        rules_list.append({
            "tp": float(rule.get("tp", _cfg.PHASE2_TP)),
            "sl": float(rule.get("sl", _cfg.PHASE2_SL)),
            "capital_pct": float(rule.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
            "conditions": list(rule["conditions"]),
        })
    return {
        "direction": direction,
        "risk_optimized": False,
        "selection_accepted": True,
        "rules_set": rules_list,
    }


def _build_rejected_output_dict(direction: str, reason: str) -> dict:
    return {
        "direction": direction,
        "risk_optimized": False,
        "selection_accepted": False,
        "selection_rejection_reason": reason,
        "rules_set": [],
    }


# ---------------------------------------------------------------------------
# Rule_Set_Selector
# ---------------------------------------------------------------------------


class Rule_Set_Selector:
    """
    Phase 3: per-symbol greedy rule selection from the Phase 2 pool.

    For each symbol, selects 0–3 rules that perform best on that symbol's
    validation data.  Rules selected for multiple symbols are merged into
    a single output rule with "symbol is X" conditions.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame (already prepared).
    val_df : pd.DataFrame
        Validation split DataFrame (already prepared).
    pool : list[dict]
        Phase 2 pool entries.  Each entry must have at minimum:
        "conditions" (list[str]).  "tp", "sl", "capital_pct" are optional
        (defaults to Phase 2 static values if absent).
    direction : str
        "long" or "short".
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        pool: list[dict],
        direction: str,
        seed: int = 42,
        cv_folds: list | None = None,
    ) -> None:
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")
        if not pool:
            raise ValueError("pool must not be empty.")

        self.direction = direction
        self.pool = pool
        self._cv_folds = cv_folds
        self._full_train_df = train_df
        self._full_val_df = val_df
        self.seed = seed

        self._val_engine, self._train_engine, self._eval_cache = (
            _build_phase3_engines(train_df, val_df, direction, pool)
        )

        (
            self._report_val_engine,
            self._report_train_engine,
            _,
        ) = _build_phase3_engines(
            self._full_train_df, self._full_val_df, direction, pool
        )

        # Discover symbol universe
        if "symbol" in val_df.columns:
            self._symbols = sorted(
                val_df["symbol"].dropna().astype(str).unique().tolist()
            )
        else:
            self._symbols = ["UNKNOWN"]

        # Pre-filter val_df per symbol
        self._symbol_dfs: dict[str, pd.DataFrame] = {}
        for sym in self._symbols:
            sym_mask = val_df["symbol"].astype(str) == sym
            self._symbol_dfs[sym] = val_df[sym_mask].reset_index(drop=True)

        # Pre-filter train_df per symbol (used for robust min(train,val) scoring)
        self._train_symbol_dfs: dict[str, pd.DataFrame] = {}
        if "symbol" in train_df.columns:
            for sym in self._symbols:
                sym_mask = train_df["symbol"].astype(str) == sym
                self._train_symbol_dfs[sym] = train_df[sym_mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> dict:
        logger.info(
            "Phase 3 [%s]: pool=%d, symbols=%d, per-symbol greedy",
            self.direction,
            len(self.pool),
            len(self._symbols),
        )

        enriched_pool = [
            {
                **entry,
                "tp": float(entry.get("tp", _cfg.PHASE2_TP)),
                "sl": float(entry.get("sl", _cfg.PHASE2_SL)),
                "capital_pct": float(entry.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
            }
            for entry in self.pool
        ]

        max_gap = float(getattr(_cfg, "PHASE3_MAX_TRAIN_VAL_GAP_PCT", 40.0))
        logger.info(
            "Phase 3 [%s]: robust scoring enabled — "
            "objective=min(train,val), gap_reject_threshold=%.1f%%",
            self.direction, max_gap,
        )

        monthly_enabled = bool(
            getattr(_cfg, "MONTHLY_VALIDATION_ENABLED", False))
        feature_names: list[str] | None = None
        if monthly_enabled:
            sample_df = pd.concat(
                [self._full_train_df, self._full_val_df], ignore_index=True)
            feature_names = _monthly_feature_names(sample_df)

        symbol_assignments: dict[str, list[int]] = {}
        # Per-symbol diagnostic data (Task 12).
        diag_per_symbol: dict[str, dict] = {}
        for sym in self._symbols:
            sym_df = self._symbol_dfs.get(sym)
            if sym_df is None or len(sym_df) == 0:
                logger.debug("Phase 3 [%s]: symbol %s has no val data, skipping",
                             self.direction, sym)
                continue

            train_sym_df = self._train_symbol_dfs.get(sym)

            sym_combined_df: pd.DataFrame | None = None
            sym_monthly_windows: list[pd.DataFrame] | None = None
            if monthly_enabled:
                sym_combined_df, sym_monthly_windows = (
                    _build_per_symbol_monthly_context(train_sym_df, sym_df)
                )

            selected = _per_symbol_greedy(
                symbol=sym,
                symbol_df=sym_df,
                pool=enriched_pool,
                direction=self.direction,
                train_symbol_df=train_sym_df,
                combined_df=sym_combined_df,
                feature_names=feature_names,
                monthly_windows=sym_monthly_windows,
                cv_folds=self._cv_folds,
            )
            if selected:
                symbol_assignments[sym] = selected
                logger.info(
                    "Phase 3 [%s]: symbol %s → %d rules selected",
                    self.direction, sym, len(selected),
                )

                # --- Accumulate per-symbol diagnostic (Task 12) ---
                if _cfg.PHASE3_DIAGNOSTIC_REPORT_ENABLED:
                    top_idx = selected[0]
                    top_rule = enriched_pool[top_idx]
                    # Re-score top rule on this symbol to get val_trades / val_return_pct.
                    result = _score_pool_rule_on_symbol(
                        top_rule, sym_df, self.direction,
                        train_symbol_df=train_sym_df,
                    )
                    val_trades = result["trades"]
                    val_return_pct = result["return_pct"]

                    # Compute train return for train_val_gap_pct.
                    train_ret = -999.0
                    if train_sym_df is not None and len(train_sym_df) > 0 and val_return_pct > -999:
                        fmt = _rule_set_to_engine_format([top_rule])
                        t_engine = CPUBacktestEngine(
                            train_sym_df, {}, self.direction, fee_pct=_cfg.FEE_PCT,
                        )
                        train_metrics_ = t_engine.simulate_rule_set(fmt)
                        train_ret = float(train_metrics_.get("total_return_pct", -999.0))

                    train_val_gap_pct = (
                        (train_ret - val_return_pct)
                        if train_ret > -999 and val_return_pct > -999
                        else 0.0
                    )
                    top_rule_condition_signature = ";".join(
                        str(c) for c in top_rule.get("conditions", [])
                    )
                    diag_per_symbol[sym] = {
                        "direction": self.direction,
                        "symbol": sym,
                        "val_trades": int(val_trades),
                        "val_return_pct": round(float(val_return_pct), 4),
                        "train_val_gap_pct": round(train_val_gap_pct, 4),
                        "n_rules_selected": len(selected),
                        "top_rule_condition_signature": top_rule_condition_signature,
                    }
            else:
                logger.info(
                    "Phase 3 [%s]: symbol %s → no rules selected",
                    self.direction, sym,
                )

        global_min = int(_cfg.PHASE3_GLOBAL_MIN_RULES)
        global_max = int(_cfg.PHASE3_GLOBAL_MAX_RULES)

        if not symbol_assignments:
            logger.warning(
                "Phase 3 [%s]: no rules selected for any symbol; trying fallback",
                self.direction,
            )
            merged_rules = _try_global_pool_fallback(
                enriched_pool,
                self._train_engine,
                self._val_engine,
                self._eval_cache,
                self.direction,
                global_min,
            )
            if merged_rules is None:
                # Lean fallback (Task 10.2) — most lenient path, skips gates.
                merged_rules = _try_lean_fallback(
                    enriched_pool, self.direction, global_min,
                )
            if merged_rules is None:
                output_dict = _build_rejected_output_dict(
                    self.direction, "no_rules_selected_for_any_symbol")
                self._persist_output(output_dict)
                return output_dict
        else:
            # Build merged rules with multi-symbol combinations when enabled
            # (Task 6).  The new path calls ``_build_symbol_specialized_variants``
            # per pool rule to find the best 1-, 2-, or 3-symbol combination
            # instead of blindly appending all symbols that selected the rule.
            max_sym_per_rule = max(
                1, int(getattr(
                    _cfg, "SYMBOL_SPECIALIZATION_MAX_SYMBOLS_PER_RULE", 3)))
            use_symbol_specialization = bool(
                getattr(_cfg, "SYMBOL_SPECIALIZATION_USE_COMBINATIONS", True))
            if max_sym_per_rule <= 1:
                merged_rules = _build_per_symbol_assigned_rules(
                    symbol_assignments, enriched_pool,
                )
            elif use_symbol_specialization:
                merged_rules = _build_multi_symbol_merged_rules(
                    symbol_assignments, enriched_pool,
                    self._train_engine, self._val_engine,
                    self._symbols,
                )
            else:
                merged_rules = _merge_per_symbol_rules(
                    symbol_assignments, enriched_pool)

            merged_rules = _sort_merged_rules_by_score(
                merged_rules,
                self._train_engine,
                self._val_engine,
            )

            if len(merged_rules) < global_min:
                logger.warning(
                    "Phase 3 [%s]: only %d merged rules, need at least %d. "
                    "Trying fallback...",
                    self.direction, len(merged_rules), global_min,
                )
                fallback_rules = _try_global_pool_fallback(
                    enriched_pool,
                    self._train_engine,
                    self._val_engine,
                    self._eval_cache,
                    self.direction,
                    global_min,
                )
                if fallback_rules is None:
                    # Lean fallback (Task 10.2) — most lenient path, skips gates.
                    fallback_rules = _try_lean_fallback(
                        enriched_pool, self.direction, global_min,
                    )
                if fallback_rules is None:
                    logger.warning(
                        "Phase 3 [%s]: fallback also failed — rejecting",
                        self.direction,
                    )
                    output_dict = _build_rejected_output_dict(
                        self.direction, "insufficient_rules_after_merge")
                    self._persist_output(output_dict)
                    return output_dict
                merged_rules = fallback_rules

        if len(merged_rules) > global_max:
            logger.warning(
                "Phase 3 [%s]: truncating %d rules to %d",
                self.direction, len(merged_rules), global_max,
            )
            merged_rules = merged_rules[:global_max]

        output_dict = _build_output_dict(merged_rules, self.direction)
        self._persist_output(output_dict)

        # Reporter: full train/val splits
        try:
            engine_fmt = _rule_set_to_engine_format(merged_rules)
            train_metrics, train_log = self._report_train_engine.simulate_rule_set(
                engine_fmt, return_logs=True
            )
            Reporter().plot_equity_curve(train_log, "train", self.direction)
            Reporter().write_per_symbol_csv(
                train_metrics, "train", direction=self.direction)
        except Exception as exc:
            logger.warning("Reporter train equity/csv failed (non-fatal): %s", exc)

        try:
            engine_fmt = _rule_set_to_engine_format(merged_rules)
            val_metrics, val_log = self._report_val_engine.simulate_rule_set(
                engine_fmt, return_logs=True
            )
            Reporter().plot_equity_curve(val_log, "validation", self.direction)
            Reporter().write_per_symbol_csv(
                val_metrics, "validation", direction=self.direction)
        except Exception as exc:
            logger.warning("Reporter validation equity/csv failed (non-fatal): %s", exc)

        # --- Write per-symbol diagnostic CSV (Task 12) ---
        if _cfg.PHASE3_DIAGNOSTIC_REPORT_ENABLED and diag_per_symbol:
            import csv as _csv

            os.makedirs(_cfg.REPORTS_DIR, exist_ok=True)
            diag_path = os.path.join(
                _cfg.REPORTS_DIR, "gen_diag_iter12.csv")
            fieldnames = [
                "direction", "symbol", "val_trades", "val_return_pct",
                "train_val_gap_pct", "n_rules_selected",
                "top_rule_condition_signature",
            ]
            file_exists = os.path.isfile(diag_path)
            with open(diag_path, "a" if file_exists else "w",
                      newline="", encoding="utf-8") as fh:
                writer = _csv.DictWriter(fh, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                for sym in sorted(diag_per_symbol.keys()):
                    writer.writerow(diag_per_symbol[sym])
            logger.info(
                "Phase 3 [%s]: diagnostic CSV written to %s (%d rows)",
                self.direction, diag_path, len(diag_per_symbol),
            )

        return output_dict

    def _persist_output(self, output_dict: dict) -> None:
        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        output_path = _OUTPUT_PATHS[self.direction]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2)
        _maybe_write_evaluator_clean(output_dict, output_path, self.direction)
        n_rules = len(output_dict.get("rules_set", []))
        logger.info(
            "Phase 3 [%s]: %d rules saved to %s",
            self.direction, n_rules, output_path,
        )

    @staticmethod
    def load_rule_set(direction: str) -> Optional[dict]:
        if direction not in _OUTPUT_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _OUTPUT_PATHS[direction]
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Rule set file is unreadable or corrupted: {path}"
            ) from exc

        _validate_rule_set_schema(data, path)
        return data

    @staticmethod
    def skip_if_valid() -> Optional[dict[str, dict]]:
        result: dict[str, dict] = {}

        for direction in ("long", "short"):
            try:
                loaded = Rule_Set_Selector.load_rule_set(direction)
                if loaded is not None and loaded.get("selection_accepted") is not False:
                    result[direction] = loaded
            except ValueError:
                pass

        if not result:
            return None
        return result
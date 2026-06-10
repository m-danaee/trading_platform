"""
phase3_rule_set.py — Rule_Set_Selector (Phase 3)

Per-symbol greedy rule selection from the Phase 2 pool.

For each symbol in the universe, selects 0–3 rules from the pool that perform
best on that symbol's validation data.  Rules selected for multiple symbols are
merged into a single output rule with "symbol is X" conditions.

Output:
    outputs/long.json  and  outputs/short.json
    (evaluator_v4.ipynb compatible format with "symbol is X" conditions)

Skip logic:
    If both files exist and pass schema validation → skip Phase 3.
    If only one exists → skip and proceed with available file.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
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

logger = logging.getLogger(__name__)

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
    while len(out) < n_rules and len(out) < len(pool):
        for rule in pool:
            key = _conditions_key(rule.get("conditions", []))
            if key not in seen:
                out.append(rule)
                seen.add(key)
                break
        else:
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


# ---------------------------------------------------------------------------
# Per-symbol greedy selection
# ---------------------------------------------------------------------------


def _score_pool_rule_on_symbol(
    rule: dict,
    symbol_df: pd.DataFrame,
    direction: str,
) -> dict:
    engine = CPUBacktestEngine(
        symbol_df, {}, direction, fee_pct=_cfg.FEE_PCT,
    )
    fmt = _rule_set_to_engine_format([rule])
    try:
        metrics = engine.simulate_rule_set(fmt)
    except Exception:
        return {"return_pct": -999.0, "trades": 0}
    return {
        "return_pct": float(metrics.get("total_return_pct", -999.0)),
        "trades": int(metrics.get("executed_trades", 0)),
    }


def _per_symbol_greedy(
    symbol: str,
    symbol_df: pd.DataFrame,
    pool: list[dict],
    direction: str,
) -> list[int]:
    min_trades = int(_cfg.PHASE3_PER_SYMBOL_MIN_TRADES)
    min_return = float(_cfg.PHASE3_PER_SYMBOL_MIN_RETURN)
    max_rules = int(_cfg.PHASE3_PER_SYMBOL_MAX_RULES)
    top_k = int(_cfg.PHASE3_PER_SYMBOL_GREEDY_TOP_K)

    scored: list[tuple[int, float, int]] = []
    for idx, rule in enumerate(pool):
        result = _score_pool_rule_on_symbol(rule, symbol_df, direction)
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

    # Round 2: test top-K extensions
    candidates = [s for s in scored if s[0] not in selected][:top_k]
    if candidates:
        best_combo = selected[:]
        best_combo_ret = best_ret
        for cand_idx, cand_ret, _ in candidates:
            combo = selected + [cand_idx]
            combo_fmt = _rule_set_to_engine_format([pool[i] for i in combo])
            engine = CPUBacktestEngine(
                symbol_df, {}, direction, fee_pct=_cfg.FEE_PCT,
            )
            try:
                metrics = engine.simulate_rule_set(combo_fmt)
                combo_ret = float(metrics.get("total_return_pct", -999.0))
            except Exception:
                combo_ret = -999.0
            if combo_ret > best_combo_ret:
                best_combo = combo[:]
                best_combo_ret = combo_ret
        selected = best_combo[:]

    if max_rules <= 2 or len(selected) < 2:
        return selected

    # Round 3: test top-K extensions on best_2
    candidates = [s for s in scored if s[0] not in selected][:top_k]
    if candidates:
        best_combo = selected[:]
        best_combo_ret = best_combo_ret
        for cand_idx, cand_ret, _ in candidates:
            combo = selected + [cand_idx]
            combo_fmt = _rule_set_to_engine_format([pool[i] for i in combo])
            engine = CPUBacktestEngine(
                symbol_df, {}, direction, fee_pct=_cfg.FEE_PCT,
            )
            try:
                metrics = engine.simulate_rule_set(combo_fmt)
                combo_ret = float(metrics.get("total_return_pct", -999.0))
            except Exception:
                combo_ret = -999.0
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
        merged.append(rule)

    merged.sort(key=lambda r: -len([c for c in r.get("conditions", [])
                                    if c.lower().startswith("symbol is")]))
    return merged


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

        symbol_assignments: dict[str, list[int]] = {}
        for sym in self._symbols:
            sym_df = self._symbol_dfs.get(sym)
            if sym_df is None or len(sym_df) == 0:
                logger.debug("Phase 3 [%s]: symbol %s has no val data, skipping",
                             self.direction, sym)
                continue

            selected = _per_symbol_greedy(
                symbol=sym,
                symbol_df=sym_df,
                pool=enriched_pool,
                direction=self.direction,
            )
            if selected:
                symbol_assignments[sym] = selected
                logger.info(
                    "Phase 3 [%s]: symbol %s → %d rules selected",
                    self.direction, sym, len(selected),
                )
            else:
                logger.info(
                    "Phase 3 [%s]: symbol %s → no rules selected",
                    self.direction, sym,
                )

        if not symbol_assignments:
            logger.warning(
                "Phase 3 [%s]: no rules selected for any symbol",
                self.direction,
            )
            output_dict = _build_rejected_output_dict(
                self.direction, "no_rules_selected_for_any_symbol")
            self._persist_output(output_dict)
            return output_dict

        merged_rules = _merge_per_symbol_rules(symbol_assignments, enriched_pool)

        global_min = int(_cfg.PHASE3_GLOBAL_MIN_RULES)
        global_max = int(_cfg.PHASE3_GLOBAL_MAX_RULES)

        if len(merged_rules) < global_min:
            logger.warning(
                "Phase 3 [%s]: only %d merged rules, need at least %d. "
                "Trying fallback...",
                self.direction, len(merged_rules), global_min,
            )
            fallback = _best_rules_from_pool_fallback(
                enriched_pool, global_min)
            engine_fmt = _rule_set_to_engine_format(fallback)
            train_m, val_m = _simulate_team(
                engine_fmt, self._train_engine, self._val_engine, self._eval_cache,
            )
            if float(val_m.get("total_return_pct", -999.0)) > float(
                _cfg.PHASE3_VAL_RETURN_FLOOR_PCT
            ):
                merged_rules = fallback
                logger.info(
                    "Phase 3 [%s]: fallback produced %d global rules",
                    self.direction, len(merged_rules),
                )
            else:
                logger.warning(
                    "Phase 3 [%s]: fallback also failed — rejecting",
                    self.direction,
                )
                output_dict = _build_rejected_output_dict(
                    self.direction, "insufficient_rules_after_merge")
                self._persist_output(output_dict)
                return output_dict

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

        return output_dict

    def _persist_output(self, output_dict: dict) -> None:
        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        output_path = _OUTPUT_PATHS[self.direction]
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(output_dict, fh, indent=2)
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
"""
cpu_engine.py — CPUBacktestEngine

Exact Python/NumPy replication of evaluator_v3.ipynb's
CapitalManagedTradeSimulator semantics.

Rule matching uses apply_dynamic_rule threshold logic (mode-independent),
exactly as the evaluator does for exported strategies.
"""

from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.condition_cache import get_or_build_rule_mask
from gpu_fuzzy_trader.backtest.symbol_conditions import (
    get_normalized_symbol_array,
    split_feature_and_symbol_conditions,
)

logger = logging.getLogger(__name__)


def _batch_eval_rule_set_pickled(
    payload: tuple["CPUBacktestEngine", list[dict]],
) -> dict:
    """Top-level worker for ProcessPoolExecutor (must be picklable)."""
    engine, rule_set = payload
    return engine.simulate_rule_set(rule_set)


# ---------------------------------------------------------------------------
# Helpers — mirrors evaluator_v3.ipynb utilities
# ---------------------------------------------------------------------------

def _normalize_direction(direction: str) -> str:
    direction = "long" if direction is None else str(direction).strip().lower()
    if direction not in {"long", "short"}:
        raise ValueError("direction must be 'long' or 'short'.")
    return direction


def precompute_release_indices(
    symbols: np.ndarray,
    symbol_bar_index: np.ndarray,
    n_rows: int,
    max_hold_candles: int,
) -> np.ndarray:
    """
    For each row, find the row index where symbol_bar_index + max_hold_candles
    is first reached within the same symbol.
    """
    release_index = np.full(n_rows, n_rows, dtype=np.int64)
    row_index = np.arange(n_rows, dtype=np.int64)
    symbol_codes, _ = pd.factorize(symbols, sort=False)

    for sym_code in np.unique(symbol_codes):
        sym_mask = symbol_codes == sym_code
        rows = row_index[sym_mask]
        bars = symbol_bar_index[sym_mask].astype(np.int64, copy=False)

        order = np.argsort(bars, kind="mergesort")
        rows_sorted = rows[order]
        bars_sorted = bars[order]

        target_bars = bars_sorted + max_hold_candles
        target_positions = np.searchsorted(
            bars_sorted, target_bars, side="left")

        valid = target_positions < len(rows_sorted)
        if np.any(valid):
            release_index[rows_sorted[valid]
                          ] = rows_sorted[target_positions[valid]]

    return release_index


def _safe_profit_factor(gross_wins: float, gross_losses: float) -> float:
    if gross_losses <= 0 and gross_wins > 0:
        return 99.0
    if gross_losses <= 0:
        return 0.0
    return gross_wins / gross_losses


def _sortino_ratio_from_returns(
    trade_returns: list[float] | np.ndarray,
    target_return: float = 0.0,
) -> float:
    """Compute a non-annualized Sortino Ratio from per-trade returns."""
    returns = np.asarray(trade_returns, dtype=np.float64)
    if returns.size == 0:
        return 0.0

    excess_returns = returns - float(target_return)
    mean_excess_return = float(np.mean(excess_returns))
    downside_returns = np.minimum(excess_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))

    from gpu_fuzzy_trader import config as _cfg

    if downside_deviation <= 0.0:
        return _cfg.SORTINO_CAP if mean_excess_return > 0.0 else 0.0

    return min(mean_excess_return / downside_deviation, _cfg.SORTINO_CAP)


def _parse_condition(condition: str) -> tuple[str, str]:
    """Parse '[feature_name] IS Fuzzy Value Name' → (feature_name, value_name)."""
    if not isinstance(condition, str) or " IS " not in condition:
        raise ValueError(
            f"Invalid condition format: {condition!r}. "
            "Expected: '[feature_name] IS Value Name'."
        )
    feature_part, value_part = condition.split(" IS ", 1)
    feature_name = feature_part.strip()
    value_name = value_part.strip()
    if not (feature_name.startswith("[") and feature_name.endswith("]")):
        raise ValueError(
            f"Feature must be inside square brackets in: {condition!r}"
        )
    feature_name = feature_name[1:-1].strip()
    if not feature_name:
        raise ValueError(f"Empty feature name in condition: {condition!r}")
    if not value_name:
        raise ValueError(f"Empty value name in condition: {condition!r}")
    return feature_name, value_name


def _apply_dynamic_rule(df: pd.DataFrame, condition: str) -> np.ndarray:
    """
    Apply one exported text condition using the original threshold logic.

    Exactly mirrors evaluator_v3.ipynb's apply_dynamic_rule — mode-independent.
    Returns a boolean NumPy array of length len(df).
    """
    feature_name, value_name = _parse_condition(condition)

    if feature_name not in df.columns:
        raise ValueError(
            f"Unknown feature in condition: {feature_name!r}. "
            "Make sure the strategy was created using the same feature set."
        )

    s = df[feature_name]

    if value_name == "Active (1)":
        return np.asarray(s == 1, dtype=bool)
    if value_name == "Inactive (0)":
        return np.asarray(s == 0, dtype=bool)
    if value_name in ("Positive", "Positive (1)"):
        return np.asarray(s == 1, dtype=bool)
    if value_name in ("Neutral", "Neutral (0)"):
        return np.asarray(s == 0, dtype=bool)
    if value_name in ("Negative", "Negative (-1)"):
        return np.asarray(s == -1, dtype=bool)

    if value_name in ("Strong Negative", "Strong Negative (e.g. Big Down Gap)"):
        return np.asarray(s <= -0.25, dtype=bool)
    if value_name == "Weak Negative":
        return np.asarray((s > -0.25) & (s <= -1e-5), dtype=bool)
    if value_name in ("Exactly Zero", "Exactly Zero (No Gap)"):
        return np.asarray((s > -1e-5) & (s <= 1e-5), dtype=bool)
    if value_name == "Weak Positive":
        return np.asarray((s > 1e-5) & (s <= 0.25), dtype=bool)
    if value_name in ("Strong Positive", "Strong Positive (e.g. Big Up Gap)"):
        return np.asarray(s > 0.25, dtype=bool)

    if value_name == "Very Low":
        return np.asarray(s <= 0.2, dtype=bool)
    if value_name == "Low":
        return np.asarray((s > 0.2) & (s <= 0.4), dtype=bool)
    if value_name == "Medium":
        return np.asarray((s > 0.4) & (s <= 0.6), dtype=bool)
    if value_name == "High":
        return np.asarray((s > 0.6) & (s <= 0.8), dtype=bool)
    if value_name == "Very High":
        return np.asarray(s > 0.8, dtype=bool)

    if value_name == "Extreme Bearish":
        return np.asarray(s <= -0.8, dtype=bool)
    if value_name == "Strong Bearish":
        return np.asarray((s > -0.8) & (s <= -0.6), dtype=bool)
    if value_name == "Bearish":
        return np.asarray((s > -0.6) & (s <= -0.4), dtype=bool)
    if value_name == "Weak Bearish":
        return np.asarray((s > -0.4) & (s <= -0.2), dtype=bool)
    if value_name == "Neutral Negative":
        return np.asarray((s > -0.2) & (s <= 0.0), dtype=bool)
    if value_name == "Neutral Positive":
        return np.asarray((s > 0.0) & (s <= 0.2), dtype=bool)
    if value_name == "Weak Bullish":
        return np.asarray((s > 0.2) & (s <= 0.4), dtype=bool)
    if value_name == "Bullish":
        return np.asarray((s > 0.4) & (s <= 0.6), dtype=bool)
    if value_name == "Strong Bullish":
        return np.asarray((s > 0.6) & (s <= 0.8), dtype=bool)
    if value_name == "Extreme Bullish":
        return np.asarray(s > 0.8, dtype=bool)

    raise ValueError(
        f"Rule value {value_name!r} is not recognized for feature {feature_name!r}."
    )


def _compute_rule_signal_mask(
    df: pd.DataFrame,
    conditions: list[str],
    rule_number: int = 1,
) -> np.ndarray:
    """
    Build one boolean signal mask (evaluator_v4 parity).

    Feature conditions are AND-ed. Symbol conditions are OR-ed, then AND-ed
    with the feature signal. Rules without symbol filters apply to all symbols.
    """
    feature_conditions, allowed_symbols = split_feature_and_symbol_conditions(
        conditions=conditions,
        rule_number=rule_number,
    )

    signal = np.ones(len(df), dtype=bool)

    for condition in feature_conditions:
        signal &= np.asarray(_apply_dynamic_rule(df, condition), dtype=bool)

    if allowed_symbols:
        normalized_symbols = get_normalized_symbol_array(df)
        signal &= np.isin(normalized_symbols, allowed_symbols)

    return signal.astype(bool, copy=False)


def _build_rule_signal_mask(
    df: pd.DataFrame,
    conditions: list[str],
    mask_cache: dict[tuple[str, ...], np.ndarray] | None = None,
    rule_number: int = 1,
) -> np.ndarray:
    """Cached wrapper around :func:`_compute_rule_signal_mask`."""
    if not conditions:
        raise ValueError("Rule must contain a non-empty 'conditions' list.")
    return get_or_build_rule_mask(df, conditions, mask_cache, rule_number)


def _build_entries_from_rule_set(
    df: pd.DataFrame,
    rule_set: list[dict],
    mask_cache: dict[tuple[str, ...], np.ndarray] | None = None,
) -> list[dict]:
    """
    Priority-based rule assignment: first matching rule wins per row.

    Mirrors evaluator_v3.ipynb's build_rule_set_entries() with signal_mask path.
    Returns a list of entry dicts sorted by row index.
    """
    if not rule_set:
        return []

    n_rows = len(df)
    assigned_mask = np.zeros(n_rows, dtype=bool)
    entries: list[dict] = []

    for rule_idx, rule_entry in enumerate(rule_set, start=1):
        if not isinstance(rule_entry, dict):
            raise ValueError("Each rule entry must be a dictionary.")

        tp = float(rule_entry["tp"])
        sl = float(rule_entry["sl"])
        capital_pct = float(rule_entry["capital_pct"])

        if not np.isfinite(capital_pct) or capital_pct <= 0:
            raise ValueError(
                f"Rule {rule_idx} has invalid capital_pct: {capital_pct}"
            )

        conditions = rule_entry.get("conditions", [])
        rule_signals = _build_rule_signal_mask(
            df, conditions, mask_cache, rule_number=rule_idx
        )

        new_match_mask = rule_signals & (~assigned_mask)
        matched_indices = np.flatnonzero(new_match_mask)
        if len(matched_indices) == 0:
            continue

        assigned_mask[matched_indices] = True

        for idx in matched_indices:
            entries.append(
                {
                    "idx": int(idx),
                    "rule_index": int(rule_idx),
                    "tp": tp,
                    "sl": sl,
                    "capital_pct": capital_pct,
                }
            )

    entries.sort(key=lambda x: x["idx"])
    return entries


# ---------------------------------------------------------------------------
# CPUBacktestEngine
# ---------------------------------------------------------------------------

class CPUBacktestEngine:
    """
    CPU backtest engine that exactly mirrors evaluator_v3.ipynb's
    CapitalManagedTradeSimulator semantics.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset (already sorted, NaN-dropped, bar-indexed).
    feature_modes : dict[str, str]
        Feature mode mapping (accepted for interface compatibility; rule
        matching uses apply_dynamic_rule threshold logic, not mode-based
        discretization).
    direction : str
        "long" or "short".
    **constants
        Optional overrides for backtest constants. Recognised keys:
        initial_capital, leverage, fee_pct, max_hold_candles,
        max_total_exposure_pct, min_position_notional.
        Defaults come from config.py.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_modes: dict[str, str],
        direction: str,
        **constants,
    ) -> None:
        self.df = df
        self.feature_modes = feature_modes  # stored but not used for matching
        self.trade_direction = _normalize_direction(direction)

        # --- Constants (config.py defaults, overridable via **constants) ---
        self.initial_capital = float(
            constants.get("initial_capital", _cfg.INITIAL_CAPITAL)
        )
        self.leverage = float(constants.get("leverage", _cfg.LEVERAGE))
        self.fee_pct = float(constants.get("fee_pct", _cfg.FEE_PCT))
        self.round_trip_fee_rate = self.fee_pct / 100.0
        self.max_hold_candles = int(
            constants.get("max_hold_candles", _cfg.MAX_HOLD_CANDLES)
        )
        self.max_total_exposure_pct = float(
            constants.get("max_total_exposure_pct",
                          _cfg.MAX_TOTAL_EXPOSURE_PCT)
        )
        self.max_total_exposure_rate = self.max_total_exposure_pct / 100.0
        self.min_position_notional = float(
            constants.get("min_position_notional", _cfg.MIN_POSITION_NOTIONAL)
        )
        self._regime_ids = constants.get("regime_ids", None)
        self._n_regimes = int(constants.get("n_regimes", 0))

        # --- Pre-extract arrays for speed ---
        entry = df["label_open_next"].values.astype(float)

        invalid_mask = (~np.isfinite(entry)) | (entry <= 0)
        if np.any(invalid_mask):
            bad = int(np.sum(invalid_mask))
            raise ValueError(
                f"Invalid label_open_next values (must be finite and positive). "
                f"Bad rows: {bad}"
            )

        self.entry_price = entry
        self.max_ret = (df["label_max_288"].values - entry) / entry * 100.0
        self.min_ret = (df["label_min_288"].values - entry) / entry * 100.0
        self.close_ret = (df["label_close_288"].values - entry) / entry * 100.0
        self.max_before_min = df["label_max_before_min"].values

        if "symbol" in df.columns:
            self.symbols = df["symbol"].astype(str).values
        else:
            self.symbols = np.array(["UNKNOWN"] * len(df))

        if "_symbol_bar_index" in df.columns:
            self.symbol_bar_index = df["_symbol_bar_index"].values.astype(int)
        else:
            self.symbol_bar_index = np.arange(len(df))

        if "datetime" in df.columns:
            self.datetimes = df["datetime"].values
        else:
            self.datetimes = np.arange(len(df))

        self.release_index = precompute_release_indices(
            self.symbols,
            self.symbol_bar_index,
            len(self.df),
            self.max_hold_candles,
        )
        self._condition_mask_cache: dict[tuple[str, ...], np.ndarray] = {}

    def _build_trade_outcome_single(
        self, idx: int, tp: float, sl: float
    ) -> tuple[float, str]:
        """
        Compute (price_return_pct, exit_reason) for a single trade.

        Mirrors evaluator_v3.ipynb's _build_trade_outcome_single exactly.
        """
        s_max = float(self.max_ret[idx])
        s_min = float(self.min_ret[idx])
        s_close = float(self.close_ret[idx])
        s_mbm = self.max_before_min[idx]

        if self.trade_direction == "long":
            hit_tp = s_max >= tp
            hit_sl = s_min <= -sl
            both_hit = hit_tp and hit_sl
            if both_hit:
                return (float(tp), "TP") if s_mbm == 1 else (float(-sl), "SL")
            if hit_tp:
                return float(tp), "TP"
            if hit_sl:
                return float(-sl), "SL"
            return float(s_close), "Time_288"

        # short
        hit_tp = s_min <= -tp
        hit_sl = s_max >= sl
        both_hit = hit_tp and hit_sl
        if both_hit:
            return (float(-sl), "SL") if s_mbm == 1 else (float(tp), "TP")
        if hit_tp:
            return float(tp), "TP"
        if hit_sl:
            return float(-sl), "SL"
        return float(-s_close), "Time_288"

    def _calculate_position_notional(
        self,
        equity: float,
        capital_pct: float,
        open_total_exposure: float,
    ) -> tuple[float, dict]:
        """
        Compute position notional and sizing info.

        Mirrors evaluator_v3.ipynb's _calculate_position_notional exactly.
        """
        capital_rate = float(capital_pct) / 100.0
        target = equity * capital_rate * self.leverage
        max_exposure = equity * self.max_total_exposure_rate * self.leverage
        remaining = max(0.0, max_exposure - open_total_exposure)
        position_notional = min(target, remaining)

        sizing_info = {
            "capital_pct": float(capital_pct),
            "round_trip_fee_rate": self.round_trip_fee_rate,
            "target_position_notional": target,
            "open_total_exposure_before": open_total_exposure,
            "remaining_total_capacity": remaining,
            "final_position_notional": position_notional,
        }
        return position_notional, sizing_info

    def _release_due_positions(
        self,
        open_positions: list,
        current_index: int,
        equity: float,
        peak_equity: float,
        max_drawdown_pct: float,
        open_total_exposure: float,
        symbol_exposure: dict,
        return_logs: bool,
        logs: list,
        stats: dict,
    ) -> tuple:
        """
        Release all positions whose release_index <= current_index.

        Mirrors evaluator_v3.ipynb's _release_due_positions exactly.
        """
        still_open = []

        for pos in open_positions:
            if pos["release_index"] <= current_index:
                equity_before_close = equity
                equity += pos["net_pnl"]

                peak_equity = max(peak_equity, equity)
                drawdown_pct = (
                    (peak_equity - equity) / peak_equity * 100.0
                    if peak_equity > 0
                    else 100.0
                )
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)

                if pos["net_pnl"] > 0:
                    stats["wins"] += 1
                    stats["gross_profit_sum"] += pos["net_pnl"]
                elif pos["net_pnl"] < 0:
                    stats["loss_count"] += 1
                    stats["gross_loss_sum"] += abs(pos["net_pnl"])

                if pos["exit_reason"] == "Time_288":
                    stats["time_closed_count"] += 1

                if self._regime_ids is not None and self._n_regimes > 0:
                    entry_idx = pos["entry_index"]
                    regime = int(self._regime_ids[entry_idx])
                    if 0 <= regime < self._n_regimes:
                        stats["regime_trade_counts"][regime] += 1
                        if pos["net_pnl"] > 0:
                            stats["regime_win_counts"][regime] += 1
                        stats["regime_net_pnl"][regime] += pos["net_pnl"]

                account_status = "ACTIVE"
                if equity <= 0:
                    account_status = "ACCOUNT_RUINED"

                if return_logs and pos["log_index"] is not None:
                    log_idx = pos["log_index"]
                    release_idx = pos["release_index"]

                    if release_idx < len(self.datetimes):
                        close_time = self.datetimes[release_idx]
                    else:
                        close_time = (
                            self.datetimes[-1] if len(
                                self.datetimes) > 0 else np.nan
                        )

                    logs[log_idx].update(
                        {
                            "Release_Index": int(release_idx),
                            "Close_Time": close_time,
                            "Equity_Before_Close": equity_before_close,
                            "Equity_After": equity,
                            "Account_Return_Pct": (
                                equity / self.initial_capital - 1.0
                            )
                            * 100.0,
                            "High_Water_Mark": peak_equity,
                            "Drawdown_Pct": drawdown_pct,
                            "Account_Status": account_status,
                            "Realized": True,
                        }
                    )

                if account_status == "ACCOUNT_RUINED":
                    stats["account_ruined"] = True

                open_total_exposure -= pos["position_notional"]
                sym = pos["symbol"]
                new_sym_exp = symbol_exposure.get(
                    sym, 0.0) - pos["position_notional"]
                if new_sym_exp > 0:
                    symbol_exposure[sym] = new_sym_exp
                elif sym in symbol_exposure:
                    del symbol_exposure[sym]
            else:
                still_open.append(pos)

        return (
            still_open,
            equity,
            peak_equity,
            max_drawdown_pct,
            open_total_exposure,
            symbol_exposure,
            stats,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simulate_rule_set_slice(
        self,
        rule_set: list[dict],
        row_start: int,
        row_end: int,
        initial_capital: float | None = None,
    ) -> dict:
        """
        Simulate a rule set on rows [row_start, row_end) without copying the df.

        Used by Phase 4 RL env to avoid per-step DataFrame/engine allocation.
        """
        entries = _build_entries_from_rule_set(
            self.df, rule_set, self._condition_mask_cache,
        )
        entries = [
            e for e in entries
            if row_start <= int(e["idx"]) < row_end
        ]
        cap = self.initial_capital if initial_capital is None else float(
            initial_capital
        )
        return self._simulate_rule_set_entries(entries, return_logs=False, initial_capital=cap)

    def simulate_rule_set(
        self,
        rule_set: list[dict],
        return_logs: bool = False,
    ) -> "dict | tuple[dict, pd.DataFrame]":
        """
        Simulate a rule set and return performance metrics.

        Parameters
        ----------
        rule_set : list[dict]
            Each dict: {"conditions": [...], "tp": float, "sl": float,
                        "capital_pct": float}
        return_logs : bool
            If True, also return a trade log DataFrame.

        Returns
        -------
        dict
            Metrics dict with keys: direction, total_return_pct,
            sortino_ratio, max_drawdown_pct, win_rate, account_ruined, loss_count,
            time_closed_count, raw_signal_count, executed_trades,
            final_equity, profit_factor, avg_position_notional,
            skipped_min_notional_count, max_simultaneous_positions,
            max_total_open_exposure, per_symbol_metrics.
        tuple[dict, pd.DataFrame]
            If return_logs=True, also returns the trade log DataFrame.
        """
        entries = _build_entries_from_rule_set(
            self.df, rule_set, self._condition_mask_cache,
        )
        return self._simulate_rule_set_entries(
            entries, return_logs=return_logs, initial_capital=self.initial_capital
        )

    def simulate_rule_set_from_cache(
        self,
        rule_set: list[dict],
        cache,
        split: str,
        return_logs: bool = False,
    ) -> "dict | tuple[dict, pd.DataFrame]":
        """Simulate using precomputed signal masks from Phase3EvalCache."""
        entries = cache.build_entries(rule_set, split)
        return self._simulate_rule_set_entries(
            entries, return_logs=return_logs, initial_capital=self.initial_capital
        )

    def simulate_rule_set_batch(
        self,
        rule_sets: list[list[dict]],
        max_workers: int | None = None,
        cache=None,
        split: str | None = None,
    ) -> list[dict]:
        """Evaluate multiple rule sets in parallel (ProcessPool, thread fallback)."""
        if not rule_sets:
            return []
        if len(rule_sets) == 1:
            if cache is not None and split is not None:
                return [self.simulate_rule_set_from_cache(
                    rule_sets[0], cache, split)]
            return [self.simulate_rule_set(rule_sets[0])]

        workers = max_workers
        if workers is None:
            workers = int(_cfg.PHASE3_BATCH_WORKERS)

        if cache is not None and split is not None:
            def _eval_one(rs: list[dict]) -> dict:
                return self.simulate_rule_set_from_cache(rs, cache, split)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(_eval_one, rule_sets))

        payloads = [(self, rs) for rs in rule_sets]
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(_batch_eval_rule_set_pickled, payloads))
        except Exception as exc:
            logger.debug(
                "ProcessPool rule-set batch failed (%s); using threads", exc)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                return list(pool.map(_batch_eval_rule_set_pickled, payloads))

    def _simulate_rule_set_entries(
        self,
        entries: list[dict],
        return_logs: bool,
        initial_capital: float,
    ) -> "dict | tuple[dict, pd.DataFrame]":
        _empty_metrics = {
            "direction": self.trade_direction,
            "total_return_pct": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "win_rate": 0.0,
            "account_ruined": False,
            "loss_count": 0,
            "time_closed_count": 0,
            "raw_signal_count": 0,
            "executed_trades": 0,
            "final_equity": initial_capital,
            "profit_factor": 0.0,
            "avg_position_notional": 0.0,
            "skipped_min_notional_count": 0,
            "max_simultaneous_positions": 0,
            "max_total_open_exposure": 0.0,
            "per_symbol_metrics": {},
        }

        if len(entries) == 0:
            return (_empty_metrics, pd.DataFrame()) if return_logs else _empty_metrics

        # --- Simulation state ---
        equity = initial_capital
        peak_equity = initial_capital
        max_drawdown_pct = 0.0

        open_positions: list[dict] = []
        logs: list[dict] = []

        stats = {
            "wins": 0,
            "loss_count": 0,
            "time_closed_count": 0,
            "gross_profit_sum": 0.0,
            "gross_loss_sum": 0.0,
            "account_ruined": False,
        }

        if self._regime_ids is not None and self._n_regimes > 0:
            stats["regime_trade_counts"] = np.zeros(self._n_regimes, dtype=np.int64)
            stats["regime_win_counts"] = np.zeros(self._n_regimes, dtype=np.int64)
            stats["regime_net_pnl"] = np.zeros(self._n_regimes, dtype=np.float64)

        executed_trades = 0
        skipped_min_notional_count = 0
        position_notional_sum = 0.0
        max_simultaneous_positions = 0
        max_total_open_exposure = 0.0
        open_total_exposure = 0.0
        symbol_exposure: dict[str, float] = {}
        trade_returns: list[float] = []

        # Per-symbol tracking (Requirement 15.1)
        sym_trades: dict[str, int] = {}
        sym_wins: dict[str, int] = {}
        sym_net_pnl: dict[str, float] = {}

        # --- Main simulation loop ---
        for entry in entries:
            idx = int(entry["idx"])
            rule_index = int(entry["rule_index"])
            tp = float(entry["tp"])
            sl = float(entry["sl"])
            capital_pct = float(entry["capital_pct"])

            # Release positions due at or before this row
            (
                open_positions,
                equity,
                peak_equity,
                max_drawdown_pct,
                open_total_exposure,
                symbol_exposure,
                stats,
            ) = self._release_due_positions(
                open_positions=open_positions,
                current_index=idx,
                equity=equity,
                peak_equity=peak_equity,
                max_drawdown_pct=max_drawdown_pct,
                open_total_exposure=open_total_exposure,
                symbol_exposure=symbol_exposure,
                return_logs=return_logs,
                logs=logs,
                stats=stats,
            )

            if stats["account_ruined"]:
                break

            symbol = self.symbols[idx]

            position_notional, sizing_info = self._calculate_position_notional(
                equity=equity,
                capital_pct=capital_pct,
                open_total_exposure=open_total_exposure,
            )

            if position_notional < self.min_position_notional:
                skipped_min_notional_count += 1
                continue

            price_return_pct, exit_reason = self._build_trade_outcome_single(
                idx, tp, sl
            )
            price_return_rate = price_return_pct / 100.0

            gross_pnl = position_notional * price_return_rate
            fee = position_notional * self.round_trip_fee_rate
            net_pnl = gross_pnl - fee
            trade_returns.append(net_pnl / equity if equity > 0.0 else 0.0)
            margin_used = position_notional / max(self.leverage, 1e-9)

            release_idx = int(self.release_index[idx])

            log_index = None
            if return_logs:
                log_index = len(logs)
                logs.append(
                    {
                        "Trade_Number": executed_trades + 1,
                        "Direction": self.trade_direction,
                        "Rule_Index": rule_index,
                        "Rule_TP": tp,
                        "Rule_SL": sl,
                        "Symbol": symbol,
                        "Entry_Time": self.datetimes[idx],
                        "Entry_Index": int(idx),
                        "Symbol_Bar_Index": int(self.symbol_bar_index[idx]),
                        "Entry_Price": self.entry_price[idx],
                        "Release_Index": release_idx,
                        "Close_Time": None,
                        "Exit_Reason": exit_reason,
                        "Price_Return_Pct": price_return_pct,
                        "Equity_Before_Entry": equity,
                        "Capital_Pct": sizing_info["capital_pct"],
                        "Round_Trip_Fee_Rate": sizing_info["round_trip_fee_rate"],
                        "Target_Position_Notional": sizing_info[
                            "target_position_notional"
                        ],
                        "Position_Notional": position_notional,
                        "Margin_Used": margin_used,
                        "Open_Total_Exposure_Before": sizing_info[
                            "open_total_exposure_before"
                        ],
                        "Open_Symbol_Exposure_Before": symbol_exposure.get(
                            symbol, 0.0
                        ),
                        "Remaining_Total_Capacity": sizing_info[
                            "remaining_total_capacity"
                        ],
                        "Gross_PnL": gross_pnl,
                        "Fee": fee,
                        "Net_PnL": net_pnl,
                        "Equity_Before_Close": None,
                        "Equity_After": None,
                        "Account_Return_Pct": None,
                        "High_Water_Mark": None,
                        "Drawdown_Pct": None,
                        "Account_Status": "OPEN",
                        "Realized": False,
                    }
                )

            open_positions.append(
                {
                    "symbol": symbol,
                    "entry_index": int(idx),
                    "release_index": release_idx,
                    "position_notional": position_notional,
                    "margin_used": margin_used,
                    "gross_pnl": gross_pnl,
                    "fee": fee,
                    "net_pnl": net_pnl,
                    "exit_reason": exit_reason,
                    "log_index": log_index,
                }
            )
            open_total_exposure += position_notional
            symbol_exposure[symbol] = (
                symbol_exposure.get(symbol, 0.0) + position_notional
            )

            executed_trades += 1
            position_notional_sum += position_notional

            # Per-symbol tracking
            sym_trades[symbol] = sym_trades.get(symbol, 0) + 1
            sym_net_pnl[symbol] = sym_net_pnl.get(symbol, 0.0) + net_pnl
            # wins tracked after release; accumulate here for later reconciliation
            # (we track wins per symbol in the release step via logs)

            max_simultaneous_positions = max(
                max_simultaneous_positions, len(open_positions)
            )
            max_total_open_exposure = max(
                max_total_open_exposure, open_total_exposure
            )

        # --- Final release: close all remaining open positions ---
        (
            open_positions,
            equity,
            peak_equity,
            max_drawdown_pct,
            open_total_exposure,
            symbol_exposure,
            stats,
        ) = self._release_due_positions(
            open_positions=open_positions,
            current_index=len(self.df),
            equity=equity,
            peak_equity=peak_equity,
            max_drawdown_pct=max_drawdown_pct,
            open_total_exposure=open_total_exposure,
            symbol_exposure=symbol_exposure,
            return_logs=return_logs,
            logs=logs,
            stats=stats,
        )

        # --- Summary metrics ---
        total_return_pct = (equity / initial_capital - 1.0) * 100.0
        sortino_ratio = _sortino_ratio_from_returns(trade_returns)
        win_rate = (
            (stats["wins"] / executed_trades) *
            100.0 if executed_trades > 0 else 0.0
        )
        profit_factor = _safe_profit_factor(
            stats["gross_profit_sum"], stats["gross_loss_sum"]
        )
        avg_position_notional = (
            position_notional_sum / executed_trades if executed_trades > 0 else 0.0
        )

        # --- Per-symbol metrics (Requirement 15.1) ---
        # Build from trade log if available; otherwise use accumulated counters
        per_symbol_metrics: dict[str, dict] = {}
        if return_logs and logs:
            logs_df_tmp = pd.DataFrame(logs)
            for sym, grp in logs_df_tmp.groupby("Symbol"):
                realized = grp[grp["Realized"] == True]
                s_trades = len(grp)
                s_wins = int((realized["Net_PnL"] > 0).sum()) if len(
                    realized) > 0 else 0
                s_net_pnl = float(grp["Net_PnL"].sum())
                s_win_rate = (s_wins / s_trades *
                              100.0) if s_trades > 0 else 0.0
                per_symbol_metrics[str(sym)] = {
                    "trade_count": s_trades,
                    "win_rate": s_win_rate,
                    "net_pnl": s_net_pnl,
                }
        else:
            # Approximate from accumulated counters (net_pnl is exact; wins not tracked per-symbol without logs)
            for sym in sym_trades:
                s_trades = sym_trades[sym]
                s_net_pnl = sym_net_pnl.get(sym, 0.0)
                per_symbol_metrics[sym] = {
                    "trade_count": s_trades,
                    "win_rate": 0.0,  # cannot compute without per-trade outcome tracking
                    "net_pnl": s_net_pnl,
                }

        metrics = {
            "direction": self.trade_direction,
            "total_return_pct": total_return_pct,
            "sortino_ratio": sortino_ratio,
            "max_drawdown_pct": max_drawdown_pct,
            "win_rate": win_rate,
            "account_ruined": bool(stats["account_ruined"]),
            "loss_count": stats["loss_count"],
            "time_closed_count": stats["time_closed_count"],
            "raw_signal_count": len(entries),
            "executed_trades": executed_trades,
            "final_equity": equity,
            "profit_factor": profit_factor,
            "avg_position_notional": avg_position_notional,
            "skipped_min_notional_count": skipped_min_notional_count,
            "max_simultaneous_positions": max_simultaneous_positions,
            "max_total_open_exposure": max_total_open_exposure,
            "per_symbol_metrics": per_symbol_metrics,
        }

        if self._regime_ids is not None and self._n_regimes > 0:
            metrics["regime_trade_counts"] = stats["regime_trade_counts"].tolist()
            metrics["regime_win_counts"] = stats["regime_win_counts"].tolist()
            metrics["regime_net_pnl"] = stats["regime_net_pnl"].tolist()

        if return_logs:
            logs_df = pd.DataFrame(logs)
            return metrics, logs_df

        return metrics

    def simulate_rule_batch(
        self,
        chromosomes: np.ndarray,
        tp: float,
        sl: float,
        capital_pct: float,
    ) -> list[dict]:
        """Evaluate a batch of rule chromosomes on CPU."""
        chromosomes = np.asarray(chromosomes, dtype=np.int32)
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
            compute_rule_signals_numpy,
            is_sparse_batch,
        )

        if chromosomes.ndim == 1:
            chromosomes = chromosomes[None, :]
        if is_sparse_batch(chromosomes) and chromosomes.ndim == 2:
            chromosomes = chromosomes[None, :, :]

        sparse_batch = is_sparse_batch(chromosomes)
        if not sparse_batch:
            B, K = chromosomes.shape
        else:
            B = chromosomes.shape[0]
        if not hasattr(self, "_data_matrix"):
            from gpu_fuzzy_trader.backtest.gpu_engine import _build_data_matrix, _MODE_NUM_CLASSES
            self._feature_names = sorted(list(self.feature_modes.keys()))
            self._data_matrix = _build_data_matrix(self.df, self._feature_names, self.feature_modes)
            self._dont_cares = np.array(
                [_MODE_NUM_CLASSES[self.feature_modes[f]] for f in self._feature_names],
                dtype=np.int32
            )

        results = []
        for b in range(B):
            if sparse_batch:
                signals = compute_rule_signals_numpy(
                    self._data_matrix, chromosomes[b],
                )
            else:
                chromosome = chromosomes[b]
                active_mask = chromosome != self._dont_cares
                condition_match = self._data_matrix == chromosome[None, :]
                effective_match = np.where(
                    active_mask[None, :], condition_match, True)
                signals = np.all(effective_match, axis=-1)
            
            matched_indices = np.flatnonzero(signals)
            entries = [
                {
                    "idx": int(idx),
                    "rule_index": 1,
                    "tp": tp,
                    "sl": sl,
                    "capital_pct": capital_pct,
                }
                for idx in matched_indices
            ]
            metrics = self._simulate_rule_set_entries(
                entries,
                return_logs=False,
                initial_capital=self.initial_capital,
            )
            metrics["raw_signal_count"] = len(matched_indices)
            metrics["skipped_min_notional_count"] = metrics.get("skipped_min_notional_count", 0)
            results.append(metrics)
        return results

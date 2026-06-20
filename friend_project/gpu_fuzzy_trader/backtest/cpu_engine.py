
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg



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


def normalize_symbol_value(value) -> str:
    """Normalize symbol filters exactly like evaluator_v5.ipynb."""
    if pd.isna(value):
        return "__MISSING_SYMBOL__"
    text = str(value).strip()
    if len(text) >= 2 and ((text[0] == text[-1] == "'") or (text[0] == text[-1] == '"')):
        text = text[1:-1].strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1].strip()
    if text.lower().startswith("symbol "):
        text = text.split(None, 1)[1].strip()
    try:
        numeric_value = float(text)
        if np.isfinite(numeric_value) and numeric_value.is_integer():
            return str(int(numeric_value))
    except (TypeError, ValueError):
        pass
    return text


def parse_symbol_condition(condition: str) -> list[str] | None:
    """Parse optional evaluator_v5-compatible symbol filters."""
    if not isinstance(condition, str):
        return None
    match = re.match(
        r"^\s*\[?\s*symbol\s*\]?\s+is\s+(.+?)\s*$",
        condition,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    value_text = match.group(1).strip()
    if not value_text:
        raise ValueError(f"Empty symbol value in condition: {condition!r}")
    values: list[str] = []
    for raw_value in re.split(r"\s*,\s*", value_text):
        normalized = normalize_symbol_value(raw_value)
        if not normalized:
            raise ValueError(f"Empty symbol value in condition: {condition!r}")
        values.append(normalized)
    return values


def split_feature_and_symbol_conditions(conditions: list[str], rule_number: int = 1) -> tuple[list[str], list[str]]:
    """Split normal feature conditions from optional OR-ed symbol filters."""
    if not isinstance(conditions, list) or not conditions:
        raise ValueError(f"Rule {rule_number} must contain a non-empty conditions list.")
    feature_conditions: list[str] = []
    symbol_values: list[str] = []
    for condition in conditions:
        parsed = parse_symbol_condition(condition)
        if parsed is None:
            feature_conditions.append(condition)
        else:
            symbol_values.extend(parsed)
    unique_symbols: list[str] = []
    seen: set[str] = set()
    for value in symbol_values:
        if value not in seen:
            seen.add(value)
            unique_symbols.append(value)
    return feature_conditions, unique_symbols


def get_normalized_symbol_array(df: pd.DataFrame) -> np.ndarray:
    """Build a normalized symbol array once, compatible with evaluator_v5."""
    if "symbol" not in df.columns:
        raise ValueError("Strategy contains a symbol filter, but dataset has no symbol column.")
    codes, uniques = pd.factorize(df["symbol"], sort=False, use_na_sentinel=True)
    normalized_uniques = np.asarray([normalize_symbol_value(v) for v in uniques], dtype=object)
    normalized_symbols = np.empty(len(df), dtype=object)
    valid = codes >= 0
    normalized_symbols[valid] = normalized_uniques[codes[valid]]
    normalized_symbols[~valid] = "__MISSING_SYMBOL__"
    return normalized_symbols


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


def _build_rule_signal_mask(df: pd.DataFrame, conditions: list[str]) -> np.ndarray:
    """AND feature conditions and optional OR-ed symbol filters.

    This mirrors evaluator_v5: conditions such as ``symbol is 1`` or
    ``[symbol] IS 1,2`` are not treated as features; they restrict the rule to
    the listed symbols.  Feature condition thresholds remain mode-independent.
    """
    feature_conditions, allowed_symbols = split_feature_and_symbol_conditions(conditions)
    signal = np.ones(len(df), dtype=bool)
    for cond in feature_conditions:
        signal &= _apply_dynamic_rule(df, cond)
    if allowed_symbols:
        signal &= np.isin(get_normalized_symbol_array(df), allowed_symbols)
    return signal.astype(bool, copy=False)


def _build_entries_from_rule_set(
    df: pd.DataFrame,
    rule_set: list[dict],
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
        rule_signals = _build_rule_signal_mask(df, conditions)

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
        self.feature_modes = feature_modes                                    
        self.feature_infos = constants.get("feature_infos")
        if self.feature_infos is None:
            self.feature_infos = [
                {"name": name, "mode": mode}
                for name, mode in self.feature_modes.items()
            ]
        self.trade_direction = _normalize_direction(direction)

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


    def simulate_rule_batch(
        self,
        chromosomes,
        tp: float,
        sl: float,
        capital_pct: float,
    ) -> list[dict]:
        """Evaluate a batch of Phase-2 chromosomes on CPU.

        This is the CPU fallback equivalent of GPUBacktestEngine.simulate_rule_batch.
        Each chromosome is decoded into one single-rule rule_set and evaluated
        with evaluator_v5-compatible CPU semantics, including symbol filters
        when present in the chromosome-derived conditions.
        """
        from gpu_fuzzy_trader.features.encoder import decode_chromosome

        arr = np.asarray(chromosomes, dtype=np.int32)
        if arr.ndim == 1:
            arr = arr[None, :]

        results: list[dict] = []
        for chrom in arr:
            try:
                conditions = decode_chromosome(chrom, self.feature_infos)
                if not conditions:
                    raise ValueError("empty decoded conditions")
                rule_set = [{
                    "conditions": conditions,
                    "tp": float(tp),
                    "sl": float(sl),
                    "capital_pct": float(capital_pct),
                }]
                results.append(self.simulate_rule_set(rule_set, return_logs=False))
            except Exception:
                results.append({
                    "direction": self.trade_direction,
                    "total_return_pct": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown_pct": 100.0,
                    "win_rate": 0.0,
                    "account_ruined": False,
                    "raw_signal_count": 0,
                    "executed_trades": 0,
                    "final_equity": self.initial_capital,
                    "profit_factor": 0.0,
                    "skipped_min_notional_count": 0,
                    "max_simultaneous_positions": 0,
                    "max_total_open_exposure": 0.0,
                    "per_symbol_metrics": {},
                })
        return results

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
        entries = _build_entries_from_rule_set(self.df, rule_set)
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
        entries = _build_entries_from_rule_set(self.df, rule_set)
        return self._simulate_rule_set_entries(
            entries, return_logs=return_logs, initial_capital=self.initial_capital
        )

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

        executed_trades = 0
        skipped_min_notional_count = 0
        position_notional_sum = 0.0
        max_simultaneous_positions = 0
        max_total_open_exposure = 0.0
        open_total_exposure = 0.0
        symbol_exposure: dict[str, float] = {}
        trade_returns: list[float] = []

        sym_trades: dict[str, int] = {}
        sym_wins: dict[str, int] = {}
        sym_losses: dict[str, int] = {}
        sym_gross_profit: dict[str, float] = {}
        sym_gross_loss: dict[str, float] = {}
        sym_net_pnl: dict[str, float] = {}

        for entry in entries:
            idx = int(entry["idx"])
            rule_index = int(entry["rule_index"])
            tp = float(entry["tp"])
            sl = float(entry["sl"])
            capital_pct = float(entry["capital_pct"])

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

            sym_trades[symbol] = sym_trades.get(symbol, 0) + 1
            sym_net_pnl[symbol] = sym_net_pnl.get(symbol, 0.0) + net_pnl
            if net_pnl > 0:
                sym_wins[symbol] = sym_wins.get(symbol, 0) + 1
                sym_gross_profit[symbol] = sym_gross_profit.get(symbol, 0.0) + net_pnl
            elif net_pnl < 0:
                sym_losses[symbol] = sym_losses.get(symbol, 0) + 1
                sym_gross_loss[symbol] = sym_gross_loss.get(symbol, 0.0) + abs(net_pnl)

            max_simultaneous_positions = max(
                max_simultaneous_positions, len(open_positions)
            )
            max_total_open_exposure = max(
                max_total_open_exposure, open_total_exposure
            )

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
                gross_profit = float(realized.loc[realized["Net_PnL"] > 0, "Net_PnL"].sum()) if len(realized) else 0.0
                gross_loss = float(abs(realized.loc[realized["Net_PnL"] < 0, "Net_PnL"].sum())) if len(realized) else 0.0
                per_symbol_metrics[str(sym)] = {
                    "trade_count": s_trades,
                    "win_rate": s_win_rate,
                    "wins": s_wins,
                    "loss_count": int((realized["Net_PnL"] < 0).sum()) if len(realized) > 0 else 0,
                    "net_pnl": s_net_pnl,
                    "profit_factor": _safe_profit_factor(gross_profit, gross_loss),
                }
        else:
            for sym in sym_trades:
                s_trades = sym_trades[sym]
                s_wins = sym_wins.get(sym, 0)
                s_losses = sym_losses.get(sym, 0)
                s_net_pnl = sym_net_pnl.get(sym, 0.0)
                s_gp = sym_gross_profit.get(sym, 0.0)
                s_gl = sym_gross_loss.get(sym, 0.0)
                per_symbol_metrics[sym] = {
                    "trade_count": s_trades,
                    "win_rate": (s_wins / s_trades * 100.0) if s_trades > 0 else 0.0,
                    "wins": s_wins,
                    "loss_count": s_losses,
                    "net_pnl": s_net_pnl,
                    "profit_factor": _safe_profit_factor(s_gp, s_gl),
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

        if return_logs:
            logs_df = pd.DataFrame(logs)
            return metrics, logs_df

        return metrics

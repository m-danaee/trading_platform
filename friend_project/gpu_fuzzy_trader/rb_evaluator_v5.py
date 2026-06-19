from __future__ import annotations

import math
import re
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg


def safe_profit_factor(gross_wins: float, gross_losses: float) -> float:
    if gross_losses <= 0 and gross_wins > 0:
        return 99.0
    if gross_losses <= 0:
        return 0.0
    return float(gross_wins / gross_losses)


def normalize_direction(direction: str | None) -> str:
    direction = "long" if direction is None else str(direction).strip().lower()
    if direction not in {"long", "short"}:
        raise ValueError("direction must be either 'long' or 'short'")
    return direction




def detect_feature_mode(series: pd.Series) -> str:
    unique_vals = series.dropna().unique()
    n_unique = len(unique_vals)
    if n_unique <= 2 and set(unique_vals).issubset({0, 1}):
        return "binary"
    if n_unique <= 3 and set(unique_vals).issubset({-1, 0, 1}):
        return "ternary"
    zero_ratio = (series == 0).mean()
    if series.min() < 0:
        return "sparse_signed" if zero_ratio > 0.3 else "signed"
    return "sparse_positive" if zero_ratio > 0.3 else "positive"


def parse_condition(condition: str) -> tuple[str, str]:
    if not isinstance(condition, str) or " IS " not in condition:
        raise ValueError(f"Invalid condition format: {condition!r}")
    feature_part, value_part = condition.split(" IS ", 1)
    feature_name = feature_part.strip()
    value_name = value_part.strip()
    if not (feature_name.startswith("[") and feature_name.endswith("]")):
        raise ValueError(f"Invalid feature format in condition: {condition!r}")
    feature_name = feature_name[1:-1].strip()
    if not feature_name or not value_name:
        raise ValueError(f"Empty feature/value in condition: {condition!r}")
    return feature_name, value_name


def normalize_symbol_value(value: Any) -> str:
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
    if not isinstance(condition, str):
        return None
    match = re.match(r"^\s*\[?\s*symbol\s*\]?\s+is\s+(.+?)\s*$", condition, flags=re.IGNORECASE)
    if match is None:
        return None
    value_text = match.group(1).strip()
    if not value_text:
        raise ValueError(f"Empty symbol value in condition: {condition!r}")
    out = []
    for raw_value in re.split(r"\s*,\s*", value_text):
        val = normalize_symbol_value(raw_value)
        if not val:
            raise ValueError(f"Empty symbol value in condition: {condition!r}")
        out.append(val)
    return out


def split_feature_and_symbol_conditions(conditions: list[str], rule_number: int = 1) -> tuple[list[str], list[str]]:
    if not isinstance(conditions, list) or len(conditions) == 0:
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
    for s in symbol_values:
        if s not in seen:
            unique_symbols.append(s)
            seen.add(s)
    return feature_conditions, unique_symbols


def get_normalized_symbol_array(df_input: pd.DataFrame) -> np.ndarray:
    if "symbol" not in df_input.columns:
        raise ValueError("Strategy contains symbol filter but dataset has no symbol column.")
    codes, uniques = pd.factorize(df_input["symbol"], sort=False, use_na_sentinel=True)
    normalized_uniques = np.array([normalize_symbol_value(v) for v in uniques], dtype=object)
    normalized_symbols = np.empty(len(df_input), dtype=object)
    valid_mask = codes >= 0
    normalized_symbols[valid_mask] = normalized_uniques[codes[valid_mask]]
    normalized_symbols[~valid_mask] = "__MISSING_SYMBOL__"
    return normalized_symbols


def apply_dynamic_rule(df_input: pd.DataFrame, rule_string: str) -> np.ndarray:
    feature_name, value_name = parse_condition(rule_string)
    if feature_name not in df_input.columns:
        raise ValueError(f"Unknown feature in strategy condition: {feature_name!r}")
    s = df_input[feature_name]

    if value_name == "Active (1)":
        return np.asarray(s == 1, dtype=bool)
    if value_name == "Inactive (0)":
        return np.asarray(s == 0, dtype=bool)
    if value_name in ["Positive", "Positive (1)"]:
        return np.asarray(s == 1, dtype=bool)
    if value_name in ["Neutral", "Neutral (0)"]:
        return np.asarray(s == 0, dtype=bool)
    if value_name in ["Negative", "Negative (-1)"]:
        return np.asarray(s == -1, dtype=bool)

    if value_name in ["Strong Negative", "Strong Negative (e.g. Big Down Gap)"]:
        return np.asarray(s <= -0.25, dtype=bool)
    if value_name == "Weak Negative":
        return np.asarray((s > -0.25) & (s <= -1e-5), dtype=bool)
    if value_name in ["Exactly Zero", "Exactly Zero (No Gap)"]:
        return np.asarray((s > -1e-5) & (s <= 1e-5), dtype=bool)
    if value_name == "Weak Positive":
        return np.asarray((s > 1e-5) & (s <= 0.25), dtype=bool)
    if value_name in ["Strong Positive", "Strong Positive (e.g. Big Up Gap)"]:
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

    raise ValueError(f"Rule value {value_name!r} is not recognized for feature {feature_name!r}.")


def build_rule_signal_mask(df_input: pd.DataFrame, conditions: list[str]) -> np.ndarray:
    feature_conditions, allowed_symbols = split_feature_and_symbol_conditions(conditions, 1)
    signal = np.ones(len(df_input), dtype=bool)
    for condition in feature_conditions:
        signal &= np.asarray(apply_dynamic_rule(df_input, condition), dtype=bool)
    if allowed_symbols:
        signal &= np.isin(get_normalized_symbol_array(df_input), allowed_symbols)
    return signal.astype(bool, copy=False)

def _read_csv_like_evaluator(path: str | os.PathLike[str]) -> pd.DataFrame:
    """Read CSV the same way evaluator_v5 reads the evaluation file."""
    path = str(path)
    try:
        df = pd.read_csv(path)
        if len(df.columns) < 5:
            df = pd.read_csv(path, sep="\t")
    except Exception:
        df = pd.read_csv(path, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_reference_dataset_schema_exact(path: str | os.PathLike[str]) -> tuple[pd.DataFrame, list[str], dict[str, str], list[str]]:
    """Mirror evaluator_v5.load_reference_dataset_schema exactly enough for scoring."""
    data = pd.read_csv(path)
    if "datetime" in data.columns:
        data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce", utc=True)
        data["datetime"] = data["datetime"].dt.tz_localize(None)
    if "datetime" in data.columns and "symbol" in data.columns:
        data = data.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    elif "datetime" in data.columns:
        data = data.sort_values("datetime").reset_index(drop=True)
    elif "symbol" in data.columns:
        data = data.sort_values("symbol").reset_index(drop=True)
    if "symbol" in data.columns:
        data["_symbol_bar_index"] = data.groupby("symbol").cumcount()
    else:
        data["_symbol_bar_index"] = np.arange(len(data))
    label_columns = ["label_open_next", "label_close_288", "label_min_288", "label_max_288", "label_max_before_min"]
    missing_labels = [c for c in label_columns if c not in data.columns]
    if missing_labels:
        raise ValueError(f"Missing required label columns: {missing_labels}")
    meta_columns = ["datetime", "symbol", "dataset_type", "_symbol_bar_index"]
    features = [c for c in data.columns if c not in label_columns + meta_columns]
    data = data.dropna(subset=label_columns).reset_index(drop=True)
    data[features] = data[features].fillna(0)
    if "symbol" in data.columns:
        data["_symbol_bar_index"] = data.groupby("symbol").cumcount()
    else:
        data["_symbol_bar_index"] = np.arange(len(data))
    modes = {col: detect_feature_mode(data[col]) for col in features}
    return data, features, modes, label_columns


def load_dataset_for_evaluation_exact(file_path: str | os.PathLike[str], feature_cols: list[str], label_cols: list[str]) -> pd.DataFrame:
    """Mirror evaluator_v5.load_and_prepare_dataset_for_evaluation."""
    df_eval = _read_csv_like_evaluator(file_path)
    missing_labels = [c for c in label_cols if c not in df_eval.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns in {file_path}: {missing_labels}")
    missing_features = [c for c in feature_cols if c not in df_eval.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns in {file_path}: {missing_features}")
    if "datetime" in df_eval.columns:
        df_eval["datetime"] = pd.to_datetime(df_eval["datetime"], errors="coerce", utc=True)
        df_eval["datetime"] = df_eval["datetime"].dt.tz_localize(None)
    if "datetime" in df_eval.columns and "symbol" in df_eval.columns:
        df_eval = df_eval.sort_values(["datetime", "symbol"]).reset_index(drop=True)
    elif "datetime" in df_eval.columns:
        df_eval = df_eval.sort_values("datetime").reset_index(drop=True)
    elif "symbol" in df_eval.columns:
        df_eval = df_eval.sort_values("symbol").reset_index(drop=True)
    df_eval = df_eval.dropna(subset=label_cols).reset_index(drop=True)
    df_eval[feature_cols] = df_eval[feature_cols].fillna(0)
    if "symbol" in df_eval.columns:
        df_eval["_symbol_bar_index"] = df_eval.groupby("symbol").cumcount()
    else:
        df_eval["_symbol_bar_index"] = np.arange(len(df_eval))
    return df_eval


def load_evaluator_v5_train_test(reference_schema_path: str | os.PathLike[str] | None = None, evaluation_file_path: str | os.PathLike[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, str], list[str]]:
    """Load train/reference and final evaluation exactly like evaluator_v5."""
    reference_schema_path = reference_schema_path or getattr(_cfg, "EVALUATOR_REFERENCE_SCHEMA_PATH", "data/train.csv")
    evaluation_file_path = evaluation_file_path or getattr(_cfg, "EVALUATOR_EVALUATION_FILE_PATH", "data/test.csv")
    df_train, feature_cols, feature_modes, label_cols = load_reference_dataset_schema_exact(reference_schema_path)
    df_eval = load_dataset_for_evaluation_exact(evaluation_file_path, feature_cols, label_cols)
    return df_train, df_eval, feature_cols, feature_modes, label_cols


_build_rule_signal_mask = build_rule_signal_mask


def _build_entries(df: pd.DataFrame, rule_set: list[dict], row_priority: np.ndarray | None = None) -> list[dict]:
    """Build rule entries using evaluator ordering."""
    if not rule_set:
        return []
    n = len(df)
    assigned_mask = np.zeros(n, dtype=bool)
    if row_priority is None:
        if "datetime" in df.columns:
            codes = pd.factorize(pd.Series(df["datetime"].values), sort=False)[0]
            missing = codes < 0
            if np.any(missing):
                mx = int(codes.max()) if len(codes) else -1
                codes[missing] = mx + 1 + np.arange(int(np.sum(missing)))
            row_priority = codes.astype(np.int64, copy=False)
        else:
            row_priority = np.arange(n, dtype=np.int64)
    normalized_symbols = get_normalized_symbol_array(df) if "symbol" in df.columns else np.array(["UNKNOWN"] * n, dtype=object)
    condition_cache: dict[tuple[str, ...], np.ndarray] = {}
    entries: list[dict] = []
    for rule_idx, rule in enumerate(rule_set, start=1):
        conditions = list(rule.get("conditions", []))
        key = tuple(conditions)
        if key not in condition_cache:
            condition_cache[key] = build_rule_signal_mask(df, conditions)
        rule_signals = condition_cache[key]
        new_match_mask = rule_signals & (~assigned_mask)
        matched_indices = np.flatnonzero(new_match_mask)
        if len(matched_indices) == 0:
            continue
        assigned_mask[matched_indices] = True
        _feature_conditions, allowed_symbols = split_feature_and_symbol_conditions(conditions, rule_idx)
        symbol_priority_map = {sym: pos for pos, sym in enumerate(allowed_symbols)}
        default_symbol_priority = len(symbol_priority_map)
        tp = float(rule["tp"])
        sl = float(rule["sl"])
        cap = float(rule["capital_pct"])
        for idx in matched_indices:
            symbol_priority = symbol_priority_map.get(normalized_symbols[idx], default_symbol_priority) if symbol_priority_map else 0
            entries.append({"idx": int(idx), "entry_priority": int(row_priority[idx]), "rule_index": int(rule_idx), "symbol_priority": int(symbol_priority), "tp": tp, "sl": sl, "capital_pct": cap})
    entries.sort(key=lambda x: (x["entry_priority"], x["rule_index"], x["symbol_priority"], x["idx"]))
    return entries


class EvaluatorV5BacktestEngine:
    """Drop-in replacement for CPUBacktestEngine with evaluator_v5 semantics."""

    def __init__(self, df: pd.DataFrame, feature_infos: dict | None = None, direction: str = "long") -> None:
        self.df = df.reset_index(drop=True)
        self.trade_direction = normalize_direction(direction)
        self.fee_pct = float(getattr(_cfg, "FEE_PCT", 0.20))
        self.round_trip_fee_rate = self.fee_pct / 100.0
        self.initial_capital = float(getattr(_cfg, "INITIAL_CAPITAL", 1000.0))
        self.leverage = float(getattr(_cfg, "LEVERAGE", 1.0)) if hasattr(_cfg, "LEVERAGE") else 1.0
        self.max_hold_candles = int(getattr(_cfg, "MAX_HOLD_CANDLES", 288))
        self.max_total_exposure_rate = float(getattr(_cfg, "MAX_TOTAL_EXPOSURE_PCT", 100.0)) / 100.0
        self.min_position_notional = float(getattr(_cfg, "MIN_POSITION_NOTIONAL", 1.0)) if hasattr(_cfg, "MIN_POSITION_NOTIONAL") else 1.0

        entry = self.df["label_open_next"].values.astype(float)
        invalid = (~np.isfinite(entry)) | (entry <= 0)
        if np.any(invalid):
            raise ValueError(f"Invalid label_open_next values found: {int(np.sum(invalid))}")
        self.entry_price = entry
        self.max_ret = (self.df["label_max_288"].values.astype(float) - entry) / entry * 100.0
        self.min_ret = (self.df["label_min_288"].values.astype(float) - entry) / entry * 100.0
        self.close_ret = (self.df["label_close_288"].values.astype(float) - entry) / entry * 100.0
        self.max_before_min = self.df["label_max_before_min"].values
        self.symbols = self.df["symbol"].astype(str).values if "symbol" in self.df.columns else np.array(["UNKNOWN"] * len(self.df))
        self.symbol_bar_index = self.df["_symbol_bar_index"].values.astype(int) if "_symbol_bar_index" in self.df.columns else np.arange(len(self.df))
        if "datetime" in self.df.columns:
            self.datetimes = self.df["datetime"].values
            entry_time_priority = pd.factorize(pd.Series(self.datetimes), sort=False)[0]
            missing_time_mask = entry_time_priority < 0
            if np.any(missing_time_mask):
                max_code = int(entry_time_priority.max()) if len(entry_time_priority) else -1
                entry_time_priority[missing_time_mask] = max_code + 1 + np.arange(int(np.sum(missing_time_mask)))
            self.entry_time_priority = entry_time_priority.astype(np.int64, copy=False)
        else:
            self.datetimes = np.arange(len(self.df))
            self.entry_time_priority = np.arange(len(self.df), dtype=np.int64)
        self.release_index = self._precompute_release_indices()

    def _precompute_release_indices(self) -> np.ndarray:
        n = len(self.df)
        release_index = np.full(n, n, dtype=np.int64)
        row_index = np.arange(n, dtype=np.int64)
        symbol_codes, _ = pd.factorize(self.symbols, sort=False)
        for sym_code in np.unique(symbol_codes):
            sym_mask = symbol_codes == sym_code
            rows = row_index[sym_mask]
            bars = self.symbol_bar_index[sym_mask].astype(np.int64, copy=False)
            order = np.argsort(bars, kind="mergesort")
            rows_sorted = rows[order]
            bars_sorted = bars[order]
            target_bars = bars_sorted + self.max_hold_candles
            target_positions = np.searchsorted(bars_sorted, target_bars, side="left")
            valid = target_positions < len(rows_sorted)
            if np.any(valid):
                release_index[rows_sorted[valid]] = rows_sorted[target_positions[valid]]
        return release_index

    def _release_due_positions(self, open_positions, current_index, equity, peak_equity, max_drawdown_pct, open_total_exposure, symbol_exposure, stats):
        still_open = []
        for pos in open_positions:
            if pos["release_index"] <= current_index:
                equity += pos["net_pnl"]
                peak_equity = max(peak_equity, equity)
                drawdown_pct = (peak_equity - equity) / peak_equity * 100.0 if peak_equity > 0 else 100.0
                max_drawdown_pct = max(max_drawdown_pct, drawdown_pct)
                if pos["net_pnl"] > 0:
                    stats["wins"] += 1
                    stats["gross_profit_sum"] += pos["net_pnl"]
                elif pos["net_pnl"] < 0:
                    stats["loss_count"] += 1
                    stats["gross_loss_sum"] += abs(pos["net_pnl"])
                if pos["exit_reason"] == "Time_288":
                    stats["time_closed_count"] += 1
                if equity <= 0:
                    stats["account_ruined"] = True
                open_total_exposure -= pos["position_notional"]
                sym = pos["symbol"]
                new_sym_exposure = symbol_exposure.get(sym, 0.0) - pos["position_notional"]
                if new_sym_exposure > 0:
                    symbol_exposure[sym] = new_sym_exposure
                elif sym in symbol_exposure:
                    del symbol_exposure[sym]
            else:
                still_open.append(pos)
        return still_open, equity, peak_equity, max_drawdown_pct, open_total_exposure, symbol_exposure, stats

    def _calculate_position_notional(self, equity: float, capital_pct: float, open_total_exposure: float) -> float:
        target_position_notional = equity * (float(capital_pct) / 100.0) * self.leverage
        max_total_exposure = equity * self.max_total_exposure_rate * self.leverage
        remaining_total_capacity = max(0.0, max_total_exposure - open_total_exposure)
        return min(target_position_notional, remaining_total_capacity)

    def _build_trade_outcome_single(self, idx: int, tp: float, sl: float) -> tuple[float, str]:
        s_max = self.max_ret[idx]
        s_min = self.min_ret[idx]
        s_close = self.close_ret[idx]
        s_mbm = self.max_before_min[idx]
        if self.trade_direction == "long":
            hit_tp = s_max >= tp
            hit_sl = s_min <= -sl
            both = hit_tp and hit_sl
            if both:
                return (float(tp), "TP") if s_mbm == 1 else (float(-sl), "SL")
            if hit_tp:
                return float(tp), "TP"
            if hit_sl:
                return float(-sl), "SL"
            return float(s_close), "Time_288"
        hit_tp = s_min <= -tp
        hit_sl = s_max >= sl
        both = hit_tp and hit_sl
        if both:
            return (float(-sl), "SL") if s_mbm == 1 else (float(tp), "TP")
        if hit_tp:
            return float(tp), "TP"
        if hit_sl:
            return float(-sl), "SL"
        return float(-s_close), "Time_288"

    def simulate_rule_set(self, rule_set: list[dict], return_logs: bool = False):
        entries = _build_entries(self.df, rule_set, row_priority=self.entry_time_priority)
        if len(entries) == 0:
            metrics = {
                "direction": self.trade_direction,
                "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "account_ruined": False,
                "loss_count": 0,
                "time_closed_count": 0,
                "raw_signal_count": 0,
                "executed_trades": 0,
                "final_equity": self.initial_capital,
                "profit_factor": 0.0,
                "avg_position_notional": 0.0,
                "skipped_min_notional_count": 0,
                "max_simultaneous_positions": 0,
                "max_total_open_exposure": 0.0,
            }
            return (metrics, pd.DataFrame()) if return_logs else metrics

        equity = self.initial_capital
        peak_equity = self.initial_capital
        max_drawdown_pct = 0.0
        open_positions: list[dict] = []
        stats = {"wins": 0, "loss_count": 0, "time_closed_count": 0, "gross_profit_sum": 0.0, "gross_loss_sum": 0.0, "account_ruined": False}
        executed_trades = 0
        skipped_min_notional_count = 0
        position_notional_sum = 0.0
        max_simultaneous_positions = 0
        max_total_open_exposure = 0.0
        open_total_exposure = 0.0
        symbol_exposure: dict[str, float] = {}

        for entry in entries:
            idx = int(entry["idx"])
            open_positions, equity, peak_equity, max_drawdown_pct, open_total_exposure, symbol_exposure, stats = self._release_due_positions(
                open_positions, idx, equity, peak_equity, max_drawdown_pct, open_total_exposure, symbol_exposure, stats
            )
            if stats["account_ruined"]:
                break
            symbol = self.symbols[idx]
            position_notional = self._calculate_position_notional(equity, float(entry["capital_pct"]), open_total_exposure)
            if position_notional < self.min_position_notional:
                skipped_min_notional_count += 1
                continue
            price_return_pct, exit_reason = self._build_trade_outcome_single(idx, float(entry["tp"]), float(entry["sl"]))
            gross_pnl = position_notional * (price_return_pct / 100.0)
            fee = position_notional * self.round_trip_fee_rate
            net_pnl = gross_pnl - fee
            release_idx = int(self.release_index[idx])
            open_positions.append({
                "symbol": symbol,
                "entry_index": idx,
                "release_index": release_idx,
                "position_notional": position_notional,
                "net_pnl": net_pnl,
                "exit_reason": exit_reason,
            })
            open_total_exposure += position_notional
            symbol_exposure[symbol] = symbol_exposure.get(symbol, 0.0) + position_notional
            executed_trades += 1
            position_notional_sum += position_notional
            max_simultaneous_positions = max(max_simultaneous_positions, len(open_positions))
            max_total_open_exposure = max(max_total_open_exposure, open_total_exposure)

        open_positions, equity, peak_equity, max_drawdown_pct, open_total_exposure, symbol_exposure, stats = self._release_due_positions(
            open_positions, len(self.df), equity, peak_equity, max_drawdown_pct, open_total_exposure, symbol_exposure, stats
        )

        total_return_pct = (equity / self.initial_capital - 1.0) * 100.0
        win_rate = (stats["wins"] / executed_trades) * 100.0 if executed_trades > 0 else 0.0
        profit_factor = safe_profit_factor(stats["gross_profit_sum"], stats["gross_loss_sum"])
        avg_position_notional = position_notional_sum / executed_trades if executed_trades > 0 else 0.0
        metrics = {
            "direction": self.trade_direction,
            "total_return_pct": float(total_return_pct),
            "max_drawdown_pct": float(max_drawdown_pct),
            "win_rate": float(win_rate),
            "account_ruined": bool(stats["account_ruined"]),
            "loss_count": int(stats["loss_count"]),
            "time_closed_count": int(stats["time_closed_count"]),
            "raw_signal_count": int(len(entries)),
            "executed_trades": int(executed_trades),
            "final_equity": float(equity),
            "profit_factor": float(profit_factor),
            "avg_position_notional": float(avg_position_notional),
            "skipped_min_notional_count": int(skipped_min_notional_count),
            "max_simultaneous_positions": int(max_simultaneous_positions),
            "max_total_open_exposure": float(max_total_open_exposure),
        }
        return (metrics, pd.DataFrame()) if return_logs else metrics

"""
cpu_engine.py — CPUBacktestEngine

Exact Python/NumPy replication of evaluator_v5.ipynb's
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
from gpu_fuzzy_trader.backtest.barrier import (
    barrier_column_names,
    configured_barrier_pairs,
)
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
# Helpers — mirrors evaluator_v5.ipynb utilities
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


def precompute_release_indices_from_offsets(
    symbols: np.ndarray,
    symbol_bar_index: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    """Map exact first-touch bar offsets to row indices in the current frame.

    ``offsets`` are computed on the complete per-symbol source tape.  A
    split/backtest frame may contain only a prefix or a validation tail, so
    the target is resolved by the stable per-symbol bar index rather than by
    assuming rows are globally contiguous.  Targets outside the current
    frame are released by the caller's final flush.
    """
    offsets = np.asarray(offsets)
    n_rows = len(offsets)
    release_index = np.full(n_rows, n_rows, dtype=np.int64)
    row_index = np.arange(n_rows, dtype=np.int64)
    symbol_codes, _ = pd.factorize(symbols, sort=False)

    for sym_code in np.unique(symbol_codes):
        sym_mask = symbol_codes == sym_code
        rows = row_index[sym_mask]
        bars = symbol_bar_index[sym_mask].astype(np.int64, copy=False)
        local_offsets = offsets[sym_mask].astype(np.int64, copy=False)

        order = np.argsort(bars, kind="mergesort")
        rows_sorted = rows[order]
        bars_sorted = bars[order]
        offsets_sorted = local_offsets[order]
        valid_offset = offsets_sorted >= 0
        target_bars = bars_sorted + np.maximum(offsets_sorted, 0)
        target_positions = np.searchsorted(
            bars_sorted, target_bars, side="left",
        )
        valid = valid_offset & (target_positions < len(rows_sorted))
        if np.any(valid):
            release_index[rows_sorted[valid]] = rows_sorted[
                target_positions[valid]
            ]

    return release_index


def _safe_profit_factor(gross_wins: float, gross_losses: float) -> float:
    if gross_losses <= 0 and gross_wins > 0:
        return 99.0
    if gross_losses <= 0:
        return 0.0
    return gross_wins / gross_losses


def _expectancy_lcb_pct(
    trade_returns: list[float] | np.ndarray,
    *,
    z_score: float,
) -> tuple[float, float, float]:
    """Return mean trade return, sample std, and a normal lower bound in %."""
    values = np.asarray(trade_returns, dtype=np.float64) * 100.0
    if values.size == 0:
        return 0.0, 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    standard_error = std / float(np.sqrt(values.size))
    return mean, std, mean - float(z_score) * standard_error


def _expected_shortfall_pct(
    trade_returns: list[float] | np.ndarray,
    *,
    tail_fraction: float,
) -> float:
    """Mean of the worst trade-return tail, expressed in percentage points."""
    values = np.asarray(trade_returns, dtype=np.float64) * 100.0
    if values.size == 0:
        return 0.0
    fraction = min(1.0, max(1.0 / values.size, float(tail_fraction)))
    tail_count = max(1, int(np.ceil(values.size * fraction)))
    return float(np.mean(np.sort(values)[:tail_count]))


def _sortino_ratio_from_returns(
    trade_returns: list[float] | np.ndarray,
    target_return: float = 0.0,
    scale_by_trades: bool = False,
) -> float:
    """Compute a non-annualized Sortino Ratio from per-trade returns."""
    returns = np.asarray(trade_returns, dtype=np.float64)
    n_trades = returns.size
    if n_trades == 0:
        return 0.0

    excess_returns = returns - float(target_return)
    mean_excess_return = float(np.mean(excess_returns))
    downside_returns = np.minimum(excess_returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside_returns))))

    from gpu_fuzzy_trader import config as _cfg

    if downside_deviation <= 0.0:
        base_sortino = _cfg.SORTINO_CAP if mean_excess_return > 0.0 else 0.0
    else:
        base_sortino = min(mean_excess_return / downside_deviation, _cfg.SORTINO_CAP)

    if scale_by_trades and hasattr(_cfg, "PHASE2_SORTINO_MIN_TRADE_THRESHOLD"):
        threshold = float(_cfg.PHASE2_SORTINO_MIN_TRADE_THRESHOLD)
        if threshold > 0:
            scale_factor = min(1.0, float(n_trades) / threshold)
            return base_sortino * scale_factor

    return base_sortino


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

    Exactly mirrors evaluator_v5.ipynb's apply_dynamic_rule — mode-independent.
    Returns a boolean NumPy array of length len(df).
    """
    feature_name, value_name = _parse_condition(condition)

    if feature_name not in df.columns:
        raise ValueError(
            f"Unknown feature in condition: {feature_name!r}. "
            "Make sure the strategy was created using the same feature set."
        )

    s = df[feature_name]
    mask_fn = _VALUE_MAP.get(value_name)
    if mask_fn is not None:
        return mask_fn(s)

    raise ValueError(
        f"Rule value {value_name!r} is not recognized for feature {feature_name!r}."
    )


def _build_value_map() -> dict:
    s_eq_0 = lambda s: np.asarray(s == 0, dtype=bool)
    s_eq_1 = lambda s: np.asarray(s == 1, dtype=bool)
    s_eq_neg1 = lambda s: np.asarray(s == -1, dtype=bool)
    return {
        "Active (1)": s_eq_1,
        "Inactive (0)": s_eq_0,
        "Positive": s_eq_1, "Positive (1)": s_eq_1,
        "Neutral": s_eq_0, "Neutral (0)": s_eq_0,
        "Negative": s_eq_neg1, "Negative (-1)": s_eq_neg1,
        "Strong Negative": lambda s: np.asarray(s <= -0.25, dtype=bool),
        "Strong Negative (e.g. Big Down Gap)": lambda s: np.asarray(s <= -0.25, dtype=bool),
        "Weak Negative": lambda s: np.asarray((s > -0.25) & (s <= -1e-5), dtype=bool),
        "Exactly Zero": lambda s: np.asarray((s > -1e-5) & (s <= 1e-5), dtype=bool),
        "Exactly Zero (No Gap)": lambda s: np.asarray((s > -1e-5) & (s <= 1e-5), dtype=bool),
        "Weak Positive": lambda s: np.asarray((s > 1e-5) & (s <= 0.25), dtype=bool),
        "Strong Positive": lambda s: np.asarray(s > 0.25, dtype=bool),
        "Strong Positive (e.g. Big Up Gap)": lambda s: np.asarray(s > 0.25, dtype=bool),
        "Very Low": lambda s: np.asarray(s <= 0.2, dtype=bool),
        "Low": lambda s: np.asarray((s > 0.2) & (s <= 0.4), dtype=bool),
        "Medium": lambda s: np.asarray((s > 0.4) & (s <= 0.6), dtype=bool),
        "High": lambda s: np.asarray((s > 0.6) & (s <= 0.8), dtype=bool),
        "Very High": lambda s: np.asarray(s > 0.8, dtype=bool),
        "Extreme Bearish": lambda s: np.asarray(s <= -0.8, dtype=bool),
        "Strong Bearish": lambda s: np.asarray((s > -0.8) & (s <= -0.6), dtype=bool),
        "Bearish": lambda s: np.asarray((s > -0.6) & (s <= -0.4), dtype=bool),
        "Weak Bearish": lambda s: np.asarray((s > -0.4) & (s <= -0.2), dtype=bool),
        "Neutral Negative": lambda s: np.asarray((s > -0.2) & (s <= 0.0), dtype=bool),
        "Neutral Positive": lambda s: np.asarray((s > 0.0) & (s <= 0.2), dtype=bool),
        "Weak Bullish": lambda s: np.asarray((s > 0.2) & (s <= 0.4), dtype=bool),
        "Bullish": lambda s: np.asarray((s > 0.4) & (s <= 0.6), dtype=bool),
        "Strong Bullish": lambda s: np.asarray((s > 0.6) & (s <= 0.8), dtype=bool),
        "Extreme Bullish": lambda s: np.asarray(s > 0.8, dtype=bool),
    }


_VALUE_MAP = _build_value_map()


def _compute_rule_signal_mask(
    df: pd.DataFrame,
    conditions: list[str],
    rule_number: int = 1,
) -> np.ndarray:
    """
    Build one boolean signal mask (evaluator_v5 parity).

    Feature conditions are AND-ed. Symbol conditions are OR-ed, then AND-ed
    with the feature signal. Rules without symbol filters apply to all symbols.
    """
    feature_conditions, allowed_symbols = split_feature_and_symbol_conditions(
        conditions=conditions,
        rule_number=rule_number,
    )

    masks = [np.asarray(_apply_dynamic_rule(df, cond), dtype=bool)
             for cond in feature_conditions]

    if allowed_symbols:
        normalized_symbols = get_normalized_symbol_array(df)
        masks.append(np.isin(normalized_symbols, allowed_symbols))

    if not masks:
        return np.ones(len(df), dtype=bool)

    if len(masks) == 1:
        return masks[0].astype(bool, copy=False)

    stacked = np.stack(masks, axis=0)
    return np.all(stacked, axis=0)


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


def compute_entry_time_priority(
    datetimes: np.ndarray,
    n_rows: int | None = None,
) -> np.ndarray:
    """Map each row to a timestamp priority code (evaluator_v5 parity)."""
    if n_rows is None:
        n_rows = len(datetimes)
    entry_time_priority = pd.factorize(pd.Series(datetimes), sort=False)[0]
    missing_time_mask = entry_time_priority < 0
    if np.any(missing_time_mask):
        max_code = int(entry_time_priority.max()) if len(
            entry_time_priority) else -1
        entry_time_priority[missing_time_mask] = (
            max_code + 1 + np.arange(int(np.sum(missing_time_mask)))
        )
    return entry_time_priority.astype(np.int64, copy=False)


def _rule_symbols_for_allocation(
    rule_entry: dict,
    conditions: list[str],
    rule_idx: int,
) -> list[str]:
    symbols = rule_entry.get("symbols")
    if symbols:
        return list(symbols)
    _, allowed_symbols = split_feature_and_symbol_conditions(
        conditions=conditions,
        rule_number=rule_idx,
    )
    return allowed_symbols


def _sort_entries_by_allocation_priority(entries: list[dict]) -> None:
    """Sort entries for v5 capital allocation (timestamp, rule, symbol, row)."""
    entries.sort(
        key=lambda x: (
            x["entry_priority"],
            x["rule_index"],
            x["symbol_priority"],
            x["idx"],
        )
    )


def _append_allocated_entries(
    entries: list[dict],
    matched_indices: np.ndarray,
    *,
    rule_idx: int,
    tp: float,
    sl: float,
    capital_pct: float,
    row_priority: np.ndarray,
    rule_symbols: list[str],
    normalized_symbols: np.ndarray | None,
) -> None:
    symbol_priority_map = {sym: pos for pos, sym in enumerate(rule_symbols)}
    default_symbol_priority = len(symbol_priority_map)

    for idx in matched_indices:
        if normalized_symbols is not None and symbol_priority_map:
            symbol_priority = symbol_priority_map.get(
                normalized_symbols[idx],
                default_symbol_priority,
            )
        else:
            symbol_priority = 0

        entries.append(
            {
                "idx": int(idx),
                "entry_priority": int(row_priority[idx]),
                "rule_index": int(rule_idx),
                "symbol_priority": int(symbol_priority),
                "tp": tp,
                "sl": sl,
                "capital_pct": capital_pct,
            }
        )


def _build_entries_from_rule_set(
    df: pd.DataFrame,
    rule_set: list[dict],
    mask_cache: dict[tuple[str, ...], np.ndarray] | None = None,
    row_priority: np.ndarray | None = None,
    normalized_symbols: np.ndarray | None = None,
    context_mask: np.ndarray | None = None,
) -> list[dict]:
    """
    Priority-based rule assignment: first matching rule wins per row.

    Mirrors evaluator_v5.ipynb build_rule_set_entries() with signal_mask path.
    Simultaneous cross-symbol signals are ordered by JSON rule order, not dataset
    symbol sort order.
    """
    if not rule_set:
        return []

    n_rows = len(df)
    if context_mask is None:
        context_mask = np.ones(n_rows, dtype=bool)
    else:
        context_mask = np.asarray(context_mask, dtype=bool)
        if len(context_mask) != n_rows:
            raise ValueError(
                "context_mask length does not match dataset length.")
    if row_priority is None:
        if "datetime" in df.columns:
            row_priority = compute_entry_time_priority(
                df["datetime"].values, n_rows)
        else:
            row_priority = np.arange(n_rows, dtype=np.int64)
    else:
        row_priority = np.asarray(row_priority)
        if len(row_priority) != n_rows:
            raise ValueError(
                "row_priority length does not match dataset length.")

    if normalized_symbols is not None:
        normalized_symbols = np.asarray(normalized_symbols, dtype=object)
        if len(normalized_symbols) != n_rows:
            raise ValueError(
                "normalized_symbols length does not match dataset length."
            )
    elif _rules_need_normalized_symbols(rule_set):
        normalized_symbols = get_normalized_symbol_array(df)

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
        # Fixed mandatory context policy ANDed into every rule's signal mask.
        rule_signals = rule_signals & context_mask

        new_match_mask = rule_signals & (~assigned_mask)
        matched_indices = np.flatnonzero(new_match_mask)
        if len(matched_indices) == 0:
            continue

        assigned_mask[matched_indices] = True
        rule_symbols = _rule_symbols_for_allocation(
            rule_entry, conditions, rule_idx
        )
        _append_allocated_entries(
            entries,
            matched_indices,
            rule_idx=rule_idx,
            tp=tp,
            sl=sl,
            capital_pct=capital_pct,
            row_priority=row_priority,
            rule_symbols=rule_symbols,
            normalized_symbols=normalized_symbols,
        )

    _sort_entries_by_allocation_priority(entries)
    return entries


def _rules_need_normalized_symbols(rule_set: list[dict]) -> bool:
    for rule_entry in rule_set:
        if not isinstance(rule_entry, dict):
            continue
        if rule_entry.get("symbols"):
            return True
        conditions = rule_entry.get("conditions", [])
        _, allowed_symbols = split_feature_and_symbol_conditions(
            conditions=conditions,
            rule_number=1,
        )
        if allowed_symbols:
            return True
    return False


# ---------------------------------------------------------------------------
# CPUBacktestEngine
# ---------------------------------------------------------------------------

class CPUBacktestEngine:
    """
    CPU backtest engine that exactly mirrors evaluator_v5.ipynb's
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
        # --- Pre-extract arrays for speed ---
        entry = df["label_open_next"].values.astype(float)

        invalid_mask = (~np.isfinite(entry)) | (entry <= 0)
        if np.any(invalid_mask):
            bad = int(np.sum(invalid_mask))
            raise ValueError(
                f"Invalid label_open_next values (must be finite and positive). "
                f"Bad rows: {bad}"
            )

        self.entry_price = entry.astype(np.float32)
        self.max_ret = ((df["label_max_288"].values - entry) / entry * 100.0).astype(np.float32)
        self.min_ret = ((df["label_min_288"].values - entry) / entry * 100.0).astype(np.float32)
        self.close_ret = ((df["label_close_288"].values - entry) / entry * 100.0).astype(np.float32)
        self.max_before_min = df["label_max_before_min"].values.astype(np.int32)

        if "symbol" in df.columns:
            self.symbols = df["symbol"].astype(str).values
        else:
            self.symbols = np.array(["UNKNOWN"] * len(df))

        if "_symbol_bar_index" in df.columns:
            self.symbol_bar_index = df["_symbol_bar_index"].values.astype(np.int32)
        else:
            self.symbol_bar_index = np.arange(len(df), dtype=np.int32)

        if "datetime" in df.columns:
            self.datetimes = df["datetime"].values
            row_times = pd.to_datetime(df["datetime"])
            if "symbol" in df.columns:
                next_times = row_times.groupby(
                    df["symbol"], sort=False, observed=False,
                ).shift(-1)
            else:
                next_times = row_times.shift(-1)
            self.entry_datetimes = next_times.fillna(
                row_times + pd.Timedelta(
                    minutes=int(_cfg.LWC_TIMEFRAME_MINUTES),
                )
            ).to_numpy()
            self.entry_time_priority = compute_entry_time_priority(
                self.datetimes, len(df)
            )
        else:
            self.datetimes = np.arange(len(df))
            self.entry_datetimes = self.datetimes
            self.entry_time_priority = np.arange(len(df), dtype=np.int64)

        self.release_index = precompute_release_indices(
            self.symbols,
            self.symbol_bar_index,
            len(self.df),
            self.max_hold_candles,
        )
        self._condition_mask_cache: dict[tuple[str, ...], np.ndarray] = {}
        self._trade_outcomes_cache: dict[tuple, np.ndarray] = {}
        self._trade_bundle_cache: dict[
            tuple, tuple[np.ndarray, np.ndarray, np.ndarray]
        ] = {}
        # Exact columns are present only on production tapes prepared by the
        # loader.  Keeping this capability opt-in preserves the historical
        # aggregate-label semantics expected by small synthetic unit fixtures
        # while making real Phase 2/RB/Phase 5 runs use first-touch exits.
        # The loader materialises the Cartesian product of the Phase 2 and RB
        # TP/SL grids.  Check that same complete contract here; checking only
        # zipped grid entries would leave valid cross-product pairs in legacy
        # overlapping-position mode.
        self._exact_barrier_available = any(
            all(column in df.columns for column in barrier_column_names(
                self.trade_direction, tp, sl,
            ))
            for tp, sl in configured_barrier_pairs()
        )
        # ------------------------------------------------------------------
        # Trend-context fixed execution mask (defense-in-depth).
        #
        # The mandatory direction + LWC-trigger policy is applied
        # deterministically here so it holds for exact CPU admission, RB
        # evaluation/composition, Phase 5, and every JSON-injected rule,
        # independent of whether a rule's condition list carries the exported
        # context conditions.  When context columns are absent (legacy /
        # synthetic fixtures) the mask is a no-op.
        # ------------------------------------------------------------------
        self._context_mask = self._build_context_mask(df)

    def _build_context_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Return a per-row boolean mask enforcing the fixed context policy.

        For a long engine the mask is
            ``tf_permission_long == 1 AND lwc_pullback_reversal_long == 1``
        and analogously for short.  Exactly matches the mandatory conditions
        injected at Phase 2 and required by Output_Writer.  When context
        columns are absent the mask is all-True (no-op).
        """
        n = len(df)
        from gpu_fuzzy_trader.config import CONTEXT_COLUMNS as _CTX_COLS
        if not any(c in df.columns for c in _CTX_COLS):
            return np.ones(n, dtype=bool)
        perm = _cfg.context_permission_column(self.trade_direction)
        trig = _cfg.context_trigger_column(self.trade_direction)
        if perm not in df.columns or trig not in df.columns:
            raise ValueError(
                f"Enriched dataframe is missing context columns required for "
                f"direction {self.trade_direction!r}: {perm}, {trig}"
            )
        perm_ok = df[perm].to_numpy() == 1
        trig_ok = df[trig].to_numpy() == 1
        return (perm_ok & trig_ok)

    def _get_trade_outcomes(self, tp: float, sl: float) -> np.ndarray:
        """Precompute (N,) price return % for all rows given fixed TP/SL."""
        return self._get_trade_bundle(tp, sl)[0]

    def _get_trade_bundle(
        self,
        tp: float,
        sl: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(return_pct, exit_offset, release_row)`` for one pair.

        Exact first-touch columns are the production contract.  The fallback
        computes the legacy aggregate-label return and fixed-horizon release
        index so old synthetic fixtures and compatibility callers continue to
        behave deterministically.
        """
        cache_key = (tp, sl, self.trade_direction)
        if cache_key in self._trade_bundle_cache:
            return self._trade_bundle_cache[cache_key]

        ret_name, off_name = barrier_column_names(
            self.trade_direction, float(tp), float(sl),
        )
        if ret_name in self.df.columns and off_name in self.df.columns:
            returns = self.df[ret_name].to_numpy(dtype=np.float32)
            offsets = self.df[off_name].to_numpy(dtype=np.int64)
            release = precompute_release_indices_from_offsets(
                self.symbols,
                self.symbol_bar_index,
                offsets,
            )
            bundle = (returns, offsets, release)
            self._trade_bundle_cache[cache_key] = bundle
            self._trade_outcomes_cache[cache_key] = returns
            return bundle

        if cache_key not in self._trade_outcomes_cache:
            # The scalar helper remains the reference path for trade logs, but
            # Phase 2/RB evaluates the same fixed TP/SL across every row.  Do
            # that preprocessing in NumPy once instead of making one Python
            # call per bar (90k+ calls on the checked-in data).
            # Keep Python-float thresholds for comparisons: the scalar
            # evaluator promotes each float32 row value to Python float before
            # applying TP/SL, and rounding the thresholds here changes exact
            # boundary exit reasons.
            tp_f = float(tp)
            sl_f = float(sl)
            max_ret = self.max_ret.astype(np.float64)
            min_ret = self.min_ret.astype(np.float64)
            if self.trade_direction == "long":
                hit_tp = max_ret >= tp_f
                hit_sl = min_ret <= -sl_f
                both_hit = hit_tp & hit_sl
                both_result = np.where(
                    self.max_before_min == 1, tp_f, -sl_f,
                )
                out = np.where(
                    both_hit,
                    both_result,
                    np.where(
                        hit_tp,
                        tp_f,
                        np.where(hit_sl, -sl_f, self.close_ret),
                    ),
                ).astype(np.float32, copy=False)
            else:
                hit_tp = min_ret <= -tp_f
                hit_sl = max_ret >= sl_f
                both_hit = hit_tp & hit_sl
                both_result = np.where(
                    self.max_before_min == 1, -sl_f, tp_f,
                )
                out = np.where(
                    both_hit,
                    both_result,
                    np.where(
                        hit_tp,
                        tp_f,
                        np.where(hit_sl, -sl_f, -self.close_ret),
                    ),
                ).astype(np.float32, copy=False)
            self._trade_outcomes_cache[cache_key] = out
        bundle = (
            self._trade_outcomes_cache[cache_key],
            np.full(len(self.df), self.max_hold_candles, dtype=np.int64),
            self.release_index,
        )
        self._trade_bundle_cache[cache_key] = bundle
        return bundle

    def _build_trade_outcome_single(
        self, idx: int, tp: float, sl: float
    ) -> tuple[float, str]:
        """
        Compute (price_return_pct, exit_reason) for a single trade.

        Mirrors evaluator_v5.ipynb's _build_trade_outcome_single exactly.
        """
        ret_name, off_name = barrier_column_names(
            self.trade_direction, float(tp), float(sl),
        )
        if ret_name in self.df.columns and off_name in self.df.columns:
            result = float(self.df[ret_name].iloc[idx])
            offset = int(self.df[off_name].iloc[idx])
            # A first-touch return is exactly +/- the configured barrier.  A
            # horizon close can coincidentally land on that value, but using
            # the exact return value is the only deterministic OHLC contract
            # available without tick data.
            tol = 1e-5
            if abs(result - float(tp)) <= tol:
                return result, "TP"
            if abs(result + float(sl)) <= tol:
                return result, "SL"
            return result, f"Time_{self.max_hold_candles}"

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
            # INTENTIONALLY UNCAPPED time-exit return.
            # This branch returns the raw close_ret/-close_ret when a trade
            # holds to MAX_HOLD_CANDLES without hitting TP or SL. A previous
            # attempt to cap this with MAX_TIME_EXIT_RETURN_PCT (commit
            # 46cb88a, default 50.0) was reverted (commit 072c527) to preserve
            # parity with evaluator_v5.ipynb (lines 958, 971), which is the
            # user's ground truth for rule testing per AGENTS.md. Re-applying
            # any cap here would create a divergence that invalidates the
            # user's standalone OOS check. If you want to re-introduce a cap,
            # update evaluator_v5.ipynb FIRST and document the change.
            return float(s_close), f"Time_{self.max_hold_candles}"

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
        # INTENTIONALLY UNCAPPED time-exit return.
        # This branch returns the raw close_ret/-close_ret when a trade
        # holds to MAX_HOLD_CANDLES without hitting TP or SL. A previous
        # attempt to cap this with MAX_TIME_EXIT_RETURN_PCT (commit
        # 46cb88a, default 50.0) was reverted (commit 072c527) to preserve
        # parity with evaluator_v5.ipynb (lines 958, 971), which is the
        # user's ground truth for rule testing per AGENTS.md. Re-applying
        # any cap here would create a divergence that invalidates the
        # user's standalone OOS check. If you want to re-introduce a cap,
        # update evaluator_v5.ipynb FIRST and document the change.
        return float(-s_close), f"Time_{self.max_hold_candles}"

    def _calculate_position_notional(
        self,
        equity: float,
        capital_pct: float,
        open_total_exposure: float,
    ) -> tuple[float, dict]:
        """
        Compute position notional and sizing info.

        Mirrors evaluator_v5.ipynb's _calculate_position_notional exactly.
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
        sym_wins: dict[str, int] | None = None,
    ) -> tuple:
        """
        Release all positions whose release_index <= current_index.

        Mirrors evaluator_v5.ipynb's _release_due_positions exactly.
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
                    if sym_wins is not None:
                        sym = pos["symbol"]
                        sym_wins[sym] = sym_wins.get(sym, 0) + 1
                    stats["gross_profit_sum"] += pos["net_pnl"]
                    # Track max single positive trade PnL for f4 concentration
                    prev_max = float(stats.get("max_single_trade_pnl", 0.0))
                    stats["max_single_trade_pnl"] = max(prev_max, pos["net_pnl"])
                elif pos["net_pnl"] < 0:
                    stats["loss_count"] += 1
                    stats["gross_loss_sum"] += abs(pos["net_pnl"])

                if pos["exit_reason"] == f"Time_{self.max_hold_candles}":
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

        Used by RB scoring to avoid repeated DataFrame/engine allocation.
        """
        normalized_symbols = None
        if _rules_need_normalized_symbols(rule_set):
            normalized_symbols = get_normalized_symbol_array(self.df)

        entries = _build_entries_from_rule_set(
            self.df,
            rule_set,
            self._condition_mask_cache,
            row_priority=self.entry_time_priority,
            normalized_symbols=normalized_symbols,
            context_mask=self._context_mask,
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
        normalized_symbols = None
        if _rules_need_normalized_symbols(rule_set):
            normalized_symbols = get_normalized_symbol_array(self.df)

        entries = _build_entries_from_rule_set(
            self.df,
            rule_set,
            self._condition_mask_cache,
            row_priority=self.entry_time_priority,
            normalized_symbols=normalized_symbols,
            context_mask=self._context_mask,
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
        """Simulate using a precomputed rule-evaluation mask cache.

        The mandatory direction + LWC-trigger context policy is enforced
        unconditionally here (defense-in-depth).  A cache may be built by any
        external rule builder that does not know about the fixed context mask,
        so every entry is post-filtered against ``self._context_mask`` before
        the shared simulation loop runs.  This guarantees the cached path can
        never admit a blocked long/short row.
        """
        entries = cache.build_entries(rule_set, split)
        entries = self._filter_entries_by_context_mask(entries)
        return self._simulate_rule_set_entries(
            entries, return_logs=return_logs, initial_capital=self.initial_capital
        )

    def _filter_entries_by_context_mask(
        self, entries: list[dict],
    ) -> list[dict]:
        """Drop cached entries on rows the fixed context mask forbids.

        ``idx`` validity is checked before indexing so an out-of-range cache
        entry fails closed instead of raising an opaque NumPy IndexError.
        """
        n_rows = len(self.df)
        out: list[dict] = []
        for entry in entries:
            idx = int(entry.get("idx", -1))
            if idx < 0 or idx >= n_rows:
                raise ValueError(
                    f"cache entry idx {idx} is out of range for a dataset of "
                    f"{n_rows} rows."
                )
            if bool(self._context_mask[idx]):
                out.append(entry)
        return out

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
            workers = int(_cfg.BACKTEST_BATCH_WORKERS)

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
            "per_symbol_metrics_available": True,
            "expectancy_pct_per_trade": 0.0,
            "trade_return_std_pct": 0.0,
            "expectancy_lcb_pct_per_trade": 0.0,
            "expected_shortfall_pct": 0.0,
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
                sym_wins=sym_wins,
            )

            if stats["account_ruined"]:
                break

            symbol = self.symbols[idx]

            # Production semantics are one net position per symbol.  A
            # repeated same-side signal while that symbol is already open is
            # ignored; the joint long/short portfolio layer handles opposite
            # direction reversal explicitly.  Legacy fixtures without exact
            # barrier columns retain their original overlapping-position
            # behaviour for backwards-compatible unit coverage.
            if (
                self._exact_barrier_available
                and symbol_exposure.get(symbol, 0.0) > 0.0
            ):
                continue

            position_notional, sizing_info = self._calculate_position_notional(
                equity=equity,
                capital_pct=capital_pct,
                open_total_exposure=open_total_exposure,
            )

            if position_notional < self.min_position_notional:
                skipped_min_notional_count += 1
                continue

            price_returns, _offsets, release_indices = self._get_trade_bundle(
                tp, sl,
            )
            price_return_pct = price_returns[idx]
            exit_reason = None
            if return_logs:
                _, exit_reason = self._build_trade_outcome_single(idx, tp, sl)
            price_return_rate = price_return_pct / 100.0

            gross_pnl = position_notional * price_return_rate
            fee = position_notional * self.round_trip_fee_rate
            net_pnl = gross_pnl - fee
            trade_returns.append(net_pnl / equity if equity > 0.0 else 0.0)
            margin_used = position_notional / max(self.leverage, 1e-9)

            release_idx = int(release_indices[idx])

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
                        "Entry_Time": self.entry_datetimes[idx],
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
            sym_wins=sym_wins,
        )

        # --- Summary metrics ---
        total_return_pct = (equity / initial_capital - 1.0) * 100.0
        sortino_ratio = _sortino_ratio_from_returns(trade_returns, scale_by_trades=True)
        expectancy, trade_return_std, expectancy_lcb = _expectancy_lcb_pct(
            trade_returns,
            z_score=float(getattr(_cfg, "PHASE2_EXPECTANCY_LCB_Z", 1.645)),
        )
        expected_shortfall = _expected_shortfall_pct(
            trade_returns,
            tail_fraction=float(
                getattr(_cfg, "PHASE2_EXPECTED_SHORTFALL_Q", 0.10)
            ),
        )
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
            for sym, grp in logs_df_tmp.groupby("Symbol", observed=False):
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
            # Approximate from accumulated counters (net_pnl is exact; wins tracked per-symbol via sym_wins)
            for sym in sym_trades:
                s_trades = sym_trades[sym]
                s_wins = sym_wins.get(sym, 0)
                s_net_pnl = sym_net_pnl.get(sym, 0.0)
                s_win_rate = (s_wins / s_trades * 100.0) if s_trades > 0 else 0.0
                per_symbol_metrics[sym] = {
                    "trade_count": s_trades,
                    "win_rate": s_win_rate,
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
            "per_symbol_metrics_available": True,
            "sum_positive_trade_pnl": float(stats.get("gross_profit_sum", 0.0)),
            "sum_negative_trade_pnl": float(stats.get("gross_loss_sum", 0.0)),
            "max_single_trade_pnl": float(stats.get("max_single_trade_pnl", 0.0)),
            "expectancy_pct_per_trade": expectancy,
            "trade_return_std_pct": trade_return_std,
            "expectancy_lcb_pct_per_trade": expectancy_lcb,
            "expected_shortfall_pct": expected_shortfall,
        }

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
        generation: int | None = None,
        is_last_gen: bool = False,
    ) -> list[dict]:
        """Evaluate a batch of rule chromosomes on CPU (vectorized signals).

        ``generation`` and ``is_last_gen`` are accepted for interface parity
        with ``GPUBacktestEngine``.  The exact CPU path has no generation-
        dependent enrichment, so the flags are intentionally ignored.
        """
        chromosomes = np.asarray(chromosomes, dtype=np.int32)
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
            is_sparse_batch,
        )

        if chromosomes.ndim == 1:
            chromosomes = chromosomes[None, :]

        n_features = len(self.feature_modes)
        if is_sparse_batch(chromosomes, n_features=n_features) and chromosomes.ndim == 2:
            chromosomes = chromosomes[None, :, :]

        sparse_batch = is_sparse_batch(chromosomes, n_features=n_features)
        if not sparse_batch:
            B, K = chromosomes.shape
        else:
            B = chromosomes.shape[0]
        if not hasattr(self, "_data_matrix"):
            from gpu_fuzzy_trader.backtest.gpu_engine import _build_data_matrix, _MODE_NUM_CLASSES
            # Chromosome positions are defined by the ordered feature list
            # supplied by Phase 1.  Preserve that order here so the exact CPU
            # admission evaluator interprets a dense chromosome identically
            # to GPUBacktestEngine and decode_chromosome.  Alphabetizing the
            # keys silently remaps genes to different columns whenever Phase
            # 1 ranks features in a non-alphabetical order.
            self._feature_names = list(self.feature_modes.keys())
            self._data_matrix = _build_data_matrix(self.df, self._feature_names, self.feature_modes)
            self._dont_cares = np.array(
                [_MODE_NUM_CLASSES[self.feature_modes[f]] for f in self._feature_names],
                dtype=np.int32
            )

        price_returns_all = self._get_trade_outcomes(tp, sl)
        results = []

        # Sparse gathering briefly materializes ``chunk × bars × slots``.
        # Keep this independently configurable from the GPU batch so a large
        # evolutionary population cannot exhaust a constrained WSL host.
        chunk_size = max(1, int(getattr(_cfg, "PHASE2_CPU_BATCH_SIZE", 16)))
        for b_start in range(0, B, chunk_size):
            b_end = min(b_start + chunk_size, B)
            chunk_chroms = chromosomes[b_start:b_end]
            bc = b_end - b_start

            if sparse_batch and self._data_matrix.shape[1] > 0:
                # Gather only the active sparse slots.  The old fallback built
                # a Python list and then accessed ``.ndim`` (so CPU fallback
                # failed for the production sparse encoding); this also avoids
                # a separate N-row comparison loop for every chromosome.
                feature_idx = chunk_chroms[:, :, 0]
                safe_idx = np.maximum(feature_idx, 0)
                gathered = np.take(
                    self._data_matrix, safe_idx, axis=1,
                ).transpose(1, 0, 2)
                matched = gathered == chunk_chroms[:, None, :, 1]
                all_signals = np.all(
                    np.where(feature_idx[:, None, :] >= 0, matched, True),
                    axis=2,
                )
            elif K > 0 and self._data_matrix.shape[1] > 0:
                # Accumulate feature comparisons in-place.  This keeps peak
                # memory at O(B*N), instead of materialising a B*N*K tensor
                # for the CPU fallback.
                all_signals = np.ones((bc, len(self.df)), dtype=bool)
                for feature_idx in range(K):
                    active = (
                        chunk_chroms[:, feature_idx]
                        != self._dont_cares[feature_idx]
                    )
                    if not np.any(active):
                        continue
                    match = (
                        self._data_matrix[:, feature_idx][None, :]
                        == chunk_chroms[:, feature_idx, None]
                    )
                    if not np.all(active):
                        match[~active] = True
                    all_signals &= match
            else:
                all_signals = np.zeros((bc, len(self.df)), dtype=bool)

            for bi in range(bc):
                signals = all_signals[bi] & self._context_mask
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

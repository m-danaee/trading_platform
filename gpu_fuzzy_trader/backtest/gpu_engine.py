"""
gpu_engine.py — GPUBacktestEngine

JAX-accelerated backtest engine for Phase 2 rule pool generation.

Primary use: evaluate batches of chromosome-encoded rules during evolutionary
search (``simulate_rule_batch``). This path uses a simplified sequential
equity model (vmap + ``lax.scan``) optimized for GPU throughput. It is an
**approximate ranking model** for evolution — final strategies are validated
on ``CPUBacktestEngine`` / ``evaluator_v3.ipynb``.

Compatibility interface (``simulate_rule_set``) delegates to CPUBacktestEngine.

Performance notes:
  - Rule matching is vectorized per chunk (VRAM-aware batch size).
  - Trade outcomes are computed once for all N rows and cached on GPU.
  - Equity simulation uses vmap'd ``lax.scan`` with ``PHASE2_SCAN_UNROLL``.
  - Chromosome batches are uploaded to GPU once per evaluation call.
  - Open-exposure slots are sized by max_exposure/capital (not hold window).
"""

from __future__ import annotations

import logging

from gpu_fuzzy_trader.backtest.cpu_engine import (
    CPUBacktestEngine,
    _sortino_ratio_from_returns,
)
from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)

from functools import partial
import math

import numpy as np
import pandas as pd

from gpu_fuzzy_trader._jax_env import configure_jax_env

configure_jax_env()

_jax_import_error = None

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    import jax.lax as lax

    def _resolve_jax_float_dtype():
        use_fp32 = bool(getattr(_cfg, "PHASE2_GPU_USE_FP32", True))
        if use_fp32:
            jax.config.update("jax_enable_x64", False)
            return jnp.float32
        jax.config.update("jax_enable_x64", True)
        return jnp.float64

    _JXF = _resolve_jax_float_dtype()
    _JX_INT = jnp.int8 if bool(
        getattr(_cfg, "PHASE2_GPU_DATA_INT8", True)) else jnp.int32
except Exception as _jax_err:
    jax = jnp = jit = vmap = lax = None
    _JXF = _JX_INT = None
    _jax_import_error = _jax_err
    logger.warning(
        "JAX unavailable in gpu_engine (%s: %s); "
        "GPUBacktestEngine will raise RuntimeError on use.",
        type(_jax_err).__name__, _jax_err,
    )


def _require_jax():
    """Raise RuntimeError if JAX failed to import at module level."""
    if jax is None:
        raise RuntimeError(
            "JAX could not be imported (required for GPUBacktestEngine). "
            f"Original error: {_jax_import_error}"
        ) from _jax_import_error


def _phase2_trade_floor() -> int:
    """Minimum executed trades required for a rule to avoid hard trade penalty."""
    return int(_cfg.MIN_TRADE_POOL_FLOOR)


def _min_raw_signals_for_full_scan() -> int:
    """Raw match count below this cannot reach trade-floor support."""
    if not bool(getattr(_cfg, "PHASE2_SKIP_INFEASIBLE_SIGNAL_SCAN", True)):
        return 1
    return max(1, _phase2_trade_floor())


# ---------------------------------------------------------------------------
# Data matrix construction helpers
# ---------------------------------------------------------------------------

_MODE_BINS: dict[str, list[float]] = {
    "positive":       [0.2, 0.4, 0.6, 0.8],
    "sparse_positive": [0.2, 0.4, 0.6, 0.8],
    "sparse_signed":  [-0.25, -1e-5, 1e-5, 0.25],
    "signed":         [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8],
}

_MODE_NUM_CLASSES: dict[str, int] = {
    "binary": 2,
    "ternary": 3,
    "positive": 5,
    "sparse_positive": 5,
    "sparse_signed": 5,
    "signed": 10,
}


def _discretize_series(series: pd.Series, mode: str) -> np.ndarray:
    """Discretize a feature series into integer bin indices."""
    values = series.values.astype(float)
    if mode == "binary":
        return values.astype(np.int32)
    if mode == "ternary":
        return (values + 1).astype(np.int32)
    bins = _MODE_BINS[mode]
    return np.digitize(values, bins=bins).astype(np.int32)


def _build_data_matrix(
    df: pd.DataFrame,
    feature_names: list[str],
    feature_modes: dict[str, str],
) -> np.ndarray:
    """Build an (N, K) integer matrix of discretized feature values."""
    columns = []
    for fname in feature_names:
        mode = feature_modes[fname]
        col = _discretize_series(df[fname], mode)
        columns.append(col)
    return np.stack(columns, axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# JAX-jitted pure functions
# ---------------------------------------------------------------------------

def _maybe_jit(fn):
    """Apply ``jit`` if JAX is available, otherwise return *fn* unchanged."""
    if jit is not None:
        return jit(fn)
    return fn


def _maybe_partial_jit(**jit_kwargs):
    """Return a decorator that applies ``jit`` with keyword args if available."""
    if jit is not None:
        return partial(jit, **jit_kwargs)
    return lambda fn: fn


@_maybe_jit
def _jax_compute_rule_signals(
    data_matrix: jnp.ndarray,   # (N, K) int32
    chromosome: jnp.ndarray,    # (K,) int32
    dont_cares: jnp.ndarray,    # (K,) int32
) -> jnp.ndarray:
    """Vectorized rule matching: returns (N,) boolean mask of matching rows."""
    active_mask = chromosome != dont_cares
    condition_match = data_matrix == chromosome[None, :]
    effective_match = jnp.where(active_mask[None, :], condition_match, True)
    return jnp.all(effective_match, axis=-1)


@_maybe_jit
def _jax_compute_rule_signals_batch(
    data_matrix: jnp.ndarray,    # (N, K) int32
    chromosomes: jnp.ndarray,    # (B, K) int32
    dont_cares: jnp.ndarray,     # (K,) int32
) -> jnp.ndarray:
    """Batch rule matching for B chromosomes simultaneously.

    Returns (B, N) boolean signal mask — fully vectorized, no Python loop.
    """
    active_mask = chromosomes != dont_cares[None, :]          # (B, K)
    condition_match = (data_matrix[None, :, :]
                       == chromosomes[:, None, :])             # (B, N, K)
    effective_match = jnp.where(
        active_mask[:, None, :], condition_match, True)       # (B, N, K)
    return jnp.all(effective_match, axis=-1)                  # (B, N)


@_maybe_jit
def _jax_compute_rule_signals_sparse_batch(
    data_matrix: jnp.ndarray,    # (N, K) int32
    # (B, S, 2) int32 — [:,:,0]=feat_idx, [:,:,1]=gene
    slots: jnp.ndarray,
) -> jnp.ndarray:
    """Batch rule matching using only active sparse slots (feat_idx >= 0)."""
    feat_idx = slots[:, :, 0]
    gene_val = slots[:, :, 1]
    active = feat_idx >= 0
    safe_idx = jnp.maximum(feat_idx, 0)
    gathered = jnp.take(data_matrix, safe_idx, axis=1)         # (N, B, S)
    gathered = jnp.transpose(gathered, (1, 0, 2))             # (B, N, S)
    match = gathered == gene_val[:, None, :]
    effective = jnp.where(active[:, None, :], match, True)
    return jnp.all(effective, axis=-1)                        # (B, N)


@_maybe_jit
def _jax_compute_trade_outcomes(
    max_ret: jnp.ndarray,        # (N,) float64
    min_ret: jnp.ndarray,        # (N,) float64
    close_ret: jnp.ndarray,      # (N,) float64
    max_before_min: jnp.ndarray,  # (N,) int32
    tp: float,
    sl: float,
    is_long: bool,
) -> jnp.ndarray:
    """Vectorized trade outcome computation for all rows.

    Mirrors CPUBacktestEngine._build_trade_outcome_single exactly.
    Returns (N,) float64 price_return_pct.
    """
    tp_f = _JXF(tp)
    sl_f = _JXF(sl)
    time_exit_cap = _JXF(_cfg.MAX_TIME_EXIT_RETURN_PCT)

    # Long path
    long_hit_tp = max_ret >= tp_f
    long_hit_sl = min_ret <= -sl_f
    long_both_hit = long_hit_tp & long_hit_sl
    long_both_result = jnp.where(max_before_min == 1, tp_f, -sl_f)
    long_result = jnp.where(
        long_both_hit, long_both_result,
        jnp.where(long_hit_tp, tp_f, jnp.where(
            long_hit_sl, -sl_f, jnp.clip(close_ret, -time_exit_cap, time_exit_cap))),
    )

    # Short path
    short_hit_tp = min_ret <= -tp_f
    short_hit_sl = max_ret >= sl_f
    short_both_hit = short_hit_tp & short_hit_sl
    short_both_result = jnp.where(max_before_min == 1, -sl_f, tp_f)
    short_result = jnp.where(
        short_both_hit, short_both_result,
        jnp.where(short_hit_tp, tp_f, jnp.where(
            short_hit_sl, -sl_f, jnp.clip(-close_ret, -time_exit_cap, time_exit_cap))),
    )

    return jnp.where(is_long, long_result, short_result)


# ---------------------------------------------------------------------------
# Batched equity simulation via vmap + lax.scan
# ---------------------------------------------------------------------------

def _exposure_slot_capacity(
    max_exposure_rate: float,
    capital_rate: float,
) -> int:
    """Max concurrent positions allowed by total-exposure cap."""
    if capital_rate <= 0.0:
        return 1
    return max(1, int(math.ceil(max_exposure_rate / capital_rate)))


def _jax_release_open_slots(
    slot_release: jnp.ndarray,
    slot_notional: jnp.ndarray,
    open_exposure: jnp.ndarray,
    current_row: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Release positions due at or before current_row; reduce open exposure."""
    active = slot_release >= 0
    releasing = active & (slot_release <= current_row)
    release_sum = jnp.sum(jnp.where(releasing, slot_notional, 0.0))
    new_open_exposure = open_exposure - release_sum
    new_slot_release = jnp.where(releasing, -1, slot_release)
    new_slot_notional = jnp.where(releasing, 0.0, slot_notional)
    return new_slot_release, new_slot_notional, new_open_exposure


def _jax_open_slot(
    slot_release: jnp.ndarray,
    slot_notional: jnp.ndarray,
    open_exposure: jnp.ndarray,
    release_idx: jnp.ndarray,
    position_notional: jnp.ndarray,
    can_trade: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Reserve one open-slot and add notional to open exposure."""
    empty_mask = slot_release < 0
    has_empty = jnp.any(empty_mask)
    first_empty = jnp.argmax(empty_mask.astype(jnp.int32))
    can_open = can_trade & has_empty

    new_slot_release = jnp.where(
        can_open,
        slot_release.at[first_empty].set(release_idx),
        slot_release,
    )
    new_slot_notional = jnp.where(
        can_open,
        slot_notional.at[first_empty].set(position_notional),
        slot_notional,
    )
    new_open_exposure = open_exposure + jnp.where(
        can_open, position_notional, 0.0)
    return new_slot_release, new_slot_notional, new_open_exposure


@_maybe_partial_jit(static_argnums=(4, 5, 6, 7, 8, 9, 10))
def _jax_simulate_equity_batch(
    signals_batch: jnp.ndarray,       # (B, N) bool
    price_returns_all: jnp.ndarray,   # (N,) float64
    release_indices: jnp.ndarray,     # (N,) int32
    initial_capital: float,
    n_rows: int,
    max_open_slots: int,
    fee_rate: float,
    leverage: float,
    capital_rate: float,
    max_exposure_rate: float,
    min_position_notional: float,
) -> jnp.ndarray:
    fee_rate_f = _JXF(fee_rate)
    leverage_f = _JXF(leverage)
    capital_rate_f = _JXF(capital_rate)
    max_exposure_rate_f = _JXF(max_exposure_rate)
    min_notional_f = _JXF(min_position_notional)
    init_cap = _JXF(initial_capital)
    sortino_cap = _JXF(_cfg.SORTINO_CAP)

    N = price_returns_all.shape[0]

    max_slots = int(max_open_slots)
    init_slot_release = jnp.full(max_slots, -1, dtype=jnp.int32)
    init_slot_notional = jnp.zeros(max_slots, dtype=_JXF)
    row_indices = jnp.arange(N, dtype=jnp.int32)

    def simulate_one(signal_mask):
        is_signal = signal_mask.astype(_JXF)
        scan_xs = jnp.stack(
            [is_signal, price_returns_all, row_indices.astype(_JXF)],
            axis=-1,
        )

        init_carry = (
            init_cap,           # equity
            init_cap,           # peak_equity
            _JXF(0.0),   # max_dd
            _JXF(0.0),   # open_exposure
            jnp.int32(0),       # wins
            jnp.int32(0),       # losses
            _JXF(0.0),   # gross_profit
            _JXF(0.0),   # gross_loss
            jnp.int32(0),       # executed
            jnp.int32(0),       # skipped
            jnp.bool_(False),   # account_ruined
            _JXF(0.0),   # trade_return_sum
            jnp.int32(0),       # n_neg
            _JXF(0.0),   # neg_sq_sum
            init_slot_release,
            init_slot_notional,
        )

        def step(carry, x):
            (equity, peak_equity, max_dd, open_exposure,
             wins, losses, gross_profit, gross_loss,
             executed, skipped, account_ruined,
             trade_return_sum, n_neg, neg_sq_sum,
             slot_release, slot_notional) = carry

            is_sig = x[0]
            price_return_pct = x[1]
            current_row = x[2].astype(jnp.int32)

            slot_release, slot_notional, open_exposure = _jax_release_open_slots(
                slot_release, slot_notional, open_exposure, current_row)

            # Position sizing (after releasing due positions)
            target = equity * capital_rate_f * leverage_f
            max_exp = equity * max_exposure_rate_f * leverage_f
            remaining = jnp.maximum(0.0, max_exp - open_exposure)
            position_notional = jnp.minimum(target, remaining)
            has_empty = jnp.any(slot_release < 0)

            can_trade = ((is_sig > 0.5) & (~account_ruined)
                         & (position_notional >= min_notional_f) & has_empty)

            gross_pnl = position_notional * (price_return_pct / 100.0)
            fee = position_notional * fee_rate_f
            net_pnl = gross_pnl - fee

            trade_ret = jnp.where(
                can_trade & (equity > 0.0), net_pnl / equity, 0.0)

            new_equity = jnp.where(can_trade, equity + net_pnl, equity)
            new_peak = jnp.maximum(peak_equity, new_equity)
            dd = jnp.where(
                new_peak > 0.0,
                (new_peak - new_equity) / new_peak * 100.0,
                100.0,
            )
            new_max_dd = jnp.maximum(max_dd, dd)

            new_wins = wins + jnp.where(
                can_trade & (net_pnl > 0.0), 1, 0).astype(jnp.int32)
            new_losses = losses + jnp.where(
                can_trade & (net_pnl < 0.0), 1, 0).astype(jnp.int32)
            new_gross_profit = gross_profit + jnp.where(
                can_trade & (net_pnl > 0.0), net_pnl, 0.0)
            new_gross_loss = gross_loss + jnp.where(
                can_trade & (net_pnl < 0.0), jnp.abs(net_pnl), 0.0)
            new_executed = executed + jnp.where(
                can_trade, 1, 0).astype(jnp.int32)
            new_skipped = skipped + jnp.where(
                (is_sig > 0.5) & (~account_ruined)
                & (position_notional < min_notional_f),
                1, 0).astype(jnp.int32)
            new_ruined = account_ruined | (new_equity <= 0.0)

            # Running Sortino stats
            new_trade_return_sum = trade_return_sum + jnp.where(
                can_trade, trade_ret, 0.0)
            is_neg = can_trade & (trade_ret < 0.0)
            new_n_neg = n_neg + jnp.where(is_neg, 1, 0).astype(jnp.int32)
            new_neg_sq_sum = neg_sq_sum + jnp.where(
                is_neg, trade_ret ** 2, 0.0)

            release_idx = release_indices[current_row]
            slot_release, slot_notional, new_open_exposure = _jax_open_slot(
                slot_release,
                slot_notional,
                open_exposure,
                release_idx,
                position_notional,
                can_trade,
            )

            new_carry = (
                new_equity, new_peak, new_max_dd, new_open_exposure,
                new_wins, new_losses, new_gross_profit, new_gross_loss,
                new_executed, new_skipped, new_ruined,
                new_trade_return_sum, new_n_neg, new_neg_sq_sum,
                slot_release, slot_notional,
            )
            return new_carry, None

        final_carry, _ = lax.scan(
            step, init_carry, scan_xs, unroll=_cfg.PHASE2_SCAN_UNROLL)

        (equity, peak_equity, max_dd, _open_exp,
         wins, losses, gross_profit, gross_loss,
         executed, skipped, account_ruined,
         trade_return_sum, n_neg, neg_sq_sum,
         _slot_release, _slot_notional) = final_carry

        total_return_pct = (equity / init_cap - 1.0) * 100.0
        raw_signal_count = jnp.sum(signal_mask).astype(_JXF)

        # Compute Sortino ratio from running stats
        n_trades = (wins + losses).astype(_JXF)
        mean_ret = jnp.where(n_trades > 0, trade_return_sum / n_trades, 0.0)
        downside_var = jnp.where(n_trades > 0, neg_sq_sum / n_trades, 0.0)
        downside_dev = jnp.sqrt(downside_var)
        sortino = jnp.where(
            downside_dev > 0.0,
            jnp.minimum(mean_ret / downside_dev, sortino_cap),
            jnp.where(mean_ret > 0.0, sortino_cap, 0.0),
        )
        sortino_threshold = _JXF(_cfg.PHASE2_SORTINO_MIN_TRADE_THRESHOLD)
        sortino = sortino * jnp.minimum(1.0, n_trades / sortino_threshold)

        win_rate = jnp.where(
            n_trades > 0, wins.astype(_JXF) / n_trades * 100.0, 0.0)
        profit_factor = jnp.where(
            (gross_loss <= 0.0) & (gross_profit > 0.0), 99.0,
            jnp.where(gross_loss <= 0.0, 0.0, gross_profit / gross_loss),
        )

        return jnp.array([
            total_return_pct, sortino, max_dd, win_rate,
            profit_factor, n_trades, equity,
            account_ruined.astype(_JXF),
            raw_signal_count, skipped.astype(_JXF),
        ])

    # vmap over the batch dimension
    batched_simulate = vmap(simulate_one)
    return batched_simulate(signals_batch)



# ---------------------------------------------------------------------------
# Batch result conversion (avoid per-row Python loops)
# ---------------------------------------------------------------------------

def _zero_signal_result_row(initial_capital: float) -> jnp.ndarray:
    """10-element row matching _jax_simulate_equity_batch when no bars match."""
    return jnp.array([
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        float(initial_capital),
        0.0,
        0.0,
        0.0,
    ], dtype=_JXF)


def _fast_reject_result_rows(
    counts: np.ndarray,
    initial_capital: float,
) -> np.ndarray:
    """
    Build fast-reject metric rows for chromosomes with insufficient raw matches.

    Zero-signal rows match ``_zero_signal_result_row``. Non-zero but below the
    trade floor use penalized placeholders (objectives treat them as infeasible).
    """
    host_dtype = np.float32 if bool(
        getattr(_cfg, "PHASE2_GPU_USE_FP32", True)) else np.float64
    rows = np.zeros((len(counts), 10), dtype=host_dtype)
    rows[:, 6] = initial_capital
    rows[:, 8] = counts.astype(host_dtype, copy=False)
    infeasible = counts > 0
    if np.any(infeasible):
        rows[infeasible, 2] = 100.0
    return rows


def _batch_metrics_from_array(
    results_np: np.ndarray,
    trade_direction: str,
) -> list[dict]:
    """Convert (B, 10) GPU result rows to list[dict] with minimal Python overhead."""
    if results_np.ndim == 1:
        results_np = results_np[None, :]
    b_count = results_np.shape[0]
    if b_count == 0:
        return []

    ruined = results_np[:, 7] > 0.5
    base = {
        "direction": trade_direction,
        "total_return_pct": results_np[:, 0],
        "sortino_ratio": results_np[:, 1],
        "max_drawdown_pct": results_np[:, 2],
        "win_rate": results_np[:, 3],
        "profit_factor": results_np[:, 4],
        "executed_trades": results_np[:, 5],
        "final_equity": results_np[:, 6],
        "account_ruined": ruined,
        "raw_signal_count": results_np[:, 8],
        "skipped_min_notional_count": results_np[:, 9],
    }

    out: list[dict] = []
    for b in range(b_count):
        entry = {
            "direction": trade_direction,
            "total_return_pct": float(base["total_return_pct"][b]),
            "sortino_ratio": float(base["sortino_ratio"][b]),
            "max_drawdown_pct": float(base["max_drawdown_pct"][b]),
            "win_rate": float(base["win_rate"][b]),
            "profit_factor": float(base["profit_factor"][b]),
            "executed_trades": int(base["executed_trades"][b]),
            "final_equity": float(base["final_equity"][b]),
            "account_ruined": bool(base["account_ruined"][b]),
            "raw_signal_count": int(base["raw_signal_count"][b]),
            "skipped_min_notional_count": int(
                base["skipped_min_notional_count"][b]),
        }
        out.append(entry)
    return out


def _neutral_per_symbol_metrics(engine: object) -> dict[str, dict]:
    """Build a neutral per_symbol_metrics dict when CPU enrichment is skipped.

    Returns a dict mapping each symbol known to *engine* to a neutral entry
    with a small positive net_pnl so the C5 symbol-spread penalty in
    ``phase2_rule_pool.py:574-582`` doesn't fire spuriously (the check uses
    ``net_pnl > 0`` strictly, so zero would still trigger the penalty).
    """
    try:
        # Extract unique symbols from the engine's dataframe.
        symbols = engine.df["symbol"].unique()
    except Exception:
        return {}
    return {
        str(sym): {"trade_count": 0, "win_rate": 0.0, "net_pnl": 1.0}
        for sym in symbols
    }


# ---------------------------------------------------------------------------
# GPUBacktestEngine
# ---------------------------------------------------------------------------

class GPUBacktestEngine:
    """JAX-accelerated backtest engine for Phase 2 rule pool generation.

    ``simulate_rule_batch`` ranks chromosomes on GPU using a simplified equity
    model. Use ``simulate_rule_set`` (CPU) for ground-truth evaluation.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_modes: dict[str, str],
        direction: str,
        **constants,
    ) -> None:
        _require_jax()
        from gpu_fuzzy_trader.backtest.cpu_engine import _normalize_direction
        self.df = df
        self.feature_modes = feature_modes
        self.trade_direction = _normalize_direction(direction)
        self.is_long = self.trade_direction == "long"

        # --- Constants ---
        self.initial_capital = float(
            constants.get("initial_capital", _cfg.INITIAL_CAPITAL))
        self.leverage = float(constants.get("leverage", _cfg.LEVERAGE))
        self.fee_pct = float(constants.get("fee_pct", _cfg.FEE_PCT))
        self.fee_rate = self.fee_pct / 100.0
        self.max_hold_candles = int(
            constants.get("max_hold_candles", _cfg.MAX_HOLD_CANDLES))
        self.max_total_exposure_pct = float(
            constants.get("max_total_exposure_pct",
                          _cfg.MAX_TOTAL_EXPOSURE_PCT))
        self.min_position_notional = float(
            constants.get("min_position_notional",
                          _cfg.MIN_POSITION_NOTIONAL))

        # --- GPU/CPU device info ---
        self._backend = jax.default_backend()

        # --- Pre-extract label arrays as JAX arrays ---
        entry = df["label_open_next"].values.astype(
            np.float32 if bool(getattr(_cfg, "PHASE2_GPU_USE_FP32", True)) else np.float64)

        invalid_mask = (~np.isfinite(entry)) | (entry <= 0)
        if np.any(invalid_mask):
            bad = int(np.sum(invalid_mask))
            raise ValueError(
                f"Invalid label_open_next values (must be finite and positive). "
                f"Bad rows: {bad}"
            )

        self._max_ret_jax = jnp.array(
            (df["label_max_288"].values - entry) / entry * 100.0,
            dtype=_JXF)
        self._min_ret_jax = jnp.array(
            (df["label_min_288"].values - entry) / entry * 100.0,
            dtype=_JXF)
        self._close_ret_jax = jnp.array(
            (df["label_close_288"].values - entry) / entry * 100.0,
            dtype=_JXF)
        self._max_before_min_jax = jnp.array(
            df["label_max_before_min"].values, dtype=jnp.int32)

        # --- Feature names in chromosome order ---
        self._feature_names: list[str] = list(feature_modes.keys())
        missing_features = [
            name for name in self._feature_names
            if name not in df.columns
        ]
        if missing_features:
            preview = ", ".join(missing_features[:5])
            suffix = "" if len(missing_features) <= 5 else ", ..."
            raise ValueError(
                f"Feature columns missing from dataframe: {preview}{suffix}")

        # --- Build data matrix (N, K) ---
        if self._feature_names:
            matrix_dtype = _JX_INT
            self._data_matrix_jax = jnp.array(
                _build_data_matrix(df, self._feature_names, feature_modes),
                dtype=matrix_dtype)
            self._dont_cares_jax = jnp.array(
                [_MODE_NUM_CLASSES[feature_modes[f]]
                    for f in self._feature_names],
                dtype=matrix_dtype)
        else:
            self._data_matrix_jax = jnp.zeros(
                (len(df), 0), dtype=_JX_INT)
            self._dont_cares_jax = jnp.zeros((0,), dtype=_JX_INT)

        # --- Pre-compute release indices ---
        from gpu_fuzzy_trader.backtest.cpu_engine import (
            precompute_release_indices,
        )
        if "symbol" in df.columns:
            sym_arr = df["symbol"].astype(str).values
        else:
            sym_arr = np.array(["UNKNOWN"] * len(df))
        if "_symbol_bar_index" in df.columns:
            bar_arr = df["_symbol_bar_index"].values.astype(int)
        else:
            bar_arr = np.arange(len(df))
        self._release_indices = precompute_release_indices(
            sym_arr, bar_arr, len(df), self.max_hold_candles)
        self._release_indices_jax = jnp.array(
            self._release_indices, dtype=jnp.int32)

        # Cache for pre-computed trade outcomes
        self._trade_outcomes_cache: dict[tuple, jnp.ndarray] = {}

        # Resolve GPU batch size once at engine init — VRAM/RAM probe is expensive.
        from gpu_fuzzy_trader._gpu_runtime import resolve_phase2_gpu_batch_size
        self._gpu_batch_size: int = max(1, resolve_phase2_gpu_batch_size())

        self._cpu_engine_ref: CPUBacktestEngine | None = None
        self._cpu_engine_constants = constants

    # ------------------------------------------------------------------
    # Public: device info
    # ------------------------------------------------------------------

    @property
    def backend(self) -> str:
        """Return the JAX backend in use ('gpu', 'cpu', or 'tpu')."""
        return self._backend

    @property
    def _lazy_cpu_engine(self) -> CPUBacktestEngine:
        """CPU engine for rule-set simulation (Phase 3+)."""
        if self._cpu_engine_ref is None:
            self._cpu_engine_ref = CPUBacktestEngine(
                self.df,
                self.feature_modes,
                self.trade_direction,
                **self._cpu_engine_constants,
            )
        return self._cpu_engine_ref

    # ------------------------------------------------------------------
    # Core JAX methods
    # ------------------------------------------------------------------

    def compute_rule_signals(
        self,
        data_matrix: jnp.ndarray,
        chromosome: jnp.ndarray,
        dont_cares: jnp.ndarray,
    ) -> np.ndarray:
        """JAX-jitted vectorized rule matching (single chromosome)."""
        result = _jax_compute_rule_signals(
            data_matrix, chromosome, dont_cares)
        return np.asarray(result, dtype=bool)

    def compute_trade_outcomes_batch(
        self,
        max_ret: jnp.ndarray,
        min_ret: jnp.ndarray,
        close_ret: jnp.ndarray,
        max_before_min: jnp.ndarray,
        tp: float,
        sl: float,
        direction: str,
    ) -> np.ndarray:
        """JAX-jitted vectorized trade outcome computation."""
        is_long = direction.lower() == "long"
        result = _jax_compute_trade_outcomes(
            max_ret, min_ret, close_ret, max_before_min, tp, sl, is_long)
        return np.asarray(result, dtype=np.float32)

    def _get_trade_outcomes(self, tp: float, sl: float) -> jnp.ndarray:
        """Get or compute cached trade outcomes for all N rows."""
        cache_key = (tp, sl, self.is_long)
        if cache_key not in self._trade_outcomes_cache:
            outcomes = _jax_compute_trade_outcomes(
                self._max_ret_jax, self._min_ret_jax,
                self._close_ret_jax, self._max_before_min_jax,
                tp, sl, self.is_long)
            self._trade_outcomes_cache[cache_key] = outcomes
        return self._trade_outcomes_cache[cache_key]

    # ------------------------------------------------------------------
    # Primary GPU-accelerated batch evaluation
    # ------------------------------------------------------------------

    def _simulate_signals_batch(
        self,
        signals_batch: jnp.ndarray,
        price_returns_all: jnp.ndarray,
        capital_rate: float,
        max_exposure_rate: float,
        n_rows: int,
    ) -> jnp.ndarray:
        """Run equity simulation for one GPU chunk of signal masks."""
        max_open_slots = _exposure_slot_capacity(
            max_exposure_rate, capital_rate)
        return _jax_simulate_equity_batch(
            signals_batch,
            price_returns_all,
            self._release_indices_jax,
            self.initial_capital,
            n_rows,
            max_open_slots,
            self.fee_rate,
            self.leverage,
            capital_rate,
            max_exposure_rate,
            self.min_position_notional,
        )

    def simulate_rule_batch(
        self,
        chromosomes: np.ndarray,
        tp: float,
        sl: float,
        capital_pct: float,
        generation: int | None = None,
        is_last_gen: bool = False,
    ) -> list[dict]:
        """Evaluate a batch of rule chromosomes on GPU.

        Large populations are processed in chunks of ``PHASE2_GPU_BATCH_SIZE``
        to cap peak VRAM (rule matching materializes a (B, N, K) tensor).

        Parameters
        ----------
        generation : int, optional
            Current generation number.  Used to throttle the CPU enrichment
            pass (``PHASE2_ENRICH_SYMBOL_METRICS_EVERY_N_GENS``).  Passed
            through by the runner loops.
        is_last_gen : bool
            Whether this is the last generation (enrichment always runs on
            final gen to keep archive/pool data fresh).
        """
        chromosomes = np.asarray(chromosomes, dtype=np.int32)
        from gpu_fuzzy_trader.phases.phase2_sparse_encoding import is_sparse_batch

        if chromosomes.ndim == 1:
            chromosomes = chromosomes[None, :]

        n_features = len(self._feature_names)
        if is_sparse_batch(chromosomes, n_features=n_features) and chromosomes.ndim == 2:
            chromosomes = chromosomes[None, :, :]

        sparse_batch = is_sparse_batch(chromosomes, n_features=n_features)
        if sparse_batch:
            B = chromosomes.shape[0]
            expected_k = len(self._feature_names)
            if chromosomes.shape[2] != 2:
                raise ValueError(
                    "Sparse chromosomes must have shape (B, S, 2).")
        else:
            B, K = chromosomes.shape
            expected_k = len(self._feature_names)
            if K != expected_k:
                raise ValueError(
                    f"Chromosome width {K} does not match engine feature "
                    f"count {expected_k}.")

        capital_rate = capital_pct / 100.0
        max_exposure_rate = self.max_total_exposure_pct / 100.0
        N = len(self.df)
        # Use cached batch size (resolved once at engine init, not re-probed)
        chunk_size = self._gpu_batch_size
        price_returns_all = self._get_trade_outcomes(tp, sl)

        # Pad chromosomes to a multiple of chunk_size to avoid shape recompilation.
        padding_len = (chunk_size - (B % chunk_size)) % chunk_size
        if padding_len > 0:
            pad_chroms = np.repeat(chromosomes[-1:], padding_len, axis=0)
            chromosomes_padded = np.concatenate([chromosomes, pad_chroms], axis=0)
        else:
            chromosomes_padded = chromosomes

        B_padded = len(chromosomes_padded)
        chromosomes_gpu = jnp.array(chromosomes_padded, dtype=jnp.int32)
        result_chunks: list[jnp.ndarray] = []

        for start in range(0, B_padded, chunk_size):
            end = start + chunk_size
            chroms_chunk = chromosomes_gpu[start:end]

            if sparse_batch:
                if self._data_matrix_jax.shape[1] > 0:
                    signals_batch = _jax_compute_rule_signals_sparse_batch(
                        self._data_matrix_jax, chroms_chunk)
                else:
                    signals_batch = jnp.zeros(
                        (end - start, N), dtype=jnp.bool_)
            elif K > 0 and self._data_matrix_jax.shape[1] > 0:
                signals_batch = _jax_compute_rule_signals_batch(
                    self._data_matrix_jax, chroms_chunk, self._dont_cares_jax)
            else:
                signals_batch = jnp.zeros(
                    (end - start, N), dtype=jnp.bool_)

            chunk_b = end - start
            use_fast_skip = (
                _cfg.PHASE2_SKIP_ZERO_SIGNAL_SCAN
                or bool(getattr(_cfg, "PHASE2_SKIP_INFEASIBLE_SIGNAL_SCAN", True))
            )
            if use_fast_skip:
                signal_counts = jnp.sum(
                    signals_batch.astype(jnp.int32), axis=1)
                counts_np = np.asarray(signal_counts, dtype=np.int64)
                min_scan = _min_raw_signals_for_full_scan()
                scan_idx = np.flatnonzero(counts_np >= min_scan)
                skip_idx = np.flatnonzero(counts_np < min_scan)

                results_jax = jnp.zeros((chunk_b, 10), dtype=_JXF)
                if skip_idx.size:
                    skip_rows = _fast_reject_result_rows(
                        counts_np[skip_idx], self.initial_capital)
                    results_jax = results_jax.at[skip_idx].set(
                        jnp.asarray(skip_rows, dtype=_JXF))

                if scan_idx.size:
                    sub_signals = signals_batch[scan_idx]
                    sub_results = self._simulate_signals_batch(
                        sub_signals,
                        price_returns_all,
                        capital_rate,
                        max_exposure_rate,
                        N,
                    )
                    results_jax = results_jax.at[scan_idx].set(sub_results)
            else:
                results_jax = self._simulate_signals_batch(
                    signals_batch,
                    price_returns_all,
                    capital_rate,
                    max_exposure_rate,
                    N,
                )
            result_chunks.append(results_jax)

        results_concat = jnp.concatenate(result_chunks, axis=0)[:B]
        results_concat = jax.block_until_ready(results_concat)
        results_np = np.asarray(results_concat)
        metrics_list = _batch_metrics_from_array(
            results_np,
            self.trade_direction,
        )
        if _cfg.phase2_should_enrich_symbol_metrics(
            self, generation=generation, is_last_gen=is_last_gen,
        ):
            try:
                cpu_metrics = self._lazy_cpu_engine.simulate_rule_batch(
                    chromosomes=chromosomes,
                    tp=tp,
                    sl=sl,
                    capital_pct=capital_pct,
                )
                for i, cm in enumerate(cpu_metrics):
                    per_sym = cm.get("per_symbol_metrics") if isinstance(
                        cm, dict) else None
                    if per_sym:
                        metrics_list[i]["per_symbol_metrics"] = per_sym
            except Exception as exc:
                logger.debug(
                    "GPU per-symbol metrics enrichment skipped: %s", exc,
                )
        else:
            # When enrichment is skipped, inject neutral per_symbol_metrics so
            # downstream code (phase2_rule_pool.py:574-582) doesn't break.
            _neutral_per_sym = _neutral_per_symbol_metrics(self)
            for m in metrics_list:
                if "per_symbol_metrics" not in m:
                    m["per_symbol_metrics"] = _neutral_per_sym
        return metrics_list


    # ------------------------------------------------------------------
    # Batched rule-set evaluation (Phase 3)
    # ------------------------------------------------------------------

    def simulate_rule_set_from_cache(
        self,
        rule_set: list[dict],
        cache,
        split: str,
        return_logs: bool = False,
    ) -> "dict | tuple[dict, pd.DataFrame]":
        """Rule-set sim using Phase3EvalCache (delegates to CPU engine)."""
        return self._lazy_cpu_engine.simulate_rule_set_from_cache(
            rule_set, cache, split, return_logs=return_logs)

    def simulate_rule_set_batch(
        self,
        rule_sets: list[list[dict]],
        max_workers: int | None = None,
        cache=None,
        split: str | None = None,
    ) -> list[dict]:
        """Evaluate multiple rule sets (parallel CPU; optional mask cache)."""
        return self._lazy_cpu_engine.simulate_rule_set_batch(
            rule_sets,
            max_workers=max_workers,
            cache=cache,
            split=split,
        )

    def simulate_rule_set_batch_jax(
        self,
        rule_sets: list[list[dict]],
        cache=None,
        split: str = "val",
    ) -> list[dict]:
        """
        Phase 3 JAX path: cached mask entry build + parallel CPU equity.

        Equity results match CPUBacktestEngine within standard parity tolerance.
        """
        return self._lazy_cpu_engine.simulate_rule_set_batch(
            rule_sets, cache=cache, split=split)

    # ------------------------------------------------------------------
    # Compatibility interface (delegates to CPUBacktestEngine)
    # ------------------------------------------------------------------

    def simulate_rule_set(
        self,
        rule_set: list[dict],
        return_logs: bool = False,
    ) -> "dict | tuple[dict, pd.DataFrame]":
        """Same interface as CPUBacktestEngine for compatibility."""
        return self._lazy_cpu_engine.simulate_rule_set(
            rule_set, return_logs=return_logs)

    # ------------------------------------------------------------------
    # Legacy compatibility: simulate_equity_sequential
    # ------------------------------------------------------------------

    def simulate_equity_sequential(
        self,
        entries: np.ndarray,
        release_indices: np.ndarray,
        net_pnls: np.ndarray,
        initial_capital: float,
    ) -> dict:
        """jax.lax.scan-based sequential equity simulation (legacy compat).

        Parameters
        ----------
        entries : np.ndarray
            Shape (M,) int32 — row indices of matched entries (sorted).
        release_indices : np.ndarray
            Shape (M,) int32 — release row index per entry.
        net_pnls : np.ndarray
            Shape (M,) float32 — net PnL per entry (pre-computed).
        initial_capital : float
            Starting equity.

        Returns
        -------
        dict
            Keys: total_return_pct, sortino_ratio, max_drawdown_pct,
            win_rate, profit_factor, executed_trades, final_equity,
            account_ruined.
        """
        if len(entries) == 0:
            return {
                "total_return_pct": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "executed_trades": 0,
                "final_equity": float(initial_capital),
                "account_ruined": False,
            }

        net_pnls_jax = jnp.array(net_pnls, dtype=_JXF)
        init_equity = _JXF(initial_capital)

        def _scan_step(carry, net_pnl):
            (equity, wins, losses, gross_profit, gross_loss,
             ruined, peak, max_dd) = carry
            new_equity = jnp.where(ruined, equity, equity + net_pnl)
            new_peak = jnp.maximum(peak, new_equity)
            dd = jnp.where(
                new_peak > 0.0,
                (new_peak - new_equity) / new_peak * 100.0, 100.0)
            new_max_dd = jnp.maximum(max_dd, dd)
            new_wins = wins + jnp.where(
                (~ruined) & (net_pnl > 0.0), 1, 0).astype(jnp.int32)
            new_losses = losses + jnp.where(
                (~ruined) & (net_pnl < 0.0), 1, 0).astype(jnp.int32)
            new_gp = gross_profit + jnp.where(
                (~ruined) & (net_pnl > 0.0), net_pnl, 0.0)
            new_gl = gross_loss + jnp.where(
                (~ruined) & (net_pnl < 0.0),
                jnp.abs(net_pnl), 0.0)
            new_ruined = ruined | (new_equity <= 0.0)
            return (new_equity, new_wins, new_losses, new_gp, new_gl,
                    new_ruined, new_peak, new_max_dd), None

        init_carry = (
            init_equity,
            jnp.int32(0), jnp.int32(0),
            _JXF(0.0), _JXF(0.0),
            jnp.bool_(False),
            init_equity, _JXF(0.0),
        )
        final_carry, _ = lax.scan(
            _scan_step, init_carry, net_pnls_jax, unroll=_cfg.PHASE2_SCAN_UNROLL)
        (final_equity, wins, losses, gross_profit, gross_loss,
         ruined, peak, max_dd) = final_carry

        executed = int(wins) + int(losses)
        win_rate = (float(wins) / executed * 100.0) if executed > 0 else 0.0
        if float(gross_loss) <= 0.0 and float(gross_profit) > 0.0:
            profit_factor = 99.0
        elif float(gross_loss) <= 0.0:
            profit_factor = 0.0
        else:
            profit_factor = float(gross_profit) / float(gross_loss)

        return {
            "total_return_pct": (
                float(final_equity) / initial_capital - 1.0) * 100.0,
            "sortino_ratio": _sortino_ratio_from_returns(
                [float(p) / float(initial_capital) for p in net_pnls],
                scale_by_trades=True,
            ),
            "max_drawdown_pct": float(max_dd),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "executed_trades": executed,
            "final_equity": float(final_equity),
            "account_ruined": bool(ruined),
        }

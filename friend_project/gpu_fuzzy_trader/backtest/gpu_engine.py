
from __future__ import annotations
from gpu_fuzzy_trader.backtest.cpu_engine import (
    CPUBacktestEngine,
    _sortino_ratio_from_returns,
)
from gpu_fuzzy_trader import config as _cfg

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from gpu_fuzzy_trader._jax_env import configure_jax_env

configure_jax_env()

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    import jax.lax as lax
    jax.config.update("jax_enable_x64", True)
except Exception as _jax_err:
    raise ImportError(
        "JAX is required for GPUBacktestEngine but could not be imported. "
        "Install it with: pip install jax jaxlib\n"
        f"Original error: {_jax_err}"
    ) from _jax_err



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
    """Discretize a feature series into integer bin indices matching the mode.

    Returns an int32 array of the same length as *series*.
    """
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
    """Build an (N, K) integer matrix of discretized feature values.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset.
    feature_names : list[str]
        Ordered list of feature column names (length K).
    feature_modes : dict[str, str]
        Mode for each feature column.

    Returns
    -------
    np.ndarray
        Shape (N, K), dtype int32.
    """
    columns = []
    for fname in feature_names:
        mode = feature_modes[fname]
        col = _discretize_series(df[fname], mode)
        columns.append(col)
    return np.stack(columns, axis=1).astype(np.int32)



@jit
def _jax_compute_rule_signals(
    data_matrix: jnp.ndarray,                 
    chromosome: jnp.ndarray,                
    dont_cares: jnp.ndarray,                
) -> jnp.ndarray:
    """Vectorized rule matching: returns boolean mask of matching rows.

    For each row, a row matches if ALL active conditions (gene != dont_care)
    have data_matrix[row, k] == chromosome[k].

    Parameters
    ----------
    data_matrix : jnp.ndarray
        Shape (N, K) — discretized feature values.
    chromosome : jnp.ndarray
        Shape (K,) — gene values for this rule.
    dont_cares : jnp.ndarray
        Shape (K,) — dont_care sentinel per feature.

    Returns
    -------
    jnp.ndarray
        Shape (N,) boolean mask.
    """
    active_mask = chromosome != dont_cares             
    condition_match = data_matrix == chromosome[None, :]          
    effective_match = jnp.where(
        active_mask[None, :], condition_match, True)          
    return jnp.all(effective_match, axis=-1)        


@jit
def _jax_compute_trade_outcomes(
    max_ret: jnp.ndarray,                      
    min_ret: jnp.ndarray,                      
    close_ret: jnp.ndarray,                    
    max_before_min: jnp.ndarray,              
    tp: float,
    sl: float,
    is_long: bool,
) -> jnp.ndarray:
    """Vectorized trade outcome computation for all rows.

    Mirrors CPUBacktestEngine._build_trade_outcome_single exactly.

    Returns
    -------
    jnp.ndarray
        Shape (N,) float64 — price_return_pct for each row.
    """
    tp_f = jnp.float64(tp)
    sl_f = jnp.float64(sl)

    long_hit_tp = max_ret >= tp_f
    long_hit_sl = min_ret <= -sl_f
    long_both_hit = long_hit_tp & long_hit_sl
    long_both_result = jnp.where(max_before_min == 1, tp_f, -sl_f)
    long_result = jnp.where(
        long_both_hit,
        long_both_result,
        jnp.where(long_hit_tp, tp_f, jnp.where(long_hit_sl, -sl_f, close_ret)),
    )

    short_hit_tp = min_ret <= -tp_f
    short_hit_sl = max_ret >= sl_f
    short_both_hit = short_hit_tp & short_hit_sl
    short_both_result = jnp.where(max_before_min == 1, -sl_f, tp_f)
    short_result = jnp.where(
        short_both_hit,
        short_both_result,
        jnp.where(short_hit_tp, tp_f, jnp.where(
            short_hit_sl, -sl_f, -close_ret)),
    )

    return jnp.where(is_long, long_result, short_result)



def _build_scan_fn(
    fee_rate: float,
    leverage: float,
    capital_rate: float,
    max_exposure_rate: float,
    min_position_notional: float,
):
    """Build a jax.lax.scan-compatible step function for equity simulation.

    The scan processes entries in chronological order. Each step:
      1. Releases positions whose release_index <= current entry index.
      2. Sizes the new position.
      3. Computes PnL and updates equity.

    Because JAX scan requires fixed-size arrays, we pre-sort entries and
    use a fixed-capacity open-positions buffer.

    Parameters
    ----------
    fee_rate : float
        Round-trip fee rate (e.g. 0.002 for 0.20%).
    leverage : float
        Position leverage multiplier.
    capital_rate : float
        capital_pct / 100.
    max_exposure_rate : float
        max_total_exposure_pct / 100.
    min_position_notional : float
        Minimum position size to execute a trade.

    Returns
    -------
    Callable
        A function suitable for jax.lax.scan.
    """
    fee_rate_f = jnp.float32(fee_rate)
    leverage_f = jnp.float32(leverage)
    capital_rate_f = jnp.float32(capital_rate)
    max_exposure_rate_f = jnp.float32(max_exposure_rate)
    min_notional_f = jnp.float32(min_position_notional)

    def scan_step(carry, x):
        """Process one entry.

        carry = (equity, open_exposure, wins, losses, gross_profit, gross_loss,
                 executed, skipped, account_ruined, peak_equity, max_dd)
        x = (entry_idx, release_idx, price_return_pct)
        """
        (equity, open_exposure, wins, losses, gross_profit, gross_loss,
         executed, skipped, account_ruined, peak_equity, max_dd) = carry

        entry_idx, release_idx, price_return_pct = x


        target = equity * capital_rate_f * leverage_f
        max_exp = equity * max_exposure_rate_f * leverage_f
        remaining = jnp.maximum(jnp.float32(0.0), max_exp - open_exposure)
        position_notional = jnp.minimum(target, remaining)

        can_trade = (~account_ruined) & (position_notional >= min_notional_f)

        gross_pnl = position_notional * (price_return_pct / jnp.float32(100.0))
        fee = position_notional * fee_rate_f
        net_pnl = gross_pnl - fee

        new_equity = jnp.where(can_trade, equity + net_pnl, equity)
        new_peak = jnp.maximum(peak_equity, new_equity)
        dd = jnp.where(
            new_peak > jnp.float32(0.0),
            (new_peak - new_equity) / new_peak * jnp.float32(100.0),
            jnp.float32(100.0),
        )
        new_max_dd = jnp.maximum(max_dd, dd)

        new_wins = wins + jnp.where(can_trade & (net_pnl > jnp.float32(0.0)),
                                    jnp.int32(1), jnp.int32(0))
        new_losses = losses + jnp.where(can_trade & (net_pnl < jnp.float32(0.0)),
                                        jnp.int32(1), jnp.int32(0))
        new_gross_profit = gross_profit + jnp.where(
            can_trade & (net_pnl > jnp.float32(0.0)), net_pnl, jnp.float32(0.0))
        new_gross_loss = gross_loss + jnp.where(
            can_trade & (net_pnl < jnp.float32(0.0)), jnp.abs(net_pnl), jnp.float32(0.0))

        new_executed = executed + \
            jnp.where(can_trade, jnp.int32(1), jnp.int32(0))
        new_skipped = skipped + jnp.where(
            (~account_ruined) & (position_notional < min_notional_f),
            jnp.int32(1), jnp.int32(0))

        new_ruined = account_ruined | (new_equity <= jnp.float32(0.0))

        new_carry = (
            new_equity, open_exposure, new_wins, new_losses,
            new_gross_profit, new_gross_loss,
            new_executed, new_skipped, new_ruined, new_peak, new_max_dd,
        )
        return new_carry, None

    return scan_step



class GPUBacktestEngine:
    """JAX-accelerated backtest engine for Phase 2 rule pool generation.

    Used exclusively during Phase 2 to evaluate batches of single-rule
    chromosomes with static TP/SL/capital_pct.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared dataset (already sorted, NaN-dropped, bar-indexed).
    feature_modes : dict[str, str]
        Feature mode mapping — used to build the data matrix.
    direction : str
        "long" or "short".
    **constants
        Optional overrides for backtest constants. Recognised keys:
        initial_capital, leverage, fee_pct, max_hold_candles,
        max_total_exposure_pct, min_position_notional.
        Defaults come from config.py.

    Raises
    ------
    ImportError
        If JAX cannot be imported (raised at module import time).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_modes: dict[str, str],
        direction: str,
        **constants,
    ) -> None:
        from gpu_fuzzy_trader.backtest.cpu_engine import _normalize_direction
        self.df = df
        self.feature_modes = feature_modes
        self.trade_direction = _normalize_direction(direction)
        self.is_long = self.trade_direction == "long"

        self.initial_capital = float(
            constants.get("initial_capital", _cfg.INITIAL_CAPITAL)
        )
        self.leverage = float(constants.get("leverage", _cfg.LEVERAGE))
        self.fee_pct = float(constants.get("fee_pct", _cfg.FEE_PCT))
        self.fee_rate = self.fee_pct / 100.0
        self.max_hold_candles = int(
            constants.get("max_hold_candles", _cfg.MAX_HOLD_CANDLES)
        )
        self.max_total_exposure_pct = float(
            constants.get("max_total_exposure_pct",
                          _cfg.MAX_TOTAL_EXPOSURE_PCT)
        )
        self.min_position_notional = float(
            constants.get("min_position_notional", _cfg.MIN_POSITION_NOTIONAL)
        )

        self._backend = jax.default_backend()

        entry = df["label_open_next"].values.astype(np.float64)
        self._max_ret_jax = jnp.array(
            (df["label_max_288"].values - entry) / entry * 100.0, dtype=jnp.float64
        )
        self._min_ret_jax = jnp.array(
            (df["label_min_288"].values - entry) / entry * 100.0, dtype=jnp.float64
        )
        self._close_ret_jax = jnp.array(
            (df["label_close_288"].values - entry) / entry * 100.0, dtype=jnp.float64
        )
        self._max_before_min_jax = jnp.array(
            df["label_max_before_min"].values, dtype=jnp.int32
        )

        self._feature_names: list[str] = list(feature_modes.keys())
        missing_features = [
            name for name in self._feature_names
            if name not in df.columns
        ]
        if missing_features:
            preview = ", ".join(missing_features[:5])
            suffix = "" if len(missing_features) <= 5 else ", ..."
            raise ValueError(
                "Feature columns missing from dataframe: "
                f"{preview}{suffix}"
            )

        if self._feature_names:
            self._data_matrix_jax = jnp.array(
                _build_data_matrix(df, self._feature_names, feature_modes),
                dtype=jnp.int32,
            )
            self._dont_cares_jax = jnp.array(
                [_MODE_NUM_CLASSES[feature_modes[f]]
                    for f in self._feature_names],
                dtype=jnp.int32,
            )
        else:
            self._data_matrix_jax = jnp.zeros((len(df), 0), dtype=jnp.int32)
            self._dont_cares_jax = jnp.zeros((0,), dtype=jnp.int32)

        from gpu_fuzzy_trader.backtest.cpu_engine import precompute_release_indices

        if "symbol" in df.columns:
            sym_arr = df["symbol"].astype(str).values
        else:
            sym_arr = np.array(["UNKNOWN"] * len(df))
        if "_symbol_bar_index" in df.columns:
            bar_arr = df["_symbol_bar_index"].values.astype(int)
        else:
            bar_arr = np.arange(len(df))
        self._release_indices = precompute_release_indices(
            sym_arr,
            bar_arr,
            len(df),
            self.max_hold_candles,
        )
        self._cpu_engine_ref: CPUBacktestEngine | None = None
        self._cpu_engine_constants = constants


    @property
    def backend(self) -> str:
        """Return the JAX backend in use ('gpu', 'cpu', or 'tpu')."""
        return self._backend

    @property
    def _lazy_cpu_engine(self) -> CPUBacktestEngine:
        """CPU engine for rule-set simulation (Phase 3+); created on first use."""
        if self._cpu_engine_ref is None:
            self._cpu_engine_ref = CPUBacktestEngine(
                self.df,
                self.feature_modes,
                self.trade_direction,
                **self._cpu_engine_constants,
            )
        return self._cpu_engine_ref


    def compute_rule_signals(
        self,
        data_matrix: jnp.ndarray,
        chromosome: jnp.ndarray,
        dont_cares: jnp.ndarray,
    ) -> np.ndarray:
        """JAX-jitted vectorized rule matching.

        Parameters
        ----------
        data_matrix : jnp.ndarray
            Shape (N, K) int32 — discretized feature values.
        chromosome : jnp.ndarray
            Shape (K,) int32 — gene values for this rule.
        dont_cares : jnp.ndarray
            Shape (K,) int32 — dont_care sentinel per feature.

        Returns
        -------
        np.ndarray
            Shape (N,) boolean mask of matching rows.
        """
        result = _jax_compute_rule_signals(data_matrix, chromosome, dont_cares)
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
        """JAX-jitted vectorized trade outcome computation.

        Parameters
        ----------
        max_ret, min_ret, close_ret : jnp.ndarray
            Shape (N,) float32 — percentage returns.
        max_before_min : jnp.ndarray
            Shape (N,) int32.
        tp, sl : float
            Take-profit and stop-loss percentages.
        direction : str
            "long" or "short".

        Returns
        -------
        np.ndarray
            Shape (N,) float32 — price_return_pct for each row.
        """
        is_long = direction.lower() == "long"
        result = _jax_compute_trade_outcomes(
            max_ret, min_ret, close_ret, max_before_min, tp, sl, is_long
        )
        return np.asarray(result, dtype=np.float32)

    def simulate_equity_sequential(
        self,
        entries: np.ndarray,
        release_indices: np.ndarray,
        net_pnls: np.ndarray,
        initial_capital: float,
    ) -> dict:
        """jax.lax.scan-based sequential equity simulation.

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
            Keys: total_return_pct, sortino_ratio, max_drawdown_pct, win_rate,
            profit_factor, executed_trades, final_equity, account_ruined.
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

        net_pnls_jax = jnp.array(net_pnls, dtype=jnp.float32)
        init_equity = jnp.float32(initial_capital)

        def _simple_scan_step(carry, net_pnl):
            equity, wins, losses, gross_profit, gross_loss, ruined, peak, max_dd = carry
            new_equity = jnp.where(ruined, equity, equity + net_pnl)
            new_peak = jnp.maximum(peak, new_equity)
            dd = jnp.where(
                new_peak > jnp.float32(0.0),
                (new_peak - new_equity) / new_peak * jnp.float32(100.0),
                jnp.float32(100.0),
            )
            new_max_dd = jnp.maximum(max_dd, dd)
            new_wins = wins + jnp.where(
                (~ruined) & (net_pnl > jnp.float32(0.0)), jnp.int32(1), jnp.int32(0))
            new_losses = losses + jnp.where(
                (~ruined) & (net_pnl < jnp.float32(0.0)), jnp.int32(1), jnp.int32(0))
            new_gp = gross_profit + jnp.where(
                (~ruined) & (net_pnl > jnp.float32(0.0)), net_pnl, jnp.float32(0.0))
            new_gl = gross_loss + jnp.where(
                (~ruined) & (net_pnl < jnp.float32(0.0)),
                jnp.abs(net_pnl), jnp.float32(0.0))
            new_ruined = ruined | (new_equity <= jnp.float32(0.0))
            return (new_equity, new_wins, new_losses, new_gp, new_gl,
                    new_ruined, new_peak, new_max_dd), None

        init_carry = (
            init_equity,
            jnp.int32(0), jnp.int32(0),
            jnp.float32(0.0), jnp.float32(0.0),
            jnp.bool_(False),
            init_equity, jnp.float32(0.0),
        )
        final_carry, _ = lax.scan(_simple_scan_step, init_carry, net_pnls_jax)
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
            "total_return_pct": (float(final_equity) / initial_capital - 1.0) * 100.0,
            "sortino_ratio": _sortino_ratio_from_returns(
                [float(net_pnl) / float(initial_capital)
                 for net_pnl in net_pnls]
            ),
            "max_drawdown_pct": float(max_dd),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "executed_trades": executed,
            "final_equity": float(final_equity),
            "account_ruined": bool(ruined),
        }


    def simulate_rule_batch(
        self,
        chromosomes: np.ndarray,
        tp: float,
        sl: float,
        capital_pct: float,
    ) -> list[dict]:
        """Evaluate a batch of rule chromosomes simultaneously.

        This is the primary GPU-accelerated method used during Phase 2.
        Each chromosome is a K-length integer array encoding gene values
        for each selected feature. Genes equal to the dont_care sentinel
        for their mode are treated as inactive conditions.

        Parameters
        ----------
        chromosomes : np.ndarray
            Shape (B, K) int32 — batch of B chromosomes, each of length K
            (number of selected features).
        tp : float
            Take-profit percentage (static during Phase 2).
        sl : float
            Stop-loss percentage (static during Phase 2).
        capital_pct : float
            Capital allocation percentage (static during Phase 2).

        Returns
        -------
        list[dict]
            List of B metrics dicts, one per chromosome. Each dict contains:
            direction, total_return_pct, max_drawdown_pct, win_rate,
            profit_factor, executed_trades, final_equity, account_ruined,
            raw_signal_count, skipped_min_notional_count.
        """
        chromosomes = np.asarray(chromosomes, dtype=np.int32)
        if chromosomes.ndim == 1:
            chromosomes = chromosomes[None, :]          

        B, K = chromosomes.shape
        expected_k = len(self._feature_names)
        if K != expected_k:
            raise ValueError(
                f"Chromosome width {K} does not match engine feature count "
                f"{expected_k}."
            )

        capital_rate = capital_pct / 100.0
        max_exposure_rate = self.max_total_exposure_pct / 100.0

        results = []

        for b in range(B):
            chrom_jax = jnp.array(chromosomes[b], dtype=jnp.int32)

            if K > 0 and self._data_matrix_jax.shape[1] > 0:
                dm = self._data_matrix_jax
                dc = self._dont_cares_jax
                signal_mask = _jax_compute_rule_signals(dm, chrom_jax, dc)
                signal_mask_np = np.asarray(signal_mask, dtype=bool)
            else:
                signal_mask_np = np.zeros(len(self.df), dtype=bool)

            matched_indices = np.flatnonzero(signal_mask_np)
            raw_signal_count = len(matched_indices)

            if raw_signal_count == 0:
                results.append({
                    "direction": self.trade_direction,
                    "total_return_pct": 0.0,
                    "sortino_ratio": 0.0,
                    "max_drawdown_pct": 0.0,
                    "win_rate": 0.0,
                    "profit_factor": 0.0,
                    "executed_trades": 0,
                    "final_equity": self.initial_capital,
                    "account_ruined": False,
                    "raw_signal_count": 0,
                    "skipped_min_notional_count": 0,
                })
                continue

            matched_max_ret = self._max_ret_jax[matched_indices]
            matched_min_ret = self._min_ret_jax[matched_indices]
            matched_close_ret = self._close_ret_jax[matched_indices]
            matched_mbm = self._max_before_min_jax[matched_indices]

            price_returns = _jax_compute_trade_outcomes(
                matched_max_ret, matched_min_ret, matched_close_ret,
                matched_mbm, tp, sl, self.is_long,
            )
            price_returns_np = np.asarray(price_returns, dtype=np.float64)

            equity = self.initial_capital
            peak_equity = self.initial_capital
            max_drawdown_pct = 0.0
            open_total_exposure = 0.0
            executed_trades = 0
            skipped_count = 0
            wins = 0
            losses = 0
            gross_profit = 0.0
            gross_loss = 0.0
            account_ruined = False
            trade_returns: list[float] = []

            open_positions: list[tuple[int, float, float]] = []

            for i, idx in enumerate(matched_indices):
                if account_ruined:
                    break

                release_idx = int(self._release_indices[idx])

                still_open = []
                for pos_release, pos_notional, pos_net_pnl in open_positions:
                    if pos_release <= idx:
                        equity += pos_net_pnl
                        peak_equity = max(peak_equity, equity)
                        dd = (peak_equity - equity) / peak_equity * \
                            100.0 if peak_equity > 0 else 100.0
                        max_drawdown_pct = max(max_drawdown_pct, dd)
                        open_total_exposure -= pos_notional
                        if equity <= 0.0:
                            account_ruined = True
                    else:
                        still_open.append(
                            (pos_release, pos_notional, pos_net_pnl))
                open_positions = still_open

                if account_ruined:
                    break

                target = equity * capital_rate * self.leverage
                max_exp = equity * max_exposure_rate * self.leverage
                remaining = max(0.0, max_exp - open_total_exposure)
                position_notional = min(target, remaining)

                if position_notional < self.min_position_notional:
                    skipped_count += 1
                    continue

                price_return_pct = float(price_returns_np[i])
                gross_pnl = position_notional * price_return_pct / 100.0
                fee = position_notional * self.fee_rate
                net_pnl = gross_pnl - fee
                trade_returns.append(net_pnl / equity if equity > 0.0 else 0.0)

                open_positions.append(
                    (release_idx, position_notional, net_pnl))
                open_total_exposure += position_notional
                executed_trades += 1

                if net_pnl > 0:
                    wins += 1
                    gross_profit += net_pnl
                elif net_pnl < 0:
                    losses += 1
                    gross_loss += abs(net_pnl)

            for pos_release, pos_notional, pos_net_pnl in open_positions:
                equity += pos_net_pnl
                peak_equity = max(peak_equity, equity)
                dd = (peak_equity - equity) / peak_equity * \
                    100.0 if peak_equity > 0 else 100.0
                max_drawdown_pct = max(max_drawdown_pct, dd)
                if equity <= 0.0:
                    account_ruined = True

            total_return_pct = (equity / self.initial_capital - 1.0) * 100.0
            sortino_ratio = _sortino_ratio_from_returns(trade_returns)
            win_rate = (wins / executed_trades *
                        100.0) if executed_trades > 0 else 0.0
            if gross_loss <= 0.0 and gross_profit > 0.0:
                profit_factor = 99.0
            elif gross_loss <= 0.0:
                profit_factor = 0.0
            else:
                profit_factor = gross_profit / gross_loss

            results.append({
                "direction": self.trade_direction,
                "total_return_pct": total_return_pct,
                "sortino_ratio": sortino_ratio,
                "max_drawdown_pct": max_drawdown_pct,
                "win_rate": win_rate,
                "profit_factor": profit_factor,
                "executed_trades": executed_trades,
                "final_equity": equity,
                "account_ruined": account_ruined,
                "raw_signal_count": raw_signal_count,
                "skipped_min_notional_count": skipped_count,
            })

        return results


    def simulate_rule_set_batch(
        self,
        rule_sets: list[list[dict]],
        max_workers: int | None = None,
    ) -> list[dict]:
        """Evaluate multiple rule sets; numerically identical to sequential CPU calls.

        Uses parallel CPU simulation via the embedded CPUBacktestEngine reference.
        Phase 3 enables this path when ``PHASE3_USE_GPU`` is True (parity-gated).

        Parameters
        ----------
        rule_sets : list[list[dict]]
            Each element is a rule set in CPUBacktestEngine format.
        max_workers : int, optional
            Thread pool size (default: min(32, len(rule_sets))).

        Returns
        -------
        list[dict]
            Metrics dict per rule set, including ``per_symbol_metrics``.
        """
        if not rule_sets:
            return []

        if len(rule_sets) == 1:
            return [self._lazy_cpu_engine.simulate_rule_set(rule_sets[0])]

        workers = max_workers
        if workers is None:
            workers = min(32, len(rule_sets))

        def _eval_one(rs: list[dict]) -> dict:
            return self._lazy_cpu_engine.simulate_rule_set(rs)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(_eval_one, rule_sets))


    def simulate_rule_set(
        self,
        rule_set: list[dict],
        return_logs: bool = False,
    ) -> "dict | tuple[dict, pd.DataFrame]":
        """Same interface as CPUBacktestEngine for compatibility.

        Delegates to CPUBacktestEngine since this method uses condition
        strings (threshold-based matching) rather than chromosomes.
        Used for Phase 3 compatibility and final evaluation.

        Parameters
        ----------
        rule_set : list[dict]
            Each dict: {"conditions": [...], "tp": float, "sl": float,
                        "capital_pct": float}
        return_logs : bool
            If True, also return a trade log DataFrame.

        Returns
        -------
        dict or tuple[dict, pd.DataFrame]
            Same as CPUBacktestEngine.simulate_rule_set.
        """
        return self._lazy_cpu_engine.simulate_rule_set(rule_set, return_logs=return_logs)

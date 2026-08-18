"""Hierarchical MTF Signal Composer and Asymmetric Soft-Veto Engine.

Combines hard LWC execution triggers with HWC (macro trend) and MWC (setup confirmation)
continuous direction/strength ensemble scores using asymmetric soft-veto logic. Neutral
or weakly supportive higher-timeframe signals never block entry triggers; only strong
contradictory evidence meeting evidence strength gates issues a veto.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence, Union
import numpy as np
import pandas as pd

from gpu_fuzzy_trader.mtf.diagnostics import compute_trade_retention_diagnostics

logger = logging.getLogger(__name__)

# Default asymmetric soft-veto thresholds calibrated on macro/intermediate dynamics
DEFAULT_V_HWC_LONG: float = 0.65
DEFAULT_V_HWC_SHORT: float = 0.60
DEFAULT_V_MWC_LONG: float = 0.60
DEFAULT_V_MWC_SHORT: float = 0.55
DEFAULT_MIN_EVIDENCE_STRENGTH: float = 0.15


def normalize_direction(direction: Any) -> str:
    """Normalize direction input to canonical 'long' or 'short'.

    Parameters
    ----------
    direction : Any
        Direction indicator ('long', 'short', 'buy', 'sell', 1, -1).

    Returns
    -------
    str
        'long' or 'short'.
    """
    s = str(direction).strip().lower()
    if s in ("long", "buy", "1", "+1"):
        return "long"
    elif s in ("short", "sell", "-1"):
        return "short"
    else:
        raise ValueError(f"Unrecognized trading direction: {direction}")


def compose_hierarchical_signals(
    lwc_triggers: Union[np.ndarray, pd.Series, Sequence[Any]],
    direction: str,
    hwc_direction: Union[np.ndarray, pd.Series, Sequence[float]],
    hwc_strength: Union[np.ndarray, pd.Series, Sequence[float]],
    mwc_direction: Union[np.ndarray, pd.Series, Sequence[float]],
    mwc_strength: Union[np.ndarray, pd.Series, Sequence[float]],
    v_hwc: float | None = None,
    v_mwc: float | None = None,
    min_strength_hwc: float = DEFAULT_MIN_EVIDENCE_STRENGTH,
    min_strength_mwc: float = DEFAULT_MIN_EVIDENCE_STRENGTH,
    *,
    v_hwc_long: float | None = None,
    v_hwc_short: float | None = None,
    v_mwc_long: float | None = None,
    v_mwc_short: float | None = None,
    min_evidence_strength_hwc: float | None = None,
    min_evidence_strength_mwc: float | None = None,
    output_dtype: Any = np.int8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Filter raw LWC entry triggers through staged HWC and MWC asymmetric soft vetoes.

    Composition logic:
      - LWC generates hard entry triggers.
      - HWC Soft-Veto:
          - Long: veto if hwc_direction < -V_HWC_LONG and hwc_strength >= min_strength_hwc.
          - Short: veto if hwc_direction > +V_HWC_SHORT and hwc_strength >= min_strength_hwc.
      - MWC Soft-Veto (applied incrementally to candidates surviving HWC):
          - Long: veto if mwc_direction < -V_MWC_LONG and mwc_strength >= min_strength_mwc.
          - Short: veto if mwc_direction > +V_MWC_SHORT and mwc_strength >= min_strength_mwc.

    Parameters
    ----------
    lwc_triggers : np.ndarray, pd.Series, or Sequence
        Array of raw LWC triggers (non-zero indicates active trigger).
    direction : str
        Trade direction: 'long' or 'short'.
    hwc_direction : np.ndarray, pd.Series, or Sequence
        Array of HWC continuous direction scores in [-1.0, +1.0].
    hwc_strength : np.ndarray, pd.Series, or Sequence
        Array of HWC evidence strength scores in [0.0, 1.0].
    mwc_direction : np.ndarray, pd.Series, or Sequence
        Array of MWC continuous direction scores in [-1.0, +1.0].
    mwc_strength : np.ndarray, pd.Series, or Sequence
        Array of MWC evidence strength scores in [0.0, 1.0].
    v_hwc : float or None, optional
        Direct HWC veto threshold override for the specified direction.
    v_mwc : float or None, optional
        Direct MWC veto threshold override for the specified direction.
    min_strength_hwc : float, default 0.15
        Minimum evidence strength required for HWC veto to activate.
    min_strength_mwc : float, default 0.15
        Minimum evidence strength required for MWC veto to activate.
    v_hwc_long : float or None, optional
        Explicit HWC veto threshold for long trades.
    v_hwc_short : float or None, optional
        Explicit HWC veto threshold for short trades.
    v_mwc_long : float or None, optional
        Explicit MWC veto threshold for long trades.
    v_mwc_short : float or None, optional
        Explicit MWC veto threshold for short trades.
    min_evidence_strength_hwc : float or None, optional
        Keyword alias for min_strength_hwc.
    min_evidence_strength_mwc : float or None, optional
        Keyword alias for min_strength_mwc.
    output_dtype : Any, default np.int8
        NumPy dtype for output signal array.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        (filtered_signals, execution_funnel_statistics)
    """
    trig = np.asarray(lwc_triggers)
    n_samples = len(trig)
    if n_samples == 0:
        empty_stats = {
            "raw_triggers": 0,
            "hwc_vetoed": 0,
            "hwc_survived": 0,
            "mwc_vetoed": 0,
            "accepted_trades": 0,
            "direction": str(direction).lower(),
            "v_hwc": 0.0,
            "v_mwc": 0.0,
            "min_strength_hwc": 0.0,
            "min_strength_mwc": 0.0,
            "raw_mask": np.empty(0, dtype=bool),
            "hwc_veto_mask": np.empty(0, dtype=bool),
            "mwc_veto_mask": np.empty(0, dtype=bool),
            "accepted_mask": np.empty(0, dtype=bool),
        }
        empty_stats["retention_diagnostics"] = compute_trade_retention_diagnostics(empty_stats)
        return np.empty(0, dtype=output_dtype), empty_stats

    norm_dir = normalize_direction(direction)
    h_dir = np.asarray(hwc_direction, dtype=np.float64)
    h_str = np.asarray(hwc_strength, dtype=np.float64)
    m_dir = np.asarray(mwc_direction, dtype=np.float64)
    m_str = np.asarray(mwc_strength, dtype=np.float64)
    for name, values in (
        ("hwc_direction", h_dir),
        ("hwc_strength", h_str),
        ("mwc_direction", m_dir),
        ("mwc_strength", m_str),
    ):
        if len(values) != n_samples:
            raise ValueError(
                f"{name} length ({len(values)}) does not match "
                f"lwc_triggers length ({n_samples})"
            )

    # Resolve minimum evidence strength gates
    eff_min_str_hwc = (
        float(min_evidence_strength_hwc)
        if min_evidence_strength_hwc is not None
        else float(min_strength_hwc)
    )
    eff_min_str_mwc = (
        float(min_evidence_strength_mwc)
        if min_evidence_strength_mwc is not None
        else float(min_strength_mwc)
    )

    # Resolve asymmetric veto thresholds based on direction
    if norm_dir == "long":
        eff_v_hwc = (
            float(v_hwc_long)
            if v_hwc_long is not None
            else (float(v_hwc) if v_hwc is not None else DEFAULT_V_HWC_LONG)
        )
        eff_v_mwc = (
            float(v_mwc_long)
            if v_mwc_long is not None
            else (float(v_mwc) if v_mwc is not None else DEFAULT_V_MWC_LONG)
        )

        # For long entries, opposing HTF score is negative (bearish)
        hwc_opposing = (h_dir < -eff_v_hwc) & (h_str >= eff_min_str_hwc)
        mwc_opposing = (m_dir < -eff_v_mwc) & (m_str >= eff_min_str_mwc)
    else:  # short
        eff_v_hwc = (
            float(v_hwc_short)
            if v_hwc_short is not None
            else (float(v_hwc) if v_hwc is not None else DEFAULT_V_HWC_SHORT)
        )
        eff_v_mwc = (
            float(v_mwc_short)
            if v_mwc_short is not None
            else (float(v_mwc) if v_mwc is not None else DEFAULT_V_MWC_SHORT)
        )

        # For short entries, opposing HTF score is positive (bullish)
        hwc_opposing = (h_dir > +eff_v_hwc) & (h_str >= eff_min_str_hwc)
        mwc_opposing = (m_dir > +eff_v_mwc) & (m_str >= eff_min_str_mwc)

    # Funnel Staging
    raw_mask = trig != 0
    hwc_veto_mask = raw_mask & hwc_opposing
    hwc_survived_mask = raw_mask & (~hwc_veto_mask)
    mwc_veto_mask = hwc_survived_mask & mwc_opposing
    accepted_mask = hwc_survived_mask & (~mwc_veto_mask)

    # Output signal array (1 for accepted trade, 0 otherwise)
    signals = np.zeros(n_samples, dtype=output_dtype)
    signals[accepted_mask] = 1

    stats: dict[str, Any] = {
        "raw_triggers": int(np.sum(raw_mask)),
        "hwc_vetoed": int(np.sum(hwc_veto_mask)),
        "hwc_survived": int(np.sum(hwc_survived_mask)),
        "mwc_vetoed": int(np.sum(mwc_veto_mask)),
        "accepted_trades": int(np.sum(accepted_mask)),
        "direction": norm_dir,
        "v_hwc": eff_v_hwc,
        "v_mwc": eff_v_mwc,
        "min_strength_hwc": eff_min_str_hwc,
        "min_strength_mwc": eff_min_str_mwc,
        "raw_mask": raw_mask,
        "hwc_veto_mask": hwc_veto_mask,
        "mwc_veto_mask": mwc_veto_mask,
        "accepted_mask": accepted_mask,
    }
    stats["retention_diagnostics"] = compute_trade_retention_diagnostics(stats)

    return signals, stats


def compose_bidirectional_signals(
    lwc_triggers: Union[np.ndarray, pd.Series, Sequence[Any]],
    hwc_direction: Union[np.ndarray, pd.Series, Sequence[float]],
    hwc_strength: Union[np.ndarray, pd.Series, Sequence[float]],
    mwc_direction: Union[np.ndarray, pd.Series, Sequence[float]],
    mwc_strength: Union[np.ndarray, pd.Series, Sequence[float]],
    v_hwc_long: float = DEFAULT_V_HWC_LONG,
    v_hwc_short: float = DEFAULT_V_HWC_SHORT,
    v_mwc_long: float = DEFAULT_V_MWC_LONG,
    v_mwc_short: float = DEFAULT_V_MWC_SHORT,
    min_strength_hwc: float = DEFAULT_MIN_EVIDENCE_STRENGTH,
    min_strength_mwc: float = DEFAULT_MIN_EVIDENCE_STRENGTH,
    output_dtype: Any = np.int8,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compose bidirectional signals from signed LWC triggers (+1 = long, -1 = short, 0 = none).

    Applies asymmetric soft-vetoes independently to long and short entry triggers.

    Parameters
    ----------
    lwc_triggers : np.ndarray, pd.Series, or Sequence
        Array with +1 for long triggers, -1 for short triggers, 0 for no trigger.
    hwc_direction : np.ndarray, pd.Series, or Sequence
        HWC continuous direction scores.
    hwc_strength : np.ndarray, pd.Series, or Sequence
        HWC evidence strength scores.
    mwc_direction : np.ndarray, pd.Series, or Sequence
        MWC continuous direction scores.
    mwc_strength : np.ndarray, pd.Series, or Sequence
        MWC evidence strength scores.
    v_hwc_long : float, default 0.65
        HWC veto threshold for long entries.
    v_hwc_short : float, default 0.60
        HWC veto threshold for short entries.
    v_mwc_long : float, default 0.60
        MWC veto threshold for long entries.
    v_mwc_short : float, default 0.55
        MWC veto threshold for short entries.
    min_strength_hwc : float, default 0.15
        Minimum evidence strength for HWC.
    min_strength_mwc : float, default 0.15
        Minimum evidence strength for MWC.
    output_dtype : Any, default np.int8
        Output array dtype.

    Returns
    -------
    tuple[np.ndarray, dict[str, Any]]
        (composed_signals, combined_stats_dict)
    """
    trig = np.asarray(lwc_triggers)
    n_samples = len(trig)
    if n_samples == 0:
        return np.empty(0, dtype=output_dtype), {
            "long": {},
            "short": {},
            "total": {
                "raw_triggers": 0,
                "accepted_trades": 0,
                "retention_diagnostics": compute_trade_retention_diagnostics(0),
            },
            "retention_diagnostics": compute_trade_retention_diagnostics(0),
        }

    long_trigs = np.where(trig > 0, 1, 0)
    short_trigs = np.where(trig < 0, 1, 0)

    long_signals, long_stats = compose_hierarchical_signals(
        lwc_triggers=long_trigs,
        direction="long",
        hwc_direction=hwc_direction,
        hwc_strength=hwc_strength,
        mwc_direction=mwc_direction,
        mwc_strength=mwc_strength,
        v_hwc=v_hwc_long,
        v_mwc=v_mwc_long,
        min_strength_hwc=min_strength_hwc,
        min_strength_mwc=min_strength_mwc,
        output_dtype=output_dtype,
    )

    short_signals, short_stats = compose_hierarchical_signals(
        lwc_triggers=short_trigs,
        direction="short",
        hwc_direction=hwc_direction,
        hwc_strength=hwc_strength,
        mwc_direction=mwc_direction,
        mwc_strength=mwc_strength,
        v_hwc=v_hwc_short,
        v_mwc=v_mwc_short,
        min_strength_hwc=min_strength_hwc,
        min_strength_mwc=min_strength_mwc,
        output_dtype=output_dtype,
    )

    # Combined output: +1 for accepted long, -1 for accepted short, 0 otherwise
    combined_signals = np.zeros(n_samples, dtype=output_dtype)
    combined_signals[long_signals > 0] = 1
    combined_signals[short_signals > 0] = -1

    total_raw = long_stats["raw_triggers"] + short_stats["raw_triggers"]
    total_hwc_veto = long_stats["hwc_vetoed"] + short_stats["hwc_vetoed"]
    total_hwc_surv = total_raw - total_hwc_veto
    total_mwc_veto = long_stats["mwc_vetoed"] + short_stats["mwc_vetoed"]
    total_accepted = long_stats["accepted_trades"] + short_stats["accepted_trades"]

    total_stats = {
        "raw_triggers": total_raw,
        "hwc_vetoed": total_hwc_veto,
        "hwc_survived": total_hwc_surv,
        "mwc_vetoed": total_mwc_veto,
        "accepted_trades": total_accepted,
    }
    total_stats["retention_diagnostics"] = compute_trade_retention_diagnostics(total_stats)

    return combined_signals, {
        "long": long_stats,
        "short": short_stats,
        "total": total_stats,
        "retention_diagnostics": total_stats["retention_diagnostics"],
    }

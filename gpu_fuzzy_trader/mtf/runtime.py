"""Frozen runtime evaluation for hierarchical MTF candidates.

This module is deliberately small and deterministic.  Training code writes the
rule thresholds and OOF evidence into the archives; this module only applies
those frozen rules to a new tape, builds causal higher-timeframe features, and
composes the resulting signals.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.data.multi_timeframe import (
    align_htf_features_causal,
    build_complete_higher_bars,
    compute_timeframe_features,
)
from gpu_fuzzy_trader.features.fuzzy_scaling import validate_rule_feature_ranges
from gpu_fuzzy_trader.backtest.symbol_conditions import (
    normalize_symbol_value,
    split_feature_and_symbol_conditions,
)
from gpu_fuzzy_trader.mtf.composer import compose_bidirectional_signals
from gpu_fuzzy_trader.mtf.diagnostics import compute_trade_retention_diagnostics
from gpu_fuzzy_trader.mtf.ensembler import (
    compute_ensemble_direction_and_strength,
    compute_rule_weights,
)

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")
_CONDITION_RE = re.compile(
    r"^\s*\[([^\]]+)\]\s*(>=|<=|==|>|<)\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*$"
)
_IS_RE = re.compile(r"^\s*\[([^\]]+)\]\s+IS\s+(.+?)\s*$")


def _value_mask(values: pd.Series, value_name: str) -> np.ndarray:
    """Apply the legacy exported fuzzy-value vocabulary to a numeric column."""
    values = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    name = str(value_name).strip().lower()
    masks: dict[str, np.ndarray] = {
        "active (1)": values == 1,
        "inactive (0)": values == 0,
        "positive": values == 1,
        "positive (1)": values == 1,
        "neutral": values == 0,
        "neutral (0)": values == 0,
        "negative": values == -1,
        "negative (-1)": values == -1,
        "strong negative": values <= -0.25,
        "weak negative": (values > -0.25) & (values <= -1e-5),
        "exactly zero": (values > -1e-5) & (values <= 1e-5),
        "weak positive": (values > 1e-5) & (values <= 0.25),
        "strong positive": values > 0.25,
        "very low": values <= 0.2,
        "low": (values > 0.2) & (values <= 0.4),
        "medium": (values > 0.4) & (values <= 0.6),
        "high": (values > 0.6) & (values <= 0.8),
        "very high": values > 0.8,
        "extreme bearish": values <= -0.8,
        "strong bearish": (values > -0.8) & (values <= -0.6),
        "bearish": (values > -0.6) & (values <= -0.4),
        "weak bearish": (values > -0.4) & (values <= -0.2),
        "neutral negative": (values > -0.2) & (values <= 0.0),
        "neutral positive": (values > 0.0) & (values <= 0.2),
        "weak bullish": (values > 0.2) & (values <= 0.4),
        "bullish": (values > 0.4) & (values <= 0.6),
        "strong bullish": (values > 0.6) & (values <= 0.8),
        "extreme bullish": values > 0.8,
    }
    if name not in masks:
        raise ValueError(f"Unknown fuzzy condition value {value_name!r}")
    return np.asarray(masks[name] & np.isfinite(values), dtype=bool)


def condition_mask(df: pd.DataFrame, condition: str) -> np.ndarray:
    """Evaluate either an exported ``IS`` condition or a numeric comparison."""
    if not isinstance(condition, str):
        raise ValueError(f"Condition must be a string, got {type(condition).__name__}")

    numeric_match = _CONDITION_RE.match(condition)
    if numeric_match:
        feature, operator, raw_value = numeric_match.groups()
        if feature not in df.columns:
            raise ValueError(f"Unknown feature in condition: {feature!r}")
        values = pd.to_numeric(df[feature], errors="coerce").to_numpy(dtype=float)
        threshold = float(raw_value)
        if operator == ">":
            mask = values > threshold
        elif operator == ">=":
            mask = values >= threshold
        elif operator == "<":
            mask = values < threshold
        elif operator == "<=":
            mask = values <= threshold
        else:
            mask = values == threshold
        return np.asarray(mask & np.isfinite(values), dtype=bool)

    is_match = _IS_RE.match(condition)
    if not is_match:
        raise ValueError(
            f"Invalid condition format: {condition!r}. Expected '[feature] IS Value' "
            "or '[feature] <numeric threshold>'."
        )
    feature, value_name = is_match.groups()
    if feature not in df.columns:
        raise ValueError(f"Unknown feature in condition: {feature!r}")
    return _value_mask(df[feature], value_name)


def rule_mask(df: pd.DataFrame, rule: Mapping[str, Any]) -> np.ndarray:
    """Return the conjunction of a frozen rule's conditions and symbol scope."""
    conditions = rule.get("conditions", [])
    if not isinstance(conditions, (list, tuple)) or not conditions:
        raise ValueError("MTF rule must contain a non-empty conditions list")
    mask = np.ones(len(df), dtype=bool)
    feature_conditions, condition_symbols = split_feature_and_symbol_conditions(
        [str(condition) for condition in conditions]
    )
    for condition in feature_conditions:
        mask &= condition_mask(df, str(condition))

    symbols = rule.get("symbols") or rule.get("eligible_symbols")
    allowed_symbols = list(condition_symbols)
    if symbols:
        allowed_symbols.extend(str(value) for value in symbols)
    if allowed_symbols:
        if "symbol" not in df.columns:
            return np.zeros(len(df), dtype=bool)
        allowed = {
            normalize_symbol_value(value)
            for value in allowed_symbols
        }
        normalized = df["symbol"].map(normalize_symbol_value)
        mask &= normalized.isin(allowed).to_numpy()
    return mask


def _or_rule_masks(frame: pd.DataFrame, rules: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """OR rule masks without relying on NumPy's scalar-only reduce initial."""
    result = np.zeros(len(frame), dtype=bool)
    for rule in rules:
        result |= rule_mask(frame, rule)
    return result


def _prefix_features(features: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = features.copy()
    rename = {
        column: f"{prefix}{column}"
        for column in renamed.columns
        if column not in ("datetime", "symbol", *_OHLCV_COLUMNS)
    }
    return renamed.rename(columns=rename)


def _prepare_target_with_history(
    raw_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Prepare a target tape with optional strictly prior indicator history."""
    if history_df is None or history_df.empty:
        frame = prepare_causal_mtf_frame(raw_df)
        return frame, np.ones(len(frame), dtype=bool)

    history = history_df.copy()
    target = raw_df.copy()
    history["_mtf_target_row"] = False
    target["_mtf_target_row"] = True
    combined = pd.concat([history, target], ignore_index=True, sort=False)
    frame = prepare_causal_mtf_frame(combined)
    target_mask = frame["_mtf_target_row"].fillna(False).to_numpy(dtype=bool)
    return frame.drop(columns=["_mtf_target_row"]), target_mask


def _slice_composition_stats(
    stats: Mapping[str, Any],
    target_mask: np.ndarray,
) -> dict[str, Any]:
    """Restrict composition funnel arrays to a target split after warm-up."""
    target_mask = np.asarray(target_mask, dtype=bool)

    def slice_one(values: Mapping[str, Any]) -> dict[str, Any]:
        raw_mask = np.asarray(
            values.get("raw_mask", np.zeros(len(target_mask))), dtype=bool
        )[target_mask]
        hwc_mask = np.asarray(
            values.get("hwc_veto_mask", np.zeros(len(target_mask))), dtype=bool
        )[target_mask]
        mwc_mask = np.asarray(
            values.get("mwc_veto_mask", np.zeros(len(target_mask))), dtype=bool
        )[target_mask]
        accepted_mask = np.asarray(
            values.get("accepted_mask", np.zeros(len(target_mask))), dtype=bool
        )[target_mask]
        out = dict(values)
        out.update({
            "raw_triggers": int(raw_mask.sum()),
            "hwc_vetoed": int(hwc_mask.sum()),
            "hwc_survived": int((raw_mask & ~hwc_mask).sum()),
            "mwc_vetoed": int(mwc_mask.sum()),
            "accepted_trades": int(accepted_mask.sum()),
            "raw_mask": raw_mask,
            "hwc_veto_mask": hwc_mask,
            "mwc_veto_mask": mwc_mask,
            "accepted_mask": accepted_mask,
        })
        out["retention_diagnostics"] = compute_trade_retention_diagnostics(out)
        return out

    if "long" not in stats and "short" not in stats:
        return slice_one(stats)

    long_stats = slice_one(stats.get("long", {}))
    short_stats = slice_one(stats.get("short", {}))
    total = {
        "raw_triggers": long_stats["raw_triggers"] + short_stats["raw_triggers"],
        "hwc_vetoed": long_stats["hwc_vetoed"] + short_stats["hwc_vetoed"],
        "hwc_survived": long_stats["hwc_survived"] + short_stats["hwc_survived"],
        "mwc_vetoed": long_stats["mwc_vetoed"] + short_stats["mwc_vetoed"],
        "accepted_trades": long_stats["accepted_trades"] + short_stats["accepted_trades"],
    }
    total["retention_diagnostics"] = compute_trade_retention_diagnostics(total)
    return {
        "long": long_stats,
        "short": short_stats,
        "total": total,
        "retention_diagnostics": total["retention_diagnostics"],
    }


def prepare_causal_mtf_frame(
    raw_df: pd.DataFrame,
    *,
    mwc_minutes: int = 60,
    hwc_minutes: int = 240,
) -> pd.DataFrame:
    """Build a raw execution frame with causally aligned HWC/MWC features."""
    required = {"datetime", "symbol", *_OHLCV_COLUMNS}
    missing = sorted(required - set(raw_df.columns))
    if missing:
        raise ValueError(f"Raw MTF frame is missing required columns: {missing}")
    if raw_df.empty:
        return raw_df.copy()

    lwc = raw_df.copy()
    lwc["datetime"] = pd.to_datetime(lwc["datetime"], errors="raise", utc=True).dt.tz_localize(None)

    for timeframe, prefix in ((mwc_minutes, "mwc_"), (hwc_minutes, "hwc_")):
        bars = build_complete_higher_bars(lwc, timeframe)
        features = _prefix_features(
            compute_timeframe_features(
                bars, timeframe, include_raw_features=False,
            ),
            prefix,
        )
        lwc = align_htf_features_causal(lwc, features, timeframe)

    # Raw tapes already carry engineered LWC features in production.  For a
    # pure OHLCV tape, provide the same causal feature family under a distinct
    # prefix without replacing raw columns.
    lwc_source = lwc.loc[:, ["datetime", "symbol", *_OHLCV_COLUMNS]].copy()
    lwc_features = _prefix_features(
        compute_timeframe_features(
            lwc_source, 15, include_raw_features=False,
        ),
        "lwc_",
    )
    for column in lwc_features.columns:
        if column not in lwc.columns:
            lwc[column] = lwc_features[column]
    lwc_feature_columns = [
        column for column in lwc.columns if str(column).startswith("lwc_")
    ]

    if lwc_feature_columns:
        # Match the repository's established feature warm-up convention while
        # keeping the imputation local to the causal LWC feature family. No
        # future value is used; unavailable indicators are neutral zeroes.
        lwc[lwc_feature_columns] = lwc[lwc_feature_columns].fillna(0.0)
    validate_rule_feature_ranges(lwc)
    return lwc


def _causal_score_columns(
    raw_df: pd.DataFrame,
    scores: pd.DataFrame | None,
    timeframe_minutes: int,
    direction_column: str,
    strength_column: str,
) -> pd.DataFrame:
    """Align a layer's OOF scores to LWC execution timestamps without filling."""
    frame = prepare_causal_mtf_frame(raw_df)
    frame[direction_column] = np.nan
    frame[strength_column] = np.nan
    if scores is None or scores.empty:
        return frame
    required = {"datetime", "symbol", "direction_score", "strength_score"}
    missing = sorted(required - set(scores.columns))
    if missing:
        raise ValueError(f"OOF scores are missing required columns: {missing}")

    source = scores.loc[
        :, ["datetime", "symbol", "direction_score", "strength_score"]
    ].copy()
    source["datetime"] = pd.to_datetime(
        source["datetime"], errors="raise", utc=True
    ).dt.tz_localize(None)
    source["direction_score"] = pd.to_numeric(
        source["direction_score"], errors="raise"
    )
    source["strength_score"] = pd.to_numeric(
        source["strength_score"], errors="raise"
    )
    source["_available_at"] = source["datetime"] + pd.Timedelta(
        minutes=int(timeframe_minutes)
    )
    if not np.isfinite(
        source[["direction_score", "strength_score"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("OOF scores contain non-finite direction or strength values")
    if source.duplicated(["symbol", "_available_at"]).any():
        raise ValueError("OOF scores contain duplicate symbol/availability timestamps")

    for symbol, group in frame.groupby("symbol", sort=False, observed=False):
        right = source[source["symbol"] == symbol].sort_values("_available_at")
        if right.empty:
            continue
        left = group.loc[:, ["datetime"]].copy()
        left["_exec_at"] = left["datetime"] + pd.Timedelta(minutes=15)
        matched = pd.merge_asof(
            left.sort_values("_exec_at"),
            right.loc[:, ["_available_at", "direction_score", "strength_score"]].rename(
                columns={"_available_at": "_exec_at"}
            ),
            on="_exec_at",
            direction="backward",
        )
        target = group.sort_values("datetime").index.to_numpy()
        frame.loc[target, direction_column] = matched["direction_score"].to_numpy()
        frame.loc[target, strength_column] = matched["strength_score"].to_numpy()
    return frame


def attach_oof_layer_scores(
    raw_df: pd.DataFrame,
    *,
    hwc_scores: pd.DataFrame | None,
    mwc_scores: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build a training frame carrying only causally available HWC/MWC OOF scores."""
    frame = _causal_score_columns(
        raw_df, hwc_scores, 240, "mtf_hwc_direction", "mtf_hwc_strength"
    )
    frame = _causal_score_columns(
        frame, mwc_scores, 60, "mtf_mwc_direction", "mtf_mwc_strength"
    )
    score_columns = [
        "mtf_hwc_direction", "mtf_hwc_strength",
        "mtf_mwc_direction", "mtf_mwc_strength",
    ]
    frame["_mtf_oof_available"] = frame[score_columns].notna().all(axis=1)
    return frame


def attach_frozen_layer_scores(
    raw_df: pd.DataFrame,
    hwc_rules: Sequence[Mapping[str, Any]],
    mwc_rules: Sequence[Mapping[str, Any]],
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Attach final full-train ensemble scores for validation or frozen OOS."""
    frame, target_mask = _prepare_target_with_history(raw_df, history_df)
    hwc_direction, hwc_strength, _ = ensemble_layer_scores(frame, hwc_rules)
    mwc_direction, mwc_strength, _ = ensemble_layer_scores(
        frame,
        mwc_rules,
        upstream_direction=hwc_direction,
    )
    frame["mtf_hwc_direction"] = hwc_direction
    frame["mtf_hwc_strength"] = hwc_strength
    frame["mtf_mwc_direction"] = mwc_direction
    frame["mtf_mwc_strength"] = mwc_strength
    frame["_mtf_oof_available"] = True
    if history_df is not None and not history_df.empty:
        frame = frame.loc[target_mask].reset_index(drop=True)
    return frame


def ensemble_layer_scores(
    frame: pd.DataFrame,
    rules: Sequence[Mapping[str, Any]],
    *,
    upstream_direction: np.ndarray | None = None,
    conditional_support_threshold: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate frozen layer rules and return direction, strength, active masks.

    When ``upstream_direction`` is supplied, it is the frozen HWC score used
    to gate conditional MWC rules: long MWC evidence requires HWC support at
    or above the threshold, and short MWC evidence requires support at or
    below its negative. This keeps runtime semantics identical to conditional
    MWC discovery rather than using the HWC score only as documentation.
    """
    rules = list(rules)
    if not rules:
        return (
            np.zeros(len(frame), dtype=np.float64),
            np.zeros(len(frame), dtype=np.float64),
            np.zeros((len(frame), 0), dtype=bool),
        )
    active = np.column_stack([rule_mask(frame, rule) for rule in rules]).astype(bool)
    if upstream_direction is not None:
        upstream = np.asarray(upstream_direction, dtype=float)
        if len(upstream) != len(frame):
            raise ValueError("upstream_direction length does not match frame")
        for index, rule in enumerate(rules):
            direction = str(rule.get("direction", "")).strip().lower()
            if direction in ("long", "1", "buy", "+1"):
                active[:, index] &= (
                    np.isfinite(upstream)
                    & (upstream >= float(conditional_support_threshold))
                )
            elif direction in ("short", "-1", "sell"):
                active[:, index] &= (
                    np.isfinite(upstream)
                    & (upstream <= -float(conditional_support_threshold))
                )
            else:
                active[:, index] = False
    weights = compute_rule_weights(rules)
    directions = [str(rule.get("direction", "")).lower() for rule in rules]
    direction, strength = compute_ensemble_direction_and_strength(
        active, directions, weights
    )
    return direction, strength, active


def _candidate_layer_components(
    candidate: Any,
    raw_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prepare one candidate's causal HWC/MWC score arrays."""
    frame, target_mask = _prepare_target_with_history(raw_df, history_df)
    hwc_direction, hwc_strength, _ = ensemble_layer_scores(
        frame, candidate.hwc_rules
    )
    mwc_direction, mwc_strength, _ = ensemble_layer_scores(
        frame,
        candidate.mwc_rules,
        upstream_direction=hwc_direction,
    )
    frame["mtf_hwc_direction"] = hwc_direction
    frame["mtf_hwc_strength"] = hwc_strength
    frame["mtf_mwc_direction"] = mwc_direction
    frame["mtf_mwc_strength"] = mwc_strength
    return (
        frame,
        target_mask,
        hwc_direction,
        hwc_strength,
        mwc_direction,
        mwc_strength,
    )


def _rule_direction(value: Any) -> str:
    normalized = str(value).strip().lower()
    if normalized in ("long", "buy", "1", "+1"):
        return "long"
    if normalized in ("short", "sell", "-1"):
        return "short"
    return ""


def _compose_candidate_rule_masks(
    candidate: Any,
    frame: pd.DataFrame,
    hwc_direction: np.ndarray,
    hwc_strength: np.ndarray,
    mwc_direction: np.ndarray,
    mwc_strength: np.ndarray,
) -> list[np.ndarray]:
    """Compose one accepted mask for each frozen LWC rule."""
    masks: list[np.ndarray] = []
    for rule in candidate.lwc_rules:
        rule_direction = _rule_direction(rule.get("direction"))
        raw = rule_mask(frame, rule)
        if candidate.direction != "bidirectional":
            if rule_direction != candidate.direction:
                masks.append(np.zeros(len(frame), dtype=bool))
                continue
            trigger = raw.astype(np.int8)
        elif rule_direction == "long":
            trigger = raw.astype(np.int8)
        elif rule_direction == "short":
            trigger = -raw.astype(np.int8)
        else:
            masks.append(np.zeros(len(frame), dtype=bool))
            continue
        composed, _ = candidate.compose(
            trigger,
            hwc_direction,
            hwc_strength,
            mwc_direction,
            mwc_strength,
        )
        masks.append(np.asarray(composed) != 0)
    return masks


def evaluate_candidate_frame(
    candidate: Any,
    raw_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
    *,
    _return_rule_masks: bool = False,
) -> tuple[np.ndarray, dict[str, Any], pd.DataFrame]:
    """Apply a frozen candidate and return signals, funnel stats, and audit rows."""
    (
        frame,
        target_mask,
        hwc_direction,
        hwc_strength,
        mwc_direction,
        mwc_strength,
    ) = _candidate_layer_components(
        candidate,
        raw_df,
        history_df=history_df,
    )
    rule_masks = (
        _compose_candidate_rule_masks(
            candidate,
            frame,
            hwc_direction,
            hwc_strength,
            mwc_direction,
            mwc_strength,
        )
        if _return_rule_masks
        else None
    )

    if candidate.direction == "bidirectional":
        long_rules = [r for r in candidate.lwc_rules if str(r.get("direction", "")).lower() == "long"]
        short_rules = [r for r in candidate.lwc_rules if str(r.get("direction", "")).lower() == "short"]
        long_mask = _or_rule_masks(frame, long_rules)
        short_mask = _or_rule_masks(frame, short_rules)
        triggers = np.where(long_mask, 1, np.where(short_mask, -1, 0))
        composer_params = candidate.composer_params
        composer_kwargs = {
            key: composer_params[key]
            for key in (
                "v_hwc_long", "v_hwc_short", "v_mwc_long", "v_mwc_short"
            )
            if key in composer_params
        }
        for canonical, alias in (
            ("min_strength_hwc", "min_evidence_strength_hwc"),
            ("min_strength_mwc", "min_evidence_strength_mwc"),
        ):
            if alias in composer_params:
                composer_kwargs[canonical] = composer_params[alias]
            elif canonical in composer_params:
                composer_kwargs[canonical] = composer_params[canonical]
        signals, stats = compose_bidirectional_signals(
            triggers,
            hwc_direction,
            hwc_strength,
            mwc_direction,
            mwc_strength,
            **composer_kwargs,
        )
        direction_labels = np.where(signals > 0, "long", np.where(signals < 0, "short", "none"))
        raw_trigger = triggers != 0
    else:
        target = candidate.direction
        trigger_rules = [
            rule for rule in candidate.lwc_rules
            if str(rule.get("direction", "")).lower() == target
        ]
        raw_trigger = _or_rule_masks(frame, trigger_rules)
        trigger_values = raw_trigger.astype(np.int8)
        signals, stats = candidate.compose(
            trigger_values,
            hwc_direction,
            hwc_strength,
            mwc_direction,
            mwc_strength,
        )
        direction_labels = np.full(len(frame), target, dtype=object)

    if candidate.direction == "bidirectional":
        long_stats = stats.get("long", {})
        short_stats = stats.get("short", {})
        hwc_veto = np.where(
            triggers > 0,
            np.asarray(
                long_stats.get("hwc_veto_mask", np.zeros(len(frame))),
                dtype=np.int8,
            ),
            np.where(
                triggers < 0,
                np.asarray(
                    short_stats.get("hwc_veto_mask", np.zeros(len(frame))),
                    dtype=np.int8,
                ),
                0,
            ),
        )
        mwc_veto = np.where(
            triggers > 0,
            np.asarray(
                long_stats.get("mwc_veto_mask", np.zeros(len(frame))),
                dtype=np.int8,
            ),
            np.where(
                triggers < 0,
                np.asarray(
                    short_stats.get("mwc_veto_mask", np.zeros(len(frame))),
                    dtype=np.int8,
                ),
                0,
            ),
        )
    else:
        hwc_veto = np.asarray(
            stats.get("hwc_veto_mask", np.zeros(len(frame))), dtype=np.int8
        )
        mwc_veto = np.asarray(
            stats.get("mwc_veto_mask", np.zeros(len(frame))), dtype=np.int8
        )

    audit = pd.DataFrame({
        "datetime": frame["datetime"].to_numpy(),
        "symbol": frame["symbol"].to_numpy() if "symbol" in frame.columns else "UNKNOWN",
        "direction": direction_labels,
        "lwc_trigger": raw_trigger.astype(np.int8),
        "hwc_veto": hwc_veto,
        "mwc_veto": mwc_veto,
        "accepted": (signals != 0).astype(np.int8),
    })
    if "_symbol_bar_index" in frame.columns:
        audit["symbol_bar_index"] = frame["_symbol_bar_index"].to_numpy()
    if history_df is not None and not history_df.empty:
        signals = signals[target_mask]
        stats = _slice_composition_stats(stats, target_mask)
        audit = audit.loc[target_mask].reset_index(drop=True)
        if rule_masks is not None:
            rule_masks = [
                np.asarray(mask, dtype=bool)[target_mask]
                for mask in rule_masks
            ]
    if _return_rule_masks:
        # This private extension is consumed only by the per-rule execution
        # path below.  Keep the public three-value API unchanged.
        return np.asarray(signals), stats, audit, rule_masks  # type: ignore[return-value]
    return np.asarray(signals), stats, audit


def evaluate_candidate_rule_masks(
    candidate: Any,
    raw_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> tuple[list[np.ndarray], dict[str, Any], pd.DataFrame]:
    """Return one causal, accepted execution mask for every frozen LWC rule.

    The masks are aligned to ``raw_df``.  If history is supplied, it is used
    only to build higher-timeframe features and is removed from the returned
    masks, statistics, and audit rows.
    """
    _, stats, audit, masks = evaluate_candidate_frame(
        candidate,
        raw_df,
        history_df=history_df,
        _return_rule_masks=True,
    )
    return list(masks or []), stats, audit

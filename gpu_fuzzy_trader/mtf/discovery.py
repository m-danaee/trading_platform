"""Purged, directional MTF rule discovery.

The production MTF stages use this deterministic candidate search as the
feature/rule evaluator behind the existing evolutionary LWC search.  Every
threshold and score is fitted inside a temporal training fold; downstream MWC
search receives only the upstream HWC OOF score frame.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Sequence

import numpy as np
import pandas as pd

from gpu_fuzzy_trader.data.multi_timeframe import (
    build_complete_higher_bars,
    compute_timeframe_features,
)
from gpu_fuzzy_trader.evolution.directional_evaluator import (
    classify_directional_labels,
    compute_forward_movement_labels,
    evaluate_conditional_directional_rule,
    evaluate_directional_rule,
    fit_directional_threshold,
)
from gpu_fuzzy_trader.mtf.cross_fitting import TemporalFold
from gpu_fuzzy_trader.mtf.ensembler import (
    compute_ensemble_direction_and_strength,
    compute_rule_weights,
)
from gpu_fuzzy_trader.mtf.runtime import condition_mask
from gpu_fuzzy_trader.research_profile import RuleSearchProfile, get_rule_search_profile

_OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class LayerDiscoveryResult:
    timeframe: str
    rules: list[dict[str, Any]]
    oof_scores: pd.DataFrame
    bars: pd.DataFrame
    feature_schema_hash: str
    data_hash: str
    theta_per_oof_fold: dict[str, float]
    theta_final_train: float
    fold_rules: dict[int, list[dict[str, Any]]]
    search_metadata: dict[str, Any] = field(default_factory=dict)
    oof_score_hash: str = ""


def _frame_hash(frame: pd.DataFrame) -> str:
    columns = sorted(str(column) for column in frame.columns)
    canonical = frame.loc[:, columns].copy()
    hashed = pd.util.hash_pandas_object(canonical, index=True).to_numpy().tobytes()
    return hashlib.sha256(
        json.dumps(columns, separators=(",", ":")).encode("utf-8") + hashed
    ).hexdigest()


def _schema_hash(frame: pd.DataFrame) -> str:
    schema = [(str(column), str(dtype)) for column, dtype in frame.dtypes.items()]
    return hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8")).hexdigest()


def _build_layer_frame(
    raw_df: pd.DataFrame,
    timeframe_minutes: int,
    role: str,
) -> pd.DataFrame:
    bars = build_complete_higher_bars(raw_df, timeframe_minutes)
    if bars.empty:
        return bars
    features = compute_timeframe_features(bars, timeframe_minutes)
    prefix = "hwc_" if role == "hwc" else "mwc_"
    rename = {
        column: f"{prefix}{column}"
        for column in features.columns
        if column not in ("datetime", "symbol", *_OHLCV_COLUMNS)
    }
    result = features.rename(columns=rename)
    result["_move"] = np.nan
    for _, group in result.groupby("symbol", sort=False, observed=False):
        indices = group.index.to_numpy()
        close = group["close"].to_numpy(dtype=float)
        atr = group[f"{prefix}atr_14"].to_numpy(dtype=float)
        moves = compute_forward_movement_labels(
            close,
            atr,
            horizon_bars=int(get_rule_search_profile(role).forward_horizon_bars),
        )
        result.loc[indices, "_move"] = moves
    return result.sort_values(["datetime", "symbol"]).reset_index(drop=True)


def _eligible_numeric_features(frame: pd.DataFrame, role: str) -> list[str]:
    prefix = "hwc_" if role == "hwc" else "mwc_"
    excluded = {"datetime", "symbol", "_move", *_OHLCV_COLUMNS}
    columns = []
    for column in frame.columns:
        if column in excluded or not str(column).startswith(prefix):
            continue
        if pd.api.types.is_numeric_dtype(frame[column]) and frame[column].notna().any():
            columns.append(str(column))
    # Keep candidate generation bounded and deterministic on low-RAM hosts.
    return sorted(columns)[:24]


def _align_upstream_scores(
    bars: pd.DataFrame,
    upstream_scores: pd.DataFrame | None,
    upstream_timeframe_minutes: int,
) -> np.ndarray | None:
    if upstream_scores is None or upstream_scores.empty:
        return None
    required = {"datetime", "symbol", "direction_score", "strength_score"}
    if not required.issubset(upstream_scores.columns):
        raise ValueError(f"Upstream OOF scores are missing {sorted(required - set(upstream_scores.columns))}")
    source = upstream_scores.copy()
    source["datetime"] = pd.to_datetime(source["datetime"], errors="raise", utc=True).dt.tz_localize(None)
    source["direction_score"] = pd.to_numeric(
        source["direction_score"], errors="raise"
    )
    source["strength_score"] = pd.to_numeric(
        source["strength_score"], errors="raise"
    )
    source["_available_at"] = source["datetime"] + pd.Timedelta(minutes=upstream_timeframe_minutes)
    if source.duplicated(["symbol", "_available_at"]).any():
        raise ValueError("Upstream OOF scores contain duplicate symbol/availability timestamps")
    if not np.isfinite(
        source[["direction_score", "strength_score"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("Upstream OOF scores contain non-finite values")
    result = np.full(len(bars), np.nan, dtype=float)
    for symbol, group in bars.groupby("symbol", sort=False, observed=False):
        left = group[["datetime"]].copy()
        right = source[source["symbol"] == symbol].sort_values("_available_at")
        if right.empty:
            continue
        matched = pd.merge_asof(
            left.sort_values("datetime"),
            right[["_available_at", "direction_score"]].rename(
                columns={"_available_at": "datetime"}
            ),
            on="datetime",
            direction="backward",
        )
        result[group.sort_values("datetime").index.to_numpy()] = matched[
            "direction_score"
        ].to_numpy(dtype=float)
    return result


def _candidate_key(feature: str, direction: str, operator: str, quantile: float) -> tuple:
    return feature, direction, operator, round(float(quantile), 6)


def _select_balanced_candidates(
    candidates: Sequence[dict[str, Any]],
    limit: int,
) -> list[dict[str, Any]]:
    """Keep directional support in bounded fold populations when available."""
    limit = max(1, int(limit))
    def ranking(item: dict[str, Any]) -> tuple[float, str, str]:
        score = item.get(
            "score",
            float(item.get("mcc", 0.0))
            + float(item.get("directional_edge", 0.0)),
        )
        return (-float(score), str(item.get("feature", "")), str(item.get("operator", "")))

    ranked = sorted(candidates, key=ranking)
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    for direction in ("long", "short"):
        first = next(
            (item for item in ranked if item["direction"] == direction), None
        )
        if first is not None and len(selected) < limit:
            selected.append(first)
            selected_ids.add(id(first))
    for item in ranked:
        if len(selected) >= limit:
            break
        if id(item) not in selected_ids:
            selected.append(item)
            selected_ids.add(id(item))
    return selected


def _directional_pareto_front(
    candidates: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the non-dominated front for directional discovery metrics.

    All objectives are maximized: directional edge, MCC, stability, and
    coverage quality (the negative coverage penalty).  This keeps HWC/MWC
    selection aligned with directional evidence instead of collapsing the
    fold population to one scalar score before balancing long and short.
    """
    if not candidates:
        return []

    def vector(item: dict[str, Any]) -> np.ndarray:
        metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
        oof = item.get("oof_metrics") if isinstance(item.get("oof_metrics"), dict) else {}
        edge = metrics.get("test_directional_edge", oof.get("directional_edge", item.get("directional_edge", 0.0)))
        mcc = metrics.get("test_mcc", oof.get("mcc", item.get("mcc", 0.0)))
        stability = item.get("stability", oof.get("stability", 1.0))
        penalty = metrics.get(
            "test_coverage_penalty",
            oof.get("coverage_penalty", item.get("coverage_penalty", 0.0)),
        )
        false_confirmation = metrics.get(
            "test_false_confirmation_penalty",
            oof.get("false_confirmation_penalty", item.get("false_confirmation_penalty", 0.0)),
        )
        values = np.asarray(
            [edge, mcc, stability, -penalty, -false_confirmation],
            dtype=float,
        )
        return np.where(np.isfinite(values), values, -np.inf)

    objective_vectors = [vector(item) for item in candidates]
    keep: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        current = objective_vectors[index]
        dominated = False
        for other_index, other in enumerate(objective_vectors):
            if index == other_index:
                continue
            if np.all(other >= current) and np.any(other > current):
                dominated = True
                break
        if not dominated:
            keep.append(candidate)
    return keep or list(candidates)


def _fit_fold_candidates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    role: str,
    upstream_train: np.ndarray | None,
    upstream_test: np.ndarray | None,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, float]:
    profile = get_rule_search_profile(role)
    train_moves = train["_move"].to_numpy(dtype=float)
    test_moves = test["_move"].to_numpy(dtype=float)
    theta = fit_directional_threshold(train_moves, profile.quantile)
    train_labels = classify_directional_labels(train_moves, theta)
    test_labels = classify_directional_labels(test_moves, theta)
    target_coverage = profile.target_coverage
    features = _eligible_numeric_features(train, role)
    candidates: list[dict[str, Any]] = []
    for feature in features:
        train_values = pd.to_numeric(train[feature], errors="coerce")
        finite_values = train_values[np.isfinite(train_values)]
        if finite_values.empty:
            continue
        for quantile in (0.35, 0.50, 0.65):
            threshold = float(finite_values.quantile(quantile))
            if not np.isfinite(threshold):
                continue
            for operator in (">=", "<="):
                condition = f"[{feature}] {operator} {threshold:.12g}"
                train_mask = condition_mask(train, condition)
                test_mask = condition_mask(test, condition)
                fold_metrics: dict[str, dict[str, float]] = {}
                for direction in ("long", "short"):
                    if role == "mwc":
                        if upstream_train is None:
                            continue
                        upstream_test_values = (
                            upstream_test
                            if upstream_test is not None
                            else np.full(len(test), np.nan)
                        )
                        edge, mcc, coverage_penalty = evaluate_conditional_directional_rule(
                            train_mask,
                            train_moves,
                            upstream_train,
                            theta,
                            direction=direction,
                            support_threshold=profile.support_threshold,
                            target_coverage=target_coverage,
                        )
                        test_edge, test_mcc, test_penalty = evaluate_conditional_directional_rule(
                            test_mask,
                            test_moves,
                            upstream_test_values,
                            theta,
                            direction=direction,
                            support_threshold=profile.support_threshold,
                            target_coverage=target_coverage,
                        )
                        support_train = np.isfinite(upstream_train) & np.isfinite(train_moves)
                        support_test = np.isfinite(upstream_test_values) & np.isfinite(test_moves)
                        if direction == "long":
                            support_train &= upstream_train >= profile.support_threshold
                            support_test &= upstream_test_values >= profile.support_threshold
                        else:
                            support_train &= upstream_train <= -profile.support_threshold
                            support_test &= upstream_test_values <= -profile.support_threshold
                    else:
                        edge, mcc, coverage_penalty = evaluate_directional_rule(
                            train_mask,
                            train_labels,
                            direction=direction,
                            target_coverage=target_coverage,
                        )
                        test_edge, test_mcc, test_penalty = evaluate_directional_rule(
                            test_mask,
                            test_labels,
                            direction=direction,
                            target_coverage=target_coverage,
                        )
                        support_train = np.ones(len(train), dtype=bool)
                        support_test = np.ones(len(test), dtype=bool)
                    fold_metrics[direction] = {
                        "directional_edge": float(edge),
                        "mcc": float(mcc),
                        "coverage_penalty": float(coverage_penalty),
                        "false_confirmation_penalty": float(
                            max(0.0, -float(edge)) if role == "mwc" else 0.0
                        ),
                        "coverage": float(np.mean(train_mask[support_train]))
                        if np.any(support_train) else 0.0,
                        "test_directional_edge": float(test_edge),
                        "test_mcc": float(test_mcc),
                        "test_coverage_penalty": float(test_penalty),
                        "test_false_confirmation_penalty": float(
                            max(0.0, -float(test_edge)) if role == "mwc" else 0.0
                        ),
                        "test_coverage": float(np.mean(test_mask[support_test]))
                        if np.any(support_test) else 0.0,
                    }
                for direction, metrics in fold_metrics.items():
                    score = (
                        metrics["directional_edge"]
                        + metrics["mcc"]
                        - metrics["coverage_penalty"]
                        - metrics["false_confirmation_penalty"]
                    )
                    candidates.append({
                        "feature": feature,
                        "direction": direction,
                        "operator": operator,
                        "quantile": quantile,
                        "threshold": threshold,
                        "condition": condition,
                        "metrics": metrics,
                        "score": float(score),
                        "test_mask": test_mask,
                    })
    pareto_candidates = _directional_pareto_front(candidates)
    selected = _select_balanced_candidates(
        pareto_candidates, max(2, profile.max_conditions * 4)
    )
    if not selected:
        return [], np.zeros(len(test)), np.zeros(len(test)), theta

    rules = []
    for item in selected:
        metrics = item["metrics"]
        rules.append({
            "timeframe": role,
            "direction": item["direction"],
            "conditions": [item["condition"]],
            "coverage": metrics["coverage"],
            "directional_edge": metrics["directional_edge"],
            "mcc": metrics["mcc"],
            "stability": 1.0,
            "stability_score": 1.0,
            "skill": metrics["directional_edge"],
            "threshold_quantile": item["quantile"],
            "threshold": item["threshold"],
            "_key": _candidate_key(
                item["feature"], item["direction"], item["operator"], item["quantile"]
            ),
            "_test_mask": item["test_mask"],
            "_test_metrics": metrics,
        })
    active = np.column_stack([rule["_test_mask"] for rule in rules]).astype(bool)
    if role == "mwc":
        upstream_values = (
            upstream_test
            if upstream_test is not None
            else np.full(len(test), np.nan)
        )
        for index, rule in enumerate(rules):
            if rule["direction"] == "long":
                active[:, index] &= (
                    np.isfinite(upstream_values)
                    & (upstream_values >= profile.support_threshold)
                )
            else:
                active[:, index] &= (
                    np.isfinite(upstream_values)
                    & (upstream_values <= -profile.support_threshold)
                )
    weights = compute_rule_weights(rules)
    directions = [rule["direction"] for rule in rules]
    direction_score, strength_score = compute_ensemble_direction_and_strength(
        active, directions, weights
    )
    return rules, direction_score, strength_score, theta


def discover_directional_layer(
    raw_df: pd.DataFrame,
    *,
    role: str,
    folds: Sequence[TemporalFold],
    upstream_oof_scores: pd.DataFrame | None = None,
) -> LayerDiscoveryResult:
    """Discover a layer from purged OOF folds and return frozen rules."""
    role = str(role).strip().lower()
    if role not in {"hwc", "mwc"}:
        raise ValueError("discover_directional_layer supports only 'hwc' and 'mwc'")
    timeframe = 240 if role == "hwc" else 60
    profile = get_rule_search_profile(role)
    bars = _build_layer_frame(raw_df, timeframe, role)
    if bars.empty:
        return LayerDiscoveryResult(
            role, [], pd.DataFrame(), bars, _schema_hash(bars), _frame_hash(bars), {}, 1.0, {}
        )
    upstream = _align_upstream_scores(
        bars,
        upstream_oof_scores,
        240 if role == "mwc" else timeframe,
    )
    fold_rules: dict[int, list[dict[str, Any]]] = {}
    fold_metrics: dict[tuple, list[dict[str, Any]]] = {}
    oof_parts: list[pd.DataFrame] = []
    theta_per_fold: dict[str, float] = {}
    for index, fold in enumerate(folds):
        if fold.is_seed:
            continue
        train_mask = (bars["datetime"] >= fold.train_start) & (bars["datetime"] < fold.test_start)
        test_mask = (bars["datetime"] >= fold.test_start) & (
            bars["datetime"] <= fold.test_end if index == len(folds) - 1 else bars["datetime"] < fold.test_end
        )
        train = bars.loc[train_mask].copy()
        test = bars.loc[test_mask].copy()
        if train.empty or test.empty:
            continue
        # Purge rows whose forward movement reaches the test start.  The layer
        # labels are already materialized on the full bar frame.
        purge = pd.Timedelta(minutes=profile.forward_horizon_bars * timeframe)
        train = train.loc[train["datetime"] < fold.test_start - purge]
        if train.empty:
            continue
        upstream_train = upstream[train.index.to_numpy()] if upstream is not None else None
        upstream_test = upstream[test.index.to_numpy()] if upstream is not None else None
        selected, direction_score, strength_score, theta = _fit_fold_candidates(
            train,
            test,
            role=role,
            upstream_train=upstream_train,
            upstream_test=upstream_test,
        )
        theta_per_fold[str(fold.fold_id)] = float(theta)
        fold_rules[fold.fold_id] = selected
        for rule in selected:
            fold_metrics.setdefault(rule["_key"], []).append(rule["_test_metrics"])
        oof_parts.append(pd.DataFrame({
            "datetime": test["datetime"].to_numpy(),
            "symbol": test["symbol"].to_numpy(),
            "direction_score": direction_score,
            "strength_score": strength_score,
            "fold_id": int(fold.fold_id),
            "is_seed": False,
        }))

    full_moves = bars["_move"].to_numpy(dtype=float)
    theta_final = fit_directional_threshold(full_moves, profile.quantile)
    frozen_rules: list[dict[str, Any]] = []
    feature_schema_hash = _schema_hash(bars.drop(columns=["_move"], errors="ignore"))
    data_hash = _frame_hash(bars.drop(columns=["_move"], errors="ignore"))
    for key, metrics_list in fold_metrics.items():
        feature, direction, operator, quantile = key
        if len(metrics_list) < 1:
            continue
        full_values = pd.to_numeric(bars[feature], errors="coerce")
        threshold = float(full_values[np.isfinite(full_values)].quantile(quantile))
        if not np.isfinite(threshold):
            continue
        mean_edge = float(np.mean([m["test_directional_edge"] for m in metrics_list]))
        mean_mcc = float(np.mean([m["test_mcc"] for m in metrics_list]))
        mcc_std = float(np.std([m["test_mcc"] for m in metrics_list]))
        mean_coverage = float(np.mean([m["test_coverage"] for m in metrics_list]))
        if mean_edge <= 0.0 or mean_mcc <= 0.0:
            continue
        condition = f"[{feature}] {operator} {threshold:.12g}"
        frozen_rules.append({
            "timeframe": role,
            "direction": direction,
            "conditions": [condition],
            "coverage": mean_coverage,
            "directional_edge": mean_edge,
            "mcc": mean_mcc,
            "stability": float(np.clip(1.0 - mcc_std, 0.0, 1.0)),
            "stability_score": float(np.clip(1.0 - mcc_std, 0.0, 1.0)),
            "skill": mean_edge,
            "oof_metrics": {
                "directional_edge": mean_edge,
                "mcc": mean_mcc,
                "coverage": mean_coverage,
                "stability": float(np.clip(1.0 - mcc_std, 0.0, 1.0)),
                "false_confirmation_penalty": float(
                    max(0.0, -mean_edge) if role == "mwc" else 0.0
                ),
                "fold_metrics": metrics_list,
                "fold_support": len(metrics_list),
            },
            "threshold_quantile": float(quantile),
            "data_hash": data_hash,
            "feature_schema_hash": feature_schema_hash,
        })
    max_rules = max(2, profile.max_conditions * 4)
    pareto_rules = _directional_pareto_front(frozen_rules)
    frozen_rules = _select_balanced_candidates(pareto_rules, max_rules)
    ensemble_weights = compute_rule_weights(frozen_rules)
    for rule, weight in zip(frozen_rules, ensemble_weights):
        rule["ensemble_weight"] = float(weight)
    oof_scores = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return LayerDiscoveryResult(
        timeframe=role,
        rules=frozen_rules,
        oof_scores=oof_scores,
        bars=bars,
        feature_schema_hash=feature_schema_hash,
        data_hash=data_hash,
        theta_per_oof_fold=theta_per_fold,
        theta_final_train=float(theta_final),
        fold_rules=fold_rules,
        search_metadata={
            "algorithm": "bounded_directional_threshold_search",
            "objective_names": [
                "directional_edge",
                "mcc",
                "stability",
                "coverage_quality",
                "false_confirmation_penalty",
            ],
            "pareto_front_size": len(pareto_rules),
            "ensemble_weight_formula": "max(0, directional_edge) * max(0, stability)",
            "plateau_metric": "directional_pareto_quality",
            "plateau_restarts": 0,
        },
        oof_score_hash=_frame_hash(oof_scores),
    )

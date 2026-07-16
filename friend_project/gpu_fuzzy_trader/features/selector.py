
from __future__ import annotations

import json
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from gpu_fuzzy_trader import config
from gpu_fuzzy_trader.features.detector import Feature_Detector

logger = logging.getLogger(__name__)


_LONG_PATH = os.path.join(config.OUTPUTS_DIR, "selected_features_long.json")
_SHORT_PATH = os.path.join(config.OUTPUTS_DIR, "selected_features_short.json")

_DIRECTION_PATHS = {
    "long": _LONG_PATH,
    "short": _SHORT_PATH,
}

_REQUIRED_KEYS = {"direction", "features"}
_REQUIRED_FEATURE_KEYS = {"name", "mode", "score"}
_VALID_DIRECTIONS = {"long", "short"}
_DISCRETE_FEATURE_MODES = frozenset({"binary", "ternary"})


def _reduce_overlap(
    ranked: dict[str, list[dict]],
    max_overlap_pct: float,
    top_k: int,
) -> dict[str, list[dict]]:
    """
    Cap long/short feature overlap and backfill each direction to top_k features.
    """
    long_feats = list(ranked.get("long", []))
    short_feats = list(ranked.get("short", []))
    long_names = {f["name"] for f in long_feats}
    short_names = {f["name"] for f in short_feats}
    shared = long_names & short_names
    max_shared = int(top_k * max_overlap_pct)

    if len(shared) > max_shared:
        long_scores = {f["name"]: f["score"] for f in long_feats}
        short_scores = {f["name"]: f["score"] for f in short_feats}
        shared_ranked = sorted(
            shared,
            key=lambda name: abs(
                long_scores.get(name, 0.0) - short_scores.get(name, 0.0)
            ),
        )
        to_remove = len(shared) - max_shared
        for name in shared_ranked[:to_remove]:
            if long_scores.get(name, 0.0) <= short_scores.get(name, 0.0):
                long_feats = [f for f in long_feats if f["name"] != name]
            else:
                short_feats = [f for f in short_feats if f["name"] != name]

    def _backfill(selected: list[dict], pool: list[dict]) -> list[dict]:
        chosen = {f["name"] for f in selected}
        for feat in pool:
            if len(selected) >= top_k:
                break
            if feat["name"] not in chosen:
                selected.append(feat)
                chosen.add(feat["name"])
        return selected[:top_k]

    return {
        "long": _backfill(long_feats, ranked.get("long", [])),
        "short": _backfill(short_feats, ranked.get("short", [])),
    }




class Feature_Selector:
    """Score and rank features separately for long and short directions."""


    def select_features(
        self, train_df: pd.DataFrame, direction: str
    ) -> list[dict]:
        """
        Select top features for the given direction.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split DataFrame (must include label columns and feature columns).
        direction : str
            "long" or "short".

        Returns
        -------
        list[dict]
            List of dicts: [{"name": str, "mode": str, "score": float}]
            Sorted by score descending, up to PHASE1_TOP_K_FEATURES entries.
        """
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

        logger.info("Phase 1 [%s]: starting feature selection", direction)

        exclude = (
            set(config.LABEL_COLUMNS)
            | set(config.META_COLUMNS)
            | set(config.INTERNAL_COLUMNS)
        )
        feature_cols = [
            c for c in train_df.columns
            if c not in exclude and not c.startswith("_")
        ]
        n_candidates = len(feature_cols)

        if not feature_cols:
            logger.warning(
                "Phase 1 [%s]: no candidate feature columns", direction)
            return []

        detector = Feature_Detector()
        feature_modes = detector.detect_all_modes(train_df, feature_cols)

        feature_cols = _remove_low_dispersion(
            train_df, feature_cols, config.PHASE1_DISPERSION_THRESHOLD
        )
        logger.info(
            "Phase 1 [%s]: %d candidates → %d after dispersion filter",
            direction, n_candidates, len(feature_cols),
        )

        if not feature_cols:
            logger.warning(
                "Phase 1 [%s]: all features removed by dispersion filter", direction)
            return []

        target = _build_target(train_df, direction)

        symbols = train_df["symbol"].unique() if "symbol" in train_df.columns else [None]
        n_symbols = len(symbols) if symbols[0] is not None else 1
        logger.info(
            "Phase 1 [%s]: scoring %d features across %d symbol(s) (MI) …",
            direction, len(feature_cols), n_symbols,
        )

        per_symbol_scores: dict[str, list[float]] = {col: [] for col in feature_cols}

        for sym in symbols:
            if sym is not None:
                mask = train_df["symbol"] == sym
                sym_df = train_df[mask]
                sym_target = target[mask]
            else:
                sym_df = train_df
                sym_target = target

            if sym_target.nunique() < 2:
                continue

            X = sym_df[feature_cols].values
            y = sym_target.values.astype(np.int32, copy=False)
            discrete_mask = _mutual_info_discrete_mask(
                feature_cols, feature_modes)

            try:
                scores = mutual_info_classif(
                    X,
                    y,
                    discrete_features=discrete_mask,
                    random_state=42,
                )
            except Exception:
                scores = np.zeros(len(feature_cols))

            for i, col in enumerate(feature_cols):
                per_symbol_scores[col].append(float(scores[i]))

        scored: list[dict] = []
        for col in feature_cols:
            sym_scores = per_symbol_scores[col]
            if not sym_scores:
                final_score = 0.0
            else:
                relevance = float(np.mean(sym_scores))
                stability = _compute_stability(sym_scores)
                final_score = relevance * stability

            sign_consistency = 0.0
            if getattr(config, "PHASE1_SIGN_CONSISTENCY_ENABLED", False):
                sign_consistency = _temporal_sign_consistency(
                    train_df, col, target, folds=int(config.PHASE1_SIGN_CONSISTENCY_FOLDS)
                )
                weight = float(config.PHASE1_SIGN_CONSISTENCY_WEIGHT)
                final_score *= max(0.0, (1.0 - weight) + weight * sign_consistency)

            scored.append({
                "name": col,
                "mode": feature_modes.get(col, "positive"),
                "score": final_score,
                "sign_consistency": sign_consistency,
            })

        n_before_redundancy = len(scored)
        scored = _remove_redundant_features(train_df, scored, threshold=0.95)
        logger.info(
            "Phase 1 [%s]: redundancy filter %d → %d features",
            direction, n_before_redundancy, len(scored),
        )

        scored.sort(key=lambda x: x["score"], reverse=True)
        candidate_k = config.PHASE1_TOP_K_FEATURES * 2
        selected = scored[:candidate_k]

        if selected:
            top = selected[0]
            logger.info(
                "Phase 1 [%s]: selected top %d (best: %s, score=%.4f)",
                direction, len(selected), top["name"], top["score"],
            )
        else:
            logger.warning("Phase 1 [%s]: no features selected", direction)

        return selected

    def run(self, train_df: pd.DataFrame) -> dict[str, list[dict]]:
        """
        Run feature selection for both directions.

        Parameters
        ----------
        train_df : pd.DataFrame
            Training split DataFrame.

        Returns
        -------
        dict[str, list[dict]]
            {"long": [...], "short": [...]}
            Also persists results to outputs/selected_features_{direction}.json.
        """
        ranked: dict[str, list[dict]] = {}
        for direction in ("long", "short"):
            ranked[direction] = self.select_features(train_df, direction)

        results = _reduce_overlap(
            ranked,
            config.PHASE1_MAX_FEATURE_OVERLAP,
            config.PHASE1_TOP_K_FEATURES,
        )

        shared = (
            {f["name"] for f in results["long"]}
            & {f["name"] for f in results["short"]}
        )
        logger.info(
            "Phase 1: overlap reduction complete — %d shared of %d max allowed",
            len(shared),
            int(config.PHASE1_TOP_K_FEATURES * config.PHASE1_MAX_FEATURE_OVERLAP),
        )

        for direction in ("long", "short"):
            features = results[direction]

            out_path = _DIRECTION_PATHS[direction]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            payload = {"direction": direction, "features": features}
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
            logger.info(
                "Phase 1 [%s]: saved %d features to %s",
                direction, len(features), out_path,
            )

        return results


    @staticmethod
    def load_and_validate(path: str) -> list[dict]:
        """
        Load and validate a feature selection JSON file.

        Parameters
        ----------
        path : str
            Path to the JSON file.

        Returns
        -------
        list[dict]
            The validated list of feature dicts.

        Raises
        ------
        ValueError
            If the file is missing, unreadable, or has an invalid schema.
        """
        if not os.path.exists(path):
            raise ValueError(f"Feature selection file not found: {path}")

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Feature selection file is unreadable or corrupted: {path}"
            ) from exc

        _validate_schema(data, path)
        return data["features"]

    @staticmethod
    def skip_if_valid() -> Optional[dict[str, list[dict]]]:
        """
        Check if output files exist and are valid.

        Returns
        -------
        dict[str, list[dict]] | None
            Loaded results if both files are valid, None if either is missing.

        Raises
        ------
        ValueError
            If a file exists but is corrupted or has an invalid schema.
        """
        long_exists = os.path.exists(_LONG_PATH)
        short_exists = os.path.exists(_SHORT_PATH)

        if not long_exists and not short_exists:
            return None

        results: dict[str, list[dict]] = {}

        for direction, path in _DIRECTION_PATHS.items():
            if not os.path.exists(path):
                return None
            features = Feature_Selector.load_and_validate(path)
            results[direction] = features

        return results




def _mutual_info_discrete_mask(
    feature_cols: list[str],
    feature_modes: dict[str, str],
) -> list[bool]:
    """
    Per-column ``discrete_features`` flags for ``mutual_info_classif``.

    Only binary/ternary columns are treated as categorical. Continuous modes
    (positive, signed, sparse_*) use k-NN MI and avoid sklearn's clustering
    metric warning for float-valued "labels".
    """
    return [
        feature_modes.get(col, "positive") in _DISCRETE_FEATURE_MODES
        for col in feature_cols
    ]


def _remove_low_dispersion(
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> list[str]:
    """
    Remove features where more than `threshold` fraction of values are identical.

    Parameters
    ----------
    df : pd.DataFrame
    feature_cols : list[str]
    threshold : float
        E.g. 0.95 means drop if >95% of values are the same.

    Returns
    -------
    list[str]
        Filtered feature column names.
    """
    kept = []
    for col in feature_cols:
        series = df[col]
        if len(series) == 0:
            continue
        top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq <= threshold:
            kept.append(col)
    return kept


def _build_target(df: pd.DataFrame, direction: str) -> pd.Series:
    """
    Build a binary success target for the given direction.

    Long success:
        label_max_288 >= label_open_next * (1 + PHASE2_TP/100)
        AND that max came BEFORE the SL was hit
        (max_before_min == 1 means max came first → TP first for long)

    Short success:
        label_min_288 <= label_open_next * (1 - PHASE2_TP/100)
        AND that min came BEFORE the max hit SL
        (max_before_min == 0 means min came first → TP first for short)

    Parameters
    ----------
    df : pd.DataFrame
    direction : str
        "long" or "short"

    Returns
    -------
    pd.Series
        Boolean series (0/1) indicating trade success.
    """
    tp = config.PHASE2_TP
    sl = config.PHASE2_SL

    open_next = df["label_open_next"]
    max_288 = df["label_max_288"]
    min_288 = df["label_min_288"]
    max_before_min = df["label_max_before_min"]

    tp_level_long = open_next * (1 + tp / 100)
    sl_level_long = open_next * (1 - sl / 100)
    tp_level_short = open_next * (1 - tp / 100)
    sl_level_short = open_next * (1 + sl / 100)

    if direction == "long":
        hit_tp = max_288 >= tp_level_long
        hit_sl = min_288 <= sl_level_long
        both_hit = hit_tp & hit_sl
        tp_first = both_hit & (max_before_min == 1)
        success = hit_tp & (~hit_sl | tp_first)
    else:         
        hit_tp = min_288 <= tp_level_short
        hit_sl = max_288 >= sl_level_short
        both_hit = hit_tp & hit_sl
        tp_first = both_hit & (max_before_min == 0)
        success = hit_tp & (~hit_sl | tp_first)

    return success.astype(int)


def _compute_stability(sym_scores: list[float]) -> float:
    """
    Compute cross-symbol stability score.

    stability = 1 - (std / mean)

    If mean is 0 or computation fails, returns 0.

    Parameters
    ----------
    sym_scores : list[float]
        Per-symbol MI scores for a single feature.

    Returns
    -------
    float
        Stability score in [−∞, 1]. Clipped to [0, 1] for practical use.
    """
    if len(sym_scores) < 2:
        return 1.0 if (sym_scores and sym_scores[0] > 0) else 0.0

    try:
        arr = np.array(sym_scores, dtype=float)
        mean_val = float(np.mean(arr))
        if mean_val == 0.0:
            return 0.0
        std_val = float(np.std(arr, ddof=0))
        stability = 1.0 - (std_val / mean_val)
        return float(np.clip(stability, 0.0, 1.0))
    except Exception:
        return 0.0


def _remove_redundant_features(
    df: pd.DataFrame,
    scored: list[dict],
    threshold: float = 0.95,
) -> list[dict]:
    """
    Within-mode redundancy removal.

    For each mode group, compute pairwise absolute correlation between features.
    If two features have |corr| > threshold, drop the one with the lower score.

    Parameters
    ----------
    df : pd.DataFrame
    scored : list[dict]
        List of {"name": str, "mode": str, "score": float}.
    threshold : float
        Correlation threshold above which a feature is considered redundant.

    Returns
    -------
    list[dict]
        Filtered list with redundant features removed.
    """
    mode_groups: dict[str, list[dict]] = {}
    for entry in scored:
        mode = entry["mode"]
        mode_groups.setdefault(mode, []).append(entry)

    kept: list[dict] = []

    for mode, group in mode_groups.items():
        if len(group) <= 1:
            kept.extend(group)
            continue

        group_sorted = sorted(group, key=lambda x: x["score"], reverse=True)
        cols = [e["name"] for e in group_sorted]

        try:
            corr_matrix = df[cols].corr().abs()
        except Exception:
            kept.extend(group_sorted)
            continue

        kept_in_group: list[dict] = []
        kept_cols: list[str] = []

        for entry in group_sorted:
            col = entry["name"]
            if not kept_cols:
                kept_in_group.append(entry)
                kept_cols.append(col)
                continue

            max_corr = corr_matrix.loc[col, kept_cols].max()
            if max_corr <= threshold:
                kept_in_group.append(entry)
                kept_cols.append(col)

        kept.extend(kept_in_group)

    return kept



def _temporal_sign_consistency(
    df: pd.DataFrame,
    feature_col: str,
    target: pd.Series,
    folds: int = 4,
) -> float:
    """Return how consistently a feature keeps the same correlation sign over time.

    Only the current training split is used.  The score is in [0, 1].  Features
    with no meaningful correlation in any fold receive 0.
    """
    if folds <= 1 or feature_col not in df.columns:
        return 0.0

    work = df[[feature_col]].copy()
    work["_target"] = target.to_numpy()
    if "datetime" in df.columns:
        work["_datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        work = work.sort_values("_datetime", kind="mergesort")
    else:
        work = work.reset_index(drop=True)

    work = work.reset_index(drop=True)
    n_rows = len(work)
    n_folds = max(1, min(int(folds), n_rows))
    boundaries = np.linspace(0, n_rows, n_folds + 1, dtype=int)
    parts = [
        work.iloc[int(a):int(b)].copy()
        for a, b in zip(boundaries[:-1], boundaries[1:])
        if int(b) > int(a)
    ]

    signs: list[int] = []
    min_abs = float(getattr(config, "PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR", 1e-5))

    for part in parts:
        if len(part) < 50 or part["_target"].nunique() < 2:
            continue
        x = pd.to_numeric(part[feature_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(part["_target"], errors="coerce")
        mask = x.notna() & y.notna()
        if int(mask.sum()) < 50 or x[mask].nunique() < 2:
            continue
        try:
            corr = float(np.corrcoef(x[mask].to_numpy(dtype=float), y[mask].to_numpy(dtype=float))[0, 1])
        except Exception:
            corr = 0.0
        if not np.isfinite(corr) or abs(corr) < min_abs:
            continue
        signs.append(1 if corr > 0 else -1)

    if not signs:
        return 0.0
    pos = signs.count(1)
    neg = signs.count(-1)
    return float(max(pos, neg) / len(signs))

def _validate_schema(data: object, path: str) -> None:
    """
    Validate the structure of a loaded feature selection JSON.

    Raises ValueError if the schema is invalid.
    """
    if not isinstance(data, dict):
        raise ValueError(
            f"Feature selection file must be a JSON object, got {type(data).__name__}: {path}"
        )

    missing_keys = _REQUIRED_KEYS - set(data.keys())
    if missing_keys:
        raise ValueError(
            f"Feature selection file missing required keys {missing_keys}: {path}"
        )

    if data["direction"] not in _VALID_DIRECTIONS:
        raise ValueError(
            f"Feature selection file has invalid direction {data['direction']!r}: {path}"
        )

    features = data["features"]
    if not isinstance(features, list):
        raise ValueError(
            f"Feature selection file 'features' must be a list: {path}"
        )

    for i, feat in enumerate(features):
        if not isinstance(feat, dict):
            raise ValueError(
                f"Feature selection file entry {i} must be a dict: {path}"
            )
        missing_feat_keys = _REQUIRED_FEATURE_KEYS - set(feat.keys())
        if missing_feat_keys:
            raise ValueError(
                f"Feature selection file entry {i} missing keys {missing_feat_keys}: {path}"
            )
        if not isinstance(feat["name"], str):
            raise ValueError(
                f"Feature selection file entry {i} 'name' must be a string: {path}"
            )
        if not isinstance(feat["mode"], str):
            raise ValueError(
                f"Feature selection file entry {i} 'mode' must be a string: {path}"
            )
        if not isinstance(feat["score"], (int, float)):
            raise ValueError(
                f"Feature selection file entry {i} 'score' must be a number: {path}"
            )

"""
selector.py — Feature_Selector (Phase 1)

Direction-specific feature scoring and ranking.

Algorithm:
  1. Exclude LABEL_COLUMNS and META_COLUMNS
  2. Detect feature modes (from training split only)
  3. Remove features where >95% of values are identical (near-zero dispersion)
  4. Build direction-specific binary success targets
  5. Score each feature per symbol using mutual information
  6. Compute cross-symbol stability score = 1 - (std / mean of per-symbol scores)
  7. Final score = relevance_score * stability_score
  8. Within-mode redundancy removal (pairwise correlation > 0.95)
  9. Select top PHASE1_TOP_K_FEATURES per direction
  10. Persist to outputs/selected_features_long.json and outputs/selected_features_short.json

Skip logic:
  - If output files exist and are valid JSON with required schema → skip (return loaded)
  - If files are missing → return None (need to run)
  - If files exist but are corrupted/invalid → raise ValueError immediately
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from gpu_fuzzy_trader import config
from gpu_fuzzy_trader.features.detector import Feature_Detector


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_LONG_PATH = os.path.join(config.OUTPUTS_DIR, "selected_features_long.json")
_SHORT_PATH = os.path.join(config.OUTPUTS_DIR, "selected_features_short.json")

_DIRECTION_PATHS = {
    "long": _LONG_PATH,
    "short": _SHORT_PATH,
}

# Required top-level keys in the output JSON
_REQUIRED_KEYS = {"direction", "features"}
# Required keys in each feature entry
_REQUIRED_FEATURE_KEYS = {"name", "mode", "score"}
_VALID_DIRECTIONS = {"long", "short"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Feature_Selector:
    """Score and rank features separately for long and short directions."""

    # ------------------------------------------------------------------
    # Core selection logic
    # ------------------------------------------------------------------

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

        # ----------------------------------------------------------------
        # Step 1: Identify candidate feature columns
        # ----------------------------------------------------------------
        exclude = set(config.LABEL_COLUMNS) | set(config.META_COLUMNS)
        feature_cols = [c for c in train_df.columns if c not in exclude]

        if not feature_cols:
            return []

        # ----------------------------------------------------------------
        # Step 2: Detect feature modes from training split only
        # ----------------------------------------------------------------
        detector = Feature_Detector()
        feature_modes = detector.detect_all_modes(train_df, feature_cols)

        # ----------------------------------------------------------------
        # Step 3: Remove near-zero dispersion features
        # ----------------------------------------------------------------
        feature_cols = _remove_low_dispersion(
            train_df, feature_cols, config.PHASE1_DISPERSION_THRESHOLD
        )

        if not feature_cols:
            return []

        # ----------------------------------------------------------------
        # Step 4: Build direction-specific binary success target
        # ----------------------------------------------------------------
        target = _build_target(train_df, direction)

        # ----------------------------------------------------------------
        # Step 5 & 6: Score per symbol, compute stability
        # ----------------------------------------------------------------
        symbols = train_df["symbol"].unique() if "symbol" in train_df.columns else [None]

        per_symbol_scores: dict[str, list[float]] = {col: [] for col in feature_cols}

        for sym in symbols:
            if sym is not None:
                mask = train_df["symbol"] == sym
                sym_df = train_df[mask]
                sym_target = target[mask]
            else:
                sym_df = train_df
                sym_target = target

            # Skip symbol if target has only one class
            if sym_target.nunique() < 2:
                continue

            X = sym_df[feature_cols].values
            y = sym_target.values

            try:
                scores = mutual_info_classif(
                    X, y, discrete_features=True, random_state=42
                )
            except Exception:
                scores = np.zeros(len(feature_cols))

            for i, col in enumerate(feature_cols):
                per_symbol_scores[col].append(float(scores[i]))

        # ----------------------------------------------------------------
        # Step 7: Final score = relevance * stability
        # ----------------------------------------------------------------
        scored: list[dict] = []
        for col in feature_cols:
            sym_scores = per_symbol_scores[col]
            if not sym_scores:
                final_score = 0.0
            else:
                relevance = float(np.mean(sym_scores))
                stability = _compute_stability(sym_scores)
                final_score = relevance * stability

            scored.append({
                "name": col,
                "mode": feature_modes.get(col, "positive"),
                "score": final_score,
            })

        # ----------------------------------------------------------------
        # Step 8: Within-mode redundancy removal (pairwise corr > 0.95)
        # ----------------------------------------------------------------
        scored = _remove_redundant_features(train_df, scored, threshold=0.95)

        # ----------------------------------------------------------------
        # Step 9: Select top K features
        # ----------------------------------------------------------------
        scored.sort(key=lambda x: x["score"], reverse=True)
        top_k = config.PHASE1_TOP_K_FEATURES
        selected = scored[:top_k]

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
        results: dict[str, list[dict]] = {}

        for direction in ("long", "short"):
            features = self.select_features(train_df, direction)
            results[direction] = features

            # Persist
            out_path = _DIRECTION_PATHS[direction]
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            payload = {"direction": direction, "features": features}
            with open(out_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)

        return results

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

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

        # If neither exists → need to run
        if not long_exists and not short_exists:
            return None

        # If one exists but not the other → validate the existing one
        # (it may be corrupted); if valid, still return None to force re-run
        # because both are required.
        results: dict[str, list[dict]] = {}

        for direction, path in _DIRECTION_PATHS.items():
            if not os.path.exists(path):
                # File missing → need to run (but don't fail)
                return None
            # File exists → validate (raises ValueError if corrupted)
            features = Feature_Selector.load_and_validate(path)
            results[direction] = features

        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


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
        # Fraction of the most common value
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
        # TP hit
        hit_tp = max_288 >= tp_level_long
        # SL hit
        hit_sl = min_288 <= sl_level_long
        # Both hit: max_before_min == 1 means max came first → TP first (success)
        both_hit = hit_tp & hit_sl
        tp_first = both_hit & (max_before_min == 1)
        # Success: TP hit and either SL not hit, or TP came first
        success = hit_tp & (~hit_sl | tp_first)
    else:  # short
        # TP hit (price dropped to TP level)
        hit_tp = min_288 <= tp_level_short
        # SL hit (price rose to SL level)
        hit_sl = max_288 >= sl_level_short
        # Both hit: max_before_min == 0 means min came first → TP first for short
        both_hit = hit_tp & hit_sl
        tp_first = both_hit & (max_before_min == 0)
        # Success: TP hit and either SL not hit, or TP came first
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
        # Only one symbol → no cross-symbol variance; treat as perfectly stable
        return 1.0 if (sym_scores and sym_scores[0] > 0) else 0.0

    try:
        arr = np.array(sym_scores, dtype=float)
        mean_val = float(np.mean(arr))
        if mean_val == 0.0:
            return 0.0
        std_val = float(np.std(arr, ddof=0))
        stability = 1.0 - (std_val / mean_val)
        # Clip to [0, 1] — negative stability means very high variance
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
    # Group by mode
    mode_groups: dict[str, list[dict]] = {}
    for entry in scored:
        mode = entry["mode"]
        mode_groups.setdefault(mode, []).append(entry)

    kept: list[dict] = []

    for mode, group in mode_groups.items():
        if len(group) <= 1:
            kept.extend(group)
            continue

        # Sort by score descending so we keep higher-scored features
        group_sorted = sorted(group, key=lambda x: x["score"], reverse=True)
        cols = [e["name"] for e in group_sorted]

        # Compute correlation matrix
        try:
            corr_matrix = df[cols].corr().abs()
        except Exception:
            kept.extend(group_sorted)
            continue

        # Greedy selection: keep a feature if it's not highly correlated
        # with any already-kept feature
        kept_in_group: list[dict] = []
        kept_cols: list[str] = []

        for entry in group_sorted:
            col = entry["name"]
            # Check correlation with all already-kept columns
            if not kept_cols:
                kept_in_group.append(entry)
                kept_cols.append(col)
                continue

            max_corr = corr_matrix.loc[col, kept_cols].max()
            if max_corr <= threshold:
                kept_in_group.append(entry)
                kept_cols.append(col)
            # else: drop this feature (redundant)

        kept.extend(kept_in_group)

    return kept


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

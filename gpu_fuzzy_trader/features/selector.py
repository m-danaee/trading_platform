"""
selector.py — Feature_Selector (Phase 1)

Direction-specific feature scoring and ranking.

Algorithm:
  1. Exclude LABEL_COLUMNS, META_COLUMNS, INTERNAL_COLUMNS, and ``_``-prefixed names
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
import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from gpu_fuzzy_trader import config
from gpu_fuzzy_trader.features.detector import Feature_Detector
from gpu_fuzzy_trader.features.regime_cluster import (
    RegimeBundle,
    fit_regime_labels,
    persist_regime_model,
)

logger = logging.getLogger(__name__)

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
# Modes with small integer codewords — use discrete MI; others are continuous.
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
    max_shared = int(top_k * max_overlap_pct)

    def _trim_shared(
        left_feats: list[dict],
        right_feats: list[dict],
    ) -> tuple[list[dict], list[dict]]:
        left_names = {f["name"] for f in left_feats}
        right_names = {f["name"] for f in right_feats}
        shared_names = left_names & right_names
        if len(shared_names) <= max_shared:
            return left_feats, right_feats

        left_scores = {f["name"]: f["score"] for f in left_feats}
        right_scores = {f["name"]: f["score"] for f in right_feats}
        shared_ranked = sorted(
            shared_names,
            key=lambda name: abs(
                left_scores.get(name, 0.0) - right_scores.get(name, 0.0)
            ),
        )
        to_remove = len(shared_ranked) - max_shared
        for name in shared_ranked[:to_remove]:
            if left_scores.get(name, 0.0) <= right_scores.get(name, 0.0):
                left_feats = [f for f in left_feats if f["name"] != name]
            else:
                right_feats = [f for f in right_feats if f["name"] != name]
        return left_feats, right_feats
    long_feats, short_feats = _trim_shared(long_feats, short_feats)

    def _backfill(
        selected: list[dict],
        pool: list[dict],
        other_selected: list[dict],
    ) -> list[dict]:
        chosen = {f["name"] for f in selected}
        other_names = {f["name"] for f in other_selected}
        for feat in pool:
            if len(selected) >= top_k:
                break
            name = feat["name"]
            if name in chosen:
                continue
            shared_now = len(chosen & other_names)
            if name in other_names and shared_now >= max_shared:
                continue
            selected.append(feat)
            chosen.add(name)
        return selected[:top_k]

    long_feats = _backfill(long_feats, ranked.get("long", []), short_feats)
    short_feats = _backfill(short_feats, ranked.get("short", []), long_feats)
    # Safety pass: backfill must not reintroduce overlap violations.
    long_feats, short_feats = _trim_shared(long_feats, short_feats)

    if len(long_feats) < top_k:
        long_feats = _backfill(long_feats, ranked.get("long", []), short_feats)
    if len(short_feats) < top_k:
        short_feats = _backfill(
            short_feats, ranked.get("short", []), long_feats)

    return {
        "long": long_feats[:top_k],
        "short": short_feats[:top_k],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Feature_Selector:
    """Score and rank features separately for long and short directions."""

    def __init__(self) -> None:
        self._regime_labels: Optional[pd.Series] = None
        self._regime_bundle: Optional[RegimeBundle] = None

    # ------------------------------------------------------------------
    # Core selection logic
    # ------------------------------------------------------------------

    def select_features(
        self,
        train_df: pd.DataFrame,
        direction: str,
        regime_labels: Optional[pd.Series] = None,
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

        # ----------------------------------------------------------------
        # Step 1: Identify candidate feature columns
        # ----------------------------------------------------------------
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
        logger.info(
            "Phase 1 [%s]: %d candidates → %d after dispersion filter",
            direction, n_candidates, len(feature_cols),
        )

        if not feature_cols:
            logger.warning(
                "Phase 1 [%s]: all features removed by dispersion filter", direction)
            return []

        # ----------------------------------------------------------------
        # Step 4: Build direction-specific binary success target
        # ----------------------------------------------------------------
        target = _build_target(train_df, direction)

        # ----------------------------------------------------------------
        # Step 5 & 6: Score per symbol, compute stability
        # ----------------------------------------------------------------
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

            # Skip symbol if target has only one class
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
                    random_state=config.get_seed(),
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
        # Step 7b: Stationarity filter (regime or chronological folds)
        # ----------------------------------------------------------------
        if config.PHASE1_STATIONARITY_FOLDS >= 2:
            stratify = config.PHASE1_STATIONARITY_STRATIFY.lower()
            rank_drift_max = config.PHASE1_STATIONARITY_RANK_DRIFT_MAX
            fold_scores: dict[str, list[float]]
            labels = regime_labels if regime_labels is not None else self._regime_labels

            if stratify == "regime":
                fold_scores = _compute_regime_stationarity_scores(
                    train_df,
                    feature_cols,
                    feature_modes,
                    target,
                    labels,
                    config.PHASE1_STATIONARITY_FOLDS,
                    config.PHASE1_REGIME_MIN_SAMPLES,
                )
                n_valid = _count_valid_stationarity_folds(fold_scores)
                if n_valid < 2:
                    logger.warning(
                        "Phase 1 [%s]: regime stationarity insufficient (%d folds); "
                        "falling back to chronological",
                        direction, n_valid,
                    )
                    fold_scores = _compute_chronological_stationarity_scores(
                        train_df,
                        feature_cols,
                        feature_modes,
                        target,
                        config.PHASE1_STATIONARITY_FOLDS,
                    )
            else:
                fold_scores = _compute_chronological_stationarity_scores(
                    train_df,
                    feature_cols,
                    feature_modes,
                    target,
                    config.PHASE1_STATIONARITY_FOLDS,
                )

            survivors = _stationarity_filter(
                fold_scores,
                config.PHASE1_STATIONARITY_CV_MAX,
                rank_drift_max,
            )
            n_before_stationarity = len(scored)
            scored = [f for f in scored if f["name"] in survivors]
            logger.info(
                "Phase 1 [%s]: stationarity filter (%s) %d → %d features "
                "(cv_max=%.2f, rank_drift_max=%d)",
                direction, stratify, n_before_stationarity, len(scored),
                config.PHASE1_STATIONARITY_CV_MAX,
                rank_drift_max,
            )
            if not scored:
                logger.warning(
                    "Phase 1 [%s]: stationarity filter removed everything; "
                    "consider relaxing PHASE1_STATIONARITY_CV_MAX or "
                    "PHASE1_STATIONARITY_RANK_DRIFT_MAX",
                    direction,
                )

        # ----------------------------------------------------------------
        # Step 8: Within-mode redundancy removal (pairwise corr > 0.95)
        # ----------------------------------------------------------------
        n_before_redundancy = len(scored)
        scored = _remove_redundant_features(train_df, scored, threshold=0.95)
        logger.info(
            "Phase 1 [%s]: redundancy filter %d → %d features",
            direction, n_before_redundancy, len(scored),
        )

        # ----------------------------------------------------------------
        # Step 9: Select top K features
        # ----------------------------------------------------------------
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
        self._regime_labels = None
        self._regime_bundle = None
        if (
            config.PHASE1_STATIONARITY_FOLDS >= 2
            and config.PHASE1_STATIONARITY_STRATIFY.lower() == "regime"
        ):
            fit_result = fit_regime_labels(
                train_df,
                n_clusters=config.PHASE1_STATIONARITY_FOLDS,
            )
            if fit_result is not None:
                self._regime_labels, self._regime_bundle = fit_result
                try:
                    persist_regime_model(
                        config.PHASE1_REGIME_MODEL_PATH,
                        self._regime_bundle,
                    )
                except Exception as exc:
                    logger.warning(
                        "Phase 1: could not persist regime model: %s", exc,
                    )

        ranked: dict[str, list[dict]] = {}
        for direction in ("long", "short"):
            ranked[direction] = self.select_features(
                train_df, direction, regime_labels=self._regime_labels,
            )

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
        if len(shared) > int(config.PHASE1_TOP_K_FEATURES * config.PHASE1_MAX_FEATURE_OVERLAP):
            logger.warning(
                "Phase 1: overlap cap violation detected after reduction "
                "(shared=%d, cap=%d)",
                len(shared),
                int(config.PHASE1_TOP_K_FEATURES *
                    config.PHASE1_MAX_FEATURE_OVERLAP),
            )

        for direction in ("long", "short"):
            features = results[direction]

            # Persist
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
        # Fraction of the most common value
        top_freq = series.value_counts(normalize=True, dropna=False).iloc[0]
        if top_freq <= threshold:
            kept.append(col)
    return kept


def _build_target(df: pd.DataFrame, direction: str) -> pd.Series:
    """
    Build a direction-specific target signal.

    Default (PHASE1_ASYMMETRIC_TARGET=True):
        Signed expected-PnL surrogate (3-class, then encoded to int target).
        Long:
            +1  TP hit and TP came first (clear win)
            -1  SL hit and SL came first (clear loss)
             0  neither side cleanly resolved
        Short: mirror logic with min_288 instead of max_288.
        This gives long and short genuinely different targets, addressing the
        "long and short feature lists nearly identical" issue observed.

    Legacy (PHASE1_ASYMMETRIC_TARGET=False):
        Binary success target — kept for ablation comparisons.

    Parameters
    ----------
    df : pd.DataFrame
    direction : str
        "long" or "short"

    Returns
    -------
    pd.Series
        Integer series with discrete classes (target encoding).
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
        sl_first = both_hit & (max_before_min == 0)
        clear_win = hit_tp & ~hit_sl
        clear_loss = hit_sl & ~hit_tp
        win = clear_win | tp_first
        loss = clear_loss | sl_first
    else:  # short
        hit_tp = min_288 <= tp_level_short
        hit_sl = max_288 >= sl_level_short
        both_hit = hit_tp & hit_sl
        # For shorts the min must come first to take TP first.
        tp_first = both_hit & (max_before_min == 0)
        sl_first = both_hit & (max_before_min == 1)
        clear_win = hit_tp & ~hit_sl
        clear_loss = hit_sl & ~hit_tp
        win = clear_win | tp_first
        loss = clear_loss | sl_first

    if config.PHASE1_ASYMMETRIC_TARGET:
        # Encode -1/0/+1 → 0/1/2 (mutual_info_classif requires non-negative ints).
        target = pd.Series(np.zeros(len(df), dtype=np.int8), index=df.index)
        target[win] = 2
        target[loss] = 0
        target[~win & ~loss] = 1
        return target

    # Legacy binary path: 1 if win, else 0.
    return win.astype(int)


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


def _count_valid_stationarity_folds(
    fold_scores: dict[str, list[float]],
) -> int:
    if not fold_scores:
        return 0
    return max(len(scores) for scores in fold_scores.values())


def _mi_scores_for_mask(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_modes: dict[str, str],
    target: pd.Series,
    mask: np.ndarray,
    min_samples: int,
) -> Optional[list[float]]:
    """Compute MI vector for rows where *mask* is True; None if insufficient."""
    if mask.sum() < min_samples:
        return None
    Xf = df.loc[mask, feature_cols].values
    yf = target.loc[mask].values.astype(np.int32, copy=False)
    if len(np.unique(yf)) < 2:
        return None
    discrete_mask = _mutual_info_discrete_mask(feature_cols, feature_modes)
    try:
        scores = mutual_info_classif(
            Xf, yf, discrete_features=discrete_mask, random_state=config.get_seed(),
        )
    except Exception:
        scores = np.zeros(len(feature_cols))
    return [float(s) for s in scores]


def _compute_chronological_stationarity_scores(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_modes: dict[str, str],
    target: pd.Series,
    n_folds: int,
) -> dict[str, list[float]]:
    """
    Per-fold MI by splitting *df* chronologically into *n_folds*.
    """
    if n_folds < 2:
        return {col: [] for col in feature_cols}

    if "datetime" in df.columns and "symbol" in df.columns:
        order = df.sort_values(["symbol", "datetime"]).index
        df_ordered = df.loc[order]
        target_ordered = target.loc[order]
    else:
        df_ordered = df
        target_ordered = target

    fold_scores: dict[str, list[float]] = {col: [] for col in feature_cols}
    n = len(df_ordered)
    min_samples = config.PHASE1_REGIME_MIN_SAMPLES
    if n < n_folds * min_samples:
        return fold_scores

    boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
    for f in range(n_folds):
        lo, hi = int(boundaries[f]), int(boundaries[f + 1])
        mask = np.zeros(n, dtype=bool)
        mask[lo:hi] = True
        scores = _mi_scores_for_mask(
            df_ordered,
            feature_cols,
            feature_modes,
            target_ordered,
            mask,
            min_samples,
        )
        if scores is None:
            continue
        for i, col in enumerate(feature_cols):
            fold_scores[col].append(scores[i])

    return fold_scores


def _compute_regime_stationarity_scores(
    df: pd.DataFrame,
    feature_cols: list[str],
    feature_modes: dict[str, str],
    target: pd.Series,
    regime_labels: Optional[pd.Series],
    n_regimes: int,
    min_samples: int,
) -> dict[str, list[float]]:
    """
    Per-regime MI using pre-fitted row labels (train only).
    """
    fold_scores: dict[str, list[float]] = {col: [] for col in feature_cols}
    if regime_labels is None or n_regimes < 2:
        return fold_scores

    labels = regime_labels.reindex(df.index)
    for regime_id in range(n_regimes):
        mask = (labels == regime_id).values
        scores = _mi_scores_for_mask(
            df, feature_cols, feature_modes, target, mask, min_samples,
        )
        if scores is None:
            continue
        for i, col in enumerate(feature_cols):
            fold_scores[col].append(scores[i])

    return fold_scores


# Backward-compatible alias for tests/imports
_compute_stationarity_scores = _compute_chronological_stationarity_scores


def _stationarity_filter(
    fold_scores: dict[str, list[float]],
    cv_max: float,
    rank_drift_max: int,
) -> set[str]:
    """
    Return the set of feature names that PASS both stationarity checks.

    A feature passes when:
      1. The coefficient of variation (std/mean) of its per-fold MI score is
         at or below *cv_max*.
      2. Its rank (descending by score) does not shift by more than
         *rank_drift_max* positions across folds (max - min rank).
    """
    feature_names = list(fold_scores.keys())
    if not feature_names:
        return set()

    n_folds = max(len(scores) for scores in fold_scores.values()) if fold_scores else 0
    if n_folds < 2:
        return set(feature_names)

    # CV check
    survivors_cv: set[str] = set()
    for col, scores in fold_scores.items():
        if not scores:
            continue
        arr = np.array(scores, dtype=float)
        mean_val = float(np.mean(arr))
        if mean_val <= 0.0:
            continue
        std_val = float(np.std(arr, ddof=0))
        cv = std_val / mean_val
        if cv <= cv_max:
            survivors_cv.add(col)

    # Rank drift check
    ranks_per_fold: dict[int, dict[str, int]] = {}
    for f in range(n_folds):
        fold_pairs = []
        for col, scores in fold_scores.items():
            if f < len(scores):
                fold_pairs.append((col, scores[f]))
        fold_pairs.sort(key=lambda x: x[1], reverse=True)
        ranks_per_fold[f] = {col: r for r, (col, _) in enumerate(fold_pairs)}

    survivors_rank: set[str] = set()
    for col in feature_names:
        ranks = [
            ranks_per_fold[f][col]
            for f in range(n_folds)
            if col in ranks_per_fold[f]
        ]
        if len(ranks) < 2:
            continue
        if (max(ranks) - min(ranks)) <= rank_drift_max:
            survivors_rank.add(col)

    return survivors_cv & survivors_rank


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

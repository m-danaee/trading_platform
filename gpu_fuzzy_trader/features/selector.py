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
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import spearmanr


from gpu_fuzzy_trader import config
from gpu_fuzzy_trader.features.detector import Feature_Detector

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


@dataclass
class Phase1SharedContext:
    """Direction-independent Phase 1 state computed once per train_df."""

    feature_cols: list[str]
    feature_modes: dict[str, str]
    targets: dict[str, pd.Series]
    feature_array: np.ndarray | None = None
    symbol_masks: dict[str, np.ndarray] | None = None


def build_phase1_shared_context(
    train_df: pd.DataFrame,
) -> Phase1SharedContext:
    """Precompute candidates, modes, and long/short targets in one pass."""
    exclude = (
        set(config.LABEL_COLUMNS)
        | set(config.META_COLUMNS)
        | set(config.INTERNAL_COLUMNS)
    )
    feature_cols = [
        c for c in train_df.columns
        if c not in exclude and not c.startswith("_")
    ]
    detector = Feature_Detector()
    feature_modes = detector.detect_all_modes(train_df, feature_cols)
    feature_cols = _remove_low_dispersion(
        train_df, feature_cols, config.PHASE1_DISPERSION_THRESHOLD,
    )
    targets = {
        direction: _build_target(train_df, direction)
        for direction in _VALID_DIRECTIONS
    }
    feature_array = (
        train_df[feature_cols].to_numpy(copy=False)
        if feature_cols else np.empty((len(train_df), 0))
    )
    return Phase1SharedContext(
        feature_cols=feature_cols,
        feature_modes=feature_modes,
        targets=targets,
        feature_array=feature_array,
        symbol_masks=_build_symbol_masks(train_df),
    )


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


def _spearman(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    if mask.sum() < 2:
        return float("nan")
    if np.std(a[mask].values) == 0.0 or np.std(b[mask].values) == 0.0:
        return float("nan")
    corr, _ = spearmanr(a[mask].values, b[mask].values)
    return float(corr)


def _get_spearman_folds(df: pd.DataFrame, n_folds: int) -> list[pd.DataFrame]:
    if "symbol" not in df.columns:
        n = len(df)
        boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
        return [df.iloc[boundaries[i]:boundaries[i+1]] for i in range(n_folds)]
    
    sort_col = "datetime" if "datetime" in df.columns else None
    per_sym_folds = {}
    for symbol, group in df.groupby("symbol", sort=True):
        g = group.sort_values(sort_col) if sort_col else group
        g = g.reset_index(drop=True)
        n = len(g)
        boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
        per_sym_folds[symbol] = [g.iloc[boundaries[i]:boundaries[i+1]] for i in range(n_folds)]
        
    folds = []
    for i in range(n_folds):
        parts = [per_sym_folds[sym][i] for sym in per_sym_folds if i < len(per_sym_folds[sym])]
        if parts:
            folds.append(pd.concat(parts, ignore_index=True))
    return folds


def _check_spearman_sign_consistency(
    df: pd.DataFrame,
    feature_cols: list[str],
    n_folds: int,
    min_folds: int,
    val_df: pd.DataFrame | None = None,
    min_abs_corr: float | None = None,
) -> set[str]:
    """
    Keep features whose Spearman sign vs ``label_close_288`` is stable across folds.

    Only correlations with ``|rho| >= min_abs_corr`` participate in the sign-flip
    check so near-zero noise does not blacklist features.
    """
    if min_abs_corr is None:
        min_abs_corr = float(config.PHASE1_SIGN_CONSISTENCY_MIN_ABS_CORR)

    folds = _get_spearman_folds(df, n_folds)
    label_col = "label_close_288"
    if label_col not in df.columns:
        return set(feature_cols)

    stable_features = set()
    for col in feature_cols:
        corrs = []
        for fold in folds:
            if col not in fold.columns:
                continue
            corr = _spearman(fold[col], fold[label_col])
            if not np.isnan(corr):
                corrs.append(corr)
        if len(corrs) < min_folds:
            continue

        significant = [c for c in corrs if abs(c) >= min_abs_corr]
        if not significant:
            stable_features.add(col)
            continue

        has_pos = any(c > 0 for c in significant)
        has_neg = any(c < 0 for c in significant)
        if has_pos and has_neg:
            logger.info(
                "Blacklisting non-stationary feature %s: Spearman signs across "
                "folds (|rho|>=%.3f): %s (all folds: %s)",
                col, min_abs_corr, significant, corrs,
            )
        else:
            stable_features.add(col)
    return stable_features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Feature_Selector:
    """Score and rank features separately for long and short directions."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Core selection logic
    # ------------------------------------------------------------------

    def select_features(
        self,
        train_df: pd.DataFrame,
        direction: str,
        shared: Phase1SharedContext | None = None,
        val_df: pd.DataFrame | None = None,
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

        if shared is not None:
            feature_cols = list(shared.feature_cols)
            feature_modes = shared.feature_modes
            target = shared.targets[direction]
            n_candidates = len(feature_cols)
            logger.info(
                "Phase 1 [%s]: using shared context (%d features after dispersion)",
                direction, len(feature_cols),
            )
        else:
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
                train_df, feature_cols, config.PHASE1_DISPERSION_THRESHOLD,
            )
            logger.info(
                "Phase 1 [%s]: %d candidates → %d after dispersion filter",
                direction, n_candidates, len(feature_cols),
            )
            if not feature_cols:
                logger.warning(
                    "Phase 1 [%s]: all features removed by dispersion filter",
                    direction,
                )
                return []
            target = _build_target(train_df, direction)

        if not feature_cols:
            logger.warning(
                "Phase 1 [%s]: no candidate feature columns", direction)
            return []

        if config.PHASE1_REQUIRE_SIGN_CONSISTENCY:
            n_folds = config.PHASE1_STATIONARITY_FOLDS
            min_folds = config.PHASE1_SIGN_CONSISTENCY_MIN_FOLDS
            stable_cols = _check_spearman_sign_consistency(
                train_df, feature_cols, n_folds, min_folds, val_df=val_df
            )
            logger.info(
                "Phase 1 [%s]: sign consistency filter kept %d/%d candidate features",
                direction, len(stable_cols), len(feature_cols)
            )
            feature_cols = [c for c in feature_cols if c in stable_cols]

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
        if shared is not None and shared.feature_array is not None:
            feature_array = _align_feature_array(
                shared.feature_array,
                shared.feature_cols,
                feature_cols,
            )
        else:
            feature_array = train_df[feature_cols].to_numpy(copy=False)
        symbol_masks = (
            shared.symbol_masks
            if shared is not None and shared.symbol_masks is not None
            else _build_symbol_masks(train_df)
        )
        target_values = target.to_numpy(copy=False)
        discrete_mask = _mutual_info_discrete_mask(
            feature_cols, feature_modes)

        for sym in symbols:
            if sym is not None:
                mask = symbol_masks[str(sym)]
            else:
                mask = symbol_masks["__all__"]

            sym_target = target_values[mask]
            if len(np.unique(sym_target)) < 2:
                continue

            X = feature_array[mask]
            try:
                scores = mutual_info_classif(
                    X,
                    sym_target.astype(np.int32, copy=False),
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
        # Step 7b: Stationarity filter (chronological folds)
        # ----------------------------------------------------------------
        if config.PHASE1_STATIONARITY_FOLDS >= 2:
            rank_drift_max = config.PHASE1_STATIONARITY_RANK_DRIFT_MAX
            stationarity_feature_array = (
                _align_feature_array(
                    shared.feature_array,
                    shared.feature_cols,
                    feature_cols,
                )
                if shared is not None and shared.feature_array is not None
                else None
            )

            fold_scores = _compute_chronological_stationarity_scores(
                train_df,
                feature_cols,
                feature_modes,
                target,
                config.PHASE1_STATIONARITY_FOLDS,
                feature_array=stationarity_feature_array,
            )

            survivors = _stationarity_filter(
                fold_scores,
                config.PHASE1_STATIONARITY_CV_MAX,
                rank_drift_max,
            )
            n_before_stationarity = len(scored)
            scored = [f for f in scored if f["name"] in survivors]
            logger.info(
                "Phase 1 [%s]: stationarity filter (chronological) %d → %d features "
                "(cv_max=%.2f, rank_drift_max=%d)",
                direction, n_before_stationarity, len(scored),
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

    def run(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> dict[str, list[dict]]:
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
        phase1_disabled = bool(getattr(config, "PHASE1_DISABLED", False))
        if phase1_disabled:
            shared = build_phase1_shared_context(train_df)
            all_infos = [
                {
                    "name": col,
                    "mode": shared.feature_modes.get(col, "positive"),
                    "score": 0.0,
                }
                for col in shared.feature_cols
            ]
            results = {"long": list(all_infos), "short": list(all_infos)}
            _persist_phase1_results(results, phase1_disabled=True)
            logger.info(
                "Phase 1 [disabled]: using all %d features (post-dispersion) "
                "for both directions",
                len(all_infos),
            )
            return results

        shared = build_phase1_shared_context(
            train_df,
        )
        logger.info(
            "Phase 1: shared preprocessing — %d features after dispersion",
            len(shared.feature_cols),
        )

        ranked: dict[str, list[dict]] = {}
        for direction in ("long", "short"):
            ranked[direction] = self.select_features(
                train_df,
                direction,
                shared=shared,
                val_df=val_df,
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
            _persist_phase1_results(
                {direction: features},
                phase1_disabled=False,
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
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                raise ValueError(
                    f"Feature selection file is unreadable or corrupted: {path}"
                ) from exc
            _validate_schema(data, path)
            cached_disabled = bool(data.get("phase1_disabled", False))
            current_disabled = bool(getattr(config, "PHASE1_DISABLED", False))
            if cached_disabled != current_disabled:
                logger.info(
                    "Phase 1: cached output was built with PHASE1_DISABLED=%s, "
                    "current config is %s; forcing re-run",
                    cached_disabled,
                    current_disabled,
                )
                return None
            results[direction] = data["features"]

        return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _persist_phase1_results(
    results: dict[str, list[dict]],
    *,
    phase1_disabled: bool,
) -> None:
    """Write Phase 1 feature JSON for each direction in *results*."""
    for direction, features in results.items():
        out_path = _DIRECTION_PATHS[direction]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        payload = {
            "direction": direction,
            "features": features,
            "phase1_disabled": bool(phase1_disabled),
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        logger.info(
            "Phase 1 [%s]: saved %d features to %s",
            direction,
            len(features),
            out_path,
        )


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


def _build_symbol_masks(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Precompute boolean row masks per symbol (or one mask for single-series data)."""
    if "symbol" not in df.columns:
        return {"__all__": np.ones(len(df), dtype=bool)}
    codes, uniques = pd.factorize(df["symbol"], sort=False)
    return {
        str(sym): (codes == idx)
        for idx, sym in enumerate(uniques)
    }


def _align_feature_array(
    feature_array: np.ndarray | None,
    source_cols: list[str],
    selected_cols: list[str],
) -> np.ndarray | None:
    """Slice precomputed matrix columns when *selected_cols* is a subset."""
    if feature_array is None:
        return None
    if source_cols == selected_cols:
        return feature_array
    col_to_idx = {name: idx for idx, name in enumerate(source_cols)}
    indices = [col_to_idx[col] for col in selected_cols]
    return feature_array[:, indices]


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
    if not feature_cols:
        return []
    n_rows = len(df)
    if n_rows == 0:
        return []
    matrix = df[feature_cols].to_numpy(copy=False)
    kept: list[str] = []
    for idx, col in enumerate(feature_cols):
        col_vals = matrix[:, idx]
        if col_vals.size == 0:
            continue
        _, counts = np.unique(col_vals, return_counts=True)
        if float(counts.max()) / float(n_rows) <= threshold:
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


def _mi_scores_for_mask(
    feature_array: np.ndarray,
    feature_cols: list[str],
    feature_modes: dict[str, str],
    target_values: np.ndarray,
    mask: np.ndarray,
    min_samples: int,
) -> Optional[list[float]]:
    """Compute MI vector for rows where *mask* is True; None if insufficient."""
    if mask.sum() < min_samples:
        return None
    Xf = feature_array[mask]
    yf = target_values[mask].astype(np.int32, copy=False)
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
    *,
    feature_array: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """
    Per-fold MI by splitting *df* chronologically into *n_folds*.
    """
    if n_folds < 2:
        return {col: [] for col in feature_cols}

    if "datetime" in df.columns and "symbol" in df.columns:
        order = df.sort_values(["symbol", "datetime"]).index
        row_idx = df.index.get_indexer(order)
        target_values = target.loc[order].to_numpy(copy=False)
        if feature_array is None:
            feature_array = df.loc[order, feature_cols].to_numpy(copy=False)
        else:
            feature_array = feature_array[row_idx]
    else:
        target_values = target.to_numpy(copy=False)
        if feature_array is None:
            feature_array = df[feature_cols].to_numpy(copy=False)

    fold_scores: dict[str, list[float]] = {col: [] for col in feature_cols}
    n = len(target_values)
    min_samples = 100
    if n < n_folds * min_samples:
        return fold_scores

    boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
    for f in range(n_folds):
        lo, hi = int(boundaries[f]), int(boundaries[f + 1])
        mask = np.zeros(n, dtype=bool)
        mask[lo:hi] = True
        scores = _mi_scores_for_mask(
            feature_array,
            feature_cols,
            feature_modes,
            target_values,
            mask,
            min_samples,
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

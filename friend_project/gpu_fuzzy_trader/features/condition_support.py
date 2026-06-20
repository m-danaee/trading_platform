
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import _apply_dynamic_rule
from gpu_fuzzy_trader.features.encoder import encode_condition, get_dont_care

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConditionSupportSummary:
    rows: int
    dead: int
    ultra_rare: int
    rare: int
    good: int
    broad: int
    very_broad: int


def _support_class(support_pct: float) -> str:
    if support_pct <= 0.0:
        return "dead"
    if support_pct < _cfg.CONDITION_SUPPORT_ULTRA_RARE_PCT:
        return "ultra_rare"
    if support_pct < _cfg.CONDITION_SUPPORT_RARE_PCT:
        return "rare"
    if support_pct > _cfg.CONDITION_SUPPORT_VERY_BROAD_PCT:
        return "very_broad"
    if support_pct > _cfg.CONDITION_SUPPORT_BROAD_PCT:
        return "broad"
    return "good"


def _base_weight(cls: str) -> float:
    if cls == "dead":
        return 0.0
    if cls == "ultra_rare":
        return 0.0 if _cfg.CONDITION_SUPPORT_DROP_ULTRA_RARE else 0.08
    if cls == "rare":
        return _cfg.CONDITION_SUPPORT_RARE_WEIGHT
    if cls == "broad":
        return _cfg.CONDITION_SUPPORT_BROAD_WEIGHT
    if cls == "very_broad":
        return _cfg.CONDITION_SUPPORT_VERY_BROAD_WEIGHT
    return 1.0


def _window_ids(df: pd.DataFrame, window_days: int) -> pd.Series:
    if "datetime" not in df.columns:
        chunk = max(1, len(df) // 8)
        return pd.Series(np.arange(len(df)) // chunk, index=df.index)
    dt = pd.to_datetime(df["datetime"], errors="coerce")
    start = dt.min()
    if pd.isna(start):
        chunk = max(1, len(df) // 8)
        return pd.Series(np.arange(len(df)) // chunk, index=df.index)
    return ((dt - start).dt.total_seconds() // (window_days * 86400)).astype(int)


def build_condition_support_table(
    train_df: pd.DataFrame,
    feature_infos: list[dict],
) -> pd.DataFrame:
    """Return one row per possible fuzzy condition with support statistics."""
    n_rows = max(int(len(train_df)), 1)
    if not feature_infos:
        return pd.DataFrame()

    symbols = train_df["symbol"] if "symbol" in train_df.columns else pd.Series(0, index=train_df.index)
    unique_symbols = list(pd.unique(symbols))
    win_ids = _window_ids(train_df, _cfg.CONDITION_SUPPORT_WINDOW_DAYS)
    unique_windows = list(pd.unique(win_ids))

    rows: list[dict] = []
    for feature_idx, fi in enumerate(feature_infos):
        name = fi["name"]
        mode = fi["mode"]
        if name not in train_df.columns:
            logger.warning("Support analysis: missing feature column %s", name)
            continue
        dc = int(get_dont_care(mode))
        for gene in range(dc):
            condition = encode_condition(name, gene, mode)
            try:
                mask = pd.Series(_apply_dynamic_rule(train_df, condition), index=train_df.index)
            except Exception as exc:
                logger.debug("Support analysis failed for %s: %s", condition, exc)
                mask = pd.Series(False, index=train_df.index)

            support_count = int(mask.sum())
            support_pct = float(support_count / n_rows)
            cls = _support_class(support_pct)

            per_symbol = []
            for sym in unique_symbols:
                sym_mask = symbols == sym
                denom = int(sym_mask.sum())
                if denom <= 0:
                    continue
                per_symbol.append(float(mask[sym_mask].sum() / denom))
            if per_symbol:
                min_symbol_support = float(np.min(per_symbol))
                max_symbol_support = float(np.max(per_symbol))
                symbol_support_std = float(np.std(per_symbol))
                active_symbol_count = int(np.sum(np.asarray(per_symbol) > 0.0))
                symbol_stability = active_symbol_count / max(len(per_symbol), 1)
            else:
                min_symbol_support = max_symbol_support = symbol_support_std = symbol_stability = 0.0
                active_symbol_count = 0

            per_window = []
            for w in unique_windows:
                w_mask = win_ids == w
                denom = int(w_mask.sum())
                if denom <= 0:
                    continue
                per_window.append(float(mask[w_mask].sum() / denom))
            if per_window:
                active_window_count = int(np.sum(np.asarray(per_window) > 0.0))
                window_stability = active_window_count / max(len(per_window), 1)
                min_window_support = float(np.min(per_window))
            else:
                active_window_count = 0
                window_stability = 0.0
                min_window_support = 0.0

            weight = _base_weight(cls)
            weight *= max(_cfg.CONDITION_SUPPORT_MIN_STABILITY_WEIGHT, symbol_stability)
            weight *= max(_cfg.CONDITION_SUPPORT_MIN_STABILITY_WEIGHT, window_stability)

            rows.append({
                "feature_idx": feature_idx,
                "feature": name,
                "mode": mode,
                "gene": gene,
                "condition": condition,
                "support_count": support_count,
                "support_pct": support_pct,
                "support_class": cls,
                "active_symbol_count": active_symbol_count,
                "min_symbol_support_pct": min_symbol_support,
                "max_symbol_support_pct": max_symbol_support,
                "symbol_support_std": symbol_support_std,
                "symbol_stability": symbol_stability,
                "active_window_count": active_window_count,
                "min_window_support_pct": min_window_support,
                "window_stability": window_stability,
                "sample_weight": float(weight),
            })

    return pd.DataFrame(rows)


def attach_condition_support_weights(
    feature_infos: list[dict],
    support_table: pd.DataFrame,
) -> list[dict]:
    """Return feature_infos copy with per-gene condition support weights."""
    out: list[dict] = [dict(fi) for fi in feature_infos]
    if support_table.empty:
        return out

    for idx, fi in enumerate(out):
        dc = int(get_dont_care(fi["mode"]))
        weights = np.ones(dc, dtype=np.float64)
        part = support_table[support_table["feature_idx"] == idx]
        for _, row in part.iterrows():
            gene = int(row["gene"])
            if 0 <= gene < dc:
                weights[gene] = float(row.get("sample_weight", 1.0))
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.ones(dc, dtype=np.float64)
        weights = weights / weights.sum()
        fi["condition_sample_weights"] = weights.tolist()
    return out


def summarise_support_table(support_table: pd.DataFrame) -> ConditionSupportSummary:
    if support_table.empty:
        return ConditionSupportSummary(0, 0, 0, 0, 0, 0, 0)
    counts = support_table["support_class"].value_counts().to_dict()
    return ConditionSupportSummary(
        rows=int(len(support_table)),
        dead=int(counts.get("dead", 0)),
        ultra_rare=int(counts.get("ultra_rare", 0)),
        rare=int(counts.get("rare", 0)),
        good=int(counts.get("good", 0)),
        broad=int(counts.get("broad", 0)),
        very_broad=int(counts.get("very_broad", 0)),
    )


def write_condition_support_report(
    support_table: pd.DataFrame,
    direction: str,
    output_dir: str | None = None,
) -> str | None:
    if support_table.empty:
        return None
    root = Path(output_dir or _cfg.REPORTS_DIR)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"condition_support_{direction}.csv"
    support_table.to_csv(path, index=False)
    return str(path)

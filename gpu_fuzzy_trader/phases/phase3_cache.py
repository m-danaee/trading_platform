"""
phase3_cache.py — Precomputed masks and per-rule validation metrics for Phase 3.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import (
    _build_rule_signal_mask,
    _build_entries_from_rule_set,
)
from gpu_fuzzy_trader.phases.phase3_objectives import (
    conditions_key,
    min_per_symbol_trades_from_metrics,
)

logger = logging.getLogger(__name__)


def build_entries_from_masks(
    rule_set: list[dict],
    masks_by_key: dict[frozenset, np.ndarray],
    n_rows: int,
) -> list[dict]:
    """
    Priority-based entry list using precomputed per-rule signal masks.
    """
    if not rule_set:
        return []

    assigned_mask = np.zeros(n_rows, dtype=bool)
    entries: list[dict] = []

    for rule_idx, rule_entry in enumerate(rule_set, start=1):
        conditions = rule_entry.get("conditions", [])
        key = conditions_key(conditions)
        if key not in masks_by_key:
            raise KeyError(f"No cached mask for conditions key {key!r}")
        rule_signals = masks_by_key[key]

        tp = float(rule_entry["tp"])
        sl = float(rule_entry["sl"])
        capital_pct = float(rule_entry["capital_pct"])

        new_match_mask = rule_signals & (~assigned_mask)
        matched_indices = np.flatnonzero(new_match_mask)
        if len(matched_indices) == 0:
            continue

        assigned_mask[matched_indices] = True
        for idx in matched_indices:
            entries.append({
                "idx": int(idx),
                "rule_index": int(rule_idx),
                "tp": tp,
                "sl": sl,
                "capital_pct": capital_pct,
            })

    entries.sort(key=lambda x: x["idx"])
    return entries


@dataclass
class Phase3EvalCache:
    """
    Caches for Phase 3 search: signal masks (train/val) and per-rule val gate stats.
    """

    train_masks: dict[frozenset, np.ndarray] = field(default_factory=dict)
    val_masks: dict[frozenset, np.ndarray] = field(default_factory=dict)
    per_rule_min_val_trades: dict[frozenset, int] = field(default_factory=dict)
    n_rows_train: int = 0
    n_rows_val: int = 0

    def build_entries(
        self,
        rule_set: list[dict],
        split: str,
    ) -> list[dict]:
        if split == "train":
            masks = self.train_masks
            n_rows = self.n_rows_train
        elif split == "val":
            masks = self.val_masks
            n_rows = self.n_rows_val
        else:
            raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        return build_entries_from_masks(rule_set, masks, n_rows)


def _rule_to_engine_format(rule: dict) -> dict:
    return {
        "conditions": rule["conditions"],
        "tp": float(rule.get("tp", _cfg.PHASE2_TP)),
        "sl": float(rule.get("sl", _cfg.PHASE2_SL)),
        "capital_pct": float(rule.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
    }


def build_phase3_eval_cache(
    pool: list[dict],
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    val_engine,
) -> Phase3EvalCache:
    """
    Precompute train/val signal masks and per-rule validation gate statistics.
    """
    cache = Phase3EvalCache(
        n_rows_train=len(train_df),
        n_rows_val=len(val_df),
    )
    seen: set[frozenset] = set()

    for rule in pool:
        key = conditions_key(rule["conditions"])
        if key in seen:
            continue
        seen.add(key)

        fmt = _rule_to_engine_format(rule)
        conditions = fmt["conditions"]

        cache.train_masks[key] = _build_rule_signal_mask(train_df, conditions)
        cache.val_masks[key] = _build_rule_signal_mask(val_df, conditions)

        try:
            metrics = val_engine.simulate_rule_set([fmt])
        except Exception as exc:
            logger.debug("per-rule val sim failed for cache: %s", exc)
            cache.per_rule_min_val_trades[key] = 0
        else:
            cache.per_rule_min_val_trades[key] = min_per_symbol_trades_from_metrics(
                metrics)

    logger.info(
        "Phase 3 eval cache: %d unique rules, train_rows=%d, val_rows=%d",
        len(seen), cache.n_rows_train, cache.n_rows_val,
    )
    return cache


def verify_mask_cache_parity(
    df: pd.DataFrame,
    rule_set: list[dict],
    cache_masks: dict[frozenset, np.ndarray],
) -> bool:
    """Return True if cached entries match direct entry build."""
    direct = _build_entries_from_rule_set(df, rule_set)
    cached = build_entries_from_masks(
        rule_set, cache_masks, len(df))
    if len(direct) != len(cached):
        return False
    for a, b in zip(direct, cached):
        if a != b:
            return False
    return True

"""Shared diagnostics for the mandatory direction-specific context contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from gpu_fuzzy_trader import config as _cfg


def context_coverage_for_direction(
    frame: pd.DataFrame,
    direction: str,
) -> dict[str, Any]:
    """Return permission, trigger, and conjunction coverage for *direction*."""
    permission = _cfg.context_permission_column(direction)
    trigger = _cfg.context_trigger_column(direction)
    missing = [
        column for column in (permission, trigger)
        if column not in frame.columns
    ]
    if missing:
        return {
            "eligible_rows": None,
            "total_rows": int(len(frame)),
            "coverage_pct": None,
            "permission_rows": None,
            "trigger_rows": None,
            "permission_only_rows": None,
            "trigger_only_rows": None,
            "neither_rows": None,
            "by_symbol": {},
            "by_symbol_detail": {},
            "missing_columns": missing,
        }

    permission_mask = frame[permission].to_numpy() == 1
    trigger_mask = frame[trigger].to_numpy() == 1
    eligible = permission_mask & trigger_mask
    permission_only = permission_mask & ~trigger_mask
    trigger_only = ~permission_mask & trigger_mask
    neither = ~permission_mask & ~trigger_mask

    by_symbol: dict[str, int] = {}
    by_symbol_detail: dict[str, dict[str, Any]] = {}
    if "symbol" in frame.columns:
        grouped = frame.groupby("symbol", sort=True, observed=False)
        for symbol, group in grouped:
            symbol_permission = group[permission].to_numpy() == 1
            symbol_trigger = group[trigger].to_numpy() == 1
            symbol_eligible = symbol_permission & symbol_trigger
            symbol_name = str(symbol)
            by_symbol[symbol_name] = int(symbol_eligible.sum())
            by_symbol_detail[symbol_name] = {
                "total_rows": int(len(group)),
                "permission_rows": int(symbol_permission.sum()),
                "trigger_rows": int(symbol_trigger.sum()),
                "eligible_rows": int(symbol_eligible.sum()),
                "permission_only_rows": int(
                    (symbol_permission & ~symbol_trigger).sum()
                ),
                "trigger_only_rows": int(
                    (~symbol_permission & symbol_trigger).sum()
                ),
                "coverage_pct": (
                    int(symbol_eligible.sum()) / max(len(group), 1) * 100.0
                ),
            }
    else:
        by_symbol["<all>"] = int(eligible.sum())
        by_symbol_detail["<all>"] = {
            "total_rows": int(len(frame)),
            "permission_rows": int(permission_mask.sum()),
            "trigger_rows": int(trigger_mask.sum()),
            "eligible_rows": int(eligible.sum()),
            "permission_only_rows": int(permission_only.sum()),
            "trigger_only_rows": int(trigger_only.sum()),
            "coverage_pct": (
                int(eligible.sum()) / max(len(frame), 1) * 100.0
            ),
        }

    eligible_rows = int(eligible.sum())
    return {
        "eligible_rows": eligible_rows,
        "total_rows": int(len(frame)),
        "coverage_pct": eligible_rows / max(len(frame), 1) * 100.0,
        "permission_rows": int(permission_mask.sum()),
        "trigger_rows": int(trigger_mask.sum()),
        "permission_only_rows": int(permission_only.sum()),
        "trigger_only_rows": int(trigger_only.sum()),
        "neither_rows": int(neither.sum()),
        # Keep this compact mapping stable for existing callers.
        "by_symbol": by_symbol,
        "by_symbol_detail": by_symbol_detail,
        "missing_columns": [],
    }


def context_coverage_report(
    frames: Mapping[str, pd.DataFrame],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Return coverage diagnostics for every named frame and direction."""
    return {
        split_name: {
            direction: context_coverage_for_direction(frame, direction)
            for direction in ("long", "short")
        }
        for split_name, frame in frames.items()
    }


def context_floor_failures(
    stats: Mapping[str, Any],
    *,
    support_floor: int | None = None,
    pool_floor: int | None = None,
    validation_floor: int | None = None,
) -> list[str]:
    """Return mathematically impossible trade-floor failures for coverage."""
    eligible = stats.get("eligible_rows")
    if eligible is None:
        return [
            "missing_context_columns:"
            + ",".join(str(value) for value in stats.get(
                "missing_columns", []
            ))
        ]

    eligible_rows = int(eligible)
    failures: list[str] = []
    if support_floor is not None and eligible_rows < int(support_floor):
        failures.append(
            f"eligible_rows={eligible_rows}<min_trade_support={int(support_floor)}"
        )
    if pool_floor is not None and eligible_rows < int(pool_floor):
        failures.append(
            f"eligible_rows={eligible_rows}<min_trade_pool_floor={int(pool_floor)}"
        )
    if validation_floor is not None and eligible_rows < int(validation_floor):
        failures.append(
            f"eligible_rows={eligible_rows}<validation_trade_floor={int(validation_floor)}"
        )
    return failures

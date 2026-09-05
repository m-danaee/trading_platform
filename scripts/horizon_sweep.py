#!/usr/bin/env python3
"""Run a train-only rule-feature catalog plus single-feature horizon diagnostic.

This is a research harness.  It changes the in-process horizon contract for
one run, rebuilds labels and exact barriers from raw OHLCV, and writes one
isolated result directory per horizon.  It does not run Phase 2, RB, or Phase
5 and does not modify the production configuration files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# Direct execution as ``.venv/bin/python scripts/horizon_sweep.py`` places the
# scripts directory first on sys.path.  Add the repository root explicitly so
# the harness uses the checkout's package rather than an installed copy.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.backtest.barrier import barrier_column_names
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.data import labels as labels_module
from gpu_fuzzy_trader.data import loader as loader_module
from gpu_fuzzy_trader.data.loader import Data_Loader
from gpu_fuzzy_trader.data.splitter import _holdout_embargo_split
from gpu_fuzzy_trader.features.encoder import encode_condition, get_dont_care
from gpu_fuzzy_trader.features.fuzzy_scaling import (
    apply_fuzzy_feature_scaling,
    fit_fuzzy_feature_scaling,
)
from gpu_fuzzy_trader.features.catalog import build_rule_feature_specs


LOGGER = logging.getLogger("horizon_sweep")
TP = 2.0
SL = 1.2
CAPITAL_PCT = 18.0


def _configure_horizon(horizon: int, output_dir: Path) -> None:
    """Patch imported constants for one isolated research process."""
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")

    # Several modules intentionally import these values for the production
    # contract.  Patch all of those bindings before loading any tape.
    cfg.MAX_HOLD_CANDLES = horizon
    cfg.TAIL_DROP_ROWS = horizon
    cfg.HOLDOUT_EMBARGO_CANDLES = horizon
    cfg.VALIDATION_PURGE_CANDLES = horizon
    cfg.PHASE2_TP = TP
    cfg.PHASE2_SL = SL
    cfg.RB_TP_GRID = (TP,)
    cfg.RB_SL_GRID = (SL,)
    cfg.OUTPUTS_DIR = str(output_dir)
    labels_module.TAIL_DROP_ROWS = horizon
    loader_module.TAIL_DROP_ROWS = horizon

def _split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the current 65/35 holdout without persisting shared artifacts."""
    train, validation = _holdout_embargo_split(frame)
    return train.reset_index(drop=True), validation.reset_index(drop=True)


def _label_balance(
    frame: pd.DataFrame,
    direction: str,
) -> dict[str, float | int]:
    """Return exact TP/SL/neutral label percentages for one frame."""
    ret_col, _ = barrier_column_names(direction, TP, SL)
    values = pd.to_numeric(frame[ret_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(values)
    values = values[valid]
    wins = np.isclose(values, TP, atol=1e-4)
    losses = np.isclose(values, -SL, atol=1e-4)
    neutral = ~(wins | losses)
    total = len(values)

    def pct(count: int) -> float:
        return float(count / total * 100.0) if total else 0.0

    return {
        "rows": int(total),
        "positive_pct": pct(int(wins.sum())),
        "negative_pct": pct(int(losses.sum())),
        "neutral_pct": pct(int(neutral.sum())),
    }


def _metric_snapshot(metrics: dict) -> dict[str, float | int | bool]:
    """Keep the metrics needed for the compact sweep report."""
    keys = (
        "total_return_pct",
        "profit_factor",
        "executed_trades",
        "raw_signal_count",
        "win_rate",
        "max_drawdown_pct",
        "time_closed_count",
        "account_ruined",
    )
    return {
        key: (
            bool(metrics.get(key, False))
            if key == "account_ruined"
            else float(metrics.get(key, 0.0))
            if key in {"total_return_pct", "profit_factor", "win_rate", "max_drawdown_pct"}
            else int(metrics.get(key, 0))
        )
        for key in keys
    }


def _state_records(
    frame: pd.DataFrame,
    feature: dict,
    direction: str,
    horizon: int,
) -> list[dict]:
    """Evaluate every fuzzy state of one feature on one split."""
    name = str(feature["name"])
    mode = str(feature["mode"])
    engine = CPUBacktestEngine(
        frame,
        {name: mode},
        direction,
        max_hold_candles=horizon,
        fee_pct=float(cfg.FEE_PCT),
        spread_bps=float(cfg.SPREAD_BPS),
        slippage_bps=float(cfg.SLIPPAGE_BPS),
        initial_capital=float(cfg.INITIAL_CAPITAL),
        leverage=float(cfg.LEVERAGE),
        max_total_exposure_pct=float(cfg.MAX_TOTAL_EXPOSURE_PCT),
        min_position_notional=float(cfg.MIN_POSITION_NOTIONAL),
    )
    records: list[dict] = []
    for gene in range(get_dont_care(mode)):
        condition = encode_condition(name, gene, mode)
        rule = [{
            "conditions": [condition],
            "tp": TP,
            "sl": SL,
            "capital_pct": CAPITAL_PCT,
        }]
        metrics = engine.simulate_rule_set(rule)
        records.append({
            "feature": name,
            "mode": mode,
            "gene": int(gene),
            "condition": condition,
            **_metric_snapshot(metrics),
        })
    return records


def _evaluate_direction(
    frames: dict[str, pd.DataFrame],
    feature_specs: list[dict],
    direction: str,
    horizon: int,
) -> dict:
    """Evaluate every catalog feature state without ranking or selection."""
    candidates: list[dict] = []
    for feature in feature_specs:
        name = str(feature["name"])
        state_rows: dict[str, dict] = {}
        for split_name, frame in frames.items():
            rows = _state_records(frame, feature, direction, horizon)
            for row in rows:
                state_rows.setdefault(int(row["gene"]), {})[split_name] = row
        for gene, split_rows in state_rows.items():
            if set(split_rows) != set(frames):
                continue
            candidates.append({
                "feature": name,
                "mode": str(feature["mode"]),
                "gene": int(gene),
                "condition": split_rows["train"]["condition"],
                "train": split_rows["train"],
                "validation": split_rows["validation"],
                "test": split_rows["test"],
            })
    candidates = [
        candidate
        for candidate in candidates
        if int(candidate["train"]["executed_trades"]) > 0
    ]
    if not candidates:
        return {
            "direction": direction,
            "rule_feature_count": len(feature_specs),
            "candidates": 0,
            "status": "no_evaluable_feature_states",
        }

    return {
        "direction": direction,
        "rule_feature_count": len(feature_specs),
        "feature_state_count": len(candidates),
        "feature_states": candidates,
        "status": "reported",
    }


def run_horizon(horizon: int, root: Path, train_path: Path, test_path: Path) -> dict:
    """Run one complete horizon diagnostic."""
    started = time.monotonic()
    horizon_dir = root / f"H{int(horizon)}"
    horizon_dir.mkdir(parents=True, exist_ok=True)
    _configure_horizon(int(horizon), horizon_dir)

    LOGGER.info("H=%d: loading raw train tape", horizon)
    loader = Data_Loader()
    train_full = loader.load_dataset(
        str(train_path),
        drop_tail=True,
        include_barrier_outcomes=True,
        require_context=False,
    )
    LOGGER.info("H=%d: loading raw test tape", horizon)
    test = loader.load_dataset(
        str(test_path),
        drop_tail=True,
        include_barrier_outcomes=True,
        require_context=False,
    )

    train, validation = _split_frame(train_full)
    scaling = fit_fuzzy_feature_scaling(train)
    for frame in (train, validation, test):
        apply_fuzzy_feature_scaling(frame, scaling)

    LOGGER.info(
        "H=%d: rows train=%d validation=%d test=%d; building rule catalog",
        horizon, len(train), len(validation), len(test),
    )
    feature_specs = build_rule_feature_specs(train)
    frames = {"train": train, "validation": validation, "test": test}

    direction_reports: dict[str, dict] = {}
    for direction in ("long", "short"):
        direction_reports[direction] = _evaluate_direction(
            frames, feature_specs, direction, int(horizon),
        )
        LOGGER.info(
            "H=%d %s: evaluated %d single-feature states",
            horizon,
            direction,
            direction_reports[direction].get("feature_state_count", 0),
        )

    report = {
        "status": "ok",
        "horizon_candles": int(horizon),
        "horizon_hours_at_15m": float(horizon * 0.25),
        "protocol": {
            "train_path": str(train_path),
            "test_path": str(test_path),
            "holdout_train_fraction": float(cfg.HOLDOUT_TRAIN_FRACTION),
            "embargo_candles": int(cfg.HOLDOUT_EMBARGO_CANDLES),
            "validation_is_full_post_embargo": True,
            "rule_catalog_uses_validation": False,
            "rule_catalog_uses_target_ranking": False,
            "phase2_run": False,
            "tp_pct": TP,
            "sl_pct": SL,
            "capital_pct": CAPITAL_PCT,
            "fee_pct": float(cfg.FEE_PCT),
            "spread_bps": float(cfg.SPREAD_BPS),
            "slippage_bps": float(cfg.SLIPPAGE_BPS),
            "initial_capital": float(cfg.INITIAL_CAPITAL),
            "scaling": "fit on train split, apply unchanged to validation/test",
        },
        "rows": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "label_balance": {
            direction: {
                split_name: _label_balance(frame, direction)
                for split_name, frame in frames.items()
            }
            for direction in ("long", "short")
        },
        "rule_features": [dict(item) for item in feature_specs],
        "directions": direction_reports,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    (horizon_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return report


def _compact_row(report: dict, direction: str) -> dict:
    direction_report = report.get("directions", {}).get(direction, {})
    return {
        "horizon": report.get("horizon_candles"),
        "direction": direction,
        "rule_feature_count": direction_report.get("rule_feature_count", 0),
        "feature_state_count": direction_report.get("feature_state_count", 0),
        "status": direction_report.get("status", "unknown"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--horizons", nargs="+", type=int, default=[8, 16, 32, 48, 96],
    )
    parser.add_argument(
        "--output-dir", default="outputs/horizon_sweep_20260905",
    )
    parser.add_argument("--train", default="data/train_new.csv")
    parser.add_argument("--test", default="data/test_new.csv")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    train_path = Path(args.train)
    test_path = Path(args.test)
    reports: list[dict] = []
    for horizon in args.horizons:
        LOGGER.info("===== H=%d (%0.2f hours) =====", horizon, horizon * 0.25)
        try:
            reports.append(run_horizon(horizon, root, train_path, test_path))
        except Exception as exc:  # keep completed horizons auditable
            LOGGER.exception("H=%d failed", horizon)
            reports.append({
                "status": "error",
                "horizon_candles": int(horizon),
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    compact = [
        row
        for report in reports
        if report.get("status") == "ok"
        for row in (_compact_row(report, "long"), _compact_row(report, "short"))
    ]
    (root / "summary.json").write_text(
        json.dumps({"reports": reports, "compact": compact}, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

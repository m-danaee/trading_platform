
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator
from gpu_fuzzy_trader.evaluation.internal_score import evaluate_strategy_file_internal
from gpu_fuzzy_trader.rb_governor import evaluate_strategy_file_governor, update_global_best, update_global_bank_and_compose
from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df


def _direction_for_run(i: int, start: str) -> str:
    if start not in ("long", "short"):
        raise ValueError("start direction must be 'long' or 'short'")
    if i % 2 == 0:
        return start
    return "short" if start == "long" else "long"


def _append_summary(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _load_internal_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not Path(_cfg.TRAIN_75_PATH).exists() or not Path(_cfg.VALIDATION_25_PATH).exists():
        from gpu_fuzzy_trader.data.loader import Data_Loader
        from gpu_fuzzy_trader.data.splitter import Data_Splitter
        full = Data_Loader().load_dataset(_cfg.TRAIN_CSV_PATH)
        return Data_Splitter().split_and_persist(full)
    return (
        downcast_numeric_df(pd.read_parquet(_cfg.TRAIN_75_PATH)),
        downcast_numeric_df(pd.read_parquet(_cfg.VALIDATION_25_PATH)),
    )


def _score_run(out_dir: Path, direction: str) -> dict[str, Any]:
    strategy_path = out_dir / f"{direction}.json"
    if not strategy_path.exists():
        raise FileNotFoundError(f"Strategy file not found: {strategy_path}")
    train_df, val_df = _load_internal_splits()
    if bool(getattr(_cfg, "RB_ENGINE_ENABLED", False)):
        score_payload = evaluate_strategy_file_governor(train_df, val_df, strategy_path)
        score_name = f"rb_score_{direction}.json"
    else:
        score_payload = evaluate_strategy_file_internal(train_df, val_df, strategy_path)
        score_name = f"internal_v4_score_{direction}.json"
    score_path = out_dir / score_name
    with score_path.open("w", encoding="utf-8") as fh:
        json.dump(score_payload, fh, indent=2, default=str)
    return score_payload


def run_auto_search(
    runs: int | None,
    output_root: str,
    start_direction: str = "long",
    resume: bool = False,
    hours: float | None = None,
    run_final_test: bool = False,
) -> Path:
    """Run alternating directional searches and return summary CSV path."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    summary_path = root / "auto_search_results.csv"
    max_runs = int(runs) if runs is not None else 10**9
    if max_runs <= 0:
        raise ValueError("runs must be positive")
    deadline = None
    if hours is not None and hours > 0:
        deadline = time.monotonic() + float(hours) * 3600.0

    run_idx = 0
    while run_idx < max_runs:
        if deadline is not None and time.monotonic() >= deadline:
            print("Auto-search time budget reached.")
            break

        run_no = run_idx + 1
        direction = _direction_for_run(run_idx, start_direction)
        out_dir = root / f"run_{run_no:03d}_{direction}"
        print("=" * 80)
        print(f"AUTO SEARCH RUN {run_no} | direction={direction} | output={out_dir}")
        print("=" * 80)

        orchestrator = Pipeline_Orchestrator(
            output_dir=str(out_dir),
            direction=direction,
            run_final_test=run_final_test,
        )
        started = datetime.now(timezone.utc).isoformat()
        status = "ok"
        error = ""
        score_payload: dict[str, Any] = {}
        try:
            orchestrator.run(force=not resume)
            score_payload = _score_run(out_dir, direction)
        except Exception as exc:
            status = "failed"
            error = repr(exc)
            print(f"Run {run_no} failed: {error}")

        finished = datetime.now(timezone.utc).isoformat()
        valid = score_payload.get("valid_metrics", {}) if score_payload else {}
        test = score_payload.get("test_metrics", {}) if score_payload else {}
        if bool(getattr(_cfg, "RB_ENGINE_ENABLED", False)) and score_payload.get("valid_metrics"):
            valid = score_payload.get("valid_metrics", {})
        train = score_payload.get("train_metrics", {}) if score_payload else {}
        fold = score_payload.get("fold_summary", {}) if score_payload else {}
        monthly = score_payload.get("monthly_summary", {}) if score_payload else {}
        row = {
            "run_no": run_no,
            "direction": direction,
            "status": status,
            "started": started,
            "finished": finished,
            "output_dir": str(out_dir),
            "internal_score": score_payload.get("internal_score", ""),
            "train_return_pct": train.get("total_return_pct", ""),
            "valid_return_pct": valid.get("total_return_pct", ""),
            "valid_profit_factor": valid.get("profit_factor", ""),
            "valid_max_drawdown_pct": valid.get("max_drawdown_pct", ""),
            "valid_win_rate": valid.get("win_rate", ""),
            "valid_executed_trades": valid.get("executed_trades", ""),
            "test_return_pct": test.get("total_return_pct", ""),
            "test_profit_factor": test.get("profit_factor", ""),
            "test_max_drawdown_pct": test.get("max_drawdown_pct", ""),
            "test_win_rate": test.get("win_rate", ""),
            "test_executed_trades": test.get("executed_trades", ""),
            "worst_fold_return_pct": fold.get("worst_return_pct", ""),
            "worst_fold_profit_factor": fold.get("worst_profit_factor", ""),
            "worst_fold_drawdown_pct": fold.get("worst_drawdown_pct", ""),
            "min_fold_trades": fold.get("min_trades", ""),
            "monthly_score": monthly.get("score", "") if isinstance(monthly, dict) else "",
            "monthly_profitable_ratio": monthly.get("profitable_ratio", "") if isinstance(monthly, dict) else "",
            "monthly_worst_return_pct": monthly.get("worst_return_pct", "") if isinstance(monthly, dict) else "",
            "monthly_worst_profit_factor": monthly.get("worst_profit_factor", "") if isinstance(monthly, dict) else "",
            "monthly_equity_slope": monthly.get("equity_slope", "") if isinstance(monthly, dict) else "",
            "error": error,
        }
        _append_summary(summary_path, row)
        if bool(getattr(_cfg, "RB_ENGINE_ENABLED", False)) and status == "ok":
            try:
                update_global_best(root, out_dir, direction, score_payload, run_no=run_no)
            except Exception as exc:
                print(f"Best-so-far update failed (non-fatal): {exc!r}")
            try:
                if bool(getattr(_cfg, "RB_GLOBAL_COMPOSE_AFTER_EACH_RUN", True)):
                    update_global_bank_and_compose(root, out_dir, direction, run_no=run_no)
            except Exception as exc:
                print(f"RB global bank/compose update failed (non-fatal): {exc!r}")
        print(f"Auto-search summary updated: {summary_path}")
        run_idx += 1

    return summary_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m gpu_fuzzy_trader.auto_search",
        description="Run alternating long/short auto-search ( may be rb/governor).",
    )
    parser.add_argument("--runs", type=int, default=None, help="Maximum number of directional runs.")
    parser.add_argument(
        "--hours",
        type=float,
        default=float(getattr(_cfg, "AUTO_SEARCH_HOURS", 24.0)),
        help="Wall-clock time budget in hours. Default: 24.",
    )
    parser.add_argument(
        "--output-root",
        default=_cfg.AUTO_SEARCH_OUTPUT_ROOT,
        help="Directory where per-run outputs and auto_search_results.csv are saved.",
    )
    parser.add_argument(
        "--start-direction",
        choices=("long", "short"),
        default=_cfg.AUTO_SEARCH_START_DIRECTION,
        help="Direction for run 1; subsequent runs alternate.",
    )
    parser.add_argument("--resume", action="store_true", help="Resume/skips valid cached outputs inside each run directory.")
    parser.add_argument(
        "--run-final-test",
        action="store_true",
        default=bool(getattr(_cfg, "AUTO_SEARCH_RUN_FINAL_TEST", False)),
        help="Also run final evaluation.",
    )
    args = parser.parse_args(argv)

    summary = run_auto_search(
        runs=args.runs,
        output_root=args.output_root,
        start_direction=args.start_direction,
        resume=args.resume,
        hours=args.hours,
        run_final_test=args.run_final_test,
    )
    print(f"\nDone. Summary: {summary}")


if __name__ == "__main__":
    main()

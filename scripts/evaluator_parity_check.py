#!/usr/bin/env python3
"""Verify outputs/long.json and outputs/short.json against evaluator_v5.ipynb."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_evaluator_namespace() -> dict:
    nb_path = ROOT / "evaluator_v5.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    ns: dict = {"__name__": "__main__"}
    skip_markers = (
        "Student Strategy Input",
        "Run Train and Test Evaluation",
        "student_strategy =",
    )
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if any(marker in src for marker in skip_markers):
            continue
        exec(compile(src, str(nb_path), "exec"), ns)
    return ns


def _evaluate(ns: dict, strategy_path: Path, split: str) -> dict:
    from gpu_fuzzy_trader import config as cfg

    if split == "test":
        eval_path = cfg.TEST_CSV_PATH
        label = "test"
    else:
        eval_path = cfg.TRAIN_CSV_PATH
        label = "train"

    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    _, df_eval, feature_cols, feature_modes, _ = ns["load_training_and_evaluation_data"](
        reference_schema_path=cfg.TRAIN_CSV_PATH,
        evaluation_file_path=eval_path,
    )
    metrics, _, _ = ns["evaluate_student_strategy_on_dataset"](
        df_eval=df_eval,
        feature_cols=feature_cols,
        feature_modes=feature_modes,
        student_strategy=strategy,
        dataset_name=label,
        return_logs=True,
    )
    return {
        "direction": strategy.get("direction", strategy_path.stem),
        "split": split,
        "total_return_pct": float(metrics["total_return_pct"]),
        "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
        "profit_factor": float(metrics["profit_factor"]),
        "win_rate": float(metrics["win_rate"]),
        "executed_trades": int(metrics["executed_trades"]),
        "account_ruined": bool(metrics["account_ruined"]),
    }


def main() -> int:
    ns = _load_evaluator_namespace()
    outputs = ROOT / "outputs"
    rows: list[dict] = []
    for direction in ("long", "short"):
        path = outputs / f"{direction}.json"
        if not path.exists():
            print(f"SKIP: missing {path}")
            continue
        for split in ("train", "test"):
            rows.append(_evaluate(ns, path, split))

    print("EVALUATOR_V5 PARITY (outputs/*.json)")
    print("-" * 72)
    for row in rows:
        print(
            f"{row['direction']:5s} {row['split']:5s}  "
            f"return={row['total_return_pct']:7.2f}%  "
            f"dd={row['max_drawdown_pct']:6.2f}%  "
            f"pf={row['profit_factor']:5.2f}  "
            f"trades={row['executed_trades']:4d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

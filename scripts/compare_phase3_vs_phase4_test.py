#!/usr/bin/env python3
"""Compare Phase 3 (pre-risk-opt) vs Phase 4 strategies on the test set via evaluator_v5."""

from __future__ import annotations
from gpu_fuzzy_trader import config as _cfg

import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# Phase 2 static risk used by Phase 3 (unchanged by Phase 4 for short in this run).
PHASE3_STATIC_TP = 2.0
PHASE3_STATIC_SL = 1.0
PHASE3_STATIC_CAPITAL_PCT = 30.0


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


def _reset_risk_to_phase3(strategy: dict) -> dict:
    out = deepcopy(strategy)
    for rule in out.get("rules_set", []):
        rule["tp"] = PHASE3_STATIC_TP
        rule["sl"] = PHASE3_STATIC_SL
        rule["capital_pct"] = PHASE3_STATIC_CAPITAL_PCT
    out.pop("risk_optimized", None)
    out.pop("deployment_accepted", None)
    out.pop("validation_gate", None)
    return out


def _evaluate(ns: dict, strategy: dict, label: str) -> dict:
    _, df_eval, feature_cols, feature_modes, _ = ns["load_training_and_evaluation_data"](
        reference_schema_path=_cfg.TRAIN_CSV_PATH,
        evaluation_file_path=_cfg.TEST_CSV_PATH,
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
        "label": label,
        "total_return_pct": float(metrics["total_return_pct"]),
        "max_drawdown_pct": float(metrics["max_drawdown_pct"]),
        "profit_factor": float(metrics["profit_factor"]),
        "win_rate": float(metrics["win_rate"]),
        "executed_trades": int(metrics["executed_trades"]),
        "account_ruined": bool(metrics["account_ruined"]),
    }


def main() -> int:
    outputs = ROOT / "outputs"
    results: list[dict] = []

    for direction in ("long", "short"):
        path = outputs / f"{direction}.json"
        if not path.is_file():
            print(f"Missing {path}", file=sys.stderr)
            continue
        phase4 = json.loads(path.read_text(encoding="utf-8"))
        phase3 = _reset_risk_to_phase3(phase4)

        print(f"\n{'=' * 72}")
        print(
            f"{direction.upper()} — rule conditions identical; only TP/SL/capital differ")
        print(
            f"Phase 3 static risk: tp={PHASE3_STATIC_TP} sl={PHASE3_STATIC_SL} capital={PHASE3_STATIC_CAPITAL_PCT}%")
        print(f"Phase 4 rules:")
        for i, rule in enumerate(phase4.get("rules_set", []), 1):
            print(
                f"  rule {i}: tp={rule['tp']} sl={rule['sl']} capital={rule['capital_pct']}%"
            )
        if direction == "short":
            print(
                "Note: pipeline log shows Phase 4 short made zero grid improvements (risk unchanged).")

    print("\nLoading evaluator_v5.ipynb engine …")
    ns = _load_evaluator_namespace()

    for direction in ("long", "short"):
        path = outputs / f"{direction}.json"
        phase4 = json.loads(path.read_text(encoding="utf-8"))
        phase3 = _reset_risk_to_phase3(phase4)
        for strategy, label in (
            (phase3, f"{direction} phase3-static-risk test"),
            (phase4, f"{direction} phase4-optimized test"),
        ):
            print(f"\nEvaluating {label} …")
            results.append(_evaluate(ns, strategy, label))

    print(f"\n{'=' * 72}")
    print("TEST SET COMPARISON (evaluator_v5)")
    print(f"{'=' * 72}")
    print(f"{'strategy':<32} {'return%':>10} {'PF':>8} {'trades':>8} {'mdd%':>8}")
    for row in results:
        print(
            f"{row['label']:<32} "
            f"{row['total_return_pct']:>10.2f} "
            f"{row['profit_factor']:>8.3f} "
            f"{row['executed_trades']:>8d} "
            f"{row['max_drawdown_pct']:>8.2f}"
        )

    out_path = outputs / "reports" / "phase3_vs_phase4_test_comparison.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

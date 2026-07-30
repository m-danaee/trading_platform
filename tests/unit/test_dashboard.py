"""Tests for the read-only artifact dashboard."""

from __future__ import annotations

import json
from pathlib import Path

from gpu_fuzzy_trader.dashboard import (
    build_dashboard_data,
    main,
    render_dashboard,
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_dashboard_handles_direction_reports_and_assets(tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    _write_json(
        reports / "rb_governor_long_report.json",
        {"direction": "long", "rb_score": 4.5, "selected_rules": 2},
    )
    _write_json(
        reports / "test_long_report.json",
        {"direction": "long", "total_return_pct": 3.2, "executed_trades": 8},
    )
    _write_json(
        reports / "generalization_diagnostics_long.json",
        {
            "split_metrics": {
                "train": {"total_return_pct": 5.0, "executed_trades": 20},
                "test": {"total_return_pct": 3.2, "executed_trades": 8},
            },
        },
    )
    _write_json(
        tmp_path / "long.json",
        {"direction": "long", "rules_set": [{"conditions": []}], "deployment_accepted": True},
    )
    (reports / "phase2_long_metrics.png").write_bytes(b"png")

    data = build_dashboard_data(tmp_path)

    assert data["accepted_directions"] == 1
    assert data["directions"]["long"]["status"] == "accepted"
    assert data["directions"]["long"]["split_metrics"]["train"]["executed_trades"] == 20
    assert "reports/phase2_long_metrics.png" in data["directions"]["long"]["assets"]
    assert data["directions"]["short"]["status"] == "missing"

    html = render_dashboard(data)
    assert "GPU Fuzzy Trader Dashboard" in html
    assert "phase2_long_metrics.png" in html
    assert "No config_audit.json found" in html


def test_dashboard_cli_writes_for_empty_or_partial_run(tmp_path: Path) -> None:
    assert main(["--output", str(tmp_path)]) == 0
    dashboard = tmp_path / "dashboard.html"
    assert dashboard.is_file()
    assert "No split metrics found" in dashboard.read_text(encoding="utf-8")


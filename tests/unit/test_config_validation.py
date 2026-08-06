"""Cross-parameter configuration and evaluator-parity tests."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from gpu_fuzzy_trader import config as cfg


def test_default_config_snapshot_is_valid_and_reports_effective_budgets() -> None:
    cfg.validate_config(n_rows=10_000, n_symbols=4)
    snapshot = cfg.effective_config_snapshot(n_rows=10_000, n_symbols=4)

    assert snapshot["phase2"]["stage_a_generations"] + snapshot["phase2"]["stage_b_generations"] == snapshot["phase2"]["generations"]
    assert snapshot["rb"]["max_total_capital"] == 100.0
    assert snapshot["rb"]["max_feasible_rules_at_min_capital"] >= cfg.RB_MAX_RULES
    assert snapshot["phase2"]["effective_min_profitable_symbols"] == 2
    assert snapshot["phase2"]["island_mode"] == cfg.PHASE2_ISLAND_MODE
    assert snapshot["phase2"]["effective_n_clusters"] == 4
    assert snapshot["gates"]["rb_min_valid_trades"] <= snapshot["gates"]["rb_ruleset_min_valid_trades"]


def test_data_dependent_symbol_requirements_fail_fast() -> None:
    with pytest.raises(cfg.ConfigError, match="PHASE2_MIN_PROFITABLE_SYMBOLS"):
        cfg.validate_config(n_rows=1000, n_symbols=1)


def test_debug_scope_caps_data_dependent_requirements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "DEBUG_SYMBOL_SCOPE_ENABLED", True)
    monkeypatch.setattr(cfg, "DEBUG_SYMBOL_COUNT", 1)
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_MODE", "cluster")
    monkeypatch.setattr(cfg, "PHASE2_N_CLUSTERS", 2)
    monkeypatch.setattr(cfg, "PHASE2_MIN_PROFITABLE_SYMBOLS", 2)
    monkeypatch.setattr(cfg, "RB_MIN_DISTINCT_SYMBOLS", 2)

    cfg.validate_config(n_rows=1000, n_symbols=1)
    snapshot = cfg.effective_config_snapshot(n_rows=1000, n_symbols=1)
    assert snapshot["phase2"]["effective_min_profitable_symbols"] == 1
    assert snapshot["phase2"]["effective_n_clusters"] == 1
    assert snapshot["rb"]["effective_min_distinct_symbols"] == 1


def test_cluster_and_rb_symbol_requirements_are_validated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "PHASE2_ISLAND_MODE", "cluster")
    monkeypatch.setattr(cfg, "PHASE2_MIN_PROFITABLE_SYMBOLS", 2)
    monkeypatch.setattr(cfg, "PHASE2_N_CLUSTERS", 3)
    with pytest.raises(cfg.ConfigError, match="PHASE2_N_CLUSTERS"):
        cfg.validate_config(n_rows=1000, n_symbols=2)

    monkeypatch.setattr(cfg, "PHASE2_N_CLUSTERS", 2)
    monkeypatch.setattr(cfg, "RB_MIN_DISTINCT_SYMBOLS", 3)
    with pytest.raises(cfg.ConfigError, match="RB_MIN_DISTINCT_SYMBOLS"):
        cfg.validate_config(n_rows=1000, n_symbols=2)


def test_evaluator_constants_match_read_only_notebook() -> None:
    notebook_path = Path(__file__).parents[2] / "evaluator_v5.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    expected = {
        "FEE_PCT": cfg.FEE_PCT,
        "MAX_HOLD_CANDLES": cfg.MAX_HOLD_CANDLES,
        "INITIAL_CAPITAL": cfg.INITIAL_CAPITAL,
        "LEVERAGE": cfg.LEVERAGE,
        "MAX_TOTAL_EXPOSURE_PCT": cfg.MAX_TOTAL_EXPOSURE_PCT,
        "MIN_POSITION_NOTIONAL": cfg.MIN_POSITION_NOTIONAL,
    }
    for name, value in expected.items():
        match = re.search(rf"(?m)^\s*{name}\s*=\s*([-+0-9.]+)", source)
        assert match, f"{name} is missing from evaluator_v5.ipynb"
        assert float(match.group(1)) == float(value), name


def test_evaluator_notebook_contains_context_and_dynamic_horizon_contract() -> None:
    notebook_path = Path(__file__).parents[2] / "evaluator_v5.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    assert "from gpu_fuzzy_trader.config import CONTEXT_COLUMNS" in source
    assert "c not in CONTEXT_COLUMNS" in source
    assert "Time_288" not in source
    assert 'f"Time_{self.max_hold_candles}"' in source


def test_exposure_cap_must_match_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "RB_MAX_TOTAL_CAPITAL", 150.0)
    with pytest.raises(cfg.ConfigError, match="must equal MAX_TOTAL_EXPOSURE_PCT"):
        cfg.validate_config()


def test_stage_budgets_must_sum_to_total(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "PHASE2_STAGE_B_GENERATIONS", 19)
    with pytest.raises(cfg.ConfigError, match="STAGE_A_GENERATIONS.*equal"):
        cfg.validate_config()


def test_capital_grid_must_make_maximum_rule_count_feasible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(cfg, "RB_CAPITAL_GRID", (7.5, 10.0, 18.0))
    with pytest.raises(cfg.ConfigError, match="minimum RB capital"):
        cfg.validate_config()


def test_threshold_ordering_is_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "RB_MIN_VALID_PF", 0.9)
    with pytest.raises(cfg.ConfigError, match="profit-factor floors"):
        cfg.validate_config()


def test_tail_embargo_and_horizon_must_be_equal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cfg, "HOLDOUT_EMBARGO_CANDLES", cfg.MAX_HOLD_CANDLES - 1)
    with pytest.raises(cfg.ConfigError, match="HOLDOUT_EMBARGO_CANDLES"):
        cfg.validate_config()


def test_audit_report_writes_the_effective_snapshot(tmp_path: Path) -> None:
    report_path = Path(cfg.write_config_audit_report(str(tmp_path), n_rows=1000, n_symbols=4))
    assert report_path.name == "config_audit.json"
    report = json.loads(report_path.read_text())
    assert report["evaluator_contract"]["max_total_exposure_pct"] == 100.0
    assert "gates" in report

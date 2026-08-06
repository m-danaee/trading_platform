"""
Unit tests for gpu_fuzzy_trader.phases.phase5_oos.OOS_Evaluator

Tests cover:
  - OOS_Evaluator constructor (default and custom test_csv_path)
  - load_strategies: missing files, valid files, invalid files, partial availability
  - prepare_test_data: delegates to Data_Loader correctly
  - _evaluate_strategy: zero-trade case, non-zero-trade case
  - _build_per_symbol_rows: correct structure
  - _save_report: creates file with correct keys
  - _save_per_symbol_csv: creates CSV with correct columns
  - run(): integration with tmp_path overrides
"""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.output.writer import Output_Writer
from gpu_fuzzy_trader.phases.phase5_oos import (
    OOS_Evaluator,
    _STRATEGY_PATHS,
    _REPORT_PATHS,
)


@pytest.fixture(autouse=True)
def _isolate_phase5_reporter_outputs(tmp_path, monkeypatch):
    """Keep Reporter plots/CSVs out of the checked-in outputs directory."""
    import gpu_fuzzy_trader.reporting.reporter as reporter_module

    monkeypatch.setattr(
        reporter_module, "_REPORTS_DIR", str(tmp_path / "reports"),
    )


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_rule_set(direction: str = "long", n_rules: int = 2) -> dict:
    """Create a minimal valid rule set dict."""
    rules = []
    for i in range(n_rules):
        rules.append({
            "conditions": [
                f"[feat_{i}] IS Very High",
                *_cfg.mandatory_context_conditions(direction),
            ],
            "tp": _cfg.PHASE2_TP,
            "sl": _cfg.PHASE2_SL,
            "capital_pct": _cfg.PHASE2_CAPITAL_PCT,
        })
    return {"direction": direction, "rules_set": rules}


def _write_rule_set(path: str, direction: str = "long", n_rules: int = 2) -> None:
    """Write a valid rule set JSON to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(_make_rule_set(direction, n_rules), fh)


def _write_selected_features(path: str, direction: str = "long", n_features: int = 2) -> None:
    """Write a valid selected-features JSON to path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        "direction": direction,
        "features": [
            {
                "name": f"feat_{i}",
                "mode": "binary",
                "score": float(n_features - i),
            }
            for i in range(n_features)
        ],
    }
    with open(path, "w") as fh:
        json.dump(data, fh)


def _make_df(
    n_rows: int = 200,
    symbols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a minimal DataFrame with all required columns."""
    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = ["SYM_A", "SYM_B"]

    rows_per_sym = n_rows // len(symbols)
    dfs = []
    for sym in symbols:
        n = rows_per_sym
        open_next = rng.uniform(100, 200, size=n)
        max_288 = open_next * rng.uniform(1.00, 1.10, size=n)
        min_288 = open_next * rng.uniform(0.90, 1.00, size=n)
        close_288 = open_next * rng.uniform(0.95, 1.05, size=n)
        max_before_min = rng.integers(0, 2, size=n)

        data = {
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "symbol": sym,
            "label_open_next": open_next,
            "label_close_288": close_288,
            "label_min_288": min_288,
            "label_max_288": max_288,
            "label_max_before_min": max_before_min.astype(float),
            "_symbol_bar_index": np.arange(n),
        }
        for i in range(5):
            data[f"feat_{i}"] = rng.uniform(0, 1, size=n)

        long_rows = np.arange(n) % 2 == 0
        data["hwc_state"] = np.where(long_rows, 1, -1)
        data["mwc_state"] = np.where(long_rows, 1, -1)
        data["lwc_state"] = np.where(long_rows, 1, -1)
        data["tf_permission_long"] = long_rows.astype(np.int8)
        data["tf_permission_short"] = (~long_rows).astype(np.int8)
        data["lwc_pullback_reversal_long"] = long_rows.astype(np.int8)
        data["lwc_pullback_reversal_short"] = (~long_rows).astype(np.int8)

        dfs.append(pd.DataFrame(data))

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: OOS_Evaluator constructor
# ---------------------------------------------------------------------------

class TestOOSEvaluatorInit:
    def test_default_test_csv_path(self):
        ev = OOS_Evaluator()
        assert ev.test_csv_path == _cfg.TEST_CSV_PATH

    def test_custom_test_csv_path(self):
        ev = OOS_Evaluator(test_csv_path="custom/path.csv")
        assert ev.test_csv_path == "custom/path.csv"

    def test_none_uses_config_default(self):
        ev = OOS_Evaluator(test_csv_path=None)
        assert ev.test_csv_path == _cfg.TEST_CSV_PATH


# ---------------------------------------------------------------------------
# Tests: load_strategies
# ---------------------------------------------------------------------------

class TestLoadStrategies:
    def test_returns_empty_when_no_files(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = str(tmp_path / "long.json")
        m._STRATEGY_PATHS["short"] = str(tmp_path / "short.json")
        try:
            result = OOS_Evaluator.load_strategies()
            assert result == {}
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_loads_strategy_with_symbol_filters(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m

        long_path = str(tmp_path / "long.json")
        rule_set = {
            "direction": "long",
            "rules_set": [
                {
                    "tp": _cfg.PHASE2_TP,
                    "sl": _cfg.PHASE2_SL,
                    "capital_pct": _cfg.PHASE2_CAPITAL_PCT,
                    "conditions": [
                        "symbol is 1",
                        "[feat_0] IS Very High",
                        "[feat_1] IS High",
                        *_cfg.mandatory_context_conditions("long"),
                    ],
                },
                {
                    "tp": _cfg.PHASE2_TP,
                    "sl": _cfg.PHASE2_SL,
                    "capital_pct": _cfg.PHASE2_CAPITAL_PCT,
                    "conditions": [
                        "[symbol] IS 2",
                        "[feat_0] IS Very High",
                        "[feat_1] IS High",
                        *_cfg.mandatory_context_conditions("long"),
                    ],
                },
            ],
        }
        Output_Writer().write(rule_set, long_path)
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = long_path
        m._STRATEGY_PATHS["short"] = str(tmp_path / "short.json")
        try:
            result = OOS_Evaluator.load_strategies()
            assert "long" in result
            assert result["long"]["rules_set"][0]["conditions"][0] == "symbol is 1"
            assert result["long"]["rules_set"][1]["conditions"][0] == "[symbol] IS 2"
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_loads_long_when_only_long_exists(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        long_path = str(tmp_path / "long.json")
        _write_rule_set(long_path, "long")
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = long_path
        m._STRATEGY_PATHS["short"] = str(tmp_path / "short.json")
        try:
            result = OOS_Evaluator.load_strategies()
            assert "long" in result
            assert "short" not in result
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_loads_short_when_only_short_exists(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        short_path = str(tmp_path / "short.json")
        _write_rule_set(short_path, "short")
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = str(tmp_path / "long.json")
        m._STRATEGY_PATHS["short"] = short_path
        try:
            result = OOS_Evaluator.load_strategies()
            assert "short" in result
            assert "long" not in result
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_loads_both_when_both_exist(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        long_path = str(tmp_path / "long.json")
        short_path = str(tmp_path / "short.json")
        _write_rule_set(long_path, "long")
        _write_rule_set(short_path, "short")
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = long_path
        m._STRATEGY_PATHS["short"] = short_path
        try:
            result = OOS_Evaluator.load_strategies()
            assert "long" in result
            assert "short" in result
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_skips_invalid_json_file(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        long_path = str(tmp_path / "long.json")
        os.makedirs(os.path.dirname(long_path), exist_ok=True)
        with open(long_path, "w") as fh:
            fh.write("{corrupted json")
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = long_path
        m._STRATEGY_PATHS["short"] = str(tmp_path / "short.json")
        try:
            result = OOS_Evaluator.load_strategies()
            assert "long" not in result
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_loaded_strategy_has_direction_and_rules_set(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        long_path = str(tmp_path / "long.json")
        _write_rule_set(long_path, "long")
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = long_path
        m._STRATEGY_PATHS["short"] = str(tmp_path / "short.json")
        try:
            result = OOS_Evaluator.load_strategies()
            assert "direction" in result["long"]
            assert "rules_set" in result["long"]
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_allowed_directions_skips_stale_file(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m

        long_path = str(tmp_path / "long.json")
        short_path = str(tmp_path / "short.json")
        _write_rule_set(long_path, "long")
        _write_rule_set(short_path, "short")
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = long_path
        m._STRATEGY_PATHS["short"] = short_path
        try:
            result = OOS_Evaluator.load_strategies(
                allowed_directions=frozenset({"short"}))
            assert "short" in result
            assert "long" not in result
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_empty_allowed_directions_skips_all(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m

        long_path = str(tmp_path / "long.json")
        _write_rule_set(long_path, "long")
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = long_path
        m._STRATEGY_PATHS["short"] = str(tmp_path / "short.json")
        try:
            result = OOS_Evaluator.load_strategies(
                allowed_directions=frozenset())
            assert result == {}
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_loaded_direction_matches_key(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        long_path = str(tmp_path / "long.json")
        _write_rule_set(long_path, "long")
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = long_path
        m._STRATEGY_PATHS["short"] = str(tmp_path / "short.json")
        try:
            result = OOS_Evaluator.load_strategies()
            assert result["long"]["direction"] == "long"
        finally:
            m._STRATEGY_PATHS.update(original)

    def test_skips_strategy_when_declared_direction_mismatches_filename(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m

        long_path = str(tmp_path / "long.json")
        _write_rule_set(long_path, "short")
        original = m._STRATEGY_PATHS.copy()
        m._STRATEGY_PATHS["long"] = long_path
        m._STRATEGY_PATHS["short"] = str(tmp_path / "short.json")
        try:
            assert OOS_Evaluator.load_strategies() == {}
        finally:
            m._STRATEGY_PATHS.update(original)


# ---------------------------------------------------------------------------
# Tests: prepare_test_data
# ---------------------------------------------------------------------------

class TestPrepareTestData:
    def test_returns_dataframe(self, tmp_path):
        """prepare_test_data should return a DataFrame."""
        # Write a minimal CSV
        df = _make_df(n_rows=600, symbols=["SYM_A"])
        # Add extra rows so tail-drop leaves something
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)
        result = OOS_Evaluator.prepare_test_data(csv_path)
        assert isinstance(result, pd.DataFrame)

    def test_has_symbol_bar_index_column(self, tmp_path):
        df = _make_df(n_rows=600, symbols=["SYM_A"])
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)
        result = OOS_Evaluator.prepare_test_data(csv_path)
        assert "_symbol_bar_index" in result.columns

    def test_no_nan_in_label_columns(self, tmp_path):
        df = _make_df(n_rows=600, symbols=["SYM_A"])
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)
        result = OOS_Evaluator.prepare_test_data(csv_path)
        for col in _cfg.LABEL_COLUMNS:
            if col in result.columns:
                assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_sorted_by_symbol_and_datetime(self, tmp_path):
        df = _make_df(n_rows=600, symbols=["SYM_A", "SYM_B"])
        # Shuffle to test sorting
        df = df.sample(frac=1, random_state=0).reset_index(drop=True)
        csv_path = str(tmp_path / "test.csv")
        df.to_csv(csv_path, index=False)
        result = OOS_Evaluator.prepare_test_data(csv_path)
        for sym, grp in result.groupby("symbol"):
            dts = grp["datetime"].values
            assert all(dts[i] <= dts[i + 1] for i in range(len(dts) - 1)), (
                f"Symbol {sym} is not sorted by datetime"
            )


# ---------------------------------------------------------------------------
# Tests: _evaluate_strategy (via OOS_Evaluator internals)
# ---------------------------------------------------------------------------

class TestEvaluateStrategy:
    def _make_evaluator(self) -> OOS_Evaluator:
        return OOS_Evaluator(test_csv_path=_cfg.TEST_CSV_PATH)

    def test_returns_metrics_dict(self):
        ev = self._make_evaluator()
        df = _make_df(n_rows=200)
        strategy = _make_rule_set("long")
        metrics, _, _log = ev._evaluate_strategy(df, strategy, "long")
        assert isinstance(metrics, dict)

    def test_metrics_has_required_keys(self):
        ev = self._make_evaluator()
        df = _make_df(n_rows=200)
        strategy = _make_rule_set("long")
        metrics, _, _log = ev._evaluate_strategy(df, strategy, "long")
        for key in ("total_return_pct", "max_drawdown_pct", "win_rate",
                    "profit_factor", "executed_trades", "account_ruined"):
            assert key in metrics, f"Missing key: {key}"

    def test_zero_trade_case_no_account_ruin(self):
        """When no trades are executed, account_ruined must be False."""
        ev = self._make_evaluator()
        df = _make_df(n_rows=200)
        # Use a condition that will never match (feat_99 doesn't exist → skip)
        # Instead use a condition that matches nothing: feat_0 IS Very High
        # but set feat_0 to all zeros (Very Low range)
        df["feat_0"] = 0.0  # Very Low, not Very High
        strategy = {
            "direction": "long",
            "rules_set": [
                {
                    "conditions": [
                        "[feat_0] IS Very High",
                        *_cfg.mandatory_context_conditions("long"),
                    ],
                    "tp": 4.0,
                    "sl": 2.0,
                    "capital_pct": 50.0,
                }
            ],
        }
        metrics, _, _log = ev._evaluate_strategy(df, strategy, "long")
        if metrics["executed_trades"] == 0:
            assert metrics["account_ruined"] is False
            assert metrics["total_return_pct"] == 0.0

    def test_zero_trade_case_total_return_is_zero(self):
        """When no trades are executed, total_return_pct must be 0.0."""
        ev = self._make_evaluator()
        df = _make_df(n_rows=200)
        df["feat_0"] = 0.0
        strategy = {
            "direction": "long",
            "rules_set": [
                {
                    "conditions": [
                        "[feat_0] IS Very High",
                        *_cfg.mandatory_context_conditions("long"),
                    ],
                    "tp": 4.0,
                    "sl": 2.0,
                    "capital_pct": 50.0,
                }
            ],
        }
        metrics, _, _log = ev._evaluate_strategy(df, strategy, "long")
        if metrics["executed_trades"] == 0:
            assert metrics["total_return_pct"] == 0.0

    def test_returns_per_symbol_rows(self):
        ev = self._make_evaluator()
        df = _make_df(n_rows=200)
        strategy = _make_rule_set("long")
        _, per_symbol_rows, _log = ev._evaluate_strategy(df, strategy, "long")
        assert isinstance(per_symbol_rows, list)

    def test_per_symbol_rows_have_required_keys(self):
        ev = self._make_evaluator()
        df = _make_df(n_rows=200)
        strategy = _make_rule_set("long")
        _, per_symbol_rows, _log = ev._evaluate_strategy(df, strategy, "long")
        for row in per_symbol_rows:
            for key in ("direction", "symbol", "trade_count", "win_rate", "net_pnl"):
                assert key in row, f"Missing key: {key}"

    def test_per_symbol_rows_direction_matches(self):
        ev = self._make_evaluator()
        df = _make_df(n_rows=200)
        strategy = _make_rule_set("short")
        _, per_symbol_rows, _log = ev._evaluate_strategy(df, strategy, "short")
        for row in per_symbol_rows:
            assert row["direction"] == "short"

    def test_simulation_error_is_explicitly_marked(self, monkeypatch):
        ev = self._make_evaluator()
        df = _make_df(n_rows=200)
        strategy = _make_rule_set("long")

        def _raise(*_args, **_kwargs):
            raise RuntimeError("synthetic engine failure")

        monkeypatch.setattr(
            "gpu_fuzzy_trader.phases.phase5_oos.CPUBacktestEngine.simulate_rule_set",
            _raise,
        )
        metrics, _, trade_log = ev._evaluate_strategy(df, strategy, "long")

        assert metrics["evaluation_status"] == "error"
        assert "synthetic engine failure" in metrics["evaluation_error"]
        assert metrics["account_ruined"] is False
        assert trade_log.empty


# ---------------------------------------------------------------------------
# Tests: _build_per_symbol_rows
# ---------------------------------------------------------------------------

class TestBuildPerSymbolRows:
    def test_empty_per_symbol_metrics_returns_empty_list(self):
        metrics = {"per_symbol_metrics": {}}
        rows = OOS_Evaluator._build_per_symbol_rows(
            metrics, "long", pd.DataFrame())
        assert rows == []

    def test_missing_per_symbol_metrics_returns_empty_list(self):
        metrics = {}
        rows = OOS_Evaluator._build_per_symbol_rows(
            metrics, "long", pd.DataFrame())
        assert rows == []

    def test_returns_one_row_per_symbol(self):
        metrics = {
            "per_symbol_metrics": {
                "SYM_A": {"trade_count": 5, "win_rate": 60.0, "net_pnl": 10.0},
                "SYM_B": {"trade_count": 3, "win_rate": 33.3, "net_pnl": -2.0},
            }
        }
        rows = OOS_Evaluator._build_per_symbol_rows(
            metrics, "long", pd.DataFrame())
        assert len(rows) == 2

    def test_row_has_correct_direction(self):
        metrics = {
            "per_symbol_metrics": {
                "SYM_A": {"trade_count": 5, "win_rate": 60.0, "net_pnl": 10.0},
            }
        }
        rows = OOS_Evaluator._build_per_symbol_rows(
            metrics, "short", pd.DataFrame())
        assert rows[0]["direction"] == "short"

    def test_row_has_correct_symbol(self):
        metrics = {
            "per_symbol_metrics": {
                "SYM_X": {"trade_count": 2, "win_rate": 50.0, "net_pnl": 5.0},
            }
        }
        rows = OOS_Evaluator._build_per_symbol_rows(
            metrics, "long", pd.DataFrame())
        assert rows[0]["symbol"] == "SYM_X"

    def test_row_values_match_metrics(self):
        metrics = {
            "per_symbol_metrics": {
                "SYM_A": {"trade_count": 7, "win_rate": 71.4, "net_pnl": 15.5},
            }
        }
        rows = OOS_Evaluator._build_per_symbol_rows(
            metrics, "long", pd.DataFrame())
        assert rows[0]["trade_count"] == 7
        assert abs(rows[0]["win_rate"] - 71.4) < 1e-6
        assert abs(rows[0]["net_pnl"] - 15.5) < 1e-6


# ---------------------------------------------------------------------------
# Tests: _save_report
# ---------------------------------------------------------------------------

class TestSaveReport:
    def _make_metrics(self, account_ruined: bool = False) -> dict:
        return {
            "direction": "long",
            "total_return_pct": 5.0,
            "max_drawdown_pct": 2.0,
            "win_rate": 55.0,
            "profit_factor": 1.5,
            "executed_trades": 10,
            "account_ruined": account_ruined,
            "final_equity": 1050.0,
            "per_symbol_metrics": {
                "SYM_A": {"trade_count": 5, "win_rate": 60.0, "net_pnl": 30.0},
            },
        }

    def test_creates_report_file(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        report_path = str(tmp_path / "reports" / "test_long_report.json")
        m._REPORT_PATHS["long"] = report_path
        try:
            ev = OOS_Evaluator()
            ev._save_report(self._make_metrics(), "long")
            assert os.path.exists(report_path)
        finally:
            m._REPORT_PATHS.update(original)

    def test_report_is_valid_json(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        report_path = str(tmp_path / "reports" / "test_long_report.json")
        m._REPORT_PATHS["long"] = report_path
        try:
            ev = OOS_Evaluator()
            ev._save_report(self._make_metrics(), "long")
            with open(report_path) as fh:
                data = json.load(fh)
            assert isinstance(data, dict)
        finally:
            m._REPORT_PATHS.update(original)

    def test_report_has_required_keys(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        report_path = str(tmp_path / "reports" / "test_long_report.json")
        m._REPORT_PATHS["long"] = report_path
        try:
            ev = OOS_Evaluator()
            ev._save_report(self._make_metrics(), "long")
            with open(report_path) as fh:
                data = json.load(fh)
            for key in ("direction", "total_return_pct", "max_drawdown_pct",
                        "win_rate", "profit_factor", "executed_trades",
                        "account_status", "final_equity"):
                assert key in data, f"Missing key: {key}"
        finally:
            m._REPORT_PATHS.update(original)

    def test_account_status_survived_when_not_ruined(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        report_path = str(tmp_path / "reports" / "test_long_report.json")
        m._REPORT_PATHS["long"] = report_path
        try:
            ev = OOS_Evaluator()
            ev._save_report(self._make_metrics(account_ruined=False), "long")
            with open(report_path) as fh:
                data = json.load(fh)
            assert data["account_status"] == "survived"
        finally:
            m._REPORT_PATHS.update(original)

    def test_account_status_ruined_when_ruined(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        report_path = str(tmp_path / "reports" / "test_long_report.json")
        m._REPORT_PATHS["long"] = report_path
        try:
            ev = OOS_Evaluator()
            ev._save_report(self._make_metrics(account_ruined=True), "long")
            with open(report_path) as fh:
                data = json.load(fh)
            assert data["account_status"] == "ruined"
        finally:
            m._REPORT_PATHS.update(original)

    def test_account_status_error_when_evaluation_failed(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        report_path = str(tmp_path / "reports" / "test_long_report.json")
        m._REPORT_PATHS["long"] = report_path
        try:
            ev = OOS_Evaluator()
            metrics = self._make_metrics()
            metrics.update(
                evaluation_status="error",
                evaluation_error="RuntimeError: synthetic failure",
            )
            ev._save_report(metrics, "long")
            with open(report_path) as fh:
                data = json.load(fh)
            assert data["evaluation_status"] == "error"
            assert data["account_status"] == "error"
            assert "synthetic failure" in data["evaluation_error"]
        finally:
            m._REPORT_PATHS.update(original)


# ---------------------------------------------------------------------------
# Tests: _save_per_symbol_csv
# ---------------------------------------------------------------------------

class TestSavePerSymbolCsv:
    def test_creates_csv_file(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        csv_path = str(tmp_path / "reports" /
                       "test_per_symbol_performance.csv")
        m._REPORT_PATHS["per_symbol"] = csv_path
        try:
            rows = [
                {"direction": "long", "symbol": "SYM_A",
                 "trade_count": 5, "win_rate": 60.0, "net_pnl": 10.0},
            ]
            OOS_Evaluator._save_per_symbol_csv(rows)
            assert os.path.exists(csv_path)
        finally:
            m._REPORT_PATHS.update(original)

    def test_csv_has_correct_columns(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        csv_path = str(tmp_path / "reports" /
                       "test_per_symbol_performance.csv")
        m._REPORT_PATHS["per_symbol"] = csv_path
        try:
            rows = [
                {"direction": "long", "symbol": "SYM_A",
                 "trade_count": 5, "win_rate": 60.0, "net_pnl": 10.0},
            ]
            OOS_Evaluator._save_per_symbol_csv(rows)
            df = pd.read_csv(csv_path)
            for col in ("direction", "symbol", "trade_count", "win_rate", "net_pnl"):
                assert col in df.columns, f"Missing column: {col}"
        finally:
            m._REPORT_PATHS.update(original)

    def test_empty_rows_creates_empty_csv_with_columns(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        csv_path = str(tmp_path / "reports" /
                       "test_per_symbol_performance.csv")
        m._REPORT_PATHS["per_symbol"] = csv_path
        try:
            OOS_Evaluator._save_per_symbol_csv([])
            df = pd.read_csv(csv_path)
            assert len(df) == 0
            for col in ("direction", "symbol", "trade_count", "win_rate", "net_pnl"):
                assert col in df.columns
        finally:
            m._REPORT_PATHS.update(original)

    def test_csv_row_count_matches_input(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        original = m._REPORT_PATHS.copy()
        csv_path = str(tmp_path / "reports" /
                       "test_per_symbol_performance.csv")
        m._REPORT_PATHS["per_symbol"] = csv_path
        try:
            rows = [
                {"direction": "long", "symbol": "SYM_A",
                 "trade_count": 5, "win_rate": 60.0, "net_pnl": 10.0},
                {"direction": "short", "symbol": "SYM_B",
                 "trade_count": 3, "win_rate": 33.3, "net_pnl": -2.0},
            ]
            OOS_Evaluator._save_per_symbol_csv(rows)
            df = pd.read_csv(csv_path)
            assert len(df) == 2
        finally:
            m._REPORT_PATHS.update(original)


# ---------------------------------------------------------------------------
# Tests: OOS_Evaluator.run() — integration
# ---------------------------------------------------------------------------

def _write_synthetic_test_csv(tmp_path: "Path", n_rows: int = 600) -> str:
    """
    Write a synthetic test CSV with all required columns (including feat_0..4)
    to a temp file and return its path.  n_rows must be > 288 per symbol so
    the tail-drop leaves data to evaluate.
    """
    df = _make_df(n_rows=n_rows, symbols=["SYM_A", "SYM_B"])
    csv_path = str(tmp_path / "synthetic_test.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


class TestOOSEvaluatorRun:
    """Integration tests using tmp_path overrides for all output paths.

    The run() method is expensive (~4s) because it loads CSV data and runs
    CPUBacktestEngine.  We use a shared class-scoped fixture that runs ev.run()
    *once* and stores the result; all simple assertion tests reuse it.
    """

    @pytest.fixture(scope="class")
    def run_result(self, tmp_path_factory):
        """Run OOS_Evaluator.run() once for the whole class (long direction only)."""
        import gpu_fuzzy_trader.phases.phase5_oos as m

        tmp_path = tmp_path_factory.mktemp("oos_run")
        orig_strategy = m._STRATEGY_PATHS.copy()
        orig_report = m._REPORT_PATHS.copy()

        for d in ("long", "short"):
            m._STRATEGY_PATHS[d] = str(tmp_path / f"{d}.json")
        m._REPORT_PATHS["long"] = str(
            tmp_path / "reports" / "test_long_report.json")
        m._REPORT_PATHS["short"] = str(
            tmp_path / "reports" / "test_short_report.json")
        m._REPORT_PATHS["per_symbol"] = str(
            tmp_path / "reports" / "test_per_symbol_performance.csv"
        )
        m._REPORT_PATHS["joint"] = str(
            tmp_path / "reports" / "test_joint_portfolio_report.json"
        )
        m._REPORT_PATHS["forward_long"] = str(
            tmp_path / "reports" / "forward_long_report.json"
        )
        m._REPORT_PATHS["forward_short"] = str(
            tmp_path / "reports" / "forward_short_report.json"
        )
        m._REPORT_PATHS["forward_joint"] = str(
            tmp_path / "reports" / "forward_joint_portfolio_report.json"
        )

        _write_rule_set(str(tmp_path / "long.json"), "long")
        csv_path = _write_synthetic_test_csv(tmp_path)

        try:
            ev = OOS_Evaluator(test_csv_path=csv_path)
            result = ev.run()
        finally:
            m._STRATEGY_PATHS.update(orig_strategy)
            m._REPORT_PATHS.update(orig_report)

        return {"result": result, "tmp_path": tmp_path, "m": m}

    def _setup_paths(self, m, tmp_path, directions=("long", "short")):
        """Override module-level path dicts and return originals (for standalone tests)."""
        orig_strategy = m._STRATEGY_PATHS.copy()
        orig_report = m._REPORT_PATHS.copy()

        for d in ("long", "short"):
            m._STRATEGY_PATHS[d] = str(tmp_path / f"{d}.json")
        m._REPORT_PATHS["long"] = str(
            tmp_path / "reports" / "test_long_report.json")
        m._REPORT_PATHS["short"] = str(
            tmp_path / "reports" / "test_short_report.json")
        m._REPORT_PATHS["per_symbol"] = str(
            tmp_path / "reports" / "test_per_symbol_performance.csv"
        )
        m._REPORT_PATHS["joint"] = str(
            tmp_path / "reports" / "test_joint_portfolio_report.json"
        )
        m._REPORT_PATHS["forward_long"] = str(
            tmp_path / "reports" / "forward_long_report.json"
        )
        m._REPORT_PATHS["forward_short"] = str(
            tmp_path / "reports" / "forward_short_report.json"
        )
        m._REPORT_PATHS["forward_joint"] = str(
            tmp_path / "reports" / "forward_joint_portfolio_report.json"
        )

        for d in directions:
            _write_rule_set(str(tmp_path / f"{d}.json"), d)

        return orig_strategy, orig_report

    def _restore_paths(self, m, orig_strategy, orig_report):
        m._STRATEGY_PATHS.update(orig_strategy)
        m._REPORT_PATHS.update(orig_report)

    def test_run_returns_empty_when_no_strategies(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        orig_s, orig_r = self._setup_paths(m, tmp_path, directions=())
        csv_path = _write_synthetic_test_csv(tmp_path)
        try:
            ev = OOS_Evaluator(test_csv_path=csv_path)
            result = ev.run()
            assert result == {}
        finally:
            self._restore_paths(m, orig_s, orig_r)

    # ---------------------------------------------------------------------------
    # Tests that reuse the shared run_result fixture (single ev.run() call)
    # ---------------------------------------------------------------------------

    def test_run_returns_dict_with_long_key(self, run_result):
        assert "long" in run_result["result"]

    def test_run_returns_dict_with_both_keys(self, tmp_path):
        """Both directions: run once for this specific case."""
        import gpu_fuzzy_trader.phases.phase5_oos as m
        orig_s, orig_r = self._setup_paths(
            m, tmp_path, directions=("long", "short"))
        csv_path = _write_synthetic_test_csv(tmp_path)
        try:
            ev = OOS_Evaluator(test_csv_path=csv_path)
            result = ev.run()
            assert "long" in result
            assert "short" in result
        finally:
            self._restore_paths(m, orig_s, orig_r)

    def test_run_creates_long_report_file(self, run_result):
        long_report = str(run_result["tmp_path"] / "reports" / "test_long_report.json")
        assert os.path.exists(long_report)

    def test_run_creates_per_symbol_csv(self, run_result):
        per_sym = str(run_result["tmp_path"] / "reports" / "test_per_symbol_performance.csv")
        assert os.path.exists(per_sym)

    def test_run_metrics_has_required_keys(self, run_result):
        metrics = run_result["result"]["long"]["test"]
        for key in ("total_return_pct", "max_drawdown_pct", "win_rate",
                    "executed_trades", "account_ruined"):
            assert key in metrics, f"Missing key: {key}"

    def test_run_long_report_is_valid_json(self, run_result):
        long_report = str(run_result["tmp_path"] / "reports" / "test_long_report.json")
        with open(long_report) as fh:
            data = json.load(fh)
        assert isinstance(data, dict)

    def test_run_per_symbol_csv_has_correct_columns(self, run_result):
        per_sym = str(run_result["tmp_path"] / "reports" / "test_per_symbol_performance.csv")
        df = pd.read_csv(per_sym)
        for col in ("direction", "symbol", "trade_count", "win_rate", "net_pnl"):
            assert col in df.columns, f"Missing column: {col}"

    def test_oos_reports_do_not_mutate_strategy_from_test_metrics(self, tmp_path):
        """Returned and saved OOS metrics come from the locked strategy."""
        import gpu_fuzzy_trader.phases.phase5_oos as m

        orig_s, orig_r = self._setup_paths(m, tmp_path, directions=("short",))
        strategy = _make_rule_set("short", n_rules=3)
        pre_test_metrics = {
            "direction": "short",
            "total_return_pct": -3.59,
            "max_drawdown_pct": 8.42,
            "win_rate": 38.0,
            "profit_factor": 0.9,
            "executed_trades": 900,
            "account_ruined": False,
            "final_equity": 964.1,
            "per_symbol_metrics": {},
        }
        split_metrics = {
            "train": {"total_return_pct": 1.0, "executed_trades": 100},
            "validation": {"total_return_pct": 0.5, "executed_trades": 80},
            "test": pre_test_metrics,
        }
        evaluate_calls = {"count": 0}

        def mock_evaluate(_self, _df, _strategy, direction):
            evaluate_calls["count"] += 1
            if evaluate_calls["count"] == 3:
                return pre_test_metrics, [], pd.DataFrame()
            split = ("train", "validation")[evaluate_calls["count"] - 1]
            return split_metrics[split], [], pd.DataFrame()

        datasets = {
            "train": _make_df(n_rows=20),
            "validation": _make_df(n_rows=20),
            "test": _make_df(n_rows=20),
        }

        with patch.object(
            OOS_Evaluator,
            "_load_datasets_by_split",
            return_value=datasets,
        ), patch.object(
            OOS_Evaluator,
            "_evaluate_strategy",
            mock_evaluate,
        ), patch.object(
            OOS_Evaluator,
            "_load_selected_features",
            return_value=[],
        ), patch(
            "gpu_fuzzy_trader.phases.phase5_oos.Reporter"
        ) as reporter_cls:
            reporter_cls.return_value.plot_equity_curve.return_value = ""
            reporter_cls.return_value.write_per_symbol_csv.return_value = None
            reporter_cls.return_value.write_strategy_evaluation_table.return_value = None
            reporter_cls.return_value.plot_per_rule_breakdown.return_value = None
            reporter_cls.return_value.plot_distribution_and_equity.return_value = None
            reporter_cls.return_value.write_spearman_correlation_report.return_value = None
            reporter_cls.return_value.write_feature_stratified_performance.return_value = None
            reporter_cls.return_value.write_generalization_diagnostics.return_value = None

            report_path = m._REPORT_PATHS["short"]
            try:
                result = OOS_Evaluator(test_csv_path=str(
                    tmp_path / "unused.csv")).run()
            finally:
                self._restore_paths(m, orig_s, orig_r)

        assert result["short"]["test"]["total_return_pct"] == pytest.approx(
            -3.59)
        assert result["short"]["test"]["executed_trades"] == 900

        with open(report_path) as fh:
            report = json.load(fh)
        assert report["total_return_pct"] == pytest.approx(-3.59)
        assert report["executed_trades"] == 900

    def test_run_creates_extended_reporting_outputs(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase5_oos as m
        import gpu_fuzzy_trader.reporting.reporter as reporter_module

        orig_s, orig_r = self._setup_paths(m, tmp_path, directions=("long",))
        orig_features = m._FEATURE_PATHS.copy()
        orig_reports_dir = reporter_module._REPORTS_DIR
        csv_path = _write_synthetic_test_csv(tmp_path)

        m._FEATURE_PATHS["long"] = str(
            tmp_path / "selected_features_long.json")
        m._FEATURE_PATHS["short"] = str(
            tmp_path / "selected_features_short.json")
        _write_selected_features(m._FEATURE_PATHS["long"], "long")
        _write_selected_features(m._FEATURE_PATHS["short"], "short")
        reporter_module._REPORTS_DIR = str(tmp_path / "reports")

        try:
            ev = OOS_Evaluator(test_csv_path=csv_path)
            ev.run()

            expected_paths = [
                m._REPORT_PATHS["long"],
                tmp_path / "reports" / "strategy_evaluation_long.csv",
                tmp_path / "reports" / "per_rule_breakdown_long.png",
                tmp_path / "reports" / "spearman_correlation_long.csv",
                tmp_path / "reports" / "feature_stratified_train_long.csv",
                tmp_path / "reports" / "feature_stratified_validation_long.csv",
                tmp_path / "reports" / "feature_stratified_test_long.csv",
                tmp_path / "reports" / "distribution_equity_test_long.png",
            ]

            for path in expected_paths:
                assert os.path.exists(path), f"Missing expected report: {path}"
        finally:
            m._FEATURE_PATHS.update(orig_features)
            reporter_module._REPORTS_DIR = orig_reports_dir
            self._restore_paths(m, orig_s, orig_r)


class TestPhase5CachedSplitFreshness:
    def test_load_datasets_uses_validated_cache_when_fresh(
        self, tmp_path, monkeypatch,
    ):
        from gpu_fuzzy_trader.data.loader import Data_Loader
        from gpu_fuzzy_trader.data.splitter import Data_Splitter
        from tests.unit.test_data_splitter import _patch_split_paths

        train_csv = tmp_path / "train.csv"
        loader = Data_Loader()
        source_df = loader.load_dataset(
            _write_synthetic_test_csv(tmp_path, n_rows=3000)
        )
        source_df.to_csv(train_csv, index=False)
        (tmp_path / "test").mkdir()
        test_csv = _write_synthetic_test_csv(tmp_path / "test")

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with _patch_split_paths(str(cache_dir))() as paths:
            Data_Splitter().split_and_persist(source_df)
            os.utime(train_csv, (1, 1))
            for path in paths.values():
                os.utime(path, (2, 2))

        monkeypatch.setattr(_cfg, "TRAIN_CSV_PATH", str(train_csv))
        monkeypatch.setattr(_cfg, "TRAIN_70_PATH", paths["train"])
        monkeypatch.setattr(_cfg, "VALIDATION_30_PATH", paths["val"])
        monkeypatch.setattr(_cfg, "VALIDATION_FITNESS_PATH", paths["fitness"])
        monkeypatch.setattr(
            _cfg, "VALIDATION_SELECTION_PATH", paths["selection"])
        monkeypatch.setattr(_cfg, "CV_FOLDS_MANIFEST_PATH", paths["manifest"])
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "holdout")

        ev = OOS_Evaluator(test_csv_path=test_csv)
        datasets = ev._load_datasets_by_split()
        assert len(datasets["train"]) > 0
        assert len(datasets["validation"]) > 0
        assert len(datasets["test"]) > 0

    def test_load_datasets_rejects_stale_parquet_cache(
        self, tmp_path, monkeypatch,
    ):
        from gpu_fuzzy_trader.data.loader import Data_Loader
        from gpu_fuzzy_trader.data.splitter import Data_Splitter, load_cached_split_if_fresh
        from tests.unit.test_data_splitter import _patch_split_paths

        train_csv = tmp_path / "train.csv"
        loader = Data_Loader()
        source_df = loader.load_dataset(_write_synthetic_test_csv(tmp_path))
        source_df.to_csv(train_csv, index=False)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        with _patch_split_paths(str(cache_dir))() as paths:
            Data_Splitter().split_and_persist(source_df)
            os.utime(train_csv, (3, 3))
            for path in paths.values():
                os.utime(path, (2, 2))

        monkeypatch.setattr(_cfg, "TRAIN_CSV_PATH", str(train_csv))
        monkeypatch.setattr(_cfg, "TRAIN_70_PATH", paths["train"])
        monkeypatch.setattr(_cfg, "VALIDATION_30_PATH", paths["val"])
        monkeypatch.setattr(_cfg, "VALIDATION_FITNESS_PATH", paths["fitness"])
        monkeypatch.setattr(
            _cfg, "VALIDATION_SELECTION_PATH", paths["selection"])
        monkeypatch.setattr(_cfg, "CV_FOLDS_MANIFEST_PATH", paths["manifest"])
        monkeypatch.setattr(_cfg, "SPLIT_MODE", "holdout")

        assert load_cached_split_if_fresh() is None


# ---------------------------------------------------------------------------
# Tests: plot_equity_curve — train, validation, test + empty trade log safety
# ---------------------------------------------------------------------------

class TestEquityCurvePlots:
    """Verify plot_equity_curve is called for all three splits and handles empty logs."""

    def _run_with_mocks(self, tmp_path, train_log, val_log, test_log):
        """Helper: run OOS_Evaluator.run() with mocked internals and capture plot calls."""
        import gpu_fuzzy_trader.phases.phase5_oos as m

        orig_s, orig_r = self._setup_paths(m, tmp_path, directions=("long",))
        strategy = _make_rule_set("long", n_rules=2)

        datasets = {
            "train": _make_df(n_rows=20),
            "validation": _make_df(n_rows=20),
            "test": _make_df(n_rows=20),
        }
        evaluate_calls = {"count": 0}
        split_order = ["train", "validation", "test"]

        def mock_evaluate(_self, _df, _strategy, direction):
            idx = evaluate_calls["count"]
            evaluate_calls["count"] += 1
            split = split_order[idx]
            logs = {"train": train_log, "validation": val_log, "test": test_log}
            tl = logs[split]
            n_trades = len(tl) if tl is not None and not tl.empty else 0
            metrics = {"total_return_pct": 1.0, "executed_trades": n_trades}
            return metrics, [], tl

        with patch.object(
            OOS_Evaluator,
            "_load_datasets_by_split",
            return_value=datasets,
        ), patch.object(
            OOS_Evaluator,
            "_evaluate_strategy",
            mock_evaluate,
        ), patch.object(
            OOS_Evaluator,
            "_load_selected_features",
            return_value=[],
        ), patch(
            "gpu_fuzzy_trader.phases.phase5_oos.Reporter"
        ) as reporter_cls:
            reporter_cls.return_value.plot_equity_curve.return_value = ""
            reporter_cls.return_value.write_per_symbol_csv.return_value = None
            reporter_cls.return_value.write_strategy_evaluation_table.return_value = None
            reporter_cls.return_value.plot_per_rule_breakdown.return_value = None
            reporter_cls.return_value.plot_distribution_and_equity.return_value = None
            reporter_cls.return_value.write_spearman_correlation_report.return_value = None
            reporter_cls.return_value.write_feature_stratified_performance.return_value = None
            reporter_cls.return_value.write_generalization_diagnostics.return_value = None

            try:
                OOS_Evaluator(test_csv_path=str(tmp_path / "unused.csv")).run()
            finally:
                self._restore_paths(m, orig_s, orig_r)

        return reporter_cls.return_value.plot_equity_curve

    def _setup_paths(self, m, tmp_path, directions=("long", "short")):
        """Override module-level path dicts and return originals."""
        orig_strategy = m._STRATEGY_PATHS.copy()
        orig_report = m._REPORT_PATHS.copy()

        for d in ("long", "short"):
            m._STRATEGY_PATHS[d] = str(tmp_path / f"{d}.json")
        m._REPORT_PATHS["long"] = str(
            tmp_path / "reports" / "test_long_report.json")
        m._REPORT_PATHS["short"] = str(
            tmp_path / "reports" / "test_short_report.json")
        m._REPORT_PATHS["per_symbol"] = str(
            tmp_path / "reports" / "test_per_symbol_performance.csv"
        )
        m._REPORT_PATHS["joint"] = str(
            tmp_path / "reports" / "test_joint_portfolio_report.json"
        )
        m._REPORT_PATHS["forward_long"] = str(
            tmp_path / "reports" / "forward_long_report.json"
        )
        m._REPORT_PATHS["forward_short"] = str(
            tmp_path / "reports" / "forward_short_report.json"
        )
        m._REPORT_PATHS["forward_joint"] = str(
            tmp_path / "reports" / "forward_joint_portfolio_report.json"
        )

        for d in directions:
            _write_rule_set(str(tmp_path / f"{d}.json"), d)

        return orig_strategy, orig_report

    def _restore_paths(self, m, orig_strategy, orig_report):
        m._STRATEGY_PATHS.update(orig_strategy)
        m._REPORT_PATHS.update(orig_report)

    def test_plot_equity_called_for_train_validation_test(self, tmp_path):
        """plot_equity_curve must be called with 'train', 'validation', and 'test'."""
        trade_log = pd.DataFrame({
            "Rule_Index": [1, 2],
            "Net_PnL": [10.0, -5.0],
            "Equity_After": [1010.0, 1005.0],
        })
        mock_plot = self._run_with_mocks(
            tmp_path, trade_log, trade_log, trade_log)

        called_splits = [call[0][1] for call in mock_plot.call_args_list]
        assert "train" in called_splits, "Missing 'train' split"
        assert "validation" in called_splits, "Missing 'validation' split"
        assert "test" in called_splits, "Missing 'test' split"

    def test_plot_equity_called_exactly_three_times(self, tmp_path):
        """plot_equity_curve must be called exactly three times (one per split)."""
        trade_log = pd.DataFrame({
            "Rule_Index": [1, 2],
            "Net_PnL": [10.0, -5.0],
            "Equity_After": [1010.0, 1005.0],
        })
        mock_plot = self._run_with_mocks(
            tmp_path, trade_log, trade_log, trade_log)
        assert mock_plot.call_count == 3, \
            f"Expected 3 calls, got {mock_plot.call_count}"

    def test_empty_train_log_does_not_crash(self, tmp_path):
        """Empty train trade log must not raise an exception."""
        trade_log = pd.DataFrame({
            "Rule_Index": [1, 2],
            "Net_PnL": [10.0, -5.0],
            "Equity_After": [1010.0, 1005.0],
        })
        empty = pd.DataFrame()
        # This should not crash
        mock_plot = self._run_with_mocks(tmp_path, empty, trade_log, trade_log)
        # Even with empty train log, all three calls should still happen
        assert mock_plot.call_count == 3

    def test_empty_validation_log_does_not_crash(self, tmp_path):
        """Empty validation trade log must not raise an exception."""
        trade_log = pd.DataFrame({
            "Rule_Index": [1, 2],
            "Net_PnL": [10.0, -5.0],
            "Equity_After": [1010.0, 1005.0],
        })
        empty = pd.DataFrame()
        mock_plot = self._run_with_mocks(tmp_path, trade_log, empty, trade_log)
        assert mock_plot.call_count == 3

    def test_empty_test_log_does_not_crash(self, tmp_path):
        """Empty test trade log must not raise an exception."""
        trade_log = pd.DataFrame({
            "Rule_Index": [1, 2],
            "Net_PnL": [10.0, -5.0],
            "Equity_After": [1010.0, 1005.0],
        })
        empty = pd.DataFrame()
        mock_plot = self._run_with_mocks(tmp_path, trade_log, trade_log, empty)
        assert mock_plot.call_count == 3

    def test_all_empty_logs_does_not_crash(self, tmp_path):
        """All three empty trade logs must not raise an exception."""
        empty = pd.DataFrame()
        mock_plot = self._run_with_mocks(tmp_path, empty, empty, empty)
        assert mock_plot.call_count == 3

    def test_none_logs_do_not_crash(self, tmp_path):
        """None trade logs (from .get()) must not raise an exception."""
        trade_log = pd.DataFrame({
            "Rule_Index": [1, 2],
            "Net_PnL": [10.0, -5.0],
            "Equity_After": [1010.0, 1005.0],
        })
        # We use None to simulate missing keys; mock_evaluate returns None
        mock_plot = self._run_with_mocks(tmp_path, None, trade_log, trade_log)
        assert mock_plot.call_count == 3

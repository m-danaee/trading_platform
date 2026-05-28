"""
Unit tests for gpu_fuzzy_trader.phases.phase4_wf_optimizer
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase4_wf_optimizer import (
    Phase4NoFeasibleTrialError,
    WalkForwardRiskOptimizer,
    _build_candidate_rule_set,
    _load_rule_set,
    _normalize_capital_pct,
    _overalloc_penalty,
    _params_within_bounds,
    _select_pareto_trial,
    build_phase4_walk_forward_splits,
    build_tail_holdout_split,
    split_validation_walk_forward,
    _OUTPUT_PATHS,
)

optuna = pytest.importorskip("optuna")


def _make_rule_set(n_rules: int = 2, direction: str = "long") -> dict:
    rules = []
    for i in range(n_rules):
        rules.append({
            "conditions": [f"[feat_{i}] IS Very High"],
            "tp": _cfg.PHASE2_TP,
            "sl": _cfg.PHASE2_SL,
            "capital_pct": 20.0,
        })
    return {"direction": direction, "rules_set": rules}


def _make_val_df(
    rows_per_sym: int = 20,
    symbols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = ["SYM_A", "SYM_B"]

    dfs = []
    for sym in symbols:
        n = rows_per_sym
        open_next = rng.uniform(100, 200, size=n)
        data = {
            "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
            "symbol": sym,
            "label_open_next": open_next,
            "label_close_288": open_next * rng.uniform(0.95, 1.05, size=n),
            "label_min_288": open_next * rng.uniform(0.90, 1.00, size=n),
            "label_max_288": open_next * rng.uniform(1.00, 1.10, size=n),
            "label_max_before_min": rng.integers(0, 2, size=n).astype(float),
            "_symbol_bar_index": np.arange(n),
        }
        for i in range(3):
            data[f"feat_{i}"] = rng.uniform(0, 1, size=n)
        dfs.append(pd.DataFrame(data))

    return pd.concat(dfs, ignore_index=True)


class TestTailHoldoutSplit:
    def test_tail_is_last_fraction_per_symbol(self):
        val_df = _make_val_df(rows_per_sym=20, symbols=["A"])
        tail = build_tail_holdout_split(val_df, fraction=0.25)
        assert len(tail) == 5
        full = val_df.sort_values("datetime")
        assert tail["datetime"].min() >= full["datetime"].iloc[15]

    def test_phase4_splits_include_tail_when_enabled(self, monkeypatch):
        monkeypatch.setattr(_cfg, "PHASE4_INCLUDE_TAIL_HOLDOUT", True)
        val_df = _make_val_df(rows_per_sym=20, symbols=["A", "B"])
        splits = build_phase4_walk_forward_splits(val_df, k=2)
        assert len(splits) == 3


class TestSplitValidationWalkForward:
    def test_two_symbols_two_windows(self):
        val_df = _make_val_df(rows_per_sym=10, symbols=["A", "B"])
        windows = split_validation_walk_forward(val_df, k=2)
        assert len(windows) == 2
        assert len(windows[0]) == 10
        assert len(windows[1]) == 10
        for w in windows:
            assert set(w["symbol"].unique()) == {"A", "B"}

    def test_chronological_within_symbol(self):
        val_df = _make_val_df(rows_per_sym=8, symbols=["A"])
        windows = split_validation_walk_forward(val_df, k=2)
        first_half = windows[0]["datetime"].tolist()
        second_half = windows[1]["datetime"].tolist()
        assert max(first_half) <= min(second_half)

    def test_raises_when_symbol_too_short(self):
        val_df = _make_val_df(rows_per_sym=1, symbols=["A"])
        with pytest.raises(ValueError, match="at least"):
            split_validation_walk_forward(val_df, k=2)

    def test_raises_on_empty_df(self):
        with pytest.raises(ValueError, match="empty"):
            split_validation_walk_forward(pd.DataFrame(), k=2)


class TestOverallocPenalty:
    def test_no_penalty_under_100(self):
        params = [{"capital_pct": 40.0}, {"capital_pct": 50.0}]
        assert _overalloc_penalty(params) == 0.0

    def test_penalty_when_over_100(self):
        params = [{"capital_pct": 50.0}, {"capital_pct": 60.0}]
        expected = 10.0 / 100.0 * _cfg.PHASE4_TOTAL_CAP_PENALTY
        assert _overalloc_penalty(params) == pytest.approx(expected)


class TestSelectParetoTrial:
    def _trial(self, number: int, sortino: float, dd: float, trades: float = 50.0):
        return SimpleNamespace(
            number=number,
            state=SimpleNamespace(name="COMPLETE"),
            values=(sortino, dd, trades),
            user_attrs={"rule_set": [{"tp": 2.0}], "worst_pf": 1.1},
        )

    def test_filters_by_drawdown_then_max_sortino(self):
        study = MagicMock()
        study.trials = [
            self._trial(0, 1.0, 20.0),
            self._trial(1, 2.5, 12.0),
            self._trial(2, 3.0, 16.0),
        ]
        study.best_trials = study.trials

        selected = _select_pareto_trial(study, max_worst_dd_pct=15.0)
        assert selected.number == 1  # trial 2 excluded (dd=16 > 15)

    def test_raises_when_filter_empty(self):
        study = MagicMock()
        study.trials = [
            self._trial(0, 5.0, 20.0),
            self._trial(1, 4.0, 18.0),
        ]
        study.best_trials = study.trials

        with pytest.raises(Phase4NoFeasibleTrialError):
            _select_pareto_trial(study, max_worst_dd_pct=15.0)

    def test_no_feasible_trial_writes_non_optimized(self, tmp_path, monkeypatch):
        import gpu_fuzzy_trader.phases.phase4_wf_optimizer as m

        out_path = tmp_path / "long.json"
        monkeypatch.setitem(m._OUTPUT_PATHS, "long", str(out_path))
        monkeypatch.setattr(_cfg, "PHASE4_N_TRIALS", 3)
        monkeypatch.setattr(_cfg, "PHASE4_WF_SPLITS", 2)
        monkeypatch.setattr(_cfg, "PHASE4_MIN_WORST_FOLD_RETURN_PCT", 99.0)

        val_df = _make_val_df(rows_per_sym=30)
        opt = WalkForwardRiskOptimizer(
            val_df=val_df,
            rule_set=_make_rule_set(n_rules=1),
            direction="long",
            n_trials=3,
            n_splits=2,
        )
        result = opt.train()
        assert result["risk_optimized"] is False
        assert result["deployment_accepted"] is False
        assert out_path.exists()


class TestNormalizeCapitalPct:
    def test_scales_when_over_limit(self):
        rules = [
            {"capital_pct": 30.0},
            {"capital_pct": 30.0},
            {"capital_pct": 50.0},
        ]
        out = _normalize_capital_pct(rules)
        total = sum(r["capital_pct"] for r in out)
        assert total == pytest.approx(100.0, rel=1e-6)

    def test_unchanged_when_under_limit(self):
        rules = [{"capital_pct": 20.0}, {"capital_pct": 30.0}]
        out = _normalize_capital_pct(rules)
        assert out[0]["capital_pct"] == 20.0
        assert out[1]["capital_pct"] == 30.0


class TestParamsWithinBounds:
    def test_valid(self):
        rs = _make_rule_set()
        rs["rules_set"][0]["tp"] = 2.5
        rs["rules_set"][0]["sl"] = 2.0
        rs["rules_set"][0]["capital_pct"] = 20.0
        assert _params_within_bounds(rs) is True

    def test_invalid_capital(self):
        rs = _make_rule_set()
        rs["rules_set"][0]["capital_pct"] = 5.0
        assert _params_within_bounds(rs) is False


class TestBuildCandidateRuleSet:
    def test_conditions_frozen(self):
        rules = _make_rule_set()["rules_set"]
        params = [{"tp": 3.0, "sl": 1.5, "capital_pct": 25.0}] * 2
        built = _build_candidate_rule_set(rules, params)
        assert built[0]["conditions"] == rules[0]["conditions"]
        assert built[0]["tp"] == 3.0


class TestWalkForwardRiskOptimizer:
    def test_constructor_rejects_invalid_direction(self):
        val_df = _make_val_df()
        with pytest.raises(ValueError, match="direction"):
            WalkForwardRiskOptimizer(val_df, _make_rule_set(), "both")

    def test_constructor_requires_rules(self):
        val_df = _make_val_df()
        with pytest.raises(ValueError, match="rules_set"):
            WalkForwardRiskOptimizer(
                val_df, {"direction": "long", "rules_set": []}, "long"
            )

    def test_skip_if_valid_missing_file(self, tmp_path, monkeypatch):
        import gpu_fuzzy_trader.phases.phase4_wf_optimizer as m

        monkeypatch.setitem(
            m._OUTPUT_PATHS, "long", str(tmp_path / "missing.json")
        )
        assert WalkForwardRiskOptimizer.skip_if_valid("long") is None

    def test_skip_if_valid_risk_optimized(self, tmp_path, monkeypatch):
        import gpu_fuzzy_trader.phases.phase4_wf_optimizer as m

        path = tmp_path / "long.json"
        data = _make_rule_set()
        data["risk_optimized"] = True
        for r in data["rules_set"]:
            r["tp"] = 2.5
            r["sl"] = 2.0
            r["capital_pct"] = 20.0
        path.write_text(json.dumps(data), encoding="utf-8")
        monkeypatch.setitem(m._OUTPUT_PATHS, "long", str(path))

        result = WalkForwardRiskOptimizer.skip_if_valid("long")
        assert result is not None
        assert result["risk_optimized"] is True

    def test_train_integration(self, tmp_path, monkeypatch):
        import gpu_fuzzy_trader.phases.phase4_wf_optimizer as m

        out_path = tmp_path / "long.json"
        monkeypatch.setitem(m._OUTPUT_PATHS, "long", str(out_path))
        monkeypatch.setattr(_cfg, "PHASE4_N_TRIALS", 5)
        monkeypatch.setattr(_cfg, "PHASE4_WF_SPLITS", 2)
        monkeypatch.setattr(_cfg, "PHASE4_N_JOBS", 1)
        monkeypatch.setattr(_cfg, "PHASE4_MIN_WORST_FOLD_RETURN_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE4_MIN_WORST_FOLD_PF", 0.0)
        monkeypatch.setattr(_cfg, "PHASE4_MIN_WORST_TRADES", 1)

        val_df = _make_val_df(rows_per_sym=30)
        opt = WalkForwardRiskOptimizer(
            val_df=val_df,
            rule_set=_make_rule_set(n_rules=1),
            direction="long",
            n_trials=5,
            n_splits=2,
        )
        result = opt.train()

        assert result["risk_optimized"] is True
        assert out_path.exists()
        assert len(result["rules_set"]) == 1
        assert opt.study is not None
        assert opt.selected_trial is not None

    def test_parallel_smoke(self, tmp_path, monkeypatch):
        import gpu_fuzzy_trader.phases.phase4_wf_optimizer as m

        out_path = tmp_path / "short.json"
        monkeypatch.setitem(m._OUTPUT_PATHS, "short", str(out_path))
        monkeypatch.setattr(_cfg, "PHASE4_N_JOBS", 2)
        monkeypatch.setattr(_cfg, "PHASE4_MIN_WORST_FOLD_RETURN_PCT", -100.0)
        monkeypatch.setattr(_cfg, "PHASE4_MIN_WORST_FOLD_PF", 0.0)
        monkeypatch.setattr(_cfg, "PHASE4_MIN_WORST_TRADES", 1)

        val_df = _make_val_df(rows_per_sym=30)
        opt = WalkForwardRiskOptimizer(
            val_df=val_df,
            rule_set=_make_rule_set(n_rules=1, direction="short"),
            direction="short",
            n_trials=10,
            n_splits=2,
            seed=99,
        )
        result = opt.train()
        assert result["risk_optimized"] is True


class TestLoadRuleSet:
    def test_missing_returns_none(self, tmp_path):
        assert _load_rule_set(str(tmp_path / "nope.json")) is None

    def test_invalid_json_returns_none(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("not json", encoding="utf-8")
        assert _load_rule_set(str(path)) is None

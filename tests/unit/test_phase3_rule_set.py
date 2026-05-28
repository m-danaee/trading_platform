"""
Unit tests for gpu_fuzzy_trader.phases.phase3_rule_set.Rule_Set_Selector

Tests cover:
  - Schema validation (_validate_rule_set_schema)
  - Duplicate rule detection (_has_duplicate_rules)
  - Symbol coverage counting (_count_symbols_with_trades)
  - Output dict building (_build_output_dict)
  - Random rule set generation (_random_rule_set)
  - Rule set to engine format conversion (_rule_set_to_engine_format)
  - NSGA-II helpers (_dominates, _non_dominated_sort, _crowding_distance)
  - Rule_Set_Selector constructor validation
  - Rule_Set_Selector.run() integration (tiny pop/gen)
  - Rule_Set_Selector.load_rule_set() / skip_if_valid() logic
  - Output JSON format compatibility
"""

from __future__ import annotations

import json
import os
import random

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase3_rule_set import (
    Rule_Set_Selector,
    _build_output_dict,
    _conditions_key,
    _count_symbols_with_trades,
    _crowding_distance,
    _dominates,
    _has_duplicate_rules,
    _non_dominated_sort,
    _random_rule_set,
    _rule_set_to_engine_format,
    _symbol_consistency_penalty,
    _validate_rule_set_schema,
    _OUTPUT_PATHS,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_pool(n: int = 5) -> list[dict]:
    """Create a minimal pool of n rules."""
    pool = []
    for i in range(n):
        pool.append({
            "conditions": [f"[feat_{i}] IS Very High"],
            "tp": _cfg.PHASE2_TP,
            "sl": _cfg.PHASE2_SL,
            "capital_pct": _cfg.PHASE2_CAPITAL_PCT,
        })
    return pool


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
            "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
            "symbol": sym,
            "label_open_next": open_next,
            "label_close_288": close_288,
            "label_min_288": min_288,
            "label_max_288": max_288,
            "label_max_before_min": max_before_min.astype(float),
            "_symbol_bar_index": np.arange(n),
        }
        # Add feature columns that match pool conditions
        for i in range(12):
            data[f"feat_{i}"] = rng.uniform(0, 1, size=n)

        dfs.append(pd.DataFrame(data))

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: _validate_rule_set_schema
# ---------------------------------------------------------------------------

class TestValidateRuleSetSchema:
    def _valid_data(self):
        return {
            "direction": "long",
            "rules_set": [
                {
                    "tp": 4.0,
                    "sl": 2.0,
                    "capital_pct": 50.0,
                    "conditions": ["[feat_0] IS Very High"],
                },
                {
                    "tp": 4.0,
                    "sl": 2.0,
                    "capital_pct": 50.0,
                    "conditions": ["[feat_1] IS Very High"],
                },
            ],
        }

    def test_valid_schema_passes(self):
        _validate_rule_set_schema(self._valid_data(), "test.json")

    def test_not_a_dict_raises(self):
        with pytest.raises(ValueError, match="must be a JSON object"):
            _validate_rule_set_schema(["not", "a", "dict"], "test.json")

    def test_missing_direction_raises(self):
        data = self._valid_data()
        del data["direction"]
        with pytest.raises(ValueError, match="missing top-level keys"):
            _validate_rule_set_schema(data, "test.json")

    def test_missing_rules_set_raises(self):
        data = self._valid_data()
        del data["rules_set"]
        with pytest.raises(ValueError, match="missing top-level keys"):
            _validate_rule_set_schema(data, "test.json")

    def test_invalid_direction_raises(self):
        data = self._valid_data()
        data["direction"] = "both"
        with pytest.raises(ValueError, match="must be 'long' or 'short'"):
            _validate_rule_set_schema(data, "test.json")

    def test_rules_set_not_list_raises(self):
        data = self._valid_data()
        data["rules_set"] = "not_a_list"
        with pytest.raises(ValueError, match="must be a list"):
            _validate_rule_set_schema(data, "test.json")

    def test_too_few_rules_raises(self):
        data = self._valid_data()
        data["rules_set"] = data["rules_set"][:1]
        with pytest.raises(ValueError, match="must have 2"):
            _validate_rule_set_schema(data, "test.json")

    def test_too_many_rules_raises(self):
        data = self._valid_data()
        rule = data["rules_set"][0].copy()
        data["rules_set"] = [rule] * 6
        with pytest.raises(ValueError, match="must have 2"):
            _validate_rule_set_schema(data, "test.json")

    def test_rule_missing_conditions_raises(self):
        data = self._valid_data()
        del data["rules_set"][0]["conditions"]
        with pytest.raises(ValueError, match="missing keys"):
            _validate_rule_set_schema(data, "test.json")

    def test_empty_conditions_raises(self):
        data = self._valid_data()
        data["rules_set"][0]["conditions"] = []
        with pytest.raises(ValueError, match="non-empty list"):
            _validate_rule_set_schema(data, "test.json")

    def test_five_rules_valid(self):
        data = self._valid_data()
        rule = data["rules_set"][0].copy()
        data["rules_set"] = [
            {**rule, "conditions": [f"[feat_{i}] IS Very High"]}
            for i in range(5)
        ]
        _validate_rule_set_schema(data, "test.json")  # should not raise


# ---------------------------------------------------------------------------
# Tests: _has_duplicate_rules
# ---------------------------------------------------------------------------

class TestHasDuplicateRules:
    def test_no_duplicates(self):
        rules = [
            {"conditions": ["[feat_0] IS Very High"]},
            {"conditions": ["[feat_1] IS Very High"]},
        ]
        assert not _has_duplicate_rules(rules)

    def test_exact_duplicate(self):
        rules = [
            {"conditions": ["[feat_0] IS Very High"]},
            {"conditions": ["[feat_0] IS Very High"]},
        ]
        assert _has_duplicate_rules(rules)

    def test_order_independent_duplicate(self):
        """Two rules with same conditions in different order are duplicates."""
        rules = [
            {"conditions": ["[feat_0] IS Very High", "[feat_1] IS Low"]},
            {"conditions": ["[feat_1] IS Low", "[feat_0] IS Very High"]},
        ]
        assert _has_duplicate_rules(rules)

    def test_single_rule_no_duplicate(self):
        rules = [{"conditions": ["[feat_0] IS Very High"]}]
        assert not _has_duplicate_rules(rules)

    def test_empty_list_no_duplicate(self):
        assert not _has_duplicate_rules([])

    def test_three_rules_one_duplicate(self):
        rules = [
            {"conditions": ["[feat_0] IS Very High"]},
            {"conditions": ["[feat_1] IS Low"]},
            {"conditions": ["[feat_0] IS Very High"]},
        ]
        assert _has_duplicate_rules(rules)


# ---------------------------------------------------------------------------
# Tests: _count_symbols_with_trades
# ---------------------------------------------------------------------------

class TestCountSymbolsWithTrades:
    def test_all_symbols_have_trades(self):
        metrics = {
            "per_symbol_metrics": {
                "SYM_A": {"trade_count": 5},
                "SYM_B": {"trade_count": 3},
            }
        }
        assert _count_symbols_with_trades(metrics) == 2

    def test_no_symbols_have_trades(self):
        metrics = {
            "per_symbol_metrics": {
                "SYM_A": {"trade_count": 0},
                "SYM_B": {"trade_count": 0},
            }
        }
        assert _count_symbols_with_trades(metrics) == 0

    def test_partial_coverage(self):
        metrics = {
            "per_symbol_metrics": {
                "SYM_A": {"trade_count": 5},
                "SYM_B": {"trade_count": 0},
                "SYM_C": {"trade_count": 2},
            }
        }
        assert _count_symbols_with_trades(metrics) == 2

    def test_empty_per_symbol_metrics(self):
        metrics = {"per_symbol_metrics": {}}
        assert _count_symbols_with_trades(metrics) == 0

    def test_missing_per_symbol_metrics(self):
        metrics = {}
        assert _count_symbols_with_trades(metrics) == 0


class TestSymbolConsistencyPenalty:
    def _metrics(self, symbols: set[str]) -> dict:
        return {
            "per_symbol_metrics": {
                s: {"trade_count": 1} for s in symbols
            }
        }

    def test_full_overlap_zero_penalty(self):
        syms = {"A", "B"}
        pen = _symbol_consistency_penalty(
            self._metrics(syms), self._metrics(syms))
        assert pen == 0.0

    def test_zero_overlap_max_penalty(self):
        pen = _symbol_consistency_penalty(
            self._metrics({"A", "B"}), self._metrics({"C", "D"}))
        assert pen == _cfg.PHASE3_SYMBOL_CONSISTENCY_WEIGHT

    def test_partial_overlap_between_extremes(self):
        train = self._metrics({"A", "B"})
        val = self._metrics({"B", "C"})
        pen = _symbol_consistency_penalty(train, val)
        assert 0.0 < pen < _cfg.PHASE3_SYMBOL_CONSISTENCY_WEIGHT


# ---------------------------------------------------------------------------
# Tests: _build_output_dict
# ---------------------------------------------------------------------------

class TestBuildOutputDict:
    def test_output_has_direction_and_rules_set(self):
        pool = _make_pool(2)
        result = _build_output_dict(pool, "long")
        assert "direction" in result
        assert "rules_set" in result
        assert result["risk_optimized"] is False

    def test_direction_is_correct(self):
        pool = _make_pool(2)
        assert _build_output_dict(pool, "long")["direction"] == "long"
        assert _build_output_dict(pool, "short")["direction"] == "short"

    def test_rules_set_has_correct_keys(self):
        pool = _make_pool(3)
        result = _build_output_dict(pool, "long")
        for rule in result["rules_set"]:
            assert "tp" in rule
            assert "sl" in rule
            assert "capital_pct" in rule
            assert "conditions" in rule

    def test_uses_phase2_defaults_when_missing(self):
        pool = [{"conditions": ["[feat_0] IS Very High"]}]
        result = _build_output_dict(pool, "long")
        rule = result["rules_set"][0]
        assert rule["tp"] == _cfg.PHASE2_TP
        assert rule["sl"] == _cfg.PHASE2_SL
        assert rule["capital_pct"] == min(
            _cfg.PHASE2_CAPITAL_PCT, _cfg.PHASE3_MAX_CAPITAL_PCT_PER_RULE)

    def test_conditions_preserved(self):
        pool = [{"conditions": ["[feat_0] IS Very High", "[feat_1] IS Low"],
                 "tp": 4.0, "sl": 2.0, "capital_pct": 50.0}]
        result = _build_output_dict(pool, "long")
        assert result["rules_set"][0]["conditions"] == [
            "[feat_0] IS Very High", "[feat_1] IS Low"]


# ---------------------------------------------------------------------------
# Tests: _rule_set_to_engine_format
# ---------------------------------------------------------------------------

class TestRuleSetToEngineFormat:
    def test_returns_list_of_dicts(self):
        pool = _make_pool(3)
        result = _rule_set_to_engine_format(pool)
        assert isinstance(result, list)
        assert len(result) == 3

    def test_each_dict_has_required_keys(self):
        pool = _make_pool(2)
        result = _rule_set_to_engine_format(pool)
        for rule in result:
            assert "conditions" in rule
            assert "tp" in rule
            assert "sl" in rule
            assert "capital_pct" in rule

    def test_uses_defaults_when_missing(self):
        pool = [{"conditions": ["[feat_0] IS Very High"]}]
        result = _rule_set_to_engine_format(pool)
        assert result[0]["tp"] == _cfg.PHASE2_TP
        assert result[0]["sl"] == _cfg.PHASE2_SL
        assert result[0]["capital_pct"] == _cfg.PHASE2_CAPITAL_PCT

    def test_strips_extra_keys(self):
        pool = [{
            "conditions": ["[feat_0] IS Very High"],
            "tp": 4.0, "sl": 2.0, "capital_pct": 50.0,
            "chromosome": [1, 2, 3],
            "objectives": {"sortino_ratio": 5.0, "total_return_pct": 5.0},
        }]
        result = _rule_set_to_engine_format(pool)
        assert "chromosome" not in result[0]
        assert "objectives" not in result[0]


# ---------------------------------------------------------------------------
# Tests: _random_rule_set
# ---------------------------------------------------------------------------

class TestRandomRuleSet:
    def test_size_within_bounds(self):
        pool = _make_pool(8)
        rng = random.Random(42)
        for _ in range(20):
            rs = _random_rule_set(pool, rng, min_rules=2, max_rules=5)
            assert 2 <= len(rs) <= 5

    def test_no_duplicate_conditions(self):
        pool = _make_pool(8)
        rng = random.Random(42)
        for _ in range(20):
            rs = _random_rule_set(pool, rng, min_rules=2, max_rules=5)
            assert not _has_duplicate_rules(rs)

    def test_rules_come_from_pool(self):
        pool = _make_pool(6)
        pool_keys = {_conditions_key(r["conditions"]) for r in pool}
        rng = random.Random(0)
        for _ in range(10):
            rs = _random_rule_set(pool, rng, min_rules=2, max_rules=4)
            for rule in rs:
                assert _conditions_key(rule["conditions"]) in pool_keys

    def test_min_rules_respected_with_small_pool(self):
        pool = _make_pool(2)
        rng = random.Random(0)
        rs = _random_rule_set(pool, rng, min_rules=2, max_rules=5)
        assert len(rs) >= 2


# ---------------------------------------------------------------------------
# Tests: NSGA-II helpers
# ---------------------------------------------------------------------------

class TestDominates:
    def test_a_dominates_b(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([2.0, 3.0, 4.0])
        assert _dominates(a, b)
        assert not _dominates(b, a)

    def test_equal_does_not_dominate(self):
        a = np.array([1.0, 2.0, 3.0])
        assert not _dominates(a, a)

    def test_incomparable(self):
        a = np.array([1.0, 3.0])
        b = np.array([2.0, 1.0])
        assert not _dominates(a, b)
        assert not _dominates(b, a)


class TestNonDominatedSort:
    def test_single_solution(self):
        obj = np.array([[1.0, 2.0, 3.0]])
        fronts = _non_dominated_sort(obj)
        assert fronts[0] == [0]

    def test_two_solutions_one_dominates(self):
        obj = np.array([[1.0, 1.0], [2.0, 2.0]])
        fronts = _non_dominated_sort(obj)
        assert 0 in fronts[0]
        assert 1 not in fronts[0]

    def test_two_incomparable_solutions(self):
        obj = np.array([[1.0, 3.0], [3.0, 1.0]])
        fronts = _non_dominated_sort(obj)
        assert set(fronts[0]) == {0, 1}

    def test_empty_objectives(self):
        obj = np.empty((0, 3))
        fronts = _non_dominated_sort(obj)
        assert fronts == [[]]


class TestCrowdingDistance:
    def test_two_solutions_get_inf(self):
        obj = np.array([[1.0, 2.0], [3.0, 4.0]])
        cd = _crowding_distance(obj, [0, 1])
        assert np.all(np.isinf(cd))

    def test_boundary_solutions_get_inf(self):
        obj = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
        cd = _crowding_distance(obj, [0, 1, 2])
        assert np.isinf(cd[0])
        assert np.isinf(cd[2])
        assert np.isfinite(cd[1])


# ---------------------------------------------------------------------------
# Tests: Rule_Set_Selector constructor
# ---------------------------------------------------------------------------

class TestRuleSetSelectorInit:
    def test_invalid_direction_raises(self):
        pool = _make_pool(3)
        df = _make_df()
        with pytest.raises(ValueError, match="direction must be"):
            Rule_Set_Selector(df, df, pool, "both")

    def test_empty_pool_raises(self):
        df = _make_df()
        with pytest.raises(ValueError, match="pool must not be empty"):
            Rule_Set_Selector(df, df, [], "long")

    def test_pool_too_small_raises(self):
        df = _make_df()
        pool = _make_pool(1)  # need at least PHASE3_MIN_RULES=2
        with pytest.raises(ValueError, match="at least"):
            Rule_Set_Selector(df, df, pool, "long")

    def test_valid_construction_long(self):
        pool = _make_pool(4)
        df = _make_df()
        sel = Rule_Set_Selector(
            df, df, pool, "long", refine_pop_size=4, refine_generations=2
        )
        assert sel.direction == "long"
        assert sel.refine_pop_size == 4
        assert sel.refine_generations == 2

    def test_valid_construction_short(self):
        pool = _make_pool(4)
        df = _make_df()
        sel = Rule_Set_Selector(
            df, df, pool, "short", refine_pop_size=4, refine_generations=2
        )
        assert sel.direction == "short"

    def test_default_refine_pop_size_from_config(self):
        pool = _make_pool(4)
        df = _make_df()
        sel = Rule_Set_Selector(df, df, pool, "long")
        assert sel.refine_pop_size == _cfg.PHASE3_REFINE_POP_SIZE

    def test_default_refine_generations_from_config(self):
        pool = _make_pool(4)
        df = _make_df()
        sel = Rule_Set_Selector(df, df, pool, "long")
        assert sel.refine_generations == _cfg.PHASE3_REFINE_GENERATIONS


# ---------------------------------------------------------------------------
# Tests: Rule_Set_Selector.run() — integration (tiny pop/gen)
# ---------------------------------------------------------------------------

class TestRuleSetSelectorRun:
    """Integration tests using tiny population and generation counts."""

    def _make_selector(
        self,
        direction: str = "long",
        n_pool: int = 6,
        refine_pop_size: int = 6,
        refine_generations: int = 3,
    ) -> Rule_Set_Selector:
        pool = _make_pool(n_pool)
        df = _make_df(n_rows=200, symbols=["SYM_A", "SYM_B"])
        return Rule_Set_Selector(
            df, df, pool, direction,
            refine_pop_size=refine_pop_size,
            refine_generations=refine_generations,
            seed=42,
        )

    def test_run_returns_dict(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            sel = self._make_selector("long")
            result = sel.run()
            assert isinstance(result, dict)
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_run_output_has_direction_and_rules_set(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            sel = self._make_selector("long")
            result = sel.run()
            assert "direction" in result
            assert "rules_set" in result
            assert result["direction"] == "long"
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_run_rules_set_size_within_bounds(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            sel = self._make_selector("long")
            result = sel.run()
            n = len(result["rules_set"])
            assert _cfg.PHASE3_MIN_RULES <= n <= _cfg.PHASE3_MAX_RULES, (
                f"Expected {_cfg.PHASE3_MIN_RULES}–{_cfg.PHASE3_MAX_RULES} rules, got {n}"
            )
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_run_no_duplicate_rules(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            sel = self._make_selector("long")
            result = sel.run()
            assert not _has_duplicate_rules(result["rules_set"]), (
                "Output rule set contains duplicate rules"
            )
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_greedy_eval_count(self):
        from gpu_fuzzy_trader.phases.phase3_greedy import greedy_rule_set_search

        pool = _make_pool(10)
        df = _make_df(n_rows=200)
        p3 = __import__(
            "gpu_fuzzy_trader.phases.phase3_rule_set",
            fromlist=["_build_phase3_engines"],
        )
        val_eng, train_eng, _, _, cache = p3._build_phase3_engines(
            df, df, "long", pool)
        _, n_evals = greedy_rule_set_search(
            pool, val_eng, train_eng, 2, 5, use_batch=False, cache=cache,
        )
        # 10 singles + 9 pairs + 8 triples + 7 quads + 6 fives = 40
        assert n_evals == 40

    def test_run_creates_output_file(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        out_path = str(tmp_path / "long.json")
        m._OUTPUT_PATHS["long"] = out_path
        try:
            sel = self._make_selector("long")
            sel.run()
            assert os.path.exists(out_path)
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_run_output_file_is_valid_json(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        out_path = str(tmp_path / "long.json")
        m._OUTPUT_PATHS["long"] = out_path
        try:
            sel = self._make_selector("long")
            sel.run()
            with open(out_path) as fh:
                data = json.load(fh)
            assert isinstance(data, dict)
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_run_output_passes_schema_validation(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        out_path = str(tmp_path / "long.json")
        m._OUTPUT_PATHS["long"] = out_path
        try:
            sel = self._make_selector("long")
            result = sel.run()
            _validate_rule_set_schema(result, "test")  # should not raise
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_run_each_rule_has_required_keys(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            sel = self._make_selector("long")
            result = sel.run()
            for rule in result["rules_set"]:
                assert "tp" in rule
                assert "sl" in rule
                assert "capital_pct" in rule
                assert "conditions" in rule
                assert len(rule["conditions"]) > 0
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_run_short_direction(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["short"] = str(tmp_path / "short.json")
        try:
            sel = self._make_selector("short")
            result = sel.run()
            assert result["direction"] == "short"
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_run_uses_phase2_static_risk_params(self, tmp_path):
        """Phase 3 output should use Phase 2 static TP/SL/capital_pct."""
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            sel = self._make_selector("long")
            result = sel.run()
            for rule in result["rules_set"]:
                assert rule["tp"] == _cfg.PHASE2_TP
                assert rule["sl"] == _cfg.PHASE2_SL
                assert rule["capital_pct"] == min(
                    _cfg.PHASE2_CAPITAL_PCT, _cfg.PHASE3_MAX_CAPITAL_PCT_PER_RULE)
        finally:
            m._OUTPUT_PATHS.update(original)


# ---------------------------------------------------------------------------
# Tests: Rule_Set_Selector.load_rule_set() / skip_if_valid()
# ---------------------------------------------------------------------------

class TestLoadRuleSet:
    def _write_valid_rule_set(self, path: str, direction: str = "long"):
        data = {
            "direction": direction,
            "rules_set": [
                {
                    "tp": 4.0,
                    "sl": 2.0,
                    "capital_pct": 50.0,
                    "conditions": ["[feat_0] IS Very High"],
                },
                {
                    "tp": 4.0,
                    "sl": 2.0,
                    "capital_pct": 50.0,
                    "conditions": ["[feat_1] IS Low"],
                },
            ],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh)

    def test_load_returns_none_when_missing(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "nonexistent.json")
        try:
            result = Rule_Set_Selector.load_rule_set("long")
            assert result is None
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_load_returns_data_when_valid(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        path = str(tmp_path / "long.json")
        self._write_valid_rule_set(path, "long")
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = path
        try:
            result = Rule_Set_Selector.load_rule_set("long")
            assert result is not None
            assert result["direction"] == "long"
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_load_raises_on_corrupted_file(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        path = str(tmp_path / "long.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write("{corrupted json")
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = path
        try:
            with pytest.raises(ValueError, match="unreadable or corrupted"):
                Rule_Set_Selector.load_rule_set("long")
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_load_raises_on_invalid_schema(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        path = str(tmp_path / "long.json")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump({"direction": "long", "rules_set": []}, fh)
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = path
        try:
            with pytest.raises(ValueError):
                Rule_Set_Selector.load_rule_set("long")
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_load_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            Rule_Set_Selector.load_rule_set("both")


class TestSkipIfValid:
    def _write_valid_rule_set(self, path: str, direction: str):
        data = {
            "direction": direction,
            "rules_set": [
                {"tp": 4.0, "sl": 2.0, "capital_pct": 50.0,
                 "conditions": ["[feat_0] IS Very High"]},
                {"tp": 4.0, "sl": 2.0, "capital_pct": 50.0,
                 "conditions": ["[feat_1] IS Low"]},
            ],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh)

    def test_returns_none_when_neither_exists(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        m._OUTPUT_PATHS["short"] = str(tmp_path / "short.json")
        try:
            result = Rule_Set_Selector.skip_if_valid()
            assert result is None
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_returns_both_when_both_valid(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        long_path = str(tmp_path / "long.json")
        short_path = str(tmp_path / "short.json")
        self._write_valid_rule_set(long_path, "long")
        self._write_valid_rule_set(short_path, "short")
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = long_path
        m._OUTPUT_PATHS["short"] = short_path
        try:
            result = Rule_Set_Selector.skip_if_valid()
            assert result is not None
            assert "long" in result
            assert "short" in result
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_returns_partial_when_only_long_exists(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        long_path = str(tmp_path / "long.json")
        self._write_valid_rule_set(long_path, "long")
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = long_path
        m._OUTPUT_PATHS["short"] = str(tmp_path / "short.json")
        try:
            result = Rule_Set_Selector.skip_if_valid()
            assert result is not None
            assert "long" in result
            assert "short" not in result
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_returns_partial_when_only_short_exists(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        short_path = str(tmp_path / "short.json")
        self._write_valid_rule_set(short_path, "short")
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        m._OUTPUT_PATHS["short"] = short_path
        try:
            result = Rule_Set_Selector.skip_if_valid()
            assert result is not None
            assert "short" in result
            assert "long" not in result
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_returns_none_when_both_corrupted(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        long_path = str(tmp_path / "long.json")
        short_path = str(tmp_path / "short.json")
        for p in (long_path, short_path):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w") as fh:
                fh.write("{bad json")
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = long_path
        m._OUTPUT_PATHS["short"] = short_path
        try:
            result = Rule_Set_Selector.skip_if_valid()
            assert result is None
        finally:
            m._OUTPUT_PATHS.update(original)

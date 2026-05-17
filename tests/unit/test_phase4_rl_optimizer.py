"""
Unit tests for gpu_fuzzy_trader.phases.phase4_rl_optimizer

Tests cover:
  - find_elbow_point: all edge cases and normal cases
  - TradingEnv: construction, reset, step, action clipping, state vector shape
  - _params_within_bounds: valid/invalid bounds checking
  - _load_rule_set: file loading and error handling
  - RL_Agent: constructor validation, skip_if_valid, train() integration
  - Action bounds enforcement
  - State vector completeness
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase4_rl_optimizer import (
    RL_Agent,
    TradingEnv,
    find_elbow_point,
    _load_rule_set,
    _params_within_bounds,
    _OUTPUT_PATHS,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_rule_set(n_rules: int = 2, direction: str = "long") -> dict:
    """Create a minimal rule set dict."""
    rules = []
    for i in range(n_rules):
        rules.append({
            "conditions": [f"[feat_{i}] IS Very High"],
            "tp": _cfg.PHASE2_TP,
            "sl": _cfg.PHASE2_SL,
            "capital_pct": _cfg.PHASE2_CAPITAL_PCT,
        })
    return {"direction": direction, "rules_set": rules}


def _make_df(
    n_rows: int = 100,
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
        for i in range(5):
            data[f"feat_{i}"] = rng.uniform(0, 1, size=n)

        dfs.append(pd.DataFrame(data))

    return pd.concat(dfs, ignore_index=True)


# ---------------------------------------------------------------------------
# Tests: find_elbow_point
# ---------------------------------------------------------------------------

class TestFindElbowPoint:
    """Tests for the find_elbow_point function."""

    def test_empty_list_returns_zero(self):
        assert find_elbow_point([]) == 0

    def test_single_element_returns_zero(self):
        assert find_elbow_point([5.0]) == 0

    def test_two_elements_returns_zero(self):
        assert find_elbow_point([1.0, 2.0]) == 0

    def test_all_equal_plateau_returns_zero(self):
        """Immediately plateauing curve → first point."""
        assert find_elbow_point([3.0, 3.0, 3.0, 3.0, 3.0]) == 0

    def test_monotonically_increasing_returns_last(self):
        """Monotonically increasing → last point."""
        curve = [1.0, 2.0, 3.0, 4.0, 5.0]
        result = find_elbow_point(curve)
        assert result == len(curve) - 1

    def test_classic_elbow_shape(self):
        """
        Classic elbow: rapid rise then plateau.
        Elbow should be near the transition point.
        """
        # Rapid rise then plateau
        curve = [0.0, 5.0, 8.0, 9.0, 9.5, 9.7, 9.8, 9.9, 9.95, 10.0]
        result = find_elbow_point(curve)
        # Elbow should be in the first half (transition region)
        assert 0 < result < len(curve) - 1

    def test_returns_valid_index(self):
        """Result must always be a valid index."""
        for length in range(1, 15):
            curve = list(range(length))
            result = find_elbow_point(curve)
            assert 0 <= result < length, f"Invalid index {result} for length {length}"

    def test_returns_int(self):
        """Return type must be int."""
        result = find_elbow_point([1.0, 2.0, 3.0])
        assert isinstance(result, int)

    def test_negative_values_handled(self):
        """Negative values in curve should not cause errors."""
        curve = [-10.0, -5.0, -2.0, -1.0, -0.5, -0.1]
        result = find_elbow_point(curve)
        assert 0 <= result < len(curve)

    def test_decreasing_curve(self):
        """Monotonically decreasing curve."""
        curve = [10.0, 8.0, 6.0, 4.0, 2.0, 1.0]
        result = find_elbow_point(curve)
        assert 0 <= result < len(curve)

    def test_sharp_elbow_at_known_position(self):
        """
        Curve with a sharp elbow at index 2:
        [0, 1, 10, 10.1, 10.2] — big jump at index 2.
        The elbow should be at or near index 2.
        """
        curve = [0.0, 1.0, 10.0, 10.1, 10.2, 10.3]
        result = find_elbow_point(curve)
        # The elbow should be near the big jump (index 1 or 2)
        assert result in (1, 2), f"Expected elbow near index 1-2, got {result}"

    def test_three_elements_middle_elbow(self):
        """Three elements with middle as elbow."""
        # [0, 5, 5] — elbow at index 1 (big jump then plateau)
        curve = [0.0, 5.0, 5.0]
        result = find_elbow_point(curve)
        assert 0 <= result < 3

    def test_large_curve_returns_valid_index(self):
        """Large curve should return a valid index."""
        curve = [float(i) ** 0.5 for i in range(100)]
        result = find_elbow_point(curve)
        assert 0 <= result < 100

    def test_curve_with_noise(self):
        """Noisy curve should still return a valid index."""
        rng = np.random.default_rng(42)
        base = [float(i) for i in range(20)]
        noise = rng.normal(0, 0.1, 20).tolist()
        curve = [b + n for b, n in zip(base, noise)]
        result = find_elbow_point(curve)
        assert 0 <= result < 20

    def test_plateau_after_rise(self):
        """Rise then long plateau — elbow near the transition."""
        curve = [0.0, 3.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        result = find_elbow_point(curve)
        # Elbow should be in the rising portion
        assert 0 <= result < len(curve)

    def test_two_equal_elements_returns_zero(self):
        """Two equal elements → plateau → first point."""
        assert find_elbow_point([5.0, 5.0]) == 0


# ---------------------------------------------------------------------------
# Tests: _params_within_bounds
# ---------------------------------------------------------------------------

class TestParamsWithinBounds:
    def _make_valid_rule_set(self) -> dict:
        return {
            "direction": "long",
            "rules_set": [
                {
                    "tp": _cfg.PHASE4_TP_MIN,
                    "sl": _cfg.PHASE4_SL_MIN,
                    "capital_pct": _cfg.PHASE4_CAPITAL_PCT_MIN,
                    "conditions": ["[feat_0] IS Very High"],
                },
                {
                    "tp": _cfg.PHASE4_TP_MAX,
                    "sl": _cfg.PHASE4_SL_MAX,
                    "capital_pct": _cfg.PHASE4_CAPITAL_PCT_MAX,
                    "conditions": ["[feat_1] IS Low"],
                },
            ],
        }

    def test_valid_bounds_returns_true(self):
        assert _params_within_bounds(self._make_valid_rule_set()) is True

    def test_tp_below_min_returns_false(self):
        rs = self._make_valid_rule_set()
        rs["rules_set"][0]["tp"] = _cfg.PHASE4_TP_MIN - 0.1
        assert _params_within_bounds(rs) is False

    def test_tp_above_max_returns_false(self):
        rs = self._make_valid_rule_set()
        rs["rules_set"][0]["tp"] = _cfg.PHASE4_TP_MAX + 0.1
        assert _params_within_bounds(rs) is False

    def test_sl_below_min_returns_false(self):
        rs = self._make_valid_rule_set()
        rs["rules_set"][0]["sl"] = _cfg.PHASE4_SL_MIN - 0.1
        assert _params_within_bounds(rs) is False

    def test_sl_above_max_returns_false(self):
        rs = self._make_valid_rule_set()
        rs["rules_set"][0]["sl"] = _cfg.PHASE4_SL_MAX + 0.1
        assert _params_within_bounds(rs) is False

    def test_capital_pct_below_min_returns_false(self):
        rs = self._make_valid_rule_set()
        rs["rules_set"][0]["capital_pct"] = _cfg.PHASE4_CAPITAL_PCT_MIN - 0.1
        assert _params_within_bounds(rs) is False

    def test_capital_pct_above_max_returns_false(self):
        rs = self._make_valid_rule_set()
        rs["rules_set"][0]["capital_pct"] = _cfg.PHASE4_CAPITAL_PCT_MAX + 0.1
        assert _params_within_bounds(rs) is False

    def test_empty_rules_returns_false(self):
        assert _params_within_bounds({"rules_set": []}) is False

    def test_missing_rules_set_returns_false(self):
        assert _params_within_bounds({}) is False

    def test_midpoint_values_valid(self):
        tp_mid = (_cfg.PHASE4_TP_MIN + _cfg.PHASE4_TP_MAX) / 2.0
        sl_mid = (_cfg.PHASE4_SL_MIN + _cfg.PHASE4_SL_MAX) / 2.0
        cap_mid = (_cfg.PHASE4_CAPITAL_PCT_MIN + _cfg.PHASE4_CAPITAL_PCT_MAX) / 2.0
        rs = {
            "rules_set": [
                {"tp": tp_mid, "sl": sl_mid, "capital_pct": cap_mid,
                 "conditions": ["[feat_0] IS Very High"]},
            ]
        }
        assert _params_within_bounds(rs) is True


# ---------------------------------------------------------------------------
# Tests: _load_rule_set
# ---------------------------------------------------------------------------

class TestLoadRuleSet:
    def test_returns_none_when_file_missing(self, tmp_path):
        result = _load_rule_set(str(tmp_path / "nonexistent.json"))
        assert result is None

    def test_returns_dict_when_valid(self, tmp_path):
        path = str(tmp_path / "test.json")
        data = {"direction": "long", "rules_set": []}
        with open(path, "w") as fh:
            json.dump(data, fh)
        result = _load_rule_set(path)
        assert result == data

    def test_returns_none_on_invalid_json(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as fh:
            fh.write("{invalid json")
        result = _load_rule_set(path)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: TradingEnv
# ---------------------------------------------------------------------------

class TestTradingEnv:
    def _make_env(self, n_rules: int = 2, direction: str = "long") -> TradingEnv:
        df = _make_df(n_rows=100)
        rule_set = _make_rule_set(n_rules=n_rules, direction=direction)
        return TradingEnv(df, rule_set, direction)

    def test_construction_long(self):
        env = self._make_env(direction="long")
        assert env.direction == "long"
        assert env.n_rules == 2

    def test_construction_short(self):
        env = self._make_env(direction="short")
        assert env.direction == "short"

    def test_n_actions_is_3_per_rule(self):
        env = self._make_env(n_rules=3)
        assert env.n_actions == 9  # 3 rules * 3 params

    def test_state_dimension(self):
        env = self._make_env(n_rules=2)
        # n_features + n_rules + 2 (equity, exposure)
        expected = env.n_features + env.n_rules + 2
        assert env.n_state == expected

    def test_reset_returns_observation_of_correct_shape(self):
        env = self._make_env(n_rules=2)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        assert obs.shape == (env.n_state,)

    def test_reset_returns_float32(self):
        env = self._make_env()
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        assert obs.dtype == np.float32

    def test_step_returns_correct_structure(self):
        env = self._make_env(n_rules=2)
        env.reset()
        action = env._default_action()
        result = env.step(action)
        # Should return (obs, reward, terminated, ...) with at least 3 elements
        assert len(result) >= 3

    def test_step_observation_shape(self):
        env = self._make_env(n_rules=2)
        env.reset()
        action = env._default_action()
        result = env.step(action)
        obs = result[0]
        assert obs.shape == (env.n_state,)

    def test_step_reward_is_float(self):
        env = self._make_env(n_rules=2)
        env.reset()
        action = env._default_action()
        result = env.step(action)
        reward = result[1]
        assert isinstance(reward, float)

    def test_clip_action_enforces_tp_bounds(self):
        env = self._make_env(n_rules=2)
        action = np.zeros(env.n_actions, dtype=np.float32)
        # Set TP way out of bounds
        action[0] = 999.0
        clipped = env._clip_action(action)
        assert clipped[0] <= _cfg.PHASE4_TP_MAX

    def test_clip_action_enforces_sl_bounds(self):
        env = self._make_env(n_rules=2)
        action = np.zeros(env.n_actions, dtype=np.float32)
        action[1] = -999.0  # SL below min
        clipped = env._clip_action(action)
        assert clipped[1] >= _cfg.PHASE4_SL_MIN

    def test_clip_action_enforces_capital_pct_bounds(self):
        env = self._make_env(n_rules=2)
        action = np.zeros(env.n_actions, dtype=np.float32)
        action[2] = 999.0  # capital_pct above max
        clipped = env._clip_action(action)
        assert clipped[2] <= _cfg.PHASE4_CAPITAL_PCT_MAX

    def test_default_action_within_bounds(self):
        env = self._make_env(n_rules=3)
        action = env._default_action()
        clipped = env._clip_action(action)
        np.testing.assert_array_equal(action, clipped)

    def test_get_current_params_returns_correct_count(self):
        env = self._make_env(n_rules=3)
        env.reset()
        params = env.get_current_params()
        assert len(params) == 3

    def test_get_current_params_has_required_keys(self):
        env = self._make_env(n_rules=2)
        env.reset()
        params = env.get_current_params()
        for p in params:
            assert "conditions" in p
            assert "tp" in p
            assert "sl" in p
            assert "capital_pct" in p

    def test_state_includes_equity_normalized(self):
        """Equity normalized should be 1.0 at start (equity == INITIAL_CAPITAL)."""
        env = self._make_env(n_rules=2)
        obs = env.reset()
        if isinstance(obs, tuple):
            obs = obs[0]
        # Last two elements are equity_norm and exposure_norm
        equity_norm = obs[-2]
        assert abs(float(equity_norm) - 1.0) < 1e-5


# ---------------------------------------------------------------------------
# Tests: RL_Agent constructor
# ---------------------------------------------------------------------------

class TestRLAgentInit:
    def test_invalid_direction_raises(self):
        df = _make_df()
        rs = _make_rule_set()
        with pytest.raises(ValueError, match="direction must be"):
            RL_Agent(df, df, rs, "both")

    def test_empty_rules_raises(self):
        df = _make_df()
        rs = {"direction": "long", "rules_set": []}
        with pytest.raises(ValueError, match="at least one rule"):
            RL_Agent(df, df, rs, "long")

    def test_valid_construction_long(self):
        df = _make_df()
        rs = _make_rule_set(direction="long")
        agent = RL_Agent(df, df, rs, "long", total_timesteps=10, elbow_window=5)
        assert agent.direction == "long"
        assert agent.total_timesteps == 10
        assert agent.elbow_window == 5

    def test_valid_construction_short(self):
        df = _make_df()
        rs = _make_rule_set(direction="short")
        agent = RL_Agent(df, df, rs, "short", total_timesteps=10, elbow_window=5)
        assert agent.direction == "short"

    def test_default_timesteps_from_config(self):
        df = _make_df()
        rs = _make_rule_set()
        agent = RL_Agent(df, df, rs, "long")
        assert agent.total_timesteps == _cfg.PHASE4_TOTAL_TIMESTEPS

    def test_default_elbow_window_from_config(self):
        df = _make_df()
        rs = _make_rule_set()
        agent = RL_Agent(df, df, rs, "long")
        assert agent.elbow_window == _cfg.PHASE4_ELBOW_WINDOW


# ---------------------------------------------------------------------------
# Tests: RL_Agent.find_elbow_point (static method)
# ---------------------------------------------------------------------------

class TestRLAgentFindElbowPoint:
    """Tests for RL_Agent.find_elbow_point static method."""

    def test_delegates_to_module_function(self):
        """Static method should produce same result as module-level function."""
        curve = [0.0, 5.0, 8.0, 9.0, 9.5, 9.7, 9.8]
        assert RL_Agent.find_elbow_point(curve) == find_elbow_point(curve)

    def test_empty_returns_zero(self):
        assert RL_Agent.find_elbow_point([]) == 0

    def test_plateau_returns_zero(self):
        assert RL_Agent.find_elbow_point([5.0, 5.0, 5.0]) == 0

    def test_monotone_returns_last(self):
        curve = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert RL_Agent.find_elbow_point(curve) == len(curve) - 1


# ---------------------------------------------------------------------------
# Tests: RL_Agent.skip_if_valid
# ---------------------------------------------------------------------------

class TestRLAgentSkipIfValid:
    def _write_rule_set(self, path: str, direction: str, tp: float = 5.0,
                        sl: float = 2.5, cap: float = 50.0):
        data = {
            "direction": direction,
            "rules_set": [
                {"tp": tp, "sl": sl, "capital_pct": cap,
                 "conditions": ["[feat_0] IS Very High"]},
                {"tp": tp, "sl": sl, "capital_pct": cap,
                 "conditions": ["[feat_1] IS Low"]},
            ],
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh)

    def test_returns_none_when_file_missing(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            result = RL_Agent.skip_if_valid("long")
            assert result is None
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_returns_data_when_valid_and_in_bounds(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        path = str(tmp_path / "long.json")
        self._write_rule_set(path, "long")
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = path
        try:
            result = RL_Agent.skip_if_valid("long")
            assert result is not None
            assert result["direction"] == "long"
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_returns_none_when_tp_out_of_bounds(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        path = str(tmp_path / "long.json")
        self._write_rule_set(path, "long", tp=_cfg.PHASE4_TP_MAX + 1.0)
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = path
        try:
            result = RL_Agent.skip_if_valid("long")
            assert result is None
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_returns_none_when_sl_out_of_bounds(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        path = str(tmp_path / "long.json")
        self._write_rule_set(path, "long", sl=_cfg.PHASE4_SL_MAX + 1.0)
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = path
        try:
            result = RL_Agent.skip_if_valid("long")
            assert result is None
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_returns_none_when_capital_pct_out_of_bounds(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        path = str(tmp_path / "long.json")
        self._write_rule_set(path, "long", cap=_cfg.PHASE4_CAPITAL_PCT_MAX + 1.0)
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = path
        try:
            result = RL_Agent.skip_if_valid("long")
            assert result is None
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError, match="direction must be"):
            RL_Agent.skip_if_valid("both")

    def test_short_direction_works(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        path = str(tmp_path / "short.json")
        self._write_rule_set(path, "short")
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["short"] = path
        try:
            result = RL_Agent.skip_if_valid("short")
            assert result is not None
            assert result["direction"] == "short"
        finally:
            m._OUTPUT_PATHS.update(original)


# ---------------------------------------------------------------------------
# Tests: RL_Agent.train() — integration (tiny timesteps)
# ---------------------------------------------------------------------------

class TestRLAgentTrain:
    """Integration tests using tiny timestep counts."""

    def _make_agent(
        self,
        direction: str = "long",
        n_rules: int = 2,
        total_timesteps: int = 40,
        elbow_window: int = 10,
    ) -> RL_Agent:
        df = _make_df(n_rows=100)
        rs = _make_rule_set(n_rules=n_rules, direction=direction)
        return RL_Agent(
            df, df, rs, direction,
            total_timesteps=total_timesteps,
            elbow_window=elbow_window,
            seed=42,
        )

    def test_train_returns_dict(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long")
            result = agent.train()
            assert isinstance(result, dict)
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_output_has_direction(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long")
            result = agent.train()
            assert result["direction"] == "long"
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_output_has_rules_set(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long")
            result = agent.train()
            assert "rules_set" in result
            assert len(result["rules_set"]) == 2
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_output_tp_within_bounds(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long")
            result = agent.train()
            for rule in result["rules_set"]:
                tp = rule["tp"]
                assert _cfg.PHASE4_TP_MIN <= tp <= _cfg.PHASE4_TP_MAX, (
                    f"TP {tp} out of bounds [{_cfg.PHASE4_TP_MIN}, {_cfg.PHASE4_TP_MAX}]"
                )
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_output_sl_within_bounds(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long")
            result = agent.train()
            for rule in result["rules_set"]:
                sl = rule["sl"]
                assert _cfg.PHASE4_SL_MIN <= sl <= _cfg.PHASE4_SL_MAX, (
                    f"SL {sl} out of bounds [{_cfg.PHASE4_SL_MIN}, {_cfg.PHASE4_SL_MAX}]"
                )
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_output_capital_pct_within_bounds(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long")
            result = agent.train()
            for rule in result["rules_set"]:
                cap = rule["capital_pct"]
                assert _cfg.PHASE4_CAPITAL_PCT_MIN <= cap <= _cfg.PHASE4_CAPITAL_PCT_MAX, (
                    f"capital_pct {cap} out of bounds"
                )
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_creates_output_file(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        out_path = str(tmp_path / "long.json")
        m._OUTPUT_PATHS["long"] = out_path
        try:
            agent = self._make_agent("long")
            agent.train()
            assert os.path.exists(out_path)
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_output_file_is_valid_json(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        out_path = str(tmp_path / "long.json")
        m._OUTPUT_PATHS["long"] = out_path
        try:
            agent = self._make_agent("long")
            agent.train()
            with open(out_path) as fh:
                data = json.load(fh)
            assert isinstance(data, dict)
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_validation_returns_populated(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long", total_timesteps=40, elbow_window=10)
            agent.train()
            assert len(agent.validation_returns) > 0
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_elbow_idx_is_valid(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long", total_timesteps=40, elbow_window=10)
            agent.train()
            assert 0 <= agent.elbow_idx < len(agent.validation_returns)
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_short_direction(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["short"] = str(tmp_path / "short.json")
        try:
            agent = self._make_agent("short")
            result = agent.train()
            assert result["direction"] == "short"
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_each_rule_has_conditions(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long")
            result = agent.train()
            for rule in result["rules_set"]:
                assert "conditions" in rule
                assert len(rule["conditions"]) > 0
        finally:
            m._OUTPUT_PATHS.update(original)

    def test_train_with_three_rules(self, tmp_path):
        import gpu_fuzzy_trader.phases.phase4_rl_optimizer as m
        original = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS["long"] = str(tmp_path / "long.json")
        try:
            agent = self._make_agent("long", n_rules=3)
            result = agent.train()
            assert len(result["rules_set"]) == 3
        finally:
            m._OUTPUT_PATHS.update(original)


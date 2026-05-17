"""
Property-based tests for gpu_fuzzy_trader.phases.phase3_rule_set.Rule_Set_Selector

Property 21: Rule Set Size Bounds
  **Validates: Requirements 9.1, 12.8**
  For any run of Rule_Set_Selector, the output rule set must have 2–5 rules.
  PHASE3_MIN_RULES = 2, PHASE3_MAX_RULES = 5.

Property 22: Rule Set Uniqueness
  **Validates: Requirements 9.4**
  For any run of Rule_Set_Selector, no two rules in the output must have
  identical condition sets (order-independent).

Property 29: Symbol Coverage Penalty Application
  **Validates: Requirements 9.5, 15.4**
  The coverage penalty must be applied when symbols_with_trades <
  PHASE3_MIN_SYMBOL_COVERAGE.  When PHASE3_MIN_SYMBOL_COVERAGE = 0, a
  zero-trade penalty must still be applied when no trades occur.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase3_rule_set import (
    Rule_Set_Selector,
    _evaluate_rule_set,
    _conditions_key,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE3_MIN_RULES = _cfg.PHASE3_MIN_RULES   # 2
PHASE3_MAX_RULES = _cfg.PHASE3_MAX_RULES   # 5
PHASE3_MIN_SYMBOL_COVERAGE = _cfg.PHASE3_MIN_SYMBOL_COVERAGE  # 7


# ---------------------------------------------------------------------------
# Helpers — minimal DataFrame and pool builders
# ---------------------------------------------------------------------------

def _make_df(
    n_rows: int,
    symbols: list[str],
    n_features: int = 3,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a minimal DataFrame with all required label columns and feature
    columns (integer-valued 0–4, matching 'positive' mode).
    """
    rng = np.random.default_rng(seed)
    rows_per_sym = max(1, n_rows // len(symbols))
    dfs = []
    for sym in symbols:
        n = rows_per_sym
        open_next = rng.uniform(100.0, 200.0, size=n)
        max_288 = open_next * rng.uniform(1.00, 1.10, size=n)
        min_288 = open_next * rng.uniform(0.90, 1.00, size=n)
        close_288 = open_next * rng.uniform(0.95, 1.05, size=n)
        max_before_min = rng.integers(0, 2, size=n).astype(float)

        data: dict[str, Any] = {
            "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
            "symbol": sym,
            "label_open_next": open_next,
            "label_close_288": close_288,
            "label_min_288": min_288,
            "label_max_288": max_288,
            "label_max_before_min": max_before_min,
            "_symbol_bar_index": np.arange(n),
        }
        for i in range(n_features):
            data[f"feat_{i}"] = rng.integers(0, 5, size=n).astype(float)

        dfs.append(pd.DataFrame(data))

    return pd.concat(dfs, ignore_index=True)


def _make_pool(n_rules: int, n_conditions_each: int = 2) -> list[dict]:
    """
    Build a minimal pool of rules with distinct condition sets.

    Each rule has `n_conditions_each` unique conditions so that the pool
    contains no duplicates.
    """
    pool = []
    for i in range(n_rules):
        conditions = [f"[feat_{i}] IS Very High", f"[feat_{i}] IS Low"]
        if n_conditions_each > 2:
            conditions.append(f"[feat_{i}] IS Medium")
        pool.append({
            "conditions": conditions[:n_conditions_each],
            "tp": float(_cfg.PHASE2_TP),
            "sl": float(_cfg.PHASE2_SL),
            "capital_pct": float(_cfg.PHASE2_CAPITAL_PCT),
        })
    return pool


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def selector_args(draw: st.DrawFn):
    """
    Generate (train_df, val_df, pool, direction) suitable for constructing a
    Rule_Set_Selector with tiny pop/gen settings.

    Varies:
      - Number of symbols (1–4)
      - Rows per symbol (30–80)
      - Pool size (PHASE3_MIN_RULES to PHASE3_MAX_RULES + 3)
      - Direction
    """
    n_symbols = draw(st.integers(min_value=1, max_value=4))
    rows_per_sym = draw(st.integers(min_value=30, max_value=80))
    pool_size = draw(st.integers(min_value=PHASE3_MIN_RULES, max_value=PHASE3_MAX_RULES + 3))
    direction = draw(st.sampled_from(["long", "short"]))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    symbols = [f"SYM_{i}" for i in range(n_symbols)]
    n_rows = rows_per_sym * n_symbols

    train_df = _make_df(n_rows, symbols, seed=seed)
    val_df = _make_df(n_rows, symbols, seed=seed + 1)
    pool = _make_pool(pool_size)

    return train_df, val_df, pool, direction


# ---------------------------------------------------------------------------
# Property 21: Rule Set Size Bounds
# Validates: Requirements 9.1, 12.8
# ---------------------------------------------------------------------------

@given(args=selector_args())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_property_21_rule_set_size_bounds(
    args: tuple[pd.DataFrame, pd.DataFrame, list[dict], str],
) -> None:
    """
    **Property 21: Rule Set Size Bounds**
    **Validates: Requirements 9.1, 12.8**

    For any run of Rule_Set_Selector (with varying datasets, pool sizes, and
    directions), the output rule set must contain between PHASE3_MIN_RULES (2)
    and PHASE3_MAX_RULES (5) rules, inclusive.

    This validates both the NSGA-II search constraint and the output schema
    constraint enforced by the Output_Writer (Requirement 12.8).
    """
    train_df, val_df, pool, direction = args

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Redirect output files to a temp dir so we don't pollute real outputs/
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original_paths = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS[direction] = os.path.join(tmp_dir, f"{direction}.json")

        try:
            selector = Rule_Set_Selector(
                train_df=train_df,
                val_df=val_df,
                pool=pool,
                direction=direction,
                pop_size=6,
                n_generations=3,
                seed=0,
            )
            result = selector.run()

            rules_set = result["rules_set"]
            n_rules = len(rules_set)

            assert PHASE3_MIN_RULES <= n_rules <= PHASE3_MAX_RULES, (
                f"Rule set has {n_rules} rules; expected [{PHASE3_MIN_RULES}, "
                f"{PHASE3_MAX_RULES}]. direction={direction}, pool_size={len(pool)}"
            )
        finally:
            m._OUTPUT_PATHS.update(original_paths)


# ---------------------------------------------------------------------------
# Property 22: Rule Set Uniqueness
# Validates: Requirements 9.4
# ---------------------------------------------------------------------------

@given(args=selector_args())
@settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_property_22_rule_set_uniqueness(
    args: tuple[pd.DataFrame, pd.DataFrame, list[dict], str],
) -> None:
    """
    **Property 22: Rule Set Uniqueness**
    **Validates: Requirements 9.4**

    For any run of Rule_Set_Selector, no two rules in the output rule set
    may have identical condition sets (order-independent comparison).

    Two rules are considered identical if their condition sets are equal as
    frozensets (i.e., the same conditions regardless of order).
    """
    train_df, val_df, pool, direction = args

    with tempfile.TemporaryDirectory() as tmp_dir:
        import gpu_fuzzy_trader.phases.phase3_rule_set as m
        original_paths = m._OUTPUT_PATHS.copy()
        m._OUTPUT_PATHS[direction] = os.path.join(tmp_dir, f"{direction}.json")

        try:
            selector = Rule_Set_Selector(
                train_df=train_df,
                val_df=val_df,
                pool=pool,
                direction=direction,
                pop_size=6,
                n_generations=3,
                seed=0,
            )
            result = selector.run()

            rules_set = result["rules_set"]

            # Check order-independent uniqueness of condition sets
            seen: set[frozenset] = set()
            for i, rule in enumerate(rules_set):
                key = _conditions_key(rule["conditions"])
                assert key not in seen, (
                    f"Rule {i} has duplicate condition set (order-independent) "
                    f"with a previous rule. conditions={rule['conditions']}"
                )
                seen.add(key)
        finally:
            m._OUTPUT_PATHS.update(original_paths)


# ---------------------------------------------------------------------------
# Property 29: Symbol Coverage Penalty Application
# Validates: Requirements 9.5, 15.4
# ---------------------------------------------------------------------------

def _make_mock_engine(
    total_return_pct: float = 5.0,
    max_drawdown_pct: float = 2.0,
    win_rate: float = 55.0,
    executed_trades: int = 30,
    per_symbol_metrics: dict | None = None,
) -> MagicMock:
    """
    Build a mock CPUBacktestEngine whose simulate_rule_set returns the
    given metrics dict.
    """
    engine = MagicMock()
    engine.simulate_rule_set.return_value = {
        "total_return_pct": total_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "win_rate": win_rate,
        "executed_trades": executed_trades,
        "per_symbol_metrics": per_symbol_metrics or {},
    }
    return engine


@st.composite
def coverage_penalty_scenario(draw: st.DrawFn):
    """
    Generate a scenario for testing coverage penalty application.

    Varies:
      - symbols_with_trades: 0 to PHASE3_MIN_SYMBOL_COVERAGE + 2
      - total_symbols: PHASE3_MIN_SYMBOL_COVERAGE to PHASE3_MIN_SYMBOL_COVERAGE + 3
      - executed_trades: 0 or positive
    """
    total_symbols = draw(
        st.integers(
            min_value=PHASE3_MIN_SYMBOL_COVERAGE,
            max_value=PHASE3_MIN_SYMBOL_COVERAGE + 3,
        )
    )
    symbols_with_trades = draw(st.integers(min_value=0, max_value=total_symbols))
    # When symbols_with_trades > 0, executed_trades must be > 0
    if symbols_with_trades > 0:
        executed_trades = draw(st.integers(min_value=1, max_value=100))
    else:
        executed_trades = 0

    return total_symbols, symbols_with_trades, executed_trades


@given(scenario=coverage_penalty_scenario())
@settings(
    max_examples=20,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_property_29_symbol_coverage_penalty_application(
    scenario: tuple[int, int, int],
) -> None:
    """
    **Property 29: Symbol Coverage Penalty Application**
    **Validates: Requirements 9.5, 15.4**

    The coverage penalty must be applied when symbols_with_trades <
    PHASE3_MIN_SYMBOL_COVERAGE.  When PHASE3_MIN_SYMBOL_COVERAGE = 0, a
    zero-trade penalty must still be applied when no trades occur.

    We test _evaluate_rule_set directly with mock engines to verify:

    1. When symbols_with_trades < PHASE3_MIN_SYMBOL_COVERAGE, the objectives
       include a positive coverage penalty proportional to the shortfall.

    2. When executed_trades == 0, the objectives include a zero-trade penalty
       regardless of the PHASE3_MIN_SYMBOL_COVERAGE setting.

    3. When symbols_with_trades >= PHASE3_MIN_SYMBOL_COVERAGE AND
       executed_trades > 0, no coverage or zero-trade penalty is applied.

    The penalty is verified by comparing objectives against a baseline
    (no-penalty) scenario with identical return/drawdown/win_rate values.
    """
    total_symbols, symbols_with_trades, executed_trades = scenario

    # Build per_symbol_metrics: first `symbols_with_trades` symbols have trades
    per_symbol_metrics: dict[str, dict] = {}
    for i in range(total_symbols):
        sym = f"SYM_{i}"
        if i < symbols_with_trades:
            per_symbol_metrics[sym] = {"trade_count": 5, "win_rate": 50.0, "net_pnl": 10.0}
        else:
            per_symbol_metrics[sym] = {"trade_count": 0, "win_rate": 0.0, "net_pnl": 0.0}

    # A simple rule set (content doesn't matter — mock engine ignores it)
    rule_set = [
        {"conditions": ["[feat_0] IS Very High"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0},
        {"conditions": ["[feat_1] IS Low"], "tp": 4.0, "sl": 2.0, "capital_pct": 50.0},
    ]

    # Val engine returns the scenario metrics
    val_engine = _make_mock_engine(
        total_return_pct=5.0,
        max_drawdown_pct=2.0,
        win_rate=55.0,
        executed_trades=executed_trades,
        per_symbol_metrics=per_symbol_metrics,
    )

    # Train engine returns a matching result (no overfitting penalty)
    train_engine = _make_mock_engine(
        total_return_pct=5.0,
        max_drawdown_pct=2.0,
        win_rate=55.0,
        executed_trades=executed_trades,
        per_symbol_metrics=per_symbol_metrics,
    )

    objectives, val_metrics = _evaluate_rule_set(rule_set, val_engine, train_engine)

    # Compute expected penalties
    expected_coverage_penalty = 0.0
    if symbols_with_trades < PHASE3_MIN_SYMBOL_COVERAGE:
        expected_coverage_penalty = (
            (PHASE3_MIN_SYMBOL_COVERAGE - symbols_with_trades) * 5.0
        )

    expected_zero_penalty = 100.0 if executed_trades == 0 else 0.0

    # Overfitting penalty: |train_return - val_return| / max(|train_return|, 1.0)
    # Both engines return 5.0, so overfitting_penalty = 0.0
    expected_overfitting_penalty = 0.0

    total_expected_penalty = (
        expected_coverage_penalty
        + expected_zero_penalty
        + expected_overfitting_penalty
    )

    # Base objectives without any penalty
    val_return = 5.0
    val_dd = 2.0
    val_wr = 55.0

    expected_f1 = -val_return + total_expected_penalty
    expected_f2 = val_dd + total_expected_penalty
    expected_f3 = -val_wr + total_expected_penalty

    assert abs(objectives[0] - expected_f1) < 1e-9, (
        f"f1 mismatch: got {objectives[0]:.6f}, expected {expected_f1:.6f}. "
        f"symbols_with_trades={symbols_with_trades}, "
        f"PHASE3_MIN_SYMBOL_COVERAGE={PHASE3_MIN_SYMBOL_COVERAGE}, "
        f"executed_trades={executed_trades}, "
        f"coverage_penalty={expected_coverage_penalty}, "
        f"zero_penalty={expected_zero_penalty}"
    )
    assert abs(objectives[1] - expected_f2) < 1e-9, (
        f"f2 mismatch: got {objectives[1]:.6f}, expected {expected_f2:.6f}."
    )
    assert abs(objectives[2] - expected_f3) < 1e-9, (
        f"f3 mismatch: got {objectives[2]:.6f}, expected {expected_f3:.6f}."
    )

    # --- Specific assertions for penalty presence/absence ---

    if symbols_with_trades < PHASE3_MIN_SYMBOL_COVERAGE:
        # Coverage penalty must be positive
        assert expected_coverage_penalty > 0.0
        # Objectives must be strictly worse than the no-penalty baseline
        assert objectives[0] > -val_return, (
            f"Coverage penalty not reflected in f1: "
            f"symbols_with_trades={symbols_with_trades} < "
            f"PHASE3_MIN_SYMBOL_COVERAGE={PHASE3_MIN_SYMBOL_COVERAGE}"
        )

    if executed_trades == 0:
        # Zero-trade penalty must be applied (100.0)
        assert objectives[0] > -val_return, (
            "Zero-trade penalty not reflected in f1 when executed_trades=0"
        )

    if symbols_with_trades >= PHASE3_MIN_SYMBOL_COVERAGE and executed_trades > 0:
        # No coverage or zero-trade penalty — objectives equal baseline
        assert abs(objectives[0] - (-val_return)) < 1e-9, (
            f"Unexpected penalty when no penalty should apply: "
            f"f1={objectives[0]:.6f}, expected {-val_return:.6f}"
        )

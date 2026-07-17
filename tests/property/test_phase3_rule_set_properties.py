"""
Property-based tests for gpu_fuzzy_trader.phases.phase3_rule_set.Rule_Set_Selector

Property 21: Rule Set Size Bounds
  **Validates: Requirements 9.1, 12.8**
  For any run of Rule_Set_Selector, the output rule set must have the correct
  number of rules (global min–max from config).

Property 22: Rule Set Uniqueness
  **Validates: Requirements 9.4**
  For any run of Rule_Set_Selector, no two rules in the output must have
  identical condition sets (order-independent).
"""

from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from hypothesis import given, HealthCheck
from hypothesis import strategies as st

from tests.property.hypothesis_config import prop_settings

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase3_objectives import (
    conditions_key as _conditions_key,
)
from gpu_fuzzy_trader.phases.phase3_rule_set import (
    Rule_Set_Selector,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE3_GLOBAL_MIN_RULES = _cfg.PHASE3_GLOBAL_MIN_RULES
PHASE3_GLOBAL_MAX_RULES = _cfg.PHASE3_GLOBAL_MAX_RULES


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
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
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


def _make_pool(
    n_rules: int,
    n_features: int = 3,
    n_conditions_each: int = 1,
) -> list[dict]:
    """
    Build a minimal pool of rules with distinct, satisfiable condition sets.

    Each rule uses conditions on different features so rows can match.
    """
    pool = []
    for i in range(n_rules):
        conditions = [f"[feat_{i % n_features}] IS Very High"]
        if n_conditions_each > 1:
            conditions.append(f"[feat_{(i + 1) % n_features}] IS Low")
        if n_conditions_each > 2:
            conditions.append(f"[feat_{(i + 2) % n_features}] IS Medium")
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
      - Pool size (PHASE3_GLOBAL_MIN_RULES to PHASE3_GLOBAL_MAX_RULES + 3)
      - Direction
    """
    n_symbols = draw(st.integers(min_value=1, max_value=4))
    rows_per_sym = draw(st.integers(min_value=30, max_value=80))
    pool_size = draw(st.integers(min_value=PHASE3_GLOBAL_MIN_RULES,
                     max_value=PHASE3_GLOBAL_MAX_RULES + 3))
    direction = draw(st.sampled_from(["long", "short"]))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    symbols = [f"SYM_{i}" for i in range(n_symbols)]
    n_rows = rows_per_sym * n_symbols

    n_feat = max(3, pool_size)
    train_df = _make_df(n_rows, symbols, n_features=n_feat, seed=seed)
    val_df = _make_df(n_rows, symbols, n_features=n_feat, seed=seed + 1)
    pool = _make_pool(pool_size, n_features=n_feat)

    return train_df, val_df, pool, direction


# ---------------------------------------------------------------------------
# Property 21: Rule Set Size Bounds
# Validates: Requirements 9.1, 12.8
# ---------------------------------------------------------------------------

@given(args=selector_args())
@prop_settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.function_scoped_fixture],
)
def test_property_21_rule_set_size_bounds(
    args: tuple[pd.DataFrame, pd.DataFrame, list[dict], str],
) -> None:
    """
    **Property 21: Rule Set Size Bounds**
    **Validates: Requirements 9.1, 12.8**

    For any run of Rule_Set_Selector (with varying datasets, pool sizes, and
    directions), the output rule set must contain between PHASE3_GLOBAL_MIN_RULES
    and PHASE3_GLOBAL_MAX_RULES rules, inclusive.

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
                seed=0,
            )
            result = selector.run()

            rules_set = result["rules_set"]
            n_rules = len(rules_set)

            if result.get("selection_accepted") is False:
                assert n_rules == 0, (
                    f"Rejected selection should yield empty rules_set, got {n_rules}"
                )
            else:
                assert PHASE3_GLOBAL_MIN_RULES <= n_rules <= PHASE3_GLOBAL_MAX_RULES, (
                    f"Rule set has {n_rules} rules; expected [{PHASE3_GLOBAL_MIN_RULES}, "
                    f"{PHASE3_GLOBAL_MAX_RULES}]. direction={direction}, pool_size={len(pool)}"
                )
        finally:
            m._OUTPUT_PATHS.update(original_paths)


# ---------------------------------------------------------------------------
# Property 22: Rule Set Uniqueness
# Validates: Requirements 9.4
# ---------------------------------------------------------------------------

@given(args=selector_args())
@prop_settings(
    max_examples=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.function_scoped_fixture],
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

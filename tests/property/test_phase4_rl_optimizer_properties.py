"""
Property-based tests for gpu_fuzzy_trader.phases.phase4_rl_optimizer

Property 24: RL Action Bounds
  **Validates: Requirements 10.3**
  For any action produced by TradingEnv._clip_action(), all values must be
  within the configured bounds:
    tp_i          ∈ [PHASE4_TP_MIN,          PHASE4_TP_MAX]          = [1.0, 10.0]
    sl_i          ∈ [PHASE4_SL_MIN,          PHASE4_SL_MAX]          = [0.5,  5.0]
    capital_pct_i ∈ [PHASE4_CAPITAL_PCT_MIN, PHASE4_CAPITAL_PCT_MAX] = [10.0, 100.0]

Property 25: RL State Vector Completeness
  **Validates: Requirements 10.2**
  For any dataset and rule set, the state vector from TradingEnv must have
  the correct dimension:
    n_features + n_rules + 2  (equity_normalized, open_exposure_normalized)

Property 26: Elbow Method Correctness
  **Validates: Requirements 10.5**
  For any validation returns curve:
    - The returned index must be a valid index (0 <= idx < len(curve))
    - For a monotonically increasing curve, the index must be the last element
    - For an immediately plateauing curve (all equal), the index must be 0
    - For any curve, the function must not raise an exception
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase4_rl_optimizer import TradingEnv, find_elbow_point


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PHASE4_TP_MIN = _cfg.PHASE4_TP_MIN                    # 1.0
PHASE4_TP_MAX = _cfg.PHASE4_TP_MAX                    # 10.0
PHASE4_SL_MIN = _cfg.PHASE4_SL_MIN                    # 0.5
PHASE4_SL_MAX = _cfg.PHASE4_SL_MAX                    # 5.0
PHASE4_CAPITAL_PCT_MIN = _cfg.PHASE4_CAPITAL_PCT_MIN  # 10.0
PHASE4_CAPITAL_PCT_MAX = _cfg.PHASE4_CAPITAL_PCT_MAX  # 100.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(
    n_rows: int,
    feature_cols: list[str],
    symbols: list[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Build a minimal DataFrame with all required label columns and the given
    feature columns (float-valued).
    """
    rng = np.random.default_rng(seed)
    if symbols is None:
        symbols = ["SYM_0"]

    rows_per_sym = max(1, n_rows // len(symbols))
    dfs = []
    for sym in symbols:
        n = rows_per_sym
        open_next = rng.uniform(100.0, 200.0, size=n)
        max_288 = open_next * rng.uniform(1.00, 1.10, size=n)
        min_288 = open_next * rng.uniform(0.90, 1.00, size=n)
        close_288 = open_next * rng.uniform(0.95, 1.05, size=n)
        max_before_min = rng.integers(0, 2, size=n).astype(float)

        data: dict = {
            "datetime": pd.date_range("2020-01-01", periods=n, freq="5min"),
            "symbol": sym,
            "label_open_next": open_next,
            "label_close_288": close_288,
            "label_min_288": min_288,
            "label_max_288": max_288,
            "label_max_before_min": max_before_min,
            "_symbol_bar_index": np.arange(n),
        }
        for col in feature_cols:
            data[col] = rng.uniform(-1.0, 1.0, size=n).astype(np.float32)

        dfs.append(pd.DataFrame(data))

    return pd.concat(dfs, ignore_index=True)


def _make_rule_set(n_rules: int, feature_cols: list[str]) -> dict:
    """
    Build a minimal rule set dict with `n_rules` rules, each referencing
    the first feature column.
    """
    feat = feature_cols[0] if feature_cols else "feat_0"
    rules = []
    for i in range(n_rules):
        rules.append({
            "conditions": [f"[{feat}] IS Very High"],
            "tp": float(_cfg.PHASE2_TP),
            "sl": float(_cfg.PHASE2_SL),
            "capital_pct": float(_cfg.PHASE2_CAPITAL_PCT),
        })
    return {"direction": "long", "rules_set": rules}


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A wide float range that includes values well outside the action bounds
_wide_float = st.floats(
    min_value=-1000.0,
    max_value=1000.0,
    allow_nan=False,
    allow_infinity=False,
)


@st.composite
def trading_env_and_raw_action(draw: st.DrawFn):
    """
    Generate a (TradingEnv, raw_action_array) pair.

    The raw action array contains arbitrary float values (including
    out-of-bounds values) so that _clip_action() is exercised.

    Varies:
      - Number of feature columns (1–8)
      - Number of rules (1–5)
      - Raw action values (arbitrary floats, possibly out of bounds)
    """
    n_features = draw(st.integers(min_value=1, max_value=8))
    n_rules = draw(st.integers(min_value=1, max_value=5))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    feature_cols = [f"feat_{i}" for i in range(n_features)]
    df = _make_df(n_rows=20, feature_cols=feature_cols, seed=seed)
    rule_set = _make_rule_set(n_rules, feature_cols)

    env = TradingEnv(df, rule_set, direction="long", feature_cols=feature_cols)

    # Generate a raw action array with n_rules * 3 values (tp, sl, cap per rule)
    raw_values = draw(
        st.lists(
            _wide_float,
            min_size=n_rules * 3,
            max_size=n_rules * 3,
        )
    )
    raw_action = np.array(raw_values, dtype=np.float32)

    return env, raw_action


@st.composite
def trading_env_args(draw: st.DrawFn):
    """
    Generate (n_features, n_rules, df, rule_set) for state-vector tests.

    Varies:
      - Number of feature columns (0–10)
      - Number of rules (1–5)
      - Number of rows in the dataset (10–50)
    """
    n_features = draw(st.integers(min_value=0, max_value=10))
    n_rules = draw(st.integers(min_value=1, max_value=5))
    n_rows = draw(st.integers(min_value=10, max_value=50))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))

    feature_cols = [f"feat_{i}" for i in range(n_features)]
    df = _make_df(n_rows=n_rows, feature_cols=feature_cols, seed=seed)
    rule_set = _make_rule_set(n_rules, feature_cols if feature_cols else ["feat_0"])

    return n_features, n_rules, df, rule_set, feature_cols


# ---------------------------------------------------------------------------
# Property 24: RL Action Bounds
# Validates: Requirements 10.3
# ---------------------------------------------------------------------------

@given(args=trading_env_and_raw_action())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_24_rl_action_bounds(
    args: tuple[TradingEnv, np.ndarray],
) -> None:
    """
    **Property 24: RL Action Bounds**
    **Validates: Requirements 10.3**

    For any action array (including values well outside the valid ranges)
    passed to TradingEnv._clip_action(), the returned array must satisfy:

      tp_i          ∈ [PHASE4_TP_MIN,          PHASE4_TP_MAX]
      sl_i          ∈ [PHASE4_SL_MIN,          PHASE4_SL_MAX]
      capital_pct_i ∈ [PHASE4_CAPITAL_PCT_MIN, PHASE4_CAPITAL_PCT_MAX]

    for every rule index i.
    """
    env, raw_action = args
    clipped = env._clip_action(raw_action)

    assert len(clipped) == env.n_rules * 3, (
        f"Clipped action length {len(clipped)} != n_rules*3 = {env.n_rules * 3}"
    )

    for i in range(env.n_rules):
        base = i * 3
        tp = float(clipped[base])
        sl = float(clipped[base + 1])
        cap = float(clipped[base + 2])

        assert PHASE4_TP_MIN <= tp <= PHASE4_TP_MAX, (
            f"Rule {i}: tp={tp} is outside [{PHASE4_TP_MIN}, {PHASE4_TP_MAX}]. "
            f"raw_action={raw_action.tolist()}"
        )
        assert PHASE4_SL_MIN <= sl <= PHASE4_SL_MAX, (
            f"Rule {i}: sl={sl} is outside [{PHASE4_SL_MIN}, {PHASE4_SL_MAX}]. "
            f"raw_action={raw_action.tolist()}"
        )
        assert PHASE4_CAPITAL_PCT_MIN <= cap <= PHASE4_CAPITAL_PCT_MAX, (
            f"Rule {i}: capital_pct={cap} is outside "
            f"[{PHASE4_CAPITAL_PCT_MIN}, {PHASE4_CAPITAL_PCT_MAX}]. "
            f"raw_action={raw_action.tolist()}"
        )


# ---------------------------------------------------------------------------
# Property 25: RL State Vector Completeness
# Validates: Requirements 10.2
# ---------------------------------------------------------------------------

@given(args=trading_env_args())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_25_rl_state_vector_completeness(
    args: tuple[int, int, pd.DataFrame, dict, list[str]],
) -> None:
    """
    **Property 25: RL State Vector Completeness**
    **Validates: Requirements 10.2**

    For any dataset and rule set, the state vector produced by TradingEnv
    must have exactly:

      n_state = n_features + n_rules + 2

    where the +2 accounts for equity_normalized and open_exposure_normalized.

    This is verified via:
      1. env.n_state attribute
      2. The shape of the observation returned by env.reset()
    """
    n_features, n_rules, df, rule_set, feature_cols = args

    env = TradingEnv(df, rule_set, direction="long", feature_cols=feature_cols)

    expected_n_state = n_features + n_rules + 2

    # 1. Check the n_state attribute
    assert env.n_state == expected_n_state, (
        f"env.n_state={env.n_state} != n_features({n_features}) + "
        f"n_rules({n_rules}) + 2 = {expected_n_state}"
    )

    # 2. Check the shape of the observation returned by reset()
    reset_result = env.reset()
    # reset() returns (obs, info) when gym is available, or just obs otherwise
    if isinstance(reset_result, tuple):
        obs = reset_result[0]
    else:
        obs = reset_result

    assert obs.shape == (expected_n_state,), (
        f"reset() observation shape {obs.shape} != ({expected_n_state},). "
        f"n_features={n_features}, n_rules={n_rules}"
    )

    # 3. Verify n_features and n_rules are stored correctly
    assert env.n_features == n_features, (
        f"env.n_features={env.n_features} != {n_features}"
    )
    assert env.n_rules == n_rules, (
        f"env.n_rules={env.n_rules} != {n_rules}"
    )


# ---------------------------------------------------------------------------
# Property 26: Elbow Method Correctness
# Validates: Requirements 10.5
# ---------------------------------------------------------------------------

@st.composite
def validation_returns_curve(draw: st.DrawFn):
    """
    Generate a list of validation return values (arbitrary floats).

    Varies:
      - Length: 1 to 50 elements
      - Values: arbitrary finite floats
    """
    length = draw(st.integers(min_value=1, max_value=50))
    values = draw(
        st.lists(
            st.floats(
                min_value=-1000.0,
                max_value=1000.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=length,
            max_size=length,
        )
    )
    return values


@given(curve=validation_returns_curve())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_26_elbow_method_valid_index(
    curve: list[float],
) -> None:
    """
    **Property 26: Elbow Method Correctness — valid index**
    **Validates: Requirements 10.5**

    For any validation returns curve, find_elbow_point() must:
      1. Not raise any exception.
      2. Return a valid index: 0 <= idx < len(curve).
    """
    # Must not raise
    idx = find_elbow_point(curve)

    assert 0 <= idx < len(curve), (
        f"find_elbow_point returned idx={idx} which is out of bounds "
        f"for a curve of length {len(curve)}. curve={curve}"
    )


@given(
    length=st.integers(min_value=3, max_value=50),
    start=st.floats(min_value=-100.0, max_value=99.0, allow_nan=False, allow_infinity=False),
    step=st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
)
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_26_elbow_method_monotonically_increasing(
    length: int,
    start: float,
    step: float,
) -> None:
    """
    **Property 26: Elbow Method Correctness — monotonically increasing curve**
    **Validates: Requirements 10.5**

    For a strictly monotonically increasing curve with length >= 3 (each
    element strictly greater than the previous), find_elbow_point() must
    return the last index (len(curve) - 1).

    A monotonically increasing curve has no elbow — the "best" checkpoint is
    always the last one.

    Note: The design document specifies a special case for 2-element curves
    where the first point (index 0) is always returned. This test therefore
    requires length >= 3 to avoid that documented edge case.
    """
    curve = [start + i * step for i in range(length)]

    idx = find_elbow_point(curve)

    assert idx == length - 1, (
        f"For a monotonically increasing curve of length {length}, "
        f"expected idx={length - 1}, got idx={idx}. "
        f"curve={curve[:5]}{'...' if length > 5 else ''}"
    )


@given(
    length=st.integers(min_value=1, max_value=50),
    value=st.floats(
        min_value=-1000.0,
        max_value=1000.0,
        allow_nan=False,
        allow_infinity=False,
    ),
)
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_property_26_elbow_method_plateau(
    length: int,
    value: float,
) -> None:
    """
    **Property 26: Elbow Method Correctness — immediately plateauing curve**
    **Validates: Requirements 10.5**

    For a curve where all values are equal (immediately plateauing),
    find_elbow_point() must return 0 (the first index).

    A flat curve has no curvature; the first point is the canonical answer.
    """
    curve = [value] * length

    idx = find_elbow_point(curve)

    assert idx == 0, (
        f"For a plateau curve of length {length} with value={value}, "
        f"expected idx=0, got idx={idx}."
    )

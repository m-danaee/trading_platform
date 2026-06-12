"""
Property-based tests for gpu_fuzzy_trader.backtest.cpu_engine.CPUBacktestEngine

Property 10: Priority-Based Rule Assignment Exclusivity
  **Validates: Requirements 5.1**
  For any dataset and any rule set, each row can be assigned to AT MOST ONE
  rule (the first matching rule). No row index may appear more than once in
  the trade log.

Property 11: Trade Outcome Correctness
  **Validates: Requirements 5.2**
  For any trade with known max_ret, min_ret, close_ret, max_before_min, tp, sl:
    Long:
      - if max_ret >= tp and min_ret > -sl  → TP
      - if min_ret <= -sl and max_ret < tp  → SL
      - if both hit → max_before_min determines order
      - if neither  → time exit (close_ret)
    Short:
      - if min_ret <= -tp and max_ret < sl  → TP
      - if max_ret >= sl and min_ret > -tp  → SL
      - if both hit → max_before_min determines order
      - if neither  → time exit (-close_ret)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck, assume
from hypothesis import strategies as st

from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_minimal_df(
    n: int,
    feature_vals: list[float],
    max_ret_pct: float = 3.0,
    min_ret_pct: float = -1.5,
    close_ret_pct: float = 1.0,
    max_before_min: int = 1,
    entry: float = 100.0,
) -> pd.DataFrame:
    """
    Build a minimal single-symbol DataFrame with n rows.

    feature_vals must have length n.
    Prices are derived from entry + percentage returns.
    """
    label_max = entry * (1.0 + max_ret_pct / 100.0)
    label_min = entry * (1.0 + min_ret_pct / 100.0)
    label_close = entry * (1.0 + close_ret_pct / 100.0)
    return pd.DataFrame(
        {
            "symbol": ["SYM"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [entry] * n,
            "label_max_288": [label_max] * n,
            "label_min_288": [label_min] * n,
            "label_close_288": [label_close] * n,
            "label_max_before_min": [max_before_min] * n,
            "feat_a": feature_vals,
        }
    )


def _make_engine(df: pd.DataFrame, direction: str = "long") -> CPUBacktestEngine:
    return CPUBacktestEngine(
        df,
        feature_modes={},
        direction=direction,
        max_hold_candles=1,
        initial_capital=10_000.0,
        leverage=1.0,
        fee_pct=0.0,
        min_position_notional=0.01,
        max_total_exposure_pct=100.0,
    )


# ---------------------------------------------------------------------------
# Strategies for Property 10
# ---------------------------------------------------------------------------

# Feature values that map to known fuzzy buckets in _apply_dynamic_rule:
#   Very Low  : <= 0.2
#   Low       : (0.2, 0.4]
#   Medium    : (0.4, 0.6]
#   High      : (0.6, 0.8]
#   Very High : > 0.8
_BUCKET_REPRESENTATIVES = {
    "Very Low":  0.1,
    "Low":       0.3,
    "Medium":    0.5,
    "High":      0.7,
    "Very High": 0.9,
}
_BUCKET_NAMES = list(_BUCKET_REPRESENTATIVES.keys())
_BUCKET_VALUES = list(_BUCKET_REPRESENTATIVES.values())


@st.composite
def overlapping_rule_set_strategy(draw: st.DrawFn) -> tuple[list[dict], list[float]]:
    """
    Generate a rule set of 1–4 rules that may overlap in their conditions,
    plus a matching list of per-row feature values.

    Each rule uses a single condition on 'feat_a' with one of the five
    positive-mode fuzzy buckets. Rules may share the same bucket (overlap).

    Returns (rule_set, feature_vals) where feature_vals has length n_rows.
    """
    n_rules = draw(st.integers(min_value=1, max_value=4))
    n_rows = draw(st.integers(min_value=5, max_value=30))

    # Each rule picks a bucket (may repeat → overlap)
    rule_buckets = [draw(st.sampled_from(_BUCKET_NAMES)) for _ in range(n_rules)]

    rule_set = [
        {
            "conditions": [f"[feat_a] IS {bucket}"],
            "tp": 2.0,
            "sl": 1.0,
            "capital_pct": 10.0,
        }
        for bucket in rule_buckets
    ]

    # Each row gets a feature value from one of the five buckets
    feature_vals = [
        draw(st.sampled_from(_BUCKET_VALUES)) for _ in range(n_rows)
    ]

    return rule_set, feature_vals


# ---------------------------------------------------------------------------
# Property 10: Priority-Based Rule Assignment Exclusivity
# Validates: Requirements 5.1
# ---------------------------------------------------------------------------

@given(data=overlapping_rule_set_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_10_no_row_assigned_to_multiple_rules(
    data: tuple[list[dict], list[float]],
) -> None:
    """
    **Property 10: Priority-Based Rule Assignment Exclusivity**
    **Validates: Requirements 5.1**

    For any dataset and any rule set (including overlapping rules), each row
    can be assigned to AT MOST ONE rule — the first matching rule in priority
    order. No row index may appear more than once in the trade log.
    """
    rule_set, feature_vals = data
    n = len(feature_vals)

    df = _make_minimal_df(n=n, feature_vals=feature_vals)
    engine = _make_engine(df, direction="long")

    metrics, trade_log = engine.simulate_rule_set(rule_set, return_logs=True)

    if trade_log.empty:
        # No trades executed — trivially satisfies the property
        return

    entry_indices = trade_log["Entry_Index"].tolist()

    # Each row index must appear at most once
    assert len(entry_indices) == len(set(entry_indices)), (
        f"Duplicate Entry_Index values found in trade log: {entry_indices}. "
        f"Each row must be assigned to at most one rule."
    )


@given(data=overlapping_rule_set_strategy())
@settings(
    max_examples=200,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_10b_first_matching_rule_wins(
    data: tuple[list[dict], list[float]],
) -> None:
    """
    **Property 10: Priority-Based Rule Assignment Exclusivity**
    **Validates: Requirements 5.1**

    When multiple rules match the same row, the trade log must record the
    FIRST matching rule's TP/SL values (not any later rule's values).
    """
    rule_set, feature_vals = data
    n = len(feature_vals)

    # Only meaningful when there are at least 2 rules
    assume(len(rule_set) >= 2)

    df = _make_minimal_df(n=n, feature_vals=feature_vals)
    engine = _make_engine(df, direction="long")

    metrics, trade_log = engine.simulate_rule_set(rule_set, return_logs=True)

    if trade_log.empty:
        return

    # For each logged trade, verify the Rule_Index is the FIRST rule that
    # matches that row's feature value.
    for _, row in trade_log.iterrows():
        idx = int(row["Entry_Index"])
        feat_val = float(df["feat_a"].iloc[idx])
        logged_rule_index = int(row["Rule_Index"])

        # Determine which rule should have matched first
        expected_rule_index = None
        for rule_pos, rule in enumerate(rule_set, start=1):
            condition = rule["conditions"][0]
            # Extract bucket name from "[feat_a] IS <bucket>"
            bucket_name = condition.split(" IS ", 1)[1]
            bucket_val = _BUCKET_REPRESENTATIVES[bucket_name]
            # Check if feat_val falls in this bucket (same representative → same bucket)
            # We compare by checking which bucket the value belongs to
            if _value_in_bucket(feat_val, bucket_name):
                expected_rule_index = rule_pos
                break

        if expected_rule_index is not None:
            assert logged_rule_index == expected_rule_index, (
                f"Row {idx} (feat_a={feat_val:.3f}): expected Rule_Index "
                f"{expected_rule_index} but got {logged_rule_index}. "
                f"First matching rule must win."
            )


def _value_in_bucket(val: float, bucket_name: str) -> bool:
    """Return True if val falls in the named positive-mode fuzzy bucket."""
    if bucket_name == "Very Low":
        return val <= 0.2
    if bucket_name == "Low":
        return 0.2 < val <= 0.4
    if bucket_name == "Medium":
        return 0.4 < val <= 0.6
    if bucket_name == "High":
        return 0.6 < val <= 0.8
    if bucket_name == "Very High":
        return val > 0.8
    return False


# ---------------------------------------------------------------------------
# Strategies for Property 11
# ---------------------------------------------------------------------------

@st.composite
def trade_scenario_strategy(draw: st.DrawFn) -> dict:
    """
    Generate a random price scenario for a single trade.

    Returns a dict with:
      direction      : "long" or "short"
      tp             : take-profit % (0.5 – 10.0)
      sl             : stop-loss %   (0.5 – 10.0)
      max_ret_pct    : max return % over hold period  (-15 to +15)
      min_ret_pct    : min return % over hold period  (-15 to +15, <= max_ret_pct)
      close_ret_pct  : close return % over hold period (-15 to +15)
      max_before_min : 0 or 1
    """
    direction = draw(st.sampled_from(["long", "short"]))
    tp = draw(st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))
    sl = draw(st.floats(min_value=0.5, max_value=10.0, allow_nan=False, allow_infinity=False))

    max_ret_pct = draw(st.floats(min_value=-15.0, max_value=15.0, allow_nan=False, allow_infinity=False))
    # min_ret must be <= max_ret
    min_ret_pct = draw(st.floats(min_value=-15.0, max_value=max_ret_pct, allow_nan=False, allow_infinity=False))
    close_ret_pct = draw(st.floats(min_value=-15.0, max_value=15.0, allow_nan=False, allow_infinity=False))
    max_before_min = draw(st.integers(min_value=0, max_value=1))

    return {
        "direction": direction,
        "tp": tp,
        "sl": sl,
        "max_ret_pct": max_ret_pct,
        "min_ret_pct": min_ret_pct,
        "close_ret_pct": close_ret_pct,
        "max_before_min": max_before_min,
    }


def _expected_outcome(
    direction: str,
    tp: float,
    sl: float,
    max_ret_pct: float,
    min_ret_pct: float,
    close_ret_pct: float,
    max_before_min: int,
    entry: float = 100.0,
) -> tuple[float, str]:
    """
    Compute the expected (price_return_pct, exit_reason) for a trade,
    mirroring the exact logic in CPUBacktestEngine._build_trade_outcome_single.

    To match the engine's floating-point behaviour exactly, we reconstruct
    max_ret / min_ret / close_ret the same way the engine does:
      label_price = entry * (1 + pct/100)
      engine_ret  = (label_price - entry) / entry * 100
    This round-trip is applied so both sides see identical float values.
    """
    import numpy as np

    def _roundtrip(pct: float) -> np.float32:
        label_price = entry * (1.0 + pct / 100.0)
        return np.float32((label_price - entry) / entry * 100.0)

    s_max = float(_roundtrip(max_ret_pct))
    s_min = float(_roundtrip(min_ret_pct))
    s_close = float(_roundtrip(close_ret_pct))

    if direction == "long":
        hit_tp = s_max >= tp
        hit_sl = s_min <= -sl
        if hit_tp and hit_sl:
            if max_before_min == 1:
                return float(tp), "TP"
            else:
                return float(-sl), "SL"
        if hit_tp:
            return float(tp), "TP"
        if hit_sl:
            return float(-sl), "SL"
        return float(s_close), "Time_288"
    else:  # short
        hit_tp = s_min <= -tp
        hit_sl = s_max >= sl
        if hit_tp and hit_sl:
            if max_before_min == 1:
                return float(-sl), "SL"
            else:
                return float(tp), "TP"
        if hit_tp:
            return float(tp), "TP"
        if hit_sl:
            return float(-sl), "SL"
        return float(-s_close), "Time_288"


# ---------------------------------------------------------------------------
# Property 11: Trade Outcome Correctness
# Validates: Requirements 5.2
# ---------------------------------------------------------------------------

@given(scenario=trade_scenario_strategy())
@settings(
    max_examples=500,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_11_trade_outcome_correctness(scenario: dict) -> None:
    """
    **Property 11: Trade Outcome Correctness**
    **Validates: Requirements 5.2**

    For any trade with known max_ret, min_ret, close_ret, max_before_min, tp, sl,
    the engine must produce the correct exit reason and price return:

    Long:
      - max_ret >= tp  AND min_ret > -sl          → TP,  return = +tp
      - min_ret <= -sl AND max_ret < tp            → SL,  return = -sl
      - both hit, max_before_min == 1              → TP first, return = +tp
      - both hit, max_before_min == 0              → SL first, return = -sl
      - neither hit                                → Time_288, return = close_ret

    Short:
      - min_ret <= -tp AND max_ret < sl            → TP,  return = +tp
      - max_ret >= sl  AND min_ret > -tp           → SL,  return = -sl
      - both hit, max_before_min == 1              → SL first, return = -sl
      - both hit, max_before_min == 0              → TP first, return = +tp
      - neither hit                                → Time_288, return = -close_ret
    """
    direction = scenario["direction"]
    tp = scenario["tp"]
    sl = scenario["sl"]
    max_ret_pct = scenario["max_ret_pct"]
    min_ret_pct = scenario["min_ret_pct"]
    close_ret_pct = scenario["close_ret_pct"]
    max_before_min = scenario["max_before_min"]

    entry = 100.0
    label_max = entry * (1.0 + max_ret_pct / 100.0)
    label_min = entry * (1.0 + min_ret_pct / 100.0)
    label_close = entry * (1.0 + close_ret_pct / 100.0)

    # label_min must be positive (entry price must be positive)
    assume(label_min > 0.0)

    df = pd.DataFrame(
        {
            "symbol": ["SYM"],
            "datetime": pd.date_range("2024-01-01", periods=1, freq="5min"),
            "_symbol_bar_index": [0],
            "label_open_next": [entry],
            "label_max_288": [label_max],
            "label_min_288": [label_min],
            "label_close_288": [label_close],
            "label_max_before_min": [max_before_min],
            "feat_a": [0.9],  # Very High → always matches the rule
        }
    )

    engine = CPUBacktestEngine(
        df,
        feature_modes={},
        direction=direction,
        max_hold_candles=1,
        initial_capital=10_000.0,
        leverage=1.0,
        fee_pct=0.0,
        min_position_notional=0.01,
        max_total_exposure_pct=100.0,
    )

    rule_set = [
        {
            "conditions": ["[feat_a] IS Very High"],
            "tp": tp,
            "sl": sl,
            "capital_pct": 10.0,
        }
    ]

    metrics, trade_log = engine.simulate_rule_set(rule_set, return_logs=True)

    # The single row must have produced exactly one trade
    assert len(trade_log) == 1, (
        f"Expected 1 trade, got {len(trade_log)}. "
        f"scenario={scenario}"
    )

    actual_return = float(trade_log["Price_Return_Pct"].iloc[0])
    actual_reason = str(trade_log["Exit_Reason"].iloc[0])

    expected_return, expected_reason = _expected_outcome(
        direction=direction,
        tp=tp,
        sl=sl,
        max_ret_pct=max_ret_pct,
        min_ret_pct=min_ret_pct,
        close_ret_pct=close_ret_pct,
        max_before_min=max_before_min,
    )

    assert actual_reason == expected_reason, (
        f"Exit reason mismatch: got '{actual_reason}', expected '{expected_reason}'. "
        f"direction={direction}, tp={tp:.4f}, sl={sl:.4f}, "
        f"max_ret={max_ret_pct:.4f}, min_ret={min_ret_pct:.4f}, "
        f"close_ret={close_ret_pct:.4f}, max_before_min={max_before_min}"
    )

    assert abs(actual_return - expected_return) < 1e-3, (
        f"Price return mismatch: got {actual_return:.10f}, expected {expected_return:.10f}. "
        f"direction={direction}, tp={tp:.4f}, sl={sl:.4f}, "
        f"max_ret={max_ret_pct:.4f}, min_ret={min_ret_pct:.4f}, "
        f"close_ret={close_ret_pct:.4f}, max_before_min={max_before_min}"
    )


@given(scenario=trade_scenario_strategy())
@settings(
    max_examples=300,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_11b_exit_reason_consistency_with_return(scenario: dict) -> None:
    """
    **Property 11: Trade Outcome Correctness**
    **Validates: Requirements 5.2**

    The exit reason and price return must be internally consistent:
      - "TP"       → price_return_pct == +tp  (positive)
      - "SL"       → price_return_pct == -sl  (negative)
      - "Time_288" → price_return_pct == close_ret (long) or -close_ret (short)
    """
    direction = scenario["direction"]
    tp = scenario["tp"]
    sl = scenario["sl"]
    max_ret_pct = scenario["max_ret_pct"]
    min_ret_pct = scenario["min_ret_pct"]
    close_ret_pct = scenario["close_ret_pct"]
    max_before_min = scenario["max_before_min"]

    entry = 100.0
    label_max = entry * (1.0 + max_ret_pct / 100.0)
    label_min = entry * (1.0 + min_ret_pct / 100.0)
    label_close = entry * (1.0 + close_ret_pct / 100.0)

    assume(label_min > 0.0)

    df = pd.DataFrame(
        {
            "symbol": ["SYM"],
            "datetime": pd.date_range("2024-01-01", periods=1, freq="5min"),
            "_symbol_bar_index": [0],
            "label_open_next": [entry],
            "label_max_288": [label_max],
            "label_min_288": [label_min],
            "label_close_288": [label_close],
            "label_max_before_min": [max_before_min],
            "feat_a": [0.9],
        }
    )

    engine = CPUBacktestEngine(
        df,
        feature_modes={},
        direction=direction,
        max_hold_candles=1,
        initial_capital=10_000.0,
        leverage=1.0,
        fee_pct=0.0,
        min_position_notional=0.01,
        max_total_exposure_pct=100.0,
    )

    rule_set = [
        {
            "conditions": ["[feat_a] IS Very High"],
            "tp": tp,
            "sl": sl,
            "capital_pct": 10.0,
        }
    ]

    metrics, trade_log = engine.simulate_rule_set(rule_set, return_logs=True)

    assert len(trade_log) == 1

    actual_return = float(trade_log["Price_Return_Pct"].iloc[0])
    actual_reason = str(trade_log["Exit_Reason"].iloc[0])

    if actual_reason == "TP":
        assert abs(actual_return - tp) < 1e-3, (
            f"TP exit must return +tp={tp:.6f}, got {actual_return:.6f}"
        )
    elif actual_reason == "SL":
        assert abs(actual_return - (-sl)) < 1e-3, (
            f"SL exit must return -sl={-sl:.6f}, got {actual_return:.6f}"
        )
    elif actual_reason == "Time_288":
        # Use the same round-trip as the engine: price → pct
        label_close = entry * (1.0 + close_ret_pct / 100.0)
        engine_close_ret = (label_close - entry) / entry * 100.0
        if direction == "long":
            expected_time_ret = engine_close_ret
        else:
            expected_time_ret = -engine_close_ret
        assert abs(actual_return - expected_time_ret) < 1e-3, (
            f"Time_288 exit must return {expected_time_ret:.6f} "
            f"(direction={direction}, close_ret={close_ret_pct:.6f}), "
            f"got {actual_return:.6f}"
        )
    else:
        pytest.fail(f"Unexpected exit reason: '{actual_reason}'")


# ---------------------------------------------------------------------------
# Shared helpers for Properties 14, 15, 28
# ---------------------------------------------------------------------------

def _make_simple_df(
    n: int,
    symbol: str = "SYM",
    entry: float = 100.0,
    label_max: float = 106.0,
    label_min: float = 97.0,
    label_close: float = 103.0,
    max_before_min: int = 1,
    feature_val: float = 0.9,
) -> pd.DataFrame:
    """Build a minimal single-symbol DataFrame for CPUBacktestEngine."""
    return pd.DataFrame(
        {
            "symbol": [symbol] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [entry] * n,
            "label_max_288": [label_max] * n,
            "label_min_288": [label_min] * n,
            "label_close_288": [label_close] * n,
            "label_max_before_min": [max_before_min] * n,
            "feat_a": [feature_val] * n,
        }
    )


def _make_engine_custom(df: pd.DataFrame, direction: str = "long", **kw) -> CPUBacktestEngine:
    return CPUBacktestEngine(df, feature_modes={}, direction=direction, **kw)


# ---------------------------------------------------------------------------
# Strategies for Property 14
# ---------------------------------------------------------------------------

@st.composite
def fee_deduction_scenario(draw: st.DrawFn) -> dict:
    """
    Generate a scenario for Property 14.

    Produces fee_pct, capital_pct, initial_capital, and tp values.
    The DataFrame is constructed so that every row fires a TP trade
    (label_max always exceeds TP threshold; label_min never hits SL).
    """
    fee_pct = draw(st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False))
    capital_pct = draw(st.floats(min_value=1.0, max_value=100.0, allow_nan=False, allow_infinity=False))
    initial_capital = draw(st.floats(min_value=100.0, max_value=100_000.0, allow_nan=False, allow_infinity=False))
    tp = draw(st.floats(min_value=0.5, max_value=20.0, allow_nan=False, allow_infinity=False))

    entry = 100.0
    # label_max always hits TP; label_min never hits SL (sl=2%)
    label_max = entry * (1.0 + (tp + 1.0) / 100.0)
    label_min = entry * 0.99

    return {
        "fee_pct": fee_pct,
        "capital_pct": capital_pct,
        "initial_capital": initial_capital,
        "tp": tp,
        "entry": entry,
        "label_max": label_max,
        "label_min": label_min,
    }


# ---------------------------------------------------------------------------
# Property 14: Fee Deduction Correctness
# Validates: Requirements 5.6
# ---------------------------------------------------------------------------

@given(scenario=fee_deduction_scenario())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_14_fee_deduction_correctness(scenario: dict) -> None:
    """
    **Property 14: Fee Deduction Correctness**
    **Validates: Requirements 5.6**

    For any trade:
      fee = position_notional * fee_pct / 100
      net_pnl = gross_pnl - fee

    This property verifies that the engine correctly deducts the round-trip
    fee from every trade's gross PnL, regardless of fee_pct, capital_pct,
    initial_capital, or TP value.
    """
    fee_pct = scenario["fee_pct"]
    capital_pct = scenario["capital_pct"]
    initial_capital = scenario["initial_capital"]
    tp = scenario["tp"]
    entry = scenario["entry"]
    label_max = scenario["label_max"]
    label_min = scenario["label_min"]

    n = 5
    df = pd.DataFrame(
        {
            "symbol": ["SYM"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [entry] * n,
            "label_max_288": [label_max] * n,
            "label_min_288": [label_min] * n,
            "label_close_288": [entry * 1.02] * n,
            "label_max_before_min": [1] * n,
            "feat_a": [0.9] * n,
        }
    )

    rule_set = [
        {
            "conditions": ["[feat_a] IS Very High"],
            "tp": tp,
            "sl": 2.0,
            "capital_pct": capital_pct,
        }
    ]

    engine = _make_engine_custom(
        df,
        direction="long",
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        max_hold_candles=1,
        min_position_notional=0.0,
    )

    metrics, logs = engine.simulate_rule_set(rule_set, return_logs=True)

    if len(logs) == 0:
        # No trades executed (e.g., position_notional below min_notional)
        return

    fee_rate = fee_pct / 100.0

    for _, row in logs.iterrows():
        position_notional = float(row["Position_Notional"])
        gross_pnl = float(row["Gross_PnL"])
        fee = float(row["Fee"])
        net_pnl = float(row["Net_PnL"])

        # fee must equal position_notional * fee_rate
        expected_fee = position_notional * fee_rate
        assert fee == pytest.approx(expected_fee, rel=1e-3, abs=1e-2), (
            f"Fee mismatch: expected position_notional * fee_rate = "
            f"{position_notional:.6f} * {fee_rate:.8f} = {expected_fee:.8f}, "
            f"got {fee:.8f}"
        )

        # net_pnl must equal gross_pnl - fee
        expected_net_pnl = gross_pnl - fee
        assert net_pnl == pytest.approx(expected_net_pnl, rel=1e-3, abs=1e-2), (
            f"Net PnL mismatch: expected gross_pnl - fee = "
            f"{gross_pnl:.8f} - {fee:.8f} = {expected_net_pnl:.8f}, "
            f"got {net_pnl:.8f}"
        )


# ---------------------------------------------------------------------------
# Strategies for Property 15
# ---------------------------------------------------------------------------

@st.composite
def equity_tracking_scenario(draw: st.DrawFn) -> dict:
    """
    Generate a scenario for Property 15.

    Produces a DataFrame with n rows where every row fires a trade.
    Rows alternate between TP wins and SL losses for a non-trivial equity path.
    """
    n = draw(st.integers(min_value=2, max_value=20))
    initial_capital = draw(st.floats(min_value=500.0, max_value=10_000.0, allow_nan=False, allow_infinity=False))
    capital_pct = draw(st.floats(min_value=5.0, max_value=50.0, allow_nan=False, allow_infinity=False))
    fee_pct = draw(st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False))
    tp = draw(st.floats(min_value=1.0, max_value=10.0, allow_nan=False, allow_infinity=False))
    sl = draw(st.floats(min_value=0.5, max_value=5.0, allow_nan=False, allow_infinity=False))

    entry = 100.0
    label_max_list = []
    label_min_list = []
    mbm_list = []
    for i in range(n):
        if i % 2 == 0:
            # Win: TP hit, SL not hit
            label_max_list.append(entry * (1.0 + (tp + 1.0) / 100.0))
            label_min_list.append(entry * 0.99)
            mbm_list.append(1)
        else:
            # Loss: SL hit, TP not hit
            label_max_list.append(entry * 1.005)
            label_min_list.append(entry * (1.0 - (sl + 1.0) / 100.0))
            mbm_list.append(0)

    return {
        "n": n,
        "initial_capital": initial_capital,
        "capital_pct": capital_pct,
        "fee_pct": fee_pct,
        "tp": tp,
        "sl": sl,
        "entry": entry,
        "label_max_list": label_max_list,
        "label_min_list": label_min_list,
        "mbm_list": mbm_list,
    }


# ---------------------------------------------------------------------------
# Property 15: Equity Tracking Consistency
# Validates: Requirements 5.7
# ---------------------------------------------------------------------------

@given(scenario=equity_tracking_scenario())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_15_equity_tracking_consistency(scenario: dict) -> None:
    """
    **Property 15: Equity Tracking Consistency**
    **Validates: Requirements 5.7**

    For any sequence of realized trades:
      final_equity = initial_capital + sum(net_pnl for all realized trades)
      total_return_pct = (final_equity / initial_capital - 1) * 100

    This property verifies that the engine's equity accounting is internally
    consistent: the reported final_equity and total_return_pct must match
    the sum of all realized net PnLs from the trade log.
    """
    n = scenario["n"]
    initial_capital = scenario["initial_capital"]
    capital_pct = scenario["capital_pct"]
    fee_pct = scenario["fee_pct"]
    tp = scenario["tp"]
    sl = scenario["sl"]
    entry = scenario["entry"]
    label_max_list = scenario["label_max_list"]
    label_min_list = scenario["label_min_list"]
    mbm_list = scenario["mbm_list"]

    df = pd.DataFrame(
        {
            "symbol": ["SYM"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [entry] * n,
            "label_max_288": label_max_list,
            "label_min_288": label_min_list,
            "label_close_288": [entry * 1.01] * n,
            "label_max_before_min": mbm_list,
            "feat_a": [0.9] * n,
        }
    )

    rule_set = [
        {
            "conditions": ["[feat_a] IS Very High"],
            "tp": tp,
            "sl": sl,
            "capital_pct": capital_pct,
        }
    ]

    engine = _make_engine_custom(
        df,
        direction="long",
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        max_hold_candles=1,
        min_position_notional=0.0,
    )

    metrics, logs = engine.simulate_rule_set(rule_set, return_logs=True)

    if len(logs) == 0:
        # No trades: equity must equal initial_capital
        assert metrics["final_equity"] == pytest.approx(initial_capital, rel=1e-9), (
            "With zero trades, final_equity must equal initial_capital."
        )
        assert metrics["total_return_pct"] == pytest.approx(0.0, abs=1e-9), (
            "With zero trades, total_return_pct must be 0.0."
        )
        return

    # Only realized trades contribute to equity
    realized_logs = logs[logs["Realized"] == True]
    sum_net_pnl = float(realized_logs["Net_PnL"].sum())

    expected_final_equity = initial_capital + sum_net_pnl
    expected_return_pct = (expected_final_equity / initial_capital - 1.0) * 100.0

    assert metrics["final_equity"] == pytest.approx(expected_final_equity, rel=1e-3, abs=1e-3), (
        f"final_equity mismatch: "
        f"initial_capital={initial_capital:.4f} + sum(net_pnl)={sum_net_pnl:.6f} "
        f"= {expected_final_equity:.6f}, but got {metrics['final_equity']:.6f}"
    )

    assert metrics["total_return_pct"] == pytest.approx(expected_return_pct, rel=1e-3, abs=1e-3), (
        f"total_return_pct mismatch: "
        f"expected {expected_return_pct:.6f}%, got {metrics['total_return_pct']:.6f}%"
    )


# ---------------------------------------------------------------------------
# Strategies for Property 28
# ---------------------------------------------------------------------------

@st.composite
def multi_symbol_scenario(draw: st.DrawFn) -> dict:
    """
    Generate a multi-symbol scenario for Property 28.

    Produces a DataFrame with 2–4 symbols, each having 3–10 rows.
    All rows fire a trade (feat_a = 0.9 → Very High).
    All trades hit TP (deterministic wins) for predictable per-symbol PnL.
    """
    num_symbols = draw(st.integers(min_value=2, max_value=4))
    symbol_names = [f"SYM_{chr(ord('A') + i)}" for i in range(num_symbols)]

    rows_per_symbol = {
        sym: draw(st.integers(min_value=3, max_value=10))
        for sym in symbol_names
    }

    initial_capital = draw(st.floats(min_value=500.0, max_value=5_000.0, allow_nan=False, allow_infinity=False))
    capital_pct = draw(st.floats(min_value=1.0, max_value=20.0, allow_nan=False, allow_infinity=False))
    fee_pct = draw(st.floats(min_value=0.01, max_value=0.5, allow_nan=False, allow_infinity=False))

    entry = 100.0
    tp = 4.0
    sl = 2.0
    # label_max always hits TP; label_min never hits SL
    label_max = entry * (1.0 + (tp + 1.0) / 100.0)
    label_min = entry * 0.99

    all_rows = []
    global_bar = 0
    for sym in symbol_names:
        n = rows_per_symbol[sym]
        for j in range(n):
            all_rows.append(
                {
                    "symbol": sym,
                    "datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=5 * global_bar),
                    "_symbol_bar_index": j,
                    "label_open_next": entry,
                    "label_max_288": label_max,
                    "label_min_288": label_min,
                    "label_close_288": entry * 1.02,
                    "label_max_before_min": 1,
                    "feat_a": 0.9,
                }
            )
            global_bar += 1

    df = pd.DataFrame(all_rows)

    return {
        "df": df,
        "rows_per_symbol": rows_per_symbol,
        "initial_capital": initial_capital,
        "capital_pct": capital_pct,
        "fee_pct": fee_pct,
        "tp": tp,
        "sl": sl,
    }


# ---------------------------------------------------------------------------
# Property 28: Per-Symbol Metrics Consistency
# Validates: Requirements 15.1
# ---------------------------------------------------------------------------

@given(scenario=multi_symbol_scenario())
@settings(
    max_examples=100,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example],
)
def test_property_28_per_symbol_metrics_consistency(scenario: dict) -> None:
    """
    **Property 28: Per-Symbol Metrics Consistency**
    **Validates: Requirements 15.1**

    For any multi-symbol dataset:
      1. sum of per-symbol trade_counts == total executed_trades
      2. sum of per-symbol net_pnl ≈ total_return_pct * initial_capital / 100
         (approximately, within floating-point tolerance)

    This property verifies that the per-symbol breakdown is internally
    consistent with the aggregate metrics reported by the engine.
    """
    df = scenario["df"]
    initial_capital = scenario["initial_capital"]
    capital_pct = scenario["capital_pct"]
    fee_pct = scenario["fee_pct"]
    tp = scenario["tp"]
    sl = scenario["sl"]

    rule_set = [
        {
            "conditions": ["[feat_a] IS Very High"],
            "tp": tp,
            "sl": sl,
            "capital_pct": capital_pct,
        }
    ]

    engine = _make_engine_custom(
        df,
        direction="long",
        initial_capital=initial_capital,
        fee_pct=fee_pct,
        max_hold_candles=1,
        min_position_notional=0.0,
    )

    metrics, logs = engine.simulate_rule_set(rule_set, return_logs=True)

    per_sym = metrics["per_symbol_metrics"]
    executed_trades = metrics["executed_trades"]
    total_return_pct = metrics["total_return_pct"]

    if executed_trades == 0:
        total_sym_trades = sum(v["trade_count"] for v in per_sym.values())
        assert total_sym_trades == 0, (
            "With zero executed_trades, sum of per-symbol trade_counts must be 0."
        )
        return

    # --- Invariant 1: sum of per-symbol trade_counts == executed_trades ---
    total_sym_trades = sum(v["trade_count"] for v in per_sym.values())
    assert total_sym_trades == executed_trades, (
        f"Sum of per-symbol trade_counts ({total_sym_trades}) != "
        f"executed_trades ({executed_trades}). "
        f"Per-symbol breakdown: {per_sym}"
    )

    # --- Invariant 2: sum of per-symbol net_pnl ≈ total_return_pct * initial_capital / 100 ---
    total_sym_net_pnl = sum(v["net_pnl"] for v in per_sym.values())
    expected_total_net_pnl = total_return_pct * initial_capital / 100.0

    # Use a generous tolerance to account for floating-point accumulation
    tolerance = max(1e-4 * abs(expected_total_net_pnl), 1e-4)
    assert abs(total_sym_net_pnl - expected_total_net_pnl) <= tolerance, (
        f"Sum of per-symbol net_pnl ({total_sym_net_pnl:.8f}) does not match "
        f"total_return_pct * initial_capital / 100 "
        f"({total_return_pct:.6f}% * {initial_capital:.2f} / 100 = {expected_total_net_pnl:.8f}). "
        f"Difference: {abs(total_sym_net_pnl - expected_total_net_pnl):.2e}"
    )

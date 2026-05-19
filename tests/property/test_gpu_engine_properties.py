"""
Property-based tests for gpu_fuzzy_trader.backtest.gpu_engine.GPUBacktestEngine

Property 16: GPU-CPU Numerical Parity
  **Validates: Requirements 6.1**
  For any valid dataset and any chromosome, GPUBacktestEngine.simulate_rule_batch()
  must produce results numerically equivalent to CPUBacktestEngine.simulate_rule_set()
  within 1e-4 relative tolerance on:
    - total_return_pct
    - max_drawdown_pct
    - win_rate
    - profit_factor

Key design notes:
  - The GPU engine uses chromosome-based matching (integer gene values vs data_matrix).
  - The CPU engine uses threshold-based matching (condition strings).
  - For parity, we use binary features (feat_binary=1) so chromosome [1] matches
    the same rows as the condition "[feat_binary] IS Active (1)".
  - All tests skip gracefully if JAX is not installed.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# JAX availability guard
# ---------------------------------------------------------------------------

jax_available = True
try:
    import jax  # noqa: F401
except ImportError:
    jax_available = False

pytestmark = pytest.mark.skipif(
    not jax_available,
    reason="JAX not installed",
)

# ---------------------------------------------------------------------------
# Conditional imports (only executed when JAX is available)
# ---------------------------------------------------------------------------

if jax_available:
    from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
    from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine


# ---------------------------------------------------------------------------
# DataFrame construction helpers
# ---------------------------------------------------------------------------

def _make_parity_df(
    n: int,
    label_max_list: list[float],
    label_min_list: list[float],
    label_close_list: list[float],
    mbm_list: list[int],
    entry: float = 100.0,
) -> pd.DataFrame:
    """
    Build a minimal single-symbol DataFrame suitable for GPU-CPU parity tests.

    Uses a single binary feature 'feat_binary' set to 1 for all rows, so:
      - CPU condition "[feat_binary] IS Active (1)" matches all rows.
      - GPU chromosome [1] (binary gene=1) also matches all rows.

    Parameters
    ----------
    n : int
        Number of rows.
    label_max_list, label_min_list, label_close_list : list[float]
        Per-row label prices (absolute, not percentages).
    mbm_list : list[int]
        Per-row max_before_min flags (0 or 1).
    entry : float
        Entry price used for all rows.
    """
    return pd.DataFrame(
        {
            "symbol": ["SYM"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": list(range(n)),
            "label_open_next": [entry] * n,
            "label_max_288": label_max_list,
            "label_min_288": label_min_list,
            "label_close_288": label_close_list,
            "label_max_before_min": mbm_list,
            "feat_binary": [1] * n,
        }
    )


def _make_engines(
    df: pd.DataFrame,
    direction: str,
    initial_capital: float = 1000.0,
    max_hold_candles: int = 5,
    fee_pct: float = 0.20,
    leverage: float = 1.0,
    max_total_exposure_pct: float = 100.0,
    min_position_notional: float = 0.01,
) -> tuple["GPUBacktestEngine", "CPUBacktestEngine"]:
    """Construct matching GPU and CPU engines with identical parameters."""
    feature_modes = {"feat_binary": "binary"}
    kwargs = dict(
        initial_capital=initial_capital,
        max_hold_candles=max_hold_candles,
        fee_pct=fee_pct,
        leverage=leverage,
        max_total_exposure_pct=max_total_exposure_pct,
        min_position_notional=min_position_notional,
    )
    gpu_eng = GPUBacktestEngine(df, feature_modes, direction, **kwargs)
    cpu_eng = CPUBacktestEngine(df, feature_modes, direction, **kwargs)
    return gpu_eng, cpu_eng


def _assert_parity(
    gpu_result: dict,
    cpu_result: dict,
    rel_tol: float = 1e-4,
    abs_tol: float = 1e-6,
    context: str = "",
) -> None:
    """
    Assert that GPU and CPU metrics are numerically equivalent within tolerance.

    Uses relative tolerance (rel_tol) when the reference value is non-trivial,
    and absolute tolerance (abs_tol) when the reference is near zero.
    """
    metrics_to_check = [
        "sortino_ratio",
        "total_return_pct",
        "max_drawdown_pct",
        "win_rate",
        "profit_factor",
    ]
    for metric in metrics_to_check:
        gpu_val = float(gpu_result[metric])
        cpu_val = float(cpu_result[metric])

        # Use relative tolerance when reference is non-trivial
        if abs(cpu_val) > abs_tol:
            rel_diff = abs(gpu_val - cpu_val) / abs(cpu_val)
            assert rel_diff <= rel_tol, (
                f"[{metric}] GPU={gpu_val:.8f} vs CPU={cpu_val:.8f} "
                f"relative diff={rel_diff:.2e} exceeds {rel_tol:.0e}. "
                f"{context}"
            )
        else:
            # Both should be near zero
            abs_diff = abs(gpu_val - cpu_val)
            assert abs_diff <= abs_tol * 100, (
                f"[{metric}] GPU={gpu_val:.8f} vs CPU={cpu_val:.8f} "
                f"absolute diff={abs_diff:.2e} (both near zero). "
                f"{context}"
            )


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

@st.composite
def parity_scenario_strategy(draw: st.DrawFn) -> dict:
    """
    Generate a random dataset and trade parameters for GPU-CPU parity testing.

    Produces:
      - n_rows: number of rows (5–40)
      - tp: take-profit % (0.5–8.0)
      - sl: stop-loss % (0.5–5.0)
      - capital_pct: capital allocation % (10–80)
      - direction: "long" or "short"
      - Per-row price scenario: label_max, label_min, label_close, max_before_min

    The binary feature 'feat_binary' is always 1, so the chromosome [1]
    and the CPU condition "[feat_binary] IS Active (1)" match identical rows.
    """
    n_rows = draw(st.integers(min_value=5, max_value=40))
    tp = draw(st.floats(min_value=0.5, max_value=8.0,
              allow_nan=False, allow_infinity=False))
    sl = draw(st.floats(min_value=0.5, max_value=5.0,
              allow_nan=False, allow_infinity=False))
    capital_pct = draw(st.floats(min_value=10.0, max_value=80.0,
                       allow_nan=False, allow_infinity=False))
    direction = draw(st.sampled_from(["long", "short"]))

    entry = 100.0

    # Per-row price scenarios
    label_max_list = []
    label_min_list = []
    label_close_list = []
    mbm_list = []

    for _ in range(n_rows):
        # max_ret_pct: how much price rose above entry (0–15%)
        max_ret_pct = draw(st.floats(min_value=0.0, max_value=15.0,
                                     allow_nan=False, allow_infinity=False))
        # min_ret_pct: how much price fell below entry (0–10%)
        min_ret_pct = draw(st.floats(min_value=0.0, max_value=10.0,
                                     allow_nan=False, allow_infinity=False))
        # close_ret_pct: close return relative to entry (-8% to +8%)
        close_ret_pct = draw(st.floats(min_value=-8.0, max_value=8.0,
                                       allow_nan=False, allow_infinity=False))
        mbm = draw(st.integers(min_value=0, max_value=1))

        label_max_list.append(entry * (1.0 + max_ret_pct / 100.0))
        label_min_list.append(entry * (1.0 - min_ret_pct / 100.0))
        label_close_list.append(entry * (1.0 + close_ret_pct / 100.0))
        mbm_list.append(mbm)

    return {
        "n_rows": n_rows,
        "tp": tp,
        "sl": sl,
        "capital_pct": capital_pct,
        "direction": direction,
        "entry": entry,
        "label_max_list": label_max_list,
        "label_min_list": label_min_list,
        "label_close_list": label_close_list,
        "mbm_list": mbm_list,
    }


# ---------------------------------------------------------------------------
# Property 16: GPU-CPU Numerical Parity
# Validates: Requirements 6.1
# ---------------------------------------------------------------------------

@given(scenario=parity_scenario_strategy())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.large_base_example],
)
def test_property_16_gpu_cpu_total_return_parity(scenario: dict) -> None:
    """
    **Property 16: GPU-CPU Numerical Parity**
    **Validates: Requirements 6.1**

    For any valid dataset and any chromosome, GPUBacktestEngine.simulate_rule_batch()
    must produce total_return_pct numerically equivalent to
    CPUBacktestEngine.simulate_rule_set() within 1e-4 relative tolerance.

    Strategy:
      - Use feat_binary=1 for all rows.
      - CPU rule: "[feat_binary] IS Active (1)" → matches all rows.
      - GPU chromosome: [1] (binary gene=1) → also matches all rows.
      - Both engines see identical matched rows and identical trade parameters.
    """
    n = scenario["n_rows"]
    tp = scenario["tp"]
    sl = scenario["sl"]
    capital_pct = scenario["capital_pct"]
    direction = scenario["direction"]

    df = _make_parity_df(
        n=n,
        label_max_list=scenario["label_max_list"],
        label_min_list=scenario["label_min_list"],
        label_close_list=scenario["label_close_list"],
        mbm_list=scenario["mbm_list"],
        entry=scenario["entry"],
    )

    gpu_eng, cpu_eng = _make_engines(df, direction, max_hold_candles=3)

    # CPU path: condition-based rule matching
    cpu_rule_set = [
        {
            "conditions": ["[feat_binary] IS Active (1)"],
            "tp": tp,
            "sl": sl,
            "capital_pct": capital_pct,
        }
    ]
    cpu_result = cpu_eng.simulate_rule_set(cpu_rule_set)

    # GPU path: chromosome-based matching (binary gene=1 → Active (1))
    chrom = np.array([[1]], dtype=np.int32)
    gpu_results = gpu_eng.simulate_rule_batch(
        chrom, tp=tp, sl=sl, capital_pct=capital_pct)
    gpu_result = gpu_results[0]

    context = (
        f"n={n}, tp={tp:.4f}, sl={sl:.4f}, capital_pct={capital_pct:.2f}, "
        f"direction={direction}"
    )

    cpu_ret = cpu_result["total_return_pct"]
    gpu_ret = gpu_result["total_return_pct"]

    if abs(cpu_ret) > 1e-6:
        rel_diff = abs(gpu_ret - cpu_ret) / abs(cpu_ret)
        assert rel_diff <= 1e-4, (
            f"total_return_pct: GPU={gpu_ret:.8f} vs CPU={cpu_ret:.8f} "
            f"relative diff={rel_diff:.2e} exceeds 1e-4. {context}"
        )
    else:
        assert abs(gpu_ret - cpu_ret) <= 1e-4, (
            f"total_return_pct: GPU={gpu_ret:.8f} vs CPU={cpu_ret:.8f} "
            f"absolute diff={abs(gpu_ret - cpu_ret):.2e} (both near zero). {context}"
        )


@given(scenario=parity_scenario_strategy())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.large_base_example],
)
def test_property_16_gpu_cpu_max_drawdown_parity(scenario: dict) -> None:
    """
    **Property 16: GPU-CPU Numerical Parity**
    **Validates: Requirements 6.1**

    For any valid dataset and any chromosome, GPUBacktestEngine.simulate_rule_batch()
    must produce max_drawdown_pct numerically equivalent to
    CPUBacktestEngine.simulate_rule_set() within 1e-4 relative tolerance.
    """
    n = scenario["n_rows"]
    tp = scenario["tp"]
    sl = scenario["sl"]
    capital_pct = scenario["capital_pct"]
    direction = scenario["direction"]

    df = _make_parity_df(
        n=n,
        label_max_list=scenario["label_max_list"],
        label_min_list=scenario["label_min_list"],
        label_close_list=scenario["label_close_list"],
        mbm_list=scenario["mbm_list"],
        entry=scenario["entry"],
    )

    gpu_eng, cpu_eng = _make_engines(df, direction, max_hold_candles=3)

    cpu_rule_set = [
        {
            "conditions": ["[feat_binary] IS Active (1)"],
            "tp": tp,
            "sl": sl,
            "capital_pct": capital_pct,
        }
    ]
    cpu_result = cpu_eng.simulate_rule_set(cpu_rule_set)

    chrom = np.array([[1]], dtype=np.int32)
    gpu_results = gpu_eng.simulate_rule_batch(
        chrom, tp=tp, sl=sl, capital_pct=capital_pct)
    gpu_result = gpu_results[0]

    context = (
        f"n={n}, tp={tp:.4f}, sl={sl:.4f}, capital_pct={capital_pct:.2f}, "
        f"direction={direction}"
    )

    cpu_dd = cpu_result["max_drawdown_pct"]
    gpu_dd = gpu_result["max_drawdown_pct"]

    # max_drawdown_pct must be non-negative
    assert gpu_dd >= 0.0, f"GPU max_drawdown_pct must be >= 0, got {gpu_dd}. {context}"

    if abs(cpu_dd) > 1e-6:
        rel_diff = abs(gpu_dd - cpu_dd) / abs(cpu_dd)
        assert rel_diff <= 1e-4, (
            f"max_drawdown_pct: GPU={gpu_dd:.8f} vs CPU={cpu_dd:.8f} "
            f"relative diff={rel_diff:.2e} exceeds 1e-4. {context}"
        )
    else:
        assert abs(gpu_dd - cpu_dd) <= 1e-4, (
            f"max_drawdown_pct: GPU={gpu_dd:.8f} vs CPU={cpu_dd:.8f} "
            f"absolute diff={abs(gpu_dd - cpu_dd):.2e} (both near zero). {context}"
        )


@given(scenario=parity_scenario_strategy())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.large_base_example],
)
def test_property_16_gpu_cpu_win_rate_parity(scenario: dict) -> None:
    """
    **Property 16: GPU-CPU Numerical Parity**
    **Validates: Requirements 6.1**

    For any valid dataset and any chromosome, GPUBacktestEngine.simulate_rule_batch()
    must produce win_rate numerically equivalent to
    CPUBacktestEngine.simulate_rule_set() within 1e-4 relative tolerance.
    """
    n = scenario["n_rows"]
    tp = scenario["tp"]
    sl = scenario["sl"]
    capital_pct = scenario["capital_pct"]
    direction = scenario["direction"]

    df = _make_parity_df(
        n=n,
        label_max_list=scenario["label_max_list"],
        label_min_list=scenario["label_min_list"],
        label_close_list=scenario["label_close_list"],
        mbm_list=scenario["mbm_list"],
        entry=scenario["entry"],
    )

    gpu_eng, cpu_eng = _make_engines(df, direction, max_hold_candles=3)

    cpu_rule_set = [
        {
            "conditions": ["[feat_binary] IS Active (1)"],
            "tp": tp,
            "sl": sl,
            "capital_pct": capital_pct,
        }
    ]
    cpu_result = cpu_eng.simulate_rule_set(cpu_rule_set)

    chrom = np.array([[1]], dtype=np.int32)
    gpu_results = gpu_eng.simulate_rule_batch(
        chrom, tp=tp, sl=sl, capital_pct=capital_pct)
    gpu_result = gpu_results[0]

    context = (
        f"n={n}, tp={tp:.4f}, sl={sl:.4f}, capital_pct={capital_pct:.2f}, "
        f"direction={direction}"
    )

    cpu_wr = cpu_result["win_rate"]
    gpu_wr = gpu_result["win_rate"]

    # win_rate must be in [0, 100]
    assert 0.0 <= gpu_wr <= 100.0, (
        f"GPU win_rate={gpu_wr} out of [0, 100] range. {context}"
    )

    if abs(cpu_wr) > 1e-6:
        rel_diff = abs(gpu_wr - cpu_wr) / abs(cpu_wr)
        assert rel_diff <= 1e-4, (
            f"win_rate: GPU={gpu_wr:.8f} vs CPU={cpu_wr:.8f} "
            f"relative diff={rel_diff:.2e} exceeds 1e-4. {context}"
        )
    else:
        assert abs(gpu_wr - cpu_wr) <= 1e-4, (
            f"win_rate: GPU={gpu_wr:.8f} vs CPU={cpu_wr:.8f} "
            f"absolute diff={abs(gpu_wr - cpu_wr):.2e} (both near zero). {context}"
        )


@given(scenario=parity_scenario_strategy())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.large_base_example],
)
def test_property_16_gpu_cpu_profit_factor_parity(scenario: dict) -> None:
    """
    **Property 16: GPU-CPU Numerical Parity**
    **Validates: Requirements 6.1**

    For any valid dataset and any chromosome, GPUBacktestEngine.simulate_rule_batch()
    must produce profit_factor numerically equivalent to
    CPUBacktestEngine.simulate_rule_set() within 1e-4 relative tolerance.

    Note: profit_factor is capped at 99.0 when there are no losses; both
    engines apply the same cap, so parity still holds.
    """
    n = scenario["n_rows"]
    tp = scenario["tp"]
    sl = scenario["sl"]
    capital_pct = scenario["capital_pct"]
    direction = scenario["direction"]

    df = _make_parity_df(
        n=n,
        label_max_list=scenario["label_max_list"],
        label_min_list=scenario["label_min_list"],
        label_close_list=scenario["label_close_list"],
        mbm_list=scenario["mbm_list"],
        entry=scenario["entry"],
    )

    gpu_eng, cpu_eng = _make_engines(df, direction, max_hold_candles=3)

    cpu_rule_set = [
        {
            "conditions": ["[feat_binary] IS Active (1)"],
            "tp": tp,
            "sl": sl,
            "capital_pct": capital_pct,
        }
    ]
    cpu_result = cpu_eng.simulate_rule_set(cpu_rule_set)

    chrom = np.array([[1]], dtype=np.int32)
    gpu_results = gpu_eng.simulate_rule_batch(
        chrom, tp=tp, sl=sl, capital_pct=capital_pct)
    gpu_result = gpu_results[0]

    context = (
        f"n={n}, tp={tp:.4f}, sl={sl:.4f}, capital_pct={capital_pct:.2f}, "
        f"direction={direction}"
    )

    cpu_pf = cpu_result["profit_factor"]
    gpu_pf = gpu_result["profit_factor"]

    # profit_factor must be non-negative
    assert gpu_pf >= 0.0, f"GPU profit_factor must be >= 0, got {gpu_pf}. {context}"

    if abs(cpu_pf) > 1e-6:
        rel_diff = abs(gpu_pf - cpu_pf) / abs(cpu_pf)
        assert rel_diff <= 1e-4, (
            f"profit_factor: GPU={gpu_pf:.8f} vs CPU={cpu_pf:.8f} "
            f"relative diff={rel_diff:.2e} exceeds 1e-4. {context}"
        )
    else:
        assert abs(gpu_pf - cpu_pf) <= 1e-4, (
            f"profit_factor: GPU={gpu_pf:.8f} vs CPU={cpu_pf:.8f} "
            f"absolute diff={abs(gpu_pf - cpu_pf):.2e} (both near zero). {context}"
        )


@given(scenario=parity_scenario_strategy())
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.large_base_example],
)
def test_property_16_gpu_cpu_all_metrics_parity(scenario: dict) -> None:
    """
    **Property 16: GPU-CPU Numerical Parity**
    **Validates: Requirements 6.1**

    Omnibus test: for any valid dataset and chromosome, ALL four metrics
    (total_return_pct, max_drawdown_pct, win_rate, profit_factor) produced by
    GPUBacktestEngine.simulate_rule_batch() must be within 1e-4 relative
    tolerance of CPUBacktestEngine.simulate_rule_set().

    This is the primary parity property required by Requirement 6.1 and 6.6.
    """
    n = scenario["n_rows"]
    tp = scenario["tp"]
    sl = scenario["sl"]
    capital_pct = scenario["capital_pct"]
    direction = scenario["direction"]

    df = _make_parity_df(
        n=n,
        label_max_list=scenario["label_max_list"],
        label_min_list=scenario["label_min_list"],
        label_close_list=scenario["label_close_list"],
        mbm_list=scenario["mbm_list"],
        entry=scenario["entry"],
    )

    gpu_eng, cpu_eng = _make_engines(df, direction, max_hold_candles=3)

    cpu_rule_set = [
        {
            "conditions": ["[feat_binary] IS Active (1)"],
            "tp": tp,
            "sl": sl,
            "capital_pct": capital_pct,
        }
    ]
    cpu_result = cpu_eng.simulate_rule_set(cpu_rule_set)

    chrom = np.array([[1]], dtype=np.int32)
    gpu_results = gpu_eng.simulate_rule_batch(
        chrom, tp=tp, sl=sl, capital_pct=capital_pct)
    gpu_result = gpu_results[0]

    context = (
        f"n={n}, tp={tp:.4f}, sl={sl:.4f}, capital_pct={capital_pct:.2f}, "
        f"direction={direction}"
    )

    _assert_parity(gpu_result, cpu_result, rel_tol=1e-4, context=context)


@given(scenario=parity_scenario_strategy())
@settings(
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow,
                           HealthCheck.large_base_example],
)
def test_property_16_zero_trades_parity(scenario: dict) -> None:
    """
    **Property 16: GPU-CPU Numerical Parity**
    **Validates: Requirements 6.1**

    When no rows match (chromosome uses dont_care=2 for binary, but we use
    a non-matching gene=0 for feat_binary=1 rows), both engines must return
    zero metrics consistently.

    This verifies the zero-trade edge case is handled identically.
    """
    n = scenario["n_rows"]
    tp = scenario["tp"]
    sl = scenario["sl"]
    capital_pct = scenario["capital_pct"]
    direction = scenario["direction"]

    # All rows have feat_binary=1; use gene=0 (Inactive) → no match
    df = _make_parity_df(
        n=n,
        label_max_list=scenario["label_max_list"],
        label_min_list=scenario["label_min_list"],
        label_close_list=scenario["label_close_list"],
        mbm_list=scenario["mbm_list"],
        entry=scenario["entry"],
    )

    gpu_eng, cpu_eng = _make_engines(df, direction, max_hold_candles=3)

    # CPU: condition that never matches (feat_binary=1 but condition checks Inactive=0)
    cpu_rule_set = [
        {
            "conditions": ["[feat_binary] IS Inactive (0)"],
            "tp": tp,
            "sl": sl,
            "capital_pct": capital_pct,
        }
    ]
    cpu_result = cpu_eng.simulate_rule_set(cpu_rule_set)

    # GPU: chromosome gene=0 (Inactive) → no match since all feat_binary=1
    chrom = np.array([[0]], dtype=np.int32)
    gpu_results = gpu_eng.simulate_rule_batch(
        chrom, tp=tp, sl=sl, capital_pct=capital_pct)
    gpu_result = gpu_results[0]

    context = (
        f"zero-trade case: n={n}, tp={tp:.4f}, sl={sl:.4f}, "
        f"capital_pct={capital_pct:.2f}, direction={direction}"
    )

    # Both must report zero trades and zero metrics
    assert gpu_result["executed_trades"] == 0, (
        f"GPU should have 0 executed_trades for non-matching chromosome. {context}"
    )
    assert cpu_result["executed_trades"] == 0, (
        f"CPU should have 0 executed_trades for non-matching condition. {context}"
    )

    assert gpu_result["total_return_pct"] == pytest.approx(0.0, abs=1e-9), (
        f"GPU total_return_pct should be 0 with no trades. {context}"
    )
    assert cpu_result["total_return_pct"] == pytest.approx(0.0, abs=1e-9), (
        f"CPU total_return_pct should be 0 with no trades. {context}"
    )

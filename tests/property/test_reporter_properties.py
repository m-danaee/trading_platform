"""
Property-based tests for gpu_fuzzy_trader.reporting.reporter.Reporter

This file contains Hypothesis composite strategies used by Properties 1–9
(tasks 8.2–8.10). No property tests are defined here yet — only the
strategies that generate valid inputs matching the data models in the design.

Data models
-----------
rule_set
    list of dicts: {"conditions": list[str], "tp": float, "sl": float,
                    "capital_pct": float}

trade_log
    pd.DataFrame with columns:
        Rule_Index          int   (1-based)
        Net_PnL             float
        Equity_After        float
        Equity_Before_Entry float (non-zero)
        Entry_Index         int   (>= 0)
        Release_Index       int   (> Entry_Index)

metrics
    dict with keys: win_rate, max_drawdown_pct, total_return_pct,
                    sortino_ratio, profit_factor  (all floats)

metrics_by_split
    dict with keys "train", "validation", "test" → metrics dict or None

trade_logs_by_split
    dict with keys "train", "validation", "test" → pd.DataFrame or None

dataset_with_features
    pd.DataFrame with feature columns (string fuzzy values) and
    label_close_288 (float)

datasets_by_split
    dict with keys "train", "validation", "test" → pd.DataFrame or None

selected_features
    list of dicts: {"name": str, "mode": str, "score": float}
"""

from __future__ import annotations
import os
from tests.property.hypothesis_config import prop_settings
from hypothesis import given, HealthCheck

import pandas as pd
import numpy as np
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FUZZY_VALUES = ["Very Low", "Low", "Medium", "High", "Very High"]
_FEATURE_MODES = ["positive", "binary", "signed"]
_SPLITS = ("train", "validation", "test")


# ---------------------------------------------------------------------------
# Strategy: rule_set_strategy
# ---------------------------------------------------------------------------

@st.composite
def rule_set_strategy(draw: st.DrawFn) -> list[dict]:
    """
    Generate a list of 1–5 rule dicts.

    Each rule has:
        conditions  : list of 1–4 non-empty condition strings
        tp          : float in (0.1, 20.0]
        sl          : float in (0.1, 20.0]
        capital_pct : float in (1.0, 100.0]
    """
    n_rules = draw(st.integers(min_value=1, max_value=5))

    rules = []
    for _ in range(n_rules):
        n_conditions = draw(st.integers(min_value=1, max_value=4))
        conditions = draw(
            st.lists(
                st.text(
                    alphabet=st.characters(
                        whitelist_categories=("Lu", "Ll", "Nd"),
                        whitelist_characters=" _[]",
                    ),
                    min_size=1,
                    max_size=40,
                ),
                min_size=n_conditions,
                max_size=n_conditions,
            )
        )
        tp = draw(
            st.floats(
                min_value=0.1,
                max_value=20.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        sl = draw(
            st.floats(
                min_value=0.1,
                max_value=20.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        capital_pct = draw(
            st.floats(
                min_value=1.0,
                max_value=100.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        rules.append(
            {
                "conditions": conditions,
                "tp": tp,
                "sl": sl,
                "capital_pct": capital_pct,
            }
        )

    return rules


# ---------------------------------------------------------------------------
# Strategy: trade_log_strategy
# ---------------------------------------------------------------------------

@st.composite
def trade_log_strategy(
    draw: st.DrawFn,
    n_rules: int | None = None,
    dataset_len: int | None = None,
) -> pd.DataFrame:
    """
    Generate a trade log DataFrame with 2–50 rows.

    Parameters
    ----------
    n_rules:
        If provided, Rule_Index values are drawn from [1, n_rules].
        Otherwise Rule_Index values are drawn from [1, 5].
    dataset_len:
        If provided, Entry_Index values are constrained to [0, dataset_len-1].
        Otherwise Entry_Index values are drawn from [0, 999].

    Columns produced
    ----------------
    Rule_Index          int   (1-based)
    Net_PnL             float (finite, may be negative)
    Equity_After        float (positive, monotonically plausible)
    Equity_Before_Entry float (positive, non-zero)
    Entry_Index         int   (>= 0)
    Release_Index       int   (> Entry_Index)
    """
    n_trades = draw(st.integers(min_value=2, max_value=50))

    max_rule_idx = n_rules if n_rules is not None else 5
    max_entry = (dataset_len - 1) if dataset_len is not None else 999

    # Ensure there is room for Release_Index > Entry_Index
    # Entry_Index must be at most max_entry - 1 so Release_Index can be +1
    effective_max_entry = max(0, max_entry - 1)

    rows = []
    equity = draw(
        st.floats(
            min_value=100.0,
            max_value=10_000.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )

    for _ in range(n_trades):
        rule_idx = draw(st.integers(min_value=1, max_value=max_rule_idx))

        equity_before = equity
        net_pnl = draw(
            st.floats(
                min_value=-equity_before * 0.5,
                max_value=equity_before * 0.5,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        equity_after = equity_before + net_pnl
        # Keep equity positive to avoid degenerate states
        equity_after = max(equity_after, 1.0)
        equity = equity_after

        entry_idx = draw(st.integers(min_value=0, max_value=effective_max_entry))
        # Release_Index must be strictly greater than Entry_Index
        release_idx = draw(
            st.integers(
                min_value=entry_idx + 1,
                max_value=entry_idx + 50,
            )
        )

        rows.append(
            {
                "Rule_Index": rule_idx,
                "Net_PnL": net_pnl,
                "Equity_After": equity_after,
                "Equity_Before_Entry": equity_before,
                "Entry_Index": entry_idx,
                "Release_Index": release_idx,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Rule_Index",
            "Net_PnL",
            "Equity_After",
            "Equity_Before_Entry",
            "Entry_Index",
            "Release_Index",
        ],
    )


# ---------------------------------------------------------------------------
# Strategy: metrics_strategy
# ---------------------------------------------------------------------------

@st.composite
def metrics_strategy(draw: st.DrawFn) -> dict:
    """
    Generate a metrics dict with reasonable float values.

    Keys: win_rate, max_drawdown_pct, total_return_pct,
          sortino_ratio, profit_factor
    """
    win_rate = draw(
        st.floats(
            min_value=0.0,
            max_value=1.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    max_drawdown_pct = draw(
        st.floats(
            min_value=0.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    total_return_pct = draw(
        st.floats(
            min_value=-100.0,
            max_value=500.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    sortino_ratio = draw(
        st.floats(
            min_value=-10.0,
            max_value=10.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    profit_factor = draw(
        st.floats(
            min_value=0.0,
            max_value=20.0,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    return {
        "win_rate": win_rate,
        "max_drawdown_pct": max_drawdown_pct,
        "total_return_pct": total_return_pct,
        "sortino_ratio": sortino_ratio,
        "profit_factor": profit_factor,
    }


# ---------------------------------------------------------------------------
# Strategy: dataset_with_features_strategy
# ---------------------------------------------------------------------------

@st.composite
def dataset_with_features_strategy(
    draw: st.DrawFn,
    feature_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Generate a dataset DataFrame with 10–100 rows.

    Parameters
    ----------
    feature_names:
        If provided, use these column names for feature columns.
        Otherwise generate 1–5 feature columns with auto-generated names.

    Columns produced
    ----------------
    <feature_name>  str   — one of the _FUZZY_VALUES strings
    label_close_288 float — finite float representing forward return
    """
    n_rows = draw(st.integers(min_value=10, max_value=100))

    if feature_names is None:
        n_features = draw(st.integers(min_value=1, max_value=5))
        feature_names = [f"feature_{chr(ord('a') + i)}" for i in range(n_features)]

    data: dict[str, list] = {}

    for feat_name in feature_names:
        data[feat_name] = draw(
            st.lists(
                st.sampled_from(_FUZZY_VALUES),
                min_size=n_rows,
                max_size=n_rows,
            )
        )

    data["label_close_288"] = draw(
        st.lists(
            st.floats(
                min_value=-50.0,
                max_value=50.0,
                allow_nan=False,
                allow_infinity=False,
            ),
            min_size=n_rows,
            max_size=n_rows,
        )
    )

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Strategy: split_logs_strategy
# ---------------------------------------------------------------------------

@st.composite
def split_logs_strategy(
    draw: st.DrawFn,
    n_rules: int | None = None,
) -> dict[str, pd.DataFrame | None]:
    """
    Generate a trade_logs_by_split dict.

    Keys: "train", "validation", "test"
    Values: pd.DataFrame (from trade_log_strategy) or None.

    At least one split will have a non-None trade log to keep tests
    meaningful. Individual splits may be None to exercise graceful handling.
    """
    # Decide which splits are present (at least one must be non-None)
    split_present = draw(
        st.lists(
            st.booleans(),
            min_size=3,
            max_size=3,
        )
    )
    # Guarantee at least one non-None split
    if not any(split_present):
        idx = draw(st.integers(min_value=0, max_value=2))
        split_present[idx] = True

    result: dict[str, pd.DataFrame | None] = {}
    for split, present in zip(_SPLITS, split_present):
        if present:
            result[split] = draw(trade_log_strategy(n_rules=n_rules))
        else:
            result[split] = None

    return result


# ---------------------------------------------------------------------------
# Strategy: stratification_scenario_strategy
# ---------------------------------------------------------------------------

@st.composite
def stratification_scenario_strategy(
    draw: st.DrawFn,
) -> tuple[
    dict[str, pd.DataFrame | None],
    list[dict],
    dict[str, pd.DataFrame | None],
]:
    """
    Generate a (trade_logs_by_split, selected_features, datasets_by_split)
    tuple where Entry_Index values in each trade log are within the bounds
    of the corresponding dataset.

    Returns
    -------
    trade_logs_by_split
        dict with keys "train", "validation", "test" → pd.DataFrame or None
    selected_features
        list of dicts with keys "name", "mode", "score"
    datasets_by_split
        dict with keys "train", "validation", "test" → pd.DataFrame or None

    Invariant
    ---------
    For every non-None split, all Entry_Index values in the trade log are
    within [0, len(dataset) - 1].
    """
    # Generate 1–4 feature names shared across all splits
    n_features = draw(st.integers(min_value=1, max_value=4))
    feature_names = [f"feat_{chr(ord('a') + i)}" for i in range(n_features)]

    # Build selected_features list
    selected_features = []
    for feat_name in feature_names:
        mode = draw(st.sampled_from(_FEATURE_MODES))
        score = draw(
            st.floats(
                min_value=0.0,
                max_value=1.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        selected_features.append({"name": feat_name, "mode": mode, "score": score})

    # Decide which splits are present (at least one must be non-None)
    split_present = draw(
        st.lists(
            st.booleans(),
            min_size=3,
            max_size=3,
        )
    )
    if not any(split_present):
        idx = draw(st.integers(min_value=0, max_value=2))
        split_present[idx] = True

    trade_logs_by_split: dict[str, pd.DataFrame | None] = {}
    datasets_by_split: dict[str, pd.DataFrame | None] = {}

    for split, present in zip(_SPLITS, split_present):
        if not present:
            trade_logs_by_split[split] = None
            datasets_by_split[split] = None
            continue

        # Generate dataset first so we know its length
        dataset = draw(dataset_with_features_strategy(feature_names=feature_names))
        dataset_len = len(dataset)
        datasets_by_split[split] = dataset

        # Generate trade log with Entry_Index constrained to dataset bounds
        trade_log = draw(
            trade_log_strategy(
                n_rules=n_features,
                dataset_len=dataset_len,
            )
        )
        trade_logs_by_split[split] = trade_log

    return trade_logs_by_split, selected_features, datasets_by_split


# ---------------------------------------------------------------------------
# Imports for property tests
# ---------------------------------------------------------------------------

import pytest

from gpu_fuzzy_trader.reporting.reporter import Reporter


# ---------------------------------------------------------------------------
# Property 1: File creation round-trip (single-return methods)
# ---------------------------------------------------------------------------

# Feature: enhanced-reporting-outputs, Property 1: file creation round-trip
# Validates: Requirements 1.1, 1.8, 2.1, 3.1, 3.8, 6.6, 7.1
@given(
    rule_set=rule_set_strategy(),
    logs=split_logs_strategy(),
    metrics=st.fixed_dictionaries({
        "train": st.one_of(st.none(), metrics_strategy()),
        "validation": st.one_of(st.none(), metrics_strategy()),
        "test": st.one_of(st.none(), metrics_strategy()),
    }),
    datasets=st.fixed_dictionaries({
        "train": st.one_of(st.none(), dataset_with_features_strategy()),
        "validation": st.one_of(st.none(), dataset_with_features_strategy()),
        "test": st.one_of(st.none(), dataset_with_features_strategy()),
    }),
    features=st.lists(
        st.fixed_dictionaries({"name": st.just("feature_a"), "mode": st.just("positive"), "score": st.floats(0, 1, allow_nan=False)}),
        min_size=1, max_size=3
    ),
)
@prop_settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.function_scoped_fixture])
def test_property_1_file_creation_round_trip(rule_set, logs, metrics, datasets, features, tmp_path):
    reporter = Reporter()
    # Test plot_per_rule_breakdown
    path = reporter.plot_per_rule_breakdown(rule_set, logs, "long", output_dir=str(tmp_path))
    assert path and os.path.exists(path) and os.path.isabs(path)
    # Test write_strategy_evaluation_table
    path = reporter.write_strategy_evaluation_table(metrics, logs, rule_set, "long", output_dir=str(tmp_path))
    assert path and os.path.exists(path) and os.path.isabs(path)
    # Test write_spearman_correlation_report
    path = reporter.write_spearman_correlation_report(datasets, features, "long", output_dir=str(tmp_path))
    assert path and os.path.exists(path) and os.path.isabs(path)


# ---------------------------------------------------------------------------
# Property 2: Invalid direction raises ValueError
# ---------------------------------------------------------------------------

# Feature: enhanced-reporting-outputs, Property 2: invalid direction raises ValueError
# Validates: Requirements 1.9, 2.9
@given(direction=st.text().filter(lambda s: s not in ("long", "short")))
@prop_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.function_scoped_fixture])
def test_property_2_invalid_direction_raises(direction, tmp_path):
    reporter = Reporter()
    with pytest.raises(ValueError):
        reporter.plot_per_rule_breakdown([], {}, direction, output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        reporter.write_strategy_evaluation_table({}, {}, [], direction, output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        reporter.write_spearman_correlation_report({}, [], direction, output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        reporter.plot_distribution_and_equity({}, direction, output_dir=str(tmp_path))
    with pytest.raises(ValueError):
        reporter.write_feature_stratified_performance({}, [], [], {}, direction, output_dir=str(tmp_path))
    # No files should have been created
    assert not any(tmp_path.iterdir())


# ---------------------------------------------------------------------------
# Imports needed for property tests
# ---------------------------------------------------------------------------

import os
import tempfile
import shutil
from pathlib import Path

from gpu_fuzzy_trader.reporting.reporter import Reporter


# ---------------------------------------------------------------------------
# Property 3: output_dir override is respected
# ---------------------------------------------------------------------------

# Feature: enhanced-reporting-outputs, Property 3: output_dir override is respected
@given(
    rule_set=rule_set_strategy(),
    logs=split_logs_strategy(),
    metrics=st.fixed_dictionaries({
        "train": st.one_of(st.none(), metrics_strategy()),
        "validation": st.one_of(st.none(), metrics_strategy()),
        "test": st.one_of(st.none(), metrics_strategy()),
    }),
    datasets=st.fixed_dictionaries({
        "train": st.one_of(st.none(), dataset_with_features_strategy()),
        "validation": st.one_of(st.none(), dataset_with_features_strategy()),
        "test": st.one_of(st.none(), dataset_with_features_strategy()),
    }),
)
@prop_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example], deadline=None)
def test_property_3_output_dir_respected(rule_set, logs, metrics, datasets):
    """
    **Validates: Requirements 6.4, 6.5, 6.6**

    For any valid inputs and any output_dir, all returned paths SHALL start
    with the provided output_dir.
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        reporter = Reporter()
        output_dir = tmp_dir

        path = reporter.plot_per_rule_breakdown(rule_set, logs, "long", output_dir=output_dir)
        assert path.startswith(output_dir)

        path = reporter.write_strategy_evaluation_table(metrics, logs, rule_set, "long", output_dir=output_dir)
        assert path.startswith(output_dir)

        path = reporter.write_spearman_correlation_report(datasets, [], "long", output_dir=output_dir)
        assert path.startswith(output_dir)

        paths = reporter.plot_distribution_and_equity(logs, "long", output_dir=output_dir)
        for p in paths:
            assert p.startswith(output_dir)

        paths = reporter.write_feature_stratified_performance(logs, [], [], datasets, "long", output_dir=output_dir)
        for p in paths:
            assert p.startswith(output_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Property 4: Strategy evaluation table schema and rule_set counts
# ---------------------------------------------------------------------------

# Feature: enhanced-reporting-outputs, Property 4: strategy evaluation table schema
@given(rule_set=rule_set_strategy())
@prop_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example])
def test_property_4_evaluation_table_schema(rule_set):
    """
    **Validates: Requirements 2.2, 2.3, 2.4**

    For any rule_set of length N with total condition count C:
    - CSV has exactly 3 rows (one per split)
    - CSV has exactly the required columns
    - num_rules == N in every row
    - num_conditions == C in every row
    """
    tmp_dir = tempfile.mkdtemp()
    try:
        reporter = Reporter()
        reporter.write_strategy_evaluation_table(
            {"train": None, "validation": None, "test": None},
            {"train": None, "validation": None, "test": None},
            rule_set,
            "long",
            output_dir=tmp_dir,
        )
        df = pd.read_csv(Path(tmp_dir) / "strategy_evaluation_long.csv")

        assert len(df) == 3
        expected_cols = {"split", "win_rate", "mdd_pct", "total_return_pct",
                         "num_rules", "num_conditions", "sortino_ratio",
                         "profit_factor", "sharpe_ratio"}
        assert set(df.columns) == expected_cols

        N = len(rule_set)
        C = sum(len(r.get("conditions", [])) for r in rule_set)
        assert (df["num_rules"] == N).all()
        assert (df["num_conditions"] == C).all()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Property 5: Sharpe ratio computation correctness
# ---------------------------------------------------------------------------

# Feature: enhanced-reporting-outputs, Property 5: Sharpe ratio computation correctness
# Validates: Requirements 2.5, 5.6
@given(trade_log=trade_log_strategy())
@prop_settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.function_scoped_fixture])
def test_property_5_sharpe_ratio_correctness(trade_log, tmp_path):
    reporter = Reporter()
    logs = {"train": trade_log, "validation": None, "test": None}
    reporter.write_strategy_evaluation_table(
        {"train": None, "validation": None, "test": None},
        logs,
        [],
        "long",
        output_dir=str(tmp_path),
    )
    df = pd.read_csv(tmp_path / "strategy_evaluation_long.csv")
    train_row = df[df["split"] == "train"].iloc[0]
    actual_sharpe = train_row["sharpe_ratio"]

    n = len(trade_log)
    if n >= 2:
        r = trade_log["Net_PnL"] / trade_log["Equity_Before_Entry"]
        std_r = r.std(ddof=1)
        if std_r != 0 and not pd.isna(std_r):
            expected_sharpe = float(r.mean() / std_r)
            assert abs(actual_sharpe - expected_sharpe) < 1e-9
        else:
            assert actual_sharpe == 0.0
    else:
        assert actual_sharpe == 0.0


# ---------------------------------------------------------------------------
# Property 6: Spearman correlation correctness and range invariant
# ---------------------------------------------------------------------------

# Feature: enhanced-reporting-outputs, Property 6: Spearman correlation correctness and range invariant
# Validates: Requirements 3.2, 3.9
@given(dataset=dataset_with_features_strategy())
@prop_settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.function_scoped_fixture])
def test_property_6_spearman_correctness(dataset, tmp_path):
    from scipy.stats import spearmanr as scipy_spearmanr
    reporter = Reporter()

    # Use the first feature column
    feature_name = [c for c in dataset.columns if c != "label_close_288"][0]
    features = [{"name": feature_name, "mode": "positive", "score": 1.0}]
    datasets = {"train": dataset, "validation": None, "test": None}

    path = reporter.write_spearman_correlation_report(
        datasets, features, "long", output_dir=str(tmp_path)
    )
    result_df = pd.read_csv(path)
    row = result_df[result_df["feature"] == feature_name].iloc[0]
    train_val = row["train_spearman"]

    # Compute expected value directly
    mask = dataset[feature_name].notna() & dataset["label_close_288"].notna()
    n_valid = mask.sum()

    if n_valid >= 2:
        expected = scipy_spearmanr(
            dataset[feature_name][mask].values,
            dataset["label_close_288"][mask].values,
        )
        expected_stat = getattr(expected, "statistic", None) or getattr(expected, "correlation", float("nan"))
        expected_float = float(expected_stat)
        if pd.isna(expected_float):
            # Constant input (e.g. all same fuzzy value) → spearmanr returns NaN;
            # the reporter also returns NaN in this case — both are consistent.
            assert pd.isna(train_val)
        else:
            assert not pd.isna(train_val)
            assert abs(train_val - expected_float) < 1e-9
            assert -1.0 <= train_val <= 1.0
    else:
        assert pd.isna(train_val)


# ---------------------------------------------------------------------------
# Property 7: Spearman output sorted by absolute train correlation
# ---------------------------------------------------------------------------

# Feature: enhanced-reporting-outputs, Property 7: Spearman output sorted by absolute train correlation
# Validates: Requirements 3.4
@given(dataset=dataset_with_features_strategy())
@prop_settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.function_scoped_fixture])
def test_property_7_spearman_sorted(dataset, tmp_path):
    reporter = Reporter()
    feature_names = [c for c in dataset.columns if c != "label_close_288"]
    features = [{"name": n, "mode": "positive", "score": 1.0} for n in feature_names]
    datasets = {"train": dataset, "validation": None, "test": None}

    path = reporter.write_spearman_correlation_report(
        datasets, features, "long", output_dir=str(tmp_path)
    )
    result_df = pd.read_csv(path)

    abs_train = result_df["train_spearman"].abs()
    # Non-NaN values should be non-increasing
    valid_abs = abs_train.dropna()
    for i in range(len(valid_abs) - 1):
        assert valid_abs.iloc[i] >= valid_abs.iloc[i + 1] - 1e-12


# ---------------------------------------------------------------------------
# Property 8: Distribution and equity skips empty splits
# ---------------------------------------------------------------------------

# Feature: enhanced-reporting-outputs, Property 8: Distribution and equity skips empty splits
# Validates: Requirements 4.1, 4.7, 4.8
@given(logs=split_logs_strategy())
@prop_settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.function_scoped_fixture])
def test_property_8_distribution_equity_skips_empty(logs, tmp_path):
    reporter = Reporter()
    result = reporter.plot_distribution_and_equity(logs, "long", output_dir=str(tmp_path))

    # Count non-empty splits
    k = sum(
        1 for split in ("train", "validation", "test")
        if logs.get(split) is not None and not (
            isinstance(logs.get(split), pd.DataFrame) and logs.get(split).empty
        )
    )

    assert len(result) == k
    for path in result:
        assert os.path.exists(path)
        assert os.path.isabs(path)


# ---------------------------------------------------------------------------
# Property 9: Feature stratification metric correctness
# ---------------------------------------------------------------------------

# Feature: enhanced-reporting-outputs, Property 9: Feature stratification metric correctness
# Validates: Requirements 5.3, 5.4, 5.5
@given(scenario=stratification_scenario_strategy())
@prop_settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow, HealthCheck.large_base_example, HealthCheck.function_scoped_fixture])
def test_property_9_stratification_metric_correctness(scenario, tmp_path):
    from gpu_fuzzy_trader import config as _cfg
    trade_logs_by_split, selected_features, datasets_by_split = scenario
    reporter = Reporter()

    result = reporter.write_feature_stratified_performance(
        trade_logs_by_split, [], selected_features, datasets_by_split,
        "long", output_dir=str(tmp_path)
    )

    assert len(result) == 3

    for path in result:
        df = pd.read_csv(path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            n = int(row["num_trades"])
            if n == 0:
                assert row["total_return_pct"] == 0.0
                assert row["win_rate"] == 0.0
            else:
                # total_return_pct and win_rate should be finite
                assert not pd.isna(row["total_return_pct"])
                assert not pd.isna(row["win_rate"])
                assert 0.0 <= row["win_rate"] <= 1.0

"""RB Governor fail-closed and stale-output regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader.output.writer import Output_Writer, ValidationError, _validate_rule_set
from gpu_fuzzy_trader.rb_governor import _strategy, run_rb_governor_pipeline


def _dummy_df(size: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    entry = rng.uniform(1.0, 2.0, size=size)
    return pd.DataFrame(
        {
            "datetime": pd.date_range("2024-01-01", periods=size, freq="5min"),
            "symbol": rng.choice(["sym_a", "sym_b"], size=size),
            "label_open_next": entry,
            "label_close_288": entry * 1.01,
            "label_min_288": entry * 0.99,
            "label_max_288": entry * 1.02,
            "label_max_before_min": np.zeros(size),
            "feat_1": rng.normal(size=size),
        }
    )


def _pool_rules() -> list[dict]:
    return [
        {
            "conditions": ["[feat_1] IS High"],
            "tp": 2.0,
            "sl": 1.0,
            "capital_pct": 5.0,
        }
    ]


def test_strategy_can_represent_an_explicit_rejected_direction() -> None:
    strategy = _strategy(
        "long",
        [],
        extra={"deployment_accepted": False, "fail_closed": True, "reason": "phase2_error"},
    )
    assert strategy["rules_set"] == []
    assert strategy["deployment_accepted"] is False
    assert strategy["fail_closed"] is True
    assert strategy["reason"] == "phase2_error"


def test_empty_phase2_pool_writes_fail_closed_output_with_reason(tmp_path: Path) -> None:
    result = run_rb_governor_pipeline(
        _dummy_df(),
        _dummy_df(),
        {"short": []},
        ("short",),
        output_dir=tmp_path,
        failure_reasons={"short": "phase2_error"},
    )

    strategy = result["short"]
    assert strategy["rules_set"] == []
    assert strategy["deployment_accepted"] is False
    assert strategy["reason"] == "phase2_error"

    report = json.loads(
        (tmp_path / "reports" / "rb_governor_short_report.json").read_text()
    )
    assert report["fail_closed"] is True
    assert report["reason"] == "phase2_error"
    assert report["phase2_status"]["reason"] == "phase2_error"


def test_no_positive_good_candidates_fail_closed_and_do_not_call_fallback(tmp_path: Path) -> None:
    with patch("gpu_fuzzy_trader.rb_governor._filter_good_rules", return_value=[]), patch(
        "gpu_fuzzy_trader.rb_governor._symbol_specialized_variants"
    ) as fallback:
        result = run_rb_governor_pipeline(
            _dummy_df(),
            _dummy_df(),
            {"long": _pool_rules()},
            ("long",),
            output_dir=tmp_path,
        )

    fallback.assert_not_called()
    assert result["long"]["rules_set"] == []
    assert result["long"]["reason"] == "no_positive_good_candidates"

    report = json.loads(
        (tmp_path / "reports" / "rb_governor_long_report.json").read_text()
    )
    assert report["selected_rules"] == 0
    assert report["fail_closed"] is True


def test_fail_closed_output_overwrites_stale_strategy(tmp_path: Path) -> None:
    stale = {
        "direction": "short",
        "rules_set": [
            {"conditions": ["[feat_1] IS High"], "tp": 2.0, "sl": 1.0, "capital_pct": 5.0}
        ],
        "deployment_accepted": True,
    }
    (tmp_path / "short.json").write_text(json.dumps(stale))

    run_rb_governor_pipeline(
        _dummy_df(),
        _dummy_df(),
        {"short": []},
        ("short",),
        output_dir=tmp_path,
        failure_reasons={"short": "missing_phase2_output"},
    )

    current = json.loads((tmp_path / "short.json").read_text())
    assert current["rules_set"] == []
    assert current["deployment_accepted"] is False
    assert current["reason"] == "missing_phase2_output"


def test_output_writer_accepts_only_explicit_empty_fail_closed_strategy() -> None:
    data = {
        "direction": "long",
        "rules_set": [],
        "deployment_accepted": False,
        "fail_closed": True,
        "reason": "phase2_error",
    }
    assert _validate_rule_set(data)["rules_set"] == []

    with pytest.raises(ValidationError, match="at least"):
        _validate_rule_set({"direction": "long", "rules_set": [], "deployment_accepted": True})


def test_written_fail_closed_strategy_is_evaluator_loadable(tmp_path: Path) -> None:
    run_rb_governor_pipeline(
        _dummy_df(),
        _dummy_df(),
        {"long": []},
        ("long",),
        output_dir=tmp_path,
        failure_reasons={"long": "empty_phase2_pool"},
    )
    loaded = Output_Writer().load_and_validate(tmp_path / "long.json")
    assert loaded["direction"] == "long"
    assert loaded["rules_set"] == []

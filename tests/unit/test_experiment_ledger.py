import json

from gpu_fuzzy_trader.validation.multiplicity import (
    read_ledger_trial_count,
    trial_count_from_counters,
)


def test_trial_count_from_ledger_counters_is_exact() -> None:
    counters = {
        "feature_alternatives": 4,
        "rules_tested": 10,
        "hyperparameter_configs": 2,
        "selection_candidates": 6,
    }

    assert trial_count_from_counters(counters) == 22


def test_read_ledger_trial_count_selects_requested_run(tmp_path) -> None:
    ledger = tmp_path / "reports" / "experiment_ledger.jsonl"
    ledger.parent.mkdir()
    ledger.write_text(
        "\n".join([
            json.dumps({
                "run_id": "old",
                "trial_counters": {"rules_tested": 2},
            }),
            json.dumps({
                "run_id": "current",
                "trial_counters": {
                    "feature_alternatives": 3,
                    "rules_tested": 5,
                },
            }),
        ])
        + "\n",
        encoding="utf-8",
    )

    assert read_ledger_trial_count(tmp_path, run_id="current") == 8

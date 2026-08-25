import json

import numpy as np

from gpu_fuzzy_trader.validation.multiplicity import (
    aggregate_seed_metrics,
    estimate_pbo,
    summarize_multiplicity,
    write_candidate_fold_matrix,
)


def test_estimate_pbo_accepts_candidate_fold_matrix() -> None:
    is_scores = np.tile(np.arange(10, dtype=float), (5, 1))
    oos_scores = is_scores + 1.0

    assert estimate_pbo(matrix={
        "is_scores": is_scores,
        "oos_scores": oos_scores,
    }) == 0.0


def test_matrix_writer_has_deterministic_candidate_fold_order(tmp_path) -> None:
    rows = [
        {"candidate_id": "candidate-2", "fold_id": 1, "is_score": 2, "oos_score": 3},
        {"candidate_id": "candidate-1", "fold_id": 2, "is_score": 1, "oos_score": 4},
        {"candidate_id": "candidate-1", "fold_id": 1, "is_score": 5, "oos_score": 6},
    ]

    path = write_candidate_fold_matrix(tmp_path, rows)

    payload = [json.loads(line) for line in path.read_text().splitlines()]
    assert [(row["candidate_id"], row["fold_id"]) for row in payload] == [
        ("candidate-1", 1),
        ("candidate-1", 2),
        ("candidate-2", 1),
    ]


def test_summary_uses_ledger_trial_count_and_matrix_pbo() -> None:
    rows = [
        {"candidate_id": "a", "fold_id": 0, "is_score": 2, "oos_score": 2},
        {"candidate_id": "b", "fold_id": 0, "is_score": 1, "oos_score": 1},
        {"candidate_id": "a", "fold_id": 1, "is_score": 2, "oos_score": 2},
        {"candidate_id": "b", "fold_id": 1, "is_score": 1, "oos_score": 1},
    ]

    summary = summarize_multiplicity(
        fold_returns=[1.0, 1.2],
        n_trials=999,
        matrix=rows,
        trial_count_ledger=7,
    )

    assert summary["candidates_tested"] == 2
    assert summary["trial_count_ledger"] == 7
    assert summary["dsr_probability"] == summary["deflated_sharpe_probability"]
    assert summary["pbo"] == 0.0
    assert "pbo_note" not in summary


def test_dsr_probability_decreases_as_trial_count_increases() -> None:
    low = summarize_multiplicity(
        fold_returns=[-1.0, 0.0, 1.0],
        n_trials=2,
    )["dsr_probability"]
    high = summarize_multiplicity(
        fold_returns=[-1.0, 0.0, 1.0],
        n_trials=20,
    )["dsr_probability"]

    assert low > high


def test_aggregate_seed_metrics_reports_mean_std_median_iqr() -> None:
    aggregate = aggregate_seed_metrics([
        {"seed": 1, "Return Long": 1.0},
        {"seed": 2, "Return Long": 3.0},
    ])

    assert aggregate["Return Long"]["mean"] == 2.0
    assert aggregate["Return Long"]["median"] == 2.0
    assert aggregate["Return Long"]["iqr"] == 1.0

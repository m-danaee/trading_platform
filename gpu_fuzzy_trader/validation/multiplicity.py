"""Selection-multiplicity diagnostics for strategy research artifacts.

The functions in this module are deliberately independent from the pipeline.
They accept both the historical ``fold -> candidate`` score lists and the
append-only candidate/fold ledger used by current runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


_IS_KEYS = ("is_score", "in_sample_score", "in_sample", "is", "IS")
_OOS_KEYS = ("oos_score", "out_of_sample_score", "out_of_sample", "oos", "OOS")
_CANDIDATE_KEYS = ("candidate_id", "candidate", "candidate_key", "id")
_FOLD_KEYS = ("fold_id", "fold", "fold_key")
_TRIAL_COUNTER_KEYS = {
    "feature_alternatives",
    "rules_tested",
    "hyperparameter_configs",
    "selection_candidates",
}


def _finite_float(value: Any) -> float | None:
    """Return a finite float, or ``None`` for an unusable score."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _first_value(mapping: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return None


def _metadata_length(value: Any) -> int | None:
    if value is None or isinstance(value, (str, bytes)):
        return None
    try:
        return len(value)
    except TypeError:
        return None


def _fold_sort_key(value: Any) -> tuple[int, int | str]:
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def _normalise_record(record: Mapping[str, Any], index: int = 0) -> dict[str, Any]:
    """Convert one ledger row to the stable public row schema."""
    candidate = _first_value(record, _CANDIDATE_KEYS)
    fold = _first_value(record, _FOLD_KEYS)
    if candidate is None:
        candidate = f"candidate-{index:06d}"
    if fold is None:
        fold = index
    try:
        fold = int(fold)
    except (TypeError, ValueError):
        fold = str(fold)

    row: dict[str, Any] = {
        "candidate_id": str(candidate),
        "fold_id": fold,
        "is_score": _finite_float(_first_value(record, _IS_KEYS)),
        "oos_score": _finite_float(_first_value(record, _OOS_KEYS)),
    }
    for key in ("source", "direction", "metric", "run_id"):
        if key in record and record[key] is not None:
            row[key] = str(record[key])
    return row


def _records_from_array(
    matrix: Any,
    *,
    candidate_ids: Sequence[Any] | None = None,
    fold_ids: Sequence[Any] | None = None,
    orientation: str = "fold_candidate",
) -> list[dict[str, Any]]:
    """Convert a numeric ``[..., 2]`` matrix to ledger rows.

    The last axis stores IS and OOS scores.  The first two axes are either
    ``fold, candidate`` or ``candidate, fold`` as specified by ``orientation``.
    """
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 3 or array.shape[-1] < 2:
        raise ValueError("matrix must have shape (folds, candidates, 2)")
    n_first, n_second = int(array.shape[0]), int(array.shape[1])
    orientation_name = _normalise_matrix_orientation(orientation)
    candidate_major = orientation_name == "candidate_fold"
    if candidate_major:
        n_candidates, n_folds = n_first, n_second
    else:
        n_folds, n_candidates = n_first, n_second
    candidates = (
        list(candidate_ids)
        if candidate_ids is not None
        else list(range(n_candidates))
    )
    folds = (
        list(fold_ids)
        if fold_ids is not None
        else list(range(n_folds))
    )
    if len(candidates) != n_candidates or len(folds) != n_folds:
        raise ValueError("matrix identifiers do not match matrix dimensions")

    rows: list[dict[str, Any]] = []
    for fold_index, fold_id in enumerate(folds):
        for candidate_index, candidate_id in enumerate(candidates):
            if candidate_major:
                values = array[candidate_index, fold_index]
            else:
                values = array[fold_index, candidate_index]
            rows.append({
                "candidate_id": str(candidate_id),
                "fold_id": fold_id,
                "is_score": values[0],
                "oos_score": values[1],
            })
    return rows


def _normalise_matrix_orientation(value: str | None) -> str:
    if value is None:
        raise ValueError(
            "raw score arrays require matrix_orientation='fold_candidate' "
            "or 'candidate_fold'"
        )
    normalized = str(value).lower().replace("-", "_")
    if normalized in {"fold_candidate", "fold_major"}:
        return "fold_candidate"
    if normalized in {"candidate_fold", "candidate_major"}:
        return "candidate_fold"
    raise ValueError(
        "matrix_orientation must be 'fold_candidate' or 'candidate_fold'"
    )


def _records_from_matrix(
    matrix: Any,
    *,
    orientation: str | None = None,
) -> list[dict[str, Any]]:
    """Accept supported ledgers and require orientation for raw arrays."""
    if hasattr(matrix, "to_dict") and not isinstance(matrix, Mapping):
        try:
            dataframe_rows = matrix.to_dict("records")
        except (TypeError, ValueError):
            dataframe_rows = None
        if isinstance(dataframe_rows, list):
            return [
                _normalise_record(row, index)
                for index, row in enumerate(dataframe_rows)
                if isinstance(row, Mapping)
            ]
    if isinstance(matrix, Mapping):
        rows = _first_value(
            matrix,
            ("rows", "records", "entries", "candidate_fold_matrix"),
        )
        if isinstance(rows, Iterable) and not isinstance(rows, (str, bytes)):
            return [
                _normalise_record(row, index)
                for index, row in enumerate(rows)
                if isinstance(row, Mapping)
            ]

        is_values = _first_value(
            matrix,
            ("is_scores", "in_sample_scores", "is_matrix", "is", "IS"),
        )
        oos_values = _first_value(
            matrix,
            (
                "oos_scores",
                "out_of_sample_scores",
                "oos_matrix",
                "oos",
                "OOS",
            ),
        )
        if is_values is not None and oos_values is not None:
            is_array = np.asarray(is_values, dtype=float)
            oos_array = np.asarray(oos_values, dtype=float)
            if is_array.shape != oos_array.shape or is_array.ndim != 2:
                raise ValueError("IS and OOS matrices must be matching 2D arrays")
            orientation_value = matrix.get("orientation")
            if orientation_value is None:
                orientation_value = orientation
            if orientation_value is None:
                candidate_values = matrix.get("candidate_ids")
                fold_values = matrix.get("fold_ids")
                candidate_count = _metadata_length(candidate_values)
                fold_count = _metadata_length(fold_values)
                candidate_major = (
                    candidate_count == is_array.shape[0]
                    and fold_count == is_array.shape[1]
                )
                fold_major = (
                    candidate_count == is_array.shape[1]
                    and fold_count == is_array.shape[0]
                )
                if candidate_major and not fold_major:
                    orientation_value = "candidate_fold"
                elif fold_major and not candidate_major:
                    orientation_value = "fold_candidate"
                else:
                    raise ValueError(
                        "paired IS/OOS mappings require explicit "
                        "matrix_orientation unless candidate_ids and "
                        "fold_ids unambiguously identify axes"
                    )
            orientation_name = _normalise_matrix_orientation(orientation_value)
            paired = np.stack((is_array, oos_array), axis=-1)
            return _records_from_array(
                paired,
                candidate_ids=matrix.get("candidate_ids"),
                fold_ids=matrix.get("fold_ids"),
                orientation=orientation_name,
            )

        nested = matrix.get("matrix")
        if nested is not None:
            nested_orientation = matrix.get("orientation") or orientation
            return _records_from_matrix(nested, orientation=nested_orientation)

        # A mapping of candidate id -> fold rows is convenient when loading a
        # hand-written fixture.  It is intentionally parsed after the explicit
        # schema above so ordinary metadata keys are not treated as candidates.
        candidate_rows: list[dict[str, Any]] = []
        for candidate_id, values in matrix.items():
            if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
                continue
            for fold_index, value in enumerate(values):
                if isinstance(value, Mapping):
                    row = dict(value)
                    row.setdefault("candidate_id", candidate_id)
                    row.setdefault("fold_id", fold_index)
                    candidate_rows.append(_normalise_record(row, fold_index))
                elif isinstance(value, Sequence) and len(value) >= 2:
                    candidate_rows.append({
                        "candidate_id": str(candidate_id),
                        "fold_id": fold_index,
                        "is_score": _finite_float(value[0]),
                        "oos_score": _finite_float(value[1]),
                    })
        if candidate_rows:
            return candidate_rows
        raise ValueError("matrix does not contain candidate/fold scores")

    if isinstance(matrix, np.ndarray):
        if matrix.ndim == 2 and matrix.dtype == object:
            rows: list[dict[str, Any]] = []
            for fold_id in range(matrix.shape[0]):
                for candidate_id in range(matrix.shape[1]):
                    value = matrix[fold_id, candidate_id]
                    if isinstance(value, Mapping):
                        row = dict(value)
                        row.setdefault("fold_id", fold_id)
                        row.setdefault("candidate_id", candidate_id)
                        rows.append(_normalise_record(row, len(rows)))
                    elif isinstance(value, Sequence) and len(value) >= 2:
                        rows.append({
                            "candidate_id": str(candidate_id),
                            "fold_id": fold_id,
                            "is_score": _finite_float(value[0]),
                            "oos_score": _finite_float(value[1]),
                        })
            if rows:
                return rows
        return _records_from_array(
            matrix,
            orientation=_normalise_matrix_orientation(orientation),
        )

    if isinstance(matrix, (str, bytes)):
        raise TypeError("matrix must be score data, not text")
    values = list(matrix)
    if not values:
        return []
    paired_raw = (
        len(values) == 2
        and (
            isinstance(matrix, tuple)
            or any(isinstance(value, np.ndarray) for value in values)
        )
        and all(
            isinstance(value, (np.ndarray, list, tuple))
            for value in values
        )
    )
    if paired_raw:
        first = np.asarray(values[0], dtype=float)
        second = np.asarray(values[1], dtype=float)
        if first.shape == second.shape and first.ndim == 2:
            return _records_from_array(
                np.stack((first, second), axis=-1),
                orientation=_normalise_matrix_orientation(orientation),
            )
    try:
        raw_array = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        raw_array = None
    if raw_array is not None and raw_array.ndim == 3:
        return _records_from_array(
            raw_array,
            orientation=_normalise_matrix_orientation(orientation),
        )
    if values and all(isinstance(value, Mapping) for value in values):
        return [
            _normalise_record(value, index)
            for index, value in enumerate(values)
        ]
    if values and all(
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) >= 4
        for value in values
    ):
        return [
            _normalise_record({
                "candidate_id": value[0],
                "fold_id": value[1],
                "is_score": value[2],
                "oos_score": value[3],
            }, index)
            for index, value in enumerate(values)
        ]
    raise ValueError("matrix must be a ledger row sequence or a paired 3D array")


def _records_to_arrays(
    records: Iterable[Mapping[str, Any]],
) -> tuple[list[str], list[Any], np.ndarray, np.ndarray]:
    rows = [_normalise_record(row, index) for index, row in enumerate(records)]
    candidate_ids = sorted(
        {str(row["candidate_id"]) for row in rows}
    )
    fold_ids = sorted(
        {row["fold_id"] for row in rows},
        key=_fold_sort_key,
    )
    candidate_index = {candidate: index for index, candidate in enumerate(candidate_ids)}
    fold_index = {str(fold): index for index, fold in enumerate(fold_ids)}
    is_scores = np.full((len(fold_ids), len(candidate_ids)), np.nan, dtype=float)
    oos_scores = np.full_like(is_scores, np.nan)
    for row in rows:
        row_fold = row["fold_id"]
        row_candidate = str(row["candidate_id"])
        # String conversion keeps int/string fold identifiers stable in the
        # index map while preserving the original fold IDs for writer output.
        target = (fold_index[str(row_fold)], candidate_index[row_candidate])
        is_value = _finite_float(row.get("is_score"))
        oos_value = _finite_float(row.get("oos_score"))
        if is_value is not None:
            is_scores[target] = is_value
        if oos_value is not None:
            oos_scores[target] = oos_value
    return candidate_ids, fold_ids, is_scores, oos_scores


def _pbo_from_arrays(
    in_sample_scores: Sequence[Sequence[float]],
    out_of_sample_scores: Sequence[Sequence[float]],
) -> float | None:
    if len(in_sample_scores) != len(out_of_sample_scores):
        raise ValueError("IS and OOS fold counts must match")
    misses = 0
    used = 0
    for is_row, oos_row in zip(in_sample_scores, out_of_sample_scores, strict=False):
        if len(is_row) != len(oos_row):
            continue
        is_values = np.asarray(is_row, dtype=float)
        oos_values = np.asarray(oos_row, dtype=float)
        valid = np.isfinite(is_values) & np.isfinite(oos_values)
        if int(np.count_nonzero(valid)) < 2:
            continue
        valid_indices = np.flatnonzero(valid)
        winner = int(valid_indices[np.argmax(is_values[valid])])
        median_oos = float(np.median(oos_values[valid]))
        misses += int(oos_values[winner] < median_oos)
        used += 1
    return float(misses / used) if used else None


def _selected_oos_scores(matrix: Any) -> list[float]:
    """Return the OOS score of each fold's IS winner for DSR fallback use."""
    _, _, is_scores, oos_scores = _records_to_arrays(_records_from_matrix(matrix))
    selected: list[float] = []
    for is_row, oos_row in zip(is_scores, oos_scores, strict=False):
        valid = np.isfinite(is_row) & np.isfinite(oos_row)
        if int(np.count_nonzero(valid)) < 2:
            continue
        valid_indices = np.flatnonzero(valid)
        winner = int(valid_indices[np.argmax(is_row[valid])])
        selected.append(float(oos_row[winner]))
    return selected


def estimate_pbo(
    in_sample_scores: Sequence[Sequence[float]] | Any | None = None,
    out_of_sample_scores: Sequence[Sequence[float]] | None = None,
    *,
    matrix: Any | None = None,
    matrix_orientation: str | None = None,
) -> float | None:
    """Estimate CSCV-style probability of backtest overfitting.

    The current ledger form is supplied as ``matrix=...`` or as the sole
    positional argument.  Each fold selects the IS winner and counts a miss
    when that candidate's OOS score is below the fold's OOS median.  The old
    two-argument ``fold -> candidate`` API remains supported.
    """
    if matrix is not None:
        records = _records_from_matrix(matrix, orientation=matrix_orientation)
        _, _, is_scores, oos_scores = _records_to_arrays(records)
        return _pbo_from_arrays(is_scores, oos_scores)
    if out_of_sample_scores is None:
        if in_sample_scores is None:
            return None
        records = _records_from_matrix(
            in_sample_scores,
            orientation=matrix_orientation,
        )
        _, _, is_scores, oos_scores = _records_to_arrays(records)
        return _pbo_from_arrays(is_scores, oos_scores)
    if in_sample_scores is None:
        raise ValueError("IS scores are required when OOS scores are supplied")
    return _pbo_from_arrays(in_sample_scores, out_of_sample_scores)


def candidate_fold_matrix_path(output_dir: str | Path) -> Path:
    """Return the canonical candidate/fold ledger path for an output root."""
    target = Path(output_dir)
    if target.suffix.lower() == ".jsonl":
        return target
    if target.name == "reports":
        return target / "candidate_fold_matrix.jsonl"
    return target / "reports" / "candidate_fold_matrix.jsonl"


def write_candidate_fold_matrix(
    output_dir: str | Path,
    rows: Iterable[Mapping[str, Any]] | Any | None = None,
    *,
    matrix: Any | None = None,
    matrix_orientation: str | None = None,
) -> Path:
    """Write candidate×fold IS/OOS rows in deterministic JSONL order."""
    if rows is None:
        rows = matrix
    if rows is None:
        normalised: list[dict[str, Any]] = []
    elif isinstance(rows, Mapping) or isinstance(rows, np.ndarray):
        normalised = _records_from_matrix(
            rows,
            orientation=matrix_orientation,
        )
    else:
        values = list(rows)
        normalised = (
            _records_from_matrix(values, orientation=matrix_orientation)
            if values
            else []
        )
    normalised = [
        _normalise_record(row, index)
        for index, row in enumerate(normalised)
        if isinstance(row, Mapping)
    ]

    def sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            str(row.get("candidate_id", "")),
            _fold_sort_key(row.get("fold_id")),
            str(row.get("source", "")),
            str(row.get("direction", "")),
        )

    # A candidate/fold pair is a single ledger observation.  The final sort is
    # applied before de-duplication so output does not depend on input ordering.
    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in sorted(normalised, key=sort_key):
        key = (
            str(row.get("candidate_id", "")),
            str(row.get("fold_id", "")),
            str(row.get("source", "")),
            str(row.get("direction", "")),
        )
        unique.setdefault(key, row)
    ordered = sorted(unique.values(), key=sort_key)

    path = candidate_fold_matrix_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in ordered:
            handle.write(
                json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            )
    return path


def read_candidate_fold_matrix(path: str | Path) -> list[dict[str, Any]]:
    """Read and normalize a candidate/fold JSONL ledger."""
    source = candidate_fold_matrix_path(path)
    if not source.exists():
        return []
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(source.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            rows.append(_normalise_record(value, index))
    return sorted(
        rows,
        key=lambda row: (
            str(row["candidate_id"]),
            _fold_sort_key(row["fold_id"]),
            str(row.get("source", "")),
            str(row.get("direction", "")),
        ),
    )


def trial_count_from_counters(counters: Mapping[str, Any] | None) -> int:
    """Return the exact sum of named trial counters in a ledger record."""
    if not isinstance(counters, Mapping):
        return 0
    direct = counters.get("trial_count_ledger")
    if direct is not None:
        try:
            return max(0, int(direct))
        except (TypeError, ValueError):
            pass

    total = 0
    found = False
    for key, value in counters.items():
        if key in _TRIAL_COUNTER_KEYS:
            try:
                total += max(0, int(value))
                found = True
            except (TypeError, ValueError):
                continue
        elif isinstance(value, Mapping):
            nested = trial_count_from_counters(value)
            if nested:
                total += nested
                found = True
    if found:
        return max(0, int(total))
    for key in ("trial_count", "trial_count_estimate"):
        try:
            if key in counters:
                return max(0, int(counters[key]))
        except (TypeError, ValueError):
            continue
    return 0


def read_ledger_trial_count(
    output_dir: str | Path,
    *,
    run_id: str | None = None,
) -> int | None:
    """Read the exact trial count for a run from the experiment ledger."""
    source = Path(output_dir)
    if source.suffix.lower() != ".jsonl":
        source = (
            source / "experiment_ledger.jsonl"
            if source.name == "reports"
            else source / "reports" / "experiment_ledger.jsonl"
        )
    if not source.exists():
        return None
    found: int | None = None
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, Mapping):
            continue
        if run_id is not None and str(record.get("run_id")) != str(run_id):
            continue
        if isinstance(record.get("trial_counters"), Mapping):
            found = trial_count_from_counters(record["trial_counters"])
        elif "trial_count_ledger" in record:
            try:
                found = max(0, int(record["trial_count_ledger"]))
            except (TypeError, ValueError):
                pass
        elif "trial_count_estimate" in record:
            try:
                found = max(0, int(record["trial_count_estimate"]))
            except (TypeError, ValueError):
                pass
    return found


def deflated_sharpe_ratio(
    observed_sharpe: float,
    *,
    n_trials: int,
    n_observations: int,
    skewness: float = 0.0,
    excess_kurtosis: float = 0.0,
) -> float:
    """Return a normal-approximation deflated Sharpe probability.

    ``n_trials`` is supplied by the research ledger when available.  The
    expected maximum null Sharpe grows with the number of tested alternatives,
    so the observed score is discounted before conversion to a probability.
    """
    observed = _finite_float(observed_sharpe)
    if observed is None:
        return 0.0
    n = max(2, int(n_observations))
    trials = max(1, int(n_trials))
    expected_max = math.sqrt(2.0 * math.log(trials)) / math.sqrt(n)
    variance = max(
        1.0e-12,
        (
            1.0
            - float(skewness) * observed
            + ((float(excess_kurtosis) - 1.0) / 4.0) * observed**2
        )
        / n,
    )
    z = (observed - expected_max) / math.sqrt(variance)
    return float(0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def summarize_multiplicity(
    *,
    fold_returns: Iterable[float] = (),
    n_trials: int | None = None,
    matrix: Any | None = None,
    matrix_orientation: str | None = None,
    candidate_fold_matrix: Any | None = None,
    trial_count_ledger: int | None = None,
    ledger_counters: Mapping[str, Any] | None = None,
    ledger_path: str | Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Create a compact, ledger-ready multiplicity report.

    ``matrix`` and ``candidate_fold_matrix`` are aliases.  When a matrix is
    present, PBO is calculated from the IS winner and OOS median per fold and
    the placeholder note is omitted.
    """
    supplied_matrix = matrix if matrix is not None else candidate_fold_matrix
    records: list[dict[str, Any]] = []
    if supplied_matrix is not None:
        try:
            records = _records_from_matrix(
                supplied_matrix,
                orientation=matrix_orientation,
            )
        except ValueError as error:
            if "matrix_orientation" in str(error):
                raise
            records = []
        except TypeError:
            records = []

    ledger_trials = trial_count_ledger
    if ledger_trials is None and ledger_counters is not None:
        ledger_trials = trial_count_from_counters(ledger_counters)
    if ledger_trials is None and ledger_path is not None:
        ledger_trials = read_ledger_trial_count(ledger_path, run_id=run_id)
    if ledger_trials is None:
        ledger_trials = n_trials
    if ledger_trials is None:
        ledger_trials = len({str(row["candidate_id"]) for row in records}) or 1
    ledger_trials = max(0, int(ledger_trials))
    trial_count = ledger_trials if ledger_trials else max(1, int(n_trials or 1))

    returns = np.asarray(list(fold_returns), dtype=float)
    returns = returns[np.isfinite(returns)]
    if returns.size == 0 and records:
        returns = np.asarray(_selected_oos_scores(records), dtype=float)
    if returns.size == 0:
        observed = 0.0
    else:
        scale = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0
        observed = (
            float(np.mean(returns) / scale)
            if scale > 0.0
            else float(np.mean(returns))
        )

    pbo: float | None = None
    if records:
        pbo = estimate_pbo(matrix=records)
    candidates_tested = len({str(row["candidate_id"]) for row in records})
    if candidates_tested == 0:
        candidates_tested = int(max(0, ledger_trials))
    dsr = deflated_sharpe_ratio(
        observed,
        n_trials=trial_count,
        n_observations=int(returns.size),
    ) if returns.size else 0.0
    summary: dict[str, Any] = {
        "trial_count": int(trial_count),
        "trial_count_ledger": int(ledger_trials),
        "trial_count_source": (
            "ledger"
            if trial_count_ledger is not None
            or ledger_counters is not None
            or ledger_path is not None
            else "fallback"
        ),
        "candidates_tested": int(candidates_tested),
        "observations": int(returns.size),
        "observed_sharpe_proxy": float(observed),
        "dsr_probability": float(dsr),
        "deflated_sharpe_probability": float(dsr),
        "pbo": pbo,
    }
    if ledger_counters is not None:
        summary["trial_count_counters"] = {
            str(key): int(value)
            for key, value in ledger_counters.items()
            if isinstance(value, (int, float))
        }
    if records:
        summary["pbo_method"] = "cscv_is_winner_vs_oos_median"
    if pbo is None:
        summary["pbo_note"] = "requires candidate-by-fold score matrix"
    return summary


def _canonical_metric_name(key: Any) -> str | None:
    text = str(key).strip()
    lowered = text.lower().replace("_", " ").replace("-", " ")
    aliases = {
        "return": "Return",
        "total return": "Return",
        "total return pct": "Return",
        "sortino": "Sortino",
        "sortino ratio": "Sortino",
        "pf": "PF",
        "profit factor": "PF",
        "mdd": "MDD",
        "max drawdown": "MDD",
        "max drawdown pct": "MDD",
        "expectancy": "Expectancy",
        "expectancy pct per trade": "Expectancy",
        "worstmonth": "WorstMonth",
        "worst month": "WorstMonth",
        "worst month return": "WorstMonth",
        "worst month return pct": "WorstMonth",
    }
    if text in {
        "Return", "Sortino", "PF", "MDD", "Expectancy", "WorstMonth",
    } or text.endswith((" Long", " Short")):
        return text
    for direction in ("Long", "Short"):
        suffix = f" {direction.lower()}"
        if lowered.endswith(suffix):
            base = _canonical_metric_name(lowered[: -len(suffix)].strip())
            if base is not None and base not in {"Long", "Short"}:
                return f"{base} {direction}"
    return aliases.get(lowered)


def _seed_metric_values(record: Mapping[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    nested = record.get("metrics")
    if isinstance(nested, Mapping):
        values.update(_seed_metric_values(nested))
    for direction in ("long", "short"):
        direction_value = record.get(direction)
        if not isinstance(direction_value, Mapping):
            continue
        for key, value in direction_value.items():
            metric = _canonical_metric_name(key)
            number = _finite_float(value)
            if metric is not None and number is not None:
                values[f"{metric} {direction.title()}"] = number
    for key, value in record.items():
        if key in {"seed", "metrics", "long", "short"}:
            continue
        metric = _canonical_metric_name(key)
        number = _finite_float(value)
        if metric is not None and number is not None:
            values[metric] = number
    return values


def aggregate_seed_metrics(
    seed_records: Iterable[Mapping[str, Any]] | Mapping[Any, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate 5–10 seed metrics with mean/std, median, and IQR.

    The helper accepts flat metric rows (for example ``Return Long``) and the
    nested ``{"long": {...}, "short": {...}}`` form used by pipeline reports.
    Missing metrics are ignored independently, which keeps a failed direction
    visible without filling the report with fabricated zeros.
    """
    if isinstance(seed_records, Mapping):
        records = [
            value | {"seed": key}
            for key, value in seed_records.items()
            if isinstance(value, Mapping)
        ]
    else:
        records = [record for record in seed_records if isinstance(record, Mapping)]
    collected: dict[str, list[float]] = {}
    for record in records:
        for metric, value in _seed_metric_values(record).items():
            collected.setdefault(metric, []).append(float(value))

    result: dict[str, Any] = {"seed_count": len(records)}
    for metric in sorted(collected):
        values = np.asarray(collected[metric], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        median = float(np.median(values))
        q1, q3 = np.percentile(values, [25.0, 75.0])
        result[metric] = {
            "count": int(len(values)),
            "mean": mean,
            "std": std,
            "mean_pm_std": f"{mean:.6g} ± {std:.6g}",
            "median": median,
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
        }
    return result


# Descriptive aliases used by report callers.
aggregate_golden_baseline = aggregate_seed_metrics
aggregate_golden_baselines = aggregate_seed_metrics


__all__ = [
    "aggregate_golden_baseline",
    "aggregate_golden_baselines",
    "aggregate_seed_metrics",
    "candidate_fold_matrix_path",
    "deflated_sharpe_ratio",
    "estimate_pbo",
    "read_candidate_fold_matrix",
    "read_ledger_trial_count",
    "summarize_multiplicity",
    "trial_count_from_counters",
    "write_candidate_fold_matrix",
]

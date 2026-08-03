"""Research-integrity utilities shared by the pipeline and Phase 5.

The consumed ``test_new.csv`` tape is a diagnostic artifact, not a tuning
surface.  This module records enough lineage to make that contract auditable:
dataset hashes, strategy identities, trial counts, and the one-shot forward
acceptance decision.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from gpu_fuzzy_trader.phases.rule_identity import strategy_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1 << 20) -> str:
    """Return a stable SHA-256 digest for *path*."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Describe a CSV dataset without changing or caching its contents."""
    source = Path(path)
    manifest: dict[str, Any] = {
        "path": str(source),
        "exists": source.exists(),
    }
    if not source.exists():
        return manifest

    manifest.update({
        "sha256": sha256_file(source),
        "bytes": int(source.stat().st_size),
    })
    try:
        frame = pd.read_csv(source, usecols=lambda column: column in {
            "datetime", "symbol",
        })
        if "datetime" in frame.columns and not frame.empty:
            dates = pd.to_datetime(frame["datetime"], errors="coerce")
            valid_dates = dates.dropna()
            if not valid_dates.empty:
                manifest["min_datetime"] = valid_dates.min().isoformat()
                manifest["max_datetime"] = valid_dates.max().isoformat()
        if "symbol" in frame.columns:
            manifest["symbols"] = sorted(
                {str(value).strip() for value in frame["symbol"].dropna().unique()}
            )
        manifest["rows_metadata_scan"] = int(len(frame))
    except Exception as exc:  # pragma: no cover - defensive data diagnostics
        manifest["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return manifest


def write_dataset_manifests(
    output_dir: str | os.PathLike[str],
    datasets: Mapping[str, str | os.PathLike[str] | None],
) -> str:
    """Write a deterministic manifest for the datasets used by a run."""
    path = Path(output_dir) / "reports" / "dataset_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": _utc_now(),
        "datasets": {
            name: dataset_manifest(dataset_path)
            for name, dataset_path in datasets.items()
            if dataset_path
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(path)
def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def count_trials(
    *,
    phase2: Mapping[str, Any] | None = None,
    rb: Mapping[str, Any] | None = None,
) -> int:
    """Estimate the number of adaptive evaluations represented by artifacts."""
    total = 0
    for value in (phase2 or {}).values():
        if isinstance(value, list):
            total += len(value)
        elif isinstance(value, Mapping):
            total += int(value.get("archive_size", 0) or 0)
            total += int(value.get("generations", 0) or 0)
    for value in (rb or {}).values():
        if not isinstance(value, Mapping):
            continue
        total += int(value.get("n_positive_single_rules", 0) or 0)
        total += int(value.get("selected_rules", 0) or 0)
        risk_history = value.get("risk_history", [])
        if isinstance(risk_history, list):
            total += len(risk_history)
    return max(0, int(total))


class ExperimentLedger:
    """Append-only JSONL ledger for research runs and adaptation decisions."""

    def __init__(self, output_dir: str | os.PathLike[str]) -> None:
        self.path = Path(output_dir) / "reports" / "experiment_ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        payload = {
            "recorded_at": _utc_now(),
            **dict(record),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(_canonical_json(payload) + "\n")
        return payload


def forward_acceptance_lock_path(
    output_dir: str | os.PathLike[str],
) -> Path:
    return Path(output_dir) / "reports" / "forward_acceptance.json"


def reserve_forward_evaluation(
    forward_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Return forward metadata, refusing a previously consumed tape."""
    source = Path(forward_path)
    digest = sha256_file(source)
    lock_path = forward_acceptance_lock_path(output_dir)
    if lock_path.exists():
        try:
            previous = json.loads(lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Forward acceptance lock is unreadable: {lock_path}"
            ) from exc
        previous_forward = previous.get("forward", previous)
        if str(previous_forward.get("sha256")) == digest:
            raise RuntimeError(
                "FORWARD_CSV_PATH has already been evaluated for acceptance; "
                "use a new strictly later tape or remove the run output."
            )
        raise RuntimeError(
            "A different forward tape is already locked for this output run. "
            "Create a new output directory for another release candidate."
        )
    return {
        "path": str(source),
        "sha256": digest,
        "reserved_at": _utc_now(),
    }


def write_forward_acceptance_record(
    output_dir: str | os.PathLike[str],
    metadata: Mapping[str, Any],
    acceptance: Mapping[str, Any],
) -> str:
    """Persist the one-shot forward decision after evaluation completes."""
    path = forward_acceptance_lock_path(output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_at": _utc_now(),
        "forward": dict(metadata),
        "acceptance": dict(acceptance),
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return str(path)

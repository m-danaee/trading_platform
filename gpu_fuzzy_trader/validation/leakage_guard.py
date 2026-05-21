"""
validation/leakage_guard.py — Leakage_Guard

Negative-control diagnostic: injects a synthetic label-derived feature and
checks whether the feature-selection stack (Phase 1) detects it.  This is a
**diagnostic only** — production Phase 1 always runs on clean (un-injected)
data.  The probe injection happens on a COPY of the training data in a
separate diagnostic step; the main pipeline never sees the probe.

LeakageAlert contract:
  - Raised ONLY when the probe is NOT selected (the feature selector failed
    to detect blatant label leakage — this is a fundamental problem).
  - When the probe IS selected (the expected, healthy case), the diagnostic
    logs its rank/score and the pipeline continues normally.

Usage:
    from gpu_fuzzy_trader.validation.leakage_guard import Leakage_Guard, LeakageAlert

    guard = Leakage_Guard()
    train_diag = guard.inject_probe(train_df.copy())
    # ... run Phase 1 on train_diag ...
    report = guard.diagnose(selected_long, selected_short)
    if not report["probe_detected"]:
        raise LeakageAlert(report["message"])
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class LeakageAlert(RuntimeError):
    """Raised when the leakage guardrail detects a fundamental problem."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Leakage_Guard:
    """Negative-control diagnostic for feature-selection leakage sensitivity."""

    def __init__(
        self,
        probe_name: str = _cfg.LEAKAGE_GUARD_FEATURE_NAME,
        enabled: bool = _cfg.LEAKAGE_GUARD_ENABLED,
    ) -> None:
        self.probe_name = probe_name
        self.enabled = enabled

    # ------------------------------------------------------------------
    # Injection
    # ------------------------------------------------------------------

    def inject_probe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a synthetic leakage-probe column from future labels.

        The probe is ``label_max_288`` — a direct copy of a 288-step-ahead
        label.  A functioning feature selector should give this column
        near-maximum MI scores and select it in its top-K.

        Always operates on a copy; never modifies the original.  Returns
        *df* unchanged when *enabled* is False or ``label_max_288`` is missing.

        Parameters
        ----------
        df : pd.DataFrame
            Training DataFrame with label columns.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with the ``_leakage_probe`` column added.
        """
        if not self.enabled:
            logger.debug("Leakage guardrail disabled; skipping probe injection.")
            return df

        if self.probe_name in df.columns:
            logger.warning(
                "Probe column '%s' already exists in DataFrame; overwriting.",
                self.probe_name,
            )

        if "label_max_288" not in df.columns:
            logger.warning(
                "label_max_288 not found; probe injection skipped."
            )
            return df

        df = df.copy()
        df[self.probe_name] = df["label_max_288"].astype(float)
        logger.info(
            "Leakage guardrail: injected probe '%s' (mirrors label_max_288, "
            "%d rows)",
            self.probe_name, len(df),
        )
        return df

    def remove_probe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove the probe column, safe to call even when absent."""
        if self.probe_name in df.columns:
            return df.drop(columns=[self.probe_name])
        return df

    # ------------------------------------------------------------------
    # Diagnose
    # ------------------------------------------------------------------

    def diagnose(
        self,
        selected_long: list[dict],
        selected_short: list[dict],
    ) -> dict[str, Any]:
        """Run the negative-control diagnostic on selected features.

        Checks whether the synthetic ``_leakage_probe`` was selected when
        present during feature selection.  The expected (healthy) outcome
        is that the probe IS detected — the selector correctly identifies
        the blatant label-derived feature.

        ``LeakageAlert`` should be raised by the CALLER when
        ``probe_detected`` is False, meaning the feature selection stack
        failed to detect label leakage.

        Parameters
        ----------
        selected_long : list[dict]
            Phase 1 selected features for long (from probe-injected data).
        selected_short : list[dict]
            Phase 1 selected features for short (from probe-injected data).

        Returns
        -------
        dict
            ``{"probe_detected": bool, "message": str, "details": dict}``
        """
        if not self.enabled:
            return {
                "probe_detected": True,
                "message": "Leakage guardrail disabled — no diagnostic run.",
                "details": {},
            }

        long_names = [f["name"] for f in selected_long]
        short_names = [f["name"] for f in selected_short]

        appeared_long = self.probe_name in long_names
        appeared_short = self.probe_name in short_names

        long_rank = None
        long_score = None
        if appeared_long:
            for i, f in enumerate(selected_long):
                if f["name"] == self.probe_name:
                    long_rank = i + 1
                    long_score = f.get("score", None)
                    break

        short_rank = None
        short_score = None
        if appeared_short:
            for i, f in enumerate(selected_short):
                if f["name"] == self.probe_name:
                    short_rank = i + 1
                    short_score = f.get("score", None)
                    break

        details = {
            "appeared_long": appeared_long,
            "appeared_short": appeared_short,
            "long_rank": long_rank,
            "long_score": long_score,
            "short_rank": short_rank,
            "short_score": short_score,
        }

        probe_detected = appeared_long and appeared_short

        if probe_detected:
            message = (
                f"Leakage diagnostic: probe '{self.probe_name}' selected by "
                f"feature selector (long rank={long_rank}, score={long_score}; "
                f"short rank={short_rank}, score={short_score}). "
                f"Expected behaviour — selector is sensitive to label leakage."
            )
        else:
            missing = []
            if not appeared_long:
                missing.append("long")
            if not appeared_short:
                missing.append("short")
            message = (
                f"Leakage diagnostic FAILED: probe '{self.probe_name}' NOT "
                f"selected for {', '.join(missing)} direction(s). "
                f"Feature selection stack may be blind to label leakage."
            )

        logger.info(message)

        return {
            "probe_detected": probe_detected,
            "message": message,
            "details": details,
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def inject_leakage_probe(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper around Leakage_Guard().inject_probe()."""
    return Leakage_Guard().inject_probe(df)


def diagnose_leakage_guard(
    selected_long: list[dict],
    selected_short: list[dict],
) -> dict[str, Any]:
    """Convenience wrapper around Leakage_Guard().diagnose()."""
    return Leakage_Guard().diagnose(selected_long, selected_short)

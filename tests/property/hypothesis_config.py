"""Shared Hypothesis settings for property tests.

Set PYTEST_LOW_MEMORY=1 to scale down example counts (for local / WSL runs).
Override the scale with HYPOTHESIS_EXAMPLE_SCALE (e.g. 0.5).
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

_LOW_MEMORY = os.environ.get("PYTEST_LOW_MEMORY", "").strip().lower() in (
    "1",
    "true",
    "yes")
_DEFAULT_SCALE = "0.25" if _LOW_MEMORY else "1.0"
_EXAMPLE_SCALE = float(os.environ.get(
    "HYPOTHESIS_EXAMPLE_SCALE", _DEFAULT_SCALE))

_DEFAULT_SUPPRESS = [
    HealthCheck.too_slow,
    HealthCheck.large_base_example,
]


def prop_settings(
    max_examples: int = 100,
    *,
    suppress_health_check: list[HealthCheck] | None = None,
    deadline: int | None | object = ...,
    **kwargs,
):
    """Hypothesis settings with optional low-memory example scaling."""
    scaled = max(5, int(max_examples * _EXAMPLE_SCALE))
    checks = (
        _DEFAULT_SUPPRESS
        if suppress_health_check is None
        else suppress_health_check
    )
    settings_kwargs: dict = {
        "max_examples": scaled,
        "suppress_health_check": checks,
        **kwargs,
    }
    if deadline is not ...:
        settings_kwargs["deadline"] = deadline
    return settings(**settings_kwargs)

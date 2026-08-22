"""Shared Hypothesis settings for property tests.

Set PYTEST_LOW_MEMORY=1 to scale down example counts (for local / WSL runs).
Override the scale with HYPOTHESIS_EXAMPLE_SCALE (e.g. 0.5).

Low-memory scale table (PYTEST_LOW_MEMORY=1):
    max_examples=500 → 50   (was 125)
    max_examples=300 → 30   (was 75)
    max_examples=200 → 20   (was 50)
    max_examples=100 → 10   (was 25)
    max_examples=50  → 5    (was 13)
"""

from __future__ import annotations

import os

from hypothesis import HealthCheck, settings

_LOW_MEMORY = os.environ.get("PYTEST_LOW_MEMORY", "").strip().lower() in (
    "1",
    "true",
    "yes",
)
# Low-memory scale reduced from 0.25 → 0.10 to cut property test runtime
_DEFAULT_SCALE = "0.10" if _LOW_MEMORY else "1.0"
_EXAMPLE_SCALE = float(os.environ.get("HYPOTHESIS_EXAMPLE_SCALE", _DEFAULT_SCALE))

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
    # Minimum floor lowered to 1 in low memory mode so tiny counts don't stay expensive
    min_floor = 1 if _LOW_MEMORY else 3
    scaled = max(min_floor, int(max_examples * _EXAMPLE_SCALE))
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
    if deadline is ...:
        if _LOW_MEMORY:
            # Feature-selection properties legitimately exceed Hypothesis's
            # 200 ms default while NumPy/Numba warm up on constrained hosts.
            settings_kwargs["deadline"] = None
    else:
        settings_kwargs["deadline"] = deadline
    return settings(**settings_kwargs)

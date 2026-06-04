"""
CPU-only runtime bootstrap for config Optuna tuning.

Must run before any ``gpu_fuzzy_trader`` import that loads JAX (e.g.
``run_pipeline``). Idempotent — safe to call from ``run_study`` and CLI.
"""

from __future__ import annotations

import os


def configure_tuning_cpu_env(*, force: bool = True) -> None:
    """
    Pin JAX to CPU and disable GPU memory preallocation for tuning hosts.

    Parameters
    ----------
    force : bool
        When True (default), always set ``JAX_PLATFORMS=cpu``. When False,
        only apply defaults that are not already set.
    """
    if force:
        os.environ["JAX_PLATFORMS"] = "cpu"
    else:
        os.environ.setdefault("JAX_PLATFORMS", "cpu")

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOGLEVEL", "3")

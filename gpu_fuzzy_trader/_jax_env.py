"""JAX/XLA runtime configuration — call before the first ``import jax``."""

from __future__ import annotations

import logging
import os


def configure_jax_env() -> None:
    """
    Reduce noisy JAX startup logs on desktop GPU / WSL setups.

    - ``JAX_PLATFORMS``: skip TPU backend probing when no TPU is present.
    - ``ABSL_MIN_LOGLEVEL`` / ``TF_CPP_MIN_LOG_LEVEL``: hide benign CUDA driver
      version parse errors from XLA on WSL.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOGLEVEL", "3")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

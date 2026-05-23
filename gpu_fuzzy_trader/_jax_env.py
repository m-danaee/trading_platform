"""JAX/XLA runtime configuration — call before the first ``import jax``."""

from __future__ import annotations

import logging
import os


def configure_jax_env() -> None:
    """
    Configure JAX/XLA runtime for optimal GPU utilization.

    - ``XLA_PYTHON_CLIENT_MEM_FRACTION``: Use 80% of GPU memory for better
      throughput during batched operations.
    - ``JAX_PLATFORMS``: skip TPU backend probing when no TPU is present.
    - ``ABSL_MIN_LOGLEVEL`` / ``TF_CPP_MIN_LOG_LEVEL``: hide benign CUDA driver
      version parse errors from XLA on WSL.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "true")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.80")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOGLEVEL", "3")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

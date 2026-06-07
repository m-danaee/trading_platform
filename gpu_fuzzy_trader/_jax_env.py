"""JAX/XLA runtime configuration — call before the first ``import jax``."""

from __future__ import annotations

import logging
import os


def configure_jax_env() -> None:
    """
    Configure JAX/XLA runtime for predictable desktop-friendly GPU usage.

    - ``XLA_PYTHON_CLIENT_PREALLOCATE``: keep memory on demand by default so
      JAX does not reserve most of the VRAM at startup.
    - ``XLA_PYTHON_CLIENT_MEM_FRACTION``: cap allocator growth when JAX does
      need to expand its GPU pool.
    - ``JAX_PLATFORMS``: skip TPU backend probing when no TPU is present.
    - ``JAX_COMPILATION_CACHE_DIR``: persist compiled XLA programs across restarts.
    - ``ABSL_MIN_LOGLEVEL`` / ``TF_CPP_MIN_LOG_LEVEL``: hide benign CUDA driver
      version parse errors from XLA on WSL.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.8")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    # Default off: Phase 2 GPU ranking uses float32; T4 FP64 throughput is poor.
    os.environ.setdefault("JAX_ENABLE_X64", "False")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOGLEVEL", "3")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

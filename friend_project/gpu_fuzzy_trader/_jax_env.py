
from __future__ import annotations

import logging
import os


def configure_jax_env() -> None:
    """
    Reduce noisy JAX startup logs on desktop GPU / WSL setups.

    - ``JAX_PLATFORMS``: skip TPU backend probing when no TPU is present.
    - ``JAX_COMPILATION_CACHE_DIR``: persist compiled XLA programs across restarts.
    - ``ABSL_MIN_LOGLEVEL`` / ``TF_CPP_MIN_LOG_LEVEL``: hide benign CUDA driver
      version parse errors from XLA on WSL.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "True")
    if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        if os.path.isdir("/content"):
            cache_dir = "/content/jax_cache"
            os.makedirs(cache_dir, exist_ok=True)
            os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir
        else:
            os.environ.setdefault(
                "JAX_COMPILATION_CACHE_DIR", "/tmp/jax_cache")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOGLEVEL", "3")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

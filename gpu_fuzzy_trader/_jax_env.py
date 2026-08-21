"""JAX/XLA runtime configuration — call before the first ``import jax``."""

from __future__ import annotations

import logging
import os
import shutil
import site
from pathlib import Path

logger = logging.getLogger(__name__)


def _cuda_package_root() -> Path | None:
    """Find a pip-installed CUDA toolkit root, if one is available."""
    candidates: list[Path] = []
    for value in (
        os.environ.get("CUDA_HOME", ""),
        os.environ.get("CUDA_PATH", ""),
        "/usr/local/cuda",
        "/opt/cuda",
    ):
        if value:
            candidates.append(Path(value))
    for site_dir in site.getsitepackages():
        candidates.append(Path(site_dir) / "nvidia" / "cu13")
    virtual_env = os.environ.get("VIRTUAL_ENV", "")
    if virtual_env:
        candidates.append(Path(virtual_env) / "nvidia" / "cu13")

    for root in candidates:
        if (root / "nvvm" / "libdevice").is_dir():
            return root
    return None


def _append_xla_flag(flags: str, flag: str) -> str:
    """Append an XLA flag once, preserving explicit user configuration."""
    flag_key = flag.split("=")[0]
    if flag_key in flags or flag in flags:
        return flags
    return f"{flags} {flag}".strip()


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

    if "JAX_COMPILATION_CACHE_DIR" not in os.environ:
        if os.path.isdir("/content"):
            cache_dir = "/content/jax_cache"
        else:
            cache_dir = "/tmp/jax_cache"
        os.environ["JAX_COMPILATION_CACHE_DIR"] = cache_dir

    cache_path = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    if cache_path:
        try:
            os.makedirs(cache_path, exist_ok=True)
        except Exception as exc:
            logger.debug("Failed to create JAX cache directory %s: %s", cache_path, exc)

    cuda_root = _cuda_package_root()
    flags = os.environ.get("XLA_FLAGS", "")
    if cuda_root is not None:
        cuda_bin = cuda_root / "bin"
        path_entries = os.environ.get("PATH", "").split(os.pathsep)
        if cuda_bin.is_dir() and str(cuda_bin) not in path_entries:
            os.environ["PATH"] = (
                f"{cuda_bin}{os.pathsep}{os.environ.get('PATH', '')}"
            )
        flags = _append_xla_flag(
            flags, f"--xla_gpu_cuda_data_dir={cuda_root}",
        )
    if shutil.which("ptxas") is None:
        # NVIDIA drivers can compile PTX when the CUDA toolkit is not present.
        # This keeps GPU execution available, while an installed NVCC toolkit
        # remains preferred because it avoids repeated driver-side JIT work.
        flags = _append_xla_flag(
            flags, "--xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found",
        )
    if flags:
        os.environ["XLA_FLAGS"] = flags
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("ABSL_MIN_LOGLEVEL", "3")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)
    logger.debug(
        "JAX env configured: platforms=%s preallocate=%s mem_fraction=%s cache_dir=%s xla_flags=%s",
        os.environ.get("JAX_PLATFORMS"),
        os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE"),
        os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION"),
        os.environ.get("JAX_COMPILATION_CACHE_DIR"),
        os.environ.get("XLA_FLAGS", ""),
    )

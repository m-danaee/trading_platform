
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

_MEMORY_LOGGING = os.environ.get("LOG_MEMORY", "").strip().lower() in (
    "1",
    "true",
    "yes",
)


def log_memory_rss(label: str) -> None:
    """Log current process RSS in MiB when LOG_MEMORY=1."""
    if not _MEMORY_LOGGING:
        return
    try:
        import psutil

        rss_mib = psutil.Process().memory_info().rss / (1024 * 1024)
        logger.info("Memory [%s]: %.1f MiB RSS", label, rss_mib)
    except ImportError:
        logger.debug("LOG_MEMORY set but psutil not installed")


def release_phase2_resources() -> None:
    """Force GC and clear JAX compilation caches between Phase 2 directions."""
    import gc

    gc.collect()
    try:
        import jax

        jax.clear_caches()
    except Exception:
        pass

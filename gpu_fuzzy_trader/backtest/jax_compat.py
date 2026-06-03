"""
Detect whether JAX / GPUBacktestEngine can be loaded on this host.

jaxlib wheels are often built with AVX (and NumPy 2.4+ with x86-64-v2). Older
CPUs and some VPS instances fail at import with RuntimeError, not ImportError.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

logger = logging.getLogger(__name__)

_GPU_ENGINE_ERRORS = (ImportError, RuntimeError, OSError)


def get_gpu_backtest_engine_class() -> Optional[Type[Any]]:
    """
    Return ``GPUBacktestEngine`` if ``gpu_engine`` imports cleanly; else ``None``.
    """
    try:
        from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine

        return GPUBacktestEngine
    except _GPU_ENGINE_ERRORS as exc:
        logger.info(
            "GPU backtest engine unavailable (%s: %s); CPU engine will be used.",
            type(exc).__name__,
            exc,
        )
        return None


def jax_gpu_backtest_available() -> bool:
    """True when ``get_gpu_backtest_engine_class()`` would succeed."""
    return get_gpu_backtest_engine_class() is not None

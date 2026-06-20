"""
Detect whether JAX / GPUBacktestEngine can be loaded on this host.

JAX can fail during import or initialisation with a variety of exceptions:
ImportError, RuntimeError, OSError, or AttributeError (e.g. AVX/NumPy
incompatibility, partially-initialised module state).  All are caught so
that callers fall back to CPUBacktestEngine.
"""

from __future__ import annotations

import logging
from typing import Any, Optional, Type

logger = logging.getLogger(__name__)

_GPU_ENGINE_ERRORS = (ImportError, RuntimeError, OSError, AttributeError)


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

"""CPU and GPU backtest engine sub-package."""

from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.backtest.jax_compat import (
    get_gpu_backtest_engine_class,
    jax_gpu_backtest_available,
)

__all__ = [
    "CPUBacktestEngine",
    "get_gpu_backtest_engine_class",
    "jax_gpu_backtest_available",
]

try:
    GPUBacktestEngine = get_gpu_backtest_engine_class()
except Exception:
    GPUBacktestEngine = None
if GPUBacktestEngine is not None:
    __all__.append("GPUBacktestEngine")

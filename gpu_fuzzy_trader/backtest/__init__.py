"""CPU and GPU backtest engine sub-package."""

from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine

__all__ = ["CPUBacktestEngine"]


def _try_import_gpu():
    try:
        from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine
        return GPUBacktestEngine
    except ImportError:
        return None


GPUBacktestEngine = _try_import_gpu()
if GPUBacktestEngine is not None:
    __all__.append("GPUBacktestEngine")

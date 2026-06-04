"""
CLI: python -m gpu_fuzzy_trader.tuning

Prefer CPU for low-RAM hosts before JAX is initialized.
"""

from __future__ import annotations

from gpu_fuzzy_trader.tuning._bootstrap import configure_tuning_cpu_env

configure_tuning_cpu_env(force=True)

if __name__ == "__main__":
    from gpu_fuzzy_trader.tuning.study_runner import main

    main()

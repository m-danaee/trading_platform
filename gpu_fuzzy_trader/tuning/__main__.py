"""
CLI: python -m gpu_fuzzy_trader.tuning

Prefer CPU for low-RAM hosts before JAX is initialized.
"""

from __future__ import annotations
from gpu_fuzzy_trader.tuning.study_runner import main

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")


if __name__ == "__main__":
    main()

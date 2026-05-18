"""
__main__.py — Entry point for `python -m gpu_fuzzy_trader.run_pipeline`

Allows the pipeline to be invoked as:
    python -m gpu_fuzzy_trader.run_pipeline

Requirements: 13.4
"""

import sys

from gpu_fuzzy_trader.run_pipeline import main

if __name__ == "__main__":
    main(sys.argv[1:])

#!/usr/bin/env python3
"""Colab validation checklist after inflated-returns fixes.

Run on Colab (not WSL) after pulling these changes:
1. Rebuild labels:  python build_train2_test2.py
2. Run pipeline:   python -m gpu_fuzzy_trader.run_pipeline --output /content/trading_platform_outputs
3. Open evaluator_v5.ipynb and evaluate outputs/long.json and outputs/short.json

This script verifies local prerequisites and config; it does not run the full pipeline.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from gpu_fuzzy_trader import config as cfg

    assert cfg.PHASE2_CAPITAL_PCT == 18.0
    assert cfg.PHASE2_JOINT_TRAIN_VAL is True
    assert cfg.RB_TRAIN_VALID_MAX_RATIO == 1.15
    assert cfg.RB_MAX_RULES == 5
    assert cfg.PHASE2_VAL_RETURN_FLOOR_PCT_SHORT == 2.0

    label_test = (
        "tests/unit/test_build_train2_labels.py::"
        "TestForwardWindowLabels::test_label_max_288_uses_future_highs_only"
    )
    subprocess.run(
        [sys.executable, "-m", "pytest", label_test, "-q"],
        cwd=ROOT,
        check=True,
    )
    print("OK: forward labels + anti-overfit config verified.")
    print("Next on Colab: rebuild train_2/test_2, run pipeline, check evaluator_v5.ipynb.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

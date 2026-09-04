"""Fresh-process import regression tests for Phase 2."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_phase2_rule_pool_imports_in_fresh_process():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(repo_root), env.get("PYTHONPATH", "")) if part
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import gpu_fuzzy_trader.phases.phase2_rule_pool",
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout

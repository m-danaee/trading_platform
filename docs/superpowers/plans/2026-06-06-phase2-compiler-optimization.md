# Phase 2 JAX Loop Unrolling Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Increase the JAX loop unrolling factor (`PHASE2_SCAN_UNROLL`) to 32 to speed up Phase 2 rule search simulation on the GPU.

**Architecture:** Change the loop unroll constant in `config.py` and update unit tests to assert the new unroll factor value.

**Tech Stack:** Python, JAX, Pytest

---

### Task 1: Update config unrolling and add unit test verification

**Files:**
* Modify: `gpu_fuzzy_trader/config.py`
* Modify: `tests/unit/test_config_additions.py`

- [ ] **Step 1: Write the failing test**

Add an assertion verifying that the config parameter `PHASE2_SCAN_UNROLL` is set to `32` in `tests/unit/test_config_additions.py`.

```python
from gpu_fuzzy_trader import config as c

def test_new_config_parameters_exist():
    assert c.PHASE2_REGIME_PROFITABILITY_GATE is True
    assert c.PHASE2_REGIME_MIN_RETURN_PER_REGIME == 0.5
    assert c.PHASE1_REQUIRE_SIGN_CONSISTENCY is True
    assert c.PHASE1_SIGN_CONSISTENCY_MIN_FOLDS == 2
    assert c.PHASE2_RECENCY_WEIGHT_ENABLED is True
    assert c.PHASE2_RECENCY_WEIGHT_FRACTION == 0.25
    assert c.PHASE2_RECENCY_WEIGHT_MULTIPLIER == 2.0
    assert c.PHASE2_REQUIRE_LAST_FOLD_POSITIVE is False
    assert c.PHASE2_SCAN_UNROLL == 32
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_config_additions.py -v`
Expected: FAIL (assertion error: `assert 16 == 32`)

- [ ] **Step 3: Write minimal implementation**

Update `PHASE2_SCAN_UNROLL` in `gpu_fuzzy_trader/config.py` from `16` to `32`:

```python
PHASE2_SCAN_UNROLL = 32
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/test_config_additions.py -v`
Expected: PASS

- [ ] **Step 5: Run all unit tests to ensure no regressions**

Run: `PYTHONPATH=. .venv/bin/pytest tests/unit/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

Run:
```bash
git add gpu_fuzzy_trader/config.py tests/unit/test_config_additions.py
git commit -m "perf: increase Phase 2 JAX loop unroll factor to 32"
```

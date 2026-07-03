# Task-21: Island RNG State Leakage & Generation-Budget Realization

**Branch:** `fix/island-rng-and-budget`
**Priority:** 🟠 High
**Depends on:** none (independent, but may conflict with task-22 on evox_runner.py)

## Problem

The 2026-07-01 root-cause audit identified two independent issues in the island
scheduler and pool generator's random-number handling:

### Problem A: RNG State Leakage Across Resumed Epochs
In `phase2_rule_pool.py`, every `run()` (line 2241), `run_epoch()` (line 2846),
and `finalize_island()` (line 2888) re-creates the RNG from the same unchanging
`self.seed`:
```python
rng = np.random.default_rng(self.seed)
```
When an island runs multiple epochs (resumed across `run_epoch` calls), each
epoch starts with the exact same RNG state, producing identical draw sequences.
This reduces cross-epoch exploration and potentially causes premature convergence
by repeatedly sampling the same crossover/mutation/initialization paths.

### Problem B: Identical Seed Shared Across All Islands
Both cluster and orphan islands are created with the same base `seed` value
(`phase2_island_scheduler.py` lines 277, 347). All 3 cluster islands plus any
orphan islands share one identical seed, reducing cross-island exploration
diversity — the different islands explore similar regions of the search space
instead of complementary ones.

## Files to Modify

1. `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — persistent advancing RNG per island
2. `gpu_fuzzy_trader/phases/phase2_island_scheduler.py` — per-island seed derivation
3. `tests/unit/test_phase2_rule_pool.py` — regression tests

## Detailed Changes

### A1: Persistent `self._rng` in Rule_Pool_Generator

In `__init__`, create a persistent RNG from `self.seed` and store as `self._rng`:
```python
self._rng = np.random.default_rng(seed)
```

Replace all three `np.random.default_rng(self.seed)` calls in `run()`,
`run_epoch()`, and `finalize_island()` with `self._rng`:
```python
# Before: rng = np.random.default_rng(self.seed)
# After:
rng = self._rng
```

This ensures the RNG state advances naturally across `run_epoch()` calls and
never replays the same sequence.

**Compatibility check**: verify that no code path relies on the ability to
replay identical RNG sequences across calls (e.g., for determinism testing).
If such reliance exists, gate the new behavior behind a flag like
`self._rng_advancing = True`.

### A2: Per-Island Seed Derivation

In `phase2_island_scheduler.py`, derive a distinct seed per island instead of
passing the same base `seed` to every `Rule_Pool_Generator`:

```python
import hashlib

def _derive_island_seed(base_seed: int, island_id: str) -> int:
    """Return a deterministic but distinct seed for each island."""
    h = hashlib.sha256(f"{base_seed}:{island_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31)
```

Use this when creating generators:
```python
# Before: seed=seed
# After: seed=_derive_island_seed(seed, island_id)
```

Apply to both cluster islands (line 347) and orphan islands (line 277).

### A3: Regression Tests

Add tests confirming:
- Two `run_epoch()` calls on the same island produce different RNG sequences (no state leakage)
- Two islands with different `island_id` produce different initial populations
- `_derive_island_seed` produces distinct outputs for distinct `island_id` values
- The persistent `self._rng` advances across calls (verify `rng.bit_generator.state` differs)

## Acceptance Criteria

- [ ] `Rule_Pool_Generator` maintains one persistent `self._rng` across its lifetime; `run()`, `run_epoch()`, and `finalize_island()` use `self._rng` instead of `np.random.default_rng(self.seed)`.
- [ ] Each island receives a distinct seed derived from `base_seed + island_id`.
- [ ] Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_island_scheduler.py -x -q`
- [ ] New test(s) assert RNG state advances across multiple `run_epoch()` calls.
- [ ] New test(s) assert distinct seeds produce distinct initial populations.
- [ ] `evaluator_v5.ipynb` NOT modified.

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_phase2_rule_pool.py tests/unit/test_phase2_island_scheduler.py -x -q
```

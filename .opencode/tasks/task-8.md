# Task 8 — Add `regime_feature_keyword` stratum initialization

## Why
The friend uses a "regime" stratum in their NSGA-III population
initialization: 25% of new chromosomes are forced to have their
**first active gene** be a "regime/volatility/trend" feature (vol,
atr, bb_width, compression, adx, dmi, semivol, etc.). This is a
feature-space proxy for regime-aware rules.

My project ALREADY has a bar-level regime detector (Task 1 didn't
add this; the regime_cluster.py has been here for a while). It
assigns a per-bar `regime` label (0=sideways, 1=bear, 2=bull) and
the Phase 2 evolution uses this in the per-regime objective. So I
have a stronger regime signal than the friend.

What the friend has that I lack: the *initialization* bias toward
regime-related features. This ensures that even if the bar-level
regime label is noisy, the chromosomes are anchored on features
that the literature associates with market regime (volatility,
trends, breakouts). The two signals (bar-level regime + feature
keywords) are complementary.

## Required reading
- `.opencode/plans/PLAN.md`
- `.opencode/CONTEXT.md` (JSON output contract)
- The friend's reference: `friend_project/gpu_fuzzy_trader/phases/phase2_rule_pool.py` lines 286-345 (`_softmax_feature_probs`, `_regime_feature_indices`, `_sample_active_indices`, `_sample_stratified_chromosome`); and `friend_project/gpu_fuzzy_trader/config.py` lines 154-160 (`PHASE2_REGIME_FEATURE_KEYWORDS`, `PHASE2_STRATA_REGIME_FRAC=0.25`).
- My existing `gpu_fuzzy_trader/phases/phase2_init.py` (the existing `Stratum = Literal["elite", "explorer"]` enum and the `assign_strata_to_indices` function).
- My existing `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (the population initialization call site).

## Behavior changes

### Step 1 — Add new config keys

```python
# Phase 2 regime stratum initialization (complements the bar-level regime label)
PHASE2_REGIME_STRATUM_ENABLED = True
PHASE2_REGIME_STRATUM_FRAC = 0.25
PHASE2_REGIME_FEATURE_KEYWORDS = (
    "vol", "atr", "bb_width", "compression", "range", "trend", "regime",
    "breakout", "drawdown", "channel", "adx", "dmi", "semivol",
)
```

### Step 2 — Extend the `Stratum` literal to include "regime"

In `gpu_fuzzy_trader/phases/phase2_init.py`:
```python
Stratum = Literal["elite", "explorer", "regime"]
```

### Step 3 — Add `_regime_feature_indices` helper

Port the friend's helper. Signature:
```python
def _regime_feature_indices(feature_infos: list[dict]) -> list[int]:
    """Return indices of features whose names look regime/volatility/trend related.
    
    Matches feature names (case-insensitive) against
    PHASE2_REGIME_FEATURE_KEYWORDS from config.
    """
```

### Step 4 — Extend `assign_strata_to_indices` to support 3 strata

In `gpu_fuzzy_trader/phases/phase2_init.py`, modify the function to
take a 3-tuple of fractions (elite, explorer, regime) instead of a
2-tuple (elite, explorer). When `PHASE2_REGIME_STRATUM_ENABLED=True`,
the new third fraction is `PHASE2_REGIME_STRATUM_FRAC=0.25`. When
False, falls back to the 2-stratum behavior (regime frac=0).

Add a new helper function `assign_three_strata_to_indices` that
returns a list of `"elite" | "explorer" | "regime"` strings. The
existing `assign_strata_to_indices` stays for backward compat.

### Step 5 — Add a `regime` stratum sampling function

In `phase2_init.py`, add `sample_regime_stratum_chromosome` that
returns a chromosome whose first active gene is a regime-feature
index. The remaining active genes are sampled uniformly (or
softmax-weighted) from the rest.

### Step 6 — Wire into the population initialization

In `phase2_rule_pool.py`'s `_init_population` (or wherever
initialization happens), check `PHASE2_REGIME_STRATUM_ENABLED`. If
True, use `assign_three_strata_to_indices` and call
`sample_regime_stratum_chromosome` for the "regime" rows. If False,
use the existing 2-stratum behavior.

## Out of scope
- Do NOT change the JSON output format.
- Do NOT modify `evaluator_v5.ipynb`.
- Do NOT touch the GPU engine or EvoX runner's NSGA-III environmental selection (the stratum affects initialization only).
- Do NOT change the bar-level regime detection (Task 1 didn't add it; it was pre-existing).
- Do NOT add Task 9 features.

## Acceptance criteria
1. All 4 new config keys are present and accessible.
2. `Stratum` literal includes "regime" (3 options).
3. `_regime_feature_indices` is importable and returns the right indices for a synthetic `feature_infos` list with regime/non-regime features.
4. `assign_three_strata_to_indices` is importable and returns a list of `"elite" | "explorer" | "regime"` strings, with the right fractions.
5. `sample_regime_stratum_chromosome` is importable and returns a chromosome whose first active gene is a regime feature.
6. The wire-in into `_init_population` is correct: 25% of non-seeded rows have their first active gene from the regime-keyword list.
7. New unit test `tests/unit/test_regime_keyword_stratum.py` with ≥ 4 cases:
   - `_regime_feature_indices` returns the right indices for a mix of regime and non-regime features.
   - `assign_three_strata_to_indices` returns the right fractions.
   - `sample_regime_stratum_chromosome` always picks a regime feature for the first active gene.
   - When `PHASE2_REGIME_STRATUM_ENABLED=False`, the existing 2-stratum behavior is preserved.
8. All existing tests pass.
9. No changes to `evaluator_v5.ipynb` or the GPU engine.

## Constraints
- Stay on `feature/task-8-regime-keyword-stratum` (off `main` after task-7 is merged).
- 12.7 GiB RAM total.
- PEP 8, type hints, module logger.
- Use only existing third-party deps.

## Files I will touch
- `gpu_fuzzy_trader/config.py` — 4 new `PHASE2_REGIME_STRATUM_*` keys
- `gpu_fuzzy_trader/phases/phase2_init.py` — add "regime" to Stratum literal; add helpers
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` — wire the new stratum into `_init_population`
- `tests/unit/test_regime_keyword_stratum.py` (new) — ≥ 4 cases

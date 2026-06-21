# Plan: Per-Symbol Phase 2 + RB Governor + Phase 5 Filtering + Colab GPU

## Goal

1. **Per-symbol Phase 2** — Evolve rule pools per symbol (each symbol's unique characteristics).
2. **Per-symbol RB Governor** — Validate each pool on its symbol's data only, produce per-symbol rule sets, merge into final strategy.
3. **Phase 5 rule filtering** — Remove rules with negative test PnL (match main project's Phase 5).
4. **GPU optimization for Colab** — Add `_gpu_runtime.py`, GPU config knobs, Colab-aware `_jax_env.py`.
5. **Colab notebook** — `friend.ipynb` for one-click Colab execution.

## Architecture

```
train.csv → per-symbol 75/25 split
  ├── train_df (all symbols)
  └── val_df (all symbols)

Phase 1 (unchanged)

Phase 2 (per-symbol):
  For each symbol S:
    train_df_S → NSGA-II/III → pool_S
    Tag rules with "symbol is S"
    Save: pools/per_symbol/phase2_{direction}_{symbol}_pool.json
  Merge for backward compat

RB Governor (per-symbol):
  For each symbol S:
    train_df_S + val_df_S + pool_S (already tagged)
    RB_REQUIRE_SYMBOL_FILTERS=False (skip _symbol_specialized_variants)
    → rule_set_S
  Merge all rule sets → outputs/{direction}.json

Phase 5 (enhanced with rule filtering):
  Load final strategy → evaluate on test
  For each rule: sum(Net_PnL on test) <= 0 → remove
  Safeguard: keep at least PHASE3_GLOBAL_MIN_RULES rules
  Re-evaluate test with cleaned rules
  Rewrite strategy file + save reports
```

---

## Tasks

### task-1: Add GPU runtime support

**Target files:**
- `gpu_fuzzy_trader/_gpu_runtime.py` (new — copy+adapt from main project)
- `gpu_fuzzy_trader/config.py` (GPU knobs)
- `gpu_fuzzy_trader/_jax_env.py` (Colab cache path)
- `gpu_fuzzy_trader/run_pipeline.py` (wire GPU runtime)
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (use resolved batch size)

**Acceptance criteria:**
- `_gpu_runtime.py` imports cleanly (no `phase2_sparse_encoding` refs)
- `PHASE2_USE_GPU`, `PHASE2_GPU_BATCH_SIZE`, `PHASE2_GPU_BATCH_SIZE_AUTO`, `PHASE2_SCAN_UNROLL`, `PHASE2_GPU_USE_FP32`, `PHASE2_GPU_DATA_INT8` in config.py
- `is_colab_runtime()` and `_apply_colab_gpu_defaults()` work
- Phase 2 GPU engine works (already has `gpu_engine.py`, `evox_runner.py`)

---

### task-2: Per-symbol Phase 2 training

**Target files:**
- `gpu_fuzzy_trader/config.py` (`PER_SYMBOL_PHASE2`, per-symbol paths)
- `gpu_fuzzy_trader/run_pipeline.py` (per-symbol loop in `_run_phase2`)
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py` (purged CV skip, symbol tagging)

**Acceptance criteria:**
- `PER_SYMBOL_PHASE2=True` loops per symbol, filters train_df, runs Phase 2
- Each rule tagged with `{"feature": "symbol", "operator": "is", "value": "S"}`
- Per-symbol pools saved to `pools/per_symbol/`
- Combined pool for backward compat
- Purged CV disabled for per-symbol (runtime override)
- `PER_SYMBOL_PHASE2=False` restores original behavior

---

### task-3: Per-symbol RB Governor validation

**Target files:**
- `gpu_fuzzy_trader/run_pipeline.py` (per-symbol RB Governor loop)
- `gpu_fuzzy_trader/rb_governor.py` (support `RB_REQUIRE_SYMBOL_FILTERS=False`)

**Acceptance criteria:**
- RB Governor called per symbol with per-symbol train/val data + pool
- `RB_REQUIRE_SYMBOL_FILTERS=False` at runtime (rules already tagged)
- Per-symbol rule sets merged into final `outputs/{direction}.json`
- Debug output: `outputs/per_symbol/{symbol}_{direction}.json`
- Backward compatible

---

### task-4: Phase 5 rule filtering (remove negative-PnL rules on test)

**Target files:**
- `gpu_fuzzy_trader/phases/phase5_oos.py` (add `_remove_negative_pnl_rules`, wire into `run()`)
- `gpu_fuzzy_trader/output/writer.py` (add `_maybe_write_evaluator_clean`)
- `gpu_fuzzy_trader/config.py` (add `PHASE5_REMOVE_NEGATIVE_PNL_RULES`, `PHASE3_GLOBAL_MIN_RULES`, `WRITE_EVALUATOR_CLEAN`)

**What:**
- Add `_remove_negative_pnl_rules()` method to `OOS_Evaluator`:
  - For each rule index in test trade log, sum `Net_PnL`
  - Remove rules with `total_pnl <= 0`
  - Safeguard: keep at least `PHASE3_GLOBAL_MIN_RULES` rules
  - Log removed/kept counts
- Wire into `run()`:
  - After evaluating all splits, call `_remove_negative_pnl_rules()` with test trade log
  - If rules removed: re-evaluate test split, rebuild `all_per_symbol`
  - Rewrite cleaned strategy to `outputs/{direction}.json`
- Add `_maybe_write_evaluator_clean()` to `output/writer.py` (writes stripped evaluator-clean JSON)
- Add config flags: `PHASE5_REMOVE_NEGATIVE_PNL_RULES=True`, `PHASE3_GLOBAL_MIN_RULES=2`, `WRITE_EVALUATOR_CLEAN=True`

**Acceptance criteria:**
- Rules with `Net_PnL <= 0` on test are removed from strategy
- Minimum rule safeguard prevents over-pruning
- Test metrics are re-evaluated after pruning
- Strategy file rewritten with cleaned rules
- Evaluator-clean JSON written
- Gated by `PHASE5_REMOVE_NEGATIVE_PNL_RULES` config flag

---

### task-5: Create `friend.ipynb` Colab notebook

**Target file:**
- `friend.ipynb` (new)

**What:**
- Model after main project's `main.ipynb`
- Cell 1: markdown instructions
- Cell 2: GitHub PAT (optional)
- Cell 3 (bootstrap): project discovery, Drive mount, dataset location
- Cell 4 (install): JAX CUDA 12, EvoX, Torch CPU for Colab T4
- Cell 5 (import/config): JAX env, set `PER_SYMBOL_PHASE2=True`, GPU knobs
- Cell 6 (run): Execute pipeline, sync to Drive

**Acceptance criteria:**
- Notebook runs on Colab T4 without errors
- Pipeline: per-symbol Phase 2 → per-symbol RB Governor → Phase 5 filtering
- Outputs synced to Google Drive

---

## Dependencies

```
task-1 → task-2 → task-3 → task-4 → task-5
(GPU)   (Phase2) (RB Gov) (Phase5) (notebook)
```

## Non-goals

- Do NOT modify Phase 1, data splitter, or features
- Do NOT change the main project
- Do NOT add new dependencies
- Do NOT modify `evaluator_v5.ipynb`

# Task-20: Kill the Per-Generation Runtime Blowup (60-230s/gen)

**Branch:** `fix/phase2-runtime-blowup`
**Priority:** 🔴 Critical
**Depends on:** none (independent of task-21/22/23)

## Problem

The 2026-07-01 Colab log shows individual NSGA-III generations (pop=200, ~145K-194K
rows/island) taking **60-230 seconds each** — at that rate a 33-gen/island budget
across 3 islands x 2 directions cannot finish in a single Colab session. Root
causes, all independently confirmed by reading the code (not just log inference):

1. **Unconditional full CPU re-simulation every batch call.**
   `GPUBacktestEngine.simulate_rule_batch` (`gpu_fuzzy_trader/backtest/gpu_engine.py:896-912`)
   always runs, in addition to the JAX GPU path:
   ```python
   if _cfg.phase2_should_enrich_symbol_metrics(self):
       cpu_metrics = self._lazy_cpu_engine.simulate_rule_batch(...)
   ```
   `phase2_should_enrich_symbol_metrics()` (`config.py:1886-1890`) just returns
   `PHASE2_GPU_ENRICH_SYMBOL_METRICS`, which defaults `True` (`config.py:963`).
   `CPUBacktestEngine.simulate_rule_batch` → `_simulate_rule_set_entries`
   (`cpu_engine.py:846-1000+`) is a **serial Python `for entry in entries:` loop**
   (`cpu_engine.py:907`) with an inner linear scan of open positions
   (`cpu_engine.py:634`). It runs on every chromosome in the batch (up to
   pop_size new offspring), and with `MIN_CONDITIONS=4` out of ~25 features many
   chromosomes match a large fraction of 145K-194K rows. This is the intended
   "ground-truth" validation engine (see module docstring, `gpu_engine.py:9-10`),
   not something meant to run every generation for every individual — it exists
   only to fill `per_symbol_metrics` for the symbol-spread penalty
   (`phase2_rule_pool.py:574-582`, confirmed load-bearing, NOT dead code).

2. **`PHASE2_JOINT_TRAIN_VAL=True` silently disables the val-cadence throttle.**
   `evox_runner.py:2216-2220` (`_run_nsga3`) and the duplicated block at
   `evox_runner.py:1691-1694` (`_run_nsga2_fallback`):
   ```python
   run_val_this_gen = (
       is_last_gen
       or bool(_cfg.PHASE2_JOINT_TRAIN_VAL)          # always True today -> short-circuits
       or (gen % int(_cfg.PHASE2_VAL_SIM_INTERVAL) == 0)
   )
   ```
   Because of the `or`, `PHASE2_VAL_SIM_INTERVAL = 2` (`config.py:629`, meant to
   throttle val simulation) is dead code whenever `PHASE2_JOINT_TRAIN_VAL=True`
   (`config.py:610`, the current default). Every generation pays for **both**
   train and val `simulate_rule_batch` calls, each separately paying cost #1
   above — i.e. up to 4x full backtests/generation (train GPU, train CPU
   enrichment, val GPU, val CPU enrichment) instead of the intended throttle.

3. **Dynamic-shape "fast reject" compaction risks JIT recompilation.**
   `gpu_engine.py:854-878`: after computing per-chromosome signal counts, the
   code masks down to only chromosomes clearing the trade floor:
   ```python
   counts_np = np.asarray(signal_counts, dtype=np.int64)  # device->host sync
   scan_idx = np.flatnonzero(counts_np >= min_scan)
   ...
   sub_signals = signals_batch[scan_idx]                   # shape = scan_idx.size (dynamic)
   sub_results = self._simulate_signals_batch(sub_signals, ...)
   ```
   `scan_idx.size` changes with the population every generation, which is a
   classic JAX retrace/recompile trigger for the wrapped `lax.scan` equity
   simulator (`PHASE2_SCAN_UNROLL=32` over ~150K-194K rows — an expensive
   compile). The warmup routine (`_gpu_runtime.py`) only pre-warms the
   all-dont-care chromosome (which matches every row, so `scan_idx.size ==
   batch_size`), a shape that essentially never recurs for real chromosomes.

4. **Minor stacking costs:** per-chunk `np.asarray(...)` sync inside the
   `for start in range(0, B_padded, chunk_size):` loop (`gpu_engine.py:857`,
   2 chunks x 2 engines = 4 syncs/gen with `PHASE2_GPU_BATCH_SIZE=128` vs
   `PHASE2_POPULATION_SIZE=200`); `resolve_phase2_gpu_batch_size()` re-probes
   VRAM/RAM (incl. an `nvidia-smi` subprocess) on every single
   `simulate_rule_batch` call (`gpu_engine.py:816`) instead of once per engine.

GPU memory is nowhere near the limit during the run (`used=0.91 GiB` of
`vram=15.0 GiB`, from the log), confirming the GPU kernel itself is not the
bottleneck — the wall-clock time is dominated by host-side Python work and
redundant simulation passes.

## Files to Modify

1. `gpu_fuzzy_trader/backtest/gpu_engine.py` — gate/replace the CPU enrichment pass; move VRAM probe out of the hot path; consider fixed-size fast-reject masking.
2. `gpu_fuzzy_trader/config.py` — val-cadence flag semantics; batch size.
3. `gpu_fuzzy_trader/evolution/evox_runner.py` — fix `run_val_this_gen` in both `_run_nsga3` and `_run_nsga2_fallback` (factor into one shared helper to prevent future drift).
4. `gpu_fuzzy_trader/_gpu_runtime.py` — memoize VRAM/RAM probe.

## Detailed Changes

### R1: Replace CPU-engine symbol enrichment with a vectorized computation

Do not call `CPUBacktestEngine.simulate_rule_batch` (a full re-simulation) just
to get `per_symbol_metrics`. The GPU path already has `signals_batch` (which
rows each chromosome matched) and per-row PnL from `_get_trade_outcomes`/
`_simulate_signals_batch`. Compute per-symbol net PnL with a vectorized
groupby (e.g. `np.bincount(symbol_ids, weights=row_pnl * signals_batch[i])` per
chromosome, or a single `(n_symbols, n_rows)` one-hot matmul against the
outcome vector) instead of re-running the whole sequential backtest. This
preserves the C5 symbol-spread penalty signal while removing the dominant
runtime cost.

If a fully vectorized rewrite is too risky short-term, as an interim step:
- Add `PHASE2_ENRICH_SYMBOL_METRICS_EVERY_N_GENS` (default e.g. 5) and only
  run the CPU enrichment pass on that cadence (always on the final gen so the
  archive/pool build still has fresh data), OR
- Only enrich the current Pareto front / newly-improved individuals, not the
  full offspring batch, since `per_symbol_metrics` only affects a penalty term
  that changes slowly.

### R2: Make `PHASE2_VAL_SIM_INTERVAL` actually throttle when `JOINT_TRAIN_VAL=True`

Replace the `or`-chain in both `_run_nsga3` (`evox_runner.py:2216-2220`) and
`_run_nsga2_fallback` (`evox_runner.py:1691-1694`) with a single shared helper:

```python
def _should_run_val_this_gen(gen: int, is_last_gen: bool) -> bool:
    if is_last_gen:
        return True
    interval = max(1, int(_cfg.PHASE2_VAL_SIM_INTERVAL))
    return gen % interval == 0
```

`PHASE2_JOINT_TRAIN_VAL` should control *whether val feeds fitness*, not
*how often val is computed* — those are separate knobs today conflated by the
`or`. When val isn't computed this generation, reuse the last-known val
metrics from `metrics_cache`/`global_metrics_cache` for the joint-fitness
computation instead of skipping the joint logic outright (avoids reintroducing
train-only overfitting on skipped generations).

### R3: Widen `PHASE2_GPU_BATCH_SIZE` given confirmed VRAM headroom

`PHASE2_GPU_BATCH_SIZE = 128` while `PHASE2_POPULATION_SIZE = 200` forces 2
dispatch rounds/engine/generation. Log shows `used=0.91 GiB` of `vram=15.0
GiB` — ample headroom. Raise the manual default (e.g. to 256) or verify
`PHASE2_GPU_BATCH_SIZE_AUTO`'s VRAM tiers allow it, so one round covers the
full population per engine call. This does not fix R1-R3's shape/enrichment
costs by itself but removes a secondary multiplier.

### R4: Memoize the VRAM/RAM probe

`resolve_phase2_gpu_batch_size()` is called on every `simulate_rule_batch`
(`gpu_engine.py:816`) and re-probes `nvidia-smi`/`/proc/meminfo`
(`_gpu_runtime.py`). Cache the result at engine-construction time or behind a
module-level `functools.lru_cache`/one-shot flag; the answer cannot change
mid-run.

## Acceptance Criteria

- [ ] CPU "ground-truth" full re-simulation no longer runs unconditionally on every `simulate_rule_batch` call during evolution; `per_symbol_metrics` is either vectorized or throttled.
- [ ] `run_val_this_gen` logic is a single shared function used by both `_run_nsga3` and `_run_nsga2_fallback`; `PHASE2_VAL_SIM_INTERVAL` demonstrably throttles val simulation even when `PHASE2_JOINT_TRAIN_VAL=True`.
- [ ] `PHASE2_GPU_BATCH_SIZE` (or its AUTO tier) covers the full population in one dispatch round where VRAM allows.
- [ ] `resolve_phase2_gpu_batch_size()` (or its VRAM/RAM probe) is not re-executed on every batch call.
- [ ] Existing tests pass: `PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_gpu_engine.py tests/unit/test_evox_runner.py tests/property/test_gpu_engine_properties.py -x -q`
- [ ] New/updated test asserting `per_symbol_metrics` net_pnl sign/values match between old (CPU) and new (vectorized) computation on a small synthetic fixture.
- [ ] `evaluator_v5.ipynb` NOT modified.

## Verification

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_gpu_engine.py tests/unit/test_evox_runner.py -x -q
```

Do NOT run the full pipeline locally (OOM risk per AGENTS.md) — validate on the
next Colab run and compare per-generation `elapsed=` deltas in the log against
this run's 60-230s baseline.

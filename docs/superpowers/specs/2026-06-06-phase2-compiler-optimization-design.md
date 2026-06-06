# Design Spec: Phase 2 JAX Loop Unrolling Optimization

## Metadata
* **Date:** 2026-06-06
* **Status:** Draft (Pending User Approval)
* **Author:** Antigravity

## 1. Context & Problem Statement
During Phase 2 rule pool generation using `SPLIT_MODE = "purged_rolling_cv"` and `CV_FOLDS = 3`, the evolutionary search algorithm (NSGA-III) runs evaluations of 400 offspring chromosomes per generation on 6 backtest engines (3 train folds and 3 validation folds sequentially).
Under the default configuration:
* `PHASE2_SCAN_UNROLL = 16`
* `PHASE2_CV_FOLD_WORKERS = 1` (sequential evaluation)

Each generation takes approximately 390 seconds to complete, which leads to long overall run times (~11 hours for 100 generations). Setting `PHASE2_CV_FOLD_WORKERS = 4` parallelizes the concurrent execution of folds on the GPU from Generation 2 onwards, but further speedups can be achieved at the JAX compiler level without modifying the underlying search space, data rows, or population constraints.

## 2. Selected Approach
To optimize the GPU backtest runtime while keeping the dataset size (`PHASE1_SAMPLING_TOTAL = 701_000`) and search configuration (`PHASE2_POPULATION_SIZE = 400`, `PHASE2_GENERATIONS = 100`) completely unchanged, we will increase the JAX loop unrolling factor in the equity simulator from 16 to 32.

### How it Works
The GPU backtesting engine evaluates rule signal masks by walking chronologically through the price returns using `lax.scan`. Unrolling the scan loop:
* Fuses multiple loop steps (32 instead of 16) into single blocks within the generated GPU kernel code.
* Reduces kernel launch overhead on the host.
* Enables the XLA compiler to optimize memory loads, instruction scheduling, and register layout on the GPU.

### Trade-offs
* **Quality of Results:** Absolutely zero impact. This is purely a compiler-level instruction. The backtest simulations remain mathematically identical.
* **Compilation Time:** A higher unroll factor slightly increases the JIT compile time during Generation 1.
* **Memory Footprint:** VRAM usage increases minimally but remains well within the Colab T4 GPU capacity (15.0 GiB).

## 3. Implementation Plan
We will make the following contiguous change in [gpu_fuzzy_trader/config.py](file:///home/danaee/trading_platform/gpu_fuzzy_trader/config.py#L215):

```diff
-PHASE2_SCAN_UNROLL = 16
+PHASE2_SCAN_UNROLL = 32
```

## 4. Verification Plan
* Run standard unit tests using `pytest tests/unit/` to verify that all modules load properly.
* Run JAX-specific backtest tests to ensure there are no compilation errors or parity issues with the CPU engine.
* Track compilation memory and ensure zero runtime failures.

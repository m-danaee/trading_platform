# Nexus Knowledge Graph — trading_platform

- **Built**: 2026-08-01T18:31:58.877Z
- **Commit**: `8b028fdf6d248408b28215e8a4c59c467f8fe9de`
- **Nodes**: 148 | **Edges**: 316

## Language stats
- markdown: 4 files / 340 lines
- jupyter: 3 files / 2929 lines
- python: 141 files / 59603 lines

## Hub nodes (top by in-degree)

| file | in | out | lines |
|---|---|---|---|
| gpu_fuzzy_trader/__init__.py | 81 | 0 | 2 |
| gpu_fuzzy_trader/phases/phase2_rule_pool.py | 22 | 18 | 4098 |
| gpu_fuzzy_trader/backtest/cpu_engine.py | 19 | 5 | 1253 |
| gpu_fuzzy_trader/evolution/evox_runner.py | 17 | 8 | 3604 |
| gpu_fuzzy_trader/config.py | 12 | 0 | 2628 |
| gpu_fuzzy_trader/phases/phase2_sparse_encoding.py | 9 | 3 | 432 |
| gpu_fuzzy_trader/rb_governor.py | 9 | 7 | 2579 |
| gpu_fuzzy_trader/backtest/df_slim.py | 8 | 1 | 74 |
| gpu_fuzzy_trader/backtest/gpu_engine.py | 8 | 5 | 1223 |
| gpu_fuzzy_trader/output/writer.py | 8 | 2 | 470 |
| gpu_fuzzy_trader/phases/phase2_island_scheduler.py | 8 | 8 | 719 |
| gpu_fuzzy_trader/phases/phase2_stage.py | 8 | 1 | 210 |

## Focus files

### gpu_fuzzy_trader/backtest/cpu_engine.py
- in_degree=19, out_degree=5
- **imported_by**: gpu_fuzzy_trader/backtest/__init__.py, gpu_fuzzy_trader/backtest/condition_cache.py, gpu_fuzzy_trader/backtest/gpu_engine.py, gpu_fuzzy_trader/phases/phase2_island_scheduler.py, gpu_fuzzy_trader/phases/phase2_rule_pool.py, gpu_fuzzy_trader/phases/phase5_oos.py, gpu_fuzzy_trader/rb_governor.py, gpu_fuzzy_trader/validation/monthly_windows.py, tests/property/test_cpu_engine_properties.py, tests/property/test_gpu_engine_properties.py, tests/unit/test_condition_cache.py, tests/unit/test_cpu_engine.py, tests/unit/test_gpu_engine.py, tests/unit/test_gpu_rule_set_batch.py, tests/unit/test_jax_compat.py, tests/unit/test_phase2_use_gpu_flag.py, tests/unit/test_rb_governor_cv_folds.py, tests/unit/test_rb_governor_tail_holdout.py, tests/unit/test_release_indices.py
- **imports**: gpu_fuzzy_trader/__init__.py, gpu_fuzzy_trader/backtest/condition_cache.py, gpu_fuzzy_trader/backtest/gpu_engine.py, gpu_fuzzy_trader/backtest/symbol_conditions.py, gpu_fuzzy_trader/phases/phase2_sparse_encoding.py

### gpu_fuzzy_trader/backtest/gpu_engine.py
- in_degree=8, out_degree=5
- **imported_by**: gpu_fuzzy_trader/_gpu_runtime.py, gpu_fuzzy_trader/backtest/cpu_engine.py, gpu_fuzzy_trader/backtest/jax_compat.py, tests/benchmark/test_phase2_gpu_throughput.py, tests/property/test_gpu_engine_properties.py, tests/unit/test_gpu_engine.py, tests/unit/test_gpu_engine_jax_failure.py, tests/unit/test_gpu_rule_set_batch.py
- **imports**: gpu_fuzzy_trader/__init__.py, gpu_fuzzy_trader/_gpu_runtime.py, gpu_fuzzy_trader/_jax_env.py, gpu_fuzzy_trader/backtest/cpu_engine.py, gpu_fuzzy_trader/phases/phase2_sparse_encoding.py

### gpu_fuzzy_trader/config.py
- in_degree=12, out_degree=0
- **imported_by**: gpu_fuzzy_trader/backtest/df_slim.py, gpu_fuzzy_trader/data/labels.py, gpu_fuzzy_trader/data/loader.py, gpu_fuzzy_trader/data/splitter.py, tests/property/test_data_loader_properties.py, tests/property/test_data_splitter_properties.py, tests/property/test_phase5_oos_properties.py, tests/unit/test_crash_fix_and_run_logging.py, tests/unit/test_data_loader.py, tests/unit/test_data_splitter.py, tests/unit/test_df_slim.py, tests/unit/test_labels.py

### RUN.md
- in_degree=0, out_degree=0

### tests/benchmark/test_phase2_gpu_throughput.py
- in_degree=0, out_degree=4
- **imports**: gpu_fuzzy_trader/__init__.py, gpu_fuzzy_trader/_gpu_runtime.py, gpu_fuzzy_trader/backtest/gpu_engine.py, gpu_fuzzy_trader/backtest/jax_compat.py

### tests/benchmark/test_phase2_numba_warmup.py
- in_degree=0, out_degree=1
- **imports**: gpu_fuzzy_trader/evolution/numba_ops.py


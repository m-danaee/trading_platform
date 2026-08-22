# Integrity and Runtime Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Fail closed on stale exact-execution inputs, reject incomplete frozen strategies, and make runtime/benchmark reporting accurately reflect the production execution path.

**Architecture:** Add a versioned, deterministic barrier-cache identity derived from the execution horizon and complete TP/SL grid, then validate the cached schema before reuse. Tighten Phase 5 archive binding to full LWC multiset equality. Keep the existing CPU-exact dispatch, but make the benchmark construct exact barriers and explicitly report that route; independently harden startup ordering and GPU detection.

**Tech Stack:** Python 3.10+, NumPy, Pandas, Numba, JAX, Pytest, PyArrow.

**Spec:** \`docs/superpowers/specs/2026-08-22-integrity-and-runtime-hardening-design.md\`

## Global Constraints

- Run commands through \`/home/danaee/trading_platform/.venv\`; use \`PYTEST_LOW_MEMORY=1\` and \`MPLCONFIGDIR=/tmp/trading-platform-mpl\` for every pytest command.
- Run focused tests sequentially; never invoke the entire suite as one local command.
- Cache reads fail closed: stale, malformed, or incomplete caches rebuild exact barrier outcomes.
- Preserve research/OOS admission policy and user-owned files outside this isolated worktree.
- Do not rewrite history or force-push; that destructive part of the design is excluded until explicitly authorized.

---

### Task 1: Version the exact-barrier cache and validate its schema

**Files:**
- Modify: \`gpu_fuzzy_trader/backtest/barrier.py\`
- Modify: \`gpu_fuzzy_trader/data/loader.py\`
- Modify: \`tests/unit/test_data_loader.py\`

**Interfaces:**
- Produce \`BARRIER_CACHE_FORMAT_VERSION: str\`.
- Produce \`barrier_cache_identity(*, horizon: int, pairs: Iterable[tuple[float, float]] | None = None) -> str\`.
- Produce \`barrier_cache_filename(tape_hash: str, *, horizon: int, pairs: Iterable[tuple[float, float]] | None = None) -> str\`.
- Extend \`required_barrier_columns(pairs: Iterable[tuple[float, float]] | None = None) -> set[str]\`.

- [ ] **Step 1: Write failing tests**

~~~python
def test_barrier_cache_rebuilds_when_risk_grid_changes(tmp_path, monkeypatch):
    monkeypatch.setattr(_cfg, 'OUTPUTS_DIR', str(tmp_path / 'outputs'))
    monkeypatch.setattr(_cfg, 'RB_TP_GRID', ())
    monkeypatch.setattr(_cfg, 'RB_SL_GRID', ())
    csv_path = _write_raw_ohlcv_csv(tmp_path, rows=TAIL_DROP_ROWS + 4)
    first = Data_Loader().load_dataset(
        str(csv_path), drop_tail=False, include_barrier_outcomes=True,
    )
    monkeypatch.setattr(_cfg, 'RB_TP_GRID', (3.0,))
    monkeypatch.setattr(_cfg, 'RB_SL_GRID', (1.2,))
    refreshed = Data_Loader().load_dataset(
        str(csv_path), drop_tail=False, include_barrier_outcomes=True,
    )
    assert required_barrier_columns().issubset(refreshed.columns)
    assert set(c for c in refreshed if c.startswith('_barrier_')) != set(
        c for c in first if c.startswith('_barrier_')
    )
~~~

~~~python
def test_barrier_cache_rebuilds_when_cached_columns_are_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(_cfg, 'OUTPUTS_DIR', str(tmp_path / 'outputs'))
    csv_path = _write_raw_ohlcv_csv(tmp_path, rows=TAIL_DROP_ROWS + 4)
    Data_Loader().load_dataset(str(csv_path), drop_tail=False, include_barrier_outcomes=True)
    cache_file = next((tmp_path / 'outputs' / '.cache' / 'barriers').glob('*.parquet'))
    cached = pd.read_parquet(cache_file)
    cached.drop(columns=[cached.columns[0]]).to_parquet(cache_file)
    rebuilt = Data_Loader().load_dataset(
        str(csv_path), drop_tail=False, include_barrier_outcomes=True,
    )
    assert {c for c in rebuilt if c.startswith('_barrier_')} == required_barrier_columns()
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: \`PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_data_loader.py -k barrier_cache -q\`

Expected: FAIL because the old key ignores risk-grid configuration and the read accepts partial schemas.

- [ ] **Step 3: Implement the minimal cache contract**

~~~python
pairs = configured_barrier_pairs()
expected_columns = required_barrier_columns(pairs)
cache_file = cache_dir / barrier_cache_filename(
    tape_hash, horizon=TAIL_DROP_ROWS, pairs=pairs,
)
if cached is not None and len(cached) == len(df) and set(cached.columns) == expected_columns:
    for column in expected_columns:
        df[column] = cached[column].values
else:
    df = attach_barrier_outcomes(df, horizon=TAIL_DROP_ROWS, pairs=pairs)
~~~

Hash a canonical JSON identity containing the named format version, integer horizon, and sorted Cartesian TP/SL pairs. Write only the exact expected barrier columns.

- [ ] **Step 4: Run loader regression**

Run: \`PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_data_loader.py -q\`

Expected: PASS.

---

### Task 2: Require exact frozen LWC multiset equality

**Files:**
- Modify: \`gpu_fuzzy_trader/phases/phase5_oos.py\`
- Modify: \`tests/unit/test_mtf_audit_fixes.py\`

**Interfaces:**
- Candidate directional LWC hashes and archive directional LWC hashes are equal as \`collections.Counter\` values, including multiplicity.

- [ ] **Step 1: Write the failing missing-rule regression**

~~~python
save_mtf_rule_archive('lwc', [lwc_rule, second_lwc_rule], archive_path, metadata={'role': 'lwc'})
strategy_payload['rules_set'] = [lwc_rule]
strategy_payload['mtf_candidate']['lwc_rules'] = [lwc_rule]
strategy_path.write_text(json.dumps(strategy_payload), encoding='utf-8')

assert 'long' not in evaluator.load_strategies()
~~~

Keep the valid fixture exact by serializing both LWC rules.

- [ ] **Step 2: Run test to verify it fails**

Run: \`PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_mtf_audit_fixes.py -k strict_binding -q\`

Expected: FAIL because a candidate can currently omit a frozen LWC rule.

- [ ] **Step 3: Implement strict equality**

~~~python
candidate_counter = Counter(candidate_lwc_hashes)
archive_counter = Counter(archive_lwc_hashes)
if candidate_counter != archive_counter:
    raise ValidationError(
        'Candidate LWC rule multiset does not exactly match the frozen LWC archive payload'
    )
~~~

- [ ] **Step 4: Run Phase 5 regressions**

Run: \`PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_mtf_audit_fixes.py -k strict_binding -q\`

Run: \`PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_phase5_oos.py -q\`

Expected: PASS.

---

### Task 3: Make benchmarks truthful about the exact route

**Files:**
- Modify: \`scripts/benchmark_t4.py\`
- Modify: \`tests/unit/test_t4_profile.py\`

**Interfaces:**
- Produce \`run_exact_engine_benchmark(batch_size: int | None = None, n_samples: int = 1000) -> dict[str, Any]\`.
- Retain \`run_jax_engine_benchmark\` as a compatibility wrapper that delegates to \`run_exact_engine_benchmark\`.
- Successful exact-engine output includes \`execution_route: 'cpu_exact'\` and \`exact_barrier_pair\`.
- Evolution output includes \`dont_care_codes: [2, 5, 3]\`.

- [ ] **Step 1: Write failing contract tests**

~~~python
def test_exact_engine_benchmark_reports_cpu_exact_route():
    result = benchmark_t4.run_exact_engine_benchmark(
        batch_size=1, n_samples=TAIL_DROP_ROWS + 8,
    )
    if result['status'] == 'skipped':
        pytest.skip(result['reason'])
    assert result['execution_route'] == 'cpu_exact'
    assert result['exact_barrier_pair'] == [_cfg.PHASE2_TP, _cfg.PHASE2_SL]

def test_evolution_benchmark_reports_production_dont_cares():
    result = benchmark_t4.run_evolution_benchmark(
        generations=0, pop_size=2, n_samples=TAIL_DROP_ROWS + 8,
    )
    assert result['dont_care_codes'] == [2, 5, 3]

def test_loader_benchmark_reports_exact_barriers(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_t4._cfg, 'OUTPUTS_DIR', str(tmp_path / 'outputs'))
    result = benchmark_t4.run_data_loader_benchmark(
        n_samples=4 * (TAIL_DROP_ROWS + 4),
    )
    assert result['barrier_column_count'] == len(required_barrier_columns())
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: \`PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_t4_profile.py -q\`

Expected: FAIL because the current benchmark reports no route and uses \`[0, 0, 0]\`.

- [ ] **Step 3: Build exact benchmark data and report its route**

~~~python
raw = _synthetic_ohlcv_frame(n_samples)
labeled = raw.merge(compute_labels(raw), on=['datetime', 'symbol'], validate='one_to_one')
exact = attach_barrier_outcomes(
    labeled,
    horizon=_cfg.TAIL_DROP_ROWS,
    pairs=[(_cfg.PHASE2_TP, _cfg.PHASE2_SL)],
)
exact = exact.dropna(subset=LABEL_COLUMNS).reset_index(drop=True)
exact['_symbol_bar_index'] = np.arange(len(exact))
~~~

Use \`_get_dont_cares(feature_infos)\` instead of hard-coded values. Make the loader benchmark request \`include_barrier_outcomes=True\` and report its barrier-column count.

- [ ] **Step 4: Re-run profile tests and a small dry run**

Run: \`PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_t4_profile.py -q\`

Run: \`/home/danaee/trading_platform/.venv/bin/python scripts/benchmark_t4.py --dry-run --component gpu --rows 400 --pop 2 --generations 0\`

Expected: PASS; output either reports \`cpu_exact\` or truthfully skips unavailable JAX hardware.

---

### Task 4: Initialize JAX configuration early and tighten T4 detection

**Files:**
- Modify: \`gpu_fuzzy_trader/run_pipeline.py\`
- Modify: \`gpu_fuzzy_trader/_gpu_runtime.py\`
- Modify: \`tests/unit/test_t4_profile.py\`

**Interfaces:**
- \`configure_jax_env()\` executes after standard-library imports and before NumPy, Pandas, and other project imports in \`run_pipeline\`.
- The VRAM fallback in \`is_t4_runtime()\` runs only when \`detect_gpu_name()\` returns \`None\`.

- [ ] **Step 1: Write failing regressions**

~~~python
def test_is_t4_runtime_rejects_known_non_t4_with_16_gib():
    _gpu_runtime.is_t4_runtime.cache_clear()
    with patch.dict('os.environ', {}, clear=True), \
         patch('gpu_fuzzy_trader._gpu_runtime.detect_gpu_name', return_value='NVIDIA RTX A4000'), \
         patch('gpu_fuzzy_trader._gpu_runtime.detect_gpu_vram_gb', return_value=16.0):
        assert _gpu_runtime.is_t4_runtime() is False
~~~

~~~python
_IMPORT_GUARD_SCRIPT = '''
import builtins
import os
for key in ('NUMBA_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
            'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.pop(key, None)
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.', 1)[0] in {'numpy', 'pandas'}:
        assert os.environ.get('OPENBLAS_NUM_THREADS')
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
import gpu_fuzzy_trader.run_pipeline
'''
assert subprocess.run(
    [sys.executable, '-c', _IMPORT_GUARD_SCRIPT],
    cwd=PROJECT_ROOT, capture_output=True, text=True,
).returncode == 0
~~~

- [ ] **Step 2: Run test to verify it fails**

Run: \`PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_t4_profile.py -q\`

Expected: FAIL on the non-T4 fixture and import-order guard.

- [ ] **Step 3: Implement startup ordering and fallback gate**

~~~python
from gpu_fuzzy_trader._jax_env import configure_jax_env
configure_jax_env()

import numpy as np
import pandas as pd
# project imports follow
~~~

~~~python
if gpu_name is None:
    vram = detect_gpu_vram_gb()
    if vram is not None and 14.5 <= vram <= 16.5:
        return True
~~~

- [ ] **Step 4: Run focused runtime regressions**

Run: \`PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_t4_profile.py tests/unit/test_jax_compat.py tests/unit/test_run_pipeline.py -q\`

Expected: PASS.

---

### Task 5: Align the environment and remove confirmed whitespace defects

**Files:**
- Modify whitespace only: \`.opencode/plans/PLAN.md\`, \`gpu_fuzzy_trader/config.py\`, \`gpu_fuzzy_trader/features/fuzzy_scaling.py\`, \`tests/unit/test_cpu_engine.py\`, \`tests/unit/test_mtf_pipeline_integration.py\`, \`tests/unit/test_t4_profile.py\`.

**Interfaces:**
- The declared dependency remains \`pandas>=2.0,<3\`.
- The local virtual environment reports Pandas 2.x and \`pip check\` succeeds.

- [ ] **Step 1: Inspect the mismatch**

Run: \`/home/danaee/trading_platform/.venv/bin/python -c "import pandas; print(pandas.__version__)"\`

Run: \`/home/danaee/trading_platform/.venv/bin/python -m pip check\`

- [ ] **Step 2: Synchronize only the declared Pandas range**

Run: \`/home/danaee/trading_platform/.venv/bin/python -m pip install 'pandas>=2.0,<3'\`

If download is sandbox-blocked, request scoped approval for this exact package operation.

- [ ] **Step 3: Remove only reported trailing whitespace**

Remove the exact \`git diff --check 0d30b98..HEAD\` findings; do not perform a whole-project formatter sweep.

- [ ] **Step 4: Verify hygiene**

Run: \`/home/danaee/trading_platform/.venv/bin/python -c "import pandas; assert pandas.__version__.split('.')[0] == '2'; print(pandas.__version__)"\`

Run: \`/home/danaee/trading_platform/.venv/bin/python -m pip check\`

Run: \`git diff --check 0d30b98..HEAD\`

Expected: Pandas 2.x, no broken requirements, and no reported whitespace errors.

---

### Task 6: Sequential regression verification

**Files:**
- Verify only; no planned source changes.

- [ ] **Step 1: Run each focused suite sequentially**

~~~bash
PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_data_loader.py -q
PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_mtf_audit_fixes.py -q
PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_phase5_oos.py -q
PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_t4_profile.py tests/unit/test_jax_compat.py tests/unit/test_run_pipeline.py -q
PYTEST_LOW_MEMORY=1 MPLCONFIGDIR=/tmp/trading-platform-mpl /home/danaee/trading_platform/.venv/bin/pytest tests/unit/test_cpu_engine.py tests/unit/test_gpu_engine.py -q
~~~

- [ ] **Step 2: Compile and inspect the worktree**

Run: \`/home/danaee/trading_platform/.venv/bin/python -m compileall -q gpu_fuzzy_trader scripts\`

Run: \`git diff --check\`

Run: \`git status --short\`

- [ ] **Step 3: Report bounded evidence**

Report focused test counts, Pandas version, and any hardware skips separately from genuine CUDA or forward/OOS acceptance.

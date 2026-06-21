# Task 5: Create `friend.ipynb` Colab Notebook

## Goal
Create a Colab-ready Jupyter notebook (`friend.ipynb`) modeled after the main project's `main.ipynb` at `/home/danaee/trading_platform/main.ipynb`. The notebook boots the friend_project on Colab T4 GPU and runs the full pipeline with per-symbol Phase 2 + per-symbol RB Governor + Phase 5 filtering.

## Target File
- **`friend_project/friend.ipynb`** (NEW)

## Notebook Structure

### Cell 1 — Markdown (instructions)
```markdown
# GPU-Fuzzy Trading Pipeline — Friend Project (Colab)

Per-symbol Phase 2 training + RB Governor + Phase 5 rule filtering.

Before running:
- Connect to a Colab T4 GPU server
- Upload the `friend_project/` folder or clone from GitHub
- Place `train.csv` and `test.csv` in Google Drive or the project `data/` folder
```

### Cell 2 — GitHub Token (optional)
```python
import os
from getpass import getpass

github_token = os.environ.get("GITHUB_TOKEN", "").strip()
if not github_token:
    github_token = getpass("GitHub classic PAT (optional, press Enter to skip): ").strip()
if github_token:
    os.environ["GITHUB_TOKEN"] = github_token
```

### Cell 3 — Bootstrap (project discovery, Drive mount, dataset)
Copy and adapt from `main.ipynb` Cell 2 (the bootstrap cell), adjusting:
- `DEFAULT_GITHUB_REPO_URL` → point to friend_project repo (or leave configurable)
- `GITHUB_CLONE_DIR` → `/content/friend_project`
- Project root discovery: look for `gpu_fuzzy_trader/` and `requirements.txt`
- `_is_project_root()`: check for `(path / "gpu_fuzzy_trader").is_dir()`
- `_repair_stale_numba_ops()`: check for numba_ops.py existence (may not exist in friend_project — handle gracefully)
- Mount Google Drive
- Discover `train.csv` and `test.csv`
- Stage CSVs to local disk (for faster I/O)
- Set `PROJECT_ROOT`, `sys.path`, `PYTHONPATH`, `os.chdir()`

Key differences from main.ipynb bootstrap:
- The friend_project may not have `gpu_fuzzy_trader/evolution/numba_ops.py` — make `_repair_stale_numba_ops()` a no-op or skip gracefully
- Use `/content/friend_project` as default clone dir
- Repo URL should be for friend_project (or configurable)

### Cell 4 — Install dependencies
```python
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

PIP = [sys.executable, "-m", "pip", "install", "-q"]

_in_colab = (
    os.environ.get("COLAB_RELEASE_TAG") is not None
    or Path("/content").is_dir()
    or importlib.util.find_spec("google.colab") is not None
)

COLAB_CORE_PINS = [
    "numpy>=1.26,<2.4",
    "pandas>=2.0,<3",
    "numba>=0.58,<0.66",
    "scikit-learn>=1.3,<2",
    "matplotlib>=3.7,<4",
    "pyarrow>=14.0",
    "optuna>=3.5.0,<5",
]
COLAB_EVOX = "evox>=1.0.0,<2"
COLAB_JAX = "jax[cuda12]==0.10.1"
COLAB_TORCH_CPU = (
    "torch==2.5.1",
    "--index-url",
    "https://download.pytorch.org/whl/cpu",
)

if _in_colab:
    print("Colab T4: installing pinned GPU stack (CUDA 12 + JAX 0.10.1 + EvoX)...")
    subprocess.check_call(PIP + ["-U", "pip", "wheel"])
    subprocess.check_call(PIP + COLAB_CORE_PINS)
    subprocess.check_call(PIP + list(COLAB_TORCH_CPU))
    subprocess.check_call(PIP + [COLAB_EVOX])
    subprocess.check_call(PIP + ["-U", COLAB_JAX])

    import jax
    print(f"JAX {jax.__version__} | backend={jax.default_backend()} | devices={jax.devices()}")
    if jax.default_backend() != "gpu":
        raise RuntimeError(
            "JAX is not using the GPU. Choose Runtime > Change runtime type > "
            "T4 GPU, then Runtime > Restart session and rerun from Cell 1."
        )
else:
    print("Installing dependencies from requirements.txt ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
```

### Cell 5 — Import and configure
```python
import importlib.metadata as _im
import os
import jax
from pathlib import Path

# RAM optimisation for Colab
os.environ["PHASE2_GPU_BATCH_SIZE"] = "64"
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")

from gpu_fuzzy_trader._jax_env import configure_jax_env
configure_jax_env()

from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator
from gpu_fuzzy_trader import config as cfg

# Enable per-symbol mode (already default in config, but ensure)
cfg.PER_SYMBOL_PHASE2 = True

# Monkeypatch scan unroll for Colab RAM
cfg.PHASE2_SCAN_UNROLL = 16

print("Import OK")
print(f"TRAIN_CSV_PATH: {Path(cfg.TRAIN_CSV_PATH).resolve()}")
print(f"TEST_CSV_PATH: {Path(cfg.TEST_CSV_PATH).resolve()}")
print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}")
print(f"Per-symbol Phase 2: {cfg.PER_SYMBOL_PHASE2}")
print(f"Phase 5 filtering: {cfg.PHASE5_REMOVE_NEGATIVE_PNL_RULES}")

for _pkg in ("evox", "optuna", "numba", "torch", "jax"):
    try:
        print(f"{_pkg}: {_im.version(_pkg)}")
    except _im.PackageNotFoundError:
        print(f"{_pkg}: not installed")

from gpu_fuzzy_trader._gpu_runtime import (
    detect_gpu_vram_gb,
    resolve_phase2_gpu_batch_size,
)

_vram = detect_gpu_vram_gb()
_vram_s = f"{_vram:.1f} GiB" if _vram is not None else "unknown"
print(f"Phase 2 GPU: batch_size={resolve_phase2_gpu_batch_size()} | scan_unroll={cfg.PHASE2_SCAN_UNROLL} | vram={_vram_s}")
```

### Cell 6 — Run pipeline
```python
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import jax

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.70")

RUN_PHASE = None  # None = full pipeline, or 1..5 for single phase
RESUME = False
USE_LOCAL_SCRATCH = True
LOCAL_OUTPUT_DIR = Path("/content/friend_project_outputs")
DRIVE_OUTPUT_DIR = Path("/content/drive/MyDrive/friend_project/outputs")
OUTPUT_DIR = str(LOCAL_OUTPUT_DIR if USE_LOCAL_SCRATCH else DRIVE_OUTPUT_DIR)

print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}", flush=True)

if USE_LOCAL_SCRATCH:
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if RESUME and DRIVE_OUTPUT_DIR.is_dir():
        shutil.copytree(DRIVE_OUTPUT_DIR, LOCAL_OUTPUT_DIR, dirs_exist_ok=True)
        print(f"Resume: copied prior outputs from Drive -> {LOCAL_OUTPUT_DIR}", flush=True)

pipeline_cmd = [
    sys.executable, "-u", "-m", "gpu_fuzzy_trader.run_pipeline",
    "--output", OUTPUT_DIR,
]
if RUN_PHASE is not None:
    pipeline_cmd.extend(["--phase", str(RUN_PHASE)])
if RESUME:
    pipeline_cmd.append("--resume")

print("Running:", " ".join(pipeline_cmd), flush=True)
t0 = time.perf_counter()

result = subprocess.run(pipeline_cmd, cwd=str(PROJECT_ROOT))
elapsed = time.perf_counter() - t0
print(f"Pipeline finished in {elapsed/60:.1f} min with code {result.returncode}", flush=True)

# Sync to Drive
if USE_LOCAL_SCRATCH and DRIVE_OUTPUT_DIR.parent.parent.exists():
    print("Syncing outputs to Google Drive...", flush=True)
    DRIVE_OUTPUT_DIR.parent.mkdir(parents=True, exist_ok=True)
    if DRIVE_OUTPUT_DIR.exists():
        shutil.rmtree(DRIVE_OUTPUT_DIR)
    shutil.copytree(LOCAL_OUTPUT_DIR, DRIVE_OUTPUT_DIR)
    
    # Also sync per-symbol pools
    POOLS_SRC = Path(PROJECT_ROOT) / "pools" / "per_symbol"
    POOLS_DST = Path("/content/drive/MyDrive/friend_project/pools/per_symbol")
    if POOLS_SRC.exists():
        POOLS_DST.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(POOLS_SRC, POOLS_DST, dirs_exist_ok=True)
    
    print(f"Outputs synced to {DRIVE_OUTPUT_DIR}", flush=True)
```

### Cell 7 — Markdown (results)
```
After the pipeline finishes:
- Per-symbol pools → pools/per_symbol/
- Per-symbol RB Governor outputs → outputs/per_symbol/{symbol}/
- Final merged strategy → outputs/long.json, outputs/short.json
- Phase 5 filtered strategy → outputs/long.json (rewritten), outputs/short.json (rewritten)
- Reports → outputs/reports/
- Drive sync → MyDrive/friend_project/outputs/
```

## Key Differences from `main.ipynb`

| Aspect | main.ipynb | friend.ipynb |
|--------|-----------|--------------|
| Project dir | `/content/trading_platform` | `/content/friend_project` |
| Repo URL | main project repo | friend_project repo (configurable) |
| Per-symbol | Not used | `cfg.PER_SYMBOL_PHASE2 = True` |
| Phase 5 filtering | Built-in | Newly added (PHASE5_REMOVE_NEGATIVE_PNL_RULES) |
| numba_ops repair | Required | Graceful skip (may not exist) |
| Drive output | `trading_platform/outputs` | `friend_project/outputs` |
| Pool sync | phase2_rule_archive | pools/per_symbol/ |

## Acceptance Criteria
- [ ] Notebook has 7 cells: 1 markdown, 2 GitHub token, 3 bootstrap, 4 install, 5 import/config, 6 run, 7 markdown results
- [ ] Bootstrap cell discovers project root, mounts Drive, finds CSVs
- [ ] Install cell installs JAX CUDA 12, EvoX, Torch CPU on Colab
- [ ] Import cell sets `PER_SYMBOL_PHASE2=True`, configures JAX, reports GPU info
- [ ] Run cell executes pipeline with `--output` to local scratch, syncs to Drive
- [ ] `_repair_stale_numba_ops()` handles missing `numba_ops.py` gracefully
- [ ] All paths reference friend_project (not main project)
- [ ] Notebook JSON is valid Jupyter format (nbformat 4, nbformat_minor 5)

## Dependencies
- All previous tasks (1-4) — the notebook runs the completed pipeline

## Handoff
Write `.opencode/handoffs/task-5-implementer.json` on completion.

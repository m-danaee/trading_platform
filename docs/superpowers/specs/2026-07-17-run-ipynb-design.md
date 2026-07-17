# Design: `run.ipynb` (local WSL runner)

**Date:** 2026-07-17  
**Status:** Approved  
**Environment:** Local Jupyter / WSL with project `.venv`

## Goal

Provide a root-level notebook that runs the **full** GPU-Fuzzy pipeline the same way as:

```bash
python -m gpu_fuzzy_trader.run_pipeline
```

## Constraints

- Primary runtime: local Jupyter (WSL), not Colab.
- Data: only `data/train.csv` (2024-01-01 → 2024-08-31) and `data/test.csv` (2024-09-01 → 2025-01-31).
- Do not run the full pipeline during agent verification (OOM risk on WSL).
- No Colab install/Drive cells; no phase-picker UI; no hyperparam editor.

## Notebook layout

1. Markdown — purpose, datasets/date ranges, WSL OOM warning.
2. Setup — chdir to repo root, import `Pipeline_Orchestrator`.
3. Data check — assert CSVs exist; print datetime min/max.
4. Knobs — `OUTPUT_DIR = "outputs"`, `FORCE = True` (CLI default is full rerun).
5. Run — `Pipeline_Orchestrator(output_dir=OUTPUT_DIR).run(force=FORCE)` + summary.
6. Markdown — output paths; next step `evaluator_v5.ipynb`.

## Success criteria

- Notebook exists at repo root as `run.ipynb`.
- Running cells uses the same orchestrator API as the CLI.
- Comments/docs reference only `train.csv` / `test.csv` and the real date spans.

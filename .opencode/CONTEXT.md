# Context

- **Active objective:** Fix Phase 2 island-mode elite erosion & migration degradation
  (diagnosed from 2026-06-25 run log: migration regresses locally-adapted elites;
  recomputed dynamic penalties + no island early-stop erode mid-epoch elites).
- **Plan:** `.opencode/plans/PLAN.md` (3 tasks: migration safety, elite preservation, island early-stop)
- **Current phase:** Planning complete — awaiting user go-ahead to start execution.
- **base_branch:** `main`
- **execution_mode:** checkpoint
- **branch_policy:** isolated
- **All tasks merged to main:** ⏳ (not started)
- **Feature branches deleted:** ⏳
- **Tests:** pending (existing suite 67/67; new unit tests to be added per task)
- **Constraints:** `.venv` for all commands; `PYTEST_LOW_MEMORY=1` for local tests;
  do NOT run the pipeline locally/WSL (OOM — Colab GPU only); do not modify
  `evaluator_v5.ipynb`.
- **Next action:** User confirms → orchestrator runs pre-dispatch isolation
  validation on `main`, then dispatches implementer for task-1 on branch
  `fix/migration-safety`. Awaits "continue task 1" signal (checkpoint mode).
# friend_project — run & test

Use the repo-root virtualenv (`.venv` in `trading_platform/`), not a separate env inside `friend_project/`.

## Tests

On local / WSL, **always** set `PYTEST_LOW_MEMORY=1` — running the full suite without it can OOM.

From the `friend_project/` directory:

```bash
cd /path/to/trading_platform/friend_project
export PYTEST_LOW_MEMORY=1
export PYTHONPATH="$(pwd)"

# All friend_project tests
../.venv/bin/python -m pytest tests/ -q

# Single module (recommended while developing)
../.venv/bin/python -m pytest tests/test_phase2_stage.py -q
../.venv/bin/python -m pytest tests/test_merge_strategies.py -q
../.venv/bin/python -m pytest tests/test_phase5_rule_filtering.py -q
../.venv/bin/python -m pytest tests/test_governor_contract.py -q
../.venv/bin/python -m pytest tests/test_evaluator_v5_symbols.py -q
../.venv/bin/python -m pytest tests/test_config_contract.py -q
```

One-liner (same as above):

```bash
cd friend_project && PYTEST_LOW_MEMORY=1 PYTHONPATH=. ../.venv/bin/python -m pytest tests/ -q
```

**Do not** run the full pipeline or heavy integration runs on WSL for smoke checks — use unit tests only. Full runs are intended for Colab GPU (`friend.ipynb`).

### Test modules

| File | Focus |
| --- | --- |
| `tests/test_phase2_stage.py` | Phase 2 two-stage hyperparams (Stage A / B) |
| `tests/test_merge_strategies.py` | Per-symbol RB Governor merge |
| `tests/test_phase5_rule_filtering.py` | OOS negative-PnL rule pruning |
| `tests/test_governor_contract.py` | RB Governor output contract |
| `tests/test_evaluator_v5_symbols.py` | Evaluator symbol handling |
| `tests/test_config_contract.py` | Config invariants |

Rule-set behaviour is validated against `evaluator_v5.ipynb` on Colab; do not change that notebook.

## Pipeline (Colab / GPU host)

Long-running search example (adjust paths for your deployment):

```bash
cd /path/to/trading_platform/friend_project
mkdir -p logs
nohup env PYTHONPATH="$(pwd)" PYTEST_LOW_MEMORY=1 \
  ../.venv/bin/python -m gpu_fuzzy_trader.auto_search \
  --hours 36 \
  --output-root outputs/trader_bigdata_governor_7h \
  --start-direction long \
  > logs/trader_bigdata_governor_7h.log 2>&1 &
```

Colab: open `friend.ipynb`, set `cfg.PHASE2_TWO_STAGE_ENABLED` and `cfg.PER_SYMBOL_PHASE2` as needed, then run the pipeline cell.

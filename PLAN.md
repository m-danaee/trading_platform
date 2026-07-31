# Certificate-first multi-symbol selection

## Summary

- Treat this as a valid fail-closed run, not a CUDA/JAX failure.
- Preserve the current concentration limits, tail-holdout gate, OOS isolation, and empty-strategy behavior.
- Target a balanced Mode-A team: BTC and ETH must each make supported, positive validation PnL contributions before deployment.

## Key changes

- Make Phase 2 re-evaluate every final archive candidate on CPU for train and validation before pool admission, rather than trusting cached GPU metrics that may lack per-symbol evidence.
- Store per-symbol train/validation metrics in pool entries; add a coverage report showing eligible rules, positive contributors per symbol, and why BTC candidates were rejected.
- Reserve up to 10 genuinely eligible pool candidates per positively contributing symbol before filling the remaining pool by the existing deployability rank. Never retain a failing rule merely to fill a quota.
- Add an RB symbol-contribution certificate: require at least `RB_MIN_DISTINCT_SYMBOLS` symbols with positive validation net PnL and at least 6 validation trades each, alongside the existing 0.55 share and 0.60 HHI limits.
- Replace single-seed greedy composition with a bounded diversification search: shortlist global leaders plus per-symbol leaders, keep a six-state beam through four diversification steps, then continue the existing score-based growth only after the certificate passes.
- Apply the same certificate to every risk-grid and profit-amplifier trial so later capital/risk changes cannot recreate ETH dominance.
- Keep tail holdout report-only during selection and fail closed afterward; record all rejected portfolio certificates in the RB report.
- Explicitly set pandas `groupby(..., observed=False)` at the warned call sites to preserve current semantics and remove the noisy FutureWarnings.

## Test plan

- Synthetic ETH-best/BTC-needed case: confirm RB chooses a balanced positive team instead of an ETH-only high-score seed.
- No viable BTC case: confirm the strategy remains empty with an explicit contribution failure reason.
- Verify CPU re-evaluation overrides unavailable/stale GPU per-symbol metrics and pool reservations never bypass admission gates.
- Verify risk optimization and profit amplification cannot break an already certified portfolio.
- Retain existing concentration, tail-holdout, stale-output, and Phase 5 skip regressions.

Run only focused tests with `PYTEST_LOW_MEMORY=1` and `.venv`; do not run the full GPU pipeline locally.

## Assumptions

- The unanswered preference defaults to the repository’s current balanced multi-symbol team mode, not specialist sleeves.
- `evaluator_v5.ipynb` and held-out test/OOS logic remain unchanged.
- If BTC has no genuinely positive, supported validation candidate after the new diagnostics, an empty result remains the correct outcome.

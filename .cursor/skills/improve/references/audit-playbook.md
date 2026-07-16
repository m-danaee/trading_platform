# Audit Playbook

What to look for, per category. Adapt depth to repo size.

A finding is only a finding with evidence. `orders/api.ts:142 issues one query per order item inside a loop` is a finding; "probably has N+1 somewhere" is not.

---

## 1. Correctness / Bugs

- Error handling: swallowed exceptions, empty catch blocks, missing error states.
- Async hazards: unawaited promises, race conditions, missing cleanup.
- Null/undefined flows: unchecked indexing, optional chaining hiding required values.
- Boundary conditions: off-by-one, empty collections, timezone assumptions.
- State machines: unhandled enum branches, impossible states in types.
- Concurrency: check-then-act, missing transactions, idempotency of retried ops.
- Type escape hatches: `any`, `cast`, `# type: ignore` clusters.
- Resource leaks: unclosed handles, connections, subscriptions.

**Trading-platform specifics:** phase boundary violations (e.g. `test_2.csv` used before Phase 5), train/val leakage across `SPLIT_MODE`, inconsistent fitness vs admission metrics, GPU/CPU path divergence in `gpu_fuzzy_trader/`.

## 2. Security

Never copy secret values into findings — `file:line` and credential type only; recommend rotation.

- Credential hygiene: hardcoded keys, committed `.env`, secrets in logs.
- Injection: SQL/shell from user data, path traversal, XSS where applicable.
- Access control: missing server-side auth, IDOR, CSRF on state-changing routes.
- Input contracts: unvalidated API bodies, unsafe file uploads.
- Dependencies: `pip-audit` / advisory scan for critical/high on reachable code.
- Production config: overly broad CORS, missing security headers, debug in prod.
- Data minimization: PII in logs or client-facing errors.

## 3. Performance

- N+1 patterns, wrong complexity, caching gaps.
- Payload size: over-fetching, missing pagination.
- GPU pipeline: redundant host↔device transfers, missing batching in phase2/phase3.
- Build/CI: slow tests, missing cache, redundant steps.

**Trading-platform specifics:** JAX memory pressure, Numba warmup gaps, full test suite OOM on WSL, phase2 population size vs GPU throughput.

## 4. Test Coverage

- Map critical paths (money, auth, data mutation, core pipeline) vs coverage.
- High churn + no tests = characterization-test candidates.
- Test quality: meaningless assertions, over-mocking, flaky patterns.
- Missing layers: unit vs integration on API boundaries.
- Verification baseline: one-command way to know the codebase works.

**Trading-platform specifics:** property tests in `tests/property/`, evaluator parity with `evaluator_v5.ipynb`, phase-specific unit tests under `tests/unit/`.

## 5. Tech Debt & Architecture

- Duplication, layering violations, circular imports.
- Dead code, god modules, inconsistent patterns.
- Abstraction mismatches: premature or missing abstractions.

**Trading-platform specifics:** RB Governor vs legacy Phase 3+4 paths, duplicated scoring logic, config knob sprawl in `config.py`.

## 6. Dependencies & Migrations

- Major-version lag on core deps (JAX, NumPy, Optuna, etc.).
- Deprecated APIs with removal timelines.
- Abandoned packages on critical paths.
- Duplicate deps solving the same problem.

## 7. DX & Tooling

- Missing typecheck/lint/format, slow feedback loops.
- Onboarding friction: wrong README steps, undocumented env vars.
- Missing or thin `AGENTS.md`.
- Error messages/logging quality.

## 8. Docs

- Public API without reference docs.
- Stale docs worse than missing (wrong setup, broken examples).
- Undocumented config knobs with cross-phase interactions.

## 9. Direction

Ground every suggestion in repo evidence — TODO clusters, unfinished flags, README promises without code, surface asymmetries, adjacent possible features, friction users work around manually.

Direction findings: Impact = product/user value; plans are usually design/spike plans, not build-everything.

---

## Finding format

```markdown
### [CATEGORY-NN] Short imperative title

- **Evidence**: `path/file.py:123` — one-sentence description.
- **Impact**: Concrete cost or failure mode.
- **Effort**: S / M / L
- **Risk**: LOW/MED/HIGH + one line why.
- **Confidence**: HIGH / MED / LOW
- **Fix sketch**: 1–3 sentences.
```

## Prioritization rubric

Order by **leverage = impact ÷ effort**, discounted by confidence and fix-risk.

1. Unblockers (verification baseline, characterization tests) float up.
2. HIGH-confidence security above equivalent non-security.
3. Prefer findings with clean verification stories.
4. "Not worth doing" is valid — record reasoning.

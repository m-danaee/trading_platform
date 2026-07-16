# Closing the Loop — execute, reconcile, issues

The advisor never edits source code. In `execute`, a separate executor subagent edits in an isolated git worktree; the advisor reviews and renders a verdict.

---

## `execute <plan>` — dispatch and review

### Preconditions

- Git repository with worktree support.
- Plan exists; dependencies DONE in `plans/README.md`.
- Run drift check; reconcile stale plans first.

### Dispatch

Spawn one `generalPurpose` subagent with isolation via git worktree. Inline the **full plan text** in the prompt.

Executor preamble: follow plan step by step; touch only in-scope files; on STOP, report; skip updating `plans/README.md` (reviewer maintains index).

Report format:

```
STATUS: COMPLETE | STOPPED
STEPS: per step — done/skipped + verification result
STOPPED BECAUSE: (if STOPPED)
FILES CHANGED: list
NOTES: deviations, surprises
```

### Review

1. Re-run every done criterion in the worktree.
2. Scope compliance: no files outside in-scope list.
3. Read full diff against "Why this matters" and repo conventions.
4. Audit new tests — meaningful assertions, not gaming criteria.
5. Confirm codelookup blast radius was addressed.

### Verdict

| Verdict | When                                          | Action                                        |
| ------- | --------------------------------------------- | --------------------------------------------- |
| APPROVE | Criteria pass, scope clean                    | Mark DONE. Present diff summary. User merges. |
| REVISE  | Fixable gaps                                  | Feedback to executor. Max 2 rounds.           |
| BLOCK   | STOP hit, scope violated, revisions exhausted | Mark BLOCKED; refine plan.                    |

Fresh worktrees lack `.venv` — executor installs deps first.

---

## `reconcile`

Per plan status in `plans/README.md`:

- **DONE** — spot-check done criteria on HEAD.
- **BLOCKED** — investigate; rewrite or REJECT.
- **IN PROGRESS** (stale) — flag to user.
- **TODO** — drift check; refresh excerpts or REJECT if fixed.

Report: verified done, refreshed, rejected, executable now.

---

## `--issues`

Only with explicit flag. Check `gh auth status`, remote, repo visibility. Warn on public repos for security findings. Create issues with `gh issue create --body-file`. Record URLs in plan Status and index.

# Multi-Timeframe Wave-Cycle Trend Following

## Objective

Add a causal HWC/MWC/LWC trading hierarchy to improve out-of-sample behavior:

```text
4h HWC regime
    +
1h MWC regime
    -> fixed directional permission
    +
15m LWC pullback-reversal trigger
    -> NSGA-III entry confirmations
    -> existing RB Governor composition and sizing
    -> locked validation, diagnostic test, and untouched forward OOS
```

The first release is trend-following only. It must not trade against the HWC
direction and must block Range and Noisy conditions.

## Frozen Trading Policy

The first implementation uses:

- HWC = 4h.
- MWC = 1h.
- LWC = 15m.
- CSV timestamps represent 15m bar-open times.
- Long entries require HWC Bullish and (MWC Bullish or MWC Range).
- Short entries require HWC Bearish and (MWC Bearish or MWC Range).
  A neutral MWC consolidation is a valid continuation while HWC retains
  direction (`CONTEXT_ALLOW_MWC_RANGE_PERMISSION`); MWC Noisy still blocks.
- Noisy, opposite, or unavailable HWC/MWC states block entries.
- LWC must show a pullback followed by a directional reversal (any opposite
  LWC print in the previous 24 bars counts by default). Optional stricter
  mode (`CONTEXT_REQUIRE_PERMISSION_ON_PULLBACK_PRINT=True`) also requires
  that historical print to have occurred while same-direction permission was
  already active. Tradeable entries still require current-row permission AND
  trigger.
- Existing RB Governor capital sizing remains unchanged.
- HWC reversal trades are deferred.
- Maximum holding period is 96 15m bars, or 24 hours.

Do not run separate Phase 2 or RB searches for each timeframe. Higher
timeframes define permission; NSGA-III searches only low-timeframe entry
confirmations.

## Four-State Regime Model

Add a deterministic, auditable classifier for Bullish, Bearish, Range, and
Noisy states. Thresholds are fitted once from the Phase-0 training prefix and
then frozen for validation, test, and forward evaluation.

Use these structural inputs on completed bars:

- Signed price efficiency: directional displacement divided by total movement.
- EMA spread normalized by ATR: trend direction and structural separation.
- Realized volatility: separates quiet consolidation from disorderly movement.

Initial classification contract:

```text
Bullish:
  signed efficiency >= positive trend threshold
  AND normalized EMA spread >= positive spread threshold

Bearish:
  signed efficiency <= negative trend threshold
  AND normalized EMA spread <= negative spread threshold

Range:
  directional efficiency is weak
  AND realized volatility is below the compression threshold

Noisy:
  everything else, including contradictory signals
```

Initial train-only pooled thresholds:

- Absolute efficiency trend threshold: 60th percentile.
- Absolute normalized EMA spread threshold: 60th percentile.
- Realized-volatility compression threshold: 40th percentile.
- Common initial structural lookback: 20 bars per cycle.
- LWC pullback lookback: 24 completed 15m states.
- Warm-up or unavailable context is Noisy, never Range.

State codes are fixed and documented:

```text
-1 = Bearish
 0 = Range
 1 = Bullish
 2 = Noisy
```

The quantile choices are frozen before validation results are reviewed. State
coverage is reported for diagnosis, but thresholds must not be repeatedly
tuned against validation, test, or forward results.

## Causal Enrichment

Add `gpu_fuzzy_trader/data/trend_context.py` to generate enriched CSVs without
overwriting raw tapes. It must:

- Validate unique `(datetime, symbol)` rows.
- Validate the 15m grid and reject irregular or ambiguous timestamps.
- Build 1h and 4h bars independently per symbol.
- Publish higher-timeframe state only after that bar is complete, built only
  from complete higher-timeframe buckets (drop incomplete leading/trailing
  buckets caused by a tape that doesn't start/end on an HTF boundary).
- Align completed states back to 15m rows with backward causal semantics.
- Fit thresholds only on the per-symbol training prefix (the exact rows that
  precede the validation split), never on the full pre-split tape, and fit an
  independent threshold set per timeframe (LWC/MWC/HWC) since realized
  volatility and efficiency distributions shift with bar duration.
- Support causal historical warm-up without emitting warm-up rows for scoring;
  forward enrichment may chain trailing train AND test history.
- Write a versioned enrichment manifest with source and history hashes.

With bar-open timestamps and next-bar execution:

- A 15m candle opened at 10:45 can affect an entry at 11:00 after it closes.
- A 1h candle opened at 10:00 and closed at 11:00 can affect the 11:00 entry.
- A 4h candle opened at 08:00 and closed at 12:00 can first affect 12:00.
- Incomplete higher-timeframe candles must never affect an earlier entry.

Generate these columns:

```text
hwc_state
mwc_state
lwc_state
tf_permission_long
tf_permission_short
lwc_pullback_reversal_long
lwc_pullback_reversal_short
```

The deterministic LWC trigger is:

```text
Long:
  current LWC state is Bullish
  AND at least one of the previous 24 completed LWC states was Bearish

Short:
  current LWC state is Bearish
  AND at least one of the previous 24 completed LWC states was Bullish
```

Optional (`CONTEXT_REQUIRE_PERMISSION_ON_PULLBACK_PRINT=True`): the historical
opposite LWC print must also have occurred while same-direction HWC/MWC
permission was active on that bar. Default is False.

The trigger is a pure LTF-timing signal and is intentionally not ANDed with
the *current*-row permission, so `permission_only`/`trigger_only` coverage
diagnostics are meaningful. A tradeable signal always requires both mandatory
conditions (`tf_permission_<dir>` AND `lwc_pullback_reversal_<dir>`) to be
active on the same row — see Mandatory Context below.

Range states may occur inside the pullback sequence. A current Noisy state
never triggers.

Recommended output locations:

```text
data/enriched/train_new_hwc_mwc_lwc.csv
data/enriched/test_new_hwc_mwc_lwc.csv
data/enriched/forward_hwc_mwc_lwc.csv
data/enriched/trend_context_manifest.json
```

The enrichment command must be deterministic and usable for every future
forward tape. Raw CSV hashes remain part of research-integrity records. The
24-bar context contract requires full re-enrichment whenever the contract
version or lookback changes; old enriched tapes, split parquets, feature
selections, Phase 2 pools, and archives must not be reused.

## Mandatory Context

The context must not be represented as ordinary NSGA-III genes. Otherwise
mutation, crossover, or a don't-care value can remove the trading policy.

Every exported rule must contain exactly these direction-specific conditions:

```text
Long:
[tf_permission_long] IS Active (1)
[lwc_pullback_reversal_long] IS Active (1)

Short:
[tf_permission_short] IS Active (1)
[lwc_pullback_reversal_short] IS Active (1)
```

The implementation must apply the same fixed masks in:

- GPU Phase 2 evolution.
- Exact CPU Phase 2 admission.
- RB Governor evaluation and composition.
- Exported strategy JSON.
- Phase 5 evaluation.
- The edited evaluator notebook.

`MIN_CONDITIONS` and `MAX_CONDITIONS` count only evolved confirmations, not
the mandatory context conditions. NSGA-III should search one to three extra
15m confirmations from the existing `ff_*` feature set.

## Pipeline Changes

### `gpu_fuzzy_trader/data/trend_context.py`

Implement validation, train-only threshold fitting, state classification,
causal timeframe alignment, LWC triggers, enriched CSV writing, and manifest
generation.

### `gpu_fuzzy_trader/config.py`

Add the timeframe, timestamp, context version, state-code, enrichment path,
classifier, permission, and trigger settings. Change the linked horizon
contract to:

```python
MAX_HOLD_CANDLES = 96
TAIL_DROP_ROWS = 96
HOLDOUT_EMBARGO_CANDLES = 96
PURGED_WF_EMBARGO_CANDLES = 96
VALIDATION_HALF_PURGE_CANDLES = 96
```

Add validation for these relationships and include the full context contract
in the effective configuration snapshot.

### `gpu_fuzzy_trader/data/loader.py`

Require valid context columns in enriched inputs. Validate state codes,
binary permissions, permission truth tables, mutual exclusivity, timestamps,
and missing values. Do not silently convert missing context to zero or Range.

Keep context columns out of ordinary feature inference while preserving them
for execution.

### `gpu_fuzzy_trader/features/selector.py`

Exclude state, permission, and deterministic trigger columns from Phase 1.
Keep ordinary 15m `ff_*` columns available as evolved confirmations.

### `gpu_fuzzy_trader/backtest/df_slim.py`

Preserve mandatory context columns through feature pruning and all slimmed
training, validation, and evaluation dataframes.

### `gpu_fuzzy_trader/backtest/gpu_engine.py`

Apply the fixed direction and LWC trigger masks to every chromosome during
evolution. Do not add these masks to the feature matrix or chromosome shape.

### `gpu_fuzzy_trader/phases/phase2_rule_pool.py`

Inject mandatory conditions after chromosome decoding and before exact CPU
admission. Ensure active-condition counts, archive entries, and pool metrics
distinguish evolved conditions from fixed context.

### `gpu_fuzzy_trader/rb_governor.py`

Assert that mandatory conditions survive candidate copying, ruleset
composition, risk-grid evaluation, profit amplification, and final strategy
writing. Do not change existing capital sizing behavior.

### `gpu_fuzzy_trader/output/writer.py`

Reject rules missing required context, containing opposite-direction context,
or containing duplicate mandatory conditions. Keep the existing evaluator JSON
schema unchanged.

### `gpu_fuzzy_trader/phases/phase5_oos.py`

Load and validate already-enriched test and forward tapes. Do not fit
thresholds, select rules, prune strategies, or rewrite strategies from Phase 5
results.

## Holding Horizon and Evaluator

Change labels, barriers, force exits, tail drops, and all relevant embargoes
from 288 to 96 bars. Existing label column names containing `_288` may remain
temporarily for schema compatibility, but manifests and documentation must
state that the actual horizon is 96 bars.

Edit `evaluator_v5.ipynb` so its horizon and label behavior match the 96-bar
pipeline contract. Verify exact parity between the notebook and the CPU
engine using identical enriched data and strategies.

Do not claim evaluator parity until the following agree:

- Entry timing.
- TP/SL barrier behavior.
- Maximum holding period.
- Fees and capital rules.
- Context condition evaluation.
- Rule ordering and signal assignment.

## Identity and Cache Invalidation

The context contract must be part of strategy, pool, archive, split, and
dataset identities. Include:

- Context algorithm version.
- Classifier formulas and fitted thresholds.
- Threshold fitting interval and source hash.
- HWC/MWC/LWC timeframes.
- Bar-open timestamp semantics.
- State-code map.
- LWC pullback lookback.
- Direction-permission policy.
- 96-bar horizon.
- Raw, history, and enriched source hashes.

Update identity and freshness checks in:

- `gpu_fuzzy_trader/phases/rule_identity.py`.
- `gpu_fuzzy_trader/phases/phase2_rule_pool.py`.
- `gpu_fuzzy_trader/data/splitter.py`.
- `gpu_fuzzy_trader/features/selector.py`.
- `gpu_fuzzy_trader/research_integrity.py`.

Old split parquets, selected features, Phase 2 pools, and archives must be
rejected when context or horizon identity differs. Resume must fail closed
instead of reusing stale artifacts.

## Warm-Up and Dataset Boundaries

Use causal history for rolling calculations:

- Validation may use preceding train bars for indicator warm-up.
- Test may use trailing train history for warm-up.
- Forward may use trailing train/test history for warm-up.
- Warm-up rows are never emitted or scored in the target tape.
- Thresholds are never refitted outside the training prefix.
- Manifests record history sources and cutoffs.

Generate labels separately per research tape. Never concatenate train, test,
and forward before label generation, since that could allow outcomes to cross
research boundaries.

## Research Protocol

Run three predeclared variants with identical seeds, budgets, risk settings,
and 96-bar horizon:

| Variant | HWC/MWC permission | LWC reversal | Purpose |
| --- | --- | --- | --- |
| A | No | No | 96-bar single-timeframe baseline |
| B | Yes | No | Measure higher-cycle filtering |
| C | Yes | Yes | Measure the complete wave-cycle policy |

Selection protocol:

1. Fit context thresholds from the Phase-0 training prefix only.
2. Run Phase 1 and Phase 2 with train-only adaptive fitness.
3. Use validation fitness for pool admission.
4. Use validation selection and the RB reserved tail for team selection.
5. Select one variant using predeclared validation criteria.
6. Treat `test_new.csv` as consumed diagnostic evidence only.
7. Evaluate the selected strategy once on a strictly newer forward tape.
8. Never update thresholds, rules, or gates from forward results.

Primary comparison criteria:

- Positive net return after fees.
- Positive expectancy lower confidence bound.
- Acceptable maximum drawdown.
- Positive performance across calendar windows.
- Supported positive contributions from BTC and ETH.
- Improvement over Variant A, not merely standalone profitability.
- Sufficient trade count after regime filtering.

Report state coverage, permissions, blocked signals, LWC triggers, trades and
PnL by HWC/MWC state, symbol, direction, and calendar window.

## Tests

Add focused tests for:

- Four-state classification and threshold edge cases.
- Train-only threshold fitting.
- 1h and 4h closed-bar publication timing.
- No future-data influence on earlier states.
- Per-symbol isolation.
- Missing, duplicate, irregular, and shuffled input rows.
- Permission truth tables and mutual exclusivity.
- LWC pullback-reversal transitions.
- Warm-up behavior for validation, test, and forward data.
- Mandatory context through GPU, CPU, Phase 2, RB, JSON, and Phase 5.
- Mutation and crossover inability to remove context.
- 96-bar labels, barriers, force exits, tail drops, and embargoes.
- Cache and archive invalidation after context or horizon changes.
- Pipeline and edited evaluator notebook parity.

Use `.venv` and focused commands with `PYTEST_LOW_MEMORY=1`. Do not run the
full local suite or full GPU pipeline on memory-constrained WSL hosts.

## Deferred Work

Do not include these in the first release:

- HWC reversal entries.
- Confidence-based position sizing.
- Separate NSGA-III searches for each timeframe.
- Per-symbol regime thresholds.
- Threshold tuning against test or forward data.
- Dynamic timeframe selection.
- Learned support/resistance classification.

After stable forward evidence, add causal HWC/MWC support and resistance
features, then evaluate reversal trades as a separate strategy family.

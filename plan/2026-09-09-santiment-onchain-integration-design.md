# Santiment On-Chain Feature Integration Design

## Objective

Add twelve Santiment-derived rule features to the eighteen supplied `ff_*`
features in `data/train_new.csv` and `data/test_new.csv`. The production Phase 2
catalog must contain exactly thirty supplied rule features. HWC and MWC remain
causal frozen context layers and are not counted as supplied rule features.

The change must not claim or guarantee profitable output. It must make the data
and model-selection process causal, reproducible, and resistant to avoidable
selection bias.

## Verified Input Contract

- The source archive contains twenty-four CSV files: twelve metrics for BTC and
  twelve metrics for ETH.
- Intraday files contain 15-minute UTC timestamps from 2023-12-31 20:30 through
  2026-09-08 20:15.
- Daily files contain UTC dates from 2023-12-31 through 2026-09-08.
- The corrected ETH file exposes `Exchange Flow Balance` under the correct file
  name.
- Funding for both assets omits the same twenty-five timestamps from
  2025-01-03 00:15 through 06:15 UTC.
- Other source series have no missing timestamps, duplicate timestamps, NaN, or
  infinite values.
- `train_new.csv` covers 2024-01-01 through 2025-12-31. `test_new.csv` covers
  2026-01-01 through 2026-07-23.

## Feature Inventory

The twelve new columns are:

1. `ff_oc_active_addresses_24h`
2. `ff_oc_age_consumed`
3. `ff_oc_exchange_flow_balance`
4. `ff_oc_funding_rate`
5. `ff_oc_mvrv_30d`
6. `ff_oc_network_profit_loss`
7. `ff_oc_open_interest`
8. `ff_oc_social_dominance`
9. `ff_oc_supply_on_exchanges`
10. `ff_oc_whale_tx_count_1m`
11. `ff_oc_network_growth`
12. `ff_oc_weighted_sentiment`

All twelve columns are floating-point values bounded to `[-1, 1]`. Missing
warm-up values remain NaN. No target or test data is used to fit a transform.

## Metric Registry

Every source metric has a checked-in registry entry with these fields:

- canonical metric name
- output feature name
- source cadence
- source aggregation
- release delay
- raw transform
- causal scale family
- scale window
- minimum history
- allowed source-gap policy

The fixed v1 registry is:

| Output feature | Aggregation | Raw representation | Scale | Release |
|---|---|---|---|---|
| `ff_oc_active_addresses_24h` | 1h `LAST` | 4h log change | centered robust | bucket end + 60m |
| `ff_oc_age_consumed` | 1h `SUM` | `log1p` | centered robust | bucket end + 60m |
| `ff_oc_exchange_flow_balance` | 1h `SUM` | signed `log1p` | zero-anchored robust | bucket end + 60m |
| `ff_oc_funding_rate` | 1h `AVG` | signed rate | zero-anchored robust | bucket end + 60m |
| `ff_oc_mvrv_30d` | 1h `LAST` | `log(max(x, eps))` | centered robust | bucket end + 60m |
| `ff_oc_network_profit_loss` | 1h `SUM` | signed `log1p` | zero-anchored robust | bucket end + 60m |
| `ff_oc_open_interest` | 1h `LAST` | 4h log change | centered robust | bucket end + 60m |
| `ff_oc_social_dominance` | 1h `AVG` | `log1p(max(x, 0))` | centered robust | bucket end + 45m |
| `ff_oc_supply_on_exchanges` | 1d `LAST` | 7d percentage change | centered robust | D+1 02:00 UTC |
| `ff_oc_whale_tx_count_1m` | 1h `SUM` | `log1p` | centered robust | bucket end + 60m |
| `ff_oc_network_growth` | 1d `LAST` | `log1p` | centered robust | D+1 02:00 UTC |
| `ff_oc_weighted_sentiment` | 1d `LAST` | signed value | zero-anchored robust | D+1 02:00 UTC |

`SUM` applies only to event and flow metrics. State metrics use `LAST`, and rate
or share metrics use `AVG`. An interval is published only after its source
bucket is closed and its release delay has elapsed.

## Causal Scaling

Scaling is independent per symbol and per metric. Statistics at timestamp `t`
use only observations strictly before `t`.

For centered robust features:

```text
q25_t = rolling_quantile(y.shift(1), 0.25, window)
center_t = rolling_quantile(y.shift(1), 0.50, window)
q75_t = rolling_quantile(y.shift(1), 0.75, window)
scale_t = (q75_t - q25_t) / 1.349
z_t = (y_t - center_t) / scale_t when scale_t > epsilon, else NaN
feature_t = tanh(clip(z_t, -5, 5) / 2.5)
```

For zero-anchored robust features:

```text
scale_t = rolling_median(abs(y.shift(1)), window)
denominator_t = 1.4826 * scale_t
z_t = y_t / denominator_t when denominator_t > epsilon, else NaN
feature_t = tanh(clip(z_t, -5, 5) / 2.5)
```

Intraday features use a 90-day window with 30 days of minimum history. Daily
features use a 180-day window with 30 observations of minimum history. The
builder reports warm-up coverage. Production use should acquire at least 180
days of prehistory. Until then, early rows remain NaN and must not be changed to
zero.

## Funding Gap Policy

Funding is a state-like rate. The builder reindexes it to the exact 15-minute
grid and applies past-only forward fill with a maximum staleness of eight hours.
It must fill the known twenty-five missing rows from the last observed funding
value. It must never use a later observation, linear interpolation, backward
fill, or a synthetic zero.

The manifest records:

- missing source timestamps
- imputed row count
- maximum observed staleness
- first and last affected timestamps

Any funding gap longer than eight hours fails the build. Any gap in another
metric also fails the build.

## Dataset Join Contract

The builder processes all source history first, then aligns the result to the
target rows. It performs a backward as-of join per symbol using
`available_at <= target datetime`.

The output must preserve, byte-for-value where practical:

- row count
- row order
- `(datetime, symbol)` keys
- OHLCV values
- the eighteen original `ff_*` columns

It adds only the twelve `ff_oc_*` columns and writes atomically through a
temporary file. The build fails on duplicate keys, unsupported symbols,
missing source metrics, range violations, unexpected gaps, or target coverage
loss. A JSON manifest stores source hashes, registry version, transform
parameters, target hashes, output hashes, coverage, and gap handling.

## Runtime Feature Contract

Production Phase 2 uses `RULE_FEATURE_SOURCE_MODE="supplied_ff"`.

- `RULE_ALLOWED_FF_FEATURES` contains exactly the eighteen existing features
  followed by the twelve `ff_oc_*` features.
- `_merge_mtf_lwc_runtime_columns` retains only allowlisted `ff_*` columns from
  enriched source tapes.
- `build_rule_feature_specs` emits exactly those thirty features when all are
  present.
- Auto-generated `lwc_*` features and raw MTF score columns are not Phase 2
  rule conditions in this mode.
- HWC and MWC scores remain frozen hierarchical context used by composition
  and veto logic.
- The existing pure-OHLCV fallback remains available under an explicit
  `RULE_FEATURE_SOURCE_MODE="generated_lwc"` profile.

This removes the current ambiguity in which supplied `ff_*` columns are listed
in config but discarded by the canonical MTF merge.

The duplicate generated features `rsi_14` and `rsi_14_midline` are collapsed to
one feature in the pure-OHLCV fallback.

## Split and Leakage Contract

- Feature definitions and registry values are frozen before test evaluation.
- Development train, validation fitness, and validation selection remain
  chronological and purged.
- Transform values may update online through past observations, but transform
  definitions and windows cannot be selected from validation selection or test.
- `test_new.csv` remains a consumed diagnostic holdout and never selects a
  feature, timeframe, seed, objective, hyperparameter, rule, or portfolio.
- Release acceptance requires a strictly newer forward OHLCV tape and one-shot
  evaluation.
- Appending or changing future source rows must not change any earlier output
  feature value.

## Search and Model-Selection Contract

Adding twelve features increases the 2-to-4-condition feature subset space by
about 7.9 times. Search budget is not increased blindly. Feature families are
screened with controlled ablations first:

- derivatives: funding and open interest
- social: social dominance and weighted sentiment
- exchange and supply: exchange flow and supply on exchanges
- network activity: active addresses and network growth
- valuation and realization: MVRV and network profit/loss
- dormant and whale activity: age consumed and whale count

The registered comparisons are technical-only, supplied eighteen, on-chain
only, full thirty, and leave-one-family-out variants. Screening uses five fixed
seeds. The two finalists use ten fixed seeds. A variant advances only when its
paired validation delta is positive in at least four of five screening seeds,
its median delta is positive, and it does not degrade the worst-seed return,
drawdown, cost-stress certificate, or symbol coverage.

Validation selection is consumed once after the variant and hyperparameters
are frozen. Test is not part of this decision.

## Global and Specialist Search

The code and README must describe the same execution mode. Add an explicit
`PHASE2_SYMBOL_MODE` with `global` and `specialist` values. Both modes use
symbol-specific feature scaling.

Specialist mode runs Phase 2 independently for each `(symbol, direction)` and
attaches immutable `source_symbols` provenance. RB composes the specialists at
portfolio level. Neither mode becomes the production default from a single
run. The default is selected from the registered development-only comparison
and then frozen.

## RB Portfolio Controls

Correlation-aware selection, marginal contribution pruning, and subset
dominance are evaluated as a named RB profile rather than silently changing
defaults. The on-chain profile starts with:

```text
RB_CORRELATION_AWARE_SELECTION = True
RB_REDUNDANCY_PENALTY = "low"
RB_MARGINAL_PRUNING = True
RB_RULESET_MUST_BEAT_SUBSETS = True
```

This profile advances only if it improves the registered validation criteria.
`RB_MAX_RULES=20` is a cap, not a target. Effective rule count, PnL correlation,
signal overlap, marginal contribution, and risk certificates determine the
final size.

## Acceptance Criteria

Data acceptance:

- exactly thirty supplied feature columns
- all twelve new columns within `[-1, 1]` when finite
- no changed target keys, OHLCV values, or original feature values
- no future-dependence under suffix-mutation tests
- daily data unavailable before D+1 02:00 UTC
- exactly twenty-five past-only funding imputations per asset for this source
  snapshot
- no unexpected source gaps

Runtime acceptance:

- production catalog contains exactly thirty supplied features
- no raw or generated LWC feature leaks into that catalog
- HWC and MWC context remains causal and frozen
- CPU and GPU condition masks agree on the new features
- old caches and archives invalidate when the on-chain manifest or feature
  contract changes

Research acceptance:

- no configuration selected from `test_new.csv`
- all ablation runs recorded in the experiment ledger
- winner selected by pre-registered paired multi-seed criteria
- cost, execution delay, regime, rule dropout, concentration, symbol coverage,
  and multiplicity diagnostics reported for the frozen finalist
- strictly newer forward tape required for release acceptance

# Santiment On-Chain Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add twelve causal Santiment features to the eighteen supplied rule features, make the production Phase 2 catalog use exactly those thirty features, and validate the resulting strategy without tuning on test data.

**Architecture:** A registry-driven on-chain builder audits the twenty-four source CSVs, repairs only the known Funding gap with capped past-only carry, applies metric-specific aggregation and release delays, creates causal bounded features, and atomically enriches the frozen train and test tapes. The canonical MTF path keeps HWC/MWC as frozen context while Phase 2 uses the thirty supplied `ff_*` features. Research variants are compared through pre-registered, purged, multi-seed development experiments before one-time diagnostic and forward evaluation.

**Tech Stack:** Python 3.10+, pandas, NumPy, pytest, Hypothesis, existing JAX/Numba CPU-GPU evaluators, JSON manifests, existing experiment ledger.

**Spec:** `docs/superpowers/specs/2026-09-09-santiment-onchain-integration-design.md`

## Global Constraints

- Run every Python or pytest command through `.venv/bin/python`.
- Set `PYTEST_LOW_MEMORY=1` for broad test groups. Do not run the full suite locally without it.
- Never use validation selection, `test_new.csv`, or a forward tape to define features, windows, lags, hyperparameters, or experiment winners.
- Preserve source row order, `(datetime, symbol)` keys, OHLCV, and all eighteen existing `ff_*` values.
- New finite feature values must be in `[-1, 1]`; causal warm-up remains NaN.
- Daily value D must not be visible before D+1 02:00 UTC.
- Funding gap repair is past-only carry with an eight-hour maximum staleness. No backward fill, interpolation, or zero fill.
- Production supplied-feature catalog size is exactly thirty. HWC/MWC context scores are not part of that count.
- Do not commit the 66 MB extracted Santiment source directory. Commit its hashes and build manifest.
- Treat `test_new.csv` as a consumed diagnostic tape. Release acceptance requires a strictly newer one-shot forward tape.

## File Map

Create:

- `gpu_fuzzy_trader/data/onchain_registry.py`: immutable metric definitions and feature-family metadata.
- `gpu_fuzzy_trader/data/onchain_pipeline.py`: source audit, gap policy, resampling, release timing, causal transforms, as-of alignment, and manifest creation.
- `scripts/build_onchain_dataset.py`: safe CLI that stages, verifies, and atomically replaces or writes enriched CSVs.
- `scripts/run_onchain_ablation.py`: registered multi-seed feature and symbol-mode experiment runner.
- `profiles/onchain_v1.json`: frozen, reviewable development decision selected in Task 12.
- `tests/unit/test_onchain_registry.py`: exact registry contract.
- `tests/unit/test_onchain_pipeline.py`: data audit, causal transforms, gap, release, and join tests.
- `tests/unit/test_onchain_dataset_builder.py`: end-to-end builder and manifest tests.
- `tests/unit/test_onchain_feature_contract.py`: exact thirty-feature runtime catalog tests.
- `tests/unit/test_phase2_symbol_modes.py`: global and specialist orchestration tests.
- `tests/unit/test_onchain_ablation.py`: experiment isolation and winner-gate tests.

Modify:

- `gpu_fuzzy_trader/config.py`: thirty-feature allowlist, feature-source mode, on-chain paths and profile controls, explicit symbol mode.
- `gpu_fuzzy_trader/features/catalog.py`: explicit supplied-feature versus generated-LWC catalog policy.
- `gpu_fuzzy_trader/features/fuzzy_scaling.py`: recognize pre-bounded continuous supplied features without fitting on validation or test.
- `gpu_fuzzy_trader/data/multi_timeframe.py`: remove the duplicate generated RSI feature.
- `gpu_fuzzy_trader/run_pipeline.py`: retain allowed supplied features in the MTF merge, enforce catalog identity, route symbol modes, and record on-chain lineage.
- `gpu_fuzzy_trader/research_integrity.py`: include on-chain manifest and feature-contract digests in run identity.
- `gpu_fuzzy_trader/rb_governor.py`: select an explicit RB profile without silently changing global defaults.
- `tests/unit/test_mtf_pipeline_integration.py`: update legacy-drop expectations and assert allowed-feature retention.
- `tests/unit/test_feature_catalog.py`: test both source modes and exact exclusions.
- `tests/unit/test_fuzzy_scaling.py`: bounded continuous feature contract.
- `tests/unit/test_research_integrity.py`: lineage and cache invalidation.
- `tests/unit/test_run_pipeline.py`: exact production catalog and context-only MTF scores.
- `README.md`: correct the actual feature and symbol-mode contracts.
- `RUN.md`: document source extraction, build, ablation, frozen evaluation, and forward acceptance.
- `.gitignore`: ignore extracted Santiment source CSVs and staged temporary CSVs.

---

### Task 1: Define the Immutable Metric Registry

**Files:**
- Create: `gpu_fuzzy_trader/data/onchain_registry.py`
- Create: `tests/unit/test_onchain_registry.py`

**Interfaces:**
- Produces: `OnchainMetricSpec`, `ONCHAIN_METRIC_SPECS`, `ONCHAIN_FEATURE_NAMES`, `ONCHAIN_REGISTRY_VERSION`, `metric_spec_by_source_column()`.
- Consumes: no runtime data.

- [ ] **Step 1: Write the failing registry inventory test**

```python
from gpu_fuzzy_trader.data.onchain_registry import (
    ONCHAIN_FEATURE_NAMES,
    ONCHAIN_METRIC_SPECS,
)


def test_registry_has_twelve_unique_bounded_feature_contracts() -> None:
    assert len(ONCHAIN_METRIC_SPECS) == 12
    assert len(ONCHAIN_FEATURE_NAMES) == 12
    assert len(set(ONCHAIN_FEATURE_NAMES)) == 12
    assert all(name.startswith("ff_oc_") for name in ONCHAIN_FEATURE_NAMES)
    assert {spec.output_feature for spec in ONCHAIN_METRIC_SPECS} == set(
        ONCHAIN_FEATURE_NAMES
    )
```

- [ ] **Step 2: Run the test and confirm the missing module failure**

Run:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_onchain_registry.py -q
```

Expected: FAIL because `gpu_fuzzy_trader.data.onchain_registry` does not exist.

- [ ] **Step 3: Implement the registry type and fixed entries**

```python
from dataclasses import dataclass
from typing import Literal

Aggregation = Literal["sum", "last", "mean"]
ScaleFamily = Literal["centered", "zero"]
RawToken = Literal[
    "identity", "log1p", "signed_log", "log_level",
    "log_change_4h", "pct_change_7d",
]


@dataclass(frozen=True)
class OnchainMetricSpec:
    source_column: str
    output_feature: str
    source_cadence: str
    target_cadence: str
    aggregation: Aggregation
    raw_transform: RawToken
    scale_family: ScaleFamily
    release_delay_minutes: int
    scale_window_observations: int
    min_periods: int
    allow_funding_gap: bool = False


ONCHAIN_REGISTRY_VERSION = "1.0.0"

ONCHAIN_METRIC_SPECS = (
    OnchainMetricSpec(
        "Active Addresses 24h", "ff_oc_active_addresses_24h",
        "15min", "1h", "last", "log_change_4h", "centered",
        60, 2160, 720,
    ),
    OnchainMetricSpec(
        "Age Consumed", "ff_oc_age_consumed",
        "15min", "1h", "sum", "log1p", "centered",
        60, 2160, 720,
    ),
    OnchainMetricSpec(
        "Exchange Flow Balance", "ff_oc_exchange_flow_balance",
        "15min", "1h", "sum", "signed_log", "zero",
        60, 2160, 720,
    ),
    OnchainMetricSpec(
        "Total Funding Rates Aggregated by Asset", "ff_oc_funding_rate",
        "15min", "1h", "mean", "identity", "zero",
        60, 2160, 720, True,
    ),
    OnchainMetricSpec(
        "MVRV Ratio Intraday (30d)", "ff_oc_mvrv_30d",
        "15min", "1h", "last", "log_level", "centered",
        60, 2160, 720,
    ),
    OnchainMetricSpec(
        "Network Realized Profit/Loss", "ff_oc_network_profit_loss",
        "15min", "1h", "sum", "signed_log", "zero",
        60, 2160, 720,
    ),
    OnchainMetricSpec(
        "Total Open Interest in USD", "ff_oc_open_interest",
        "15min", "1h", "last", "log_change_4h", "centered",
        60, 2160, 720,
    ),
    OnchainMetricSpec(
        "Social Dominance", "ff_oc_social_dominance",
        "15min", "1h", "mean", "log1p", "centered",
        45, 2160, 720,
    ),
    OnchainMetricSpec(
        "Supply on Exchanges", "ff_oc_supply_on_exchanges",
        "1d", "1d", "last", "pct_change_7d", "centered",
        120, 180, 30,
    ),
    OnchainMetricSpec(
        "Whale Transaction Count (>1m USD)", "ff_oc_whale_tx_count_1m",
        "15min", "1h", "sum", "log1p", "centered",
        60, 2160, 720,
    ),
    OnchainMetricSpec(
        "Network Growth", "ff_oc_network_growth",
        "1d", "1d", "last", "log1p", "centered",
        120, 180, 30,
    ),
    OnchainMetricSpec(
        "Weighted sentiment (Total)", "ff_oc_weighted_sentiment",
        "1d", "1d", "last", "identity", "zero",
        120, 180, 30,
    ),
)

ONCHAIN_FEATURE_NAMES = tuple(
    spec.output_feature for spec in ONCHAIN_METRIC_SPECS
)

_SPEC_BY_SOURCE_COLUMN = {
    spec.source_column: spec for spec in ONCHAIN_METRIC_SPECS
}


def metric_spec_by_source_column(source_column: str) -> OnchainMetricSpec:
    try:
        return _SPEC_BY_SOURCE_COLUMN[source_column]
    except KeyError as exc:
        raise ValueError(f"Unsupported Santiment metric: {source_column}") from exc
```

The hourly window is 2160 observations, equal to 90 days. The daily window is
180 observations. Funding is the only entry with `allow_funding_gap=True`.

- [ ] **Step 4: Add exact aggregation and release tests**

```python
def test_flow_state_and_daily_policies_are_explicit() -> None:
    specs = {spec.output_feature: spec for spec in ONCHAIN_METRIC_SPECS}
    assert specs["ff_oc_exchange_flow_balance"].aggregation == "sum"
    assert specs["ff_oc_open_interest"].aggregation == "last"
    assert specs["ff_oc_funding_rate"].aggregation == "mean"
    assert specs["ff_oc_social_dominance"].release_delay_minutes == 45
    assert specs["ff_oc_supply_on_exchanges"].source_cadence == "1d"
    assert specs["ff_oc_supply_on_exchanges"].release_delay_minutes == 120
```

- [ ] **Step 5: Run tests and commit**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_onchain_registry.py -q
git add gpu_fuzzy_trader/data/onchain_registry.py tests/unit/test_onchain_registry.py
git commit -m "feat: define Santiment metric registry"
```

---

### Task 2: Audit Source Files and Repair Only the Funding Gap

**Files:**
- Create: `gpu_fuzzy_trader/data/onchain_pipeline.py`
- Create: `tests/unit/test_onchain_pipeline.py`

**Interfaces:**
- Consumes: `ONCHAIN_METRIC_SPECS`.
- Produces: `load_and_audit_sources(source_dir: Path) -> tuple[dict[tuple[str, str], pd.Series], dict]` and `repair_funding_grid(series: pd.Series, max_staleness: pd.Timedelta) -> tuple[pd.Series, dict]`.

- [ ] **Step 1: Write failing source-contract tests**

Create fixtures for two assets and all twelve source columns with this helper:

```python
def write_complete_source_fixture(tmp_path: Path) -> dict[tuple[str, str], Path]:
    paths = {}
    assets = (("Bitcoin", "BTCUSDT"), ("Ethereum", "ETHUSDT"))
    for asset_name, symbol in assets:
        for position, spec in enumerate(ONCHAIN_METRIC_SPECS, start=1):
            cadence = "15min" if spec.source_cadence == "15min" else "1D"
            periods = 16 if cadence == "15min" else 10
            index = pd.date_range("2023-12-01", periods=periods, freq=cadence, tz="UTC")
            values = np.arange(1, periods + 1, dtype=float) + position
            path = tmp_path / f"{asset_name} metric-{position}.csv"
            pd.DataFrame({"Date": index, spec.source_column: values}).to_csv(
                path, index=False
            )
            paths[(symbol, spec.source_column)] = path
    return paths
```

Assert rejection of duplicate dates, non-finite values, missing metrics,
unsupported symbols, and gaps in non-Funding metrics.

```python
def test_non_funding_gap_fails_closed(tmp_path: Path) -> None:
    paths = write_complete_source_fixture(tmp_path)
    path = paths[("BTCUSDT", "Total Open Interest in USD")]
    frame = pd.read_csv(path).drop(index=3)
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unexpected timestamp gap"):
        load_and_audit_sources(tmp_path)
```

Repeat the mutation pattern for a duplicated `Date`, `np.inf`, one deleted
source file, and a file whose asset prefix is `Solana`. Assert the exact error
tokens `duplicate`, `non-finite`, `missing source metrics`, and
`unsupported asset`, respectively.

- [ ] **Step 2: Write the failing twenty-five-row Funding test**

```python
def test_funding_gap_uses_only_past_value() -> None:
    index = pd.date_range("2025-01-02 23:45", periods=31, freq="15min", tz="UTC")
    series = pd.Series(np.arange(31, dtype=float), index=index)
    series = series.drop(index[2:27])
    repaired, audit = repair_funding_grid(series, pd.Timedelta(hours=8))
    assert audit["imputed_rows"] == 25
    assert repaired.loc[index[2]:index[26]].eq(series.iloc[1]).all()
    assert repaired.loc[index[27]] == series.loc[index[27]]
```

- [ ] **Step 3: Run both tests and confirm failure**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_onchain_pipeline.py -k "gap or source_contract" -vv
```

- [ ] **Step 4: Implement strict CSV discovery and audit**

The loader must identify metrics from the second CSV column, not from the file
name. Normalize file assets through one fixed map:

```python
ASSET_PREFIX_TO_SYMBOL = {"Bitcoin": "BTCUSDT", "Ethereum": "ETHUSDT"}
```

Parse `Date` as UTC, coerce the value column to float, require two columns,
require unique increasing timestamps, and require one source for every
`(symbol, source_column)` pair. Reject extra duplicate representations.

- [ ] **Step 5: Implement capped past-only Funding carry**

```python
def repair_funding_grid(series, max_staleness=pd.Timedelta(hours=8)):
    full = pd.date_range(series.index.min(), series.index.max(), freq="15min", tz="UTC")
    reindexed = series.reindex(full)
    missing = reindexed.isna()
    repaired = reindexed.ffill(limit=int(max_staleness / pd.Timedelta(minutes=15)))
    unresolved = repaired.isna() & missing
    if unresolved.any():
        raise ValueError("Funding gap exceeds maximum staleness")
    return repaired, {
        "imputed_rows": int(missing.sum()),
        "first_missing": missing[missing].index.min().isoformat() if missing.any() else None,
        "last_missing": missing[missing].index.max().isoformat() if missing.any() else None,
        "max_staleness_minutes": int(max_staleness / pd.Timedelta(minutes=1)),
    }
```

Do not call `bfill`, `interpolate`, or `fillna(0)` anywhere in this module.

- [ ] **Step 6: Add an eight-hour fail-closed test and run the module tests**

```python
def test_funding_gap_over_eight_hours_fails() -> None:
    index = pd.date_range("2025-01-01", periods=50, freq="15min", tz="UTC")
    series = pd.Series(1.0, index=index).drop(index=index[1:35])
    with pytest.raises(ValueError, match="maximum staleness"):
        repair_funding_grid(series, pd.Timedelta(hours=8))
```

Run and commit:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_onchain_pipeline.py -q
git add gpu_fuzzy_trader/data/onchain_pipeline.py tests/unit/test_onchain_pipeline.py
git commit -m "feat: audit on-chain sources and repair funding gaps"
```

---

### Task 3: Implement Metric-Specific Aggregation and Availability

**Files:**
- Modify: `gpu_fuzzy_trader/data/onchain_pipeline.py`
- Modify: `tests/unit/test_onchain_pipeline.py`

**Interfaces:**
- Produces: `aggregate_metric(series: pd.Series, spec: OnchainMetricSpec) -> pd.DataFrame` with `value` and `available_at` columns.

- [ ] **Step 1: Write failing aggregation tests**

```python
from dataclasses import replace

BASE_SPEC = next(
    spec for spec in ONCHAIN_METRIC_SPECS
    if spec.output_feature == "ff_oc_active_addresses_24h"
)
DAILY_SPEC = next(
    spec for spec in ONCHAIN_METRIC_SPECS
    if spec.output_feature == "ff_oc_supply_on_exchanges"
)


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [("sum", 10.0), ("last", 4.0), ("mean", 2.5)],
)
def test_closed_hour_uses_metric_specific_aggregation(aggregation, expected):
    source = pd.Series(
        [1.0, 2.0, 3.0, 4.0],
        index=pd.date_range("2024-01-01", periods=4, freq="15min", tz="UTC"),
    )
    spec = replace(BASE_SPEC, aggregation=aggregation, release_delay_minutes=60)
    result = aggregate_metric(source, spec)
    assert result.iloc[0]["value"] == expected
    assert result.iloc[0]["available_at"] == pd.Timestamp("2024-01-01 02:00", tz="UTC")
```

The source timestamps describe 15-minute interval starts. The 00:00 to 01:00
bucket closes at 01:00 and becomes available at 02:00 after a 60-minute delay.

- [ ] **Step 2: Write the failing daily-release test**

```python
def test_daily_value_is_released_next_day_at_0200() -> None:
    source = pd.Series([7.0], index=[pd.Timestamp("2024-01-03", tz="UTC")])
    result = aggregate_metric(source, DAILY_SPEC)
    assert result.iloc[0]["available_at"] == pd.Timestamp("2024-01-04 02:00", tz="UTC")
```

- [ ] **Step 3: Implement complete-bucket aggregation**

For non-Funding intraday metrics require four observations per 1-hour bucket.
Funding is already repaired to a complete 15-minute grid in Task 2. Use
left-labeled, left-closed UTC resampling. Set `available_at` from the bucket end,
not the bucket label. Daily values use D+1 02:00 UTC.

- [ ] **Step 4: Add future-suffix invariance test**

```python
def test_future_source_change_cannot_change_earlier_aggregates() -> None:
    base = pd.Series(
        np.arange(40 * 24 * 4, dtype=float),
        index=pd.date_range(
            "2024-01-01", periods=40 * 24 * 4, freq="15min", tz="UTC"
        ),
    )
    changed = base.copy()
    changed.iloc[-1] = 1e12
    a = aggregate_metric(base, BASE_SPEC)
    b = aggregate_metric(changed, BASE_SPEC)
    pd.testing.assert_frame_equal(a.iloc[:-1], b.iloc[:-1])
```

- [ ] **Step 5: Run and commit**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_onchain_pipeline.py -k "aggregation or released or future_source" -q
git add gpu_fuzzy_trader/data/onchain_pipeline.py tests/unit/test_onchain_pipeline.py
git commit -m "feat: add causal on-chain release timing"
```

---

### Task 4: Build the Twelve Causal Bounded Features

**Files:**
- Modify: `gpu_fuzzy_trader/data/onchain_pipeline.py`
- Modify: `tests/unit/test_onchain_pipeline.py`

**Interfaces:**
- Produces: `transform_metric(aggregated: pd.DataFrame, spec: OnchainMetricSpec) -> pd.DataFrame` and `build_symbol_feature_frame(sources: dict, symbol: str) -> pd.DataFrame`.

- [ ] **Step 1: Write failing bound and warm-up tests**

```python
INTRADAY_CENTERED_SPEC = next(
    spec for spec in ONCHAIN_METRIC_SPECS
    if spec.output_feature == "ff_oc_age_consumed"
)


def aggregate_fixture(periods, cadence, multiplier=1.0, constant=None):
    values = (
        np.full(periods, constant, dtype=float)
        if constant is not None
        else (np.arange(periods, dtype=float) + 1.0) * multiplier
    )
    return pd.DataFrame(
        {
            "value": values,
            "available_at": pd.date_range(
                "2023-01-01", periods=periods, freq=cadence, tz="UTC"
            ),
        }
    ).set_index("available_at", drop=False)


def test_centered_transform_is_bounded_and_keeps_warmup_nan() -> None:
    frame = aggregate_fixture(periods=900, cadence="1h")
    result = transform_metric(frame, INTRADAY_CENTERED_SPEC)
    assert result[INTRADAY_CENTERED_SPEC.output_feature].iloc[:720].isna().all()
    finite = result[INTRADAY_CENTERED_SPEC.output_feature].dropna()
    assert finite.between(-1.0, 1.0).all()
```

- [ ] **Step 2: Write failing causal and symbol-isolation tests**

```python
def test_transform_is_suffix_invariant() -> None:
    base = aggregate_fixture(periods=1000, cadence="1h")
    changed = base.copy()
    changed.loc[changed.index[-1], "value"] *= 1000
    a = transform_metric(base, INTRADAY_CENTERED_SPEC)
    b = transform_metric(changed, INTRADAY_CENTERED_SPEC)
    pd.testing.assert_series_equal(
        a.iloc[:-1, 0], b.iloc[:-1, 0], check_names=False
    )


def test_btc_values_do_not_fit_eth_scale() -> None:
    btc = aggregate_fixture(periods=1000, cadence="1h", multiplier=1000)
    eth = aggregate_fixture(periods=1000, cadence="1h", multiplier=1)
    eth_before = transform_metric(eth, INTRADAY_CENTERED_SPEC)
    btc["value"] *= 100
    transform_metric(btc, INTRADAY_CENTERED_SPEC)
    eth_after = transform_metric(eth, INTRADAY_CENTERED_SPEC)
    pd.testing.assert_frame_equal(eth_before, eth_after)
```

- [ ] **Step 3: Implement raw representations**

Use exact functions:

```python
def signed_log1p(values):
    return np.sign(values) * np.log1p(np.abs(values))

def log_change(values, periods):
    safe = values.where(values > 0.0)
    return np.log(safe).diff(periods)
```

For hourly state metrics, four hours equals four aggregated observations. For
daily Supply, seven days equals seven observations.

- [ ] **Step 4: Implement prior-only centered and zero-anchored scales**

```python
def _rolling_center_scale(values, window, min_periods):
    history = values.shift(1)
    rolling = history.rolling(window, min_periods=min_periods)
    q25 = rolling.quantile(0.25)
    center = rolling.quantile(0.50)
    q75 = rolling.quantile(0.75)
    return center, (q75 - q25) / 1.349

def _bounded_z(z):
    return np.tanh(z.clip(-5.0, 5.0) / 2.5)
```

For centered scaling use `(value - center)/IQR_scale`. For zero-anchored
scaling use `value/(1.4826*rolling_median(abs(history)))`. Replace a zero scale
with NaN. Do not use an epsilon to turn a constant series into an extreme
feature. Constant or insufficient-history output must remain NaN and later fail
the coverage or dispersion audit.

- [ ] **Step 5: Add a constant-series and infinity rejection test**

```python
def test_constant_or_invalid_series_does_not_create_extreme_feature() -> None:
    constant = aggregate_fixture(periods=1000, cadence="1h", constant=1.0)
    result = transform_metric(constant, INTRADAY_CENTERED_SPEC)
    assert result.iloc[:, 0].dropna().empty
```

- [ ] **Step 6: Run and commit**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_onchain_pipeline.py -k "transform or bounded or suffix or symbol or constant" -q
git add gpu_fuzzy_trader/data/onchain_pipeline.py tests/unit/test_onchain_pipeline.py
git commit -m "feat: derive bounded causal on-chain features"
```

---

### Task 5: Build and Verify Enriched Train and Test Tapes

**Files:**
- Create: `scripts/build_onchain_dataset.py`
- Create: `tests/unit/test_onchain_dataset_builder.py`
- Modify: `gpu_fuzzy_trader/data/onchain_pipeline.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `enrich_target_frame(target: pd.DataFrame, feature_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]` and CLI options `--source-dir`, `--train`, `--test`, `--output-train`, `--output-test`, `--manifest`, `--replace`.

- [ ] **Step 1: Write the failing backward-as-of tests**

```python
def target_fixture(datetimes):
    return pd.DataFrame({
        "datetime": pd.to_datetime(datetimes, utc=True),
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 10.0,
    })


def feature_fixture(available_at, value):
    frame = pd.DataFrame({
        "available_at": [pd.Timestamp(available_at, tz="UTC")],
        "ff_oc_funding_rate": [value],
    })
    return frame


def test_feature_is_not_visible_before_available_at() -> None:
    target = target_fixture([
        "2024-01-02 01:45", "2024-01-02 02:00", "2024-01-02 02:15"
    ])
    features = feature_fixture(
        available_at="2024-01-02 02:00", value=0.5
    )
    enriched, _ = enrich_target_frame(target, {"BTCUSDT": features})
    assert np.isnan(enriched.loc[0, "ff_oc_funding_rate"])
    assert enriched.loc[1:, "ff_oc_funding_rate"].eq(0.5).all()
```

- [ ] **Step 2: Write invariance and exact-column tests**

```python
def complete_target_fixture():
    frame = target_fixture(pd.date_range("2024-02-01", periods=8, freq="15min"))
    for position, name in enumerate(config.LEGACY_RULE_ALLOWED_FF_FEATURES):
        frame[name] = float(position % 5)
    return frame


def complete_feature_frames():
    frame = pd.DataFrame({
        "available_at": [pd.Timestamp("2024-01-31", tz="UTC")],
        **{name: [0.25] for name in ONCHAIN_FEATURE_NAMES},
    })
    return {"BTCUSDT": frame}


def write_target(path: Path) -> Path:
    complete_target_fixture().to_csv(path, index=False)
    return path


def test_enrichment_preserves_source_and_adds_exactly_twelve_columns() -> None:
    source = complete_target_fixture()
    enriched, audit = enrich_target_frame(source, complete_feature_frames())
    pd.testing.assert_frame_equal(
        enriched[source.columns], source, check_dtype=False
    )
    assert set(enriched.columns) - set(source.columns) == set(ONCHAIN_FEATURE_NAMES)
    assert len(enriched) == len(source)
    assert audit["duplicate_target_keys"] == 0
```

- [ ] **Step 3: Implement per-symbol backward as-of alignment**

Sort temporary copies by time, run `pd.merge_asof(..., direction="backward")`
per symbol, restore `_target_order`, then compare every original column and key.
Reject duplicate target keys before the merge.

- [ ] **Step 4: Implement the manifest**

Write deterministic JSON with:

```json
{
  "registry_version": "1.0.0",
  "source_files": {"filename.csv": "sha256"},
  "source_audit": {},
  "target_inputs": {"train": "sha256", "test": "sha256"},
  "outputs": {"train": "sha256", "test": "sha256"},
  "feature_names": [],
  "coverage_by_symbol_and_feature": {},
  "funding_gap": {},
  "build_parameters": {}
}
```

Exclude wall-clock time from the deterministic digest. Store optional
`created_at` outside the hashed `contract` object.

- [ ] **Step 5: Implement atomic staging and explicit replacement**

The CLI writes `*.staged.csv`, re-reads and verifies both staged files, writes
the manifest, and only calls `os.replace` when `--replace` is present. Without
`--replace`, it writes the requested output paths and leaves source files
untouched.

- [ ] **Step 6: Test failed verification leaves originals unchanged**

```python
def test_failed_build_never_replaces_original(tmp_path, monkeypatch) -> None:
    train = write_target(tmp_path / "train_new.csv")
    before = train.read_bytes()
    def fail_verification(*_args, **_kwargs):
        raise ValueError("forced staged verification failure")
    monkeypatch.setattr(builder, "verify_staged_dataset", fail_verification)
    with pytest.raises(ValueError):
        builder.main([
            "--source-dir", str(tmp_path / "sources"),
            "--train", str(train), "--test", str(train),
            "--output-train", str(train), "--output-test", str(tmp_path / "test.csv"),
            "--manifest", str(tmp_path / "manifest.json"), "--replace",
        ])
    assert train.read_bytes() == before
```

Import the CLI module as `builder`. Copy the complete twelve-metric source
fixture function shown in Task 2 into this independent test module and call it
with `tmp_path / "sources"` before invoking the CLI.

- [ ] **Step 7: Run and commit**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit/test_onchain_dataset_builder.py tests/unit/test_onchain_pipeline.py -q
git add scripts/build_onchain_dataset.py gpu_fuzzy_trader/data/onchain_pipeline.py tests/unit/test_onchain_dataset_builder.py .gitignore
git commit -m "feat: build causally enriched train and test tapes"
```

---

### Task 6: Make the Production Phase 2 Catalog Use Exactly Thirty Features

**Files:**
- Modify: `gpu_fuzzy_trader/config.py`
- Modify: `gpu_fuzzy_trader/features/catalog.py`
- Modify: `gpu_fuzzy_trader/features/fuzzy_scaling.py`
- Modify: `gpu_fuzzy_trader/run_pipeline.py`
- Modify: `tests/unit/test_feature_catalog.py`
- Modify: `tests/unit/test_fuzzy_scaling.py`
- Create: `tests/unit/test_onchain_feature_contract.py`
- Modify: `tests/unit/test_mtf_pipeline_integration.py`

**Interfaces:**
- Produces: `RULE_FEATURE_SOURCE_MODE` with `supplied_ff` and `generated_lwc` values.
- Changes: `_merge_mtf_lwc_runtime_columns()` retains allowlisted supplied features.

- [ ] **Step 1: Write the failing exact-thirty catalog test**

```python
def production_feature_fixture() -> pd.DataFrame:
    frame = pd.DataFrame({
        "datetime": pd.date_range("2024-02-01", periods=4, freq="15min"),
        "symbol": "BTCUSDT",
        "open": 100.0, "high": 101.0, "low": 99.0,
        "close": 100.0, "volume": 10.0,
    })
    for position, name in enumerate(config.LEGACY_RULE_ALLOWED_FF_FEATURES):
        frame[name] = [float((position + row) % 5) for row in range(4)]
    for position, name in enumerate(ONCHAIN_FEATURE_NAMES):
        frame[name] = np.linspace(-0.8, 0.8, 4) + position * 1e-4
    frame["lwc_unapproved"] = [-0.5, 0.0, 0.5, 1.0]
    frame["mtf_hwc_direction"] = [-1.0, 0.0, 1.0, 0.0]
    return frame


def test_supplied_mode_catalog_is_exactly_thirty(monkeypatch) -> None:
    monkeypatch.setattr(config, "RULE_FEATURE_SOURCE_MODE", "supplied_ff")
    frame = production_feature_fixture()
    specs = build_rule_feature_specs(frame)
    assert [item["name"] for item in specs] == list(
        config.RULE_ALLOWED_FF_FEATURES
    )
    assert len(specs) == 30
    assert not any(item["name"].startswith("lwc_") for item in specs)
    assert not any(item["name"].startswith("mtf_") for item in specs)
```

- [ ] **Step 2: Change config validation from hard-coded eighteen to the explicit contract**

Append `ONCHAIN_FEATURE_NAMES` to the existing eighteen names and validate:

```python
RULE_FEATURE_SOURCE_MODE = "supplied_ff"
RULE_ALLOWED_FF_FEATURES = (*LEGACY_RULE_ALLOWED_FF_FEATURES, *ONCHAIN_FEATURE_NAMES)
_config_check(len(RULE_ALLOWED_FF_FEATURES) == 30, "...")
```

Also validate that all names are unique and start with `ff_`, and that source
mode belongs to `{"supplied_ff", "generated_lwc"}`.

- [ ] **Step 3: Implement explicit catalog policy**

In `candidate_rule_feature_columns`:

```python
if config.RULE_FEATURE_SOURCE_MODE == "supplied_ff":
    missing = [name for name in allowed_ff if name not in train_df.columns]
    if missing:
        raise ValueError(f"Missing supplied rule features: {missing}")
    return list(allowed_ff)
```

Keep the existing safe generated-LWC logic only in `generated_lwc` mode.

- [ ] **Step 4: Write and implement MTF retention test**

Replace the legacy expectation in
`test_lwc_runtime_merge_keeps_own_features_but_not_raw_htf_features` with:

```python
assert "ff_donchian_width_20" in merged.columns
assert "ff_oc_funding_rate" in merged.columns
assert "ff_unapproved" not in merged.columns
assert not any(column.startswith("hwc_") for column in merged.columns)
assert not any(column.startswith("mwc_") for column in merged.columns)
```

Change `_merge_mtf_lwc_runtime_columns` so `base_columns` includes only
`column in RULE_ALLOWED_FF_FEATURES` in addition to its current base and label
columns. Do not retain arbitrary `ff_*` or stale HTF features.

- [ ] **Step 5: Make pre-bounded on-chain floats an explicit scaling contract**

Extend the scaling manifest with:

```python
features[name] = {"kind": "bounded_continuous", "scale": 1.0}
```

only when `name in ONCHAIN_FEATURE_NAMES` and every finite training value is in
`[-1, 1]`. `apply_fuzzy_feature_scaling` leaves these values unchanged. Reject
an out-of-range on-chain feature before Phase 2.

- [ ] **Step 6: Run targeted tests and commit**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_onchain_feature_contract.py \
  tests/unit/test_feature_catalog.py \
  tests/unit/test_fuzzy_scaling.py \
  tests/unit/test_mtf_pipeline_integration.py -q
git add gpu_fuzzy_trader/config.py gpu_fuzzy_trader/features/catalog.py \
  gpu_fuzzy_trader/features/fuzzy_scaling.py gpu_fuzzy_trader/run_pipeline.py \
  tests/unit/test_onchain_feature_contract.py tests/unit/test_feature_catalog.py \
  tests/unit/test_fuzzy_scaling.py tests/unit/test_mtf_pipeline_integration.py
git commit -m "feat: activate thirty supplied rule features"
```

---

### Task 7: Remove MTF Redundancy and Bind Caches to On-Chain Lineage

**Files:**
- Modify: `gpu_fuzzy_trader/data/multi_timeframe.py`
- Modify: `gpu_fuzzy_trader/research_integrity.py`
- Modify: `gpu_fuzzy_trader/run_pipeline.py`
- Modify: `tests/unit/test_multi_timeframe.py`
- Modify: `tests/unit/test_research_integrity.py`
- Modify: `tests/unit/test_run_pipeline.py`

**Interfaces:**
- Produces: `feature_contract_digest() -> str` and on-chain lineage in dataset and experiment manifests.

- [ ] **Step 1: Write the failing duplicate-RSI test**

```python
def test_generated_timeframe_features_have_no_duplicate_rsi_alias() -> None:
    result = compute_timeframe_features(ohlcv_fixture(), 15)
    assert "rsi_14_midline" in result.columns
    assert "rsi_14" not in result.columns
```

- [ ] **Step 2: Remove `rsi_14` assignment and keep `rsi_14_midline`**

Delete only the duplicate public feature. Do not change RSI computation,
scaling, labels, or archive semantics beyond the feature-contract digest.

- [ ] **Step 3: Write failing lineage invalidation test**

```python
def write_onchain_manifest(tmp_path: Path, registry_version: str) -> Path:
    path = tmp_path / "onchain_feature_manifest.json"
    path.write_text(
        json.dumps({
            "contract": {
                "registry_version": registry_version,
                "feature_names": list(ONCHAIN_FEATURE_NAMES),
            }
        }, sort_keys=True),
        encoding="utf-8",
    )
    return path


def test_onchain_manifest_change_invalidates_feature_contract(tmp_path) -> None:
    first = write_onchain_manifest(tmp_path, registry_version="1.0.0")
    digest_a = feature_contract_digest(first)
    second = write_onchain_manifest(tmp_path, registry_version="1.0.1")
    digest_b = feature_contract_digest(second)
    assert digest_a != digest_b
```

- [ ] **Step 4: Implement deterministic feature-contract identity**

Hash the ordered thirty feature names, `RULE_FEATURE_SOURCE_MODE`, registry
version, transform specification, and on-chain manifest contract digest. Add it
to Phase 2 resume identity, MTF archive identity, dataset manifest, and
experiment ledger records. A mismatch must reject all cached splits, Phase 2
pools, LWC archives, and RB outputs.

- [ ] **Step 5: Run targeted tests and commit**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_multi_timeframe.py \
  tests/unit/test_research_integrity.py \
  tests/unit/test_run_pipeline.py -q
git add gpu_fuzzy_trader/data/multi_timeframe.py \
  gpu_fuzzy_trader/research_integrity.py gpu_fuzzy_trader/run_pipeline.py \
  tests/unit/test_multi_timeframe.py tests/unit/test_research_integrity.py \
  tests/unit/test_run_pipeline.py
git commit -m "fix: bind MTF artifacts to the on-chain feature contract"
```

---

### Task 8: Make Global and Specialist Phase 2 Modes Explicit

**Files:**
- Create: `tests/unit/test_phase2_symbol_modes.py`
- Modify: `gpu_fuzzy_trader/config.py`
- Modify: `gpu_fuzzy_trader/run_pipeline.py`
- Modify: `gpu_fuzzy_trader/phases/phase2_rule_pool.py`

**Interfaces:**
- Produces: `PHASE2_SYMBOL_MODE`, `_run_phase2_global()`, `_run_phase2_specialists()`, and immutable `source_symbols` rule provenance.

- [ ] **Step 1: Write the failing config and routing tests**

```python
def test_invalid_symbol_mode_fails_config(monkeypatch) -> None:
    monkeypatch.setattr(config, "PHASE2_SYMBOL_MODE", "implicit")
    with pytest.raises(ValueError, match="PHASE2_SYMBOL_MODE"):
        config.validate_config()


def test_specialist_mode_runs_each_symbol_and_direction(monkeypatch) -> None:
    calls = capture_generator_calls(monkeypatch)
    run_phase2_with_fixture(mode="specialist")
    assert {(call.symbols, call.direction) for call in calls} == {
        (("BTCUSDT",), "long"), (("BTCUSDT",), "short"),
        (("ETHUSDT",), "long"), (("ETHUSDT",), "short"),
    }
```

- [ ] **Step 2: Add explicit config without choosing a winner**

```python
PHASE2_SYMBOL_MODE = "global"
PHASE2_SYMBOL_MODE_CHOICES = ("global", "specialist")
```

The initial default remains `global` to preserve behavior until Task 10 runs the
registered comparison. Logs and manifests must always show the resolved mode.

- [ ] **Step 3: Extract current global execution unchanged**

Move the current one-generator-per-direction path into
`_run_phase2_global`. Preserve resume identity and output schema.

- [ ] **Step 4: Implement specialist routing with independent data slices**

For each symbol and direction, slice train, validation, and CV folds before
constructing `Rule_Pool_Generator`. Provide the matching
`config.island_hyperparams(n_rows=...)`. Attach:

```python
rule["source_symbols"] = [symbol]
rule["island_symbols"] = [symbol]
rule["phase2_symbol_mode"] = "specialist"
```

Fail closed if provenance is missing or a rule carries more than its one source
symbol. Do not migrate candidates across symbols in v1. Cross-symbol migration
would mix thresholds whose feature distributions differ.

- [ ] **Step 5: Test RB receives both specialist pools without rewriting provenance**

```python
def test_specialist_provenance_survives_phase2_and_rb(monkeypatch) -> None:
    pools = run_specialist_fixture(monkeypatch)
    assert {tuple(rule["source_symbols"]) for rule in pools["long"]} == {
        ("BTCUSDT",), ("ETHUSDT",)
    }
    selected = run_rb_fixture(pools)
    assert all(len(rule["source_symbols"]) == 1 for rule in selected)
```

- [ ] **Step 6: Run and commit**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_phase2_symbol_modes.py \
  tests/unit/test_rb_island_source_symbols.py \
  tests/unit/test_run_pipeline.py -q
git add gpu_fuzzy_trader/config.py gpu_fuzzy_trader/run_pipeline.py \
  gpu_fuzzy_trader/phases/phase2_rule_pool.py \
  tests/unit/test_phase2_symbol_modes.py
git commit -m "feat: make Phase 2 symbol mode explicit"
```

---

### Task 9: Register RB Diversity Profile and Controlled Ablations

**Files:**
- Create: `scripts/run_onchain_ablation.py`
- Create: `tests/unit/test_onchain_ablation.py`
- Modify: `gpu_fuzzy_trader/config.py`
- Modify: `gpu_fuzzy_trader/rb_governor.py`
- Modify: `gpu_fuzzy_trader/optuna_search.py`

**Interfaces:**
- Produces: `ONCHAIN_FEATURE_FAMILIES`, `RB_PROFILE`, registered variant IDs, fixed seed sets, paired advancement report.

- [ ] **Step 1: Write the failing experiment-matrix test**

```python
def test_registered_variants_do_not_include_test_data() -> None:
    variants = registered_variants()
    assert {variant.name for variant in variants} >= {
        "generated_lwc_baseline", "supplied_18", "onchain_12", "hybrid_30"
    }
    assert all("test" not in variant.selection_splits for variant in variants)
    assert all("forward" not in variant.selection_splits for variant in variants)
```

- [ ] **Step 2: Define immutable feature families and variants**

Use the six two-feature families from the design spec. Register:

- generated LWC technical baseline
- supplied eighteen baseline
- on-chain twelve only
- full hybrid thirty
- six full-hybrid leave-one-family-out variants
- full hybrid in global and specialist symbol modes
- finalist with historical RB profile
- finalist with on-chain diversity RB profile

- [ ] **Step 3: Add an explicit RB profile resolver**

```python
RB_PROFILE = "historical"
RB_PROFILES = {
    "historical": {},
    "onchain_diverse_v1": {
        "RB_CORRELATION_AWARE_SELECTION": True,
        "RB_REDUNDANCY_PENALTY": "low",
        "RB_MARGINAL_PRUNING": True,
        "RB_RULESET_MUST_BEAT_SUBSETS": True,
    },
}
```

Resolve this once at run start and record it. Do not mutate module-level config
inside RB. Remove the current silent canonical policy override for settings
that belong to the selected profile.

- [ ] **Step 4: Implement fixed seeds and paired advancement gate**

Use screening seeds `(11, 42, 73, 101, 137)` and finalist seeds
`(11, 23, 42, 73, 101, 137, 173, 211, 251, 307)`.

```python
def advances(candidate_rows, baseline_rows):
    paired = pair_by_seed(candidate_rows, baseline_rows)
    deltas = [row.candidate_score - row.baseline_score for row in paired]
    return {
        "pass": (
            sum(delta > 0 for delta in deltas) >= 4
            and np.median(deltas) > 0
            and min(row.candidate_return for row in paired)
                >= min(row.baseline_return for row in paired)
            and all(row.cost_stress_pass for row in paired)
            and all(row.symbol_coverage_pass for row in paired)
        ),
        "paired_deltas": deltas,
    }
```

Do not add test metrics to `candidate_score` or `advances`.

- [ ] **Step 5: Test test-path rejection and append-only experiment recording**

```python
def test_ablation_runner_rejects_test_as_selection_input(tmp_path) -> None:
    with pytest.raises(ValueError, match="selection data"):
        run_ablation(selection_path=tmp_path / "test_new.csv")
```

Record variant, seed, data hashes, feature contract, symbol mode, RB profile,
Phase 2 budget, and all reported metrics in the existing experiment ledger.

- [ ] **Step 6: Run and commit**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_onchain_ablation.py \
  tests/unit/test_rb_correlation.py \
  tests/unit/test_marginal.py \
  tests/unit/test_optuna_search.py -q
git add scripts/run_onchain_ablation.py tests/unit/test_onchain_ablation.py \
  gpu_fuzzy_trader/config.py gpu_fuzzy_trader/rb_governor.py \
  gpu_fuzzy_trader/optuna_search.py
git commit -m "feat: register on-chain ablation and RB profiles"
```

---

### Task 10: Build the Real Dataset and Prove the Data Contract

**Files:**
- Modify generated data: `data/train_new.csv`, `data/test_new.csv`
- Create generated manifest: `data/onchain_feature_manifest.json`

**Interfaces:**
- Consumes: corrected twenty-four-file Santiment directory.
- Produces: verified thirty-feature train and test tapes.

- [ ] **Step 1: Extract sources to an ignored directory**

```bash
mkdir -p data/onchain/source-v1
unzip -q '/absolute/path/Onchain-Data(1).zip' -d data/onchain/source-v1
```

Confirm `git status --short` does not list the extracted CSVs.

- [ ] **Step 2: Build staged outputs without replacement**

```bash
.venv/bin/python scripts/build_onchain_dataset.py \
  --source-dir data/onchain/source-v1 \
  --train data/train_new.csv \
  --test data/test_new.csv \
  --output-train data/train_new.onchain.staged.csv \
  --output-test data/test_new.onchain.staged.csv \
  --manifest data/onchain_feature_manifest.json
```

- [ ] **Step 3: Inspect the manifest acceptance fields**

Require:

```text
source files = 24
new feature count = 12
total supplied feature count = 30
BTC Funding imputed rows = 25
ETH Funding imputed rows = 25
unexpected gaps = 0
duplicate target keys = 0
changed original values = 0
finite range violations = 0
```

- [ ] **Step 4: Run a suffix-mutation replay on copied source data**

Change only source observations after `2026-07-23 21:45` in a temporary source
copy, rebuild, and assert both train and test output hashes are unchanged. This
proves that post-test Santiment data does not affect earlier features.

- [ ] **Step 5: Atomically replace the two target CSVs**

Run the builder again with `--replace` only after Steps 2 through 4 pass:

```bash
.venv/bin/python scripts/build_onchain_dataset.py \
  --source-dir data/onchain/source-v1 \
  --train data/train_new.csv \
  --test data/test_new.csv \
  --output-train data/train_new.csv \
  --output-test data/test_new.csv \
  --manifest data/onchain_feature_manifest.json \
  --replace
```

- [ ] **Step 6: Verify checked-in data shape and immutable original columns**

```bash
.venv/bin/python scripts/build_onchain_dataset.py \
  --source-dir data/onchain/source-v1 \
  --train data/train_new.csv \
  --test data/test_new.csv \
  --manifest data/onchain_feature_manifest.json \
  --verify-only
```

Expected train rows: 140352. Expected test rows: 39152. Expected total columns:
37, consisting of seven base columns and thirty supplied features.

- [ ] **Step 7: Commit generated tapes and manifest separately**

```bash
git add data/train_new.csv data/test_new.csv data/onchain_feature_manifest.json
git commit -m "data: add causal Santiment features"
```

Keep this commit separate because it is large and must be reviewable or
revertible without reverting runtime code.

---

### Task 11: Run Targeted Regression, CPU-GPU Parity, and Debug Pipeline

**Files:**
- No new source files unless a failing test exposes a defect.

**Interfaces:**
- Produces: test and smoke-run evidence for the frozen feature contract.

- [ ] **Step 1: Run all data, feature, split, and MTF tests in low-memory mode**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_onchain_registry.py \
  tests/unit/test_onchain_pipeline.py \
  tests/unit/test_onchain_dataset_builder.py \
  tests/unit/test_onchain_feature_contract.py \
  tests/unit/test_data_loader.py \
  tests/unit/test_data_splitter.py \
  tests/unit/test_feature_catalog.py \
  tests/unit/test_fuzzy_scaling.py \
  tests/unit/test_multi_timeframe.py \
  tests/unit/test_mtf_pipeline_integration.py -q
```

- [ ] **Step 2: Run rule-engine parity tests**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_cpu_engine.py \
  tests/unit/test_gpu_engine.py \
  tests/unit/test_phase2_batch_evaluator_parity.py \
  tests/property/test_encoder_properties.py -q
```

Add a fixture with all twelve new features to parity tests if they currently
exercise only binary or ordinal inputs.

- [ ] **Step 3: Run research-integrity and RB tests**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest \
  tests/unit/test_research_integrity.py \
  tests/unit/test_experiment_ledger.py \
  tests/unit/test_multiplicity.py \
  tests/unit/test_rb_correlation.py \
  tests/unit/test_marginal.py \
  tests/unit/test_phase2_symbol_modes.py \
  tests/unit/test_onchain_ablation.py -q
```

- [ ] **Step 4: Run a one-symbol, small-budget end-to-end smoke test**

Use environment overrides that do not change production config:

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python - <<'PY'
from gpu_fuzzy_trader import config
from gpu_fuzzy_trader.run_pipeline import Pipeline_Orchestrator

config.DEBUG_SYMBOL_SCOPE_ENABLED = True
config.DEBUG_SYMBOL = "BTCUSDT"
config.DEBUG_SYMBOL_COUNT = 1
config.PHASE2_POPULATION_SIZE = 32
config.PHASE2_GENERATIONS = 2
Pipeline_Orchestrator(output_dir="outputs/onchain_smoke").run(force=True)
PY
```

Inspect logs and manifests. Require `rule_features=30`, no range violation, no
stale archive reuse, and no read of `test_new.csv` before Phase 5.

- [ ] **Step 5: Run a CUDA smoke test on the target GPU host**

Run the same debug scope with GPU enabled. Require CPU-GPU mask and aggregate
metric parity within existing tolerances. Record hardware, JAX version, peak
memory, and elapsed time in the experiment ledger.

- [ ] **Step 6: Commit only verified defect fixes**

For each verified defect, stage the exact source and regression-test paths from
the failing test output. Review them with `git diff --name-only --cached`, then
commit:

```bash
git diff --name-only --cached
git commit -m "fix: close on-chain integration regressions"
```

Do not create this commit when verification makes no source changes.

---

### Task 12: Select the Research Profile Without Test Tuning

**Files:**
- Generated outputs under a new run-specific output directory.
- Modify only after evidence: `gpu_fuzzy_trader/config.py`, `README.md`, `RUN.md`.

**Interfaces:**
- Consumes: registered variants and development-only splits.
- Produces: frozen feature, symbol-mode, RB, and Phase 2 profile plus a rejection or advancement report.

- [ ] **Step 1: Run five-seed screening variants**

```bash
.venv/bin/python scripts/run_onchain_ablation.py \
  --stage screening \
  --seeds 11 42 73 101 137 \
  --selection-split validation_fitness \
  --output-root outputs/onchain_screening_v1
```

Do not change Phase 2 population or generations between feature variants.

- [ ] **Step 2: Apply the paired advancement gate**

Reject any candidate that fails four-of-five positive paired score deltas,
positive median delta, worst-seed non-degradation, cost-stress, or symbol
coverage. A rejected on-chain family stays out even if one seed has a large
gain.

- [ ] **Step 3: Run ten-seed finalists**

```bash
.venv/bin/python scripts/run_onchain_ablation.py \
  --stage finalist \
  --seeds 11 23 42 73 101 137 173 211 251 307 \
  --selection-split validation_fitness \
  --output-root outputs/onchain_finalists_v1
```

- [ ] **Step 4: Freeze the winner before consuming validation selection**

Write `profiles/onchain_v1.json` containing feature list, registry digest, symbol
mode, RB profile, Phase 2 hyperparameters, seeds, cost model, TP/SL identity,
and code commit. Hash it and append it to the experiment ledger.

- [ ] **Step 5: Evaluate the frozen winner once on validation selection**

No retry, threshold adjustment, family change, or hyperparameter change is
allowed after this evaluation. If it fails, record rejection and begin a new
research generation with a new untouched selection window. Do not tune against
the failure.

- [ ] **Step 6: Update defaults only when the registered winner passes**

Set the winning `PHASE2_SYMBOL_MODE` and `RB_PROFILE` in config. If the hybrid
thirty-feature variant does not beat the supplied-eighteen baseline, keep the
baseline production profile and retain the on-chain builder as an experimental
dataset path. Do not force all twelve features into production for cosmetic
feature count.

- [ ] **Step 7: Commit the frozen development decision**

```bash
git add gpu_fuzzy_trader/config.py profiles/onchain_v1.json
git commit -m "research: freeze validated on-chain strategy profile"
```

---

### Task 13: Document the Truthful Workflow and Perform Final Verification

**Files:**
- Modify: `README.md`
- Modify: `RUN.md`

**Interfaces:**
- Produces: reproducible operator instructions and final verification evidence.

- [ ] **Step 1: Update README contracts**

Document:

- thirty supplied features versus HWC/MWC context
- `supplied_ff` and `generated_lwc` modes
- metric-specific release timing
- Funding past-only capped carry
- exact global versus specialist behavior
- test as consumed diagnostic only
- forward tape as the only release-acceptance input

Remove the unconditional claim that production already uses specialist islands
unless Task 12 selected that mode.

- [ ] **Step 2: Add exact RUN commands**

Include the dataset build, verify-only, screening, finalist, debug pipeline,
full GPU pipeline, diagnostic test, and one-shot forward commands. State which
commands consume a protected split.

- [ ] **Step 3: Run documentation and config smoke checks**

```bash
.venv/bin/python -m gpu_fuzzy_trader.run_pipeline --help
.venv/bin/python -c "from gpu_fuzzy_trader import config; config.validate_config(); print(len(config.RULE_ALLOWED_FF_FEATURES))"
git diff --check
```

Expected feature count: `30`.

- [ ] **Step 4: Run the broad low-memory suite**

```bash
PYTEST_LOW_MEMORY=1 .venv/bin/python -m pytest tests/unit tests/property -q
```

Run benchmark and CUDA tests separately on the GPU host. Do not report them as
passing if the target hardware was not tested.

- [ ] **Step 5: Review generated manifests and repository status**

Require no extracted raw Santiment CSV in git, no staged CSV, no unexpected
cache artifact, a clean `git diff --check`, and manifests whose hashes match the
committed tapes and frozen profile.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md RUN.md
git commit -m "docs: document causal on-chain research workflow"
```

---

## Execution Order and Stop Conditions

Execute Tasks 1 through 7 before building real datasets. Stop immediately if:

- any non-Funding source gap appears
- the Funding gap exceeds eight hours
- a source suffix mutation changes an earlier feature
- daily data becomes visible before D+1 02:00 UTC
- any finite on-chain feature is outside `[-1, 1]`
- any target row, OHLCV value, or original `ff_*` value changes
- the production catalog is not exactly thirty features
- a cache survives a feature-contract digest change
- an experiment reads test or forward data for selection

Tasks 8 and 9 make competing strategy profiles explicit. Task 12 selects among
them using development-only evidence. A good software result is a verified,
causal pipeline. A profitable result is not guaranteed and must not be created
by repeated tuning against `test_new.csv`.

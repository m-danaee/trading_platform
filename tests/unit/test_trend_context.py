import numpy as np
import json

import pandas as pd
import pytest

from gpu_fuzzy_trader import config as cfg
from gpu_fuzzy_trader.data import trend_context as tc

_B = cfg.CONTEXT_STATE_CODES["bearish"]
_R = cfg.CONTEXT_STATE_CODES["range"]
_U = cfg.CONTEXT_STATE_CODES["bullish"]
_N = cfg.CONTEXT_STATE_CODES["noisy"]


_DRIFT = {"AA": 0.002, "BB": -0.001}


def _raw_tape(n=1500, seed=7):
    idx = pd.date_range("2024-01-01", periods=n, freq="15min")
    rows = []
    for sym in ("AA", "BB"):
        rng = np.random.default_rng(seed + abs(hash(sym)) % 100)
        volume = rng.uniform(10, 100, n)
        close = 100 * np.exp(np.cumsum(rng.normal(_DRIFT.get(sym, 0.0), 0.01, n)))
        openp = np.concatenate([[100.0], close[:-1]])
        high = np.maximum(openp, close) * (1 + rng.uniform(0, 0.002, n))
        low = np.minimum(openp, close) * (1 - rng.uniform(0, 0.002, n))
        rawf = rng.normal(0, 1, n)
        for i in range(n):
            rows.append((idx[i], sym, float(openp[i]), float(high[i]), float(low[i]), float(close[i]), float(volume[i]), float(rawf[i])))
    return pd.DataFrame(rows, columns=["datetime", "symbol", "open", "high", "low", "close", "volume", "ff_raw"])


def _enriched():
    raw = _raw_tape()
    th = tc.fit_all_thresholds(raw)
    return tc.generate_enriched_frame(raw, th)


class TestContextContract:
    def test_selected_lookback_and_threshold_quantiles_are_frozen(self):
        contract = cfg.context_contract()

        assert cfg.LWC_PULLBACK_LOOKBACK == 24
        assert contract["lwc_pullback_lookback"] == 24
        assert contract["algorithm_version"] == "regime_v6_ungated_pullback_print"
        assert contract["permission_policy"]["mwc_range_allowed"] is True
        assert contract["trigger_policy"][
            "require_permission_on_pullback_print"
        ] is False
        assert contract["efficiency_trend_threshold_quantile"] == 0.60
        assert contract["ema_spread_trend_threshold_quantile"] == 0.60
        assert contract["volatility_compression_quantile"] == 0.40

    def test_default_pullback_window_uses_previous_24_states(self):
        lwc = np.array([_B] + [_R] * 23 + [_U, _U], dtype=np.int8)
        idx = pd.RangeIndex(len(lwc))
        hwc = pd.Series(np.full(len(lwc), _U, dtype=np.int8), index=idx)
        mwc = pd.Series(np.full(len(lwc), _U, dtype=np.int8), index=idx)
        symbols = pd.Series(["AA"] * len(lwc), index=idx)

        out = tc.compute_permissions_and_triggers(
            hwc, mwc, pd.Series(lwc, index=idx), symbols,
        )

        assert out["lwc_pullback_reversal_long"].iloc[24] == 1
        assert out["lwc_pullback_reversal_long"].iloc[25] == 0

    def test_mwc_range_allows_directional_reentry(self):
        idx = pd.RangeIndex(2)
        symbols = pd.Series(["AA", "AA"], index=idx)
        lwc_long = pd.Series([_B, _U], index=idx)
        long_out = tc.compute_permissions_and_triggers(
            pd.Series([_U, _U], index=idx),
            pd.Series([_R, _R], index=idx),
            lwc_long,
            symbols,
            lookback=1,
        )
        assert long_out["tf_permission_long"].tolist() == [1, 1]
        assert long_out["lwc_pullback_reversal_long"].tolist() == [0, 1]

        lwc_short = pd.Series([_U, _B], index=idx)
        short_out = tc.compute_permissions_and_triggers(
            pd.Series([_B, _B], index=idx),
            pd.Series([_R, _R], index=idx),
            lwc_short,
            symbols,
            lookback=1,
        )
        assert short_out["tf_permission_short"].tolist() == [1, 1]
        assert short_out["lwc_pullback_reversal_short"].tolist() == [0, 1]


class TestThresholdDeterminism:
    def test_reproducible_thresholds(self):
        t1 = tc.fit_thresholds(_raw_tape(n=2000))
        t2 = tc.fit_thresholds(_raw_tape(n=2000))
        assert list(t1) == list(t2)
        for k in t1:
            assert t1[k] == t2[k], k

    def test_exclusive_permissions(self):
        out, _ = _enriched()
        for sym, grp in out.groupby("symbol"):
            both = grp[cfg.context_permission_column("long")] + grp[cfg.context_permission_column("short")]
            assert (both <= 1).all(), "long/short permission must be mutually exclusive"


class TestTrainPrefixOnlyFitting:
    """Regression: thresholds must never see validation-period rows."""

    def test_validation_tail_volatility_does_not_change_thresholds(self):
        raw = _raw_tape(n=2000)
        baseline_prefix = tc.build_train_prefix(raw)
        baseline = tc.fit_all_thresholds(baseline_prefix)

        perturbed = raw.copy()
        rng = np.random.default_rng(0)
        for _symbol, g in raw.groupby("symbol", observed=False):
            n = len(g)
            train_end = cfg.train_prefix_row_count(n)
            tail_idx = g.sort_values("datetime").index[train_end:]
            shock = 1.0 + rng.normal(0, 5.0, len(tail_idx))
            perturbed.loc[tail_idx, "close"] = perturbed.loc[tail_idx, "close"] * shock
            perturbed.loc[tail_idx, "high"] = np.maximum(
                perturbed.loc[tail_idx, "high"], perturbed.loc[tail_idx, "close"])
            perturbed.loc[tail_idx, "low"] = np.minimum(
                perturbed.loc[tail_idx, "low"], perturbed.loc[tail_idx, "close"])

        perturbed_prefix = tc.build_train_prefix(perturbed)
        assert perturbed_prefix.reset_index(drop=True).equals(
            baseline_prefix.reset_index(drop=True)
        ), "perturbing only the validation tail must not change the train prefix"

        perturbed_thresholds = tc.fit_all_thresholds(perturbed_prefix)
        for tf in ("lwc", "mwc", "hwc"):
            for key in (
                "efficiency_abs_trend_threshold",
                "ema_spread_abs_trend_threshold",
                "volatility_compression_threshold",
            ):
                assert baseline[tf][key] == perturbed_thresholds[tf][key], (tf, key)


class TestPerTimeframeThresholds:
    def test_thresholds_are_fitted_independently_per_timeframe(self):
        raw = _raw_tape(n=3000)
        thresholds = tc.fit_all_thresholds(raw)
        assert set(thresholds) == {"lwc", "mwc", "hwc"}
        # Realized-volatility (and efficiency/spread) distributions shift with
        # bar duration, so a single 15m-fitted set must not be reused as-is.
        assert (
            thresholds["lwc"]["volatility_compression_threshold"]
            != thresholds["mwc"]["volatility_compression_threshold"]
        )
        assert (
            thresholds["mwc"]["volatility_compression_threshold"]
            != thresholds["hwc"]["volatility_compression_threshold"]
        )


class TestIncompleteBoundaryBars:
    def test_build_higher_bars_drops_incomplete_boundary_bucket(self):
        # Tape starts at 05:00, so the 04:00-08:00 4h bucket only has 12 of
        # the required 16 15m candles and must never be treated as complete.
        times = pd.date_range("2024-01-01 05:00", periods=20, freq="15min")
        df = pd.DataFrame({
            "datetime": times,
            "symbol": "X",
            "open": 100.0, "high": 101.0, "low": 99.0,
            "close": 100.0, "volume": 1.0,
        })
        bars = tc.build_higher_bars(df, 240)
        assert pd.Timestamp("2024-01-01 04:00") not in set(bars["datetime"])
        assert pd.Timestamp("2024-01-01 08:00") not in set(bars["datetime"])

        raw_bars = tc.build_higher_bars(df, 240, require_complete=False)
        assert pd.Timestamp("2024-01-01 04:00") in set(raw_bars["datetime"])


class TestPermissionGatedPullback:
    def test_default_allows_pullback_without_historical_permission(self):
        """Default v6: opposite LWC print counts even if permission was off."""
        idx = pd.RangeIndex(3)
        symbols = pd.Series(["AA"] * 3, index=idx)
        hwc = pd.Series([_B, _U, _U], index=idx)
        mwc = pd.Series([_B, _U, _U], index=idx)
        lwc = pd.Series([_B, _R, _U], index=idx)
        assert cfg.CONTEXT_REQUIRE_PERMISSION_ON_PULLBACK_PRINT is False
        out = tc.compute_permissions_and_triggers(
            hwc, mwc, lwc, symbols, lookback=2)
        assert out["tf_permission_long"].tolist() == [0, 1, 1]
        assert out["lwc_pullback_reversal_long"].iloc[2] == 1

    def test_gated_mode_requires_permission_at_the_historical_bar(self, monkeypatch):
        monkeypatch.setattr(
            cfg, "CONTEXT_REQUIRE_PERMISSION_ON_PULLBACK_PRINT", True)
        idx = pd.RangeIndex(3)
        symbols = pd.Series(["AA"] * 3, index=idx)

        # Bearish LWC print occurs while long permission is OFF; permission
        # then turns on; current LWC turns Bullish. The stale bearish print
        # from before permission was active must not count when gated.
        hwc = pd.Series([_B, _U, _U], index=idx)
        mwc = pd.Series([_B, _U, _U], index=idx)
        lwc = pd.Series([_B, _R, _U], index=idx)
        out = tc.compute_permissions_and_triggers(hwc, mwc, lwc, symbols, lookback=2)
        assert out["tf_permission_long"].tolist() == [0, 1, 1]
        assert out["lwc_pullback_reversal_long"].iloc[2] == 0

        # Same LWC path, but permission is active throughout: the Bearish
        # print now occurred during an active uptrend and must trigger.
        hwc2 = pd.Series([_U, _U, _U], index=idx)
        mwc2 = pd.Series([_U, _U, _U], index=idx)
        out2 = tc.compute_permissions_and_triggers(hwc2, mwc2, lwc, symbols, lookback=2)
        assert out2["lwc_pullback_reversal_long"].iloc[2] == 1


class TestLoaderContext:
    def test_loader_keeps_enriched_columns(self, tmp_path):
        from gpu_fuzzy_trader.data.loader import Data_Loader

        out, _ = _enriched()
        p = tmp_path / "train.csv"
        out.to_csv(p, index=False)
        df = Data_Loader().load_dataset(str(p), drop_tail=True, include_barrier_outcomes=True)
        assert len(df) > 0
        for col in cfg.CONTEXT_COLUMNS:
            assert col in df.columns

    def test_loader_rejects_invalid_state_code(self, tmp_path):
        from gpu_fuzzy_trader.data.loader import Data_Loader

        out, _ = _enriched()
        out.loc[out.index[0], cfg.CONTEXT_COLUMNS[0]] = 99
        p = tmp_path / "bad.csv"
        out.to_csv(p, index=False)
        
        with pytest.raises(ValueError):
            Data_Loader().load_dataset(str(p), drop_tail=True, include_barrier_outcomes=True)

    def test_loader_accepts_mwc_range_permission(self):
        """Regression: the loader's truth table must match the actual policy
        (CONTEXT_ALLOW_MWC_RANGE_PERMISSION), not a stale strict variant."""
        from gpu_fuzzy_trader.data.loader import validate_context_columns

        n = 30
        symbols = pd.Series(["AA"] * n)
        hwc = pd.Series([_U] * n)
        mwc = pd.Series([_R] * n)
        lwc = pd.Series(([_B, _U] * ((n // 2) + 1))[:n])
        exec_cols = tc.compute_permissions_and_triggers(
            hwc, mwc, lwc, symbols, lookback=5)
        assert int(exec_cols["tf_permission_long"].sum()) == n

        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "symbol": symbols,
            "hwc_state": hwc, "mwc_state": mwc, "lwc_state": lwc,
            **{c: exec_cols[c] for c in exec_cols.columns},
        })
        validate_context_columns(df)

    def test_loader_rejects_corrupted_trigger(self, tmp_path):
        from gpu_fuzzy_trader.data.loader import Data_Loader

        lookback = cfg.LWC_PULLBACK_LOOKBACK
        n = lookback + 10
        symbols = pd.Series(["AA"] * n)
        hwc = pd.Series([_U] * n)
        mwc = pd.Series([_U] * n)
        # Bearish print at position 0 (permission active throughout), then
        # Range until the guaranteed trigger row at index == lookback, whose
        # bar_index (== lookback) is itself checkable by the loader.
        lwc_vals = [_R] * n
        lwc_vals[0] = _B
        lwc_vals[lookback] = _U
        lwc = pd.Series(lwc_vals)
        exec_cols = tc.compute_permissions_and_triggers(hwc, mwc, lwc, symbols)
        assert exec_cols["lwc_pullback_reversal_long"].iloc[lookback] == 1

        out = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
            "symbol": symbols,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0,
            "volume": 1.0,
            "hwc_state": hwc, "mwc_state": mwc, "lwc_state": lwc,
            **{c: exec_cols[c] for c in exec_cols.columns},
        })
        trig_col = cfg.context_trigger_column("long")
        out.loc[lookback, trig_col] = 0  # corrupt the guaranteed trigger
        p = tmp_path / "corrupt_trigger.csv"
        out.to_csv(p, index=False)
        with pytest.raises(ValueError, match="does not match the recomputed"):
            Data_Loader().load_dataset(
                str(p), drop_tail=False, include_barrier_outcomes=False)


class TestInputAndBoundaryContracts:
    def test_regular_but_off_grid_timestamps_are_rejected(self):
        frame = pd.DataFrame({
            "datetime": pd.date_range(
                "2024-01-01 00:07", periods=4, freq="15min"),
            "symbol": "AA",
        })
        with pytest.raises(ValueError, match="aligned to the 15m grid"):
            tc.validate_input_frame(frame)

    def test_enrich_tape_emits_exact_staggered_target_keys(self):
        raw = _raw_tape(n=120)
        starts = {"AA": 60, "BB": 70}
        history_parts = []
        target_parts = []
        for symbol, group in raw.groupby("symbol", sort=False):
            group = group.sort_values("datetime")
            cut = starts[symbol]
            history_parts.append(group.iloc[:cut])
            target_parts.append(group.iloc[cut:])
        history = pd.concat(history_parts, ignore_index=True)
        target = pd.concat(target_parts, ignore_index=True)
        # Threshold fitting needs enough history to build >CONTEXT_STRUCTURAL_
        # LOOKBACK complete HWC/MWC bars; use an independent larger tape so
        # this test stays focused on enrich_tape's target-key contract.
        thresholds = tc.fit_all_thresholds(_raw_tape(n=2000))

        enriched = tc.enrich_tape(target, history, thresholds)
        expected = set(zip(target["datetime"], target["symbol"]))
        actual = set(zip(enriched["datetime"], enriched["symbol"]))
        assert actual == expected
        assert len(enriched) == len(target)

    def test_manifest_records_each_actual_raw_source(self, tmp_path):
        raw_paths = {}
        enriched_paths = {}
        for name in ("train", "test", "forward"):
            raw_path = tmp_path / f"raw_{name}.csv"
            enriched_path = tmp_path / f"enriched_{name}.csv"
            pd.DataFrame({
                "datetime": [f"2024-01-0{len(raw_paths) + 1}"],
                "symbol": [name],
            }).to_csv(raw_path, index=False)
            enriched_path.write_text(f"enriched-{name}\n", encoding="utf-8")
            raw_paths[name] = str(raw_path)
            enriched_paths[name] = str(enriched_path)

        manifest = tc.build_manifest(
            {"threshold": 1.0},
            train_source=raw_paths["train"],
            tapes=enriched_paths,
            tape_sources=raw_paths,
        )
        for name in raw_paths:
            source = manifest["tapes"][name]["source"]
            assert source["path"] == raw_paths[name]
            assert source["sha256"] == tc.sha256_file(raw_paths[name])


class TestTriggerPerSymbol:
    """Regression: symbols arrive interleaved (sorted by datetime then symbol),
    so triggers must be computed per symbol via grouping, never via contiguous
    row ranges.  The old ``_symbol_index_ranges`` approach zeroed all triggers
    on interleaved rows and could cross symbol history if rows were contiguous
    but mis-sorted."""

    def test_interleaved_symbols_produce_expected_triggers(self):
        # Long permission on for every row (HWC/MWC all Bullish).
        n_a = 6
        n_b = 6
        lwc_a = np.array([_B, _B, _B, _U, _U, _U], dtype=np.int8)
        lwc_b = np.array([_U, _U, _U, _B, _B, _B], dtype=np.int8)
        symbols = []
        lwc = []
        for i in range(max(n_a, n_b)):
            if i < n_a:
                symbols.append("A")
                lwc.append(lwc_a[i])
            if i < n_b:
                symbols.append("B")
                lwc.append(lwc_b[i])

        n = len(lwc)
        idx = pd.RangeIndex(n)
        hwc = pd.Series(np.full(n, _U, dtype=np.int8), index=idx)
        mwc = pd.Series(np.full(n, _U, dtype=np.int8), index=idx)
        lwc_s = pd.Series(np.array(lwc, dtype=np.int8), index=idx)
        sym_s = pd.Series(symbols, index=idx)

        out = tc.compute_permissions_and_triggers(
            hwc, mwc, lwc_s, sym_s, lookback=3)

        trig_long = out["lwc_pullback_reversal_long"].to_numpy()
        trig_short = out["lwc_pullback_reversal_short"].to_numpy()
        # A rows live at global even indices; A's 4th/5th/6th rows are the
        # BULL states preceded by 3 Bearish -> all must trigger.
        a_bull_globals = [2 * i for i in range(3, 6)]
        assert (trig_long[a_bull_globals] == 1).all(), trig_long
        # B rows (odd globals) must never fire the long trigger.
        assert (trig_long[1::2] == 0).all()
        assert int(trig_long.sum()) == 3
        # Default ungated policy: B's Bearish rows with prior Bullish LWC still
        # fire the short timing trigger even though short permission is off
        # (HWC/MWC are Bullish). Tradeable entries still need both columns.
        b_bear_globals = [2 * i + 1 for i in range(3, 6)]
        assert (trig_short[b_bear_globals] == 1).all(), trig_short
        assert (trig_short[0::2] == 0).all()
        assert int(trig_short.sum()) == 3
        # Per-symbol reference (contiguous layout) matches the interleaved one.
        ref = tc.compute_permissions_and_triggers(
            hwc, mwc, lwc_s, sym_s, lookback=3)
        assert (out["tf_permission_long"].to_numpy() == 1).all()
        assert (out["tf_permission_short"].to_numpy() == 0).all()
        assert ref["lwc_pullback_reversal_long"].tolist() == trig_long.tolist()
        assert ref["lwc_pullback_reversal_short"].tolist(
        ) == trig_short.tolist()

    def test_pullback_window_does_not_cross_symbol(self):
        # A B-symbol Bearish row adjacent to an A-symbol row must not enter A's
        # pullback window.  Case 1: A is [BULL, BULL] at globals 0 and 2, B is
        # Bearish at global 1.  A's row at global 2 sees only its own prior
        # BULL -> no trigger, even though a naive array window of size 2 would
        # include B's Bearish at global 1.
        lwc = np.array([_U, _B, _U], dtype=np.int8)
        symbols = ["A", "B", "A"]
        idx = pd.RangeIndex(3)
        hwc = pd.Series(np.full(3, _U, dtype=np.int8), index=idx)
        mwc = pd.Series(np.full(3, _U, dtype=np.int8), index=idx)
        out = tc.compute_permissions_and_triggers(
            hwc, mwc, pd.Series(lwc, index=idx), pd.Series(symbols, index=idx),
            lookback=2)
        trig = out["lwc_pullback_reversal_long"].to_numpy()
        assert trig[0] == 0
        assert trig[1] == 0
        assert trig[2] == 0

        # Case 2: A's own Bearish history (inside A's window) still triggers.
        # A rows at globals 0,2,3 with lwc [_B,_B,_U]; B is Bearish at global 1.
        lwc2 = np.array([_B, _B, _B, _U], dtype=np.int8)
        symbols2 = ["A", "B", "A", "A"]
        idx2 = pd.RangeIndex(4)
        hwc2 = pd.Series(np.full(4, _U, dtype=np.int8), index=idx2)
        mwc2 = pd.Series(np.full(4, _U, dtype=np.int8), index=idx2)
        out2 = tc.compute_permissions_and_triggers(
            hwc2, mwc2, pd.Series(lwc2, index=idx2),
            pd.Series(symbols2, index=idx2), lookback=2)
        trig2 = out2["lwc_pullback_reversal_long"].to_numpy()
        # A's global 3 uses its OWN Bearish at global 2 -> trigger.
        assert trig2[3] == 1
        # B (global 1) is current Bearish -> never triggers a long pullback.
        assert trig2[1] == 0
        assert int(trig2.sum()) == 1


class TestCausalPublicationTiming:
    """Higher-timeframe state is only published after that bar completes and
    aligned to 15m rows with backward-causal semantics.  Monkeypatch
    ``_classify_hf_bars`` to isolate the alignment timing from the indicators."""

    def _rows(self):
        times = pd.date_range("2024-01-01 08:00", "2024-01-01 13:00", freq="15min")
        return pd.DataFrame(
            {
                "datetime": times,
                "symbol": "X",
                "open": 100.0, "high": 101.0, "low": 99.0,
                "close": 100.0, "volume": 10.0,
            }
        )

    def test_one_hour_published_at_close_time(self, monkeypatch):
        rows = self._rows()
        hf = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-01 10:00"]),
                "symbol": "X",
            }
        )
        monkeypatch.setattr(
            tc, "_classify_hf_bars",
            lambda g, th: np.array([_U], dtype=np.int8))
        aligned = tc.align_completed_states_to_rows(
            rows, hf, {}, int(cfg.MWC_TIMEFRAME_MINUTES))
        row_at = {t: s for t, s in zip(rows["datetime"], aligned.to_numpy())}
        # The 10:45 signal row executes at 11:00, so it can use the 1h bar
        # that completed exactly at that next-open execution instant.
        assert row_at[pd.Timestamp("2024-01-01 10:30")] == _N
        assert row_at[pd.Timestamp("2024-01-01 10:45")] == _U
        assert row_at[pd.Timestamp("2024-01-01 11:00")] == _U
        assert row_at[pd.Timestamp("2024-01-01 12:00")] == _U

    def test_four_hour_08_00_bar_published_at_12_00(self, monkeypatch):
        rows = self._rows()
        hf = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2024-01-01 08:00"]),
                "symbol": "X",
            }
        )
        monkeypatch.setattr(
            tc, "_classify_hf_bars",
            lambda g, th: np.array([_B], dtype=np.int8))
        aligned = tc.align_completed_states_to_rows(
            rows, hf, {}, int(cfg.HWC_TIMEFRAME_MINUTES))
        row_at = {t: s for t, s in zip(rows["datetime"], aligned.to_numpy())}
        # The 11:45 signal executes at 12:00 and may use the 08:00-12:00 bar.
        assert row_at[pd.Timestamp("2024-01-01 11:30")] == _N
        assert row_at[pd.Timestamp("2024-01-01 11:45")] == _B
        assert row_at[pd.Timestamp("2024-01-01 12:00")] == _B
        assert row_at[pd.Timestamp("2024-01-01 13:00")] == _B
    def test_cpu_mask_gates_every_rule(self, tmp_path):
        from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
        from gpu_fuzzy_trader.data.loader import Data_Loader

        out, _ = _enriched()
        p = tmp_path / "e.csv"
        out.to_csv(p, index=False)
        df = Data_Loader().load_dataset(str(p), drop_tail=True, include_barrier_outcomes=True)
        eng = CPUBacktestEngine(df, {}, "long")
        perm = df[cfg.context_permission_column("long")].to_numpy().astype(bool)
        trig = df[cfg.context_trigger_column("long")].to_numpy().astype(bool)
        eligible = (perm & trig).sum()
        raw_positive = (df["ff_raw"].to_numpy() > 0).sum()
        rule = {"tp": 5.0, "sl": 5.0, "capital_pct": 10.0, "conditions": ["[ff_raw] IS Positive"]}
        m, _ = eng.simulate_rule_set([rule], return_logs=True)
        # The permissive feature rule fires on ~half the rows; the context mask
        # must cap trades at the eligible (permission & LWC-trigger) count.
        assert m["raw_signal_count"] <= eligible
        assert m["raw_signal_count"] <= raw_positive


class TestWriterContract:
    def test_context_free_rule_rejected(self):
        from gpu_fuzzy_trader.output.writer import _validate_rule_set, ValidationError

        strat = {
            "direction": "long",
            "rules_set": [{
                "tp": 2.0, "sl": 1.0, "capital_pct": 10.0,
                "conditions": ["[ff_raw] IS Positive"],
            }],
        }
        with pytest.raises(ValidationError, match="missing mandatory context"):
            _validate_rule_set(strat)

    def test_opposite_direction_context_rejected(self):
        from gpu_fuzzy_trader.output.writer import _validate_rule_set, ValidationError

        strat = {
            "direction": "short",
            "rules_set": [{
                "tp": 2.0, "sl": 1.0, "capital_pct": 10.0,
                "conditions": [
                    "[ff_raw] IS Positive",
                    "[tf_permission_long] IS Active (1)",
                    "[lwc_pullback_reversal_long] IS Active (1)",
                ],
            }],
        }
        
        with pytest.raises(ValidationError):
            _validate_rule_set(strat)

    def test_declared_context_missing_mandatory_rejected(self):
        from gpu_fuzzy_trader.output.writer import _validate_rule_set, ValidationError

        strat = {
            "direction": "long",
            "rules_set": [{
                "tp": 2.0, "sl": 1.0, "capital_pct": 10.0,
                "conditions": ["[hwc_state] IS Bullish"],
            }],
        }
        
        with pytest.raises(ValidationError):
            _validate_rule_set(strat)


class TestRbContract:
    def test_mandatory_context_survives_strategy(self):
        from gpu_fuzzy_trader import rb_governor as rb

        rules = [{
            "tp": 2.0, "sl": 1.0, "capital_pct": 10.0,
            "conditions": [
                "[feature] IS Positive",
                "[tf_permission_long] IS Active (1)",
                "[lwc_pullback_reversal_long] IS Active (1)",
            ],
        }]
        s = rb._strategy("long", rules)
        assert s["direction"] == "long"
        for cond in cfg.mandatory_context_conditions("long"):
            assert any(cond in c for r in s["rules_set"] for c in r["conditions"])

    def test_missing_permission_fails_closed(self):
        from gpu_fuzzy_trader import rb_governor as rb

        rules = [{
            "tp": 2.0, "sl": 1.0, "capital_pct": 10.0,
            "conditions": ["[feature] IS Positive", "[lwc_pullback_reversal_long] IS Active (1)"],
        }]
        with pytest.raises(AssertionError):
            rb._strategy("long", rules)


class TestContextIdentity:
    def test_manifest_threshold_or_source_change_invalidates_digest(
        self, tmp_path, monkeypatch,
    ):
        manifest_path = tmp_path / "trend_context_manifest.json"
        monkeypatch.setattr(cfg, "ENRICHED_MANIFEST_PATH", str(manifest_path))
        base = {
            "thresholds": {"efficiency_abs_trend_threshold": 0.5},
            "threshold_fitting": {
                "source": {"sha256": "raw-a", "interval": ["a", "b"]},
                "train_only": True,
            },
            "history_sources": {"test": {"sha256": "history-a"}},
            "tapes": {"train": {"sha256": "enriched-a"}},
        }
        manifest_path.write_text(json.dumps(base), encoding="utf-8")
        digest_a = cfg.context_contract_digest()

        changed_threshold = json.loads(json.dumps(base))
        changed_threshold["thresholds"]["efficiency_abs_trend_threshold"] = 0.6
        manifest_path.write_text(json.dumps(changed_threshold), encoding="utf-8")
        digest_b = cfg.context_contract_digest()

        changed_source = json.loads(json.dumps(base))
        changed_source["threshold_fitting"]["source"]["sha256"] = "raw-b"
        manifest_path.write_text(json.dumps(changed_source), encoding="utf-8")
        digest_c = cfg.context_contract_digest()

        assert len({digest_a, digest_b, digest_c}) == 3

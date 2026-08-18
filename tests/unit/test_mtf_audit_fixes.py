"""Unit tests verifying fixes for the 5 MTF audit issues.

1. OOF leakage prevention during Pareto candidate selection.
2. Multi-symbol signal and row alignment between splitter and CPUBacktestEngine.
3. Per-symbol MCC and minimum fold-support hard admission constraints.
4. Consistent MTF mode in run_from_phase2 / run_phase and artifact reuse with --resume.
5. Strict binding of Phase 5 serialized strategy candidates to verified archive payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.data.splitter import _holdout_embargo_split
from gpu_fuzzy_trader.mtf.archives import (
    compute_archive_hash,
    compute_rule_hash,
    save_mtf_rule_archive,
)
from gpu_fuzzy_trader.mtf.candidate import HierarchicalStrategyCandidate
from gpu_fuzzy_trader.mtf.cross_fitting import (
    DEFAULT_HWC_PURGE_MINUTES,
    TemporalFold,
    build_master_temporal_folds,
)
from gpu_fuzzy_trader.mtf.discovery import (
    _directional_pareto_front,
    _passes_cross_symbol_mcc_admission,
    canonicalize_oof_scores,
    discover_directional_layer,
    hash_oof_scores,
)
from gpu_fuzzy_trader.mtf.runtime import (
    evaluate_candidate_frame,
    prepare_causal_mtf_frame,
)
from gpu_fuzzy_trader.phases.phase5_oos import OOS_Evaluator, ValidationError
from gpu_fuzzy_trader.run_pipeline import Pipeline_Runner


def test_oof_leakage_prevention_in_pareto_vector():
    """Verify Pareto vector uses train metrics during candidate search, not test metrics."""
    # Candidate A: high train metrics, poor test metrics
    cand_a = {
        "metrics": {
            "directional_edge": 0.30,
            "mcc": 0.25,
            "coverage_penalty": 0.0,
            "false_confirmation_penalty": 0.0,
            "test_directional_edge": 0.01,
            "test_mcc": 0.01,
            "test_coverage_penalty": 0.5,
            "test_false_confirmation_penalty": 0.5,
        },
        "stability": 1.0,
    }
    # Candidate B: poor train metrics, artificially high test metrics
    cand_b = {
        "metrics": {
            "directional_edge": 0.02,
            "mcc": 0.01,
            "coverage_penalty": 0.5,
            "false_confirmation_penalty": 0.5,
            "test_directional_edge": 0.50,
            "test_mcc": 0.40,
            "test_coverage_penalty": 0.0,
            "test_false_confirmation_penalty": 0.0,
        },
        "stability": 1.0,
    }
    front = _directional_pareto_front([cand_a, cand_b])
    # Candidate A dominates Candidate B on train metrics
    assert cand_a in front
    assert cand_b not in front


def test_multi_symbol_signal_alignment_with_cpu_backtester():
    """Verify symbol-blocked frames maintain exact row-for-row signal alignment in CPU backtest."""
    # 40 BTC bars followed by 40 ETH bars (symbol-blocked, like splitter output)
    n = 40
    dates = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
    btc_df = pd.DataFrame({
        "datetime": dates,
        "symbol": ["BTCUSDT"] * n,
        "open": 100.0 + np.arange(n) * 0.1,
        "high": 101.0 + np.arange(n) * 0.1,
        "low": 99.0 + np.arange(n) * 0.1,
        "close": 100.5 + np.arange(n) * 0.1,
        "volume": 10.0,
        "label_open_next": 100.5 + np.arange(n) * 0.1,
        "label_close_288": 105.0 + np.arange(n) * 0.1,
        "label_min_288": 98.0 + np.arange(n) * 0.1,
        "label_max_288": 106.0 + np.arange(n) * 0.1,
        "label_max_before_min": 1,
    })
    eth_df = pd.DataFrame({
        "datetime": dates,
        "symbol": ["ETHUSDT"] * n,
        "open": 10.0 + np.arange(n) * 0.01,
        "high": 11.0 + np.arange(n) * 0.01,
        "low": 9.0 + np.arange(n) * 0.01,
        "close": 10.5 + np.arange(n) * 0.01,
        "volume": 50.0,
        "label_open_next": 10.5 + np.arange(n) * 0.01,
        "label_close_288": 12.0 + np.arange(n) * 0.01,
        "label_min_288": 8.0 + np.arange(n) * 0.01,
        "label_max_288": 13.0 + np.arange(n) * 0.01,
        "label_max_before_min": 1,
    })
    split_df = pd.concat([btc_df, eth_df], ignore_index=True)

    # Candidate rule matches only BTC (e.g. open >= 50)
    candidate = HierarchicalStrategyCandidate(
        direction="long",
        lwc_rules=[{
            "timeframe": "lwc",
            "direction": "long",
            "conditions": ["[open] >= 50.0"],
            "coverage": 0.5,
        }],
        composer_params={
            "v_hwc_long": 0.65,
            "v_mwc_long": 0.60,
            "min_evidence_strength_hwc": 0.15,
            "min_evidence_strength_mwc": 0.15,
        },
    )

    signals, stats, audit = evaluate_candidate_frame(candidate, split_df)
    assert len(signals) == len(split_df)
    # BTC rows (indices 0..39) should match (open >= 50 is True)
    assert (signals[:n] == 1).all()
    # ETH rows (indices 40..79) should NOT match (open < 50)
    assert (signals[n:] == 0).all()

    # Audit frame symbols must match input split_df row for row
    assert (audit["symbol"].to_numpy() == split_df["symbol"].to_numpy()).all()

    # CPU backtest engine on split_df
    engine = CPUBacktestEngine(split_df, {}, "long")
    metrics, trade_log = engine.simulate_signal_mask(
        signals != 0,
        tp=2.0,
        sl=1.2,
        capital_pct=18.0,
        return_logs=True,
    )
    # All executed trades must be on BTCUSDT, none on ETHUSDT
    if not trade_log.empty:
        assert (trade_log["Symbol"] == "BTCUSDT").all()


def test_cross_symbol_mcc_and_fold_support_admission():
    """Admitted rules must have fold support and strictly positive MCC on every symbol."""
    n = 200
    dates = pd.date_range("2024-01-01 00:00", periods=n, freq="4h")
    btc = pd.DataFrame({
        "datetime": dates,
        "symbol": "BTCUSDT",
        "open": 100.0 + np.sin(np.linspace(0, 10, n)) * 5.0,
        "high": 105.0,
        "low": 95.0,
        "close": 100.0 + np.sin(np.linspace(0, 10, n)) * 5.0,
        "volume": 1000.0,
    })
    eth = pd.DataFrame({
        "datetime": dates,
        "symbol": "ETHUSDT",
        "open": 50.0 + np.cos(np.linspace(0, 10, n)) * 5.0,
        "high": 55.0,
        "low": 45.0,
        "close": 50.0 + np.cos(np.linspace(0, 10, n)) * 5.0,
        "volume": 500.0,
    })
    raw_df = pd.concat([btc, eth], ignore_index=True).sort_values(
        ["datetime", "symbol"]).reset_index(drop=True)

    folds = build_master_temporal_folds(
        raw_df, n_folds=3, embargo_minutes=DEFAULT_HWC_PURGE_MINUTES)
    discovery = discover_directional_layer(raw_df, role="hwc", folds=folds)

    for rule in discovery.rules:
        oof_meta = rule.get("oof_metrics", {})
        assert oof_meta.get("fold_support", 0) >= 2
        fold_metrics = oof_meta.get("fold_metrics", [])
        assert _passes_cross_symbol_mcc_admission(
            fold_metrics, ("BTCUSDT", "ETHUSDT"))


def test_cross_symbol_mcc_zero_rejection_probe():
    """Production admission rejects MCC 0.0 on any required symbol."""
    metrics_pass = [
        {"test_symbol_mccs": {"BTCUSDT": 0.05, "ETHUSDT": 0.05}},
        {"test_symbol_mccs": {"BTCUSDT": 0.06, "ETHUSDT": 0.06}},
    ]
    metrics_fail_zero = [
        {"test_symbol_mccs": {"BTCUSDT": 0.10, "ETHUSDT": 0.0}},
        {"test_symbol_mccs": {"BTCUSDT": 0.12, "ETHUSDT": 0.0}},
    ]
    metrics_fail_nan = [
        {"test_symbol_mccs": {"BTCUSDT": 0.10, "ETHUSDT": float("nan")}},
        {"test_symbol_mccs": {"BTCUSDT": 0.12, "ETHUSDT": 0.05}},
    ]
    symbols = ("BTCUSDT", "ETHUSDT")
    assert _passes_cross_symbol_mcc_admission(metrics_pass, symbols) is True
    assert _passes_cross_symbol_mcc_admission(
        metrics_fail_zero, symbols) is False
    assert _passes_cross_symbol_mcc_admission(
        metrics_fail_nan, symbols) is False


def test_identical_symbols_can_admit_hwc_rules():
    """A constructed two-symbol tape with shared signal must admit at least one HWC rule."""
    n = 1600
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    wave = np.sin(np.linspace(0, 24, n))

    def _symbol_frame(symbol: str, scale: float) -> pd.DataFrame:
        close = 100.0 * scale + wave * 8.0 * scale
        return pd.DataFrame({
            "datetime": dates,
            "symbol": symbol,
            "open": close,
            "high": close + 0.5 * scale,
            "low": close - 0.5 * scale,
            "close": close,
            "volume": 1000.0 * scale,
        })

    raw_df = pd.concat(
        [_symbol_frame("BTCUSDT", 1.0), _symbol_frame("ETHUSDT", 0.5)],
        ignore_index=True,
    ).sort_values(["datetime", "symbol"]).reset_index(drop=True)
    folds = build_master_temporal_folds(
        raw_df, n_folds=3, embargo_minutes=DEFAULT_HWC_PURGE_MINUTES)
    discovery = discover_directional_layer(raw_df, role="hwc", folds=folds)
    assert len(discovery.rules) > 0
    for rule in discovery.rules:
        assert _passes_cross_symbol_mcc_admission(
            rule["oof_metrics"]["fold_metrics"],
            ("BTCUSDT", "ETHUSDT"),
        )


def test_oof_sidecar_hash_survives_json_roundtrip():
    """Saved OOF hash must match the JSON sidecar after reload."""
    oof = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=8, freq="4h"),
        "symbol": ["BTCUSDT", "ETHUSDT"] * 4,
        "direction_score": np.linspace(-0.4, 0.5, 8),
        "strength_score": np.linspace(0.1, 0.8, 8),
        "fold_id": 2,
        "is_seed": False,
    })
    saved = hash_oof_scores(oof)
    payload = json.dumps(oof.to_dict(orient="records"),
                         indent=2, sort_keys=True, default=str)
    loaded = canonicalize_oof_scores(
        pd.DataFrame.from_records(json.loads(payload)))
    assert hash_oof_scores(loaded) == saved


def test_resume_reuses_valid_hwc_mwc_artifacts(tmp_path):
    """Resume reuses a discovered archive and rejects a tampered dataset hash."""
    runner = Pipeline_Runner()
    runner._output_dir = str(tmp_path / "outputs")
    runner._log_path = str(tmp_path / "run.log")
    runner._create_output_dirs()

    n = 1200
    dates = pd.date_range("2024-01-01", periods=n, freq="15min")
    close = 100.0 + np.sin(np.linspace(0, 18, n)) * 4.0
    df = pd.DataFrame({
        "datetime": dates,
        "symbol": ["BTCUSDT"] * n,
        "open": close,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 10.0,
    })
    folds = build_master_temporal_folds(
        df, n_folds=3, embargo_minutes=DEFAULT_HWC_PURGE_MINUTES)

    first = runner.run_phase1_hwc(df, folds=folds, force=True)
    archive_path = Path(runner._output_dir) / \
        "rule_archives" / "hwc" / "hwc_rules.json"
    oof_path = archive_path.with_name("hwc_oof_scores.json")
    first_payload = json.loads(archive_path.read_text(encoding="utf-8"))
    first_oof = oof_path.read_text(encoding="utf-8")
    first_hash = first_payload["archive_hash"]
    metadata = dict(first_payload.get("metadata", {}))

    reused = runner.run_phase1_hwc(df, folds=folds, force=False)
    assert [rule.get("conditions") for rule in reused] == [
        rule.get("conditions") for rule in first]
    assert json.loads(archive_path.read_text(
        encoding="utf-8"))["archive_hash"] == first_hash

    planted = [{
        "timeframe": "hwc",
        "direction": "long",
        "conditions": ["[PLANTED] >= 1"],
        "coverage": 0.5,
        "directional_edge": 0.2,
        "mcc": 0.2,
        "stability": 1.0,
        "skill": 0.2,
        "data_hash": metadata.get("dataset_hash", "planted"),
        "feature_schema_hash": metadata.get("feature_schema_hash", "planted"),
        "oof_metrics": {
            "directional_edge": 0.2,
            "mcc": 0.2,
            "coverage": 0.5,
            "stability": 1.0,
        },
    }]
    save_mtf_rule_archive("hwc", planted, archive_path, metadata=metadata)
    oof_path.write_text(first_oof, encoding="utf-8")
    planted_reused = runner.run_phase1_hwc(df, folds=folds, force=False)
    assert planted_reused[0]["conditions"] == ["[PLANTED] >= 1"]

    metadata_bad = dict(metadata, dataset_hash="tampered_hash")
    save_mtf_rule_archive("hwc", planted, archive_path, metadata=metadata_bad)
    oof_path.write_text(first_oof, encoding="utf-8")
    after_tamper = runner.run_phase1_hwc(df, folds=folds, force=False)
    assert all("[PLANTED]" not in str(rule.get("conditions"))
               for rule in after_tamper)

    save_mtf_rule_archive("hwc", planted, archive_path, metadata=metadata)
    tampered_oof = json.loads(first_oof)
    if tampered_oof:
        tampered_oof[0]["direction_score"] = 0.999
    else:
        tampered_oof = [{
            "datetime": "2024-01-01 00:00:00",
            "symbol": "BTCUSDT",
            "direction_score": 0.999,
        }]
    oof_path.write_text(json.dumps(tampered_oof), encoding="utf-8")
    after_sidecar = runner.run_phase1_hwc(df, folds=folds, force=False)
    assert all("[PLANTED]" not in str(rule.get("conditions"))
               for rule in after_sidecar)


def test_phase5_strict_binding_to_archive_payloads(tmp_path):
    """Verify Phase 5 fails closed if serialized candidate rules diverge from archive payloads."""
    output_dir = tmp_path / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    hwc_dir = output_dir / "rule_archives" / "hwc"
    mwc_dir = output_dir / "rule_archives" / "mwc"
    lwc_dir = output_dir / "rule_archives" / "lwc"
    hwc_dir.mkdir(parents=True, exist_ok=True)
    mwc_dir.mkdir(parents=True, exist_ok=True)
    lwc_dir.mkdir(parents=True, exist_ok=True)

    hwc_rule = {"timeframe": "hwc", "direction": "long", "conditions": [
        "[open] >= 100"], "coverage": 0.5, "directional_edge": 0.2, "mcc": 0.2, "stability": 1.0, "skill": 0.2}
    mwc_rule = {"timeframe": "mwc", "direction": "long", "conditions": [
        "[rsi_14] >= 50"], "coverage": 0.4, "directional_edge": 0.15, "mcc": 0.15, "stability": 1.0, "skill": 0.15}
    lwc_rule = {
        "timeframe": "lwc",
        "direction": "long",
        "conditions": ["[open] >= 0"],
        "coverage": 0.5,
        "directional_edge": 0.1,
        "mcc": 0.1,
        "stability": 1.0,
        "skill": 0.1,
        "tp": 2.0,
        "sl": 1.2,
        "capital_pct": 18.0,
    }

    hwc_hash = save_mtf_rule_archive(
        "hwc", [hwc_rule], hwc_dir / "hwc_rules.json", metadata={"role": "hwc"})
    mwc_hash = save_mtf_rule_archive(
        "mwc", [mwc_rule], mwc_dir / "mwc_rules.json", metadata={"role": "mwc"})
    lwc_hash = save_mtf_rule_archive(
        "lwc", [lwc_rule], lwc_dir / "lwc_rules.json", metadata={"role": "lwc"})

    manifest = {
        "frozen_runtime": True,
        "archives": {
            "hwc_archive_hash": hwc_hash,
            "mwc_archive_hash": mwc_hash,
            "lwc_archive_hash": lwc_hash,
        },
    }
    import hashlib
    manifest_sha = hashlib.sha256(json.dumps(
        manifest, sort_keys=True).encode("utf-8")).hexdigest()

    # Corrupted candidate with modified HWC rule
    corrupted_hwc_rule = dict(hwc_rule, conditions=["[open] >= 999"])
    strategy_payload = {
        "direction": "long",
        "rules_set": [lwc_rule],
        "mtf_manifest": manifest,
        "provenance": {"mtf_manifest_hash": manifest_sha},
        "mtf_candidate": {
            "direction": "long",
            "lwc_rules": [lwc_rule],
            "hwc_rules": [corrupted_hwc_rule],
            "mwc_rules": [mwc_rule],
            "mtf_manifest": manifest,
        },
    }
    strategy_path = output_dir / "long.json"
    strategy_path.write_text(json.dumps(strategy_payload), encoding="utf-8")

    from gpu_fuzzy_trader import config as _cfg
    from gpu_fuzzy_trader.phases import phase5_oos as p5_mod
    evaluator = OOS_Evaluator()
    evaluator.output_dir = str(output_dir)

    orig_paths = _cfg.MTF_ARCHIVE_PATHS
    orig_strategy_paths = p5_mod._STRATEGY_PATHS
    try:
        _cfg.MTF_ARCHIVE_PATHS = {
            "hwc": str(hwc_dir / "hwc_rules.json"),
            "mwc": str(mwc_dir / "mwc_rules.json"),
            "lwc": str(lwc_dir / "lwc_rules.json"),
        }
        p5_mod._STRATEGY_PATHS = {
            "long": str(strategy_path),
            "short": str(output_dir / "short.json"),
        }
        loaded = evaluator.load_strategies()
        # Corrupted HWC rule strategy must fail validation and not be loaded
        assert "long" not in loaded

        # Test LWC multiset count divergence (candidate has duplicate rule not in archive multiset)
        strategy_payload_lwc_multiset = {
            "direction": "long",
            "rules_set": [lwc_rule, lwc_rule],
            "mtf_manifest": manifest,
            "provenance": {"mtf_manifest_hash": manifest_sha},
            "mtf_candidate": {
                "direction": "long",
                "lwc_rules": [lwc_rule, lwc_rule],
                "hwc_rules": [hwc_rule],
                "mwc_rules": [mwc_rule],
                "mtf_manifest": manifest,
            },
        }
        strategy_path.write_text(json.dumps(
            strategy_payload_lwc_multiset), encoding="utf-8")
        loaded_multiset = evaluator.load_strategies()
        assert "long" not in loaded_multiset

        # Test valid exact matching candidate
        strategy_payload_valid = {
            "direction": "long",
            "rules_set": [lwc_rule],
            "mtf_manifest": manifest,
            "provenance": {"mtf_manifest_hash": manifest_sha},
            "mtf_candidate": {
                "direction": "long",
                "lwc_rules": [lwc_rule],
                "hwc_rules": [hwc_rule],
                "mwc_rules": [mwc_rule],
                "mtf_manifest": manifest,
            },
        }
        strategy_path.write_text(json.dumps(
            strategy_payload_valid), encoding="utf-8")
        loaded_valid = evaluator.load_strategies()
        assert "long" in loaded_valid
    finally:
        _cfg.MTF_ARCHIVE_PATHS = orig_paths
        p5_mod._STRATEGY_PATHS = orig_strategy_paths

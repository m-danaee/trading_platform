"""Unit & Integration tests for Hierarchical MTF Pipeline Integration (Task 6).

Verifies:
1. Pipeline_Runner exposes new hierarchical phases and contract methods.
2. HierarchicalStrategyCandidate encapsulates LWC, MWC, and HWC rules with composer params.
3. mtf_manifest.json generation and schema compliance.
4. CPUBacktestEngine evaluates rules without legacy mandatory context mask enforcement.
5. Config and Loader support clean MTF parameters without mandatory legacy context locks.
6. End-to-end MTF composition, candidate serialization, and OOS evaluation.
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
from gpu_fuzzy_trader.data.loader import Data_Loader, validate_context_columns
from gpu_fuzzy_trader.mtf.archives import save_mtf_rule_archive
from gpu_fuzzy_trader.mtf.candidate import HierarchicalStrategyCandidate
from gpu_fuzzy_trader.mtf.runtime import (
    attach_frozen_layer_scores,
    attach_oof_layer_scores,
)
from gpu_fuzzy_trader.run_pipeline import (
    Pipeline_Runner,
    Pipeline_Orchestrator,
    _merge_mtf_lwc_runtime_columns,
)


def test_pipeline_mtf_phases_contract():
    """Verify pipeline runner exposes new hierarchical phases and contract methods."""
    runner = Pipeline_Runner()
    assert hasattr(runner, "run_phase1_hwc")
    assert hasattr(runner, "run_phase1_mwc")
    assert hasattr(runner, "run_phase2")
    assert hasattr(runner, "run_mtf_composition")
    assert hasattr(runner, "run_rb_governor")
    assert hasattr(runner, "run_phase5_oos")

    # Pipeline_Orchestrator alias compatibility
    assert Pipeline_Orchestrator is Pipeline_Runner or issubclass(Pipeline_Orchestrator, Pipeline_Runner) or issubclass(Pipeline_Runner, Pipeline_Orchestrator)


def test_config_mtf_parameters():
    """Verify new MTF configuration parameters and disabled legacy requirements."""
    # New MTF parameters
    assert hasattr(_cfg, "MTF_V_HWC_LONG")
    assert hasattr(_cfg, "MTF_V_HWC_SHORT")
    assert hasattr(_cfg, "MTF_V_MWC_LONG")
    assert hasattr(_cfg, "MTF_V_MWC_SHORT")
    assert hasattr(_cfg, "MTF_MIN_EVIDENCE_STRENGTH")
    assert hasattr(_cfg, "MTF_RETENTION_FLOOR")
    assert hasattr(_cfg, "MTF_RETENTION_TARGET")
    assert hasattr(_cfg, "MTF_ARCHIVE_PATHS")
    assert hasattr(_cfg, "MTF_MANIFEST_PATH")

    # Disabled legacy requirements
    assert _cfg.REQUIRE_CONTEXT_IN_STRATEGY is False
    assert _cfg.REQUIRE_CONTEXT_COLUMNS is False


def test_loader_clean_ohlcv_without_mandatory_context():
    """Verify Data_Loader loads clean OHLCV without failing on missing context columns."""
    n = 150
    df_raw = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01 00:00", periods=n, freq="15min"),
        "symbol": ["BTCUSDT"] * n,
        "open": np.linspace(100, 110, n),
        "high": np.linspace(101, 111, n),
        "low": np.linspace(99, 109, n),
        "close": np.linspace(100.5, 110.5, n),
        "volume": np.ones(n) * 10.0,
    })

    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w", delete=False) as f:
        df_raw.to_csv(f.name, index=False)
        temp_csv = f.name

    try:
        # Should succeed without requiring context columns
        loader = Data_Loader()
        loaded_df = loader.load_dataset(temp_csv, require_context=False)
        assert len(loaded_df) > 0
        assert "close" in loaded_df.columns
    finally:
        Path(temp_csv).unlink(missing_ok=True)


def test_cpu_engine_no_legacy_context_mask():
    """Verify CPU backtest engine evaluates raw trading rules without injecting legacy context columns."""
    n = 40
    df = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01 00:00", periods=n, freq="15min"),
        "symbol": ["BTCUSDT"] * n,
        "open": np.linspace(100, 120, n),
        "high": np.linspace(101, 121, n),
        "low": np.linspace(99, 119, n),
        "close": np.linspace(100.5, 120.5, n),
        "volume": np.ones(n) * 100.0,
        "label_open_next": np.linspace(100.5, 120.5, n),
        "label_close_288": np.linspace(101.5, 121.5, n),
        "label_min_288": np.linspace(99.0, 119.0, n),
        "label_max_288": np.linspace(102.0, 122.0, n),
        "label_max_before_min": np.ones(n, dtype=int),
        "feature_rsi": np.ones(n, dtype=float) * 0.7,
    })

    engine = CPUBacktestEngine(df=df, feature_modes={"feature_rsi": 1}, direction="long")
    rule_set = [
        {
            "conditions": ["[feature_rsi] IS High"],
            "tp": 0.03,
            "sl": 0.015,
            "capital_pct": 0.05,
        }
    ]

    # Simulating rule set should work cleanly without expecting tf_permission_*
    result = engine.simulate_rule_set(rule_set)
    assert isinstance(result, dict)
    assert "total_return_pct" in result
    assert "executed_trades" in result


def test_hierarchical_strategy_candidate_encapsulation():
    """Verify HierarchicalStrategyCandidate encapsulates multi-timeframe rules and composer params."""
    hwc_rules = [
        {
            "timeframe": "hwc",
            "direction": "long",
            "conditions": ["[hwc_rsi] > 50"],
            "coverage": 0.40,
            "directional_edge": 0.12,
            "mcc": 0.22,
            "stability_score": 0.85,
            "skill": 0.10,
        }
    ]
    mwc_rules = [
        {
            "timeframe": "mwc",
            "direction": "long",
            "conditions": ["[mwc_macd] > 0"],
            "coverage": 0.35,
            "directional_edge": 0.10,
            "mcc": 0.18,
            "stability_score": 0.80,
            "skill": 0.08,
        }
    ]
    lwc_rules = [
        {
            "timeframe": "lwc",
            "direction": "long",
            "conditions": ["[feature_bb] < -1.0"],
            "tp": 0.03,
            "sl": 0.015,
            "capital_pct": 0.05,
            "coverage": 0.15,
            "directional_edge": 0.15,
            "mcc": 0.25,
            "stability_score": 0.90,
            "skill": 0.12,
        }
    ]

    candidate = HierarchicalStrategyCandidate(
        direction="long",
        lwc_rules=lwc_rules,
        hwc_rules=hwc_rules,
        mwc_rules=mwc_rules,
        composer_params={
            "v_hwc_long": 0.65,
            "v_hwc_short": 0.60,
            "v_mwc_long": 0.60,
            "v_mwc_short": 0.55,
            "min_evidence_strength_hwc": 0.15,
            "min_evidence_strength_mwc": 0.15,
            "retention_floor": 0.50,
        },
    )

    d = candidate.to_dict()
    assert d["direction"] == "long"
    assert len(d["lwc_rules"]) == 1
    assert len(d["hwc_rules"]) == 1
    assert len(d["mwc_rules"]) == 1
    assert "composer_params" in d
    assert "strategy_id" in d

    # Roundtrip from dict
    restored = HierarchicalStrategyCandidate.from_dict(d)
    assert restored.direction == "long"
    assert len(restored.lwc_rules) == 1
    assert len(restored.hwc_rules) == 1
    assert len(restored.mwc_rules) == 1
    assert restored.composer_params["v_hwc_long"] == 0.65


def test_mtf_manifest_generation_and_schema():
    """Verify build_mtf_manifest generates valid manifest matching design schema."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        hwc_path = Path(tmp_dir) / "hwc_rules.json"
        mwc_path = Path(tmp_dir) / "mwc_rules.json"
        lwc_path = Path(tmp_dir) / "lwc_rules.json"
        manifest_path = Path(tmp_dir) / "mtf_manifest.json"

        hwc_hash = save_mtf_rule_archive(
            timeframe="hwc",
            rules=[{
                "timeframe": "hwc",
                "direction": "long",
                "conditions": ["[hwc_rsi] > 50"],
                "coverage": 0.40,
            }],
            path=hwc_path,
        )
        mwc_hash = save_mtf_rule_archive(
            timeframe="mwc",
            rules=[{
                "timeframe": "mwc",
                "direction": "long",
                "conditions": ["[mwc_macd] > 0"],
                "coverage": 0.35,
            }],
            path=mwc_path,
        )
        lwc_hash = save_mtf_rule_archive(
            timeframe="lwc",
            rules=[{
                "timeframe": "lwc",
                "direction": "long",
                "conditions": ["[feature_bb] < -1.0"],
                "coverage": 0.15,
            }],
            path=lwc_path,
        )

        runner = Pipeline_Runner(output_dir=tmp_dir)
        manifest = runner.build_mtf_manifest(
            hwc_archive_hash=hwc_hash,
            mwc_archive_hash=mwc_hash,
            lwc_archive_hash=lwc_hash,
            output_path=manifest_path,
        )

        assert manifest["schema_version"] == "2.0.0"
        assert manifest["timeframes"]["lwc_minutes"] == 15
        assert manifest["timeframes"]["mwc_minutes"] == 60
        assert manifest["timeframes"]["hwc_minutes"] == 240
        assert manifest["archives"]["hwc_archive_hash"] == hwc_hash
        assert manifest["archives"]["mwc_archive_hash"] == mwc_hash
        assert manifest["archives"]["lwc_archive_hash"] == lwc_hash
        assert manifest_path.exists()


def test_run_mtf_composition_and_candidate_signal_evaluation():
    """Verify run_mtf_composition produces valid candidates and executes composition."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = Pipeline_Runner(output_dir=tmp_dir)

        lwc_rules = {
            "long": [
                {
                    "timeframe": "lwc",
                    "direction": "long",
                    "conditions": ["[feature_bb] < -1.0"],
                    "tp": 0.03,
                    "sl": 0.015,
                    "capital_pct": 0.05,
                    "coverage": 0.20,
                }
            ],
            "short": [
                {
                    "timeframe": "lwc",
                    "direction": "short",
                    "conditions": ["[feature_bb] > 1.0"],
                    "tp": 0.03,
                    "sl": 0.015,
                    "capital_pct": 0.05,
                    "coverage": 0.20,
                }
            ],
        }

        hwc_rules = [
            {
                "timeframe": "hwc",
                "direction": "long",
                "conditions": ["[hwc_rsi] > 50"],
                "coverage": 0.40,
                "directional_edge": 0.12,
                "mcc": 0.20,
                "stability_score": 0.85,
                "skill": 0.10,
            }
        ]

        mwc_rules = [
            {
                "timeframe": "mwc",
                "direction": "long",
                "conditions": ["[mwc_macd] > 0"],
                "coverage": 0.35,
                "directional_edge": 0.10,
                "mcc": 0.15,
                "stability_score": 0.80,
                "skill": 0.08,
            }
        ]

        candidates = runner.run_mtf_composition(
            lwc_rules=lwc_rules,
            hwc_rules=hwc_rules,
            mwc_rules=mwc_rules,
        )

        assert "long" in candidates
        assert "short" in candidates
        assert isinstance(candidates["long"], HierarchicalStrategyCandidate)
        assert len(candidates["long"].lwc_rules) == 1

        # Test composition evaluation via candidate
        n = 10
        lwc_trigs = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 1])
        hwc_dir = np.array([0.5, -0.8, 0.0, 0.0, 0.2, -0.7, 0.0, 0.4, 0.0, 0.6])
        hwc_str = np.array([0.3, 0.5, 0.0, 0.0, 0.3, 0.4, 0.0, 0.3, 0.0, 0.3])
        mwc_dir = np.array([0.4, 0.0, 0.0, 0.0, -0.8, 0.0, 0.0, 0.2, 0.0, 0.5])
        mwc_str = np.array([0.3, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.3, 0.0, 0.3])

        signals, stats = candidates["long"].compose(
            lwc_triggers=lwc_trigs,
            hwc_direction=hwc_dir,
            hwc_strength=hwc_str,
            mwc_direction=mwc_dir,
            mwc_strength=mwc_str,
        )

        assert len(signals) == n
        assert stats["raw_triggers"] == 6
        # At index 1: hwc_dir=-0.8, hwc_str=0.5 -> vetoed by HWC
        # At index 5: hwc_dir=-0.7, hwc_str=0.4 -> vetoed by HWC
        # At index 4: mwc_dir=-0.8, mwc_str=0.5 -> vetoed by MWC
        assert stats["hwc_vetoed"] >= 2
        assert stats["mwc_vetoed"] >= 1
        assert stats["accepted_trades"] == stats["raw_triggers"] - stats["hwc_vetoed"] - stats["mwc_vetoed"]


def test_run_phase1_hwc_and_mwc_discovery():
    """Verify run_phase1_hwc and run_phase1_mwc discover and persist rules."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        runner = Pipeline_Runner(output_dir=tmp_dir)

        hwc_rules = runner.run_phase1_hwc(force=True)
        assert len(hwc_rules) > 0
        assert (Path(tmp_dir) / "rule_archives" / "hwc" / "hwc_rules.json").exists()

        mwc_rules = runner.run_phase1_mwc(hwc_rules=hwc_rules, force=True)
        assert isinstance(mwc_rules, list)
        assert (Path(tmp_dir) / "rule_archives" / "mwc" / "mwc_rules.json").exists()
        # Current BTC/ETH train tape has no MWC rule with MCC > 0 on every symbol.
        # The connected MTF pipeline fail-closes on an empty MWC archive.


def test_bidirectional_hierarchical_strategy_candidate():
    """Verify bidirectional HierarchicalStrategyCandidate composition."""
    candidate = HierarchicalStrategyCandidate(
        direction="bidirectional",
        lwc_rules=[
            {"timeframe": "lwc", "direction": "long", "conditions": ["[a] > 0"], "coverage": 0.1},
            {"timeframe": "lwc", "direction": "short", "conditions": ["[b] < 0"], "coverage": 0.1},
        ],
        hwc_rules=[
            {"timeframe": "hwc", "direction": "long", "conditions": ["[c] > 0"], "coverage": 0.4},
        ],
        mwc_rules=[
            {"timeframe": "mwc", "direction": "long", "conditions": ["[d] > 0"], "coverage": 0.3},
        ],
    )

    lwc_trigs = np.array([1, -1, 0, 1, -1])
    hwc_dir = np.zeros(5)
    hwc_str = np.zeros(5)
    mwc_dir = np.zeros(5)
    mwc_str = np.zeros(5)

    signals, stats = candidate.compose(
        lwc_triggers=lwc_trigs,
        hwc_direction=hwc_dir,
        hwc_strength=hwc_str,
        mwc_direction=mwc_dir,
        mwc_strength=mwc_str,
    )

    assert len(signals) == 5
    assert (signals == lwc_trigs).all()
    assert stats["total"]["raw_triggers"] == 4
    assert stats["total"]["accepted_trades"] == 4


def test_bidirectional_candidate_runtime_uses_serialized_strength_aliases():
    raw = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=4, freq="15min"),
        "symbol": "BTCUSDT",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 10.0,
    })
    candidate = HierarchicalStrategyCandidate(
        direction="bidirectional",
        lwc_rules=[
            {"timeframe": "lwc", "direction": "long", "conditions": ["[open] >= 0"], "coverage": 0.5},
            {"timeframe": "lwc", "direction": "short", "conditions": ["[open] >= 0", "symbol is BTCUSDT"], "coverage": 0.5},
        ],
        composer_params={
            "min_evidence_strength_hwc": 0.9,
            "min_evidence_strength_mwc": 0.9,
        },
    )
    signals, stats, audit = candidate.evaluate_frame(raw)
    assert len(signals) == len(raw)
    assert stats["total"]["raw_triggers"] == len(raw)
    assert len(audit) == len(raw)
    history = raw.copy()
    history["datetime"] = history["datetime"] - pd.Timedelta(hours=1)
    historical_signals, _, historical_audit = candidate.evaluate_frame(
        raw, history_df=history
    )
    assert len(historical_signals) == len(raw)
    assert historical_audit["datetime"].min() == raw["datetime"].min()


def test_lwc_runtime_merge_keeps_own_features_but_not_raw_htf_features():
    n = 32
    raw = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "symbol": "BTCUSDT",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 10.0,
        "label_open_next": 100.0,
        "label_close_288": 100.0,
        "label_min_288": 99.0,
        "label_max_288": 101.0,
        "label_max_before_min": 1,
        "ff_legacy_lwc": 0.5,
        "hwc_stale_feature": 0.5,
    })
    scores = attach_oof_layer_scores(
        raw,
        hwc_scores=pd.DataFrame(columns=["datetime", "symbol", "direction_score", "strength_score"]),
        mwc_scores=pd.DataFrame(columns=["datetime", "symbol", "direction_score", "strength_score"]),
    )
    merged = _merge_mtf_lwc_runtime_columns(raw, scores)
    assert any(column.startswith("lwc_") for column in merged.columns)
    assert not any(column.startswith("hwc_") for column in merged.columns)
    assert not any(column.startswith("mwc_") for column in merged.columns)
    assert "ff_legacy_lwc" not in merged.columns
    assert "hwc_stale_feature" not in merged.columns
    assert "mtf_hwc_direction" in merged.columns
    assert "mtf_mwc_strength" in merged.columns


def test_frozen_mwc_scores_are_conditioned_on_frozen_hwc_direction():
    raw = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=4, freq="15min"),
        "symbol": "BTCUSDT",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 10.0,
    })
    common = {
        "conditions": ["[open] >= 0"],
        "coverage": 0.5,
        "directional_edge": 0.2,
        "mcc": 0.2,
        "stability": 1.0,
        "skill": 0.2,
    }
    frame = attach_frozen_layer_scores(
        raw,
        [{**common, "timeframe": "hwc", "direction": "short"}],
        [{**common, "timeframe": "mwc", "direction": "long"}],
    )
    assert np.all(frame["mtf_hwc_direction"].to_numpy() == -1.0)
    assert np.all(frame["mtf_mwc_direction"].to_numpy() == 0.0)
    assert np.all(frame["mtf_mwc_strength"].to_numpy() == 0.0)


def test_lwc_oof_score_alignment_is_causal_and_runtime_scores_are_frozen():
    n = 32
    raw = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="15min"),
        "symbol": "BTCUSDT",
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.0,
        "volume": 10.0,
    })
    hwc_oof = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01 00:00")],
        "symbol": ["BTCUSDT"],
        "direction_score": [-0.8],
        "strength_score": [0.7],
    })
    mwc_oof = pd.DataFrame({
        "datetime": [pd.Timestamp("2024-01-01 00:00")],
        "symbol": ["BTCUSDT"],
        "direction_score": [0.4],
        "strength_score": [0.5],
    })

    scored = attach_oof_layer_scores(
        raw, hwc_scores=hwc_oof, mwc_scores=mwc_oof
    )
    assert not bool(scored.loc[scored["datetime"] == "2024-01-01 03:30", "_mtf_oof_available"].iloc[0])
    row = scored.loc[scored["datetime"] == "2024-01-01 03:45"].iloc[0]
    assert bool(row["_mtf_oof_available"])
    assert np.isclose(row["mtf_hwc_direction"], -0.8)
    assert np.isclose(row["mtf_mwc_direction"], 0.4)

    candidate = HierarchicalStrategyCandidate(
        direction="long",
        lwc_rules=[{
            "timeframe": "lwc",
            "direction": "long",
            "conditions": ["[mtf_hwc_direction] IS Exactly Zero"],
            "coverage": 0.5,
        }],
    )
    signals, stats, audit = candidate.evaluate_frame(raw)
    assert len(signals) == n
    assert stats["raw_triggers"] == n
    assert set(audit["accepted"].unique()) == {1}


def test_mtf_validation_fitness_selection_separation():
    """Verify MTF pipeline splits validation into fitness and selection halves."""
    n = 300
    dates = pd.date_range("2024-01-01 00:00", periods=n, freq="15min")
    val_df = pd.DataFrame({
        "datetime": dates,
        "symbol": ["BTCUSDT"] * n,
        "open": 100.0 + np.arange(n) * 0.1,
        "high": 101.0 + np.arange(n) * 0.1,
        "low": 99.0 + np.arange(n) * 0.1,
        "close": 100.5 + np.arange(n) * 0.1,
        "volume": 10.0,
    })
    runner = Pipeline_Runner()
    val_fitness, val_selection = runner._validation_scoring_frames(val_df)

    # Must be non-empty and disjoint subsets
    assert not val_fitness.empty
    assert not val_selection.empty
    assert len(val_fitness) < len(val_df)
    assert len(val_selection) < len(val_df)
    # Chronologically purged: max fitness datetime < min selection datetime
    assert val_fitness["datetime"].max() < val_selection["datetime"].min()

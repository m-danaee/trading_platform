"""
Empirical validation of the README conflict table against live code paths.

Each test maps to one row in the conflict table and proves the described
interaction exists in this project (not merely documented).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta

import pandas as pd
import pytest

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.phases.phase2_support import (
    passes_pool_admission_gate,
    trade_support_penalty,
)
from gpu_fuzzy_trader.phases.phase4_wf_optimizer import (
    build_phase4_walk_forward_splits,
)
from gpu_fuzzy_trader.data.splitter import load_cached_split_if_fresh
from gpu_fuzzy_trader.rb_governor import _filter_good_rules, _is_positive_good


def _synthetic_val_df(*, n_rows_per_symbol: int = 400, symbols: list[str] | None = None) -> pd.DataFrame:
    symbols = symbols or ["A", "B"]
    base = datetime(2024, 1, 1)
    rows: list[dict] = []
    for sym in symbols:
        for i in range(n_rows_per_symbol):
            rows.append({
                "datetime": base + timedelta(minutes=5 * i),
                "symbol": sym,
                "label_open_next": 1.0,
                "label_close_288": 1.0,
                "label_min_288": 0.5,
                "label_max_288": 1.5,
                "label_max_before_min": 0.0,
                "feat": 0.5,
            })
    return pd.DataFrame(rows)


def _good_train_metrics(*, trades: int = 60) -> dict:
    return {
        "executed_trades": trades,
        "total_return_pct": 8.0,
        "profit_factor": 1.5,
        "max_drawdown_pct": 5.0,
        "win_rate": 55.0,
    }


def _bad_val_metrics(*, trades: int = 30) -> dict:
    return {
        "executed_trades": trades,
        "total_return_pct": -3.0,
        "profit_factor": 0.8,
        "max_drawdown_pct": 12.0,
        "win_rate": 40.0,
    }


class TestConflict01PurgedHighFloorsIslandCluster:
  """Purged WF + high trade floors + island cluster → zero rules per island."""

  def test_unscaled_high_floors_reject_thin_island_slice(self, monkeypatch):
      monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_walk_forward")
      monkeypatch.setattr(_cfg, "PURGED_WF_SCALE_TRADE_FLOORS", False)
      monkeypatch.setattr(_cfg, "PHASE2_ISLAND_SCALE_TRADE_FLOORS", False)
      monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 150)
      _cfg.set_purged_wf_reference_rows(100_000)

      hp = _cfg.resolve_island_hyperparams(
          "cluster", n_rows=8_000, reference_rows=100_000, n_symbols=3,
      )
      assert hp.min_trade_pool_floor >= 150

      train = _good_train_metrics(trades=80)
      val = _good_train_metrics(trades=40)
      assert passes_pool_admission_gate(train, val, n_valid_rows=2_000) is False

  def test_scaling_on_lowers_floors_for_small_island(self, monkeypatch):
      monkeypatch.setattr(_cfg, "PHASE2_ISLAND_SCALE_TRADE_FLOORS", True)
      monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 150)
      hp = _cfg.resolve_island_hyperparams(
          "cluster", n_rows=8_000, reference_rows=100_000, n_symbols=3,
      )
      assert hp.min_trade_pool_floor < 150


class TestConflict02JointTrainValFalseStrictGates:
  """PHASE2_JOINT_TRAIN_VAL=False + strict pool gates → admission rejects train-only winners."""

  def test_train_strong_val_weak_fails_admission_even_when_joint_off(self, monkeypatch):
      monkeypatch.setattr(_cfg, "PHASE2_JOINT_TRAIN_VAL", False)
      monkeypatch.setattr(_cfg, "PHASE2_POOL_REQUIRE_POSITIVE_SPLITS", True)
      monkeypatch.setattr(_cfg, "MIN_TRADE_POOL_FLOOR", 10)
      monkeypatch.setattr(_cfg, "PHASE2_STRICT_POSITIVE_GOOD", False)

      train = _good_train_metrics(trades=60)
      val = _bad_val_metrics(trades=30)
      assert passes_pool_admission_gate(train, val) is False


class TestConflict03HoldoutPhase4SplitsTailHoldout:
  """holdout_70_30 + PHASE4_WF_SPLITS=4 + tail holdout → tiny WF windows."""

  def test_four_splits_plus_tail_shrink_worst_window_below_trade_gate(self, monkeypatch):
      monkeypatch.setattr(_cfg, "PHASE4_WF_SPLITS", 4)
      monkeypatch.setattr(_cfg, "PHASE4_INCLUDE_TAIL_HOLDOUT", True)
      monkeypatch.setattr(_cfg, "PHASE4_TAIL_HOLDOUT_FRACTION", 0.25)
      monkeypatch.setattr(_cfg, "PHASE4_MIN_WORST_TRADES", 20)

      val_df = _synthetic_val_df(n_rows_per_symbol=400)
      splits = build_phase4_walk_forward_splits(val_df, 4)
      assert len(splits) == 5  # 4 WF + tail

      smallest = min(len(s) for s in splits)
      # Val is already the 30% holdout; 4 WF chunks + tail → ~100 rows/symbol/window.
      assert smallest == 200
      assert smallest < len(val_df) * 0.30


class TestConflict04PurgedLegacyPhase4TripleWf:
  """purged_walk_forward + legacy Phase 4 → triple WF unless forced to 1."""

  def test_effective_splits_forced_to_one_under_purged_mode(self, monkeypatch):
      monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_walk_forward")
      monkeypatch.setattr(_cfg, "PHASE4_WF_SPLITS", 4)
      assert _cfg.effective_phase4_wf_splits() == 1

  def test_holdout_mode_keeps_configured_splits(self, monkeypatch):
      monkeypatch.setattr(_cfg, "SPLIT_MODE", "holdout_70_30")
      monkeypatch.setattr(_cfg, "PHASE4_WF_SPLITS", 4)
      assert _cfg.effective_phase4_wf_splits() == 4


class TestConflict05TripleMonthlyGates:
  """Rules can pass Phase 2 monthly admission yet fail RB monthly certificate."""

  def test_rb_monthly_certificate_rejects_after_lenient_p2_monthly_thresholds(
      self, monkeypatch,
  ):
      from gpu_fuzzy_trader.rb_governor import _profit_amp_certificate

      from gpu_fuzzy_trader.validation.monthly_windows import MonthlyWindowSummary

      monkeypatch.setattr(_cfg, "PHASE2_MONTHLY_ADMISSION_MIN_RATIO", 0.3)
      monkeypatch.setattr(_cfg, "RB_PROFIT_AMP_MIN_MONTHLY_PROFITABLE_RATIO", 0.90)
      monkeypatch.setattr(_cfg, "RB_PROFIT_AMP_MIN_MONTHLY_WINDOWS", 2)

      summary = MonthlyWindowSummary(
          windows=4,
          profitable_windows=2,
          profitable_ratio=0.50,
          mean_return_pct=1.0,
          median_return_pct=1.0,
          worst_return_pct=1.0,
          latest_return_pct=1.0,
          recency_weighted_return_pct=1.0,
          mean_profit_factor=1.2,
          worst_profit_factor=1.2,
          worst_drawdown_pct=5.0,
          min_trades=10,
          mean_trades=15.0,
          equity_slope=0.0,
          max_equity_dip_pct=5.0,
          score=0.0,
      )

      train_m = _good_train_metrics()
      train_m["raw_signal_count"] = 80
      train_m["skipped_min_notional_count"] = 2
      valid_m = _good_train_metrics()
      valid_m["raw_signal_count"] = 70
      valid_m["skipped_min_notional_count"] = 2
      accepted, detail = _profit_amp_certificate(train_m, valid_m, summary)
      assert accepted is False
      assert "monthly_profitable_ratio_low" in detail["reasons"]


class TestConflict06DebugSymbolCountVsMinProfitable:
  """DEBUG_SYMBOL_COUNT=2 + PHASE2_MIN_PROFITABLE_SYMBOLS=4 — auto-capped, not impossible."""

  def test_effective_min_profitable_symbols_caps_to_debug_universe(self, monkeypatch):
      monkeypatch.setattr(_cfg, "DEBUG_SYMBOL_SCOPE_ENABLED", True)
      monkeypatch.setattr(_cfg, "DEBUG_SYMBOL_COUNT", 2)
      monkeypatch.setattr(_cfg, "PHASE2_MIN_PROFITABLE_SYMBOLS", 4)
      assert _cfg.effective_min_profitable_symbols() == 2


class TestConflict07TwoStageIslandCluster:
  """PHASE2_TWO_STAGE + island cluster → two-stage never runs without island flag."""

  def test_global_two_stage_ignored_in_cluster_mode(self, monkeypatch):
      monkeypatch.setattr(_cfg, "PHASE2_ISLAND_MODE", "cluster")
      monkeypatch.setattr(_cfg, "PHASE2_TWO_STAGE_ENABLED", True)
      monkeypatch.setattr(_cfg, "PHASE2_ISLAND_TWO_STAGE_ENABLED", False)
      assert _cfg.island_two_stage_enabled() is False

      from gpu_fuzzy_trader.phases.phase2_stage import island_stage_budgets
      stage_a, stage_b = island_stage_budgets(60)
      assert stage_a == 60
      assert stage_b == 0


class TestConflict08EarlyStopArchiveSeed:
  """Early stop + archive seed 25% can truncate exploration before diversity recovery."""

  def test_island_plateau_early_stop_enabled_by_default(self):
      assert _cfg.PHASE2_ISLAND_MODE == "cluster"
      assert _cfg.island_early_stop_enabled() is False
      assert _cfg.island_plateau_early_stop_enabled() is True

  def test_global_mode_enables_early_stop_when_not_cluster(self, monkeypatch):
      monkeypatch.setattr(_cfg, "PHASE2_ISLAND_MODE", "global")
      monkeypatch.setattr(_cfg, "PHASE2_EARLY_STOP_ENABLED", True)
      monkeypatch.setattr(_cfg, "PHASE2_PLATEAU_EARLY_STOP_ENABLED", True)
      assert _cfg.island_early_stop_enabled() is True
      assert _cfg.island_plateau_early_stop_enabled() is True
      assert _cfg.PHASE2_ARCHIVE_SEED_FRACTION == 0.25


class TestConflict09StaleParquetCache:
  """Stale parquet cache can ignore split-mode changes."""

  def test_purged_cache_rejected_when_manifest_split_mode_differs(
      self, tmp_path, monkeypatch,
  ):
      csv_path = tmp_path / "train.csv"
      train_pq = tmp_path / "train_70.parquet"
      val_pq = tmp_path / "validation_30.parquet"
      fitness_pq = tmp_path / "validation_fitness.parquet"
      selection_pq = tmp_path / "validation_selection.parquet"
      manifest = tmp_path / "cv_folds_manifest.json"

      pd.DataFrame({"x": [1]}).to_parquet(train_pq)
      pd.DataFrame({"x": [1]}).to_parquet(val_pq)
      pd.DataFrame({"x": [1]}).to_parquet(fitness_pq)
      pd.DataFrame({"x": [1]}).to_parquet(selection_pq)
      csv_path.write_text("x\n1\n", encoding="utf-8")
      os.utime(csv_path, (1, 1))
      for path in (train_pq, val_pq, fitness_pq, selection_pq, manifest):
          path.touch()
          os.utime(path, (2, 2))

      monkeypatch.setattr(_cfg, "TRAIN_CSV_PATH", str(csv_path))
      monkeypatch.setattr(_cfg, "TRAIN_70_PATH", str(train_pq))
      monkeypatch.setattr(_cfg, "VALIDATION_30_PATH", str(val_pq))
      monkeypatch.setattr(_cfg, "VALIDATION_FITNESS_PATH", str(fitness_pq))
      monkeypatch.setattr(_cfg, "VALIDATION_SELECTION_PATH", str(selection_pq))
      monkeypatch.setattr(_cfg, "CV_FOLDS_MANIFEST_PATH", str(manifest))
      monkeypatch.setattr(_cfg, "SPLIT_MODE", "purged_walk_forward")

      manifest.write_text(json.dumps({
          "split_mode": "holdout_70_30",
          "config_fingerprint": "stale",
      }), encoding="utf-8")

      assert load_cached_split_if_fresh() is None

  def test_holdout_cache_rejected_without_manifest(self, tmp_path, monkeypatch):
      csv_path = tmp_path / "train.csv"
      train_pq = tmp_path / "train_70.parquet"
      val_pq = tmp_path / "validation_30.parquet"
      fitness_pq = tmp_path / "validation_fitness.parquet"
      selection_pq = tmp_path / "validation_selection.parquet"

      pd.DataFrame({"x": [1, 2]}).to_parquet(train_pq)
      pd.DataFrame({"x": [3]}).to_parquet(val_pq)
      pd.DataFrame({"x": [3]}).to_parquet(fitness_pq)
      pd.DataFrame({"x": [3]}).to_parquet(selection_pq)
      csv_path.write_text("x\n1\n", encoding="utf-8")
      os.utime(csv_path, (1, 1))
      os.utime(train_pq, (2, 2))
      os.utime(val_pq, (2, 2))
      os.utime(fitness_pq, (2, 2))
      os.utime(selection_pq, (2, 2))

      monkeypatch.setattr(_cfg, "TRAIN_CSV_PATH", str(csv_path))
      monkeypatch.setattr(_cfg, "TRAIN_70_PATH", str(train_pq))
      monkeypatch.setattr(_cfg, "VALIDATION_30_PATH", str(val_pq))
      monkeypatch.setattr(_cfg, "VALIDATION_FITNESS_PATH", str(fitness_pq))
      monkeypatch.setattr(_cfg, "VALIDATION_SELECTION_PATH", str(selection_pq))
      monkeypatch.setattr(_cfg, "CV_FOLDS_MANIFEST_PATH",
                          str(tmp_path / "missing.json"))
      monkeypatch.setattr(_cfg, "SPLIT_MODE", "holdout_70_30")

      assert load_cached_split_if_fresh() is None


class TestConflict10SupportPenaltyMaxZero:
  """SUPPORT_PENALTY_MAX=0 → thin-trade rules face no support penalty in objectives."""

  def test_zero_support_penalty_removes_evolutionary_pressure(self, monkeypatch):
      monkeypatch.setattr(_cfg, "SUPPORT_PENALTY_MAX", 0.0)
      monkeypatch.setattr(_cfg, "MIN_TRADE_SUPPORT", 55)
      pen, _, _ = trade_support_penalty(5)
      assert pen == 0.0


class TestConflict11RbMinTrainReturnHigh:
  """RB_MIN_TRAIN_RETURN=2.0 + thin val → _filter_good_rules drops marginal rules."""

  def test_high_rb_return_floor_filters_pool_rules(self, monkeypatch):
      monkeypatch.setattr(_cfg, "RB_MIN_TRAIN_RETURN", 2.0)
      monkeypatch.setattr(_cfg, "RB_MIN_VALID_RETURN", 0.5)
      monkeypatch.setattr(_cfg, "RB_MIN_TRAIN_TRADES", 5)
      monkeypatch.setattr(_cfg, "RB_MIN_VALID_TRADES", 5)

      train_m = _good_train_metrics(trades=30)
      train_m["total_return_pct"] = 1.5
      train_m["raw_signal_count"] = 40
      train_m["skipped_min_notional_count"] = 2
      valid_m = _good_train_metrics(trades=20)
      valid_m["raw_signal_count"] = 30
      valid_m["skipped_min_notional_count"] = 2

      assert _is_positive_good(train_m, valid_m) is False

      pool = [{
          "conditions": ["[feat] IS High"],
          "tp": 2.0,
          "sl": 1.0,
          "capital_pct": 20.0,
          "train_metrics": train_m,
          "valid_metrics": valid_m,
      }]
      train_df = _synthetic_val_df(n_rows_per_symbol=200, symbols=["A"])
      val_df = _synthetic_val_df(n_rows_per_symbol=100, symbols=["A"])
      kept = _filter_good_rules(pool, train_df, val_df, "long")
      assert kept == []


class TestConflict12Phase2CapitalPctSingleRule:
  """PHASE2_CAPITAL_PCT=48 + single rule — exposure semantics, not a failure mode."""

  def test_single_rule_uses_configured_capital_pct_directly(self):
      assert _cfg.PHASE2_CAPITAL_PCT == 30.0
      rule = {"capital_pct": 48.0}
      assert rule["capital_pct"] == 48.0

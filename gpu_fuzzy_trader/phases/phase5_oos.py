"""
phase5_oos.py — OOS_Evaluator (Phase 5)

Final out-of-sample evaluation on the held-out test_new.csv.

Workflow:
  1. Load outputs/long.json and outputs/short.json via Output_Writer.load_and_validate()
      (handles the case where only one strategy file exists)
  2. Prepare train, validation, and test data with the same pipeline as training:
         - Sort by (datetime, symbol)
         - Drop last 288 rows per symbol
         - Drop NaN label rows
         - Fill feature NaN with 0
         - Compute _symbol_bar_index
  3. Evaluate each available strategy on train / validation / test using
      CPUBacktestEngine.simulate_rule_set() with return_logs=True
  4. Compute per-symbol breakdowns from the test trade logs
  5. Handle zero-trade case: report 0% total return; do NOT report account ruin
      unless equity actually reached zero
  6. Save outputs in outputs/reports/ including the existing test JSON/CSV files
      plus the new cross-split reporting artifacts

Requirements: 11.1, 11.2, 11.3, 11.4, 11.5, 11.6
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.backtest.df_slim import downcast_numeric_df
from gpu_fuzzy_trader.data.loader import Data_Loader
from gpu_fuzzy_trader.data.splitter import Data_Splitter
from gpu_fuzzy_trader.features.selector import Feature_Selector
from gpu_fuzzy_trader.output.writer import (
    Output_Writer,
    ValidationError,
)
from gpu_fuzzy_trader.reporting.reporter import Reporter

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_STRATEGY_PATHS: dict[str, str] = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "long.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "short.json"),
}

_REPORT_PATHS: dict[str, str] = {
    "long": os.path.join(_cfg.REPORTS_DIR, "test_long_report.json"),
    "short": os.path.join(_cfg.REPORTS_DIR, "test_short_report.json"),
    "per_symbol": os.path.join(_cfg.REPORTS_DIR, "test_per_symbol_performance.csv"),
}

_FEATURE_PATHS: dict[str, str] = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "selected_features_long.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "selected_features_short.json"),
}


# ---------------------------------------------------------------------------
# OOS_Evaluator
# ---------------------------------------------------------------------------

class OOS_Evaluator:
    """
    Out-of-sample evaluator for Phase 5.

    Loads the final long/short strategies, prepares the test dataset with the
    same pipeline as training, runs the CPU backtest engine, and saves
    performance reports.

    Parameters
    ----------
    test_csv_path : str or None
        Path to the test CSV file.  Defaults to ``config.TEST_CSV_PATH``.
    """

    def __init__(self, test_csv_path: str | None = None) -> None:
        self.test_csv_path: str = test_csv_path or _cfg.TEST_CSV_PATH

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        allowed_directions: frozenset[str] | None = None,
    ) -> dict:
        """
        Run out-of-sample evaluation.

        Parameters
        ----------
        allowed_directions : frozenset[str] | None
            When set (full pipeline run), only these directions are loaded from
            disk. Use an empty frozenset to skip all directions (e.g. RB
            produced no rule sets this run). ``None`` loads every valid strategy
            file (standalone Phase 5).

        Returns
        -------
        dict
            Keys are strategy directions ("long", "short") present in the
            outputs directory.  Each value is the metrics dict returned by
            ``CPUBacktestEngine.simulate_rule_set()``.

        Also saves:
          - outputs/reports/test_long_report.json
          - outputs/reports/test_short_report.json
          - outputs/reports/test_per_symbol_performance.csv
        """
        # 1. Load strategies (whichever are available for this run)
        strategies = self.load_strategies(
            allowed_directions=allowed_directions)
        if not strategies:
            if allowed_directions is not None and not allowed_directions:
                logger.warning(
                    "Phase 5: no directions produced in the current pipeline run; "
                    "skipping OOS evaluation (stale strategy files are ignored)."
                )
            else:
                logger.warning(
                    "No strategy files found in %s. "
                    "Run Phase 2 and RB Governor first.",
                    _cfg.OUTPUTS_DIR,
                )
            return {}

        # 2. Prepare train / validation / test data
        datasets_by_split = self._load_datasets_by_split()
        test_df = datasets_by_split["test"]

        # 3. Evaluate each strategy on all splits and build reports
        results: dict[str, dict] = {}
        all_per_symbol: list[dict] = []

        for direction, strategy in strategies.items():
            logger.info(
                "Evaluating %s strategy on train / validation / test …", direction)

            metrics_by_split: dict[str, dict] = {}
            trade_logs_by_split: dict[str, pd.DataFrame] = {}

            for split, split_df in datasets_by_split.items():
                metrics, per_symbol_rows, trade_log = self._evaluate_strategy(
                    split_df, strategy, direction
                )
                metrics_by_split[split] = metrics
                trade_logs_by_split[split] = trade_log

                if split == "test":
                    all_per_symbol.extend(per_symbol_rows)

            results[direction] = {
                split: metrics_by_split[split]
                for split in ("train", "validation", "test")
                if split in metrics_by_split
            }

            test_metrics = metrics_by_split.get("test", {})
            test_return = float(test_metrics.get("total_return_pct", 0.0))
            if test_return < -5.0:
                logger.warning(
                    "Phase 5 [%s]: FAIL — test return %.2f%% is negative. "
                    "Strategy does not generalize.",
                    direction,
                    test_return,
                )

            # 4. Save per-direction report
            self._save_report(test_metrics, direction)

            selected_features = self._load_selected_features(direction)
            rule_set = strategy.get("rules_set", [])
            reporter = Reporter()

            try:
                reporter.plot_equity_curve(
                    trade_logs_by_split.get("test"), "test", direction)
            except Exception as exc:
                logger.warning(
                    "Reporter.plot_equity_curve (test/%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                reporter.plot_equity_curve(
                    trade_logs_by_split.get("train"), "train", direction)
            except Exception as exc:
                logger.warning(
                    "Reporter.plot_equity_curve (train/%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                reporter.plot_equity_curve(
                    trade_logs_by_split.get("validation"), "validation", direction)
            except Exception as exc:
                logger.warning(
                    "Reporter.plot_equity_curve (validation/%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                reporter.write_per_symbol_csv(
                    test_metrics, "test", direction=direction)
            except Exception as exc:
                logger.warning(
                    "Reporter.write_per_symbol_csv (test/%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                reporter.write_strategy_evaluation_table(
                    metrics_by_split,
                    trade_logs_by_split,
                    rule_set,
                    direction,
                )
            except Exception as exc:
                logger.warning(
                    "Reporter.write_strategy_evaluation_table (%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                reporter.plot_per_rule_breakdown(
                    rule_set,
                    trade_logs_by_split,
                    direction,
                )
            except Exception as exc:
                logger.warning(
                    "Reporter.plot_per_rule_breakdown (%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                reporter.plot_distribution_and_equity(
                    trade_logs_by_split,
                    direction,
                )
            except Exception as exc:
                logger.warning(
                    "Reporter.plot_distribution_and_equity (%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                reporter.write_spearman_correlation_report(
                    datasets_by_split,
                    selected_features,
                    direction,
                )
            except Exception as exc:
                logger.warning(
                    "Reporter.write_spearman_correlation_report (%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                reporter.write_feature_stratified_performance(
                    trade_logs_by_split,
                    rule_set,
                    selected_features,
                    datasets_by_split,
                    direction,
                )
            except Exception as exc:
                logger.warning(
                    "Reporter.write_feature_stratified_performance (%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                reporter.write_generalization_diagnostics(
                    metrics_by_split=metrics_by_split,
                    selected_features=selected_features,
                    datasets_by_split=datasets_by_split,
                    direction=direction,
                )
            except Exception as exc:
                logger.warning(
                    "Reporter.write_generalization_diagnostics (%s) failed (non-fatal): %s",
                    direction, exc,
                )

        # 5. Save per-symbol CSV
        self._save_per_symbol_csv(all_per_symbol)

        return results

    @staticmethod
    def load_strategies(
        allowed_directions: frozenset[str] | None = None,
    ) -> dict[str, dict]:
        """
        Load long.json and short.json via Output_Writer.load_and_validate().

        Parameters
        ----------
        allowed_directions : frozenset[str] | None
            Restrict to these directions. ``None`` loads any valid on-disk file.

        Returns a dict with keys "long" and/or "short" for whichever files
        exist and pass validation.  Missing or invalid files are silently
        skipped (with a WARNING log).
        """
        writer = Output_Writer()
        strategies: dict[str, dict] = {}

        for direction, path in _STRATEGY_PATHS.items():
            if (
                allowed_directions is not None
                and direction not in allowed_directions
            ):
                logger.info(
                    "Skipping %s strategy file %s: not produced in current "
                    "pipeline run",
                    direction,
                    path,
                )
                continue
            if not os.path.exists(path):
                logger.warning(
                    "Strategy file not found, skipping %s direction: %s",
                    direction,
                    path,
                )
                continue
            try:
                data = writer.load_and_validate(path)
                strategies[direction] = data
                logger.info("Loaded %s strategy from %s", direction, path)
            except ValidationError as exc:
                logger.warning(
                    "Strategy file failed validation, skipping %s direction: %s — %s",
                    direction,
                    path,
                    exc,
                )

        return strategies

    @staticmethod
    def prepare_test_data(test_csv_path: str) -> pd.DataFrame:
        """
        Prepare test data using Data_Loader.load_dataset().

        Applies the same preparation pipeline as training:
          1. Load CSV
          2. Sort by (symbol, datetime)
          3. Drop last 288 rows per symbol
          4. Drop NaN label rows
          5. Fill feature NaN with 0
          6. Compute _symbol_bar_index

        Parameters
        ----------
        test_csv_path : str
            Path to the test CSV file.

        Returns
        -------
        pd.DataFrame
            Prepared test DataFrame.
        """
        loader = Data_Loader()
        return loader.load_dataset(test_csv_path)

    def _load_datasets_by_split(self) -> dict[str, pd.DataFrame]:
        """Load prepared train, validation, and test datasets."""
        from gpu_fuzzy_trader.data.splitter import Data_Splitter, load_cached_split_if_fresh

        datasets: dict[str, pd.DataFrame] = {}

        cached = load_cached_split_if_fresh()
        if cached is not None:
            train_df, val_df, _, _, _ = cached
            datasets["train"] = train_df
            datasets["validation"] = val_df
            datasets["test"] = self.prepare_test_data(self.test_csv_path)
            logger.info(
                "Loaded cached train / validation splits and prepared test data: train=%d, validation=%d, test=%d",
                len(datasets["train"]),
                len(datasets["validation"]),
                len(datasets["test"]),
            )
            return datasets

        loader = Data_Loader()
        splitter = Data_Splitter()
        train_full = loader.load_dataset(_cfg.TRAIN_CSV_PATH)
        train_df, val_df, _cv_folds = splitter.split_and_persist(train_full)
        test_df = self.prepare_test_data(self.test_csv_path)

        datasets["train"] = train_df
        datasets["validation"] = val_df
        datasets["test"] = test_df

        logger.info(
            "Loaded and prepared datasets: train=%d, validation=%d, test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return datasets

    @staticmethod
    def _load_selected_features(direction: str) -> list[dict]:
        """Load selected features for a direction when available."""
        path = _FEATURE_PATHS.get(direction)
        if path is None or not os.path.exists(path):
            logger.warning(
                "Selected features file not found, skipping %s direction: %s",
                direction,
                path,
            )
            return []

        try:
            return Feature_Selector.load_and_validate(path)
        except ValueError as exc:
            logger.warning(
                "Selected features file failed validation, skipping %s direction: %s — %s",
                direction,
                path,
                exc,
            )
            return []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evaluate_strategy(
        self,
        test_df: pd.DataFrame,
        strategy: dict,
        direction: str,
    ) -> tuple[dict, list[dict], pd.DataFrame]:
        """
        Evaluate a single strategy on the test DataFrame.

        Returns
        -------
        metrics : dict
            Performance metrics dict from CPUBacktestEngine.
        per_symbol_rows : list[dict]
            Flat list of per-symbol metric dicts (for CSV output).
        trade_log : pd.DataFrame
            Trade log DataFrame (for equity curve reporting).
        """
        rule_set = strategy.get("rules_set", [])

        # feature_modes is not used for rule matching (threshold-based),
        # but the engine interface requires it.
        engine = CPUBacktestEngine(
            df=test_df,
            feature_modes={},
            direction=direction,
        )

        try:
            metrics, trade_log = engine.simulate_rule_set(
                rule_set, return_logs=True)
        except ValueError as exc:
            logger.warning(
                "Phase 5 [%s]: simulate_rule_set failed for this split; treating as no trades: %s",
                direction,
                exc,
            )
            metrics = {
                "direction": direction,
                "total_return_pct": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown_pct": 0.0,
                "win_rate": 0.0,
                "account_ruined": False,
                "loss_count": 0,
                "time_closed_count": 0,
                "raw_signal_count": 0,
                "executed_trades": 0,
                "final_equity": _cfg.INITIAL_CAPITAL,
                "profit_factor": 0.0,
                "avg_position_notional": 0.0,
                "skipped_min_notional_count": 0,
                "max_simultaneous_positions": 0,
                "max_total_open_exposure": 0.0,
                "per_symbol_metrics": {},
            }
            trade_log = pd.DataFrame()

        # Requirement 11.4: zero-trade case — do NOT report account ruin
        # unless equity actually reached zero.
        if metrics["executed_trades"] == 0:
            metrics["account_ruined"] = False
            metrics["total_return_pct"] = 0.0

        # Requirement 11.5: per-symbol breakdowns
        per_symbol_rows = self._build_per_symbol_rows(
            metrics, direction, trade_log
        )

        return metrics, per_symbol_rows, trade_log

    @staticmethod
    def _build_per_symbol_rows(
        metrics: dict,
        direction: str,
        trade_log: pd.DataFrame,
    ) -> list[dict]:
        """
        Build a flat list of per-symbol metric dicts for CSV output.

        Uses the per_symbol_metrics already computed by the engine (which
        leverages the trade log when return_logs=True for accurate win rates).
        """
        rows: list[dict] = []
        per_sym = metrics.get("per_symbol_metrics", {})

        for symbol, sym_metrics in per_sym.items():
            rows.append(
                {
                    "direction": direction,
                    "symbol": symbol,
                    "trade_count": sym_metrics.get("trade_count", 0),
                    "win_rate": sym_metrics.get("win_rate", 0.0),
                    "net_pnl": sym_metrics.get("net_pnl", 0.0),
                }
            )

        return rows

    def _save_report(self, metrics: dict, direction: str) -> None:
        """Save a per-direction JSON report to outputs/reports/."""
        report_path = _REPORT_PATHS[direction]
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        # Build a clean, serialisable report dict
        report = {
            "direction": direction,
            "total_return_pct": metrics.get("total_return_pct", 0.0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "executed_trades": metrics.get("executed_trades", 0),
            "account_status": (
                "ruined" if metrics.get(
                    "account_ruined", False) else "survived"
            ),
            "final_equity": metrics.get("final_equity", _cfg.INITIAL_CAPITAL),
            "per_symbol_metrics": metrics.get("per_symbol_metrics", {}),
        }

        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

        logger.info("Saved %s report to %s", direction, report_path)

    @staticmethod
    def _save_per_symbol_csv(rows: list[dict]) -> None:
        """Save the combined per-symbol performance CSV."""
        csv_path = _REPORT_PATHS["per_symbol"]
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

        if rows:
            df = pd.DataFrame(rows)
        else:
            df = pd.DataFrame(
                columns=["direction", "symbol",
                         "trade_count", "win_rate", "net_pnl"]
            )

        df.to_csv(csv_path, index=False)
        logger.info("Saved per-symbol performance CSV to %s", csv_path)

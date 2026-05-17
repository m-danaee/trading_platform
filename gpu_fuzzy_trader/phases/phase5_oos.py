"""
phase5_oos.py — OOS_Evaluator (Phase 5)

Final out-of-sample evaluation on the held-out test.csv.

Workflow:
  1. Load outputs/long.json and outputs/short.json via Output_Writer.load_and_validate()
     (handles the case where only one strategy file exists)
  2. Prepare data/test.csv with the same pipeline as training:
       - Sort by (symbol, datetime)
       - Drop last 288 rows per symbol
       - Drop NaN label rows
       - Fill feature NaN with 0
       - Compute _symbol_bar_index
  3. Evaluate each available strategy using CPUBacktestEngine.simulate_rule_set()
     with return_logs=True
  4. Compute per-symbol breakdowns from trade logs
  5. Handle zero-trade case: report 0% total return; do NOT report account ruin
     unless equity actually reached zero
  6. Save outputs:
       outputs/reports/test_long_report.json
       outputs/reports/test_short_report.json
       outputs/reports/test_per_symbol_performance.csv

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
from gpu_fuzzy_trader.data.loader import Data_Loader
from gpu_fuzzy_trader.output.writer import Output_Writer, ValidationError
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

    def run(self) -> dict:
        """
        Run out-of-sample evaluation.

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
        # 1. Load strategies (whichever are available)
        strategies = self.load_strategies()
        if not strategies:
            logger.warning(
                "No strategy files found in %s. "
                "Run Phase 3 (and optionally Phase 4) first.",
                _cfg.OUTPUTS_DIR,
            )
            return {}

        # 2. Prepare test data
        logger.info("Preparing test data from %s", self.test_csv_path)
        test_df = self.prepare_test_data(self.test_csv_path)
        logger.info(
            "Test data prepared: %d rows, %d symbols",
            len(test_df),
            test_df["symbol"].nunique() if "symbol" in test_df.columns else 0,
        )

        # 3. Evaluate each strategy
        results: dict[str, dict] = {}
        all_per_symbol: list[dict] = []

        for direction, strategy in strategies.items():
            logger.info("Evaluating %s strategy on test data …", direction)
            metrics, per_symbol_rows, test_log = self._evaluate_strategy(
                test_df, strategy, direction
            )
            results[direction] = metrics
            all_per_symbol.extend(per_symbol_rows)

            # 4. Save per-direction report
            self._save_report(metrics, direction)

            # Reporter: equity curve and per-symbol CSV for test split
            try:
                Reporter().plot_equity_curve(test_log, "test", direction)
            except Exception as exc:
                logger.warning(
                    "Reporter.plot_equity_curve (test/%s) failed (non-fatal): %s",
                    direction, exc,
                )
            try:
                Reporter().write_per_symbol_csv(metrics, "test")
            except Exception as exc:
                logger.warning(
                    "Reporter.write_per_symbol_csv (test/%s) failed (non-fatal): %s",
                    direction, exc,
                )

        # 5. Save per-symbol CSV
        self._save_per_symbol_csv(all_per_symbol)

        return results

    @staticmethod
    def load_strategies() -> dict[str, dict]:
        """
        Load long.json and short.json via Output_Writer.load_and_validate().

        Returns a dict with keys "long" and/or "short" for whichever files
        exist and pass validation.  Missing or invalid files are silently
        skipped (with a WARNING log).
        """
        writer = Output_Writer()
        strategies: dict[str, dict] = {}

        for direction, path in _STRATEGY_PATHS.items():
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

        metrics, trade_log = engine.simulate_rule_set(rule_set, return_logs=True)

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
                "ruined" if metrics.get("account_ruined", False) else "survived"
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
                columns=["direction", "symbol", "trade_count", "win_rate", "net_pnl"]
            )

        df.to_csv(csv_path, index=False)
        logger.info("Saved per-symbol performance CSV to %s", csv_path)

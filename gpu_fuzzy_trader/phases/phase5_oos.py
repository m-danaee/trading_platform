"""
phase5_oos.py — OOS_Evaluator (Phase 5)

Final out-of-sample diagnostics on the consumed test_new.csv, with an optional
untouched forward tape used for acceptance.

Workflow:
  1. Load outputs/long.json and outputs/short.json via Output_Writer.load_and_validate()
      (handles the case where only one strategy file exists)
  2. Prepare train, validation, and test data with the same pipeline as training:
         - Sort by (datetime, symbol)
         - Attach exact first-touch outcomes, then drop rows with unavailable labels
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
import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.backtest.joint_engine import JointPortfolioEngine
from gpu_fuzzy_trader.data.loader import Data_Loader, validate_context_columns
from gpu_fuzzy_trader.features.fuzzy_scaling import (
    apply_fuzzy_feature_scaling,
    fit_fuzzy_feature_scaling,
)
from gpu_fuzzy_trader.features.selector import Feature_Selector
from gpu_fuzzy_trader.output.writer import (
    Output_Writer,
    ValidationError,
)
from gpu_fuzzy_trader.research_integrity import (
    reserve_forward_evaluation,
    write_forward_acceptance_record,
)
from gpu_fuzzy_trader.reporting.reporter import Reporter
from gpu_fuzzy_trader.mtf.runtime import evaluate_candidate_rule_masks
from gpu_fuzzy_trader.validation.uncertainty import compute_trade_uncertainty

logger = logging.getLogger(__name__)


class _Phase5JSONEncoder(json.JSONEncoder):
    """Keep numeric report values numeric instead of stringifying NumPy scalars."""

    def default(self, value):  # noqa: D401 - JSONEncoder hook
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return str(value)

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
    "joint": os.path.join(_cfg.REPORTS_DIR, "test_joint_portfolio_report.json"),
    "forward_long": os.path.join(_cfg.REPORTS_DIR, "forward_long_report.json"),
    "forward_short": os.path.join(_cfg.REPORTS_DIR, "forward_short_report.json"),
    "forward_joint": os.path.join(
        _cfg.REPORTS_DIR, "forward_joint_portfolio_report.json"
    ),
}

# Reporter outputs are derived from the current Phase 5 run.  Clear this
# fixed, explicit set before evaluation so a fail-closed/no-strategy run
# cannot leave an older equity curve or CSV looking current.  Phase 2/RB
# artifacts are intentionally not included here.
_PHASE5_DERIVED_REPORT_NAMES: tuple[str, ...] = (
    "test_long_report.json",
    "test_short_report.json",
    "test_joint_portfolio_report.json",
    "test_per_symbol_performance.csv",
    "forward_long_report.json",
    "forward_short_report.json",
    "forward_joint_portfolio_report.json",
    "test_long_equity.png",
    "test_short_equity.png",
    "train_long_equity.png",
    "train_short_equity.png",
    "validation_long_equity.png",
    "validation_short_equity.png",
    "test_long_per_symbol_performance.csv",
    "test_short_per_symbol_performance.csv",
    "train_long_per_symbol_performance.csv",
    "train_short_per_symbol_performance.csv",
    "validation_long_per_symbol_performance.csv",
    "validation_short_per_symbol_performance.csv",
    "train_per_symbol_performance.csv",
    "validation_per_symbol_performance.csv",
    "strategy_evaluation_long.csv",
    "strategy_evaluation_short.csv",
    "per_rule_breakdown_long.png",
    "per_rule_breakdown_short.png",
    "distribution_equity_test_long.png",
    "distribution_equity_test_short.png",
    "distribution_equity_train_long.png",
    "distribution_equity_train_short.png",
    "distribution_equity_validation_long.png",
    "distribution_equity_validation_short.png",
    "spearman_correlation_long.csv",
    "spearman_correlation_short.csv",
    "feature_stratified_train_long.csv",
    "feature_stratified_train_short.csv",
    "feature_stratified_validation_long.csv",
    "feature_stratified_validation_short.csv",
    "feature_stratified_test_long.csv",
    "feature_stratified_test_short.csv",
    "generalization_diagnostics_long.json",
    "generalization_diagnostics_short.json",
)

_FEATURE_PATHS: dict[str, str] = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "selected_features_long.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "selected_features_short.json"),
}


def _validate_forward_metadata(
    test_meta: pd.DataFrame,
    forward_meta: pd.DataFrame,
) -> dict[str, Any]:
    """Validate source-tape identity before a forward evaluation is consumed."""
    expected_seconds = int(getattr(_cfg, "CONTEXT_BAR_SECONDS", 15 * 60))
    expected_delta = pd.Timedelta(seconds=expected_seconds)

    def normalise(
        frame: pd.DataFrame,
        name: str,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        required = {"datetime", "symbol"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} tape is missing metadata columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} tape is empty; refusing forward evaluation")
        timestamps = pd.to_datetime(
            frame["datetime"], errors="raise", utc=True,
        )
        if timestamps.isna().any():
            raise ValueError(f"{name} tape contains missing timestamps")
        symbols = frame["symbol"].astype("string").str.strip().str.upper()
        if symbols.isna().any() or (symbols == "").any():
            raise ValueError(f"{name} tape contains missing or empty symbols")
        normalised = pd.DataFrame({
            "datetime": timestamps,
            "symbol": symbols,
        })
        if normalised.duplicated(["datetime", "symbol"]).any():
            raise ValueError(
                f"{name} tape contains duplicate (datetime, symbol) bars"
            )

        first_by_symbol: dict[str, str] = {}
        last_by_symbol: dict[str, str] = {}
        counts: dict[str, int] = {}
        for symbol, group in normalised.groupby("symbol", sort=True):
            ordered = group.sort_values("datetime")["datetime"]
            deltas = ordered.diff().dropna()
            if not deltas.eq(expected_delta).all():
                bad = deltas.loc[~deltas.eq(expected_delta)].iloc[0]
                raise ValueError(
                    f"{name} tape is not contiguous at {symbol}: "
                    f"observed interval {bad}, expected {expected_delta}"
                )
            first_by_symbol[str(symbol)] = ordered.iloc[0].isoformat()
            last_by_symbol[str(symbol)] = ordered.iloc[-1].isoformat()
            counts[str(symbol)] = len(ordered)

        ranges = set(zip(first_by_symbol.values(), last_by_symbol.values()))
        if len(ranges) > 1:
            raise ValueError(
                f"{name} tape does not cover all symbols on the same interval"
            )
        return normalised, {
            "rows": len(normalised),
            "symbols": sorted(counts),
            "rows_per_symbol": counts,
            "min_datetime": normalised["datetime"].min().isoformat(),
            "max_datetime": normalised["datetime"].max().isoformat(),
            "contiguous_cadence_seconds": expected_seconds,
        }

    test, test_summary = normalise(test_meta, "test")
    forward, forward_summary = normalise(forward_meta, "forward")
    test_symbols = set(test_summary["symbols"])
    forward_symbols = set(forward_summary["symbols"])
    if forward_symbols != test_symbols:
        raise ValueError(
            "FORWARD_CSV_PATH must contain exactly the consumed test symbol "
            f"universe {sorted(test_symbols)}; got {sorted(forward_symbols)}."
        )

    test_last = test.groupby("symbol")["datetime"].max()
    forward_first = forward.groupby("symbol")["datetime"].min()
    overlap = [
        str(symbol)
        for symbol in sorted(test_symbols)
        if forward_first[symbol] <= test_last[symbol]
    ]
    if overlap:
        raise ValueError(
            "FORWARD_CSV_PATH overlaps the consumed test tape for symbols: "
            f"{overlap}"
        )

    if forward["datetime"].min() <= test["datetime"].max():
        raise ValueError(
            "FORWARD_CSV_PATH must start strictly after the complete "
            "consumed test tape; refusing overlapping acceptance data."
        )

    minimum_rows = int(getattr(_cfg, "TAIL_DROP_ROWS", 0)) + 1
    too_short = {
        symbol: count
        for symbol, count in forward_summary["rows_per_symbol"].items()
        if int(count) < minimum_rows
    }
    if too_short:
        raise ValueError(
            "FORWARD_CSV_PATH must contain more than the label horizon for "
            f"every symbol; short tapes: {too_short}"
        )

    return {
        "expected_bar_seconds": expected_seconds,
        "test": test_summary,
        "forward": forward_summary,
        "validation_contract": (
            "strictly_later_same_universe_contiguous_15m_tape; "
            "minimum_one_post_horizon_row_per_symbol"
        ),
    }


def _forward_positive_result(result: dict[str, Any] | None) -> bool:
    """Return whether a forward result is a successful positive return."""
    if not isinstance(result, dict) or str(
        result.get("evaluation_status", "")
    ) != "ok":
        return False
    try:
        total_return = float(result.get("total_return_pct", -float("inf")))
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(total_return) and total_return > 0.0)


def _forward_result_passes_gate(
    result: dict[str, Any] | None,
    *,
    minimum_trades: int | None = None,
    require_ci_positive: bool | None = None,
) -> bool:
    """Apply the non-selection acceptance gate to one forward result."""
    if not _forward_positive_result(result):
        return False
    required_trades = int(
        getattr(_cfg, "PHASE5_FORWARD_MIN_TRADES", 10)
        if minimum_trades is None else minimum_trades
    )
    try:
        if int(result.get("executed_trades", 0) or 0) < required_trades:
            return False
    except (TypeError, ValueError):
        return False
    ci_required = bool(
        getattr(_cfg, "PHASE5_FORWARD_REQUIRE_CI_POSITIVE", True)
        if require_ci_positive is None else require_ci_positive
    )
    if not ci_required:
        return True
    uncertainty = result.get("trade_uncertainty", {})
    ci = (
        uncertainty.get("mean_trade_return_ci95_pct", [])
        if isinstance(uncertainty, dict)
        else []
    )
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return False
    try:
        lower_bound = float(ci[0])
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(lower_bound) and lower_bound > 0.0)


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
        Path to the consumed diagnostic test CSV. Defaults to
        ``config.TEST_CSV_PATH``.
    forward_csv_path : str or None
        Optional untouched future CSV. If absent, no deployment acceptance is
        possible and test results are explicitly diagnostic-only.
    """

    def __init__(
        self,
        test_csv_path: str | None = None,
        forward_csv_path: str | None = None,
        run_id: str | None = None,
    ) -> None:
        self.test_csv_path: str = test_csv_path or _cfg.TEST_CSV_PATH
        self.run_id = str(run_id) if run_id else None
        configured_forward = (
            forward_csv_path
            if forward_csv_path is not None
            else getattr(_cfg, "FORWARD_CSV_PATH", None)
        )
        self.forward_csv_path: str | None = (
            str(configured_forward).strip() if configured_forward else None
        )
        self._forward_acceptance_metadata: dict | None = None
        self._forward_validation_summary: dict[str, Any] | None = None

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
        # Reports are derived artifacts.  Remove the previous run's small,
        # fixed set of Phase 5 reports before loading anything so a failed or
        # partial run cannot leave an apparently current result on disk.
        self._clear_previous_reports()

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

        # 3. Evaluate each strategy on all splits and build reports
        results: dict[str, dict] = {}
        all_per_symbol: list[dict] = []

        for direction, strategy in strategies.items():
            logger.info(
                "Evaluating %s strategy on train / validation / test …", direction)

            metrics_by_split: dict[str, dict] = {}
            trade_logs_by_split: dict[str, pd.DataFrame] = {}

            for split, split_df in datasets_by_split.items():
                history_df = None
                is_mtf_strategy = "mtf_candidate" in strategy
                if is_mtf_strategy and split == "validation":
                    history_df = datasets_by_split.get("train")
                elif is_mtf_strategy and split == "test":
                    history_df = pd.concat(
                        [datasets_by_split["train"], datasets_by_split["validation"]],
                        ignore_index=True,
                        sort=False,
                    )
                elif is_mtf_strategy and split == "forward":
                    history_df = pd.concat(
                        [
                            datasets_by_split["train"],
                            datasets_by_split["validation"],
                            datasets_by_split["test"],
                        ],
                        ignore_index=True,
                        sort=False,
                    )
                if is_mtf_strategy:
                    metrics, per_symbol_rows, trade_log = self._evaluate_strategy(
                        split_df,
                        strategy,
                        direction,
                        history_df=history_df,
                    )
                else:
                    # Keep the established three-argument call contract for
                    # legacy evaluators and lightweight test doubles.
                    metrics, per_symbol_rows, trade_log = self._evaluate_strategy(
                        split_df, strategy, direction
                    )
                split_seed_offset = {
                    "train": 0,
                    "validation": 101,
                }.get(split, 202)
                metrics["trade_uncertainty"] = compute_trade_uncertainty(
                    trade_log,
                    seed=(
                        int(getattr(_cfg, "GLOBAL_SEED", 42) or 42)
                        + (1 if direction == "long" else 2)
                        + split_seed_offset
                    ),
                )
                metrics_by_split[split] = metrics
                trade_logs_by_split[split] = trade_log

                if split == "test":
                    all_per_symbol.extend(per_symbol_rows)

            results[direction] = dict(metrics_by_split)

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
            self._save_report(test_metrics, direction, split="test")
            if "forward" in metrics_by_split:
                self._save_report(
                    metrics_by_split["forward"], direction, split="forward",
                )

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
        # The direction-specific reports above are useful for diagnosing each
        # specialist.  Deployment still owns one account, so run a second
        # joint pass that nets BTC/ETH and long/short positions and applies
        # same-side suppression/reversal at the next tradable open.
        joint_by_split: dict[str, dict] = {}
        joint_test_logs = pd.DataFrame()
        try:
            for split, split_df in datasets_by_split.items():
                joint_history_df = None
                if any(
                    "mtf_candidate" in strategy
                    for strategy in strategies.values()
                ):
                    if split == "validation":
                        joint_history_df = datasets_by_split.get("train")
                    elif split == "test":
                        joint_history_df = pd.concat(
                            [
                                datasets_by_split["train"],
                                datasets_by_split["validation"],
                            ],
                            ignore_index=True,
                            sort=False,
                        )
                    elif split == "forward":
                        joint_history_df = pd.concat(
                            [
                                datasets_by_split["train"],
                                datasets_by_split["validation"],
                                datasets_by_split["test"],
                            ],
                            ignore_index=True,
                            sort=False,
                        )
                joint_metrics, joint_logs = JointPortfolioEngine(
                    split_df,
                ).simulate(
                    strategies,
                    return_logs=True,
                    history_df=joint_history_df,
                )
                # Forward acceptance is fail-closed.  The joint engine is a
                # successful evaluator only when it explicitly reports it.
                joint_metrics.setdefault("evaluation_status", "ok")
                split_seed_offset = {
                    "train": 0,
                    "validation": 101,
                }.get(split, 202)
                joint_metrics["trade_uncertainty"] = compute_trade_uncertainty(
                    joint_logs,
                    seed=int(getattr(_cfg, "GLOBAL_SEED", 42) or 42)
                    + split_seed_offset,
                )
                joint_by_split[split] = joint_metrics
                if split == "test":
                    joint_test_logs = joint_logs
            results["joint_portfolio"] = dict(joint_by_split)
            self._save_report(
                joint_by_split.get("test", {}), "joint_portfolio", split="test",
            )
            if "forward" in joint_by_split:
                self._save_report(
                    joint_by_split["forward"],
                    "joint_portfolio",
                    split="forward",
                )
            all_per_symbol.extend(
                self._build_per_symbol_rows(
                    joint_by_split.get("test", {}),
                    "joint_portfolio",
                    joint_test_logs,
                )
            )
            logger.info(
                "Phase 5 [joint]: test return %.2f%%, trades=%d, reversals=%d",
                float(joint_by_split.get("test", {}).get("total_return_pct", 0.0)),
                int(joint_by_split.get("test", {}).get("executed_trades", 0)),
                int(joint_by_split.get("test", {}).get("reversal_count", 0)),
            )
        except Exception as exc:
            logger.error(
                "Phase 5 joint portfolio evaluation failed (non-fatal to "
                "direction reports): %s",
                exc,
                exc_info=True,
            )

        self._save_per_symbol_csv(all_per_symbol)

        # The consumed test period is intentionally never an acceptance gate.
        # A forward decision requires both direction specialists and the joint
        # account to be profitable on a new, untouched period.
        forward_available = "forward" in datasets_by_split

        forward_direction_ok = all(
            _forward_result_passes_gate(results.get(direction, {}).get("forward"))
            for direction in ("long", "short")
        )
        forward_joint_ok = _forward_result_passes_gate(
            results.get("joint_portfolio", {}).get("forward")
        )
        results["acceptance"] = {
            "status": (
                "accepted"
                if forward_available and forward_direction_ok and forward_joint_ok
                else "diagnostic_only"
                if not forward_available
                else "rejected"
            ),
            "forward_dataset": self.forward_csv_path if forward_available else None,
            "test_dataset_is_diagnostic_only": True,
            "requires_positive_long_short_and_joint_forward_return": True,
            "requires_successful_nonzero_trade_evaluation": True,
            "requires_minimum_forward_trades": int(
                getattr(_cfg, "PHASE5_FORWARD_MIN_TRADES", 10)
            ),
            "requires_positive_bootstrap_mean_return_ci": bool(
                getattr(_cfg, "PHASE5_FORWARD_REQUIRE_CI_POSITIVE", True)
            ),
            "long_positive": bool(
                _forward_positive_result(results.get("long", {}).get("forward"))
            ),
            "short_positive": bool(
                _forward_positive_result(results.get("short", {}).get("forward"))
            ),
            "long_gate_pass": bool(
                _forward_result_passes_gate(
                    results.get("long", {}).get("forward")
                )
            ),
            "short_gate_pass": bool(
                _forward_result_passes_gate(
                    results.get("short", {}).get("forward")
                )
            ),
            "joint_positive": bool(
                _forward_positive_result(
                    results.get("joint_portfolio", {}).get("forward")
                )
            ),
            "joint_gate_pass": bool(forward_joint_ok),
        }

        if forward_available:
            results["acceptance"]["forward_tape_validation"] = (
                self._forward_validation_summary
            )
        if forward_available and self._forward_acceptance_metadata is not None:
            write_forward_acceptance_record(
                _cfg.OUTPUTS_DIR,
                self._forward_acceptance_metadata,
                results["acceptance"],
            )

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
                with open(path, encoding="utf-8") as fh:
                    raw_data = json.load(fh)
                if (
                    isinstance(raw_data, dict)
                    and raw_data.get("deployment_accepted") is False
                ):
                    logger.warning(
                        "Strategy explicitly rejected by RB, skipping %s direction: %s",
                        direction,
                        path,
                    )
                    continue
                data = writer.load_and_validate(path)
                declared_direction = str(data.get("direction", "")).strip().lower()
                if declared_direction != direction:
                    raise ValidationError(
                        f"strategy direction {declared_direction!r} does not "
                        f"match {direction}.json"
                    )
                if "mtf_candidate" in data:
                    manifest = data.get("mtf_manifest")
                    if not isinstance(manifest, dict) or manifest.get("frozen_runtime") is not True:
                        raise ValidationError(
                            "MTF strategy is missing a frozen mtf_manifest contract"
                        )
                    candidate_payload = data.get("mtf_candidate")
                    if not isinstance(candidate_payload, dict):
                        raise ValidationError(
                            "MTF strategy is missing its serialized candidate"
                        )
                    if candidate_payload.get("mtf_manifest") != manifest:
                        raise ValidationError(
                            "MTF candidate manifest does not match strategy manifest"
                        )
                    provenance = data.get("provenance")
                    expected_manifest_hash = hashlib.sha256(
                        json.dumps(manifest, sort_keys=True).encode("utf-8")
                    ).hexdigest()
                    if (
                        not isinstance(provenance, dict)
                        or str(provenance.get("mtf_manifest_hash", ""))
                        != expected_manifest_hash
                    ):
                        raise ValidationError(
                            "MTF strategy manifest hash is missing or does not match"
                        )
                    from gpu_fuzzy_trader.mtf.archives import (
                        compute_rule_hash,
                        load_mtf_archive_payload,
                    )

                    declared_archives = manifest.get("archives", {})
                    if not isinstance(declared_archives, dict):
                        raise ValidationError(
                            "MTF frozen manifest has an invalid archives mapping"
                        )
                    loaded_payloads: dict[str, dict[str, Any]] = {}
                    for timeframe in ("hwc", "mwc", "lwc"):
                        expected_hash = str(
                            declared_archives.get(f"{timeframe}_archive_hash", "")
                        )
                        if not expected_hash:
                            raise ValidationError(
                                f"MTF frozen manifest is missing the {timeframe} archive hash"
                            )
                        archive_path = getattr(_cfg, "MTF_ARCHIVE_PATHS", {}).get(timeframe)
                        if not archive_path or not os.path.exists(archive_path):
                            raise ValidationError(
                                f"MTF {timeframe} archive required by frozen manifest is missing"
                            )
                        payload = load_mtf_archive_payload(archive_path)
                        if str(payload.get("archive_hash", "")) != expected_hash:
                            raise ValidationError(
                                f"MTF {timeframe} archive hash does not match frozen manifest"
                            )
                        loaded_payloads[timeframe] = payload

                    # Strictly bind serialized candidate rules to verified archive contents
                    candidate_hwc_hashes = [
                        compute_rule_hash(r) for r in candidate_payload.get("hwc_rules", [])
                    ]
                    archive_hwc_hashes = [
                        compute_rule_hash(r) for r in loaded_payloads["hwc"].get("rules", [])
                    ]
                    if candidate_hwc_hashes != archive_hwc_hashes:
                        raise ValidationError(
                            "Candidate HWC rules do not match the frozen HWC archive payload"
                        )

                    candidate_mwc_hashes = [
                        compute_rule_hash(r) for r in candidate_payload.get("mwc_rules", [])
                    ]
                    archive_mwc_hashes = [
                        compute_rule_hash(r) for r in loaded_payloads["mwc"].get("rules", [])
                    ]
                    if candidate_mwc_hashes != archive_mwc_hashes:
                        raise ValidationError(
                            "Candidate MWC rules do not match the frozen MWC archive payload"
                        )

                    from collections import Counter

                    candidate_lwc_rules = candidate_payload.get(
                        "lwc_rules", [])
                    if not candidate_lwc_rules:
                        raise ValidationError(
                            "MTF candidate is missing LWC rules")
                    candidate_lwc_hashes = [
                        compute_rule_hash(r) for r in candidate_lwc_rules
                    ]
                    strategy_rules_hashes = [
                        compute_rule_hash(r) for r in raw_data.get("rules_set", [])
                    ]
                    if candidate_lwc_hashes != strategy_rules_hashes:
                        raise ValidationError(
                            "Candidate LWC rules do not match the strategy rules_set"
                        )

                    archive_lwc_rules = loaded_payloads["lwc"].get("rules", [])
                    archive_directional_rules = [
                        r for r in archive_lwc_rules
                        if str(r.get("direction", "")).lower() == direction.lower()
                    ]
                    matching_archive_rules = (
                        archive_directional_rules
                        if archive_directional_rules
                        else archive_lwc_rules
                    )
                    archive_lwc_hashes = [
                        compute_rule_hash(r) for r in matching_archive_rules
                    ]
                    candidate_counter = Counter(candidate_lwc_hashes)
                    archive_counter = Counter(archive_lwc_hashes)
                    if candidate_counter != archive_counter:
                        raise ValidationError(
                            "Candidate LWC rule multiset does not exactly "
                            "match the frozen LWC archive payload"
                        )


                strategies[direction] = data
                logger.info("Loaded %s strategy from %s", direction, path)
            except (OSError, json.JSONDecodeError, ValidationError) as exc:
                logger.warning(
                    "Strategy file failed validation, skipping %s direction: %s — %s",
                    direction,
                    path,
                    exc,
                )

        return strategies

    @staticmethod
    def prepare_test_data(test_csv_path: str, *, require_context: bool | None = None) -> pd.DataFrame:
        """
        Prepare test data using Data_Loader.load_dataset().

        Applies the same preparation pipeline as training:
          1. Load CSV
          2. Sort by (datetime, symbol)
          3. Attach exact first-touch outcomes before label-tail trimming
          4. Drop rows with unavailable labels
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
        if require_context is None:
            require_context = bool(getattr(_cfg, "REQUIRE_CONTEXT_COLUMNS", False))
        return loader.load_dataset(
            test_csv_path,
            drop_tail=True,
            include_barrier_outcomes=True,
            require_context=require_context,
        )

    def _load_datasets_by_split(self) -> dict[str, pd.DataFrame]:
        """Load prepared train, validation, and test datasets."""
        from gpu_fuzzy_trader.data.splitter import Data_Splitter, load_cached_split_if_fresh

        datasets: dict[str, pd.DataFrame] = {}

        mtf_mode = bool(getattr(_cfg, "MTF_PIPELINE_ENABLED", False))
        if mtf_mode and any(
            Path(path).name.endswith("_hwc_mwc_lwc.csv")
            for path in (_cfg.TRAIN_CSV_PATH, self.test_csv_path)
        ):
            raise RuntimeError(
                "Frozen MTF OOS evaluation requires raw 15m tapes; enriched "
                "HWC/MWC/LWC inputs are not valid canonical sources."
            )
        cached = None if mtf_mode else load_cached_split_if_fresh()
        if cached is not None:
            train_df, val_df, _, _, _ = cached
            # Cached splits are an optimization, never an exemption from the
            # same context contract required for Phase 2 and OOS scoring.
            if not mtf_mode:
                validate_context_columns(train_df)
                validate_context_columns(val_df)
            datasets["train"] = train_df
            datasets["validation"] = val_df
            datasets["test"] = self.prepare_test_data(self.test_csv_path)
            forward_df = self._load_forward_data()
            if forward_df is not None:
                datasets["forward"] = forward_df
            scaling = fit_fuzzy_feature_scaling(datasets["train"])
            for frame in datasets.values():
                apply_fuzzy_feature_scaling(frame, scaling)
            logger.info(
                "Loaded cached train / validation splits and prepared test data: train=%d, validation=%d, test=%d",
                len(datasets["train"]),
                len(datasets["validation"]),
                len(datasets["test"]),
            )
            return datasets

        loader = Data_Loader()
        splitter = Data_Splitter()
        train_full = loader.load_dataset(
            _cfg.TRAIN_CSV_PATH,
            drop_tail=False,
            include_barrier_outcomes=True,
            require_context=(
                False
                if mtf_mode
                else bool(getattr(_cfg, "REQUIRE_CONTEXT_COLUMNS", False))
            ),
        )
        train_full = train_full.dropna(
            subset=list(_cfg.LABEL_COLUMNS),
        ).reset_index(drop=True)
        train_df, val_df, _cv_folds = splitter.split_and_persist(train_full)
        test_df = self.prepare_test_data(self.test_csv_path)
        datasets["train"] = train_df
        datasets["validation"] = val_df
        datasets["test"] = test_df
        forward_df = self._load_forward_data()
        if forward_df is not None:
            datasets["forward"] = forward_df
        scaling = fit_fuzzy_feature_scaling(datasets["train"])
        for frame in datasets.values():
            apply_fuzzy_feature_scaling(frame, scaling)

        logger.info(
            "Loaded and prepared datasets: train=%d, validation=%d, test=%d",
            len(train_df),
            len(val_df),
            len(test_df),
        )
        return datasets

    def _load_forward_data(self) -> pd.DataFrame | None:
        """Load and validate a strictly newer, untouched forward period."""
        if not self.forward_csv_path:
            return None
        if not os.path.exists(self.forward_csv_path):
            raise FileNotFoundError(
                "Configured FORWARD_CSV_PATH does not exist: "
                f"{self.forward_csv_path}"
            )
        if bool(getattr(_cfg, "FORWARD_ACCEPTANCE_ONCE", True)):
            self._forward_acceptance_metadata = reserve_forward_evaluation(
                self.forward_csv_path,
                _cfg.OUTPUTS_DIR,
            )

        # Compare source-tape timestamps, including rows whose labels are not
        # yet available.  This prevents accidentally treating the unlabelled
        # tail of test_new.csv as a new forward period.
        test_meta = pd.read_csv(
            self.test_csv_path, usecols=["datetime", "symbol"],
        )
        forward_meta = pd.read_csv(
            self.forward_csv_path, usecols=["datetime", "symbol"],
        )
        self._forward_validation_summary = _validate_forward_metadata(
            test_meta,
            forward_meta,
        )
        prepared = self.prepare_test_data(self.forward_csv_path)
        expected_symbols = set(self._forward_validation_summary["test"]["symbols"])
        prepared_symbols = {
            str(value).strip().upper()
            for value in prepared.get("symbol", pd.Series(dtype=str)).dropna().unique()
        }
        if prepared.empty or prepared_symbols != expected_symbols:
            raise ValueError(
                "FORWARD_CSV_PATH has no complete post-horizon rows for the "
                "consumed symbol universe"
            )
        expected_counts = {
            symbol: int(count) - int(getattr(_cfg, "TAIL_DROP_ROWS", 0))
            for symbol, count in self._forward_validation_summary[
                "forward"
            ]["rows_per_symbol"].items()
        }
        actual_counts = {
            str(symbol).strip().upper(): int(len(group))
            for symbol, group in prepared.groupby(
                "symbol", sort=True, observed=False,
            )
        }
        if actual_counts != expected_counts:
            raise ValueError(
                "FORWARD_CSV_PATH lost rows during label preparation; "
                f"expected {expected_counts}, got {actual_counts}"
            )
        return prepared

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

    @staticmethod
    def _clear_previous_reports() -> None:
        """Remove only known Phase 5 artifacts from the active report root."""
        for report_path in _REPORT_PATHS.values():
            Path(report_path).unlink(missing_ok=True)
        # Derive the root from the patched long-report path so tests and
        # alternate output directories remain isolated from the repository.
        report_root = Path(_REPORT_PATHS["long"]).parent
        for name in _PHASE5_DERIVED_REPORT_NAMES:
            (report_root / name).unlink(missing_ok=True)

    def _evaluate_strategy(
        self,
        test_df: pd.DataFrame,
        strategy: dict,
        direction: str,
        history_df: pd.DataFrame | None = None,
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

        if "mtf_candidate" in strategy:
            from gpu_fuzzy_trader.mtf.candidate import HierarchicalStrategyCandidate

            try:
                candidate = HierarchicalStrategyCandidate.from_dict(
                    strategy["mtf_candidate"]
                )
                rule_masks, composition_stats, _audit = evaluate_candidate_rule_masks(
                    candidate,
                    test_df,
                    history_df=history_df,
                )
                engine = CPUBacktestEngine(
                    test_df,
                    feature_modes={},
                    direction=direction,
                )
                metrics, trade_log = engine.simulate_signal_masks(
                    rule_masks,
                    candidate.lwc_rules,
                    return_logs=True,
                )
                if not isinstance(metrics, dict):
                    raise TypeError(
                        "simulate_signal_masks returned a non-dict metrics object"
                    )
                metrics.setdefault("evaluation_status", "ok")
                if int(metrics.get("executed_trades", 0) or 0) == 0:
                    metrics["account_ruined"] = False
                    metrics["total_return_pct"] = 0.0
                metrics["mtf_composition"] = {
                    "frozen": True,
                    "retention_diagnostics": composition_stats.get(
                        "retention_diagnostics", {}
                    ),
                    "manifest_hash": strategy.get("provenance", {}).get(
                        "mtf_manifest_hash", ""
                    ),
                }
                return metrics, self._build_per_symbol_rows(metrics, direction, trade_log), trade_log
            except Exception as exc:
                logger.error(
                    "Phase 5 [%s]: frozen MTF evaluation failed: %s",
                    direction,
                    exc,
                    exc_info=True,
                )
                return (
                    self._evaluation_error_metrics(
                        direction, f"{type(exc).__name__}: {exc}"
                    ),
                    [],
                    pd.DataFrame(),
                )

        if bool(getattr(_cfg, "REQUIRE_CONTEXT_IN_STRATEGY", False)):
            ctx_columns = set(getattr(_cfg, "CONTEXT_COLUMNS", ()))
            if any(c in test_df.columns for c in ctx_columns):
                mandatory = _cfg.mandatory_context_conditions(direction)
                for rule in rule_set:
                    present = [str(c).strip() for c in rule.get("conditions", [])]
                    missing = [m for m in mandatory if m not in present]
                    if missing:
                        raise ValueError(
                            f"Phase 5 [{direction}]: strategy rule is missing "
                            f"mandatory context conditions {missing} on an "
                            f"enriched tape. Refusing to evaluate an incomplete "
                            "strategy contract."
                        )

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
            if not isinstance(metrics, dict):
                raise TypeError(
                    "simulate_rule_set returned a non-dict metrics object"
                )
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Phase 5 [%s]: simulate_rule_set failed for this split; "
                "marking evaluation as error: %s",
                direction,
                error_text,
                exc_info=True,
            )
            metrics = self._evaluation_error_metrics(direction, error_text)
            trade_log = pd.DataFrame()

        metrics.setdefault("evaluation_status", "ok")

        # Requirement 11.4: zero-trade case — do NOT report account ruin
        # unless equity actually reached zero.
        if int(metrics.get("executed_trades", 0) or 0) == 0:
            metrics["account_ruined"] = False
            metrics["total_return_pct"] = 0.0

        # Requirement 11.5: per-symbol breakdowns
        per_symbol_rows = self._build_per_symbol_rows(
            metrics, direction, trade_log
        )

        return metrics, per_symbol_rows, trade_log

    @staticmethod
    def _evaluation_error_metrics(direction: str, error_text: str) -> dict:
        """Return an explicit, non-success result for a failed split."""
        return {
            "direction": direction,
            "evaluation_status": "error",
            "evaluation_error": error_text,
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
            "per_symbol_metrics_available": False,
        }

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

    def _save_report(
        self,
        metrics: dict,
        direction: str,
        *,
        split: str = "test",
    ) -> None:
        """Save a split report, marking consumed test data as diagnostic-only."""
        if split == "forward":
            report_key = (
                "forward_joint" if direction == "joint_portfolio"
                else f"forward_{direction}"
            )
            fallback_name = f"forward_{direction}_report.json"
        else:
            report_key = (
                "joint" if direction == "joint_portfolio" else direction
            )
            fallback_name = f"test_{direction}_report.json"
        report_path = _REPORT_PATHS.get(
            report_key,
            os.path.join(_cfg.REPORTS_DIR, fallback_name),
        )
        Path(report_path).parent.mkdir(parents=True, exist_ok=True)

        # Build a clean, serialisable report dict
        evaluation_status = str(metrics.get("evaluation_status", "ok"))
        report = {
            "run_id": self.run_id,
            "direction": direction,
            "split": split,
            "acceptance_status": (
                "diagnostic_only" if split == "test" else "forward_candidate"
            ),
            "evaluation_status": evaluation_status,
            "total_return_pct": metrics.get("total_return_pct", 0.0),
            "max_drawdown_pct": metrics.get("max_drawdown_pct", 0.0),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "executed_trades": metrics.get("executed_trades", 0),
            "sortino_ratio": metrics.get("sortino_ratio", 0.0),
            "expectancy_pct_per_trade": metrics.get(
                "expectancy_pct_per_trade", 0.0,
            ),
            "expectancy_lcb_pct_per_trade": metrics.get(
                "expectancy_lcb_pct_per_trade", 0.0,
            ),
            "expected_shortfall_pct": metrics.get(
                "expected_shortfall_pct", 0.0,
            ),
            "turnover_multiple": metrics.get("turnover_multiple", 0.0),
            "max_simultaneous_positions": metrics.get(
                "max_simultaneous_positions", 0,
            ),
            "max_total_open_exposure": metrics.get(
                "max_total_open_exposure", 0.0,
            ),
            "total_fees": metrics.get("total_fees", 0.0),
            "fee_pct": metrics.get("fee_pct", getattr(_cfg, "FEE_PCT", 0.0)),
            "spread_bps": metrics.get(
                "spread_bps", getattr(_cfg, "SPREAD_BPS", 0.0),
            ),
            "slippage_bps": metrics.get(
                "slippage_bps", getattr(_cfg, "SLIPPAGE_BPS", 0.0),
            ),
            "effective_fee_rate": metrics.get(
                "effective_fee_rate", 0.0,
            ),
            "account_status": (
                "error" if evaluation_status == "error" else
                "ruined" if metrics.get("account_ruined", False) else
                "survived"
            ),
            "final_equity": metrics.get("final_equity", _cfg.INITIAL_CAPITAL),
            "per_symbol_metrics": metrics.get("per_symbol_metrics", {}),
        }
        if metrics.get("trade_uncertainty") is not None:
            report["trade_uncertainty"] = metrics["trade_uncertainty"]
        if metrics.get("evaluation_error"):
            report["evaluation_error"] = str(metrics["evaluation_error"])

        with open(report_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, cls=_Phase5JSONEncoder)

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

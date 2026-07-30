"""
reporter.py — Reporting and visualization for the GPU-Fuzzy Trading Pipeline.

Generates:
  - Phase 2 generation metrics plots (objectives vs. generation)
    - Phase 2 PnL plots (mean/best Pareto Sortino vs. generation)
  - Equity curve plots (train, validation, test)
  - Per-symbol performance CSVs
  - RB risk-grid and validation diagnostics

All plots are saved as PNG files (not displayed).
"""

from __future__ import annotations
from scipy.stats import spearmanr
from gpu_fuzzy_trader import config as _cfg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import logging
import os
from typing import Any, List

import matplotlib
import matplotlib.dates as mdates
# Non-interactive backend — must be set before importing pyplot
matplotlib.use("Agg")


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level output paths (overridable in tests)
# ---------------------------------------------------------------------------
_REPORTS_DIR = _cfg.REPORTS_DIR


def _bucket_series_by_mode(series: pd.Series, mode: str) -> pd.Series:
    """
    Map a numeric feature column into the discrete fuzzy buckets that the
    backtest engine uses (mirrors ``_apply_dynamic_rule`` thresholds in
    ``cpu_engine.py``). Used by the feature-stratified report so we can
    measure per-bucket trade performance without relying on pre-discretized
    string columns (raw features are stored as floats).

    Returns a Series of dtype object containing string bucket names. Values
    that don't match any bucket map to ``None``. If *series* already holds
    string bucket labels (legacy callers and tests pre-discretize their
    features), return a copy unchanged.
    """
    s = series

    # Pass-through: column already holds string bucket labels.
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        sample = s.dropna()
        if not sample.empty and isinstance(sample.iloc[0], str):
            return s.astype(object)

    if mode in ("binary",):
        out = pd.Series(np.full(len(s), None, dtype=object), index=s.index)
        out[s == 0] = "Inactive (0)"
        out[s == 1] = "Active (1)"
        return out

    if mode in ("ternary",):
        out = pd.Series(np.full(len(s), None, dtype=object), index=s.index)
        out[s == -1] = "Negative (-1)"
        out[s == 0] = "Neutral (0)"
        out[s == 1] = "Positive (1)"
        return out

    if mode in ("sparse_signed",):
        bins = [-np.inf, -0.25, -1e-5, 1e-5, 0.25, np.inf]
        labels = [
            "Strong Negative", "Weak Negative", "Exactly Zero",
            "Weak Positive", "Strong Positive",
        ]
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(object)

    if mode in ("positive", "sparse_positive"):
        bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
        labels = ["Very Low", "Low", "Medium", "High", "Very High"]
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(object)

    if mode == "signed":
        bins = [-np.inf, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, np.inf]
        labels = [
            "Extreme Bearish", "Strong Bearish", "Bearish", "Weak Bearish",
            "Neutral Negative", "Neutral Positive",
            "Weak Bullish", "Bullish", "Strong Bullish", "Extreme Bullish",
        ]
        return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(object)

    # Unknown mode — fall back to "positive" buckets so we still emit something useful.
    bins = [-np.inf, 0.2, 0.4, 0.6, 0.8, np.inf]
    labels = ["Very Low", "Low", "Medium", "High", "Very High"]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True).astype(object)


class Reporter:
    """
    Generates visual and tabular reports for each pipeline phase.

    All output files are written to ``outputs/reports/`` by default.
    The directory is created automatically if it does not exist.
    """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_dir(path: str) -> None:
        """Create parent directories for *path* if they do not exist."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    @staticmethod
    def _compute_mdd(equity_series: pd.Series) -> float:
        """Compute maximum drawdown percentage from an equity series.

        Parameters
        ----------
        equity_series:
            Series of equity values (e.g. ``Equity_After`` column).

        Returns
        -------
        float
            Maximum percentage drop from peak to trough, or ``0.0`` if the
            series is empty.
        """
        if equity_series.empty:
            return 0.0
        peak = equity_series.cummax()
        drawdown = (peak - equity_series) / peak.replace(0, np.nan) * 100
        return float(drawdown.max(skipna=True)) if not drawdown.empty else 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plot_per_rule_breakdown(
        self,
        rule_set: list,
        trade_logs_by_split: dict,
        direction: str,
        output_dir: str | None = None,
    ) -> str:
        """Plot a per-rule performance breakdown across train/validation/test splits.

        Produces a 2×2 subplot figure with grouped bar charts showing four
        metrics for each rule: total PnL, win rate, number of trades, and
        maximum drawdown percentage.

        Parameters
        ----------
        rule_set:
            List of rule dicts, each with ``"conditions"``, ``"tp"``,
            ``"sl"``, and ``"capital_pct"`` keys.
        trade_logs_by_split:
            Dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping
            to ``pd.DataFrame | None``.
        direction:
            ``"long"`` or ``"short"``; raises ``ValueError`` otherwise.
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        str
            Absolute path to the saved PNG file.

        Raises
        ------
        ValueError
            If ``direction`` is not ``"long"`` or ``"short"``.
        """
        # Step 1 — Resolve output directory
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(
            reports_dir, f"per_rule_breakdown_{direction}.png"
        )

        # Step 2 — Validate direction (before creating any file)
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        # Step 3 — Ensure output directory exists
        self._ensure_dir(out_path)

        # Step 4 — Compute per-rule metrics for each split
        _SPLITS = ("train", "validation", "test")
        _COLORS = {
            "train": "#4C72B0",
            "validation": "#DD8452",
            "test": "#55A868",
        }

        n_rules = len(rule_set)

        # Metrics containers: dict[split] -> list of values (one per rule)
        metrics: dict[str, dict[str, list]] = {
            split: {
                "total_pnl": [],
                "win_rate": [],
                "num_trades": [],
                "mdd_pct": [],
            }
            for split in _SPLITS
        }

        for split in _SPLITS:
            log = trade_logs_by_split.get(
                split) if trade_logs_by_split else None

            # Treat None or empty DataFrame as zero-trade log
            if log is None or (isinstance(log, pd.DataFrame) and log.empty):
                for _ in range(n_rules):
                    metrics[split]["total_pnl"].append(0.0)
                    metrics[split]["win_rate"].append(0.0)
                    metrics[split]["num_trades"].append(0)
                    metrics[split]["mdd_pct"].append(0.0)
                continue

            # Rule_Index in the trade log is 1-based per the data model
            for rule_idx in range(1, n_rules + 1):
                if "Rule_Index" in log.columns:
                    filtered = log[log["Rule_Index"] == rule_idx]
                else:
                    filtered = pd.DataFrame()

                if filtered.empty:
                    metrics[split]["total_pnl"].append(0.0)
                    metrics[split]["win_rate"].append(0.0)
                    metrics[split]["num_trades"].append(0)
                    metrics[split]["mdd_pct"].append(0.0)
                else:
                    total_pnl = float(filtered["Net_PnL"].sum(
                    )) if "Net_PnL" in filtered.columns else 0.0
                    win_rate = (
                        float((filtered["Net_PnL"] > 0).mean() * 100)
                        if "Net_PnL" in filtered.columns
                        else 0.0
                    )
                    num_trades = len(filtered)
                    mdd_pct = (
                        self._compute_mdd(filtered["Equity_After"])
                        if "Equity_After" in filtered.columns
                        else 0.0
                    )
                    metrics[split]["total_pnl"].append(total_pnl)
                    metrics[split]["win_rate"].append(win_rate)
                    metrics[split]["num_trades"].append(num_trades)
                    metrics[split]["mdd_pct"].append(mdd_pct)

        # Step 5 — Build 2×2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Per-Rule Breakdown — {direction.capitalize()}")

        subplot_specs = [
            (axes[0, 0], "total_pnl", "Total PnL"),
            (axes[0, 1], "win_rate", "Win Rate (%)"),
            (axes[1, 0], "num_trades", "Number of Trades"),
            (axes[1, 1], "mdd_pct", "Max Drawdown (%)"),
        ]

        x = np.arange(n_rules)
        width = 0.25
        x_labels = [f"Rule {i + 1}" for i in range(n_rules)]

        for ax, metric_key, metric_title in subplot_specs:
            ax.bar(
                x - width,
                metrics["train"][metric_key],
                width,
                label="train",
                color=_COLORS["train"],
            )
            ax.bar(
                x,
                metrics["validation"][metric_key],
                width,
                label="validation",
                color=_COLORS["validation"],
            )
            ax.bar(
                x + width,
                metrics["test"][metric_key],
                width,
                label="test",
                color=_COLORS["test"],
            )
            ax.set_title(metric_title)
            ax.set_xticks(x)
            ax.set_xticklabels(x_labels)
            ax.legend()
            ax.grid(True, alpha=0.3, axis="y")

        fig.tight_layout()

        # Step 6 — Save at 100 DPI
        fig.savefig(out_path, dpi=100)

        # Step 7 — Close figure
        plt.close(fig)

        # Step 8 — Log and return absolute path
        abs_path = os.path.abspath(out_path)
        logger.info("Saved per-rule breakdown plot: %s", abs_path)
        return abs_path

    def plot_phase2_metrics(
        self,
        history: List[dict],
        direction: str,
        output_dir: str | None = None,
    ) -> str:
        """
        Plot Phase 2 objectives vs. generation and save to PNG.

        Parameters
        ----------
        history:
            List of dicts, each with keys ``"generation"``, ``"mean_f1"``,
            ``"mean_f2"``, ``"mean_f3"``.
        direction:
            ``"long"`` or ``"short"``.
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        str
            Absolute path to the saved PNG file.
        """
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(reports_dir, f"phase2_{direction}_metrics.png")
        self._ensure_dir(out_path)

        fig, ax = plt.subplots(figsize=(10, 6))

        if not history:
            ax.set_title(f"Phase 2 Metrics — {direction} (no data)")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Objective value")
            fig.tight_layout()
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            logger.warning(
                "plot_phase2_metrics: empty history for direction=%s", direction)
            return out_path

        generations = [entry.get("generation", i)
                       for i, entry in enumerate(history)]
        mean_f1 = [entry.get("mean_f1", 0.0) for entry in history]
        mean_f2 = [entry.get("mean_f2", 0.0) for entry in history]
        mean_f3 = [entry.get("mean_f3", 0.0) for entry in history]
        mean_raw_train = [
            entry.get("mean_raw_train_return_pct", 0.0) for entry in history
        ]
        mean_val = [entry.get("mean_val_return_pct", 0.0) for entry in history]

        ax.plot(generations, mean_f1,
                label="mean_f1 (−Sortino)", color="tab:blue")
        ax.plot(generations, mean_f2,
                label="mean_f2 (drawdown)", color="tab:orange")
        f3_label = "mean_f3 (−total_return)" if _cfg.PHASE2_USE_TOTAL_RETURN_OBJ else "mean_f3 (−win_rate)"
        ax.plot(generations, mean_f3,
                label=f3_label, color="tab:green")
        if any(v != 0.0 for v in mean_raw_train):
            ax.plot(
                generations,
                mean_raw_train,
                label="mean raw train return %",
                color="tab:red",
                linestyle="--",
            )
        if any(v != 0.0 for v in mean_val):
            ax.plot(
                generations,
                mean_val,
                label="mean val return %",
                color="tab:purple",
                linestyle="--",
            )

        ax.set_title(f"Phase 2 Objectives vs. Generation — {direction}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Objective value")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

        logger.info("Saved Phase 2 metrics plot: %s", out_path)
        return out_path

    def plot_phase2_pnl(
        self,
        history: List[dict],
        direction: str,
        output_dir: str | None = None,
    ) -> str:
        """
        Plot Phase 2 mean/best Pareto Sortino Ratio vs. generation and save to PNG.

        Parameters
        ----------
        history:
            List of dicts, each with keys ``"generation"``,
            ``"mean_sortino_ratio"``, and ``"best_sortino_ratio"``.
        direction:
            ``"long"`` or ``"short"``.
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        str
            Absolute path to the saved PNG file.
        """
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(reports_dir, f"phase2_{direction}_pnl.png")
        self._ensure_dir(out_path)

        fig, ax = plt.subplots(figsize=(10, 6))

        if not history:
            ax.set_title(
                f"Phase 2 Sortino per Generation — {direction} (no data)")
            ax.set_xlabel("Generation")
            ax.set_ylabel("Sortino Ratio")
            fig.tight_layout()
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            logger.warning(
                "plot_phase2_pnl: empty history for direction=%s", direction)
            return out_path

        generations = [entry.get("generation", i)
                       for i, entry in enumerate(history)]
        mean_return = [
            entry.get("mean_sortino_ratio", entry.get(
                "mean_total_return_pct", 0.0))
            for entry in history
        ]
        best_return = [
            entry.get("best_sortino_ratio", entry.get(
                "best_total_return_pct", 0.0))
            for entry in history
        ]
        mean_val = [entry.get("mean_val_return_pct", 0.0) for entry in history]

        ax.plot(
            generations,
            mean_return,
            label="Mean Pareto Sortino Ratio",
            color="tab:blue",
        )
        ax.plot(
            generations,
            best_return,
            label="Best Pareto Sortino Ratio",
            color="tab:orange",
        )
        if any(v != 0.0 for v in mean_val):
            ax.plot(
                generations,
                mean_val,
                label="Mean Pareto val return %",
                color="tab:purple",
                linestyle="--",
            )
        ax.axhline(y=0.0, color="gray", linestyle="--", linewidth=0.8)

        ax.set_title(f"Phase 2 Sortino per Generation — {direction}")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Sortino Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

        logger.info("Saved Phase 2 PnL plot: %s", out_path)
        return out_path

    def plot_equity_curve(
        self,
        trade_log: pd.DataFrame,
        split: str,
        direction: str,
        output_dir: str | None = None,
    ) -> str:
        """
        Plot equity curve from a trade log and save to PNG.

        Parameters
        ----------
        trade_log:
            DataFrame with at least an ``"Equity_After"`` column.
        split:
            One of ``"train"``, ``"validation"``, or ``"test"``.
        direction:
            ``"long"`` or ``"short"``.
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        str
            Absolute path to the saved PNG file.
        """
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(reports_dir, f"{split}_{direction}_equity.png")
        self._ensure_dir(out_path)

        fig, ax = plt.subplots(figsize=(12, 5))

        if trade_log is None or trade_log.empty or "Equity_After" not in trade_log.columns:
            ax.set_title(f"Equity Curve — {split} / {direction} (no trades)")
            ax.set_xlabel("Trade #")
            ax.set_ylabel("Equity")
            fig.tight_layout()
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            logger.warning(
                "plot_equity_curve: empty trade_log for split=%s direction=%s",
                split,
                direction,
            )
            return out_path

        equity = trade_log["Equity_After"].values

        # Check if Entry_Time is available for date-based x-axis (Task 10.3).
        has_entry_time = (
            "Entry_Time" in trade_log.columns
            and trade_log["Entry_Time"].notna().any()
        )

        if has_entry_time:
            # Sort by Entry_Time so the equity curve is chronological.
            plot_df = trade_log.sort_values("Entry_Time").reset_index(drop=True)
            equity = plot_df["Equity_After"].values
            x_dates = plot_df["Entry_Time"]
            ax.plot(x_dates, equity, color="tab:blue", linewidth=1.2)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            ax.set_xlabel("Date")
            fig.autofmt_xdate()
        else:
            trade_indices = range(len(equity))
            ax.plot(trade_indices, equity, color="tab:blue", linewidth=1.2)
            ax.set_xlabel("Trade #")

        ax.axhline(
            y=_cfg.INITIAL_CAPITAL,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label=f"Initial capital ({_cfg.INITIAL_CAPITAL:.0f})",
        )

        ax.set_title(f"Equity Curve — {split} / {direction}")
        ax.set_ylabel("Equity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

        logger.info("Saved equity curve: %s", out_path)
        return out_path

    def write_per_symbol_csv(
        self,
        metrics: dict,
        split: str,
        direction: str | None = None,
        output_dir: str | None = None,
    ) -> str:
        """
        Write per-symbol performance metrics to a CSV file.

        Parameters
        ----------
        metrics:
            Dict containing a ``"per_symbol_metrics"`` key whose value is a
            dict mapping symbol names to metric dicts (``trade_count``,
            ``win_rate``, ``net_pnl``).
        split:
            One of ``"train"``, ``"validation"``, or ``"test"``.
        direction:
            Optional strategy direction (``"long"`` or ``"short"``). When
            provided, the filename includes direction to avoid long/short
            overwrites for the same split.
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        str
            Absolute path to the saved CSV file.
        """
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        filename = (
            f"{split}_{direction}_per_symbol_performance.csv"
            if direction in ("long", "short")
            else f"{split}_per_symbol_performance.csv"
        )
        out_path = os.path.join(reports_dir, filename)
        self._ensure_dir(out_path)

        per_symbol = metrics.get("per_symbol_metrics", {})

        rows = []
        for symbol, sym_metrics in per_symbol.items():
            rows.append(
                {
                    "symbol": symbol,
                    "trade_count": sym_metrics.get("trade_count", 0),
                    "win_rate": sym_metrics.get("win_rate", 0.0),
                    "net_pnl": sym_metrics.get("net_pnl", 0.0),
                }
            )

        df = pd.DataFrame(
            rows,
            columns=["symbol", "trade_count", "win_rate", "net_pnl"],
        )
        df.to_csv(out_path, index=False)

        logger.info("Saved per-symbol CSV: %s", out_path)
        return out_path

    def write_strategy_evaluation_table(
        self,
        metrics_by_split: dict,
        trade_logs_by_split: dict,
        rule_set: list,
        direction: str,
        output_dir: str | None = None,
    ) -> str:
        """
        Write a strategy evaluation table (one row per split) to a CSV file.

        Parameters
        ----------
        metrics_by_split:
            Dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping to
            metrics dicts from ``CPUBacktestEngine.simulate_rule_set`` (or ``None``).
        trade_logs_by_split:
            Dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping to
            ``pd.DataFrame | None``.
        rule_set:
            List of rule dicts, each with a ``"conditions"`` key.
        direction:
            ``"long"`` or ``"short"``; raises ``ValueError`` otherwise.
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        str
            Absolute path to the saved CSV file.
        """
        # Step 1: Resolve output directory
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(
            reports_dir, f"strategy_evaluation_{direction}.csv"
        )

        # Step 2: Validate direction
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        # Step 3: Ensure directory exists (only when output_dir not provided)
        if output_dir is None:
            self._ensure_dir(out_path)

        # Step 4: Compute rule-set-level counts (constant across all splits)
        num_rules = len(rule_set)
        num_conditions = sum(len(r.get("conditions", [])) for r in rule_set)

        # Step 5: Build one row per split
        rows = []
        for split in ("train", "validation", "test"):
            metrics = (metrics_by_split or {}).get(split) or {}
            log = (trade_logs_by_split or {}).get(split)

            # Source metrics from the metrics dict; default to 0.0 if absent/None
            win_rate = float(metrics.get("win_rate") or 0.0)
            mdd_pct = float(metrics.get("max_drawdown_pct") or 0.0)
            total_return_pct = float(metrics.get("total_return_pct") or 0.0)
            sortino_ratio = float(metrics.get("sortino_ratio") or 0.0)
            profit_factor = float(metrics.get("profit_factor") or 0.0)

            # Compute sharpe_ratio from trade log
            sharpe_ratio = 0.0
            if log is not None and not (
                isinstance(log, pd.DataFrame) and log.empty
            ):
                try:
                    if (
                        isinstance(log, pd.DataFrame)
                        and "Net_PnL" in log.columns
                        and "Equity_Before_Entry" in log.columns
                        and len(log) >= 2
                    ):
                        r = log["Net_PnL"] / log["Equity_Before_Entry"]
                        std_r = r.std(ddof=1)
                        if std_r != 0 and not pd.isna(std_r):
                            sharpe_ratio = float(r.mean() / std_r)
                except Exception:
                    sharpe_ratio = 0.0

            rows.append(
                {
                    "split": split,
                    "win_rate": win_rate,
                    "mdd_pct": mdd_pct,
                    "total_return_pct": total_return_pct,
                    "num_rules": num_rules,
                    "num_conditions": num_conditions,
                    "sortino_ratio": sortino_ratio,
                    "profit_factor": profit_factor,
                    "sharpe_ratio": sharpe_ratio,
                }
            )

        # Step 6: Write CSV
        df = pd.DataFrame(
            rows,
            columns=[
                "split",
                "win_rate",
                "mdd_pct",
                "total_return_pct",
                "num_rules",
                "num_conditions",
                "sortino_ratio",
                "profit_factor",
                "sharpe_ratio",
            ],
        )
        df.to_csv(out_path, index=False)

        # Step 7: Log and return
        logger.info("Saved strategy evaluation table: %s", out_path)
        return os.path.abspath(out_path)

    @staticmethod
    def _spearman(a: pd.Series, b: pd.Series) -> float:
        """Compute Spearman correlation between two series, dropping NaN-paired rows.

        Parameters
        ----------
        a, b:
            Input series of equal length.

        Returns
        -------
        float
            Spearman correlation coefficient, or ``NaN`` if fewer than 2
            non-NaN paired rows remain.
        """
        mask = a.notna() & b.notna()
        if mask.sum() < 2:
            return float("nan")
        result = spearmanr(a[mask].values, b[mask].values)
        stat = getattr(result, "statistic", None) or getattr(
            result, "correlation", float("nan"))
        return float(stat)

    def write_spearman_correlation_report(
        self,
        datasets_by_split: dict,
        selected_features: list,
        direction: str,
        output_dir: str | None = None,
    ) -> str:
        """Write a Spearman correlation report between each feature and forward returns.

        For each selected feature, computes the Spearman correlation with
        ``label_close_288`` independently on each split (train, validation,
        test) after dropping NaN-paired rows.

        Parameters
        ----------
        datasets_by_split:
            Dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping
            to ``pd.DataFrame | None``.
        selected_features:
            List of dicts with at least a ``"name"`` key identifying the
            feature column name.
        direction:
            ``"long"`` or ``"short"``; used only for the output filename.
            Raises ``ValueError`` if not one of these values.
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        str
            Absolute path to the saved CSV file.

        Raises
        ------
        ValueError
            If ``direction`` is not ``"long"`` or ``"short"``.
        """
        # Step 1 — Resolve output directory
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(
            reports_dir, f"spearman_correlation_{direction}.csv"
        )

        # Step 2 — Validate direction (before creating any file)
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        # Step 3 — Ensure output directory exists (only when output_dir not provided)
        if output_dir is None:
            self._ensure_dir(out_path)

        _LABEL_COL = "label_close_288"
        _SPLITS = ("train", "validation", "test")

        # Step 4 — Compute Spearman correlation for each feature × split
        rows = []
        for feat in selected_features:
            feat_name = feat["name"]
            corr_values: dict[str, float] = {}

            for split in _SPLITS:
                dataset = (datasets_by_split or {}).get(split)

                # None or empty dataset → NaN
                if dataset is None or (isinstance(dataset, pd.DataFrame) and dataset.empty):
                    corr_values[split] = float("nan")
                    continue

                # Missing label column → NaN for all features on this split
                if _LABEL_COL not in dataset.columns:
                    corr_values[split] = float("nan")
                    continue

                # Missing feature column → NaN
                if feat_name not in dataset.columns:
                    corr_values[split] = float("nan")
                    continue

                # Compute Spearman (helper handles NaN-dropping and row count check)
                corr_values[split] = self._spearman(
                    dataset[feat_name], dataset[_LABEL_COL]
                )

            rows.append(
                {
                    "feature": feat_name,
                    "train_spearman": corr_values.get("train", float("nan")),
                    "validation_spearman": corr_values.get("validation", float("nan")),
                    "test_spearman": corr_values.get("test", float("nan")),
                }
            )

        # Step 5 — Build DataFrame
        df = pd.DataFrame(
            rows,
            columns=["feature", "train_spearman",
                     "validation_spearman", "test_spearman"],
        )

        # Step 6 — Sort: abs(train_spearman) descending, then feature ascending (stable)
        df["_abs_train"] = df["train_spearman"].abs()
        df = df.sort_values(
            by=["_abs_train", "feature"],
            ascending=[False, True],
            kind="stable",
            na_position="last",
        ).drop(columns=["_abs_train"]).reset_index(drop=True)

        # Step 7 — Write CSV with index=False
        df.to_csv(out_path, index=False)

        # Step 8 — Log and return absolute path
        abs_path = os.path.abspath(out_path)
        logger.info("Saved Spearman correlation report: %s", abs_path)
        return abs_path

    def plot_distribution_and_equity(
        self,
        trade_logs_by_split: dict,
        direction: str,
        output_dir: str | None = None,
    ) -> list[str]:
        """Plot concurrent-positions histogram, time-between-trades histogram,
        and an equity curve annotated with trade entry / exit points for each
        non-empty split.

        Parameters
        ----------
        trade_logs_by_split:
            Dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping
            to ``pd.DataFrame | None``.
        direction:
            ``"long"`` or ``"short"``; raises ``ValueError`` otherwise.
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        list[str]
            Absolute paths to the saved PNG files, one per non-empty split.
            Splits with ``None`` or empty trade logs are skipped.

        Raises
        ------
        ValueError
            If ``direction`` is not ``"long"`` or ``"short"``.
        """
        # Step 1 — Resolve output directory
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR

        # Step 2 — Validate direction (before creating any file)
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        _SPLITS = ("train", "validation", "test")
        result_paths: list[str] = []

        # Step 3 — Iterate over splits
        for split in _SPLITS:
            log = (trade_logs_by_split or {}).get(split)

            # Skip None or empty trade logs
            if log is None or (isinstance(log, pd.DataFrame) and log.empty):
                logger.warning(
                    "plot_distribution_and_equity: empty trade log for "
                    "split=%s direction=%s — skipping",
                    split,
                    direction,
                )
                continue

            out_path = os.path.join(
                reports_dir,
                f"distribution_equity_{split}_{direction}.png",
            )

            # Ensure output directory exists (only when output_dir not provided)
            if output_dir is None:
                self._ensure_dir(out_path)

            # Step 4a — Compute Concurrent_Open_Positions
            if "Release_Index" in log.columns and not log["Release_Index"].empty:
                max_idx = int(log["Release_Index"].max())
                concurrent = [
                    int(
                        ((log["Entry_Index"] <= idx) &
                         (log["Release_Index"] > idx)).sum()
                    )
                    for idx in range(max_idx + 1)
                ]
            else:
                concurrent = []

            # Step 4b — Compute Time_Between_Trades
            if "Entry_Index" in log.columns:
                sorted_log = log.sort_values("Entry_Index", kind="stable")
                diffs = sorted_log["Entry_Index"].diff().dropna()
            else:
                sorted_log = log
                diffs = pd.Series(dtype=float)

            # Step 5 — Build 3-panel figure
            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(
                f"Distribution & Equity — {split.capitalize()} / {direction.capitalize()}"
            )

            # top-left: concurrent positions
            ax_conc = fig.add_subplot(2, 2, 1)
            # top-right: time between trades
            ax_time = fig.add_subplot(2, 2, 2)
            # bottom: equity curve (full width)
            ax_eq = fig.add_subplot(2, 1, 2)

            # Concurrent positions histogram
            ax_conc.hist(concurrent, bins="auto",
                         color="#4C72B0", edgecolor="white")
            ax_conc.set_title("Concurrent Open Positions")
            ax_conc.set_xlabel("Concurrent Positions")
            ax_conc.set_ylabel("Frequency")
            ax_conc.grid(True, alpha=0.3)

            # Time-between-trades histogram
            if len(diffs) > 0:
                ax_time.hist(diffs.values, bins="auto",
                             color="#DD8452", edgecolor="white")
            ax_time.set_title("Time Between Trades (candles)")
            ax_time.set_xlabel("Candles Between Entries")
            ax_time.set_ylabel("Frequency")
            ax_time.grid(True, alpha=0.3)

            # Step 5d — Equity curve: x = trade sequence (1..N), y = Equity_After
            n_trades = len(log)
            seq = np.arange(1, n_trades + 1)
            equity_vals = log["Equity_After"].values

            ax_eq.plot(seq, equity_vals, color="tab:blue",
                       linewidth=1.2, zorder=1)

            if "Equity_Before_Entry" in log.columns:
                entry_vals = log["Equity_Before_Entry"].values
                ax_eq.vlines(
                    seq,
                    entry_vals,
                    equity_vals,
                    color="#A0A0A0",
                    alpha=0.25,
                    linewidth=0.8,
                    zorder=1,
                )
                ax_eq.scatter(
                    seq,
                    entry_vals,
                    marker="o",
                    facecolors="none",
                    edgecolors="#4C72B0",
                    s=28,
                    linewidths=0.9,
                    zorder=3,
                    label="Entry",
                )

            ax_eq.scatter(
                seq,
                equity_vals,
                marker="x",
                color="#C44E52",
                s=30,
                zorder=3,
                label="Exit",
            )
            ax_eq.set_title("Equity Curve")
            ax_eq.set_xlabel("Trade #")
            ax_eq.set_ylabel("Equity")
            ax_eq.grid(True, alpha=0.3)
            ax_eq.legend(loc="best")

            fig.tight_layout()

            # Step 6 — Save at 100 DPI
            fig.savefig(out_path, dpi=100)

            # Step 7 — Close figure
            plt.close(fig)

            # Step 8 — Log and append absolute path
            abs_path = os.path.abspath(out_path)
            logger.info("Saved distribution & equity plot: %s", abs_path)
            result_paths.append(abs_path)

        return result_paths

    def write_feature_stratified_performance(
        self,
        trade_logs_by_split: dict,
        rule_set: list,
        selected_features: list,
        datasets_by_split: dict,
        direction: str,
        output_dir: str | None = None,
    ) -> list[str]:
        """Write feature-stratified performance metrics to CSV files (one per split).

        For each split, for each selected feature, for each unique fuzzy value
        in that feature's column, computes performance metrics for the subset
        of trades whose entry candle has that fuzzy value.

        Feature columns store raw floats; we bucket them into the same
        mode-aware fuzzy bins the backtest engine uses (mirrors
        evaluator_v5.ipynb thresholds).

        Parameters
        ----------
        trade_logs_by_split:
            Dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping
            to ``pd.DataFrame | None``.
        rule_set:
            List of rule dicts (used for context; not directly used in computation).
        selected_features:
            List of dicts with at least a ``"name"`` key identifying the
            feature column name and a ``"mode"`` key (positive/signed/...).
        datasets_by_split:
            Dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping
            to ``pd.DataFrame | None``.
        direction:
            ``"long"`` or ``"short"``; raises ``ValueError`` otherwise.
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        list[str]
            Absolute paths to the saved CSV files, one per split (always 3).

        Raises
        ------
        ValueError
            If ``direction`` is not ``"long"`` or ``"short"``.
        """
        # Step 1 — Resolve output directory
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR

        # Step 2 — Validate direction (before creating any file)
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        _SPLITS = ("train", "validation", "test")
        _COLUMNS = ["feature", "fuzzy_value", "split", "num_trades",
                    "total_return_pct", "win_rate", "sharpe_ratio"]
        result_paths: list[str] = []

        # Step 3 — Iterate over splits
        for split in _SPLITS:
            out_path = os.path.join(
                reports_dir,
                f"feature_stratified_{split}_{direction}.csv",
            )

            # Ensure output directory exists (only when output_dir not provided)
            if output_dir is None:
                self._ensure_dir(out_path)

            dataset = (datasets_by_split or {}).get(split)
            trade_log = (trade_logs_by_split or {}).get(split)

            # If dataset is None/empty → write header-only CSV
            if dataset is None or (isinstance(dataset, pd.DataFrame) and dataset.empty):
                logger.warning(
                    "write_feature_stratified_performance: empty dataset for "
                    "split=%s direction=%s — writing header-only CSV",
                    split,
                    direction,
                )
                pd.DataFrame(columns=_COLUMNS).to_csv(out_path, index=False)
                result_paths.append(os.path.abspath(out_path))
                continue

            # If trade log is None/empty → write header-only CSV
            if trade_log is None or (isinstance(trade_log, pd.DataFrame) and trade_log.empty):
                logger.warning(
                    "write_feature_stratified_performance: empty trade log for "
                    "split=%s direction=%s — writing header-only CSV",
                    split,
                    direction,
                )
                pd.DataFrame(columns=_COLUMNS).to_csv(out_path, index=False)
                result_paths.append(os.path.abspath(out_path))
                continue

            # Step 4 — Build valid_log with out-of-bounds check (vectorised)
            valid_mask = trade_log["Entry_Index"].between(0, len(dataset) - 1)
            if not valid_mask.all():
                n_oob = int((~valid_mask).sum())
                logger.warning(
                    "write_feature_stratified_performance: %d out-of-bounds "
                    "Entry_Index value(s) in split=%s — skipping those trades",
                    n_oob,
                    split,
                )
            valid_log = trade_log[valid_mask].copy()

            # Step 5 — Iterate over features
            rows: list[dict] = []
            for feat in selected_features:
                feat_name = feat["name"]
                feat_mode = feat.get("mode", "positive")

                # Skip feature if column absent from dataset
                if feat_name not in dataset.columns:
                    logger.warning(
                        "write_feature_stratified_performance: feature column "
                        "'%s' absent from dataset for split=%s — skipping",
                        feat_name,
                        split,
                    )
                    continue

                # Bucket feature values by mode (matches engine semantics)
                series = dataset[feat_name]
                bucketed = _bucket_series_by_mode(series, feat_mode)
                fuzzy_values = [v for v in bucketed.dropna().unique()]

                if not fuzzy_values:
                    continue

                # Attach feature value to each valid trade (vectorised lookup)
                valid_log["_feat_val"] = bucketed.iloc[
                    valid_log["Entry_Index"].values
                ].values

                # Step 6 — Compute metrics per fuzzy_value stratum
                for fuzzy_value in fuzzy_values:
                    stratum = valid_log[valid_log["_feat_val"] == fuzzy_value]
                    num_trades = len(stratum)

                    if num_trades == 0:
                        # Zero-trade stratum: include row with all metrics = 0.0
                        rows.append({
                            "feature": feat_name,
                            "fuzzy_value": fuzzy_value,
                            "split": split,
                            "num_trades": 0,
                            "total_return_pct": 0.0,
                            "win_rate": 0.0,
                            "sharpe_ratio": 0.0,
                        })
                        continue

                    # total_return_pct
                    if _cfg.INITIAL_CAPITAL != 0:
                        total_return_pct = float(
                            stratum["Net_PnL"].sum() /
                            _cfg.INITIAL_CAPITAL * 100
                        )
                    else:
                        total_return_pct = 0.0

                    # win_rate
                    win_rate = float(
                        (stratum["Net_PnL"] > 0).sum()) / num_trades

                    # sharpe_ratio: mean(r) / std(r, ddof=1) where r = Net_PnL / Equity_Before_Entry
                    sharpe_ratio = 0.0
                    if num_trades >= 2:
                        try:
                            r = stratum["Net_PnL"] / \
                                stratum["Equity_Before_Entry"]
                            std_r = r.std(ddof=1)
                            if std_r != 0 and not pd.isna(std_r):
                                sharpe_ratio = float(r.mean() / std_r)
                        except Exception:
                            sharpe_ratio = 0.0

                    rows.append({
                        "feature": feat_name,
                        "fuzzy_value": fuzzy_value,
                        "split": split,
                        "num_trades": num_trades,
                        "total_return_pct": total_return_pct,
                        "win_rate": win_rate,
                        "sharpe_ratio": sharpe_ratio,
                    })

            # Step 7 — Write CSV
            df = pd.DataFrame(rows, columns=_COLUMNS)
            df.to_csv(out_path, index=False)

            # Step 8 — Log and append absolute path
            abs_path = os.path.abspath(out_path)
            logger.info(
                "Saved feature-stratified performance CSV: %s", abs_path
            )
            result_paths.append(abs_path)

        return result_paths

    def write_generalization_diagnostics(
        self,
        metrics_by_split: dict,
        selected_features: list,
        datasets_by_split: dict,
        direction: str,
        output_dir: str | None = None,
    ) -> str:
        """Write compact train/validation/test generalization diagnostics to JSON.

        The report is designed for rapid failure-mode triage and includes:
          - split-level return/profit-factor/trade summaries
          - return decay and sign-flip checks across splits
          - per-split symbol concentration (HHI + top symbol share)
          - feature-bucket concentration for selected features
        """
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(
            reports_dir, f"generalization_diagnostics_{direction}.json"
        )
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )
        if output_dir is None:
            self._ensure_dir(out_path)

        split_rows: dict[str, dict[str, Any]] = {}
        for split in ("train", "validation", "test"):
            m = (metrics_by_split or {}).get(split) or {}
            split_rows[split] = {
                "total_return_pct": float(m.get("total_return_pct", 0.0)),
                "profit_factor": float(m.get("profit_factor", 0.0)),
                "win_rate": float(m.get("win_rate", 0.0)),
                "max_drawdown_pct": float(m.get("max_drawdown_pct", 0.0)),
                "executed_trades": int(m.get("executed_trades", 0)),
            }

        train_ret = split_rows["train"]["total_return_pct"]
        val_ret = split_rows["validation"]["total_return_pct"]
        test_ret = split_rows["test"]["total_return_pct"]
        split_shift = {
            "train_to_validation_delta_pct": float(val_ret - train_ret),
            "validation_to_test_delta_pct": float(test_ret - val_ret),
            "train_to_test_delta_pct": float(test_ret - train_ret),
            "train_to_test_sign_flip": bool((train_ret >= 0) != (test_ret >= 0)),
        }

        symbol_concentration: dict[str, dict[str, float | str]] = {}
        for split in ("train", "validation", "test"):
            m = (metrics_by_split or {}).get(split) or {}
            per_sym = m.get("per_symbol_metrics", {}) or {}
            pnls = []
            top_symbol = ""
            top_abs = 0.0
            for sym, v in per_sym.items():
                if not isinstance(v, dict):
                    continue
                val = float(v.get("net_pnl", 0.0))
                abs_val = abs(val)
                pnls.append(abs_val)
                if abs_val > top_abs:
                    top_abs = abs_val
                    top_symbol = str(sym)
            total_abs = float(np.sum(pnls)) if pnls else 0.0
            if total_abs <= 0.0:
                hhi = 0.0
                top_share = 0.0
            else:
                shares = np.asarray(pnls, dtype=np.float64) / total_abs
                hhi = float(np.sum(shares * shares))
                top_share = float(np.max(shares))
            symbol_concentration[split] = {
                "hhi_abs_pnl": hhi,
                "top_symbol_share_abs_pnl": top_share,
                "top_symbol": top_symbol,
            }

        feature_concentration: dict[str, dict[str, dict[str, float | str | int]]] = {}
        for split in ("train", "validation", "test"):
            ds = (datasets_by_split or {}).get(split)
            if ds is None or ds.empty:
                feature_concentration[split] = {}
                continue
            split_fc: dict[str, dict[str, float | str | int]] = {}
            for feat in selected_features:
                feat_name = feat.get("name")
                feat_mode = feat.get("mode", "positive")
                if not feat_name or feat_name not in ds.columns:
                    continue
                bucketed = _bucket_series_by_mode(ds[feat_name], feat_mode)
                counts = bucketed.value_counts(dropna=True)
                if counts.empty:
                    continue
                total = int(counts.sum())
                top_bucket = str(counts.index[0])
                top_count = int(counts.iloc[0])
                split_fc[str(feat_name)] = {
                    "top_bucket": top_bucket,
                    "top_bucket_share": float(top_count / max(total, 1)),
                    "unique_buckets": int(len(counts)),
                }
            feature_concentration[split] = split_fc

        payload: dict[str, Any] = {
            "direction": direction,
            "split_metrics": split_rows,
            "split_shift": split_shift,
            "symbol_concentration": symbol_concentration,
            "feature_bucket_concentration": feature_concentration,
        }
        with open(out_path, "w", encoding="utf-8") as fh:
            import json
            json.dump(payload, fh, indent=2)
        logger.info("Saved generalization diagnostics: %s", out_path)
        return os.path.abspath(out_path)


from __future__ import annotations
from scipy.stats import spearmanr
from gpu_fuzzy_trader import config as _cfg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import logging
import os
from typing import List

import matplotlib
matplotlib.use("Agg")


logger = logging.getLogger(__name__)

_REPORTS_DIR = _cfg.REPORTS_DIR


class Reporter:
    """
    Generates visual and tabular reports for each pipeline phase.

    All output files are written to ``outputs/reports/`` by default.
    The directory is created automatically if it does not exist.
    """


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


    def plot_per_rule_breakdown(
        self,
        rule_set: list,
        trade_logs_by_split: dict,
        direction: str,
        output_dir: str | None = None,
    ) -> str:
        """Plot a per-rule performance breakdown across evaluation splits.

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
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(
            reports_dir, f"per_rule_breakdown_{direction}.png"
        )

        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        self._ensure_dir(out_path)

        _SPLITS = ("train", "validation", "test")
        _COLORS = {
            "train": "#4C72B0",
            "validation": "#DD8452",
            "test": "#55A868",
        }

        n_rules = len(rule_set)

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

            if log is None or (isinstance(log, pd.DataFrame) and log.empty):
                for _ in range(n_rules):
                    metrics[split]["total_pnl"].append(0.0)
                    metrics[split]["win_rate"].append(0.0)
                    metrics[split]["num_trades"].append(0)
                    metrics[split]["mdd_pct"].append(0.0)
                continue

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

        fig.savefig(out_path, dpi=100)

        plt.close(fig)

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

        ax.plot(generations, mean_f1,
                label="mean_f1 (−Sortino)", color="tab:blue")
        ax.plot(generations, mean_f2,
                label="mean_f2 (drawdown)", color="tab:orange")
        ax.plot(generations, mean_f3,
                label="mean_f3 (−win_rate)", color="tab:green")

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
        trade_indices = range(len(equity))

        ax.plot(trade_indices, equity, color="tab:blue", linewidth=1.2)
        ax.axhline(
            y=_cfg.INITIAL_CAPITAL,
            color="gray",
            linestyle="--",
            linewidth=0.8,
            label=f"Initial capital ({_cfg.INITIAL_CAPITAL:.0f})",
        )

        ax.set_title(f"Equity Curve — {split} / {direction}")
        ax.set_xlabel("Trade #")
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
        output_dir:
            Override the output directory (used in tests).

        Returns
        -------
        str
            Absolute path to the saved CSV file.
        """
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(
            reports_dir, f"{split}_per_symbol_performance.csv")
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
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(
            reports_dir, f"strategy_evaluation_{direction}.csv"
        )

        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        if output_dir is None:
            self._ensure_dir(out_path)

        num_rules = len(rule_set)
        num_conditions = sum(len(r.get("conditions", [])) for r in rule_set)

        rows = []
        for split in ("train", "validation", "test"):
            metrics = (metrics_by_split or {}).get(split) or {}
            log = (trade_logs_by_split or {}).get(split)

            win_rate = float(metrics.get("win_rate") or 0.0)
            mdd_pct = float(metrics.get("max_drawdown_pct") or 0.0)
            total_return_pct = float(metrics.get("total_return_pct") or 0.0)
            sortino_ratio = float(metrics.get("sortino_ratio") or 0.0)
            profit_factor = float(metrics.get("profit_factor") or 0.0)

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

        logger.info("Saved strategy evaluation table: %s", out_path)
        return os.path.abspath(out_path)

    def plot_rl_curve(
        self,
        validation_returns: List[float],
        elbow_idx: int,
        direction: str,
        output_dir: str | None = None,
    ) -> str:
        """
        Plot RL training curve (validation return vs. checkpoint) with the
        elbow point marked, and save to PNG.

        Parameters
        ----------
        validation_returns:
            List of validation returns recorded at each evaluation checkpoint.
        elbow_idx:
            Index into *validation_returns* identifying the elbow point.
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
        out_path = os.path.join(
            reports_dir, f"phase4_{direction}_rl_curve.png")
        self._ensure_dir(out_path)

        fig, ax = plt.subplots(figsize=(10, 5))

        if not validation_returns:
            ax.set_title(f"RL Training Curve — {direction} (no data)")
            ax.set_xlabel("Checkpoint")
            ax.set_ylabel("Validation Return (%)")
            fig.tight_layout()
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            logger.warning(
                "plot_rl_curve: empty validation_returns for direction=%s", direction
            )
            return out_path

        checkpoints = list(range(len(validation_returns)))

        ax.plot(
            checkpoints,
            validation_returns,
            color="tab:blue",
            linewidth=1.2,
            label="Validation return",
        )

        clamped_idx = max(0, min(elbow_idx, len(validation_returns) - 1))
        ax.axvline(
            x=clamped_idx,
            color="tab:red",
            linestyle="--",
            linewidth=1.0,
            label=f"Elbow point (checkpoint {clamped_idx})",
        )
        ax.scatter(
            [clamped_idx],
            [validation_returns[clamped_idx]],
            color="tab:red",
            zorder=5,
            s=60,
        )

        ax.set_title(f"RL Training Curve — {direction}")
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Validation Return (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(out_path, dpi=100)
        plt.close(fig)

        logger.info("Saved RL curve: %s", out_path)
        return out_path

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
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR
        out_path = os.path.join(
            reports_dir, f"spearman_correlation_{direction}.csv"
        )

        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        if output_dir is None:
            self._ensure_dir(out_path)

        _LABEL_COL = "label_close_288"
        _SPLITS = ("train", "validation", "test")

        rows = []
        for feat in selected_features:
            feat_name = feat["name"]
            corr_values: dict[str, float] = {}

            for split in _SPLITS:
                dataset = (datasets_by_split or {}).get(split)

                if dataset is None or (isinstance(dataset, pd.DataFrame) and dataset.empty):
                    corr_values[split] = float("nan")
                    continue

                if _LABEL_COL not in dataset.columns:
                    corr_values[split] = float("nan")
                    continue

                if feat_name not in dataset.columns:
                    corr_values[split] = float("nan")
                    continue

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

        df = pd.DataFrame(
            rows,
            columns=["feature", "train_spearman",
                     "validation_spearman", "test_spearman"],
        )

        df["_abs_train"] = df["train_spearman"].abs()
        df = df.sort_values(
            by=["_abs_train", "feature"],
            ascending=[False, True],
            kind="stable",
            na_position="last",
        ).drop(columns=["_abs_train"]).reset_index(drop=True)

        df.to_csv(out_path, index=False)

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
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR

        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        _SPLITS = ("train", "validation", "test")
        result_paths: list[str] = []

        for split in _SPLITS:
            log = (trade_logs_by_split or {}).get(split)

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

            if output_dir is None:
                self._ensure_dir(out_path)

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

            if "Entry_Index" in log.columns:
                sorted_log = log.sort_values("Entry_Index", kind="stable")
                diffs = sorted_log["Entry_Index"].diff().dropna()
            else:
                sorted_log = log
                diffs = pd.Series(dtype=float)

            fig = plt.figure(figsize=(14, 10))
            fig.suptitle(
                f"Distribution & Equity — {split.capitalize()} / {direction.capitalize()}"
            )

            ax_conc = fig.add_subplot(2, 2, 1)
            ax_time = fig.add_subplot(2, 2, 2)
            ax_eq = fig.add_subplot(2, 1, 2)

            ax_conc.hist(concurrent, bins="auto",
                         color="#4C72B0", edgecolor="white")
            ax_conc.set_title("Concurrent Open Positions")
            ax_conc.set_xlabel("Concurrent Positions")
            ax_conc.set_ylabel("Frequency")
            ax_conc.grid(True, alpha=0.3)

            if len(diffs) > 0:
                ax_time.hist(diffs.values, bins="auto",
                             color="#DD8452", edgecolor="white")
            ax_time.set_title("Time Between Trades (candles)")
            ax_time.set_xlabel("Candles Between Entries")
            ax_time.set_ylabel("Frequency")
            ax_time.grid(True, alpha=0.3)

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

            fig.savefig(out_path, dpi=100)

            plt.close(fig)

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

        Parameters
        ----------
        trade_logs_by_split:
            Dict with keys ``"train"``, ``"validation"``, ``"test"`` mapping
            to ``pd.DataFrame | None``.
        rule_set:
            List of rule dicts (used for context; not directly used in computation).
        selected_features:
            List of dicts with at least a ``"name"`` key identifying the
            feature column name.
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
        reports_dir = output_dir if output_dir is not None else _REPORTS_DIR

        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}"
            )

        _SPLITS = ("train", "validation", "test")
        _COLUMNS = ["feature", "fuzzy_value", "split", "num_trades",
                    "total_return_pct", "win_rate", "sharpe_ratio"]
        result_paths: list[str] = []

        for split in _SPLITS:
            out_path = os.path.join(
                reports_dir,
                f"feature_stratified_{split}_{direction}.csv",
            )

            if output_dir is None:
                self._ensure_dir(out_path)

            dataset = (datasets_by_split or {}).get(split)
            trade_log = (trade_logs_by_split or {}).get(split)

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

            rows: list[dict] = []
            for feat in selected_features:
                feat_name = feat["name"]

                if feat_name not in dataset.columns:
                    logger.warning(
                        "write_feature_stratified_performance: feature column "
                        "'%s' absent from dataset for split=%s — skipping",
                        feat_name,
                        split,
                    )
                    continue

                raw_vals = dataset[feat_name].dropna()
                fuzzy_values = [
                    v for v in raw_vals.unique() if isinstance(v, str)]

                if not fuzzy_values:
                    continue

                valid_log["_feat_val"] = dataset[feat_name].iloc[
                    valid_log["Entry_Index"].values
                ].values

                for fuzzy_value in fuzzy_values:
                    stratum = valid_log[valid_log["_feat_val"] == fuzzy_value]
                    num_trades = len(stratum)

                    if num_trades == 0:
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

                    if _cfg.INITIAL_CAPITAL != 0:
                        total_return_pct = float(
                            stratum["Net_PnL"].sum() /
                            _cfg.INITIAL_CAPITAL * 100
                        )
                    else:
                        total_return_pct = 0.0

                    win_rate = float(
                        (stratum["Net_PnL"] > 0).sum()) / num_trades

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

            df = pd.DataFrame(rows, columns=_COLUMNS)
            df.to_csv(out_path, index=False)

            abs_path = os.path.abspath(out_path)
            logger.info(
                "Saved feature-stratified performance CSV: %s", abs_path
            )
            result_paths.append(abs_path)

        return result_paths

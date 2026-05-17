"""
reporter.py — Reporting and visualization for the GPU-Fuzzy Trading Pipeline.

Generates:
  - Phase 2 generation metrics plots (objectives vs. generation)
  - Equity curve plots (train, validation, test)
  - Per-symbol performance CSVs
  - Phase 4 RL training curve plots with elbow point marked

All plots are saved as PNG files (not displayed).
"""

from __future__ import annotations

import logging
import os
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import pandas as pd

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level output paths (overridable in tests)
# ---------------------------------------------------------------------------
_REPORTS_DIR = _cfg.REPORTS_DIR


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            logger.warning("plot_phase2_metrics: empty history for direction=%s", direction)
            return out_path

        generations = [entry.get("generation", i) for i, entry in enumerate(history)]
        mean_f1 = [entry.get("mean_f1", 0.0) for entry in history]
        mean_f2 = [entry.get("mean_f2", 0.0) for entry in history]
        mean_f3 = [entry.get("mean_f3", 0.0) for entry in history]

        ax.plot(generations, mean_f1, label="mean_f1 (−return)", color="tab:blue")
        ax.plot(generations, mean_f2, label="mean_f2 (drawdown)", color="tab:orange")
        ax.plot(generations, mean_f3, label="mean_f3 (−win_rate)", color="tab:green")

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
        out_path = os.path.join(reports_dir, f"{split}_per_symbol_performance.csv")
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
        out_path = os.path.join(reports_dir, f"phase4_{direction}_rl_curve.png")
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

        # Mark the elbow point if the index is valid
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

#!/usr/bin/env python3
"""Plot Phase 1 rolling-regression regime detection on train.csv."""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# Repo root on path
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from gpu_fuzzy_trader import config
from gpu_fuzzy_trader.features.regime_cluster import fit_regime_labels

REGIME_COLORS = {
    0: "#bdbdbd",  # sideways
    1: "#e57373",  # bearish
    2: "#81c784",  # bullish
}
REGIME_NAMES = {0: "Sideways (0)", 1: "Bearish (1)", 2: "Bullish (2)"}


def _resolve_price_column(columns: list[str], requested: str | None) -> str:
    if requested:
        if requested not in columns:
            raise ValueError(f"Column {requested!r} not in dataset. Available: {columns[:12]}…")
        return requested
    for name in ("close", "label_open_next", "label_close_288"):
        if name in columns:
            return name
    raise ValueError("No price column found (expected close, label_open_next, or label_close_288).")


def _daily_price_and_regime(
    df_sym: pd.DataFrame,
    price_col: str,
    regimes: pd.Series,
) -> pd.DataFrame:
    out = df_sym[["datetime", price_col]].copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    out["regime"] = regimes.reindex(df_sym.index).values
    out = out.set_index("datetime").sort_index()
    daily = out.resample("D").agg({price_col: "last", "regime": "last"}).dropna()
    daily["regime"] = daily["regime"].astype(int)
    return daily


def _shade_regimes(ax, daily: pd.DataFrame, price_col: str) -> None:
    if daily.empty:
        return
    dates = daily.index
    prices = daily[price_col].values
    regimes = daily["regime"].values.astype(int)
    ax.plot(dates, prices, color="#212121", linewidth=0.9, zorder=3)
    block_start = 0
    for i in range(1, len(regimes) + 1):
        if i == len(regimes) or regimes[i] != regimes[block_start]:
            regime = int(regimes[block_start])
            ax.axvspan(
                dates[block_start], dates[i - 1],
                color=REGIME_COLORS[regime], alpha=0.35, zorder=1,
            )
            block_start = i


def plot_regimes(
    train_path: str,
    output_path: str,
    price_col: str | None = None,
    symbols: list | None = None,
) -> str:
    header = pd.read_csv(train_path, nrows=0)
    price_col = _resolve_price_column(list(header.columns), price_col)
    usecols = ["datetime", "symbol", price_col]
    df = pd.read_csv(train_path, usecols=usecols)
    df["datetime"] = pd.to_datetime(df["datetime"])

    # Pipeline regime logic reads label_open_next; map chosen price column there.
    df_reg = df.rename(columns={price_col: "label_open_next"})
    fit_result = fit_regime_labels(df_reg)
    if fit_result is None:
        raise RuntimeError("Regime detection returned no labels.")
    labels, bundle = fit_result

    sym_list = symbols if symbols else sorted(df["symbol"].unique())
    n = len(sym_list)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, 3.2 * nrows), sharex=False, squeeze=False,
    )
    axes_flat = axes.ravel()

    for idx, sym in enumerate(sym_list):
        ax = axes_flat[idx]
        mask = df["symbol"] == sym
        daily = _daily_price_and_regime(
            df.loc[mask], price_col, labels.loc[mask],
        )
        _shade_regimes(ax, daily, price_col)
        ax.set_title(f"Symbol {sym}", fontsize=10, fontweight="bold")
        ax.set_ylabel(price_col, fontsize=8)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=7)

    for j in range(len(sym_list), len(axes_flat)):
        axes_flat[j].set_visible(False)

    legend_patches = [
        mpatches.Patch(facecolor=REGIME_COLORS[k], alpha=0.5, label=REGIME_NAMES[k])
        for k in (0, 1, 2)
    ]
    fig.legend(
        handles=legend_patches, loc="upper center", ncol=3, fontsize=9,
        frameon=True, bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Phase 1 regime detection on train.csv\n"
        f"Price: {price_col} · fast={bundle['fast_window']}d slow={bundle['slow_window']}d "
        f"med={bundle['med_window']}d min_block={bundle['min_days']}d",
        fontsize=12, fontweight="bold", y=1.06,
    )
    fig.supxlabel("Date (daily resample)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", default=config.TRAIN_CSV_PATH, help="Path to train.csv",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(config.OUTPUTS_DIR, "reports", "regime_detection_train.png"),
        help="Output PNG path",
    )
    parser.add_argument(
        "--price-col",
        default=None,
        help="Price column (default: close if present, else label_open_next)",
    )
    parser.add_argument(
        "--symbols", nargs="*", type=int, default=None,
        help="Subset of symbol ids (default: all)",
    )
    args = parser.parse_args()
    path = plot_regimes(args.train, args.output, args.price_col, args.symbols)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

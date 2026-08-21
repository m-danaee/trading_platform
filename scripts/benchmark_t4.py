#!/usr/bin/env python3
"""Benchmark and hardware baseline profiler for T4 / local environments.

Probes host and GPU hardware capabilities, runs synthetic micro-benchmarks
(GPU backtest engine and Data Loader if available), and outputs a structured
JSON report to stdout (with --dry-run) or to outputs/reports/t4_profile.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader._gpu_runtime import (
    detect_hardware_profile,
    resolve_phase2_gpu_batch_size,
)

logger = logging.getLogger("benchmark_t4")


def run_data_loader_benchmark(n_samples: int = 50000) -> dict[str, Any]:
    """Run a quick micro-benchmark on synthetic feature framing and Data_Loader."""
    import tempfile
    import numpy as np
    import pandas as pd
    from gpu_fuzzy_trader.data.loader import Data_Loader

    rng = np.random.default_rng(42)
    symbols = ["SYM_A", "SYM_B", "SYM_C", "SYM_D"]
    rows_per_sym = n_samples // len(symbols)
    dfs = []
    for sym in symbols:
        dt = pd.date_range("2024-01-01", periods=rows_per_sym, freq="5min")
        dfs.append(pd.DataFrame({
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": sym,
            "open": rng.uniform(100, 110, rows_per_sym),
            "high": rng.uniform(110, 120, rows_per_sym),
            "low": rng.uniform(90, 100, rows_per_sym),
            "close": rng.uniform(95, 115, rows_per_sym),
            "volume": rng.uniform(1000, 5000, rows_per_sym),
            "feature_1": rng.uniform(-1.0, 1.0, rows_per_sym),
            "feature_2": rng.uniform(-1.0, 1.0, rows_per_sym),
            "feature_3": rng.uniform(-1.0, 1.0, rows_per_sym),
        }))
    synth_df = pd.concat(dfs, ignore_index=True)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        synth_df.to_csv(f.name, index=False)
        tmp_path = f.name

    try:
        t0 = time.perf_counter()
        loaded = Data_Loader().load_dataset(tmp_path, require_context=False)
        elapsed = time.perf_counter() - t0
        loader_ms = elapsed * 1000.0
        return {
            "status": "success",
            "rows": len(loaded),
            "loader_ms": round(loader_ms, 2),
            "elapsed_sec": round(elapsed, 4),
            "symbol_is_category": str(loaded["symbol"].dtype) == "category",
            "datetime_is_datetime64": pd.api.types.is_datetime64_any_dtype(loaded["datetime"]),
        }
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def run_jax_engine_benchmark(batch_size: int | None = None) -> dict[str, Any]:
    """Run synthetic GPUBacktestEngine simulate_rule_batch if JAX is functional."""
    t0 = time.perf_counter()
    try:
        import jax
        import numpy as np
        import pandas as pd

        from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine

        resolved_batch = batch_size or resolve_phase2_gpu_batch_size()
        n = 1000
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "symbol": ["BENCH"] * n,
            "datetime": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "_symbol_bar_index": np.arange(n),
            "label_open_next": rng.uniform(90, 110, n),
            "label_max_288": rng.uniform(92, 115, n),
            "label_min_288": rng.uniform(85, 98, n),
            "label_close_288": rng.uniform(90, 110, n),
            "label_max_before_min": rng.integers(0, 2, n),
            "feat_a": rng.integers(0, 2, n),
            "feat_b": rng.integers(0, 5, n),
            "feat_c": rng.integers(0, 3, n),
        })
        feature_modes = {"feat_a": "binary", "feat_b": "positive", "feat_c": "ternary"}
        engine = GPUBacktestEngine(df, feature_modes, "long")

        k = len(feature_modes)
        chrom = np.zeros((resolved_batch, k), dtype=np.int32)

        # Warmup
        engine.simulate_rule_batch(
            chrom,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            generation=1,
            is_last_gen=False,
        )
        try:
            jax.block_until_ready(jax.numpy.zeros(1))
        except Exception:
            pass

        t_start = time.perf_counter()
        engine.simulate_rule_batch(
            chrom,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            generation=1,
            is_last_gen=False,
        )
        try:
            jax.block_until_ready(jax.numpy.zeros(1))
        except Exception:
            pass
        t_exec = time.perf_counter() - t_start

        return {
            "status": "success",
            "batch_size": resolved_batch,
            "exec_time_sec": round(t_exec, 4),
            "total_time_sec": round(time.perf_counter() - t0, 4),
        }
    except Exception as exc:
        return {
            "status": "skipped",
            "reason": str(exc),
            "total_time_sec": round(time.perf_counter() - t0, 4),
        }


def collect_profile_report(
    skip_bench: bool = False,
    component: str = "all",
    rows: int = 50000,
) -> dict[str, Any]:
    """Collect complete hardware snapshot and benchmark metrics."""
    hw = detect_hardware_profile()
    batch_resolved = resolve_phase2_gpu_batch_size()

    report: dict[str, Any] = {
        "timestamp": time.time(),
        "hardware": {
            "gpu_name": hw.gpu_name,
            "vram_gb": hw.vram_gb,
            "ram_gb": hw.ram_gb,
            "cpu_count": hw.cpu_count,
            "jax_backend": hw.jax_backend,
            "devices": hw.devices,
            "is_t4": hw.is_t4,
        },
        "resolved_config": {
            "batch_size": batch_resolved,
            "scan_unroll": _cfg.PHASE2_SCAN_UNROLL,
            "cpu_route_large_data": getattr(_cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", True),
            "workers": getattr(_cfg, "BACKTEST_BATCH_WORKERS", 1),
        },
        "benchmarks": {},
    }

    if not skip_bench:
        if component in ("all", "loader"):
            report["benchmarks"]["data_loader"] = run_data_loader_benchmark(n_samples=rows)
        if component in ("all", "gpu"):
            report["benchmarks"]["gpu_engine"] = run_jax_engine_benchmark(batch_resolved)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="T4 Hardware Profiler & Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print JSON to stdout without writing file")
    parser.add_argument("--output", type=str, default="", help="Path to write JSON profile")
    parser.add_argument("--skip-micro-bench", action="store_true", help="Skip running synthetic JAX/Loader micro-bench")
    parser.add_argument("--component", type=str, default="all", choices=["all", "loader", "gpu"], help="Micro-bench component to run")
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows for data loader benchmark")
    args = parser.parse_args()

    report = collect_profile_report(
        skip_bench=args.skip_micro_bench,
        component=args.component,
        rows=args.rows,
    )
    rendered = json.dumps(report, indent=2)

    if args.dry_run or not args.output:
        print(rendered)

    if not args.dry_run:
        out_path = Path(args.output) if args.output else _PROJECT_ROOT / "outputs" / "reports" / "t4_profile.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")
        print(f"Profile saved to: {out_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Phase 2 GPU runtime helpers: VRAM-aware batch size and JAX warmup."""

from __future__ import annotations

import logging
import os
import subprocess
from typing import TYPE_CHECKING

from gpu_fuzzy_trader import config as _cfg

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine


def detect_gpu_vram_gb() -> float | None:
    """Return total GPU VRAM in GiB via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        first_line = out.strip().splitlines()[0].strip()
        mib = float(first_line)
        return mib / 1024.0
    except Exception:
        return None


def detect_gpu_memory_used_gb() -> float | None:
    """Return current GPU memory used in GiB via nvidia-smi, or None."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        first_line = out.strip().splitlines()[0].strip()
        mib = float(first_line)
        return mib / 1024.0
    except Exception:
        return None


def resolve_phase2_gpu_batch_size() -> int:
    """
    Return Phase 2 GPU vmap chunk size.

    Priority: ``PHASE2_GPU_BATCH_SIZE`` env override > VRAM heuristic >
    ``config.PHASE2_GPU_BATCH_SIZE``.

    Heuristic (when ``PHASE2_GPU_BATCH_SIZE_AUTO`` is not disabled):

    Peak VRAM scales ~linearly with batch size (rule match is O(B×N×K)).
    With ``PHASE2_CV_FOLD_WORKERS=1`` only one fold runs at a time, so T4
    (15 GiB) can use larger batches than the old multi-fold parallel cap.

    - <= 8 GiB: 16
    - <= 12 GiB: 32
    - <= 16 GiB: 64  (Colab T4 — ~3–4 GiB peak at B=64, N=300k)
    - <= 24 GiB: 96
    - > 24 GiB: config default
    """
    env_override = os.environ.get("PHASE2_GPU_BATCH_SIZE", "").strip()
    if env_override:
        return max(1, int(env_override))

    auto = os.environ.get(
        "PHASE2_GPU_BATCH_SIZE_AUTO", "true").strip().lower()
    if auto in ("0", "false", "no"):
        return max(1, int(_cfg.PHASE2_GPU_BATCH_SIZE))

    config_default = max(1, int(_cfg.PHASE2_GPU_BATCH_SIZE))
    vram = detect_gpu_vram_gb()
    if vram is None:
        return config_default
    if vram <= 8.0:
        return min(config_default, 16)
    if vram <= 12.0:
        return min(config_default, 32)
    if vram <= 16.0:
        return min(config_default, 64)
    if vram <= 24.0:
        return min(config_default, 96)
    return config_default


def log_gpu_runtime_config() -> None:
    """Log resolved Phase 2 GPU knobs once at startup."""
    try:
        import jax

        backend = jax.default_backend()
        devices = jax.devices()
    except Exception:
        backend = "unknown"
        devices = []

    batch = resolve_phase2_gpu_batch_size()
    vram = detect_gpu_vram_gb()
    vram_str = f"{vram:.1f} GiB" if vram is not None else "unknown"
    used = detect_gpu_memory_used_gb()
    used_str = f"{used:.2f} GiB" if used is not None else "unknown"
    logger.info(
        "Phase 2 GPU runtime: backend=%s devices=%s vram=%s used=%s "
        "batch_size=%d scan_unroll=%d fp32=%s data_int8=%s",
        backend,
        devices,
        vram_str,
        used_str,
        batch,
        _cfg.PHASE2_SCAN_UNROLL,
        getattr(_cfg, "PHASE2_GPU_USE_FP32", True),
        getattr(_cfg, "PHASE2_GPU_DATA_INT8", True),
    )


def _warmup_engine(engine: object, batch_size: int = 1) -> None:
    """Run a representative ``simulate_rule_batch`` to compile JAX kernels."""
    import numpy as np

    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
        empty_slots,
        use_sparse_slots,
    )

    target = getattr(engine, "_inner", engine)
    n = max(1, int(batch_size))

    if use_sparse_slots():
        slots = np.tile(empty_slots()[None, :, :], (n, 1, 1))
        engine.simulate_rule_batch(
            slots,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
        return

    k = int(target._data_matrix_jax.shape[1])
    if k == 0:
        chrom = np.zeros((n, 0), dtype=np.int32)
    else:
        dc = int(np.asarray(target._dont_cares_jax)[0])
        chrom = np.full((n, k), dc, dtype=np.int32)

    engine.simulate_rule_batch(
        chrom,
        tp=_cfg.PHASE2_TP,
        sl=_cfg.PHASE2_SL,
        capital_pct=_cfg.PHASE2_CAPITAL_PCT,
    )


def _iter_warmup_targets(*engines: object | None) -> list[object]:
    """Expand CV facades into per-fold engines for JAX warmup."""
    targets: list[object] = []
    for engine in engines:
        if engine is None:
            continue
        fold_engines = getattr(engine, "_fold_engines", None)
        if fold_engines:
            targets.extend(fold_engines)
        elif hasattr(engine, "simulate_rule_batch"):
            targets.append(engine)
    return targets


def warmup_phase2_gpu_kernels(
    engine: object,
    val_engine: object | None = None,
) -> None:
    """
    Compile JAX kernels with representative shapes before evolution.

    Warms every fold engine in train (and optional val) CV facades at full
    production batch size so Generation 1 does not pay lazy JIT costs.
    """
    batch_size = resolve_phase2_gpu_batch_size()
    targets = _iter_warmup_targets(engine, val_engine)
    if not targets:
        return

    for target in targets:
        _warmup_engine(target, batch_size=batch_size)

    used = detect_gpu_memory_used_gb()
    used_str = f"{used:.2f} GiB" if used is not None else "unknown"
    logger.info(
        "Phase 2 JAX warmup complete (%d engines, batch_size=%d, gpu_used=%s)",
        len(targets),
        batch_size,
        used_str,
    )


def configure_phase2_gpu_runtime(
    engine: object,
    val_engine: object | None = None,
) -> None:
    """Log GPU config and warm up JAX kernels when Phase 2 uses GPU."""
    if not _cfg.PHASE2_USE_GPU:
        return
    log_gpu_runtime_config()
    try:
        warmup_phase2_gpu_kernels(engine, val_engine=val_engine)
    except Exception as exc:
        logger.debug("Phase 2 JAX warmup skipped: %s", exc)

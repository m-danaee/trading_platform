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

# Avoid re-compiling identical (n_rows, K, regime, batch) shapes across islands.
_WARMED_SIGNATURES: set[tuple] = set()


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


def detect_system_ram_gb() -> float | None:
    """Return total system RAM in GiB, or None if unavailable."""
    try:
        with open("/proc/meminfo", encoding="ascii") as meminfo:
            for line in meminfo:
                if line.startswith("MemTotal:"):
                    kib = int(line.split()[1])
                    return kib / (1024.0 * 1024.0)
    except Exception:
        pass
    try:
        import psutil

        return psutil.virtual_memory().total / (1024.0 ** 3)
    except Exception:
        return None


def _vram_batch_cap(vram: float | None, config_default: int) -> int:
    """VRAM-tier cap for Phase 2 GPU vmap chunk size."""
    if vram is None:
        # Unknown GPU: avoid using the full config default (often tuned for big GPUs).
        return min(config_default, 32)
    if vram <= 8.0:
        return min(config_default, 16)
    if vram <= 12.0:
        return min(config_default, 32)
    if vram <= 16.0:
        return min(config_default, 128)
    if vram <= 24.0:
        return min(config_default, 96)
    return config_default


def _ram_batch_cap(ram: float | None) -> int | None:
    """Host-RAM tier cap; JAX compile + CV engines dominate Colab SIGKILL."""
    if ram is None:
        return None
    if ram <= 13.0:
        return 32
    if ram <= 16.0:
        return 64
    return None


def resolve_phase2_gpu_batch_size() -> int:
    """
    Return Phase 2 GPU vmap chunk size.

    Priority: ``PHASE2_GPU_BATCH_SIZE`` env override > config auto flag >
    VRAM heuristic > ``config.PHASE2_GPU_BATCH_SIZE``.

    Heuristic (when ``PHASE2_GPU_BATCH_SIZE_AUTO`` is enabled):

    Peak VRAM scales ~linearly with batch size (rule match is O(B×N×K)).
    Host RAM also spikes during JAX XLA compile at the warmup batch size.

    VRAM tiers (applied first):
    - unknown: min(config, 32)
    - <= 8 GiB: 16
    - <= 12 GiB: 32
    - <= 16 GiB: 128  (Colab T4)
    - <= 24 GiB: 96
    - > 24 GiB: config default

    Host RAM tiers (final cap):
    - <= 13 GiB: 32  (Colab free tier — avoids SIGKILL during compile)
    - <= 16 GiB: 64
    """
    env_override = os.environ.get("PHASE2_GPU_BATCH_SIZE", "").strip()
    if env_override:
        return max(1, int(env_override))

    auto_default = (
        "true" if getattr(_cfg, "PHASE2_GPU_BATCH_SIZE_AUTO", True) else "false"
    )
    auto = os.environ.get(
        "PHASE2_GPU_BATCH_SIZE_AUTO", auto_default,
    ).strip().lower()
    if auto in ("0", "false", "no"):
        return max(1, int(_cfg.PHASE2_GPU_BATCH_SIZE))

    config_default = max(1, int(_cfg.PHASE2_GPU_BATCH_SIZE))
    vram_cap = _vram_batch_cap(detect_gpu_vram_gb(), config_default)
    ram_cap = _ram_batch_cap(detect_system_ram_gb())
    if ram_cap is not None:
        return min(vram_cap, ram_cap)
    return vram_cap


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
    ram = detect_system_ram_gb()
    ram_str = f"{ram:.1f} GiB" if ram is not None else "unknown"
    used = detect_gpu_memory_used_gb()
    used_str = f"{used:.2f} GiB" if used is not None else "unknown"
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    logger.info(
        "Phase 2 GPU runtime: backend=%s devices=%s vram=%s ram=%s used=%s "
        "batch_size=%d scan_unroll=%d fp32=%s data_int8=%s jax_cache=%s",
        backend,
        devices,
        vram_str,
        ram_str,
        used_str,
        batch,
        _cfg.PHASE2_SCAN_UNROLL,
        getattr(_cfg, "PHASE2_GPU_USE_FP32", True),
        getattr(_cfg, "PHASE2_GPU_DATA_INT8", True),
        cache_dir or "disabled",
    )


def _resolve_warmup_inner(engine: object) -> object:
    return getattr(engine, "_inner", engine)


def _warmup_signature(engine: object, batch_size: int) -> tuple:
    """Hashable key for JAX kernels already compiled for this engine shape."""
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import use_sparse_slots

    inner = _resolve_warmup_inner(engine)
    n_rows = len(getattr(inner, "df", ()))
    if hasattr(inner, "_data_matrix_jax"):
        k = int(inner._data_matrix_jax.shape[1])
    else:
        k = 0
    encoding = "sparse" if use_sparse_slots() else "dense"
    return (n_rows, k, int(batch_size), encoding)


def _warmup_engine(engine: object, batch_size: int = 1) -> None:
    """Run a representative ``simulate_rule_batch`` to compile JAX kernels."""
    import numpy as np

    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
        empty_slots,
        use_sparse_slots,
    )

    signature = _warmup_signature(engine, batch_size)
    if signature in _WARMED_SIGNATURES:
        return

    target = _resolve_warmup_inner(engine)
    n = max(1, int(batch_size))

    if use_sparse_slots():
        slots = np.tile(empty_slots()[None, :, :], (n, 1, 1))
        target.simulate_rule_batch(
            slots,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
    else:
        k = int(target._data_matrix_jax.shape[1])
        if k == 0:
            chrom = np.zeros((n, 0), dtype=np.int32)
        else:
            dc = int(np.asarray(target._dont_cares_jax)[0])
            chrom = np.full((n, k), dc, dtype=np.int32)

        target.simulate_rule_batch(
            chrom,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )

    # Force XLA compile to finish before evolution starts.
    try:
        import jax

        jax.block_until_ready(jax.numpy.zeros(1))
    except Exception:
        pass

    _WARMED_SIGNATURES.add(signature)


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
        logger.warning("Phase 2 JAX warmup: no engines to warm")
        return

    warmed = 0
    skipped = 0
    for target in targets:
        sig = _warmup_signature(target, batch_size)
        if sig in _WARMED_SIGNATURES:
            skipped += 1
            continue
        _warmup_engine(target, batch_size=batch_size)
        warmed += 1

    used = detect_gpu_memory_used_gb()
    used_str = f"{used:.2f} GiB" if used is not None else "unknown"
    logger.info(
        "Phase 2 JAX warmup complete (%d engines warmed, %d skipped, "
        "batch_size=%d, gpu_used=%s, signatures=%d)",
        warmed,
        skipped,
        batch_size,
        used_str,
        len(_WARMED_SIGNATURES),
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
        logger.warning(
            "Phase 2 JAX warmup failed — Generation 1 will pay JIT compile cost: %s",
            exc,
            exc_info=True,
        )

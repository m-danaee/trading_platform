"""Phase 2 GPU runtime helpers: VRAM-aware batch size and JAX warmup."""

from __future__ import annotations

import functools
import logging
import os
import subprocess

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.research_profile import HardwareProfile

logger = logging.getLogger(__name__)

# Avoid re-compiling identical (n_rows, K, batch) shapes across islands.
_WARMED_SIGNATURES: set[tuple] = set()


@functools.lru_cache(maxsize=1)
def detect_gpu_name() -> str | None:
    """Return GPU product name via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=5,
            stderr=subprocess.DEVNULL,
        )
        first_line = out.strip().splitlines()[0].strip()
        return first_line if first_line else None
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
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


@functools.lru_cache(maxsize=1)
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
    """VRAM-tier cap for Phase 2 GPU vmap chunk size.

    Tiers:
    - unknown: min(config, 64)
    - <= 8 GiB: min(config, 64)
    - <= 12 GiB: min(config, 256)
    - <= 16 GiB: min(config, 256)  (Colab T4 / generic T4)
    - <= 24 GiB: min(config, 256)  (RTX 3090 / 4090 / A10G — sustain full 256 batch)
    - > 24 GiB: config default (A100 / H100 / etc.)
    """
    if vram is None:
        # Unknown GPU: avoid using the full config default (often tuned for
        # desktop cards with more memory).
        return min(config_default, 64)
    if vram <= 8.0:
        return min(config_default, 64)
    if vram <= 12.0:
        # A 12-GiB RTX 3080 Ti can hold a full 256-rule chunk with the
        # int8 feature matrix and float32 kernels used by Phase 2.
        return min(config_default, 256)
    if vram <= 16.0:
        return min(config_default, 256)
    if vram <= 24.0:
        return min(config_default, 256)
    return config_default


def _ram_batch_cap(
    ram: float | None,
    gpu_route_active: bool = False,
) -> int | None:
    """Host-RAM tier cap; JAX compile + CV engines dominate Colab SIGKILL.

    When GPU route is active (PHASE2_GPU_CPU_ROUTE_LARGE_DATA is False),
    13-16 GiB host RAM is raised from 64 to 128 (or 256) to prevent throttling T4.
    For small RAM <= 12 GiB, cap at 64 to avoid SIGKILL during XLA compile.
    """
    if ram is None:
        return None
    if ram <= 12.0:
        return 64
    if ram <= 13.0:
        return 128 if gpu_route_active else 64
    if ram <= 16.0:
        return 256
    return None


@functools.lru_cache(maxsize=1)
def resolve_phase2_gpu_batch_size() -> int:
    """
    Return Phase 2 GPU vmap chunk size (cached — one probe per process).

    Priority: ``PHASE2_GPU_BATCH_SIZE`` env override > config auto flag >
    VRAM heuristic > ``config.PHASE2_GPU_BATCH_SIZE``.

    Heuristic (when ``PHASE2_GPU_BATCH_SIZE_AUTO`` is enabled):

    Peak VRAM scales ~linearly with batch size (rule match is O(B×N×K)).
    Host RAM also spikes during JAX XLA compile at the warmup batch size.

    VRAM tiers (applied first):
    - unknown: min(config, 64)
    - <= 8 GiB: min(config, 64)
    - <= 12 GiB: min(config, 256)
    - <= 16 GiB: min(config, 256)  (Colab T4 / generic T4 — covers full pop)
    - <= 24 GiB: min(config, 256)
    - > 24 GiB: config default

    Host RAM tiers (final cap):
    - <= 12 GiB: 64  (small hosts — avoids SIGKILL during compile)
    - <= 13 GiB: 128 when GPU route active, else 64
    - <= 16 GiB: 256 (desktop 12-GiB GPUs remain throughput-bound)

    Note: ``@lru_cache`` ensures nvidia-smi and /proc/meminfo are probed only
    once per process, not on every ``simulate_rule_batch`` call.
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
    gpu_route_active = not getattr(_cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", True)
    ram_cap = _ram_batch_cap(detect_system_ram_gb(), gpu_route_active=gpu_route_active)
    if ram_cap is not None:
        return min(vram_cap, ram_cap)
    return vram_cap


@functools.lru_cache(maxsize=1)
def is_t4_runtime() -> bool:
    """True when running on an NVIDIA Tesla T4 GPU or explicit T4 env override."""
    env_t4 = os.environ.get("GPU_OPT_T4", "").strip().lower()
    if env_t4 in ("1", "true", "yes"):
        return True
    if env_t4 in ("0", "false", "no"):
        return False

    gpu_name = detect_gpu_name()
    if gpu_name and "t4" in gpu_name.lower():
        return True

    # Fallback heuristic: 14.5 - 16.5 GiB VRAM on Linux without specific name
    vram = detect_gpu_vram_gb()
    if vram is not None and 14.5 <= vram <= 16.5:
        return True

    return False


@functools.lru_cache(maxsize=1)
def detect_hardware_profile() -> HardwareProfile:
    """Snapshot local host hardware and active JAX backend info."""
    try:
        import jax

        backend = jax.default_backend()
        devices = [str(d) for d in jax.devices()]
    except Exception:
        backend = "unknown"
        devices = []

    return HardwareProfile(
        gpu_name=detect_gpu_name(),
        vram_gb=detect_gpu_vram_gb(),
        ram_gb=detect_system_ram_gb(),
        cpu_count=os.cpu_count() or 1,
        jax_backend=backend,
        devices=devices,
        is_t4=is_t4_runtime(),
    )


def log_gpu_runtime_config() -> None:
    """Log resolved Phase 2 GPU knobs once at startup."""
    hw = detect_hardware_profile()
    batch = resolve_phase2_gpu_batch_size()
    vram_str = f"{hw.vram_gb:.1f} GiB" if hw.vram_gb is not None else "unknown"
    ram_str = f"{hw.ram_gb:.1f} GiB" if hw.ram_gb is not None else "unknown"
    used = detect_gpu_memory_used_gb()
    used_str = f"{used:.2f} GiB" if used is not None else "unknown"
    gpu_name_str = hw.gpu_name or "none"
    cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR", "")
    logger.info(
        "Phase 2 runtime: gpu=%s backend=%s devices=%s vram=%s ram=%s used=%s "
        "cpu_count=%d workers=%d is_t4=%s batch_size=%d scan_unroll=%d fp32=%s "
        "data_int8=%s large_window_cpu_route=%s jax_cache=%s",
        gpu_name_str,
        hw.jax_backend,
        hw.devices,
        vram_str,
        ram_str,
        used_str,
        hw.cpu_count,
        getattr(_cfg, "BACKTEST_BATCH_WORKERS", 1),
        hw.is_t4,
        batch,
        _cfg.PHASE2_SCAN_UNROLL,
        getattr(_cfg, "PHASE2_GPU_USE_FP32", True),
        getattr(_cfg, "PHASE2_GPU_DATA_INT8", True),
        getattr(_cfg, "PHASE2_GPU_CPU_ROUTE_LARGE_DATA", True),
        cache_dir or "disabled",
    )


def _resolve_warmup_inner(engine: object) -> object:
    return getattr(engine, "_inner", engine)


def _warmup_signature(
    engine: object,
    batch_size: int,
    cluster_id: str | int | None = None,
) -> tuple:
    """Hashable key for JAX kernels already compiled for this engine shape.

    When *cluster_id* is provided it is appended as the last tuple element
    so that ``evict_cluster_signatures`` can filter signatures per cluster.
    """
    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import use_sparse_slots

    inner = _resolve_warmup_inner(engine)
    n_rows = len(getattr(inner, "df", ()))
    if hasattr(inner, "_data_matrix_jax"):
        k = int(inner._data_matrix_jax.shape[1])
    else:
        k = 0
    encoding = "sparse" if use_sparse_slots() else "dense"
    base = (n_rows, k, int(batch_size), encoding)
    if cluster_id is not None:
        return base + (str(cluster_id),)
    return base


def _warmup_engine(
    engine: object,
    batch_size: int = 1,
    cluster_id: str | int | None = None,
) -> None:
    """Run a representative ``simulate_rule_batch`` to compile JAX kernels.

    The signature is tagged with *cluster_id* so the later eviction helper
    can filter per-cluster entries from ``_WARMED_SIGNATURES``.
    """
    import numpy as np

    from gpu_fuzzy_trader.phases.phase2_sparse_encoding import (
        empty_slots,
        use_sparse_slots,
    )

    signature = _warmup_signature(engine, batch_size, cluster_id=cluster_id)
    if signature in _WARMED_SIGNATURES:
        return

    target = _resolve_warmup_inner(engine)
    n = max(1, int(batch_size))

    # Large-window RTX 4050 runs may intentionally use the optimized CPU
    # ranking path.  Do not warm a GPU kernel with an all-wildcard sparse rule:
    # that would perform a very expensive CPU admission pass without helping
    # the subsequent evolution calls.
    if (
        hasattr(target, "_should_route_batch_to_cpu")
        and target._should_route_batch_to_cpu(n)
    ):
        logger.info(
            "Phase 2 warmup: routing batch %d to CPU (skipped JAX kernel compile).",
            n,
        )
        _WARMED_SIGNATURES.add(signature)
        return

    if use_sparse_slots():
        slots = np.tile(empty_slots()[None, :, :], (n, 1, 1))
        target.simulate_rule_batch(
            slots,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            # Compile the GPU kernel without triggering the optional CPU
            # per-symbol enrichment pass.  ``generation=None`` means
            # "always enrich" in GPUBacktestEngine.
            generation=1,
            is_last_gen=False,
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
            generation=1,
            is_last_gen=False,
        )

    # Force XLA compile to finish before evolution starts.
    try:
        import jax

        jax.block_until_ready(jax.numpy.zeros(1))
    except Exception:
        pass

    _WARMED_SIGNATURES.add(signature)
    logger.debug("Phase 2 JAX warmup done for signature: %s", signature)


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
    cluster_id: str | int | None = None,
) -> None:
    """
    Compile JAX kernels with representative shapes before evolution.

    Warms every fold engine in train (and optional val) CV facades at full
    production batch size (and half-batch if >32) so Generation 1 does not pay
    lazy JIT compile costs.

    When *cluster_id* is provided the compiled signatures are tagged so
    ``evict_cluster_signatures`` can evict them when the cluster finishes.
    """
    batch_size = resolve_phase2_gpu_batch_size()
    targets = _iter_warmup_targets(engine, val_engine)
    if not targets:
        logger.warning("Phase 2 JAX warmup: no engines to warm")
        return

    # Pre-warm at full batch size and intermediate batch sizes if applicable
    warmup_batches = [batch_size]
    if batch_size > 32:
        warmup_batches.append(batch_size // 2)

    warmed = 0
    skipped = 0
    for b in warmup_batches:
        for target in targets:
            sig = _warmup_signature(target, b, cluster_id=cluster_id)
            if sig in _WARMED_SIGNATURES:
                skipped += 1
                continue
            _warmup_engine(target, batch_size=b, cluster_id=cluster_id)
            warmed += 1

    used = detect_gpu_memory_used_gb()
    used_str = f"{used:.2f} GiB" if used is not None else "unknown"
    logger.info(
        "Phase 2 JAX warmup complete (%d engines warmed, %d skipped, "
        "batch_size=%d, gpu_used=%s, signatures=%d, cluster_id=%s)",
        warmed,
        skipped,
        batch_size,
        used_str,
        len(_WARMED_SIGNATURES),
        cluster_id or "none",
    )


def evict_cluster_signatures(cluster_id: str | int | None = None) -> int:
    """Evict JAX compiled signatures for a completed cluster.

    Removes entries from ``_WARMED_SIGNATURES`` that belong to a specific
    cluster and tries to free the JAX-compiled programs. Returns the number
    of signatures evicted.

    When *cluster_id* is ``None``, **all** signatures are evicted (useful
    between long / short directions).
    """
    global _WARMED_SIGNATURES

    if cluster_id is not None:
        before = len(_WARMED_SIGNATURES)
        cid_str = str(cluster_id)
        to_evict = {
            sig
            for sig in _WARMED_SIGNATURES
            if (
                isinstance(sig, tuple)
                and len(sig) >= 2
                and str(sig[-1]) == cid_str
            )
        }
        _WARMED_SIGNATURES -= to_evict
        evicted = before - len(_WARMED_SIGNATURES)
    else:
        evicted = len(_WARMED_SIGNATURES)
        _WARMED_SIGNATURES.clear()

    if evicted > 0:
        try:
            import jax

            if hasattr(jax, "clear_caches"):
                jax.clear_caches()
        except Exception:
            pass
        import gc as _gc

        _gc.collect()

    return evicted


def configure_phase2_gpu_runtime(
    engine: object,
    val_engine: object | None = None,
    cluster_id: str | int | None = None,
) -> None:
    """Log GPU config and warm up JAX kernels when Phase 2 uses GPU.

    *cluster_id* is forwarded to ``warmup_phase2_gpu_kernels`` so every
    compiled signature is tagged with the cluster identifier, enabling
    per-cluster eviction later.
    """
    if not _cfg.PHASE2_USE_GPU:
        return
    # The large-window route can now select CPU before creating any JAX arrays.
    # Do not initialize a JAX device or run a synthetic CPU warmup merely
    # because the global GPU flag is enabled; neither action benefits a CPU
    # engine and both consume scarce host memory on WSL/small-RAM machines.
    targets = _iter_warmup_targets(engine, val_engine)
    if not any(hasattr(_resolve_warmup_inner(target), "_data_matrix_jax") for target in targets):
        logger.info("Phase 2 CPU backend selected; skipping JAX runtime warmup.")
        return
    log_gpu_runtime_config()
    try:
        warmup_phase2_gpu_kernels(
            engine, val_engine=val_engine, cluster_id=cluster_id,
        )
    except Exception as exc:
        logger.warning(
            "Phase 2 JAX warmup failed — Generation 1 will pay JIT compile cost: %s",
            exc,
            exc_info=True,
        )

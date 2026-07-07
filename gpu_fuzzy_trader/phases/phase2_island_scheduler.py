"""
phase2_island_scheduler.py — Cluster-island Phase 2 orchestration.

Runs hybrid K-means symbol clusters with fixed total generation budget,
guarded inter-cluster migration, and orphan-symbol mini-runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
from gpu_fuzzy_trader.features.symbol_cluster import (
    SYMBOL_CLUSTERS_PATH,
    build_hybrid_symbol_clusters,
    persist_symbol_clusters,
)
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    Rule_Pool_Generator,
    _get_dont_cares,
    _merge_archive_entries,
    _sample_df,
)
from gpu_fuzzy_trader.phases.phase2_support import (
    passes_evolution_deployability_preview,
    passes_pool_admission_gate,
)

logger = logging.getLogger(__name__)


def compute_cluster_generation_budgets(
    total_gens: int,
    cluster_ids: list[str],
) -> dict[str, int]:
    """Split *total_gens* across *cluster_ids* (base + remainder).

    Each cluster receives ``total_gens // len(cluster_ids)`` generations;
    the first ``total_gens % len(cluster_ids)`` clusters get one extra.

    Parameters
    ----------
    total_gens : int
        Total generation budget across all clusters.
    cluster_ids : list[str]
        Sorted list of cluster identifiers.

    Returns
    -------
    dict[str, int]
        Mapping ``{cluster_id: generation_budget}``.
    """
    k = max(1, len(cluster_ids))
    base = max(1, total_gens // k)
    remainder = total_gens % k
    return {
        cid: base + (1 if idx < remainder else 0)
        for idx, cid in enumerate(cluster_ids)
    }


def _derive_island_seed(base_seed: int | None, island_id: str) -> int | None:
    """Return a deterministic but distinct seed per island from base seed + id.

    Parameters
    ----------
    base_seed : int or None
        The base random seed. If None, returns None.
    island_id : str
        Unique identifier for the island (e.g. cluster ID or 'orphan_SYM').

    Returns
    -------
    int or None
        A derived seed in [0, 2**31), or None when base_seed is None.
    """
    if base_seed is None:
        return None
    h = hashlib.sha256(f"{base_seed}:{island_id}".encode()).digest()
    return int.from_bytes(h[:4], "big") % (2**31)


def _derive_epoch_seed(base_seed: int | None, epoch_idx: int) -> int | None:
    """Return a deterministic per-epoch seed derived from *base_seed* + epoch.

    Used by :func:`_run_cluster_islands` to rotate the train-window start
    bar each epoch so every generation backtests a different contiguous
    sub-window of the training data.

    → fixes audit finding #1 (per-epoch window resampling is dead),
      implements N2

    The derivation mode is controlled by
    :attr:`config.PHASE2_PER_EPOCH_WINDOW_SEED_MODE`:

    * ``"hash_island_epoch"`` (default) — hash(base_seed, epoch_idx)
      via SHA-256, deterministic, no RNG state leak.

    Raises ``ValueError`` for unknown seed-mode values.

    Parameters
    ----------
    base_seed : int or None
        The island's base seed (already derived from global seed + island_id).
        If None, returns None.
    epoch_idx : int
        0-indexed epoch number for this cluster.

    Returns
    -------
    int or None
        A derived seed in [0, 2**31), or None when base_seed is None.
    """
    if base_seed is None:
        return None
    mode = _cfg.PHASE2_PER_EPOCH_WINDOW_SEED_MODE
    if mode == "hash_island_epoch":
        h = hashlib.sha256(f"{base_seed}:epoch_{epoch_idx}".encode()).digest()
        return int.from_bytes(h[:4], "big") % (2**31)
    raise ValueError(
        f"Unknown PHASE2_PER_EPOCH_WINDOW_SEED_MODE={mode!r}. "
        f"Supported modes: 'hash_island_epoch'."
    )


def _load_phase1_feature_lists() -> tuple[list[dict], list[dict]]:
    from gpu_fuzzy_trader.features import selector as sel_mod

    out_long: list[dict] = []
    out_short: list[dict] = []
    for direction, path in sel_mod._DIRECTION_PATHS.items():
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
        feats = list(payload.get("features", []))
        if direction == "long":
            out_long = feats
        else:
            out_short = feats
    return out_long, out_short


def _migrant_to_metrics(
    migrant: dict,
    train_engine: CPUBacktestEngine,
    val_engine: CPUBacktestEngine | None,
    feature_infos: list[dict],
) -> tuple[dict, dict | None]:
    """Backtest one migrant chromosome on receiver cluster engines."""
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _evaluate_chromosome

    chrom = np.asarray(migrant["chromosome"], dtype=np.int32)
    dont_cares = _get_dont_cares(feature_infos)
    pareto_front: list[np.ndarray] = []
    _, metrics = _evaluate_chromosome(
        chrom,
        dont_cares,
        train_engine,
        pareto_front,
        val_engine=val_engine,
        island_hyperparams=getattr(train_engine, "_island_hyperparams", None),
    )
    val_metrics = None
    if val_engine is not None:
        val_list = val_engine.simulate_rule_batch(
            chromosomes=np.asarray([chrom], dtype=np.int32),
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
        val_metrics = val_list[0] if val_list else None
    return metrics, val_metrics


def filter_migrants_for_cluster(
    migrants: list[dict],
    receiver: Rule_Pool_Generator,
    *,
    source_cluster_id: str,
    target_cluster_id: str,
) -> list[dict]:
    """Accept only migrants that pass deployability on the receiver cluster slice."""
    if not migrants:
        return []

    receiver._ensure_engines()
    train_engine = receiver._engine
    val_engine = receiver._val_engine
    hp = receiver.island_hyperparams
    accepted: list[dict] = []

    for migrant in migrants:
        try:
            train_m, val_m = _migrant_to_metrics(
                migrant,
                train_engine,
                val_engine,
                receiver.feature_infos,
            )
        except Exception as exc:
            logger.debug(
                "migration_rejected source=%s target=%s reason=eval_error %s",
                source_cluster_id,
                target_cluster_id,
                exc,
            )
            continue

        if _cfg.PHASE2_MIGRATION_REQUIRE_DEPLOYABILITY:
            if not passes_evolution_deployability_preview(
                train_m,
                val_m,
                island_hyperparams=hp,
            ):
                logger.debug(
                    "migration_rejected source=%s target=%s reason=deployability",
                    source_cluster_id,
                    target_cluster_id,
                )
                continue

        if val_m is not None:
            min_val_ret = float(_cfg.PHASE2_MIGRATION_MIN_VAL_RETURN_PCT)
            if float(val_m.get("total_return_pct", -999.0)) < min_val_ret:
                logger.debug(
                    "migration_rejected source=%s target=%s reason=val_return",
                    source_cluster_id,
                    target_cluster_id,
                )
                continue
            min_trades = (
                int(_cfg.PHASE2_MIGRATION_MIN_VAL_TRADES)
                if _cfg.PHASE2_MIGRATION_MIN_VAL_TRADES is not None
                else (hp.min_trade_pool_floor if hp else _cfg.MIN_TRADE_POOL_FLOOR)
            )
            if int(val_m.get("executed_trades", 0)) < min_trades:
                logger.debug(
                    "migration_rejected source=%s target=%s reason=val_trades",
                    source_cluster_id,
                    target_cluster_id,
                )
                continue

        if not passes_pool_admission_gate(train_m, val_m):
            logger.debug(
                "migration_rejected source=%s target=%s reason=admission",
                source_cluster_id,
                target_cluster_id,
            )
            continue

        accepted.append(migrant)

    if accepted:
        logger.info(
            "Phase 2 migration: cluster %s → %s accepted %d/%d migrants",
            source_cluster_id,
            target_cluster_id,
            len(accepted),
            len(migrants),
        )
    return accepted


def _reference_sample_rows(train_df: pd.DataFrame, seed: int | None) -> int:
    sampled = _sample_df(
        train_df,
        _cfg.PHASE1_SAMPLING_TOTAL,
        random_state=seed if seed is not None else _cfg.PHASE2_SEED,
    )
    return len(sampled)


def _score_rule_on_symbol_val(
    rule: dict,
    val_sym_df: pd.DataFrame,
    direction: str,
) -> dict:
    """Lightweight val-only score for orphan detection."""
    fmt = [{
        "conditions": list(rule.get("conditions", [])),
        "tp": float(rule.get("tp", _cfg.PHASE2_TP)),
        "sl": float(rule.get("sl", _cfg.PHASE2_SL)),
        "capital_pct": float(rule.get("capital_pct", _cfg.PHASE2_CAPITAL_PCT)),
    }]
    try:
        engine = CPUBacktestEngine(
            val_sym_df, {}, direction, fee_pct=_cfg.FEE_PCT,
        )
        metrics = engine.simulate_rule_set(fmt)
        return {
            "return_pct": float(metrics.get("total_return_pct", -999.0)),
            "trades": int(metrics.get("executed_trades", 0)),
        }
    except Exception:
        return {"return_pct": -999.0, "trades": 0}


def _symbol_has_viable_pool_rule(
    pool: list[dict],
    sym: str,
    val_df: pd.DataFrame,
    direction: str,
) -> bool:
    sym_val = val_df[val_df["symbol"].astype(str) == str(sym)]
    if sym_val.empty:
        return True
    min_trades = int(_cfg.PHASE2_ORPHAN_MIN_VAL_TRADES)
    min_ret = float(_cfg.PHASE2_ORPHAN_MIN_VAL_RETURN_PCT)
    for entry in pool:
        score = _score_rule_on_symbol_val(entry, sym_val, direction)
        if score["trades"] >= min_trades and score["return_pct"] >= min_ret:
            return True
    return False


def _run_orphan_boosts(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pool: list[dict],
    feature_infos: list[dict],
    direction: str,
    reference_rows: int,
    seed: int | None,
    cv_folds: list | None,
) -> list[dict]:
    if not _cfg.PHASE2_ORPHAN_ENABLED:
        return pool

    symbols = sorted(train_df["symbol"].dropna().astype(str).unique().tolist())
    orphans = [
        sym for sym in symbols
        if not _symbol_has_viable_pool_rule(pool, sym, val_df, direction)
    ]
    if not orphans:
        return pool

    logger.info(
        "Phase 2 [%s]: orphan boost for symbols %s",
        direction,
        orphans,
    )
    extra: list[dict] = []
    for sym in orphans:
        sym_train = _cfg.filter_df_to_symbols(train_df, [sym])
        sym_val = _cfg.filter_df_to_symbols(val_df, [sym])
        sampled_rows = len(_sample_df(
            sym_train, _cfg.PHASE1_SAMPLING_TOTAL, random_state=seed))
        hp = _cfg.resolve_island_hyperparams(
            "orphan", sampled_rows, reference_rows, n_symbols=1,
        )
        try:
            gen = Rule_Pool_Generator(
                train_df=sym_train,
                val_df=sym_val,
                feature_infos=feature_infos,
                direction=direction,
                pop_size=int(_cfg.PHASE2_ORPHAN_POPULATION_SIZE),
                n_generations=int(_cfg.PHASE2_ORPHAN_GENERATIONS),
                seed=_derive_island_seed(seed, f"{direction}_orphan_{sym}"),
                cv_folds=cv_folds,
                island_id=f"orphan_{sym}",
                source_symbols=[sym],
                island_hyperparams=hp,
                island_profile="orphan",
            )
            n_gens = int(_cfg.PHASE2_ORPHAN_GENERATIONS)
            epoch = int(_cfg.PHASE2_ISLAND_EPOCH_GENERATIONS)
            while gen._island_generations_done < n_gens:
                remaining = n_gens - gen._island_generations_done
                gen.run_epoch(n_generations=min(epoch, remaining))
                gen.park_engines()
            orphan_pool = gen.finalize_island()
        except Exception as exc:
            logger.warning(
                "Phase 2 [%s]: orphan mini-run failed for %s: %s",
                direction,
                sym,
                exc,
            )
            continue
        for row in orphan_pool:
            tagged = dict(row)
            tagged["source_symbols"] = [str(sym)]
            tagged["orphan_boost"] = True
            extra.append(tagged)

    if extra:
        pool = _merge_archive_entries(pool + extra)
    return pool


def _should_skip_epoch(remaining: int) -> bool:
    """Check if an epoch should be skipped due to small remaining budget.

    Engine rebuild takes ~30s (JAX compilation, GPU memory allocation).
    Epochs with fewer than the configured minimum generations provide
    negligible evolutionary benefit relative to the rebuild cost.
    """
    return remaining < int(_cfg.PHASE2_ISLAND_MIN_EPOCH_GENERATIONS)


def _should_migrate_this_round(round_index: int, interval: int) -> bool:
    """Return True if migration should fire on this outer round (1-indexed).

    .. note::
       This helper is **legacy/unused** in the active sequential scheduler.
       ``_run_cluster_islands`` performs **sequential post-cluster chain
       migration** (cluster N → N+1 after cluster N finishes) and does NOT
       use epoch-round interval checking.  The helper is kept for reference
       and for any future round-robin mode.

    Parameters
    ----------
    round_index : int
        The current outer-round count (starts at 0, incremented after each
        ``while`` iteration in ``_run_cluster_islands`` if round-robin mode
        were active).
    interval : int
        The configured migration epoch interval (``PHASE2_MIGRATION_EPOCH_INTERVAL``).
        Values <= 0 are treated as "never migrate".

    Returns
    -------
    bool
        True when migration should be performed this round.
    """
    if interval <= 0:
        return False
    return round_index % interval == 0


def _run_cluster_islands(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_infos: list[dict],
    direction: str,
    cluster_map: dict[str, list[str]],
    reference_rows: int,
    seed: int | None,
    cv_folds: list | None,
) -> list[dict]:
    cluster_ids = sorted(cluster_map.keys(),
                         key=lambda x: int(x) if x.isdigit() else x)
    n_clusters = len(cluster_ids)
    total_gens = int(_cfg.PHASE2_ISLAND_TOTAL_GENERATIONS)
    cluster_budgets = compute_cluster_generation_budgets(total_gens, cluster_ids)
    epoch_gens = int(_cfg.PHASE2_ISLAND_EPOCH_GENERATIONS)

    generators: dict[str, Rule_Pool_Generator] = {}
    for cid in cluster_ids:
        gens_per_cluster = cluster_budgets[cid]
        syms = cluster_map[cid]
        scoped_train = _cfg.filter_df_to_symbols(train_df, syms)
        scoped_val = _cfg.filter_df_to_symbols(val_df, syms)
        sampled_rows = len(
            _sample_df(scoped_train, _cfg.PHASE1_SAMPLING_TOTAL,
                       random_state=seed),
        )
        hp = _cfg.resolve_island_hyperparams(
            "cluster", sampled_rows, reference_rows, n_symbols=len(syms),
        )
        generators[cid] = Rule_Pool_Generator(
            train_df=scoped_train,
            val_df=scoped_val,
            feature_infos=feature_infos,
            direction=direction,
            n_generations=gens_per_cluster,
            seed=_derive_island_seed(seed, f"{direction}_{cid}"),
            cv_folds=cv_folds,
            island_id=cid,
            source_symbols=syms,
            island_hyperparams=hp,
            island_profile="cluster",
            reference_rows=reference_rows,
            defer_warmup=True,
        )

    # ------------------------------------------------------------------
    # Sequential per-cluster processing (Fix 5: sequential cluster warmup)
    #
    # Each cluster is processed one at a time: warm → run epochs → evict.
    # This ensures only 1 cluster's JAX signatures are alive at any time,
    # reducing peak VRAM from ~4 signatures to ~2.
    #
    # Migration: after a cluster finishes, its best individuals are
    # forwarded to the next cluster as seed archives.
    # ------------------------------------------------------------------
    from gpu_fuzzy_trader._gpu_runtime import (
        evict_cluster_signatures,
        warmup_phase2_gpu_kernels,
    )

    cluster_order = list(cluster_ids)
    for idx, cid in enumerate(cluster_order):
        gen = generators[cid]
        gens_per_cluster = cluster_budgets[cid]

        # 1. Warm this cluster's GPU engines before its first epoch
        try:
            warmup_phase2_gpu_kernels(
                gen._engine,
                gen._val_engine,
                cluster_id=cid,
            )
        except Exception as exc:
            logger.debug(
                "Phase 2 [%s]: warmup failed for cluster %s: %s",
                direction, cid, exc,
            )

        # 2. Run all epochs for this cluster sequentially
        epoch_idx = 0
        while gen._island_generations_done < gens_per_cluster:
            remaining = gens_per_cluster - gen._island_generations_done

            # Skip tiny remaining epochs (engine rebuild ~30s with negligible benefit)
            if _should_skip_epoch(remaining):
                min_gens = int(_cfg.PHASE2_ISLAND_MIN_EPOCH_GENERATIONS)
                logger.info(
                    "Phase 2 [%s]: skipping final epoch for cluster %s "
                    "(remaining=%d < PHASE2_ISLAND_MIN_EPOCH_GENERATIONS=%d)",
                    direction, cid, remaining, min_gens,
                )
                gen._island_generations_done = gens_per_cluster  # exit loop cleanly
                break

            # Per-epoch window rotation: re-sample for epoch > 0 so each
            # epoch sees a different contiguous sub-window (Fix 1).
            if epoch_idx > 0 and _cfg.PHASE2_PER_EPOCH_WINDOW_ROTATION:
                gen.resample_train_for_epoch(epoch_idx)

            gen.run_epoch(n_generations=min(epoch_gens, remaining))
            gen.park_engines()
            epoch_idx += 1

        # 3. Evict this cluster's JAX signatures after all epochs are done
        try:
            evicted = evict_cluster_signatures(cluster_id=cid)
            if evicted > 0:
                logger.info(
                    "Phase 2 [%s]: evicted %d signatures for cluster %s",
                    direction, evicted, cid,
                )
        except Exception as exc:
            logger.debug(
                "Phase 2 [%s]: evict_cluster_signatures failed for cluster %s: %s",
                direction, cid, exc,
            )

        # 4. Migration: forward best individuals to the next cluster
        if (
            _cfg.PHASE2_MIGRATION_ENABLED
            and n_clusters > 1
            and idx + 1 < len(cluster_order)
        ):
            next_cid = cluster_order[idx + 1]
            next_gen = generators[next_cid]
            from gpu_fuzzy_trader.evolution.evox_runner import (
                extract_deployable_migrants,
            )

            if gen._evolution_state is not None:
                try:
                    migrants = extract_deployable_migrants(
                        gen._evolution_state,
                        top_k=int(_cfg.PHASE2_MIGRATION_TOP_K),
                    )
                    if migrants:
                        inbound = filter_migrants_for_cluster(
                            migrants[: int(_cfg.PHASE2_MIGRATION_TOP_K)],
                            next_gen,
                            source_cluster_id=cid,
                            target_cluster_id=next_cid,
                        )
                        if inbound:
                            next_gen.set_pending_migrant_seeds(
                                _merge_archive_entries(inbound),
                            )
                except Exception as exc:
                    logger.debug(
                        "Phase 2 [%s]: migration from cluster %s to %s failed: %s",
                        direction, cid, next_cid, exc,
                    )

    # Free cache memory across all cluster generators before finalizing
    # (Fix 2: RAM quick wins — clear global metrics cache between clusters).
    from gpu_fuzzy_trader.evolution.evox_runner import clear_global_metrics_cache

    for cid in cluster_ids:
        gen = generators.get(cid)
        if gen is not None and gen._evolution_state is not None:
            clear_global_metrics_cache(gen._evolution_state.global_metrics_cache)

    cluster_pools: list[dict] = []
    for cid, gen in list(generators.items()):
        pool_part = gen.finalize_island()
        annotated = Rule_Pool_Generator._annotate_archive_entries(
            pool_part,
            source_symbols=cluster_map.get(cid, []),
        )
        cluster_pools.extend(annotated)
        # (Fix 3: RAM quick wins — explicit engine teardown + GC)
        # (Fix 5: sequential cluster warmup — evict JAX signatures for this cluster)
        # Eviction already done in sequential processing above; re-call is safe.
        try:
            from gpu_fuzzy_trader._gpu_runtime import evict_cluster_signatures

            evicted = evict_cluster_signatures(cluster_id=cid)
            if evicted > 0:
                logger.info(
                    "Phase 2 [%s]: evicted %d signatures for cluster %s (cleanup)",
                    direction,
                    evicted,
                    cid,
                )
        except Exception as exc:
            logger.debug(
                "Phase 2 [%s]: evict_cluster_signatures failed for cluster %s: %s",
                direction,
                cid,
                exc,
            )
        del generators[cid]
        import gc as _gc
        _gc.collect()

    return _merge_archive_entries(cluster_pools)


def run_cluster_phase2(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_infos: list[dict],
    direction: str,
    cv_folds: list | None = None,
    seed: int | None = None,
) -> list[dict]:
    """Run cluster-island Phase 2 for one direction and return merged pool."""
    seed = seed if seed is not None else _cfg.PHASE2_SEED
    reference_rows = _reference_sample_rows(train_df, seed)

    feat_long, feat_short = _load_phase1_feature_lists()
    if not feat_long:
        feat_long = feature_infos
    if not feat_short:
        feat_short = feature_infos

    cluster_payload = build_hybrid_symbol_clusters(
        train_df,
        feat_long,
        feat_short,
        n_clusters=int(_cfg.PHASE2_N_CLUSTERS),
        random_state=seed,
    )
    persist_symbol_clusters(SYMBOL_CLUSTERS_PATH, cluster_payload)
    cluster_map: dict[str, list[str]] = cluster_payload["clusters"]

    logger.info(
        "Phase 2 [%s]: cluster island mode K=%d reference_rows=%d",
        direction,
        len(cluster_map),
        reference_rows,
    )
    migration_status = (
        "enabled sequential post-cluster chain"
        if _cfg.PHASE2_MIGRATION_ENABLED
        else "disabled (independent islands)"
    )
    receiver_guards = (
        "deployability+admission+val_return"
        if _cfg.PHASE2_MIGRATION_REQUIRE_DEPLOYABILITY
        else "admission"
    )
    logger.info(
        "Phase 2 [%s]: island mode migration=%s"
        " (top_k=%d, seed_frac=%.2f, receiver_guards=%s)",
        direction,
        migration_status,
        int(_cfg.PHASE2_MIGRATION_TOP_K),
        float(_cfg.PHASE2_MIGRATION_SEED_FRACTION),
        receiver_guards,
    )

    pool = _run_cluster_islands(
        train_df,
        val_df,
        feature_infos,
        direction,
        cluster_map,
        reference_rows,
        seed,
        cv_folds,
    )

    pool = _run_orphan_boosts(
        train_df,
        val_df,
        pool,
        feature_infos,
        direction,
        reference_rows,
        seed,
        cv_folds,
    )

    pool_path = _cfg.phase2_pool_path(direction)
    history_path = _cfg.phase2_history_path(direction)
    os.makedirs(os.path.dirname(pool_path) or ".", exist_ok=True)
    with open(pool_path, "w", encoding="utf-8") as fh:
        json.dump(pool, fh, indent=2)
    with open(history_path, "w", encoding="utf-8") as fh:
        json.dump({"mode": "cluster_islands",
                  "clusters": cluster_map}, fh, indent=2)

    return pool

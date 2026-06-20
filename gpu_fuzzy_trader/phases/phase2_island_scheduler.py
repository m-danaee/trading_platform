"""
phase2_island_scheduler.py — Cluster-island Phase 2 orchestration.

Runs hybrid K-means symbol clusters with fixed total generation budget,
guarded inter-cluster migration, and orphan-symbol mini-runs.
"""

from __future__ import annotations

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
    _filter_pool_by_admission,
    _get_dont_cares,
    _merge_archive_entries,
    _sample_df,
)
from gpu_fuzzy_trader.phases.phase2_support import (
    passes_evolution_deployability_preview,
    passes_pool_admission_gate,
)

logger = logging.getLogger(__name__)


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
                seed=seed,
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
    gens_per_cluster = max(1, total_gens // max(1, n_clusters))
    epoch_gens = int(_cfg.PHASE2_ISLAND_EPOCH_GENERATIONS)

    generators: dict[str, Rule_Pool_Generator] = {}
    for cid in cluster_ids:
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
            seed=seed,
            cv_folds=cv_folds,
            island_id=cid,
            source_symbols=syms,
            island_hyperparams=hp,
            island_profile="cluster",
            reference_rows=reference_rows,
        )

    epoch_counter = 0
    while any(g._island_generations_done < gens_per_cluster for g in generators.values()):
        for cid in cluster_ids:
            gen = generators[cid]
            if gen._island_generations_done >= gens_per_cluster:
                continue
            remaining = gens_per_cluster - gen._island_generations_done
            gen.run_epoch(n_generations=min(epoch_gens, remaining))
            gen.park_engines()
            epoch_counter += 1

        if (
            epoch_counter % int(_cfg.PHASE2_MIGRATION_EPOCH_INTERVAL) == 0
            and n_clusters > 1
        ):
            from gpu_fuzzy_trader.evolution.evox_runner import (
                extract_deployable_migrants,
            )

            migrants_by_source: dict[str, list[dict]] = {}
            for src_id, src_gen in generators.items():
                if src_gen._evolution_state is None:
                    migrants_by_source[src_id] = []
                else:
                    migrants_by_source[src_id] = extract_deployable_migrants(
                        src_gen._evolution_state,
                        top_k=int(_cfg.PHASE2_MIGRATION_TOP_K),
                    )

            for tgt_id, tgt_gen in generators.items():
                inbound: list[dict] = []
                for src_id, migrants in migrants_by_source.items():
                    if src_id == tgt_id:
                        continue
                    inbound.extend(
                        filter_migrants_for_cluster(
                            migrants[: int(_cfg.PHASE2_MIGRATION_TOP_K)],
                            tgt_gen,
                            source_cluster_id=src_id,
                            target_cluster_id=tgt_id,
                        )
                    )
                if inbound:
                    tgt_gen.set_pending_migrant_seeds(
                        _merge_archive_entries(inbound),
                    )

    cluster_pools: list[dict] = []
    for cid, gen in generators.items():
        pool_part = gen.finalize_island()
        annotated = Rule_Pool_Generator._annotate_archive_entries(
            pool_part,
            source_symbols=cluster_map.get(cid, []),
        )
        cluster_pools.extend(annotated)

    merged = _merge_archive_entries(cluster_pools)
    return _filter_pool_by_admission(merged)


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

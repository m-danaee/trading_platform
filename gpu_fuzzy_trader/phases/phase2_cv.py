"""
Phase 2 — purged CV fold engine construction and conservative metric aggregation.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.data.cv_folds import PurgedFold
from gpu_fuzzy_trader.phases.phase2_support import (
    passes_pool_admission_cv_fold,
    passes_pool_admission_gate,
)
from gpu_fuzzy_trader.phases.phase2_rule_pool import (
    Rule_Pool_Generator,
    _sample_df,
)

logger = logging.getLogger(__name__)


def _merge_metrics_worst_case(
    current: dict | None,
    new: dict,
) -> dict:
    """Conservative merge: min return/Sortino/PF, max drawdown, min win rate."""
    if current is None:
        return dict(new)

    out = dict(current)
    ret = float(new.get("total_return_pct", 0.0))
    out["total_return_pct"] = min(
        float(out.get("total_return_pct", 0.0)), ret)

    sortino = float(new.get(
        "sortino_ratio", new.get("total_return_pct", 0.0)))
    out["sortino_ratio"] = min(
        float(out.get("sortino_ratio", out.get("total_return_pct", 0.0))),
        sortino,
    )

    dd = float(new.get("max_drawdown_pct", 0.0))
    out["max_drawdown_pct"] = max(
        float(out.get("max_drawdown_pct", 0.0)), dd)

    wr = float(new.get("win_rate", 0.0))
    out["win_rate"] = min(float(out.get("win_rate", 0.0)), wr)

    pf = float(new.get("profit_factor", 0.0))
    out["profit_factor"] = min(float(out.get("profit_factor", pf)), pf)

    trades = int(new.get("executed_trades", 0))
    out["executed_trades"] = min(
        int(out.get("executed_trades", trades)), trades)

    # Merge regime stats element-wise (worst-case minimums)
    for k in ("regime_net_pnl", "regime_trade_counts", "regime_win_counts"):
        if k in new:
            new_val = list(new[k])
            if k in out:
                out[k] = [min(c, n) for c, n in zip(out[k], new_val)]
            else:
                out[k] = new_val

    return out


class _FoldBacktestEngine:
    """Delegates simulate_rule_batch to an inner engine (GPU or CPU)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        for attr in ("_regime_row_fractions", "_regime_row_counts", "backend"):
            if hasattr(inner, attr):
                setattr(self, attr, getattr(inner, attr))

    def simulate_rule_batch(self, **kwargs: Any) -> list[dict]:
        return self._inner.simulate_rule_batch(**kwargs)


def _n_fold_workers(n_folds: int) -> int:
    """Resolve the number of parallel fold-evaluation threads."""
    cfg_val = int(_cfg.PHASE2_CV_FOLD_WORKERS)
    return n_folds if cfg_val <= 0 else min(cfg_val, n_folds)


class PurgedCVTrainEngine:
    """
    Train-side facade: batch metrics are worst-case across CV train folds.

    Folds are evaluated in parallel via ThreadPoolExecutor (see
    ``PHASE2_CV_FOLD_WORKERS`` in config). JAX releases the GIL during GPU
    dispatch so multiple folds can overlap on the XLA stream; NumPy/CPU folds
    also benefit from overlapping Python overhead.

    Used as ``engine`` in Phase 2 evolution when CV is enabled.
    """

    def __init__(self, fold_engines: list[Any]) -> None:
        self._fold_engines = fold_engines
        if fold_engines:
            fr = getattr(fold_engines[0], "_regime_row_fractions", None)
            self._regime_row_fractions = fr

    def simulate_rule_batch(self, **kwargs: Any) -> list[dict]:
        if not self._fold_engines:
            raise RuntimeError("PurgedCVTrainEngine has no fold engines")

        n_folds = len(self._fold_engines)
        n_workers = _n_fold_workers(n_folds)

        if n_workers <= 1 or n_folds == 1:
            # Sequential path — safe for any backend.
            merged: list[dict | None] = [None] * 0
            for eng in self._fold_engines:
                batch = eng.simulate_rule_batch(**kwargs)
                if not merged:
                    merged = [None] * len(batch)
                for j, met in enumerate(batch):
                    merged[j] = _merge_metrics_worst_case(merged[j], met)
            return [m if m is not None else {} for m in merged]

        # Parallel path — each fold dispatched to its own thread.
        try:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(eng.simulate_rule_batch, **kwargs): idx
                    for idx, eng in enumerate(self._fold_engines)
                }
                batches: list[list[dict]] = [None] * n_folds  # type: ignore[list-item]
                for fut in as_completed(futures):
                    batches[futures[fut]] = fut.result()
        except Exception:
            logger.debug(
                "PurgedCVTrainEngine parallel eval failed; retrying sequentially",
                exc_info=True,
            )
            return self._simulate_rule_batch_sequential(**kwargs)

        merged2: list[dict | None] = [None] * len(batches[0])
        for batch in batches:
            for j, met in enumerate(batch):
                merged2[j] = _merge_metrics_worst_case(merged2[j], met)
        return [m if m is not None else {} for m in merged2]

    def _simulate_rule_batch_sequential(self, **kwargs: Any) -> list[dict]:
        merged: list[dict | None] = [None] * 0
        for eng in self._fold_engines:
            batch = eng.simulate_rule_batch(**kwargs)
            if not merged:
                merged = [None] * len(batch)
            for j, met in enumerate(batch):
                merged[j] = _merge_metrics_worst_case(merged[j], met)
        return [m if m is not None else {} for m in merged]


class PurgedCVValEngine:
    """
    Val-side facade: worst-case metrics across CV validation folds.

    Folds are evaluated in parallel via ThreadPoolExecutor (see
    ``PHASE2_CV_FOLD_WORKERS`` in config).
    """

    def __init__(self, fold_engines: list[Any]) -> None:
        self._fold_engines = fold_engines
        if fold_engines:
            rc = getattr(fold_engines[0], "_regime_row_counts", None)
            self._regime_row_counts = rc

    def simulate_rule_batch(self, **kwargs: Any) -> list[dict]:
        if not self._fold_engines:
            raise RuntimeError("PurgedCVValEngine has no fold engines")

        n_folds = len(self._fold_engines)
        n_workers = _n_fold_workers(n_folds)

        if n_workers <= 1 or n_folds == 1:
            merged: list[dict | None] = [None] * 0
            for eng in self._fold_engines:
                batch = eng.simulate_rule_batch(**kwargs)
                if not merged:
                    merged = [None] * len(batch)
                for j, met in enumerate(batch):
                    merged[j] = _merge_metrics_worst_case(merged[j], met)
            return [m if m is not None else {} for m in merged]

        try:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(eng.simulate_rule_batch, **kwargs): idx
                    for idx, eng in enumerate(self._fold_engines)
                }
                batches: list[list[dict]] = [None] * n_folds  # type: ignore[list-item]
                for fut in as_completed(futures):
                    batches[futures[fut]] = fut.result()
        except Exception:
            logger.debug(
                "PurgedCVValEngine parallel eval failed; retrying sequentially",
                exc_info=True,
            )
            return self._simulate_rule_batch_sequential(**kwargs)

        merged2: list[dict | None] = [None] * len(batches[0])
        for batch in batches:
            for j, met in enumerate(batch):
                merged2[j] = _merge_metrics_worst_case(merged2[j], met)
        return [m if m is not None else {} for m in merged2]

    def _simulate_rule_batch_sequential(self, **kwargs: Any) -> list[dict]:
        merged: list[dict | None] = [None] * 0
        for eng in self._fold_engines:
            batch = eng.simulate_rule_batch(**kwargs)
            if not merged:
                merged = [None] * len(batch)
            for j, met in enumerate(batch):
                merged[j] = _merge_metrics_worst_case(merged[j], met)
        return [m if m is not None else {} for m in merged]


def build_cv_fold_engines(
    folds: list[PurgedFold],
    feature_infos: list[dict],
    direction: str,
    *,
    seed: int | None,
    builder: Rule_Pool_Generator,
) -> tuple[PurgedCVTrainEngine | None, PurgedCVValEngine | None]:
    """
    Build purged CV train/val engine facades for Phase 2.

    Returns (None, None) when *folds* is empty.
    """
    if not folds:
        return None, None

    feature_names = [fi["name"] for fi in feature_infos]
    from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df
    from gpu_fuzzy_trader.phases.phase2_rule_pool import _prepare_regime_context

    train_engines: list[Any] = []
    val_engines: list[Any] = []

    sample_seed = seed if seed is not None else _cfg.PHASE2_SEED

    for fold in folds:
        train_sampled = _sample_df(
            fold.train_df,
            _cfg.PHASE1_SAMPLING_TOTAL,
            random_state=sample_seed,
        )
        val_sampled = _sample_df(
            fold.val_df,
            _cfg.PHASE1_SAMPLING_TOTAL,
            random_state=sample_seed,
        )
        train_regime_ids, _, n_regimes = _prepare_regime_context(train_sampled)
        val_regime_ids, _, val_n_regimes = _prepare_regime_context(val_sampled)

        slim_train = slim_backtest_df(train_sampled, feature_names)
        slim_val = slim_backtest_df(val_sampled, feature_names)

        train_eng = builder._build_engine_for_df(
            slim_train,
            regime_ids=train_regime_ids,
            n_regimes=n_regimes,
        )
        val_eng = builder._build_engine_for_df(
            slim_val,
            regime_ids=val_regime_ids,
            n_regimes=val_n_regimes,
        )
        if val_regime_ids is not None:
            val_eng._regime_row_counts = np.bincount(
                val_regime_ids.astype(np.int64),
                minlength=val_n_regimes,
            ).astype(np.int64)

        train_engines.append(_FoldBacktestEngine(train_eng))
        val_engines.append(_FoldBacktestEngine(val_eng))

    logger.info(
        "Phase 2 [%s]: purged CV engines built (%d folds, embargo=%d bars)",
        direction,
        len(folds),
        _cfg.CV_EMBARGO_BARS,
    )
    return PurgedCVTrainEngine(train_engines), PurgedCVValEngine(val_engines)


def _simulate_one_chrom(
    engine: Any,
    chrom: np.ndarray,
) -> dict:
    batch = engine.simulate_rule_batch(
        chromosomes=chrom[None, :],
        tp=_cfg.PHASE2_TP,
        sl=_cfg.PHASE2_SL,
        capital_pct=_cfg.PHASE2_CAPITAL_PCT,
    )
    return batch[0] if batch else {}


def evaluate_purged_cv_pool_admission(
    train_cv: PurgedCVTrainEngine,
    val_cv: PurgedCVValEngine,
    chrom: np.ndarray,
) -> tuple[bool, dict, dict | None, int]:
    """
    Per-fold pool admission for purged CV.

    A rule passes when at least ``PHASE2_CV_POOL_MIN_FOLDS_PASS`` folds satisfy
    ``passes_pool_admission_cv_fold``. Returned train/val metrics are the
    worst-case merge (same as evolution facades) for pool JSON storage.

    The fourth return value is the number of folds that passed; persist it on
    pool entries so post-merge filters do not reject rules on negative merged
    metrics alone.
    """
    train_folds = train_cv._fold_engines
    val_folds = val_cv._fold_engines
    if not train_folds or len(train_folds) != len(val_folds):
        return False, {}, None, 0

    folds_passing = 0
    last_fold_val_ret = 0.0
    for idx, (train_eng, val_eng) in enumerate(zip(train_folds, val_folds)):
        try:
            train_m = _simulate_one_chrom(train_eng, chrom)
            val_m = _simulate_one_chrom(val_eng, chrom)
        except Exception:
            continue
        if passes_pool_admission_cv_fold(train_m, val_m):
            folds_passing += 1
        if idx == len(train_folds) - 1:
            last_fold_val_ret = float(val_m.get("total_return_pct", 0.0))

    min_pass = min(
        int(_cfg.PHASE2_CV_POOL_MIN_FOLDS_PASS),
        len(train_folds),
    )
    admitted = folds_passing >= min_pass
    if admitted and _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE:
        if last_fold_val_ret <= 0.0:
            admitted = False

    try:
        merged_train = _simulate_one_chrom(train_cv, chrom)
        merged_val = _simulate_one_chrom(val_cv, chrom)
    except Exception:
        merged_train, merged_val = {}, None

    if admitted and merged_val is not None and bool(_cfg.PHASE2_CV_MERGED_GATE_HARD):
        if not passes_pool_admission_gate(merged_train, merged_val):
            admitted = False

    return admitted, merged_train, merged_val, folds_passing


def evaluate_purged_cv_pool_admission_batch(
    train_cv: PurgedCVTrainEngine,
    val_cv: PurgedCVValEngine,
    chroms: np.ndarray,
    direction: str = "",
) -> list[tuple[bool, dict, dict | None, int]]:
    """
    Batched CV pool admission: evaluate *all* chroms in one
    ``simulate_rule_batch`` call per fold instead of one call per chromosome.

    This eliminates the silent hang after gen 80/80 where N×(2×n_folds+2)
    individual backtest calls were made without any progress logging.

    Returns a list of ``(admitted, merged_train, merged_val, folds_passing)``
    tuples in the same order as *chroms* — same contract per entry as
    ``evaluate_purged_cv_pool_admission``.
    """
    import time as _time

    train_folds = train_cv._fold_engines
    val_folds = val_cv._fold_engines
    n = len(chroms)
    if not train_folds or len(train_folds) != len(val_folds) or n == 0:
        return [(False, {}, None, 0)] * n

    min_pass = min(int(_cfg.PHASE2_CV_POOL_MIN_FOLDS_PASS), len(train_folds))
    folds_passing_arr = np.zeros(n, dtype=np.int32)
    last_fold_val_ret = np.zeros(n, dtype=np.float64)

    tag = f"Phase 2 [{direction}] CV admission" if direction else "Phase 2 CV admission"
    t0 = _time.monotonic()
    logger.info(
        "%s: batched evaluation of %d archive chromosomes (%d folds)",
        tag, n, len(train_folds),
    )

    # One batch call per fold pair — avoids N×6 individual backtest calls.
    for fold_idx, (train_eng, val_eng) in enumerate(zip(train_folds, val_folds)):
        try:
            train_batch = train_eng.simulate_rule_batch(
                chromosomes=chroms,
                tp=_cfg.PHASE2_TP,
                sl=_cfg.PHASE2_SL,
                capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            )
            val_batch = val_eng.simulate_rule_batch(
                chromosomes=chroms,
                tp=_cfg.PHASE2_TP,
                sl=_cfg.PHASE2_SL,
                capital_pct=_cfg.PHASE2_CAPITAL_PCT,
            )
        except Exception as exc:
            logger.debug("%s: fold %d batch eval failed: %s", tag, fold_idx, exc)
            continue
        for i in range(n):
            t_m = train_batch[i] if i < len(train_batch) else {}
            v_m = val_batch[i] if i < len(val_batch) else {}
            if passes_pool_admission_cv_fold(t_m, v_m):
                folds_passing_arr[i] += 1
            if fold_idx == len(train_folds) - 1:
                last_fold_val_ret[i] = float(v_m.get("total_return_pct", 0.0))
        logger.info(
            "%s: fold %d/%d done — elapsed=%.1fs",
            tag, fold_idx + 1, len(train_folds), _time.monotonic() - t0,
        )

    # Merged worst-case across all folds — one batch call each.
    try:
        merged_train_batch = train_cv.simulate_rule_batch(
            chromosomes=chroms,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
    except Exception:
        merged_train_batch = [{} for _ in range(n)]
    try:
        merged_val_batch = val_cv.simulate_rule_batch(
            chromosomes=chroms,
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
    except Exception:
        merged_val_batch = [None for _ in range(n)]

    results: list[tuple[bool, dict, dict | None, int]] = []
    for i in range(n):
        fp = int(folds_passing_arr[i])
        admitted = fp >= min_pass
        if admitted and _cfg.PHASE2_REQUIRE_LAST_FOLD_POSITIVE:
            if last_fold_val_ret[i] <= 0.0:
                admitted = False
        m_train = merged_train_batch[i] if i < len(merged_train_batch) else {}
        m_val = merged_val_batch[i] if i < len(merged_val_batch) else None
        if admitted and m_val is not None and bool(_cfg.PHASE2_CV_MERGED_GATE_HARD):
            if not passes_pool_admission_gate(m_train, m_val):
                admitted = False
        results.append((admitted, m_train, m_val, fp))

    n_admitted = sum(1 for r in results if r[0])
    logger.info(
        "%s: batch complete — %d/%d admitted in %.1fs",
        tag, n_admitted, n, _time.monotonic() - t0,
    )
    return results

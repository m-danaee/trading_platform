"""
phase2_rule_pool.py — Rule_Pool_Generator (Phase 2)

GPU-accelerated multi-objective evolutionary search for fuzzy trading rules.

Uses NSGA-III via EvoX when available; falls back to NumPy NSGA-II when EvoX
is not installed.

Chromosome encoding:
    chromosome = [gene_0, gene_1, ..., gene_{K-1}]
    gene_i ∈ {0, ..., num_classes_i - 1, dont_care_i}
    dont_care_i = num_classes_i  (inactive condition)

Three objectives (all minimised):
    f1 = -sortino_ratio
    f2 = max_drawdown_pct
    f3 = -win_rate

Penalties:
    support_penalty        — if executed_trades < MIN_TRADE_SUPPORT
    diversity_penalty      — Hamming distance in chromosome space
    condition_count_penalty — active conditions outside [MIN_CONDITIONS, MAX_CONDITIONS]

Static risk parameters during Phase 2:
    TP = PHASE2_TP, SL = PHASE2_SL, capital_pct = PHASE2_CAPITAL_PCT
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Optional

import numpy as np
import pandas as pd

from gpu_fuzzy_trader import config as _cfg
from gpu_fuzzy_trader.features.encoder import decode_chromosome, get_dont_care
from gpu_fuzzy_trader.log_progress import maybe_log_generation
from gpu_fuzzy_trader.reporting.reporter import Reporter

logger = logging.getLogger(__name__)


def trade_support_penalty(executed: int) -> float:
    """Graduated penalty when executed trades fall below MIN_TRADE_SUPPORT."""
    if executed >= _cfg.MIN_TRADE_SUPPORT:
        return 0.0
    shortfall = (_cfg.MIN_TRADE_SUPPORT - executed) / _cfg.MIN_TRADE_SUPPORT
    return min(shortfall ** 2 * _cfg.SUPPORT_PENALTY_MAX, _cfg.SUPPORT_PENALTY_MAX)


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_POOL_PATHS = {
    "long": _cfg.PHASE2_POOL_PATHS["long"],
    "short": _cfg.PHASE2_POOL_PATHS["short"],
}
_HISTORY_PATHS = {
    "long": _cfg.PHASE2_HISTORY_PATHS["long"],
    "short": _cfg.PHASE2_HISTORY_PATHS["short"],
}
_ARCHIVE_PATHS = dict(_cfg.PHASE2_ARCHIVE_PATHS)

# ---------------------------------------------------------------------------
# EvoX optional import
# ---------------------------------------------------------------------------

try:
    from evox.core import Algorithm as EvoxAlgorithm  # type: ignore
    _EVOX_AVAILABLE = True
except ImportError:
    EvoxAlgorithm = None  # type: ignore[misc, assignment]
    _EVOX_AVAILABLE = False

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_dont_cares(feature_infos: list[dict]) -> np.ndarray:
    """Return array of dont_care sentinels for each feature."""
    return np.array([get_dont_care(fi["mode"]) for fi in feature_infos], dtype=np.int32)


def _count_active_conditions(chromosome: np.ndarray, dont_cares: np.ndarray) -> int:
    """Count genes that are NOT dont_care (i.e. active conditions)."""
    return int(np.sum(chromosome != dont_cares))


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Hamming distance between two integer arrays."""
    return int(np.sum(a != b))


def _pareto_sortino_stats(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> dict[str, float]:
    """Aggregate raw Sortino Ratio over the current Pareto front."""
    if not pareto_indices:
        return {"mean_sortino_ratio": 0.0, "best_sortino_ratio": 0.0}
    returns = [
        float(metrics_cache[i].get("sortino_ratio",
              metrics_cache[i].get("total_return_pct", 0.0)))
        for i in pareto_indices
    ]
    return {
        "mean_sortino_ratio": float(np.mean(returns)),
        "best_sortino_ratio": float(np.max(returns)),
    }


_pareto_return_stats = _pareto_sortino_stats


def _sample_df(df: pd.DataFrame, total_rows: int) -> pd.DataFrame:
    """
    Sample up to *total_rows* rows from *df*, distributed equally across symbols.

    If a symbol has fewer rows than its share, all its rows are used.
    """
    if "symbol" not in df.columns:
        return df.sample(n=min(total_rows, len(df)), random_state=42)

    symbols = df["symbol"].unique()
    n_sym = len(symbols)
    rows_per_sym = max(1, total_rows // n_sym)

    parts = []
    for sym in symbols:
        sym_df = df[df["symbol"] == sym]
        n = min(rows_per_sym, len(sym_df))
        parts.append(sym_df.sample(n=n, random_state=42))

    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------

def _evaluate_chromosome(
    chromosome: np.ndarray,
    dont_cares: np.ndarray,
    engine,  # GPUBacktestEngine or CPUBacktestEngine
    pareto_front: list[np.ndarray],
) -> tuple[np.ndarray, dict]:
    """
    Evaluate a single chromosome and return (objectives, metrics).

    objectives = [f1, f2, f3] (all minimised, with penalties applied).
    """
    active = _count_active_conditions(chromosome, dont_cares)

    # Condition count penalty
    cond_penalty = 0.0
    if active < _cfg.MIN_CONDITIONS:
        cond_penalty = (_cfg.MIN_CONDITIONS - active) * 10.0
    elif active > _cfg.MAX_CONDITIONS:
        cond_penalty = (active - _cfg.MAX_CONDITIONS) * 10.0

    # Evaluate via backtest engine
    try:
        metrics_list = engine.simulate_rule_batch(
            chromosomes=chromosome[None, :],
            tp=_cfg.PHASE2_TP,
            sl=_cfg.PHASE2_SL,
            capital_pct=_cfg.PHASE2_CAPITAL_PCT,
        )
        metrics = metrics_list[0]
    except Exception as exc:
        logger.debug("simulate_rule_batch failed: %s", exc)
        metrics = {
            "sortino_ratio": 0.0,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 100.0,
            "win_rate": 0.0,
            "executed_trades": 0,
        }

    sortino_ratio = float(metrics.get(
        "sortino_ratio", metrics.get("total_return_pct", 0.0)))
    max_dd = float(metrics.get("max_drawdown_pct", 100.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    executed = int(metrics.get("executed_trades", 0))

    support_penalty = trade_support_penalty(executed)

    # Diversity penalty: Hamming distance to nearest Pareto-front member
    diversity_penalty = 0.0
    if pareto_front:
        min_hamming = min(_hamming_distance(chromosome, pf)
                          for pf in pareto_front)
        if min_hamming == 0:
            diversity_penalty = 5.0  # identical rule already in front

    f1 = -sortino_ratio + support_penalty + diversity_penalty + cond_penalty
    f2 = max_dd + support_penalty + diversity_penalty + cond_penalty
    f3 = -win_rate + support_penalty + diversity_penalty + cond_penalty

    objectives = np.array([f1, f2, f3], dtype=np.float64)
    return objectives, metrics


# ---------------------------------------------------------------------------
# NSGA-II helpers (non-dominated sorting + crowding distance)
# ---------------------------------------------------------------------------

def _dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Return True if solution *a* dominates *b* (all objectives ≤, at least one <)."""
    return bool(np.all(a <= b) and np.any(a < b))


def _non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """
    NSGA-II non-dominated sorting.

    Parameters
    ----------
    objectives : np.ndarray
        Shape (N, M) — N solutions, M objectives (all minimised).

    Returns
    -------
    list[list[int]]
        Fronts in order: fronts[0] is the Pareto front, fronts[1] is the
        second front, etc.  Always returns at least one (possibly empty) front.
    """
    n = len(objectives)
    if n == 0:
        return [[]]

    domination_count = np.zeros(n, dtype=int)
    dominated_by: list[list[int]] = [[] for _ in range(n)]
    first_front: list[int] = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(objectives[i], objectives[j]):
                dominated_by[i].append(j)
            elif _dominates(objectives[j], objectives[i]):
                domination_count[i] += 1
        if domination_count[i] == 0:
            first_front.append(i)

    fronts: list[list[int]] = [first_front]
    current_front = 0

    while fronts[current_front]:
        next_front: list[int] = []
        for i in fronts[current_front]:
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
        else:
            break

    return fronts


def _crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """
    Compute crowding distance for solutions in *front*.

    Parameters
    ----------
    objectives : np.ndarray
        Shape (N, M).
    front : list[int]
        Indices of solutions in this front.

    Returns
    -------
    np.ndarray
        Shape (len(front),) crowding distances.
    """
    n = len(front)
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    front_obj = objectives[front]  # (n, M)
    M = front_obj.shape[1]

    for m in range(M):
        order = np.argsort(front_obj[:, m])
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        obj_range = front_obj[order[-1], m] - front_obj[order[0], m]
        if obj_range == 0:
            continue
        for k in range(1, n - 1):
            distances[order[k]] += (
                front_obj[order[k + 1], m] - front_obj[order[k - 1], m]
            ) / obj_range

    return distances


# ---------------------------------------------------------------------------
# Population initialisation
# ---------------------------------------------------------------------------

def _init_population(
    pop_size: int,
    feature_infos: list[dict],
    rng: np.random.Generator,
    dont_care_prob: float = 0.5,
    seeded_chromosomes: np.ndarray | list[np.ndarray] | None = None,
    seed_fraction: float | None = None,
) -> np.ndarray:
    """
    Initialise a population of chromosomes.

    Each gene is either a random valid class index or the dont_care sentinel,
    chosen with probability *dont_care_prob*.

    Parameters
    ----------
    pop_size : int
    feature_infos : list[dict]
        Each dict must have "mode" key.
    rng : np.random.Generator
    dont_care_prob : float
        Probability that a gene is set to dont_care (inactive).
    seeded_chromosomes : np.ndarray | list[np.ndarray] | None
        Optional archive chromosomes to seed into the initial population.
    seed_fraction : float | None
        Fraction of the population to seed from *seeded_chromosomes*.

    Returns
    -------
    np.ndarray
        Shape (pop_size, K) int32.
    """
    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    population = np.zeros((pop_size, K), dtype=np.int32)
    if seed_fraction is None:
        seed_fraction = _cfg.PHASE2_ARCHIVE_SEED_FRACTION

    seed_rows: list[np.ndarray] = []
    if seeded_chromosomes is not None:
        seed_array = np.asarray(seeded_chromosomes, dtype=np.int32)
        if seed_array.ndim == 1:
            seed_array = seed_array[None, :]
        if seed_array.ndim != 2:
            raise ValueError(
                "seeded_chromosomes must be a 1D or 2D array-like value")
        if seed_array.shape[1] != K:
            raise ValueError(
                f"seeded_chromosomes must have {K} genes per chromosome, got {seed_array.shape[1]}"
            )

        seen: set[tuple[int, ...]] = set()
        for row in seed_array:
            key = tuple(int(v) for v in row.tolist())
            if key in seen:
                continue
            seen.add(key)
            repaired = row.astype(np.int32, copy=True)
            for k, dc in enumerate(dont_cares):
                gene = int(repaired[k])
                if gene < 0:
                    repaired[k] = 0
                elif gene > int(dc):
                    repaired[k] = int(dc)
            seed_rows.append(repaired)

    seed_count = 0
    if seed_rows and pop_size > 0 and seed_fraction > 0:
        seed_count = min(
            pop_size,
            max(1, int(round(pop_size * seed_fraction))),
            len(seed_rows),
        )

    seeded_mask = np.zeros(pop_size, dtype=bool)
    if seed_count > 0:
        seed_positions = rng.choice(pop_size, size=seed_count, replace=False)
        for position, chrom in zip(seed_positions, seed_rows[:seed_count]):
            population[position] = chrom
        seeded_mask[seed_positions] = True

    for k, fi in enumerate(feature_infos):
        dc = dont_cares[k]
        num_classes = dc  # dont_care = num_classes
        for i in np.where(~seeded_mask)[0]:
            if rng.random() < dont_care_prob:
                population[i, k] = dc
            else:
                population[i, k] = int(rng.integers(0, num_classes))

    return population


# ---------------------------------------------------------------------------
# Crossover and mutation
# ---------------------------------------------------------------------------

def _crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Uniform crossover: each gene independently chosen from either parent."""
    K = len(parent_a)
    mask = rng.random(K) < 0.5
    child_a = np.where(mask, parent_a, parent_b).astype(np.int32)
    child_b = np.where(mask, parent_b, parent_a).astype(np.int32)
    return child_a, child_b


def _mutate(
    chromosome: np.ndarray,
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    rng: np.random.Generator,
    mutation_rate: float = 0.1,
) -> np.ndarray:
    """
    Mutate a chromosome in-place (returns a copy).

    Each gene is mutated with probability *mutation_rate*:
      - If currently dont_care → random valid class index
      - If currently active → dont_care or a different valid class index
    """
    child = chromosome.copy()
    K = len(child)
    for k in range(K):
        if rng.random() < mutation_rate:
            dc = dont_cares[k]
            num_classes = dc
            if child[k] == dc:
                # Activate: pick a random class
                child[k] = int(rng.integers(0, num_classes))
            else:
                # Deactivate or change class
                if rng.random() < 0.3:
                    child[k] = dc
                else:
                    child[k] = int(rng.integers(0, num_classes))
    return child


# ---------------------------------------------------------------------------
# Fallback evolutionary algorithm (NSGA-II style)
# ---------------------------------------------------------------------------

def _metrics_dict_from_population(
    population: np.ndarray,
    metrics_cache: list[dict],
) -> dict[tuple, dict]:
    """Map chromosome tuples to cached fitness metrics."""
    out: dict[tuple, dict] = {}
    for i, met in enumerate(metrics_cache):
        if met:
            out[tuple(population[i].tolist())] = met
    return out


# ---------------------------------------------------------------------------
# Pool construction
# ---------------------------------------------------------------------------

def _build_pool_from_archive(
    archive: list[np.ndarray],
    feature_infos: list[dict],
    dont_cares: np.ndarray,
    engine,
    metrics_by_chrom: dict[tuple, dict] | None = None,
) -> list[dict]:
    """
    Convert a list of Pareto-front chromosomes into pool JSON entries.

    Each entry:
    {
        "chromosome": [...],
        "conditions": [...],
        "objectives": {"sortino_ratio": ..., "max_drawdown_pct": ..., "win_rate": ...},
        "executed_trades": ...
    }
    """
    pool: list[dict] = []
    seen: set[tuple] = set()

    for chrom in archive:
        key = tuple(chrom.tolist())
        if key in seen:
            continue
        seen.add(key)

        active = _count_active_conditions(chrom, dont_cares)
        if active < _cfg.MIN_CONDITIONS or active > _cfg.MAX_CONDITIONS:
            continue

        metrics = None
        if metrics_by_chrom is not None:
            metrics = metrics_by_chrom.get(key)
        if metrics is None:
            try:
                metrics_list = engine.simulate_rule_batch(
                    chromosomes=chrom[None, :],
                    tp=_cfg.PHASE2_TP,
                    sl=_cfg.PHASE2_SL,
                    capital_pct=_cfg.PHASE2_CAPITAL_PCT,
                )
                metrics = metrics_list[0]
            except Exception:
                continue

        executed = int(metrics.get("executed_trades", 0))
        if executed < _cfg.MIN_TRADE_POOL_FLOOR:
            continue

        try:
            conditions = decode_chromosome(chrom, feature_infos)
        except Exception:
            continue

        if not conditions:
            continue

        pool.append({
            "chromosome": chrom.tolist(),
            "conditions": conditions,
            "objectives": {
                "sortino_ratio": float(metrics.get("sortino_ratio", metrics.get("total_return_pct", 0.0))),
                "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
            },
            "executed_trades": executed,
        })

    return pool


def _archive_feature_signature(feature_infos: list[dict]) -> list[dict[str, str]]:
    """Return the ordered feature signature used to validate archive reuse."""
    return [
        {"name": fi["name"], "mode": fi["mode"]}
        for fi in feature_infos
    ]


def _read_json_payload(path: str) -> object | None:
    """Read JSON from *path* and return None when the file cannot be loaded."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _archive_objective_vector(entry: dict) -> np.ndarray:
    """Convert an archive entry into the minimisation objectives used for ranking."""
    objectives = entry.get("objectives", {})
    sortino_ratio = float(objectives.get(
        "sortino_ratio", objectives.get("total_return_pct", 0.0)))
    return np.array(
        [
            -sortino_ratio,
            float(objectives.get("max_drawdown_pct", 0.0)),
            -float(objectives.get("win_rate", 0.0)),
        ],
        dtype=np.float64,
    )


def _is_better_archive_entry(candidate: dict, incumbent: dict) -> bool:
    """Return True when *candidate* should replace *incumbent* for the same chromosome."""
    candidate_vec = _archive_objective_vector(candidate)
    incumbent_vec = _archive_objective_vector(incumbent)
    if _dominates(candidate_vec, incumbent_vec):
        return True
    if _dominates(incumbent_vec, candidate_vec):
        return False
    return tuple(candidate_vec.tolist()) < tuple(incumbent_vec.tolist())


def _merge_archive_entries(
    entries: list[dict],
    max_size: int = _cfg.PHASE2_ARCHIVE_MAX_SIZE,
) -> list[dict]:
    """Deduplicate and rank archive entries, keeping the best *max_size* rules."""
    if not entries:
        return []

    deduped: dict[tuple, dict] = {}
    for entry in entries:
        key = tuple(entry["chromosome"])
        current = deduped.get(key)
        if current is None or _is_better_archive_entry(entry, current):
            deduped[key] = entry

    unique_entries = list(deduped.values())
    if not unique_entries:
        return []

    objectives = np.vstack([
        _archive_objective_vector(entry) for entry in unique_entries
    ])
    fronts = _non_dominated_sort(objectives)

    selected: list[int] = []
    for front in fronts:
        if not front:
            continue
        if len(selected) + len(front) <= max_size:
            selected.extend(front)
        else:
            crowding = _crowding_distance(objectives, front)
            order = np.argsort(-crowding)
            need = max_size - len(selected)
            selected.extend(int(front[j]) for j in order[:need])
            break

    return [unique_entries[i] for i in selected[:max_size]]


def _validate_archive_payload(
    payload: object,
    path: str,
    feature_infos: list[dict],
) -> None:
    """Validate the archive JSON structure and feature compatibility."""
    if not isinstance(payload, dict):
        raise ValueError(
            f"Phase 2 archive must be a JSON object, got {type(payload).__name__}: {path}"
        )

    required_keys = {"version", "direction", "feature_signature", "rules"}
    missing = required_keys - set(payload.keys())
    if missing:
        raise ValueError(f"Phase 2 archive missing keys {missing}: {path}")

    if payload["direction"] not in _ARCHIVE_PATHS:
        raise ValueError(
            f"Phase 2 archive has invalid direction {payload['direction']!r}: {path}"
        )

    expected_signature = _archive_feature_signature(feature_infos)
    if payload["feature_signature"] != expected_signature:
        raise ValueError(f"Phase 2 archive feature signature mismatch: {path}")

    if not isinstance(payload["rules"], list):
        raise ValueError(f"Phase 2 archive 'rules' must be a list: {path}")

    _validate_pool_schema(payload["rules"], path)

    dont_cares = _get_dont_cares(feature_infos)
    for i, entry in enumerate(payload["rules"]):
        chromosome = entry["chromosome"]
        if len(chromosome) != len(feature_infos):
            raise ValueError(
                f"Phase 2 archive entry {i} chromosome length mismatch: {path}"
            )
        for j, gene in enumerate(chromosome):
            if not isinstance(gene, (int, np.integer)):
                raise ValueError(
                    f"Phase 2 archive entry {i} gene {j} must be an int: {path}"
                )
            if gene < 0 or gene > int(dont_cares[j]):
                raise ValueError(
                    f"Phase 2 archive entry {i} gene {j} out of range: {path}"
                )


# ---------------------------------------------------------------------------
# EvoX-compatible optimizer (optional)
# ---------------------------------------------------------------------------

if _EVOX_AVAILABLE:
    class FuzzyRuleOptimizer(EvoxAlgorithm):  # type: ignore[misc]
        """
        EvoX-compatible multi-objective optimizer for fuzzy rule evolution.

        Wraps the NSGA-II fallback as an EvoX Algorithm for interface
        compatibility. When EvoX is available, this class can be used
        directly with EvoX workflows.
        """

        def __init__(
            self,
            feature_infos: list[dict],
            engine,
            pop_size: int = _cfg.PHASE2_POPULATION_SIZE,
            n_generations: int = _cfg.PHASE2_GENERATIONS,
        ):
            self.feature_infos = feature_infos
            self.engine = engine
            self.pop_size = pop_size
            self.n_generations = n_generations
            self._rng = np.random.default_rng(42)
            self._population: Optional[np.ndarray] = None
            self._objectives: Optional[np.ndarray] = None

        def setup(self, key=None):
            """Initialise population with dont_care sparsity."""
            self._population = _init_population(
                self.pop_size, self.feature_infos, self._rng
            )
            dont_cares = _get_dont_cares(self.feature_infos)
            self._objectives = np.full((self.pop_size, 3), np.inf)
            self._dont_cares = dont_cares
            return self

        def step(self, state=None):
            """Run one generation step."""
            if self._population is None:
                self.setup()
            # Evaluate
            for i in range(self.pop_size):
                if np.any(np.isinf(self._objectives[i])):
                    obj, _ = _evaluate_chromosome(
                        self._population[i], self._dont_cares, self.engine, []
                    )
                    self._objectives[i] = obj
            return self


# ---------------------------------------------------------------------------
# Rule_Pool_Generator
# ---------------------------------------------------------------------------

class Rule_Pool_Generator:
    """
    Phase 2: GPU-accelerated multi-objective evolutionary rule pool generation.

    Generates separate Pareto-front rule pools for long and short directions.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training split DataFrame (already prepared by Data_Loader + Data_Splitter).
    feature_infos : list[dict]
        Output of Feature_Selector.select_features(): list of
        {"name": str, "mode": str, "score": float}.
    direction : str
        "long" or "short".
    pop_size : int, optional
        Override PHASE2_POPULATION_SIZE (useful for testing).
    n_generations : int, optional
        Override PHASE2_GENERATIONS (useful for testing).
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        train_df: pd.DataFrame,
        feature_infos: list[dict],
        direction: str,
        pop_size: int | None = None,
        n_generations: int | None = None,
        seed: int = 42,
    ) -> None:
        if direction not in ("long", "short"):
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")
        if not feature_infos:
            raise ValueError("feature_infos must not be empty.")

        self.direction = direction
        self.feature_infos = feature_infos
        self.pop_size = pop_size if pop_size is not None else _cfg.PHASE2_POPULATION_SIZE
        self.n_generations = n_generations if n_generations is not None else _cfg.PHASE2_GENERATIONS
        self.seed = seed
        self._feature_signature = _archive_feature_signature(feature_infos)

        # Sample training data to budget, then slim to backtest-only columns
        sampled = _sample_df(train_df, _cfg.PHASE1_SAMPLING_TOTAL)
        feature_names = [fi["name"] for fi in feature_infos]
        from gpu_fuzzy_trader.backtest.df_slim import slim_backtest_df

        self._train_df = slim_backtest_df(sampled, feature_names)

        # Build feature_modes dict for engine
        self._feature_modes = {fi["name"]: fi["mode"] for fi in feature_infos}

        # Initialise backtest engine (GPU preferred, CPU fallback)
        self._engine = self._build_engine()
        from gpu_fuzzy_trader._memory import log_memory_rss

        log_memory_rss(f"Phase2 [{direction}] engine init")

    # ------------------------------------------------------------------
    # Engine construction
    # ------------------------------------------------------------------

    def _build_engine(self):
        """Build GPUBacktestEngine if JAX available, else CPUBacktestEngine."""
        try:
            from gpu_fuzzy_trader.backtest.gpu_engine import GPUBacktestEngine
            engine = GPUBacktestEngine(
                self._train_df,
                self._feature_modes,
                self.direction,
            )
            logger.info(
                "Phase 2 using GPUBacktestEngine (backend: %s)", engine.backend)
            return engine
        except ImportError:
            logger.warning(
                "JAX not available; falling back to CPUBacktestEngine for Phase 2.")
            from gpu_fuzzy_trader.backtest.cpu_engine import CPUBacktestEngine
            return CPUBacktestEngine(
                self._train_df,
                self._feature_modes,
                self.direction,
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> list[dict]:
        """
        Run the evolutionary search and return the Pareto-front pool.

        Also persists results to:
                    pools/phase2_{direction}_pool.json
                    pools/phase2_{direction}_history.json

        Returns
        -------
        list[dict]
            Pareto-front rules in pool JSON schema.
        """
        rng = np.random.default_rng(self.seed)

        logger.info(
            "Phase 2 [%s]: pop=%d, gen=%d, features=%d",
            self.direction, self.pop_size, self.n_generations, len(
                self.feature_infos)
        )

        from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution

        logger.info(
            "Phase 2 [%s]: algorithm=%s",
            self.direction, _cfg.PHASE2_ALGORITHM,
        )

        archive_state = self.load_archive(self.direction, self.feature_infos)
        seed_chromosomes = None
        if archive_state and archive_state["rules"]:
            seed_chromosomes = np.asarray(
                [entry["chromosome"] for entry in archive_state["rules"]],
                dtype=np.int32,
            )
            logger.info(
                "Phase 2 [%s]: seeding from archive with %d rules",
                self.direction,
                len(archive_state["rules"]),
            )

        progress_tag = "Phase 2 [%s] NSGA-III" % self.direction
        pool, history = run_phase2_evolution(
            feature_infos=self.feature_infos,
            engine=self._engine,
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            rng=rng,
            log_tag=progress_tag,
            seed_chromosomes=seed_chromosomes,
        )

        pool_path = _POOL_PATHS[self.direction]
        history_path = _HISTORY_PATHS[self.direction]
        pool_dir = os.path.dirname(pool_path)
        history_dir = os.path.dirname(history_path)
        if pool_dir:
            os.makedirs(pool_dir, exist_ok=True)
        if history_dir and history_dir != pool_dir:
            os.makedirs(history_dir, exist_ok=True)

        with open(pool_path, "w", encoding="utf-8") as fh:
            json.dump(pool, fh, indent=2)
        with open(history_path, "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=2)

        logger.info(
            "Phase 2 [%s]: pool_size=%d, saved to %s",
            self.direction, len(pool), pool_path
        )

        # Reporter: plot Phase 2 generation metrics
        try:
            reporter = Reporter()
            reporter.plot_phase2_metrics(history, self.direction)
            reporter.plot_phase2_pnl(history, self.direction)
        except Exception as exc:
            logger.warning(
                "Reporter Phase 2 plots failed (non-fatal): %s", exc)

        self._release_resources()
        return pool

    def _release_resources(self) -> None:
        """Drop engine and sampled data to free RAM before the next direction."""
        self._engine = None
        self._train_df = None
        from gpu_fuzzy_trader._memory import log_memory_rss, release_phase2_resources

        log_memory_rss(f"Phase2 [{self.direction}] after release")
        release_phase2_resources()

    @staticmethod
    def load_pool(direction: str) -> Optional[list[dict]]:
        """
        Load existing pool if valid, return None if missing.

        Parameters
        ----------
        direction : str
            "long" or "short".

        Returns
        -------
        list[dict] | None
            Loaded pool if file exists and is valid JSON, None if missing.

        Raises
        ------
        ValueError
            If the file exists but is corrupted or has an invalid schema.
        """
        if direction not in _POOL_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _POOL_PATHS[direction]
        if not os.path.exists(path):
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                pool = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(
                f"Phase 2 pool file is unreadable or corrupted: {path}"
            ) from exc

        _validate_pool_schema(pool, path)
        return pool

    @staticmethod
    def load_archive(
        direction: str,
        feature_infos: list[dict],
    ) -> Optional[dict]:
        """
        Load a compatible persistent archive if it exists, otherwise return None.

        Archive files are ignored when they are corrupt or were built from an
        incompatible feature signature.
        """
        if direction not in _ARCHIVE_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _ARCHIVE_PATHS[direction]
        if not os.path.exists(path):
            return None

        payload = _read_json_payload(path)
        if payload is None:
            logger.warning(
                "Phase 2 archive file is unreadable or corrupted: %s", path)
            return None

        try:
            _validate_archive_payload(payload, path, feature_infos)
        except ValueError as exc:
            logger.info("Ignoring Phase 2 archive at %s: %s", path, exc)
            return None

        rules = _merge_archive_entries(payload["rules"])
        return {
            "version": int(payload.get("version", 1)),
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": rules,
        }

    @staticmethod
    def save_archive(
        direction: str,
        feature_infos: list[dict],
        rules: list[dict],
    ) -> list[dict]:
        """Merge the latest pool into the persistent archive and write it atomically."""
        if direction not in _ARCHIVE_PATHS:
            raise ValueError(
                f"direction must be 'long' or 'short', got {direction!r}")

        path = _ARCHIVE_PATHS[direction]
        existing_rules: list[dict] = []
        raw_payload = _read_json_payload(path)
        if raw_payload is not None:
            try:
                _validate_archive_payload(raw_payload, path, feature_infos)
            except ValueError as exc:
                logger.info(
                    "Replacing invalid Phase 2 archive at %s: %s", path, exc)
            else:
                existing_rules = list(raw_payload["rules"])

        merged = _merge_archive_entries(existing_rules + list(rules))
        if not merged:
            return []

        payload = {
            "version": 1,
            "direction": direction,
            "feature_signature": _archive_feature_signature(feature_infos),
            "rules": merged,
        }

        archive_dir = os.path.dirname(path)
        if archive_dir:
            os.makedirs(archive_dir, exist_ok=True)

        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp_path, path)
        return merged

    @staticmethod
    def skip_if_valid(direction: str) -> Optional[list[dict]]:
        """
        Return loaded pool if valid, None if need to run.

        Parameters
        ----------
        direction : str
            "long" or "short".

        Returns
        -------
        list[dict] | None
            Loaded pool if file exists and is valid, None otherwise.
        """
        try:
            return Rule_Pool_Generator.load_pool(direction)
        except ValueError:
            return None


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

def _validate_pool_schema(pool: object, path: str) -> None:
    """
    Validate the structure of a loaded pool JSON.

    Raises ValueError if the schema is invalid.
    """
    if not isinstance(pool, list):
        raise ValueError(
            f"Phase 2 pool must be a JSON array, got {type(pool).__name__}: {path}"
        )

    for i, entry in enumerate(pool):
        if not isinstance(entry, dict):
            raise ValueError(
                f"Phase 2 pool entry {i} must be a dict: {path}"
            )
        required_keys = {"chromosome", "conditions",
                         "objectives", "executed_trades"}
        missing = required_keys - set(entry.keys())
        if missing:
            raise ValueError(
                f"Phase 2 pool entry {i} missing keys {missing}: {path}"
            )
        if not isinstance(entry["chromosome"], list):
            raise ValueError(
                f"Phase 2 pool entry {i} 'chromosome' must be a list: {path}"
            )
        if not isinstance(entry["conditions"], list):
            raise ValueError(
                f"Phase 2 pool entry {i} 'conditions' must be a list: {path}"
            )
        if not isinstance(entry["objectives"], dict):
            raise ValueError(
                f"Phase 2 pool entry {i} 'objectives' must be a dict: {path}"
            )
        required_obj_keys = {"sortino_ratio",
                             "max_drawdown_pct", "win_rate"}
        missing_obj = required_obj_keys - set(entry["objectives"].keys())
        if missing_obj:
            raise ValueError(
                f"Phase 2 pool entry {i} 'objectives' missing keys {missing_obj}: {path}"
            )
        if not isinstance(entry["executed_trades"], int):
            raise ValueError(
                f"Phase 2 pool entry {i} 'executed_trades' must be an int: {path}"
            )

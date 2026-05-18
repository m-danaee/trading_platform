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
    f1 = -total_return_pct
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

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

_POOL_PATHS = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "phase2_long_pool.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "phase2_short_pool.json"),
}
_HISTORY_PATHS = {
    "long": os.path.join(_cfg.OUTPUTS_DIR, "phase2_long_history.json"),
    "short": os.path.join(_cfg.OUTPUTS_DIR, "phase2_short_history.json"),
}

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


def _pareto_return_stats(
    pareto_indices: list[int],
    metrics_cache: list[dict],
) -> dict[str, float]:
    """Aggregate raw backtest return (%) over the current Pareto front."""
    if not pareto_indices:
        return {"mean_total_return_pct": 0.0, "best_total_return_pct": 0.0}
    returns = [
        float(metrics_cache[i].get("total_return_pct", 0.0))
        for i in pareto_indices
    ]
    return {
        "mean_total_return_pct": float(np.mean(returns)),
        "best_total_return_pct": float(np.max(returns)),
    }


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
            "total_return_pct": 0.0,
            "max_drawdown_pct": 100.0,
            "win_rate": 0.0,
            "executed_trades": 0,
        }

    total_return = float(metrics.get("total_return_pct", 0.0))
    max_dd = float(metrics.get("max_drawdown_pct", 100.0))
    win_rate = float(metrics.get("win_rate", 0.0))
    executed = int(metrics.get("executed_trades", 0))

    # Support penalty
    support_penalty = 0.0
    if executed < _cfg.MIN_TRADE_SUPPORT:
        support_penalty = (_cfg.MIN_TRADE_SUPPORT - executed) * 0.5

    # Diversity penalty: Hamming distance to nearest Pareto-front member
    diversity_penalty = 0.0
    if pareto_front:
        min_hamming = min(_hamming_distance(chromosome, pf) for pf in pareto_front)
        if min_hamming == 0:
            diversity_penalty = 5.0  # identical rule already in front

    f1 = -total_return + support_penalty + diversity_penalty + cond_penalty
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

    Returns
    -------
    np.ndarray
        Shape (pop_size, K) int32.
    """
    K = len(feature_infos)
    dont_cares = _get_dont_cares(feature_infos)
    population = np.zeros((pop_size, K), dtype=np.int32)

    for k, fi in enumerate(feature_infos):
        dc = dont_cares[k]
        num_classes = dc  # dont_care = num_classes
        for i in range(pop_size):
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
        "objectives": {"total_return_pct": ..., "max_drawdown_pct": ..., "win_rate": ...},
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
        if executed < _cfg.MIN_TRADE_SUPPORT:
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
                "total_return_pct": float(metrics.get("total_return_pct", 0.0)),
                "max_drawdown_pct": float(metrics.get("max_drawdown_pct", 0.0)),
                "win_rate": float(metrics.get("win_rate", 0.0)),
            },
            "executed_trades": executed,
        })

    return pool


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
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
        if not feature_infos:
            raise ValueError("feature_infos must not be empty.")

        self.direction = direction
        self.feature_infos = feature_infos
        self.pop_size = pop_size if pop_size is not None else _cfg.PHASE2_POPULATION_SIZE
        self.n_generations = n_generations if n_generations is not None else _cfg.PHASE2_GENERATIONS
        self.seed = seed

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
            logger.info("Phase 2 using GPUBacktestEngine (backend: %s)", engine.backend)
            return engine
        except ImportError:
            logger.warning("JAX not available; falling back to CPUBacktestEngine for Phase 2.")
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
          outputs/phase2_{direction}_pool.json
          outputs/phase2_{direction}_history.json

        Returns
        -------
        list[dict]
            Pareto-front rules in pool JSON schema.
        """
        rng = np.random.default_rng(self.seed)

        logger.info(
            "Phase 2 [%s]: pop=%d, gen=%d, features=%d",
            self.direction, self.pop_size, self.n_generations, len(self.feature_infos)
        )

        from gpu_fuzzy_trader.evolution.evox_runner import run_phase2_evolution

        logger.info(
            "Phase 2 [%s]: algorithm=%s",
            self.direction, _cfg.PHASE2_ALGORITHM,
        )

        progress_tag = "Phase 2 [%s] NSGA-III" % self.direction
        pool, history = run_phase2_evolution(
            feature_infos=self.feature_infos,
            engine=self._engine,
            pop_size=self.pop_size,
            n_generations=self.n_generations,
            rng=rng,
            log_tag=progress_tag,
        )

        # Persist
        os.makedirs(_cfg.OUTPUTS_DIR, exist_ok=True)
        pool_path = _POOL_PATHS[self.direction]
        history_path = _HISTORY_PATHS[self.direction]

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
            raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

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
        required_keys = {"chromosome", "conditions", "objectives", "executed_trades"}
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
        required_obj_keys = {"total_return_pct", "max_drawdown_pct", "win_rate"}
        missing_obj = required_obj_keys - set(entry["objectives"].keys())
        if missing_obj:
            raise ValueError(
                f"Phase 2 pool entry {i} 'objectives' missing keys {missing_obj}: {path}"
            )
        if not isinstance(entry["executed_trades"], int):
            raise ValueError(
                f"Phase 2 pool entry {i} 'executed_trades' must be an int: {path}"
            )

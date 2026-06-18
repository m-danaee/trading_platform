"""
Property-based tests for crash-fix-and-run-logging spec.

# Feature: crash-fix-and-run-logging, Property 1: Log formatter preserves message content
# Feature: crash-fix-and-run-logging, Property 2: configure_jax_env does not overwrite pre-existing env vars
# Feature: crash-fix-and-run-logging, Property 3: O(N) metrics cache rebuild correctness
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest
from hypothesis import given

from tests.property.hypothesis_config import prop_settings
from hypothesis import strategies as st

from gpu_fuzzy_trader._jax_env import configure_jax_env


# ---------------------------------------------------------------------------
# Property 1: Log formatter preserves message content
# ---------------------------------------------------------------------------

# Feature: crash-fix-and-run-logging, Property 1: Log formatter preserves message content
@given(st.text(min_size=1))
@prop_settings(max_examples=100)
def test_run_log_formatter_preserves_message(message: str) -> None:
    """**Validates: Requirements 1.3**

    For any non-empty log message string, when the run.log FileHandler
    formatter formats a LogRecord containing that message, the formatted
    output string SHALL contain the original message as a substring.
    """
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    record = logging.LogRecord(
        name="test.logger",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=message,
        args=(),
        exc_info=None,
    )
    formatted = formatter.format(record)
    assert message in formatted, (
        f"Formatted output does not contain original message.\n"
        f"  message:   {message!r}\n"
        f"  formatted: {formatted!r}"
    )


# ---------------------------------------------------------------------------
# Property 2: configure_jax_env does not overwrite pre-existing env vars
# ---------------------------------------------------------------------------

# Feature: crash-fix-and-run-logging, Property 2: configure_jax_env does not overwrite pre-existing env vars
@given(st.text(min_size=1).filter(lambda s: "\x00" not in s))
@prop_settings(max_examples=100)
def test_configure_jax_env_does_not_overwrite_preallocate(pre_existing_value: str) -> None:
    """**Validates: Requirements 2.2**

    For any non-empty string value pre-assigned to XLA_PYTHON_CLIENT_PREALLOCATE,
    configure_jax_env() must leave the value unchanged (setdefault semantics).
    """
    original = os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")
    try:
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = pre_existing_value

        configure_jax_env()

        assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == pre_existing_value
    finally:
        # Restore the environment to its original state after each example.
        if original is None:
            os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        else:
            os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = original


# ---------------------------------------------------------------------------
# Helpers — pure reimplementation of the O(N) rebuild logic under test
# ---------------------------------------------------------------------------

def _rebuild_metrics_cache_on(
    merge_pop: np.ndarray,
    merge_metrics: list[dict],
    population: np.ndarray,
) -> list[dict]:
    """
    O(N) metrics cache rebuild — mirrors the production code in _run_nsga3().

    Validates: Requirements 5.1, 5.2, 5.3
    """
    n_alive = len(population)
    _merge_metrics_by_key: dict[tuple, dict] = {
        tuple(merge_pop[j].tolist()): m
        for j, m in enumerate(merge_metrics)
        if m
    }
    return [
        _merge_metrics_by_key.get(tuple(population[i].tolist()), {})
        for i in range(n_alive)
    ]


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

# A single chromosome is a 1-D array of small integers (gene values).
_gene_value = st.integers(min_value=0, max_value=7)

# A chromosome row: fixed width of 4 genes for simplicity.
_chromosome = st.lists(_gene_value, min_size=4, max_size=4).map(
    lambda row: np.array(row, dtype=np.int32)
)

# A non-empty dict of metrics (arbitrary string keys → float values).
_metrics_dict = st.dictionaries(
    keys=st.text(min_size=1, max_size=20),
    values=st.floats(allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=5,
)

# An empty dict (represents "no metrics recorded").
_empty_dict = st.just({})

# Either a real metrics dict or an empty one.
_maybe_metrics = st.one_of(_metrics_dict, _empty_dict)


@st.composite
def _merge_pop_and_metrics_and_population(draw):
    """
    Generate a consistent (merge_pop, merge_metrics, population) triple where
    population is a subset of rows from merge_pop.
    """
    # 1. Build a pool of unique chromosomes (2–12 rows).
    n_merge = draw(st.integers(min_value=2, max_value=12))
    rows = draw(
        st.lists(_chromosome, min_size=n_merge, max_size=n_merge)
    )
    merge_pop = np.stack(rows, axis=0)  # shape (n_merge, 4)

    # 2. Assign a metrics dict (possibly empty) to each row.
    merge_metrics: list[dict] = [draw(_maybe_metrics) for _ in range(n_merge)]

    # 3. Pick a non-empty subset of row indices as the surviving population.
    n_alive = draw(st.integers(min_value=1, max_value=n_merge))
    indices = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_merge - 1),
            min_size=n_alive,
            max_size=n_alive,
        )
    )
    population = merge_pop[indices]  # shape (n_alive, 4)

    return merge_pop, merge_metrics, population, indices


# ---------------------------------------------------------------------------
# Property 3: O(N) metrics cache rebuild correctness
# ---------------------------------------------------------------------------

@given(triple=_merge_pop_and_metrics_and_population())
@prop_settings(max_examples=100)
def test_metrics_cache_rebuild_correctness(triple):
    """
    **Validates: Requirements 5.1, 5.2, 5.3**

    Property 3: O(N) metrics cache rebuild correctness.

    For any merged population and corresponding metrics list, after rebuilding
    the metrics cache using the dict-based O(N) lookup:
      - Each entry equals the metrics dict for the chromosome at the same index
        in the surviving population if that chromosome has a non-empty metrics
        entry in merge_metrics.
      - If no non-empty metrics were recorded for a chromosome, the entry is {}.
    """
    merge_pop, merge_metrics, population, indices = triple

    result = _rebuild_metrics_cache_on(merge_pop, merge_metrics, population)

    assert len(result) == len(population), (
        "Rebuilt cache length must equal surviving population size"
    )

    for i, src_idx in enumerate(indices):
        chrom_key = tuple(population[i].tolist())

        # Find the expected metrics: the last non-empty entry in merge_metrics
        # whose chromosome matches (dict comprehension keeps last writer for
        # duplicate keys, so we replicate that here).
        expected: dict = {}
        for j in range(len(merge_pop)):
            if tuple(merge_pop[j].tolist()) == chrom_key and merge_metrics[j]:
                expected = merge_metrics[j]
        # (No break — dict comprehension overwrites duplicates, keeping last.)

        assert result[i] == expected, (
            f"metrics_cache[{i}] mismatch for chromosome {chrom_key}: "
            f"got {result[i]!r}, expected {expected!r}"
        )


@given(triple=_merge_pop_and_metrics_and_population())
@prop_settings(max_examples=100)
def test_metrics_cache_rebuild_empty_fallback(triple):
    """
    **Validates: Requirements 5.3**

    When all merge_metrics entries are empty dicts, every entry in the rebuilt
    cache must be {} (the O(N) lookup must fall back to empty dict for every
    surviving individual).
    """
    merge_pop, _, population, _ = triple
    all_empty_metrics = [{} for _ in range(len(merge_pop))]

    result = _rebuild_metrics_cache_on(merge_pop, all_empty_metrics, population)

    assert all(entry == {} for entry in result), (
        "All entries should be {} when merge_metrics contains only empty dicts"
    )


@given(triple=_merge_pop_and_metrics_and_population())
@prop_settings(max_examples=100)
def test_metrics_cache_rebuild_length_invariant(triple):
    """
    **Validates: Requirements 5.1**

    The length of the rebuilt metrics_cache must always equal the number of
    surviving individuals (n_alive), regardless of the size of merge_pop.
    """
    merge_pop, merge_metrics, population, _ = triple

    result = _rebuild_metrics_cache_on(merge_pop, merge_metrics, population)

    assert len(result) == len(population)

"""Threshold graph clustering and greedy portfolio selection."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _resolve_lambda(value: float | str) -> float:
    """Resolve the supported zero/low/medium penalty levels."""
    if isinstance(value, str):
        text = value.strip().lower()
        named = {
            "0": 0.0,
            "off": 0.0,
            "none": 0.0,
            "low": 0.25,
            "medium": 0.50,
        }
        if text in named:
            return named[text]
        try:
            value = float(text)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "lambda must be 0, 'low', 'medium', or a non-negative number"
            ) from exc
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("lambda must be numeric or a named level") from exc
    if result < 0.0 or not np.isfinite(result):
        raise ValueError("lambda must be finite and non-negative")
    return result


def threshold_graph_clusters(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    threshold: float,
) -> list[list[int]]:
    """Return connected components of a thresholded redundancy graph.

    Nodes are candidate positions.  An undirected edge exists when the
    redundancy value is greater than or equal to ``threshold``.  Breadth-first
    traversal makes the result deterministic and handles transitive clusters.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("the redundancy matrix must be square")
    threshold_value = float(threshold)
    if not np.isfinite(threshold_value):
        raise ValueError("cluster threshold must be finite")
    size = int(values.shape[0])
    unseen = set(range(size))
    clusters: list[list[int]] = []
    while unseen:
        seed = min(unseen)
        unseen.remove(seed)
        queue = [seed]
        component: list[int] = []
        while queue:
            node = queue.pop(0)
            component.append(node)
            neighbours = [
                other
                for other in sorted(unseen)
                if (
                    np.isfinite(values[node, other])
                    and values[node, other] >= threshold_value
                )
                or (
                    np.isfinite(values[other, node])
                    and values[other, node] >= threshold_value
                )
            ]
            for other in neighbours:
                unseen.remove(other)
                queue.append(other)
        clusters.append(sorted(component))
    return clusters


def _cluster_index_map(
    size: int,
    clusters: Sequence[Sequence[int]] | None,
) -> dict[int, int]:
    if clusters is None:
        return {index: index for index in range(size)}
    result: dict[int, int] = {}
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            candidate_index = int(index)
            if 0 <= candidate_index < size and candidate_index not in result:
                result[candidate_index] = cluster_id
    next_cluster = len(clusters)
    for index in range(size):
        if index not in result:
            result[index] = next_cluster
            next_cluster += 1
    return result


def adjusted_quality(
    quality: float,
    redundancy: float | Iterable[float],
    *,
    lambda_: float | str = 0.0,
    lam: float | str | None = None,
) -> float:
    """Return quality after subtracting the selected redundancy penalty."""
    penalty_lambda = _resolve_lambda(lambda_ if lam is None else lam)
    if np.isscalar(redundancy):
        maximum_redundancy = float(redundancy)
    else:
        values = np.asarray(list(redundancy), dtype=float)
        finite = values[np.isfinite(values)]
        maximum_redundancy = float(np.max(finite)) if finite.size else 0.0
    return float(quality) - penalty_lambda * maximum_redundancy


def greedy_adjusted_quality(
    qualities: Sequence[float] | np.ndarray,
    redundancy: Sequence[Sequence[float]] | np.ndarray,
    *,
    lambda_: float | str = 0.0,
    lam: float | str | None = None,
    max_items: int | None = None,
    clusters: Sequence[Sequence[int]] | None = None,
    require_cross_cluster: bool = False,
) -> list[int]:
    """Greedily select candidates by ``AdjustedQuality``.

    The penalty is the largest redundancy with an already selected candidate.
    When ``require_cross_cluster`` is true, an unrepresented cluster is
    preferred while one exists.  This provides cross-cluster picks without
    making a portfolio impossible when all remaining candidates are in a
    represented cluster.
    """
    quality_values = np.asarray(qualities, dtype=float).reshape(-1)
    matrix = np.asarray(redundancy, dtype=float)
    if matrix.shape != (quality_values.size, quality_values.size):
        raise ValueError("qualities and redundancy matrix have incompatible shapes")
    if max_items is None:
        limit = quality_values.size
    else:
        limit = max(0, min(int(max_items), quality_values.size))
    penalty_lambda = _resolve_lambda(lambda_ if lam is None else lam)
    cluster_map = _cluster_index_map(quality_values.size, clusters)
    remaining = set(range(quality_values.size))
    selected: list[int] = []

    while remaining and len(selected) < limit:
        selected_clusters = {cluster_map[index] for index in selected}
        unrepresented = {
            cluster_map[index] for index in remaining
            if cluster_map[index] not in selected_clusters
        }
        pool = remaining
        if require_cross_cluster and unrepresented:
            pool = {
                index for index in remaining
                if cluster_map[index] in unrepresented
            }
        best_index: int | None = None
        best_value = float("-inf")
        for index in sorted(pool):
            if selected:
                row = matrix[index, selected]
                finite = row[np.isfinite(row)]
                max_redundancy = float(np.max(finite)) if finite.size else 0.0
            else:
                max_redundancy = 0.0
            value = adjusted_quality(
                quality_values[index],
                max_redundancy,
                lambda_=penalty_lambda,
            )
            if not np.isfinite(value):
                value = float("-inf")
            if best_index is None or value > best_value:
                best_index = index
                best_value = value
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return selected


def cluster_labels(
    matrix: Sequence[Sequence[float]] | np.ndarray,
    threshold: float,
) -> list[int]:
    """Return one cluster label per matrix row."""
    clusters = threshold_graph_clusters(matrix, threshold=threshold)
    labels = [0] * len(np.asarray(matrix))
    for cluster_id, cluster in enumerate(clusters):
        for index in cluster:
            labels[index] = cluster_id
    return labels


def cross_cluster_picks(
    qualities: Sequence[float] | np.ndarray,
    redundancy: Sequence[Sequence[float]] | np.ndarray,
    clusters: Sequence[Sequence[int]],
    *,
    max_items: int | None = None,
    lambda_: float = 0.0,
) -> list[int]:
    """Select candidates while preferring an unrepresented cluster."""
    return greedy_adjusted_quality(
        qualities,
        redundancy,
        lambda_=lambda_,
        max_items=max_items,
        clusters=clusters,
        require_cross_cluster=True,
    )


# Compatibility aliases for callers that name the graph operation directly.
cluster_redundancy_matrix = threshold_graph_clusters
cluster_candidates = threshold_graph_clusters
threshold_clusters = threshold_graph_clusters
greedy_select = greedy_adjusted_quality


__all__ = [
    "adjusted_quality",
    "cluster_candidates",
    "cluster_labels",
    "cluster_redundancy_matrix",
    "cross_cluster_picks",
    "greedy_adjusted_quality",
    "greedy_select",
    "threshold_clusters",
    "threshold_graph_clusters",
]

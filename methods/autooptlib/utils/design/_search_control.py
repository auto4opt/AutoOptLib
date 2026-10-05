"""State and representation utilities for search-based algorithm design."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np


def _finite_values(value: Any, *, decimals: int) -> Any:
    if value is None:
        return None
    array = np.asarray(value)
    if array.dtype.kind in {"i", "u", "b"}:
        return array.astype(int).reshape(-1).tolist()
    rounded = np.round(array.astype(float), decimals=decimals).reshape(-1)
    return [float(item) for item in rounded]


def _canonical_json_value(value: Any, *, decimals: int) -> Any:
    """Round finite phenotype floats recursively for stable semantic keys."""

    if isinstance(value, Mapping):
        return {
            str(key): _canonical_json_value(item, decimals=decimals)
            for key, item in value.items()
        }
    if isinstance(value, np.ndarray):
        return _canonical_json_value(value.tolist(), decimals=decimals)
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item, decimals=decimals) for item in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if not np.isfinite(number):
            return number
        rounded = float(np.round(number, decimals=decimals))
        return 0.0 if rounded == 0.0 else rounded
    return value


def active_operator_indices(candidate: Any) -> tuple[int, ...]:
    indices: set[int] = set()
    for path in getattr(candidate, "operator", None) or []:
        matrix = np.asarray(path, dtype=int)
        if matrix.size:
            indices.update(int(value) for value in matrix.reshape(-1))
    return tuple(sorted(value for value in indices if value > 0))


def representation_payload(
    candidate: Any,
    *,
    include_continuous_parameters: bool = True,
    decimals: int = 12,
) -> dict[str, Any]:
    """Return a canonical genotype payload, excluding inactive parameters."""
    if include_continuous_parameters:
        # Candidate records and released algorithms use the decoded, portable
        # pathway representation.  Two different internal genotypes can
        # decode to the same executable algorithm (for example after repair or
        # component aliases).  Deduplicating only the raw matrices therefore
        # allowed a small number of semantically identical algorithms to be
        # evaluated twice.  Prefer the exact representation written to the
        # candidate ledger so online admission and the final audit agree.
        try:
            from ...serialization import algorithm_to_dict

            document = algorithm_to_dict(candidate)
            pathways = document["pathways"]
            if document.get("execution_semantics") == "stream_graph_v2":
                # Parallel stream pathways are unordered when they terminate
                # at the same update sink.  Canonicalize that graph symmetry
                # so merely swapping the two serialized branches cannot spend
                # another Search evaluation or diversity slot.  Distinct
                # path-local updates remain ordered because applying different
                # replacement operators can be order-sensitive.
                update_keys = {
                    json.dumps(
                        _canonical_json_value(
                            {
                                "update": pathway.get("update"),
                                "parameter": pathway.get("update_parameter"),
                            },
                            decimals=decimals,
                        ),
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    for pathway in pathways
                }
                if len(update_keys) == 1:
                    pathways = sorted(
                        pathways,
                        key=lambda pathway: json.dumps(
                            _canonical_json_value(pathway, decimals=decimals),
                            sort_keys=True,
                            separators=(",", ":"),
                        ),
                    )
        except (AttributeError, IndexError, TypeError, ValueError):
            # Lightweight controller unit tests and partially constructed
            # candidates may not be decoded yet.  The genotype remains the
            # appropriate fallback until a phenotype exists.
            pathways = None
        if pathways is not None:
            from ._population import copy_configuration

            configuration = copy_configuration(candidate)
            return {
                "configuration": _canonical_json_value(
                    configuration, decimals=decimals
                ),
                "pathways": _canonical_json_value(pathways, decimals=decimals),
            }

    raw_paths = list(getattr(candidate, "operator", None) or [])
    phenotype_groups = getattr(candidate, "operator_pheno", None) or []
    phenotype_paths = phenotype_groups[0] if phenotype_groups else []
    if (
        len(raw_paths) > 1
        and len(phenotype_paths) == len(raw_paths)
        and all(hasattr(path, "stages") for path in phenotype_paths)
        and len({str(path.update) for path in phenotype_paths}) == 1
    ):
        raw_paths.sort(
            key=lambda path: tuple(np.asarray(path, dtype=int).reshape(-1).tolist())
        )
    operators = [np.asarray(path, dtype=int).tolist() for path in raw_paths]
    payload: dict[str, Any] = {"operator": operators}
    if include_continuous_parameters:
        parameter = getattr(candidate, "parameter", None) or []
        active = active_operator_indices(candidate)
        payload["parameter"] = {
            str(index): _finite_values(parameter[index - 1][0], decimals=decimals)
            for index in active
            if index <= len(parameter)
            and parameter[index - 1] is not None
            and parameter[index - 1][0] is not None
        }
    return payload


def representation_key(
    candidate: Any,
    *,
    include_continuous_parameters: bool = True,
    decimals: int = 12,
) -> str:
    document = representation_payload(
        candidate,
        include_continuous_parameters=include_continuous_parameters,
        decimals=decimals,
    )
    encoded = json.dumps(document, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def graph_tokens(candidate: Any) -> frozenset[str]:
    """Encode graph nodes and directed edges for a scale-free diversity metric."""
    tokens: set[str] = set()
    paths = list(getattr(candidate, "operator", None) or [])
    if paths:
        terminal_nodes = []
        for path in paths:
            matrix = np.asarray(path, dtype=int)
            terminal_nodes.append(int(matrix[-1, 1]) if matrix.size else -1)
        if len(set(terminal_nodes)) == 1:
            paths.sort(
                key=lambda path: tuple(np.asarray(path, dtype=int).reshape(-1).tolist())
            )
    for path_index, path in enumerate(paths):
        matrix = np.asarray(path, dtype=int)
        for row_index, (source, target) in enumerate(matrix.tolist()):
            tokens.add(f"p{path_index}:r{row_index}:n{int(source)}")
            tokens.add(f"p{path_index}:r{row_index}:e{int(source)}>{int(target)}")
    return frozenset(tokens)


def graph_distance(left: Any, right: Any) -> float:
    """Normalized graph-token Jaccard distance in ``[0, 1]``."""
    left_tokens = graph_tokens(left)
    right_tokens = graph_tokens(right)
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return 1.0 - len(left_tokens & right_tokens) / len(union)


def candidate_cost(candidate: Any) -> float:
    performance = np.asarray(getattr(candidate, "performance", []), dtype=float)
    # A partially evaluated candidate must never look artificially strong by
    # averaging only the cells that happened to finish.  Search normally passes
    # an explicit training-ledger cost, but this fallback is also used by the
    # global archive and diversity selector.
    if performance.size == 0 or not np.all(np.isfinite(performance)):
        return float("inf")
    return float(np.mean(performance))


def update_global_archive(
    archive: Iterable[Any],
    candidates: Iterable[Any],
    *,
    size: int,
    cost: Callable[[Any], float] = candidate_cost,
) -> list[Any]:
    """Keep the best unique candidates seen over the entire design run."""
    unique: dict[str, Any] = {}
    for candidate in [*archive, *candidates]:
        key = representation_key(candidate)
        incumbent = unique.get(key)
        if incumbent is None or cost(candidate) < cost(incumbent):
            unique[key] = candidate
    ordered = sorted(unique.values(), key=cost)
    return [deepcopy(candidate) for candidate in ordered[: max(1, int(size))]]


def performance_diversity_select(
    candidates: Sequence[Any],
    *,
    count: int,
    diversity_weight: float,
    global_best: Any | None = None,
    costs: Sequence[float] | None = None,
) -> list[Any]:
    """Greedily balance performance rank and distance from selected graphs."""
    if count <= 0 or not candidates:
        return []
    pool: list[Any] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = representation_key(candidate)
        if key not in seen:
            seen.add(key)
            pool.append(candidate)
    if global_best is not None:
        best_key = representation_key(global_best)
        if best_key not in seen:
            pool.append(global_best)
            seen.add(best_key)
    if len(pool) <= count:
        return pool

    if costs is None:
        cost_by_key = {
            representation_key(candidate): candidate_cost(candidate)
            for candidate in candidates
        }
    else:
        if len(costs) != len(candidates):
            raise ValueError("costs must align with candidates")
        cost_by_key = {
            representation_key(candidate): float(value)
            for candidate, value in zip(candidates, costs)
        }
    pool_costs = np.asarray(
        [
            cost_by_key.get(representation_key(candidate), candidate_cost(candidate))
            for candidate in pool
        ]
    )
    order = np.argsort(pool_costs, kind="stable")
    ranks = np.empty(len(pool), dtype=float)
    ranks[order] = np.arange(len(pool), dtype=float)
    if len(pool) > 1:
        ranks /= len(pool) - 1

    if global_best is None:
        first = int(order[0])
    else:
        global_key = representation_key(global_best)
        first = next(
            (
                idx
                for idx, candidate in enumerate(pool)
                if representation_key(candidate) == global_key
            ),
            int(order[0]),
        )
    selected = [first]
    remaining = set(range(len(pool))) - {first}
    weight = float(np.clip(diversity_weight, 0.0, 1.0))
    while remaining and len(selected) < count:
        best_index = min(
            remaining,
            key=lambda idx: (
                (1.0 - weight) * ranks[idx]
                - weight
                * min(graph_distance(pool[idx], pool[chosen]) for chosen in selected),
                pool_costs[idx],
                idx,
            ),
        )
        selected.append(best_index)
        remaining.remove(best_index)
    return [pool[index] for index in selected]


def performance_plateau_select(
    candidates: Sequence[Any],
    *,
    count: int,
    costs: Sequence[float],
    global_best: Any | None = None,
) -> list[Any]:
    """Select strictly by cost and use graph novelty only to break exact ties.

    Unlike weighted diversity selection, this rule can never retain a worse
    candidate for the sake of diversity.  It enables neutral movement across
    flat objective plateaus while the separate global archive preserves the
    best algorithm found so far.
    """

    if count <= 0 or not candidates:
        return []
    if len(costs) != len(candidates):
        raise ValueError("costs must align with candidates")

    pool: list[Any] = []
    pool_costs: list[float] = []
    positions: dict[str, int] = {}
    for candidate, value in zip(candidates, costs):
        key = representation_key(candidate)
        cost = float(value)
        position = positions.get(key)
        if position is None:
            positions[key] = len(pool)
            pool.append(candidate)
            pool_costs.append(cost)
        elif cost < pool_costs[position]:
            pool[position] = candidate
            pool_costs[position] = cost
    if len(pool) <= count:
        return pool

    selected: list[int] = []
    remaining = set(range(len(pool)))
    if global_best is not None:
        global_key = representation_key(global_best)
        global_index = positions.get(global_key)
        if global_index is not None and pool_costs[global_index] == min(pool_costs):
            selected.append(global_index)
            remaining.remove(global_index)

    while remaining and len(selected) < count:
        best_cost = min(pool_costs[index] for index in remaining)
        tied = [index for index in remaining if pool_costs[index] == best_cost]
        if selected and len(tied) > 1:
            chosen = max(
                tied,
                key=lambda index: (
                    min(graph_distance(pool[index], pool[other]) for other in selected),
                    -index,
                ),
            )
        else:
            chosen = min(tied)
        selected.append(chosen)
        remaining.remove(chosen)
    return [pool[index] for index in selected]


__all__ = [
    "active_operator_indices",
    "candidate_cost",
    "graph_distance",
    "graph_tokens",
    "performance_diversity_select",
    "performance_plateau_select",
    "representation_key",
    "representation_payload",
    "update_global_archive",
]

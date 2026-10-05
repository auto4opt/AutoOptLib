"""Shared helpers for the AutoOpt component translations."""

from __future__ import annotations

import re
from typing import Any, Sequence

import numpy as np


def _name_variants(name: str) -> list[str]:
    # Preserve a deterministic precedence order.  A set made alias resolution
    # depend on Python's hash seed when an object exposed more than one spelling
    # (for example both ``N`` and ``n``), so identical seeded runs could read a
    # different field in another process.
    variants = [name, name.lower(), name.upper()]
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()
    variants.append(snake)
    if "_" in name:
        parts = name.split("_")
        camel = parts[0].lower() + "".join(part.title() for part in parts[1:])
        pascal = "".join(part.title() for part in parts)
        variants.extend((camel, pascal))
    else:
        parts = re.findall(r"[A-Za-z][^A-Z]*", name)
        if parts:
            variants.append("_".join(part.lower() for part in parts))
            camel = parts[0].lower() + "".join(part.title() for part in parts[1:])
            pascal = "".join(part.title() for part in parts)
            variants.extend((camel, pascal))
    return list(dict.fromkeys(variant for variant in variants if variant))


def flex_get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    for candidate in _name_variants(name):
        if isinstance(obj, dict) and candidate in obj:
            return obj[candidate]
        if hasattr(obj, candidate):
            attr = getattr(obj, candidate)
            return attr() if callable(attr) else attr
    return default


def ensure_rng(*candidates: Any) -> np.random.Generator:
    for candidate in candidates:
        if isinstance(candidate, np.random.Generator):
            return candidate
        if isinstance(candidate, dict):
            nested = candidate.get("rng")
            if isinstance(nested, np.random.Generator):
                return nested
        if hasattr(candidate, "rng"):
            nested = getattr(candidate, "rng")
            if isinstance(nested, np.random.Generator):
                return nested
    return np.random.default_rng()


def selection_count(auxiliary: Any, problem: Any, default: int) -> int:
    """Return the executor-requested parent count for a selection call.

    ``problem.N`` is the survivor population size, not necessarily the number
    of parents required by the current operator.  In particular, crossover
    needs two parents per offspring and lambda may differ from mu.
    """

    requested = flex_get(auxiliary, "_selection_count", None)
    if requested is None:
        requested = flex_get(problem, "N", default)
    count = int(requested)
    if count < 0:
        raise ValueError("Selection count cannot be negative.")
    return count


def to_numpy(data: Any) -> np.ndarray:
    return np.asarray(data)


def extract_fits(solution: Any) -> np.ndarray:
    fits = flex_get(solution, "fits")
    if fits is not None:
        return np.asarray(fits, dtype=float).reshape(-1)
    if isinstance(solution, Sequence):
        values = []
        for item in solution:
            val = flex_get(item, "fit")
            if val is None:
                val = flex_get(item, "fitness")
            if val is None:
                val = flex_get(item, "fits")
            if val is None:
                raise ValueError("Each solution must expose fit/fitness value")
            arr = np.asarray(val)
            if arr.size != 1:
                raise ValueError(f"Expected a scalar-like value, got shape={arr.shape}")
            values.append(float(arr.item()))
        return np.asarray(values, dtype=float)
    raise ValueError("Solution must provide fitness information")


def solution_as_list(solution: Any) -> list[Any]:
    if isinstance(solution, list):
        return solution
    if isinstance(solution, Sequence):
        return list(solution)
    raise TypeError("Solution collection must be indexable")


def pairwise_distances(a: np.ndarray) -> np.ndarray:
    """Return Euclidean distances without materializing an ``n x n x d`` cube.

    The direct broadcast formulation is especially costly for ``choose_nich``:
    that component can run once per generation, including valid configurations
    with a large population and a single offspring.  Computing squared distances
    from the Gram matrix keeps the temporary storage quadratic in the population
    size and delegates the dominant operation to optimized matrix multiplication.
    """

    values = np.asarray(a, dtype=float)
    if values.ndim != 2:
        raise ValueError("Pairwise distances require a two-dimensional array.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Pairwise distances require finite coordinates.")
    scale = max(1.0, float(np.max(np.abs(values), initial=0.0)))
    normalized = values / scale
    squared_norms = np.einsum("ij,ij->i", normalized, normalized)
    squared_distances = squared_norms[:, None] + squared_norms[None, :]
    # Some BLAS builds leak floating-point status flags from earlier calls;
    # the arithmetic below still has the same IEEE result in that case.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        squared_distances -= 2.0 * (normalized @ normalized.T)
    # Round-off can produce tiny negative values, including on the diagonal.
    np.maximum(squared_distances, 0.0, out=squared_distances)
    np.sqrt(squared_distances, out=squared_distances)
    with np.errstate(over="ignore", invalid="ignore"):
        squared_distances *= scale
    np.nan_to_num(
        squared_distances,
        copy=False,
        nan=0.0,
        posinf=np.finfo(float).max,
        neginf=0.0,
    )
    np.fill_diagonal(squared_distances, 0.0)
    return squared_distances


def randperm(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.permutation(n)


def reshape_pairs(index: Sequence[int]) -> np.ndarray:
    array = np.asarray(index)
    if array.size % 2:
        raise ValueError("Index array must have even length for pairing")
    return array.reshape(-1, 2)


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def ensure_column(vector: Any) -> np.ndarray:
    arr = np.asarray(vector)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

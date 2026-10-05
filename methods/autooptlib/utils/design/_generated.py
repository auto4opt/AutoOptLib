"""Shared candidate evaluation for grammar-generated algorithm designs."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from . import Design


def evaluate_generated_designs(
    sequences: Iterable[Sequence[int] | np.ndarray],
    *,
    normalize: Callable[[Sequence[int] | np.ndarray], list[int]],
    decode: Callable[[Sequence[int] | np.ndarray], Design],
    problems: Any,
    data: Any,
    setting: Any,
    train_count: int,
    test_count: int,
    parallel: bool = False,
) -> tuple[list[Design], list[Design]]:
    """Evaluate, deduplicate, rank, and held-out test generated candidates."""

    seed = getattr(setting, "Seed", getattr(setting, "seed", None))
    rng = np.random.default_rng(seed)
    setting.rng = rng
    train_indices = rng.permutation(train_count).tolist()
    test_indices = (rng.permutation(test_count) + train_count).tolist()
    train_rng_state = deepcopy(rng.bit_generator.state)
    rows = [normalize(row) for row in sequences]
    keys = [tuple(row) for row in rows]
    unique_rows = list(dict.fromkeys(keys))
    decoded = [decode(row) for row in unique_rows]
    runtime = None
    if parallel:
        from ...runtime import DesignEvaluationRuntime

        runtime = DesignEvaluationRuntime(problems, data, setting)
        runtime.evaluate(decoded, setting, train_indices)
    else:
        for algorithm in decoded:
            rng.bit_generator.state = deepcopy(train_rng_state)
            algorithm.evaluate(problems, data, setting, train_indices)
    evaluated = {
        row: deepcopy(algorithm) for row, algorithm in zip(unique_rows, decoded)
    }

    candidates: list[Design] = []
    best_trace: list[Design] = []
    best_cost = np.inf
    for key in keys:
        algorithm = deepcopy(evaluated[key])
        candidates.append(algorithm)
        cost = float(np.mean(algorithm.performance[train_indices, :]))
        if cost < best_cost:
            best_cost = cost
            best_trace.append(algorithm)

    rng.bit_generator.state = deepcopy(train_rng_state)
    rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
    candidates.sort(
        key=lambda algorithm: float(np.mean(algorithm.performance[train_indices, :]))
    )
    finalists = candidates[: int(setting.AlgN)]
    test_rng_state = deepcopy(rng.bit_generator.state)
    if runtime is not None:
        if test_indices:
            runtime.evaluate(finalists, setting, test_indices)
        setting.EvalRuntimeStats = runtime.statistics.as_dict()
        runtime.close()
    else:
        if test_indices:
            for algorithm in finalists:
                rng.bit_generator.state = deepcopy(test_rng_state)
                algorithm.evaluate(problems, data, setting, test_indices)
    rng.bit_generator.state = deepcopy(test_rng_state)
    rng.integers(0, np.iinfo(np.uint64).max, dtype=np.uint64)
    # Do not sort on held-out performance: test instances are reporting-only.
    return finalists, best_trace


__all__ = ["evaluate_generated_designs"]

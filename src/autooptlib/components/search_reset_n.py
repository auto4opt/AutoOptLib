"""Reset a fixed number of discrete variables in every solution."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import ensure_rng, flex_get


def _extract_matrix(solution: Any) -> np.ndarray:
    decs = flex_get(solution, "decs")
    array = np.asarray(decs if decs is not None else solution, dtype=int)
    if array.ndim != 2:
        raise ValueError("Solution must be 2-D for search_reset_n")
    return array.copy()


def _minimum_dimension(problem: Any) -> int:
    problems = problem if isinstance(problem, (list, tuple)) else [problem]
    dimensions = []
    for item in problems:
        bound = np.asarray(flex_get(item, "bound", np.empty((2, 1))))
        if bound.ndim == 2:
            dimensions.append(int(bound.shape[1]))
    return max(1, min(dimensions, default=1))


def search_reset_n(*args):
    """Randomly reset ``n`` distinct variables of each solution.

    This component completes the Python component set used by the published
    ALDes discrete vocabulary.  Its mode-based interface mirrors the other
    AutoOptLib components and the original MATLAB implementation.
    """

    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1]
        parameter = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, problem)

        new = _extract_matrix(solution)
        bounds = np.asarray(flex_get(problem, "bound"), dtype=int)
        lower, upper = bounds
        count, dimension = new.shape
        requested = (
            1
            if parameter is None
            else int(round(float(np.asarray(parameter).reshape(-1)[0])))
        )
        reset_count = min(max(requested, 1), dimension)

        for row in range(count):
            indices = rng.choice(dimension, size=reset_count, replace=False)
            for column in np.atleast_1d(indices):
                column = int(column)
                if lower[column] == upper[column]:
                    continue
                current = new[row, column]
                candidate = int(rng.integers(lower[column], upper[column] + 1))
                while candidate == current:
                    candidate = int(rng.integers(lower[column], upper[column] + 1))
                new[row, column] = candidate
        return new, aux

    if mode == "parameter":
        problem = args[0] if len(args) > 1 else None
        dimension = _minimum_dimension(problem)
        return np.array([1.0, float(max(1, dimension))]), None

    if mode == "behavior":
        return [["LS", "small"], ["GS", "large"]], None

    raise ValueError(f"Unsupported mode: {mode}")

"""Python translation of cross_point_n."""

from __future__ import annotations

import numpy as np

from ._n_point_crossover import execute_n_point_crossover
from ._utils import flex_get


def _maximum_cut_points(problem):
    problems = problem if isinstance(problem, (list, tuple)) else [problem]
    dimensions = []
    for item in problems:
        bound = np.asarray(flex_get(item, "bound", np.empty((2, 1))))
        if bound.ndim == 2:
            dimensions.append(int(bound.shape[1]))
    return max(1, min(dimensions, default=1) - 1)


def cross_point_n(*args):
    mode = args[-1]
    if mode == "execute":
        para = args[2] if len(args) > 2 else None
        n_points = (
            int(round(float(np.asarray(para).reshape(-1)[0])))
            if para is not None
            else 1
        )
        if n_points < 1:
            raise ValueError("The number of crossover points must be positive.")
        return execute_n_point_crossover(args, n_points)
    if mode == "parameter":
        problem = args[0] if len(args) > 1 else None
        return [1, _maximum_cut_points(problem)], None
    if mode == "parameter_type":
        return ["integer"], None
    if mode == "behavior":
        return [["LS", "small"], ["GS", "large"]], None
    raise ValueError(f"Unsupported mode: {mode}")

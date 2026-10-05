"""Python translation of update_greedy."""

from __future__ import annotations

import numpy as np

from ._utils import extract_fits, flex_get, solution_as_list


def update_greedy(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        sol_list = solution_as_list(solution)
        fitness = extract_fits(solution)
        n = int(flex_get(problem, "N", len(sol_list)))
        n = min(n, len(sol_list))
        order = np.argsort(fitness)[:n]
        return [sol_list[i] for i in order], None

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

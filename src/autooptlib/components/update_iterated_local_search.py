"""Acceptance rule for random-restart iterated local search presets."""

from __future__ import annotations

import numpy as np

from ._utils import extract_fits, flex_get, solution_as_list


def update_iterated_local_search(*args):
    """Accept local improvements, but start every restart unconditionally.

    The objective tracker and solve history retain the global incumbent. The
    working solution must nevertheless accept a worse reinitialization so the
    next local-search sweep actually starts in a new basin.
    """

    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        auxiliary = args[3] if len(args) > 3 and isinstance(args[3], dict) else {}
        sol_list = solution_as_list(solution)
        n = min(int(flex_get(problem, "N", len(sol_list))), len(sol_list))
        operator = str(auxiliary.get("_active_search_operator", ""))
        if operator.startswith("reinit_"):
            return sol_list[-n:], auxiliary
        fitness = extract_fits(solution)
        order = np.argsort(fitness, kind="stable")[:n]
        return [sol_list[int(index)] for index in order], auxiliary

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

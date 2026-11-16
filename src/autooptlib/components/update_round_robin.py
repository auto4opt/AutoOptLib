"""Python translation of update_round_robin."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, flex_get, solution_as_list


def update_round_robin(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        sol_list = solution_as_list(solution)
        pop_size = len(sol_list)
        if pop_size == 0:
            return [], None
        try:
            fitness = np.fromiter(
                (float(item.fit) for item in sol_list),
                dtype=float,
                count=pop_size,
            )
        except (AttributeError, TypeError, ValueError):
            fitness = extract_fits(solution)

        n = int(flex_get(problem, "N", pop_size))
        n = min(n, pop_size)
        k = min(10, n - 1)
        if k <= 0:
            selected = np.argsort(fitness)[:n]
            return [sol_list[i] for i in selected], None

        # The k smallest independent uniforms form a uniform k-subset.  Draw
        # every individual's opponents together to avoid one Python-level
        # ``Generator.choice`` call per individual and generation.
        random_keys = rng.random((pop_size, pop_size))
        opponents = np.argpartition(random_keys, k - 1, axis=1)[:, :k]
        win = np.sum(fitness[:, None] <= fitness[opponents], axis=1)

        rank = np.argsort(win)[::-1]
        chosen = rank[:n]
        return [sol_list[i] for i in chosen], None

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

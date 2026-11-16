"""Python translation of update_pairwise."""

from __future__ import annotations

from ._utils import flex_get, solution_as_list


def update_pairwise(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        sol_list = solution_as_list(solution)
        problem = args[1] if len(args) > 1 else None
        n = min(int(flex_get(problem, "N", len(sol_list) // 2)), len(sol_list))
        old = sol_list[:n]
        offspring = sol_list[n:]
        selected = list(old)
        # Map arbitrary lambda offspring to N parents cyclically.  Every
        # parent survives unless one of its assigned offspring is better.
        for offspring_index, child in enumerate(offspring):
            parent_index = offspring_index % n
            if child.fit < selected[parent_index].fit:
                selected[parent_index] = child
        return selected, None

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

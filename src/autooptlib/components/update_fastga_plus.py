"""eoPlusReplacement compatibility component for Search v6.0."""

from __future__ import annotations

from ._utils import flex_get, solution_as_list


def update_fastga_plus(*args):
    """Merge offspring before parents, then keep the best ``mu`` solutions.

    ParadisEO's ``eoPlusReplacement`` receives parents and offspring
    separately, appends the parents to the offspring buffer, and truncates it.
    The stream executor exposes one parent-first list to update components, so
    this wrapper restores the native merge order before ranking.  Python's
    stable sort also gives a deterministic policy for equal-fitness solutions,
    for which C++ ``std::sort`` itself specifies no ordering.
    """

    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        values = solution_as_list(solution)
        population_size = int(flex_get(problem, "N", len(values) // 2))
        if population_size <= 0 or len(values) < population_size:
            raise ValueError("FastGA plus replacement requires a parent population.")
        parents = values[:population_size]
        offspring = values[population_size:]
        merged = [*offspring, *parents]
        return sorted(merged, key=lambda item: float(item.fit))[:population_size], None
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

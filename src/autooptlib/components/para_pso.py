"""Python translation of para_pso."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.solve import SolutionSet
from ._utils import flex_get
from .update_pairwise import update_pairwise


def _ensure_solution_set(solution: Any) -> SolutionSet:
    if isinstance(solution, SolutionSet):
        return solution
    return SolutionSet(solution)


def para_pso(*args):
    solution = _ensure_solution_set(args[0])
    aux = args[2] if len(args) > 2 else None

    if aux is None:
        aux = {}
    if "Pbest" not in aux:
        return aux

    pbest_set = aux["Pbest"]
    update_count = min(len(pbest_set), len(solution))
    combined = list(pbest_set)[:update_count] + list(solution)[:update_count]
    selected, _ = update_pairwise(combined, "execute")
    # A strict ProbFE boundary can truncate the last particle batch. Preserve
    # personal bests for particles that were not evaluated in that batch.
    aux["Pbest"] = SolutionSet(selected + list(pbest_set)[update_count:])

    objs = np.asarray([flex_get(sol, "obj") for sol in aux["Pbest"]], dtype=float)
    best_idx = int(np.argmin(objs))
    aux["Gbest"] = aux["Pbest"][best_idx]
    return aux

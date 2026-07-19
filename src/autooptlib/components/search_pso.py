"""Python translation of search_pso."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..utils.solve import SolutionSet
from ._utils import ensure_rng, flex_get


def _ensure_solution_set(solution: Any) -> SolutionSet:
    if isinstance(solution, SolutionSet):
        return solution
    return SolutionSet(solution)


def _extract_dec_matrix(solution_set: SolutionSet) -> np.ndarray:
    decs = solution_set.decs()
    return np.asarray(decs, dtype=float)


def search_pso(*args):
    """Particle swarm optimization particle fly operator."""

    mode = args[-1]
    if mode == "execute":
        solution_obj = args[0]
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        inertia = float(np.asarray(para).reshape(-1)[0]) if para is not None else 0.5

        sol_set = _ensure_solution_set(solution_obj)
        decs = _extract_dec_matrix(sol_set)
        n, d = decs.shape

        if aux is None or not isinstance(aux, dict):
            aux = {}

        if "Pbest" not in aux:
            aux["Pbest"] = SolutionSet([sol for sol in sol_set])
            # The original PSO component seeds the global guide with a random
            # particle; ParaPSO promotes the best guide after evaluation.
            aux["Gbest"] = aux["Pbest"][int(rng.integers(n))]
            aux["V"] = np.zeros((n, d))

        velocity = np.asarray(aux.get("V", np.zeros((n, d))), dtype=float)
        pbest = _extract_dec_matrix(aux["Pbest"])
        gbest = np.asarray(flex_get(aux["Gbest"], "dec"), dtype=float).reshape(1, -1)

        # MATLAB draws one coefficient per particle and broadcasts it across
        # dimensions, rather than drawing a coefficient per coordinate.
        r1 = rng.random((n, 1))
        r2 = rng.random((n, 1))
        velocity = (
            inertia * velocity + 2 * r1 * (pbest - decs) + 2 * r2 * (gbest - decs)
        )
        offspring = decs + velocity

        aux["V"] = velocity
        return offspring, aux

    if mode == "parameter":
        return [0, 0.5], None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

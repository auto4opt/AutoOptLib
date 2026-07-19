"""Python port of ImproveRate (Utilities/General/ImproveRate.m)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _fitness_from_solution(solution: Any) -> float:
    con = getattr(solution, "cons", None)
    if callable(con):
        con = con()
    obj = getattr(solution, "objs", None)
    if callable(obj):
        obj = obj()
    if obj is None:
        obj = getattr(solution, "obj", None)
    if con is None:
        return float(np.asarray(obj).reshape(1)[0])
    con_sum = float(np.sum(np.maximum(0.0, np.asarray(con))))
    feasible = con_sum <= 0.0
    base = float(np.asarray(obj).reshape(1)[0])
    return base if feasible else (con_sum + 1e8)


def improve_rate(
    solution: Any, improve: Sequence[float] | None, inner_g: int, typ: str
):
    """Compute improvement vector as in MATLAB version.

    Returns the updated ``improve`` vector of length k+1 (default k=3), where
    ``improve[0]`` is the current improvement rate and the remaining entries
    track the best-so-far fitness over last k iterations.
    """
    k = 3
    if inner_g == 1 or improve is None:
        improve = [1.0] + [0.0] * k
    else:
        improve = list(improve)
        if len(improve) < k + 1:
            improve.extend([0.0] * (k + 1 - len(improve)))

    if typ == "solution":
        # support vectorized input or a container with .fits
        fits = getattr(solution, "fits", None)
        if callable(fits):
            fitness = np.asarray(fits()).reshape(-1)
        elif fits is not None:
            fitness = np.asarray(fits).reshape(-1)
        else:
            # compute fitness from each item
            try:
                items = list(solution)
            except TypeError:
                items = [solution]
            fitness = np.asarray(
                [_fitness_from_solution(it) for it in items], dtype=float
            )
        best = float(np.min(fitness)) if fitness.size else float("inf")
    elif typ == "algorithm":
        # expect object exposing avePerformAll or similar
        vals = getattr(solution, "avePerformAll", None)
        if callable(vals):
            arr = np.asarray(vals()).reshape(-1)
        else:
            arr = np.asarray(vals if vals is not None else [], dtype=float)
        best = float(np.min(arr)) if arr.size else float("inf")
    else:
        raise ValueError("typ must be 'solution' or 'algorithm'")

    improve.append(best)
    improve.pop(1)

    if inner_g >= k:
        temp_rate = []
        for i in range(1, k):
            num = improve[i] - improve[i + 1]
            den = improve[i]
            temp_rate.append(num / den if den != 0 else 0.0)
        improve[0] = max(temp_rate) if temp_rate else 1.0

    return improve

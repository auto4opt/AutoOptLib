"""Python port of ImproveRate (Utilities/General/ImproveRate.m)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .._constraint_order import solution_key


class _ImprovementWindow(list):
    """Keep feasibility alongside numeric values without changing the list API."""

    def __init__(self, values, feasibility):
        super().__init__(values)
        self.feasibility = feasibility


def improve_rate(
    solution: Any, improve: Sequence[float] | None, inner_g: int, typ: str
):
    """Compute improvement vector as in MATLAB version.

    Returns the updated ``improve`` vector of length k+1 (default k=3), where
    ``improve[0]`` is the current improvement rate and the remaining entries
    track the best-so-far fitness over last k iterations.
    """
    k = 3
    feasibility = list(getattr(improve, "feasibility", [None] * k))
    if inner_g == 1 or improve is None:
        improve = [1.0] + [0.0] * k
        feasibility = [None] * k
    else:
        improve = list(improve)
        if len(improve) < k + 1:
            improve.extend([0.0] * (k + 1 - len(improve)))

    best_infeasible = None
    if typ == "solution":
        try:
            items = list(solution)
        except TypeError:
            items = None
        if items is not None:
            best_infeasible, best = min(
                map(solution_key, items), default=(True, float("inf"))
            )
        else:
            fits = getattr(solution, "fits", None)
            if fits is None:
                best_infeasible, best = solution_key(solution)
            else:
                if callable(fits):
                    fits = fits()
                fitness = np.asarray(fits).reshape(-1)
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
    feasibility = [*feasibility[1:], best_infeasible]

    if inner_g >= k:
        temp_rate = []
        for i in range(1, k):
            previous = float(improve[i])
            current = float(improve[i + 1])
            previous_class, current_class = feasibility[i - 1 : i + 1]
            if (
                previous_class is not None
                and current_class is not None
                and previous_class != current_class
            ):
                rate = 1.0 if previous_class else -1.0
            elif np.isposinf(previous) and np.isfinite(current):
                rate = 1.0
            elif not np.isfinite(previous) or not np.isfinite(current):
                rate = 0.0
            else:
                scale = max(abs(previous), abs(current), float(np.finfo(float).eps))
                rate = (previous - current) / scale
            temp_rate.append(rate)
        improve[0] = max(temp_rate) if temp_rate else 1.0

    return _ImprovementWindow(improve, feasibility)

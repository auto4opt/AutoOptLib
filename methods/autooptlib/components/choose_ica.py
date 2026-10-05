"""Selection operator for Imperialist Competitive Algorithm (ICA)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import extract_fits


def choose_ica(*args: Any):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        para = args[2] if len(args) > 2 else None

        fitness = extract_fits(solution)
        order = np.argsort(fitness)

        top_k = 1
        if para is not None:
            val = np.asarray(para).reshape(-1)[0]
            try:
                top_k = max(1, int(round(val)))
            except (TypeError, ValueError):
                top_k = 1
        n = len(fitness)
        if n == 0:
            return np.array([], dtype=int), None
        top_k = min(top_k, n)
        return order, {"imperialist_count": top_k}

    if mode == "parameter":
        problem = args[0] if args else None
        population_size = max(1, int(getattr(problem, "N", 1)))
        return np.array([[1, population_size]], dtype=float), None
    if mode == "parameter_type":
        return ["integer"], None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

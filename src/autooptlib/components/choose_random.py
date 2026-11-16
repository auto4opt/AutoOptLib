"""Uniform random parent selection with replacement."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, selection_count


def choose_random(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)
        population_size = int(extract_fits(solution).size)
        if population_size == 0:
            return np.array([], dtype=int), None
        count = selection_count(aux, problem, population_size)
        return rng.integers(0, population_size, size=count, dtype=int), None
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

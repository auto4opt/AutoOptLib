"""Python translation of choose_tournament."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, selection_count


def choose_tournament(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        fitness = extract_fits(solution)
        pop_size = fitness.shape[0]
        if pop_size == 0:
            return np.array([], dtype=int), None

        k = 2
        n = selection_count(aux, problem, pop_size)

        match_index = rng.integers(0, pop_size, size=(n, k))
        match_fitness = fitness[match_index]
        row_min = match_fitness.min(axis=1)
        winners = []
        for i in range(n):
            candidates = np.where(match_fitness[i] == row_min[i])[0]
            winners.append(match_index[i, candidates[0]])
        return np.asarray(winners, dtype=int), None

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

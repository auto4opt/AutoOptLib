"""Two-way stochastic tournament parent selection.

The better contestant wins with probability 0.75, matching the selector
registered by the paper experiment's pinned ParadisEO ``eoFastGA`` runner.
"""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, selection_count

WINNER_PROBABILITY = 0.75


def choose_stochastic_tournament(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)
        fitness = extract_fits(solution)
        population_size = int(fitness.size)
        if population_size == 0:
            return np.array([], dtype=int), None
        count = selection_count(aux, problem, population_size)
        contestants = rng.integers(0, population_size, size=(count, 2))
        first = contestants[:, 0]
        second = contestants[:, 1]
        first_is_better = fitness[first] <= fitness[second]
        better = np.where(first_is_better, first, second)
        worse = np.where(first_is_better, second, first)
        select_better = rng.random(count) < WINNER_PROBABILITY
        return np.where(select_better, better, worse).astype(int), None
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

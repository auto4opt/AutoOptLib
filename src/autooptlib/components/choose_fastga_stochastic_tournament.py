"""eoFastGA stochastic-tournament selector with native draw ordering."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, selection_count

WINNER_PROBABILITY = 0.75


def choose_fastga_stochastic_tournament(*args):
    """Run each two-way tournament before drawing the next tournament.

    The generic vectorized selector has the same marginal distribution, but it
    draws all contestants before all winner coins.  eoFastGA calls its
    ``eoSelectOne`` object once per parent, interleaving two contestant draws
    and one coin for every selection.  The distinction affects every later
    stochastic event when a crossover requests two parents.
    """

    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        auxiliary = args[3] if len(args) > 3 else None
        rng = ensure_rng(auxiliary, problem)
        fitness = extract_fits(solution)
        population_size = int(fitness.size)
        if population_size == 0:
            return np.empty(0, dtype=int), None
        count = selection_count(auxiliary, problem, population_size)
        selected = np.empty(count, dtype=int)
        for index in range(count):
            first = int(rng.integers(0, population_size))
            second = int(rng.integers(0, population_size))
            return_better = bool(rng.random() < WINNER_PROBABILITY)
            if fitness[first] <= fitness[second]:
                selected[index] = first if return_better else second
            else:
                selected[index] = second if return_better else first
        return selected, None
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

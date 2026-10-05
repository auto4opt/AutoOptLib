"""Uniform selection from the best fraction of a population."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, selection_count


def choose_elite_fraction(*args):
    """Select parents uniformly from the best ``p`` fraction.

    ``p`` is represented on the closed interval ``[0, 1]`` so the generic
    parameter machinery can optimize it without an artificial positive lower
    bound.  The effective elite set is never empty: ``p == 0`` selects only
    the best individual and ``p == 1`` selects from the whole population.
    """

    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        parameter = args[2] if len(args) > 2 else None
        auxiliary = args[3] if len(args) > 3 else None
        rng = ensure_rng(auxiliary, problem)

        fitness = extract_fits(solution)
        population_size = int(fitness.size)
        if population_size == 0:
            return np.array([], dtype=int), None
        fraction = (
            float(np.asarray(parameter, dtype=float).reshape(-1)[0])
            if parameter is not None
            else 0.2
        )
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("Elite fraction p must be finite and in [0, 1].")

        elite_count = max(1, int(np.ceil(fraction * population_size)))
        # Stable ordering makes ties reproducible; sampling within the elite
        # set remains stochastic and uses the run's shared RNG.
        elite = np.argsort(fitness, kind="stable")[:elite_count]
        count = selection_count(auxiliary, problem, population_size)
        selected = elite[rng.integers(0, elite_count, size=count)]
        return np.asarray(selected, dtype=int), None

    if mode == "parameter":
        return [0.0, 1.0], None

    if mode == "behavior":
        return [["LS", "small"], ["GS", "large"]], None

    raise ValueError(f"Unsupported mode: {mode}")

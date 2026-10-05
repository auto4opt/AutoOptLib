"""Python translation of choose_roulette_wheel."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, selection_count


def choose_roulette_wheel(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        fitness = extract_fits(solution)
        if fitness.size == 0:
            return np.array([], dtype=int), None
        n = selection_count(aux, problem, fitness.shape[0])

        # ParadisEO's proportional selector samples in direct proportion to a
        # maximized fitness. AutoOptLib stores minimization costs, so the exact
        # equivalent weights are ``worst_cost - cost`` (plus a tiny positive
        # floor so equal/worst individuals remain selectable).
        finite = np.isfinite(fitness)
        if not np.any(finite):
            return rng.integers(0, fitness.size, size=n, dtype=int), None
        scale = max(1.0, float(np.max(np.abs(fitness[finite]))))
        normalized = fitness / scale
        worst = float(np.max(normalized[finite]))
        floor = np.finfo(float).eps
        # The positive floor belongs only to valid individuals.  Adding it to
        # the whole vector made NaN/Inf candidates selectable with a tiny but
        # real probability.
        # Normalize before subtraction so opposite-sign finite values near the
        # floating-point limit cannot overflow to infinite weights.
        weights = np.where(finite, worst - normalized + floor, 0.0)
        cdf = np.cumsum(weights)
        cdf /= cdf[-1]

        random_values = rng.random(n)
        indices = np.searchsorted(cdf, random_values, side="left")
        return indices.astype(int), None

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

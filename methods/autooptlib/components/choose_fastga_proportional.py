"""ParadisEO proportional selection with a defined signed-fitness extension."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, selection_count


def choose_fastga_proportional(*args):
    """Select in proportion to the original maximizing PBO fitness.

    AutoOptLib stores PBO fitness as a minimization cost ``-raw``.  The native
    eoFastGA runner stores ``raw`` in an ``eoMaximizingFitness`` and its
    proportional selector uses those values directly when they are finite and
    non-negative.  Several transformed IOH PBO instances legitimately return
    negative raw values, however.  ``eoProportionalSelect`` then calls
    ``std::upper_bound`` on a decreasing/non-partitioned cumulative array, for
    which the C++ algorithm's precondition is not satisfied.  Preserve exact
    native behaviour on its valid domain and use a stable affine shift on the
    signed domain so every Search-v6 graph remains executable.
    """

    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        auxiliary = args[3] if len(args) > 3 else None
        rng = ensure_rng(auxiliary, problem)
        costs = extract_fits(solution)
        if costs.size == 0:
            return np.empty(0, dtype=int), None
        count = selection_count(auxiliary, problem, costs.size)
        raw_fitness = -np.asarray(costs, dtype=float)
        finite = np.isfinite(raw_fitness)
        if np.all(finite) and np.all(raw_fitness >= 0.0):
            # Exact eoProportionalSelect path.  In particular, an all-zero
            # population falls through to the final individual below.
            weights = raw_fitness
        elif np.any(finite):
            # Normalize before shifting so opposite-sign values near the
            # floating-point limit cannot overflow during subtraction.
            scale = max(1.0, float(np.max(np.abs(raw_fitness[finite]))))
            normalized = raw_fitness[finite] / scale
            shifted = normalized - float(np.min(normalized))
            if not np.any(shifted > 0.0):
                shifted = np.ones_like(shifted)
            else:
                # Keep every finite individual selectable, including the
                # current worst.  Invalid/non-finite records retain zero mass.
                shifted = shifted + np.finfo(float).eps
            weights = np.zeros_like(raw_fitness)
            weights[finite] = shifted
        else:
            # This is defensive for custom objectives.  Official PBO tasks are
            # finite, but a malformed population must not crash the complete
            # design run or make a non-finite candidate selectable by weight.
            return rng.integers(0, costs.size, size=count, dtype=int), None

        cumulative = np.cumsum(weights, dtype=float)
        fortunes = rng.random(count) * float(cumulative[-1])
        # std::upper_bound is equivalent to side='right'.  When the total is
        # zero, the native selector falls through to population.back().
        selected = np.searchsorted(cumulative, fortunes, side="right")
        selected = np.minimum(selected, costs.size - 1)
        return selected.astype(int), None
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

"""Current-to-pbest/1 differential-evolution trial-vector generator."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, flex_get


def _decisions(parent_obj) -> np.ndarray:
    decisions = flex_get(parent_obj, "decs", None)
    if decisions is None:
        decisions = parent_obj
    result = np.asarray(decisions, dtype=float)
    if result.ndim != 2:
        raise ValueError("DE parents must form a two-dimensional matrix.")
    return result


def _sample_excluding(
    rng: np.random.Generator,
    population_size: int,
    excluded: set[int],
) -> int:
    available = [index for index in range(population_size) if index not in excluded]
    if available:
        return int(available[int(rng.integers(0, len(available)))])
    # Degenerate lambda/population settings remain executable. In that case
    # distinct DE donors do not exist and reuse is the only total semantics.
    return int(rng.integers(0, population_size))


def search_de_current_to_pbest(*args):
    """Generate trials using current-to-pbest/1 with binomial crossover.

    Parameters are ``F``, ``CR`` and the independent DE elite fraction
    ``p_de``.  At least one donor coordinate is copied into every trial even
    when ``CR == 0``, as required by binomial DE crossover.
    """

    mode = args[-1]
    if mode == "execute":
        parent_obj = args[0]
        problem = args[1] if len(args) > 1 else None
        parameter = args[2] if len(args) > 2 else None
        auxiliary = args[3] if len(args) > 3 else None
        rng = ensure_rng(auxiliary, problem)

        targets = _decisions(parent_obj)
        context = (
            auxiliary.get("_population_context")
            if isinstance(auxiliary, dict)
            else None
        )
        population_obj = parent_obj if context is None else context
        population = _decisions(population_obj)
        population_size, dimension = population.shape
        if population_size == 0 or dimension == 0 or targets.shape[0] == 0:
            return targets.copy(), auxiliary
        if targets.shape[1] != dimension:
            raise ValueError("DE targets and population must have the same dimension.")
        fitness = extract_fits(population_obj)
        if fitness.size != population_size:
            raise ValueError("DE population fitness and decision counts must match.")
        target_indices = None
        fastga_match_first = bool(
            isinstance(auxiliary, dict)
            and auxiliary.get("_fastga_match_first_target", False)
        )
        if isinstance(auxiliary, dict) and not fastga_match_first:
            raw_indices = auxiliary.get("_target_population_indices")
            if raw_indices is not None:
                target_indices = np.asarray(raw_indices, dtype=int).reshape(-1)
                if target_indices.size != targets.shape[0]:
                    target_indices = None

        if parameter is None:
            scale, crossover_rate, elite_fraction = 0.5, 0.5, 0.2
        else:
            values = np.asarray(parameter, dtype=float).reshape(-1)
            if values.size != 3:
                raise ValueError(
                    "current-to-pbest requires parameters F, CR, and p_de."
                )
            scale, crossover_rate, elite_fraction = map(float, values)
        if not all(
            np.isfinite(value) and 0.0 <= value <= 1.0
            for value in (scale, crossover_rate, elite_fraction)
        ):
            raise ValueError("F, CR, and p_de must be finite and in [0, 1].")

        elite_count = max(1, int(np.ceil(elite_fraction * population_size)))
        elite = np.argsort(fitness, kind="stable")[:elite_count]
        target_fitness = extract_fits(parent_obj)
        trials = targets.copy()
        for trial_index in range(targets.shape[0]):
            if fastga_match_first:
                matches = [
                    index
                    for index in range(population_size)
                    if float(fitness[index]) == float(target_fitness[trial_index])
                    and np.array_equal(population[index], targets[trial_index])
                ]
                if not matches:
                    raise ValueError(
                        "FastGA current-to-pbest target is not a clone of the "
                        "current population."
                    )
                target_index = matches[0]
            else:
                target_index = (
                    int(target_indices[trial_index])
                    if target_indices is not None
                    and 0 <= int(target_indices[trial_index]) < population_size
                    else None
                )
            pbest_index = int(elite[int(rng.integers(0, elite_count))])
            first_index = _sample_excluding(
                rng,
                population_size,
                set() if target_index is None else {target_index},
            )
            second_index = _sample_excluding(
                rng,
                population_size,
                {first_index} if target_index is None else {target_index, first_index},
            )
            donor = (
                targets[trial_index]
                + scale * (population[pbest_index] - targets[trial_index])
                + scale * (population[first_index] - population[second_index])
            )
            if fastga_match_first:
                # CurrentToPBestMutation draws the forced coordinate before
                # its coordinate loop and short-circuits the CR coin at that
                # position.  Keep legacy Search's historical vectorized order
                # below unchanged.
                forced = int(rng.integers(0, dimension))
                mask = np.zeros(dimension, dtype=bool)
                for coordinate in range(dimension):
                    mask[coordinate] = coordinate == forced or bool(
                        rng.random() < crossover_rate
                    )
            else:
                mask = rng.random(dimension) < crossover_rate
                mask[int(rng.integers(0, dimension))] = True
            trials[trial_index, mask] = donor[mask]
        return trials, auxiliary

    if mode == "parameter":
        return [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]], None

    if mode == "behavior":
        return [
            ["LS", "small", "large", "small"],
            ["GS", "large", "small", "large"],
        ], None

    raise ValueError(f"Unsupported mode: {mode}")

"""BIPOP-CMA-ES, SHADE and binary ILS comparison algorithms."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from comparisons.shared.objectives import TrackedIOHObjective
from comparisons.shared.protocol import EvaluationTask


def _bipop_cmaes(
    task: EvaluationTask, objective: TrackedIOHObjective
) -> tuple[str, dict[str, Any]]:
    if task.suite != "bbob":
        raise ValueError("BIPOP-CMA-ES is only registered for BBOB.")
    try:
        import cma
    except ImportError as exc:  # pragma: no cover - dependency specific
        raise ImportError("Install the official `cma==4.4.4` package.") from exc
    rng = np.random.default_rng(task.seed)
    lower, upper = objective.bounds

    def x0() -> np.ndarray:
        # pycma calls a callable x0 for every restart. A fixed vector silently
        # restarts every BIPOP run from the same point.
        return rng.uniform(lower, upper)

    sigma0 = float(np.mean(upper - lower) / 5.0)
    options = {
        "bounds": [lower.tolist(), upper.tolist()],
        "maxfevals": int(task.budget),
        "seed": int(task.seed),
        "verbose": -9,
        "verb_log": 0,
        "verb_disp": 0,
    }
    cma.fmin2(
        lambda x: objective(np.asarray(x, dtype=float)),
        x0,
        sigma0,
        options,
        # A fixed restart count can terminate before the registered FE budget
        # on easy functions. Keep the official BIPOP scheduler and make only
        # the experiment's FE cap binding.
        restarts={"maxrestarts": math.inf, "maxfevals": int(task.budget)},
        bipop=True,
    )
    return "CMA-ES/pycma", {
        "version": getattr(cma, "__version__", None),
        "restart_initialization": "fresh seeded uniform x0 per restart",
        "restart_termination": "registered FE budget",
    }


def _positive_cauchy(rng: np.random.Generator, center: float) -> float:
    value = -1.0
    while value <= 0:
        value = center + 0.1 * float(rng.standard_cauchy())
    return min(value, 1.0)


def _update_shade_archive(
    archive: list[np.ndarray],
    additions: list[np.ndarray],
    capacity: int,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Apply the corrected SHADE 1.1.1 author's archive update exactly."""

    for parent in additions:
        candidate = np.array(parent, dtype=float, copy=True)
        if len(archive) < capacity:
            archive.append(candidate)
        else:
            archive[int(rng.integers(capacity))] = candidate
    return archive


def _shade(
    task: EvaluationTask, objective: TrackedIOHObjective
) -> tuple[str, dict[str, Any]]:
    """SHADE 1.1.1 author mechanics with the protocol's common population."""

    if task.suite != "bbob":
        raise ValueError("SHADE is only registered for BBOB.")
    if task.dimension < 1 or task.budget < 1:
        raise ValueError("SHADE requires a positive dimension and FE budget.")
    rng = np.random.default_rng(task.seed)
    lower, upper = objective.bounds.astype(float)
    dimension = task.dimension
    population_size = min(task.budget, max(4, int(task.population_size)))
    population = rng.uniform(lower, upper, size=(population_size, dimension))
    fitness = np.asarray([objective(row) for row in population])
    memory_size = dimension
    memory_f = np.full(memory_size, 0.5)
    memory_cr = np.full(memory_size, 0.5)
    memory_index = 0
    archive_capacity = 2 * population_size
    archive: list[np.ndarray] = []
    p_best_rate = 0.1
    p_count = min(
        population_size,
        max(2, int(math.floor(population_size * p_best_rate + 0.5))),
    )
    strict_improvements = 0
    tie_replacements = 0

    while objective.evaluations < task.budget:
        successful_f: list[float] = []
        successful_cr: list[float] = []
        improvements: list[float] = []
        archive_additions: list[np.ndarray] = []
        trials: list[np.ndarray] = []
        trial_f: list[float] = []
        trial_cr: list[float] = []
        order = np.argsort(fitness)
        union = np.vstack((population, np.asarray(archive))) if archive else population
        for i in range(population_size):
            if objective.evaluations + len(trials) >= task.budget:
                break
            memory_slot = int(rng.integers(memory_size))
            scale = _positive_cauchy(rng, float(memory_f[memory_slot]))
            crossover = (
                0.0
                if memory_cr[memory_slot] == -1
                else float(np.clip(rng.normal(memory_cr[memory_slot], 0.1), 0.0, 1.0))
            )
            pbest = int(rng.choice(order[:p_count]))
            choices = np.delete(np.arange(population_size), i)
            r1 = int(rng.choice(choices))
            union_choices = np.arange(len(union))
            forbidden = {i, r1}
            valid_r2 = [index for index in union_choices if index not in forbidden]
            r2 = int(rng.choice(valid_r2))
            mutant = population[i] + scale * (
                population[pbest] - population[i] + population[r1] - union[r2]
            )
            # SHADE midpoint repair from the corrected author implementation.
            below = mutant < lower
            above = mutant > upper
            mutant[below] = (lower[below] + population[i][below]) / 2
            mutant[above] = (upper[above] + population[i][above]) / 2
            mask = rng.random(dimension) < crossover
            mask[int(rng.integers(dimension))] = True
            trials.append(np.where(mask, mutant, population[i]))
            trial_f.append(scale)
            trial_cr.append(crossover)

        if not trials:
            break
        trial_values = np.asarray([objective(row) for row in trials])
        for i, (trial, value, scale, crossover) in enumerate(
            zip(trials, trial_values, trial_f, trial_cr)
        ):
            if value <= fitness[i]:
                previous_fitness = float(fitness[i])
                if value < previous_fitness:
                    archive_additions.append(population[i].copy())
                    improvement = float(previous_fitness - value)
                    successful_f.append(scale)
                    successful_cr.append(crossover)
                    improvements.append(improvement)
                    strict_improvements += 1
                else:
                    # The author's corrected 1.1.1 source accepts equal-fitness
                    # offspring but excludes them from the archive and success
                    # history used to update M_F and M_CR.
                    tie_replacements += 1
                population[i] = trial
                fitness[i] = value
        if archive_additions:
            archive = _update_shade_archive(
                archive, archive_additions, archive_capacity, rng
            )
        if successful_f:
            weights = np.asarray(improvements, dtype=float)
            weights /= weights.sum()
            sf = np.asarray(successful_f)
            scr = np.asarray(successful_cr)
            memory_f[memory_index] = float(
                np.sum(weights * sf**2) / np.sum(weights * sf)
            )
            cr_denominator = float(np.sum(weights * scr))
            memory_cr[memory_index] = (
                -1.0
                if cr_denominator == 0.0
                else float(np.sum(weights * scr**2) / cr_denominator)
            )
            memory_index = (memory_index + 1) % memory_size
    return "SHADE 1.1.1 author mechanics (common-population protocol)", {
        "version": "1.1.1",
        "author_source": "https://ryojitanabe.github.io/publication",
        "population_size": population_size,
        "memory_size": memory_size,
        "archive_capacity": archive_capacity,
        "archive_rate": 2.0,
        "archive_update": "append until full, then random replacement per parent",
        "tie_selection": "replace parent without archive or history update",
        "strict_improvements": strict_improvements,
        "tie_replacements": tie_replacements,
        "archive_entries": len(archive),
        "terminal_cr_memory": "-1 when all successful CR values are zero",
        "p_best_rate": p_best_rate,
        "p_best_count": p_count,
        "common_population_override": True,
    }


def _iterated_local_search(
    task: EvaluationTask, objective: TrackedIOHObjective
) -> tuple[str, dict[str, Any]]:
    """Binary ILS with exact one-bit local optima and two-bit perturbations."""

    if task.suite != "pbo":
        raise ValueError("The registered binary ILS is only available for PBO.")
    if task.dimension < 1 or task.budget < 1:
        raise ValueError("ILS requires a positive dimension and FE budget.")
    rng = np.random.default_rng(task.seed)
    local_sweeps = 0
    accepted_local_moves = 0
    perturbations = 0
    accepted_perturbations = 0

    def local_search(decision: np.ndarray, cost: float) -> tuple[np.ndarray, float]:
        nonlocal local_sweeps, accepted_local_moves
        current = np.array(decision, dtype=int, copy=True)
        current_cost = float(cost)
        while objective.evaluations < task.budget:
            improved = False
            for coordinate in rng.permutation(task.dimension):
                if objective.evaluations >= task.budget:
                    return current, current_cost
                neighbor = current.copy()
                neighbor[int(coordinate)] = 1 - neighbor[int(coordinate)]
                neighbor_cost = float(objective(neighbor))
                if neighbor_cost < current_cost:
                    current = neighbor
                    current_cost = neighbor_cost
                    accepted_local_moves += 1
                    improved = True
                    break
            local_sweeps += 1
            if not improved:
                return current, current_cost
        return current, current_cost

    incumbent = rng.integers(0, 2, size=task.dimension, dtype=int)
    incumbent_cost = float(objective(incumbent))
    incumbent, incumbent_cost = local_search(incumbent, incumbent_cost)
    perturbation_strength = min(2, task.dimension)
    while objective.evaluations < task.budget:
        candidate = incumbent.copy()
        coordinates = rng.choice(
            task.dimension, size=perturbation_strength, replace=False
        )
        candidate[coordinates] = 1 - candidate[coordinates]
        candidate_cost = float(objective(candidate))
        perturbations += 1
        candidate, candidate_cost = local_search(candidate, candidate_cost)
        if candidate_cost < incumbent_cost:
            incumbent = candidate
            incumbent_cost = candidate_cost
            accepted_perturbations += 1

    return "Binary iterated local search", {
        "local_search": "random-order first-improvement one-bit neighborhood",
        "perturbation": "flip two distinct bits (one when D=1)",
        "acceptance": "strictly improving local optimum",
        "perturbation_strength": perturbation_strength,
        "local_sweeps": local_sweeps,
        "accepted_local_moves": accepted_local_moves,
        "perturbations": perturbations,
        "accepted_perturbations": accepted_perturbations,
    }

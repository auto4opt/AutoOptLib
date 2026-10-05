"""Steady-state GA replacement primitives."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ._utils import ensure_rng, flex_get, solution_as_list

VictimSelector = Callable[[list[Any], np.random.Generator], int]


def _split_parents_offspring(
    solution: Any, problem: Any
) -> tuple[list[Any], list[Any]]:
    values = solution_as_list(solution)
    mu = int(flex_get(problem, "N", len(values) // 2))
    if mu <= 0 or len(values) < mu:
        raise ValueError("SSGA replacement requires a non-empty parent population.")
    parents = list(values[:mu])
    offspring = list(values[mu:])
    if len(offspring) > mu:
        raise ValueError("SSGA replacement requires offspring_size <= population_size.")
    return parents, offspring


def replace_parents(
    solution: Any,
    problem: Any,
    auxiliary: Any,
    victim_selector: VictimSelector,
) -> list[Any]:
    """Remove one parent per offspring, then insert every offspring."""

    parents, offspring = _split_parents_offspring(solution, problem)
    rng = ensure_rng(auxiliary)
    for _child in offspring:
        victim = int(victim_selector(parents, rng))
        if not 0 <= victim < len(parents):
            raise RuntimeError("SSGA victim selector returned an invalid index.")
        parents.pop(victim)
    parents.extend(offspring)
    return parents


def replace_parents_fastga(
    solution: Any,
    problem: Any,
    auxiliary: Any,
    victim_selector: VictimSelector,
) -> list[Any]:
    """Match eoReduceMerge, including its zero-survivor fast path.

    ParadisEO's tournament truncate returns immediately when lambda equals mu;
    it does not consume victim-selection RNG in that case.  The legacy Search
    replacement keeps its historical behaviour in :func:`replace_parents`.
    """

    parents, offspring = _split_parents_offspring(solution, problem)
    survivors = len(parents) - len(offspring)
    if survivors == 0:
        return offspring
    rng = ensure_rng(auxiliary)
    while len(parents) > survivors:
        victim = int(victim_selector(parents, rng))
        if not 0 <= victim < len(parents):
            raise RuntimeError("SSGA victim selector returned an invalid index.")
        parents.pop(victim)
    parents.extend(offspring)
    return parents


def worst_victim(parents: list[Any], _rng: np.random.Generator) -> int:
    return int(np.argmax([float(parent.fit) for parent in parents]))


def deterministic_tournament_victim(
    parents: list[Any], rng: np.random.Generator
) -> int:
    contestants = rng.integers(0, len(parents), size=2)
    first, second = int(contestants[0]), int(contestants[1])
    return first if parents[first].fit >= parents[second].fit else second


def stochastic_tournament_victim(parents: list[Any], rng: np.random.Generator) -> int:
    contestants = rng.integers(0, len(parents), size=2)
    first, second = int(contestants[0]), int(contestants[1])
    worse, better = (
        (first, second)
        if parents[first].fit >= parents[second].fit
        else (second, first)
    )
    return worse if float(rng.random()) < 0.75 else better

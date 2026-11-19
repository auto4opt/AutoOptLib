"""Run registered components with JSON-shaped populations over local MCP."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace
from typing import Literal
from uuid import uuid4

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from . import Solution, SolutionSet, get_component, get_component_contract

Number = int | float


class ComponentPopulation(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    decisions: list[list[Number]] = Field(min_length=1)
    fitness: list[float] | None = None

    def solutions(self, dimension: int, integral: bool) -> SolutionSet:
        values = np.asarray(self.decisions)
        if values.shape != (len(self.decisions), dimension):
            raise ValueError(
                "Decisions must form a matrix matching the bounds dimension."
            )
        if integral and np.any(values != np.floor(values)):
            raise ValueError(
                "Discrete and permutation components require integer decisions."
            )
        values = values.astype(np.int64 if integral else np.float64)
        fitness = (
            self.fitness if self.fitness is not None else [float("nan")] * len(values)
        )
        if len(fitness) != len(values):
            raise ValueError("Supply one fitness value per decision row.")
        return SolutionSet(Solution(x, f, 0.0, f) for x, f in zip(values, fitness))


class ComponentProblem(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    bounds: tuple[list[Number], list[Number]]
    problem_type: Literal["continuous", "discrete", "permutation"] = "continuous"
    population_size: int | None = Field(default=None, gt=0)
    max_generations: int = Field(default=20, gt=0)


class ComponentSession:
    """Retain engine-shaped state for successive component calls in this process."""

    def __init__(self):
        self.states: OrderedDict[str, dict] = OrderedDict()

    def apply(
        self,
        name,
        population,
        domain,
        parameters,
        backend,
        seed,
        state_id,
        generation,
        inner_generation,
        count,
        archive,
        active_search,
        context,
        targets,
        data,
    ):
        contract = get_component_contract(name)
        bounds = np.asarray(domain.bounds)
        if not bounds.shape[1] or np.any(bounds[0] > bounds[1]):
            raise ValueError("Supply ordered bounds of shape (2, dimension).")
        dimension = bounds.shape[1]
        kind = domain.problem_type
        integral = kind != "continuous"
        if integral and np.any(bounds != np.floor(bounds)):
            raise ValueError(
                "Discrete and permutation components require integer bounds."
            )
        if kind not in contract.problem_types or dimension < contract.min_dimension:
            raise ValueError(f"{name} does not support this problem type or dimension.")
        if contract.binary and not (np.all(bounds[0] == 0) and np.all(bounds[1] == 1)):
            raise ValueError(f"{name} requires binary bounds.")
        contract.validate_parameters(parameters, dimension)

        solutions = population.solutions(dimension, integral)
        if contract.needs_fitness and population.fitness is None:
            raise ValueError(f"{name} requires fitness for every population row.")
        problem = SimpleNamespace(
            N=domain.population_size or len(solutions),
            Gmax=domain.max_generations,
            type=[kind, "static", "certain"],
            dimension=dimension,
            bound=bounds.astype(np.int64 if integral else np.float64),
        )
        if state_id is not None and state_id not in self.states:
            raise ValueError(
                "Component state is unknown or expired in this MCP process."
            )
        state = (
            deepcopy(self.states[state_id])
            if state_id
            else {"rng": np.random.default_rng(seed)}
        )
        rng = state.setdefault("rng", np.random.default_rng(seed))
        problem.rng = rng
        component = get_component(name, native=backend == "native")
        result = {}

        if contract.category == "parameter":
            if name.startswith("para_cma") and "cma_Disturb" not in state:
                raise ValueError("Run search_cma first and pass its state_id.")
            if name == "para_pso" and "Pbest" not in state:
                problem_record = get_component("search_pso", native=backend == "native")
                _, state = problem_record(
                    solutions,
                    problem,
                    None,
                    state,
                    generation,
                    inner_generation,
                    data,
                    "execute",
                )
            updated = component(solutions, problem, state, "solution")
        elif contract.category == "archive":
            if isinstance(archive, ComponentPopulation):
                archive = list(archive.solutions(dimension, integral))
            value, updated = component(
                solutions,
                [] if archive is None else archive,
                problem,
                parameters,
                generation,
                inner_generation,
                data,
                "execute",
            )
            if name == "archive_statistic":
                result["statistics"] = _numeric_rows(value)
            elif value is not None:
                result.update(_population_result(value))
        else:
            transient = {
                "_active_search_operator": active_search,
                "_selection_count": count,
                "_population_context": context.solutions(dimension, integral)
                if context
                else None,
                "_target_population_indices": np.asarray(targets, dtype=np.int64)
                if targets is not None
                else None,
            }
            state.update(
                {key: value for key, value in transient.items() if value is not None}
            )
            value, updated = component(
                solutions,
                problem,
                parameters,
                state,
                generation,
                inner_generation,
                data,
                "execute",
            )
            if contract.category == "choose":
                result["indices"] = np.asarray(value).tolist()
            elif contract.category == "update":
                result.update(_population_result(value))
            else:
                result["decisions"] = _numeric_rows(value)
        if isinstance(updated, dict):
            state = updated
        state.setdefault("rng", rng)
        for key in (
            "_selection_count",
            "_population_context",
            "_target_population_indices",
        ):
            state.pop(key, None)
        invalidated = state.pop("_stream_invalidated_mask", None)
        if invalidated is not None:
            result["invalidated"] = np.asarray(invalidated, dtype=bool).tolist()
        new_id = uuid4().hex
        self.states[new_id] = state
        if len(self.states) > 64:
            self.states.popitem(last=False)
        return {"state_id": new_id, **result}


def _numeric_rows(value):
    array = np.asarray(value)
    return np.where(np.isfinite(array), array, None).tolist()


def _population_result(value):
    rows = list(value)
    return {
        "decisions": _numeric_rows([row.dec for row in rows]),
        "fitness": _numeric_rows([row.fit for row in rows]),
    }

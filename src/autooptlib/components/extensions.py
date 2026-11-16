"""Typed Python extension protocol, adapted once to the existing executor."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable

import numpy as np

from .contracts import ComponentContract


@dataclass(frozen=True)
class ComponentRequest:
    source: Any
    problem: Any
    parameters: tuple[float, ...] | None
    state: dict
    generation: int
    inner: int
    data: Any
    population: Any = None
    selection_count: int | None = None
    target_indices: Any = None
    offspring_size: int | None = None
    active_search: str | None = None

    @property
    def rng(self):
        return self.state["rng"]

    @property
    def decisions(self):
        from ._utils import decision_matrix

        return decision_matrix(self.source)


@dataclass(frozen=True)
class ComponentResult:
    value: Any
    state: dict


@dataclass(frozen=True)
class _TypedComponent:
    function: Callable
    contract: ComponentContract
    search_bounds: tuple

    def __call__(self, *args):
        mode = args[-1]
        if mode == "parameter":
            return (
                None if not self.search_bounds else np.array(self.search_bounds)
            ), None
        if mode == "parameter_type":
            return tuple(p.kind for p in self.contract.parameters), None
        if mode == "behavior":
            return ["", ""], None
        if mode not in {"execute", "solution", "algorithm"}:
            raise ValueError(f"Unsupported extension mode {mode}")
        values = list(args[:-1]) + [None] * 7
        source, problem, parameters, state, generation, inner, data = values[:7]
        state = {} if state is None else state
        if not isinstance(state, dict) or "rng" not in state:
            raise ValueError(f"{self.contract.name} requires the runtime state/rng")
        outer_state = state
        state = dict(
            outer_state.get("_typed_components", {}).get(self.contract.name, {})
        )
        state["rng"] = outer_state["rng"]
        self.contract.validate_parameters(
            parameters, np.asarray(problem.bound).shape[1]
        )
        if self.contract.needs_fitness:
            if not all(getattr(row, "fit", None) is not None for row in source):
                raise ValueError(f"{self.contract.name} requires evaluated source rows")
        if self.contract.input_kind == "decisions":
            from ._utils import decision_matrix

            source = decision_matrix(source)
        if parameters is None and self.contract.parameters:
            defaults = tuple(p.default for p in self.contract.parameters)
            if any(v is None for v in defaults):
                raise ValueError(f"{self.contract.name} requires explicit parameters")
            parameters = defaults
        request = ComponentRequest(
            source,
            problem,
            None if parameters is None else tuple(float(v) for v in parameters),
            state,
            0 if generation is None else int(generation),
            1 if inner is None else int(inner),
            data,
            population=outer_state.get("_population_context", source),
            selection_count=outer_state.get("_selection_count"),
            target_indices=outer_state.get("_target_population_indices"),
            offspring_size=outer_state.get("_stream_generation_target"),
            active_search=outer_state.get("_active_search_operator"),
        )
        authoritative_rng = state["rng"]
        result = self.function(request)
        if not isinstance(result, ComponentResult) or not isinstance(
            result.state, dict
        ):
            raise TypeError(
                f"{self.contract.name} must return ComponentResult(value, state)"
            )
        if result.state.get("rng") is not authoritative_rng:
            raise ValueError(f"{self.contract.name} must retain the authoritative rng")
        if self.contract.category == "choose":
            indices = np.asarray(result.value)
            if (
                indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or np.any(indices < 0)
                or np.any(indices >= len(source))
            ):
                raise ValueError(
                    f"{self.contract.name} returned invalid selection indices"
                )
        elif self.contract.category == "search":
            a = np.asarray(result.value)
            if (
                a.ndim != 2
                or a.shape[1] != np.asarray(problem.bound).shape[1]
                or a.dtype.kind not in "fiu"
                or not np.all(np.isfinite(a))
            ):
                raise ValueError(f"{self.contract.name} returned invalid decisions")
        elif self.contract.category == "update":
            from ..utils.solve import SolutionSet

            if not isinstance(result.value, SolutionSet):
                raise TypeError(
                    f"{self.contract.name} must return a SolutionSet of survivors"
                )
        for field in self.contract.state_fields:
            if field not in result.state:
                raise ValueError(
                    f"{self.contract.name} did not return declared state field {field}"
                )
        merged = dict(outer_state)
        components = dict(outer_state.get("_typed_components", {}))
        components[self.contract.name] = result.state
        merged["_typed_components"] = components
        return result.value, merged


def register_operator(
    name: str,
    function: Callable,
    *,
    contract: ComponentContract,
    search_bounds=(),
    replace_existing=False,
):
    """Register a typed extension for real Python solve/search/worker execution."""
    from . import register_component

    if not callable(function) or contract.name != name:
        raise ValueError("Extension name/function must match its contract")
    if contract.backends != ("python",):
        raise ValueError("Python extensions must declare only the python backend")
    if contract.category not in {"choose", "search", "update"}:
        raise ValueError("Typed extensions currently execute as choose/search/update")
    if contract.parent_arity != (2 if name.startswith("cross_") else 1):
        raise ValueError("Two-parent operators must use the cross_ naming family")
    bounds = tuple(tuple(float(v) for v in row) for row in search_bounds)
    if len(bounds) != len(contract.parameters) or any(len(row) != 2 for row in bounds):
        raise ValueError("One search interval is required per declared parameter")
    for p, row in zip(contract.parameters, bounds):
        if not np.all(np.isfinite(row)) or row[0] > row[1]:
            raise ValueError("Invalid search interval")
        if row[0] < p.lower or row[1] > p.upper:
            raise ValueError("Search interval exceeds execution parameter bounds")
        if p.kind == "integer" and np.ceil(row[0]) > np.floor(row[1]):
            raise ValueError("Search interval contains no legal integer")
    contract = replace(
        contract,
        require_parameters=bool(contract.parameters)
        and any(p.default is None for p in contract.parameters),
    )
    adapter = _TypedComponent(function, contract, bounds)
    register_component(
        name,
        adapter,
        category=contract.category,
        problem_types=contract.problem_types,
        replace=replace_existing,
    )
    return adapter

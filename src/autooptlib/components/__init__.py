"""AutoOpt component library (Python port)."""

from __future__ import annotations

from importlib import import_module
from typing import Callable, Dict, Iterable

_COMPONENT_MODULES = {
    "archive_best": "archive_best",
    "archive_diversity": "archive_diversity",
    "archive_statistic": "archive_statistic",
    "archive_tabu": "archive_tabu",
    "choose_brainstorm": "choose_brainstorm",
    "choose_ica": "choose_ica",
    "choose_traverse": "choose_traverse",
    "choose_tournament": "choose_tournament",
    "choose_roulette_wheel": "choose_roulette_wheel",
    "choose_nich": "choose_nich",
    "update_always": "update_always",
    "update_greedy": "update_greedy",
    "update_round_robin": "update_round_robin",
    "update_pairwise": "update_pairwise",
    "update_simulated_annealing": "update_simulated_annealing",
    "cross_point_one": "cross_point_one",
    "cross_point_two": "cross_point_two",
    "cross_point_uniform": "cross_point_uniform",
    "cross_point_n": "cross_point_n",
    "cross_arithmetic": "cross_arithmetic",
    "cross_sim_binary": "cross_sim_binary",
    "search_de_current": "search_de_current",
    "search_de_current_best": "search_de_current_best",
    "search_de_random": "search_de_random",
    "search_eda": "search_eda",
    "search_ica": "search_ica",
    "search_mu_gaussian": "search_mu_gaussian",
    "search_mu_cauchy": "search_mu_cauchy",
    "search_mu_uniform": "search_mu_uniform",
    "search_mu_polynomial": "search_mu_polynomial",
    "search_cma": "search_cma",
    "search_pso": "search_pso",
    "reinit_continuous": "reinit_continuous",
    "reinit_discrete": "reinit_discrete",
    "reinit_permutation": "reinit_permutation",
    "search_reset_one": "search_reset_one",
    "search_reset_rand": "search_reset_rand",
    "search_reset_creep": "search_reset_creep",
    "search_swap": "search_swap",
    "search_swap_multi": "search_swap_multi",
    "search_scramble": "search_scramble",
    "search_insert": "search_insert",
    "cross_order_two": "cross_order_two",
    "cross_order_n": "cross_order_n",
    "para_cma": "para_cma",
    "para_pso": "para_pso",
    "para_cmaes": "para_cmaes",
}

_cache: Dict[str, Callable] = {}
_custom_components: Dict[str, Callable] = {}
_custom_component_specs: Dict[str, tuple[str, frozenset[str]]] = {}

_VALID_CATEGORIES = {"choose", "search", "update"}
_VALID_PROBLEM_TYPES = {"continuous", "discrete", "permutation"}


def _infer_category(name: str) -> str | None:
    prefix = name.split("_", 1)[0]
    if prefix == "choose":
        return "choose"
    if prefix in {"search", "cross", "reinit"}:
        return "search"
    if prefix == "update":
        return "update"
    return None


def register_component(
    name: str,
    component: Callable,
    *,
    category: str | None = None,
    problem_types: Iterable[str] = ("continuous", "discrete", "permutation"),
    replace: bool = False,
) -> None:
    """Register a user-defined component for the current Python process.

    The callable must follow the same mode-based protocol as built-in
    components.  Registration is explicit so extensions do not need to edit
    AutoOptLib's internal component map.  ``category`` may be ``choose``,
    ``search``, or ``update`` and is inferred from conventional component
    names.  Compatible registered components are included by :func:`space`.
    """
    if not isinstance(name, str) or not name.isidentifier():
        raise ValueError("Component names must be valid Python identifiers.")
    if not callable(component):
        raise TypeError("component must be callable")
    resolved_category = category or _infer_category(name)
    if resolved_category not in _VALID_CATEGORIES:
        raise ValueError(
            "category must be 'choose', 'search', or 'update'; it cannot be "
            f"inferred from {name!r}"
        )
    resolved_types = frozenset(str(value).lower() for value in problem_types)
    if not resolved_types or not resolved_types <= _VALID_PROBLEM_TYPES:
        raise ValueError(
            "problem_types must contain one or more of: continuous, discrete, permutation"
        )
    if not replace and (name in _COMPONENT_MODULES or name in _custom_components):
        raise ValueError(f"Component {name!r} is already registered.")
    _custom_components[name] = component
    _custom_component_specs[name] = (resolved_category, resolved_types)
    _cache.pop(name, None)


def list_components() -> tuple[str, ...]:
    """Return all built-in and user-registered component names."""
    return tuple(sorted(set(_COMPONENT_MODULES) | set(_custom_components)))


def compatible_custom_components(category: str, problem_type: str) -> tuple[str, ...]:
    """Return registered extensions compatible with a design-space section."""
    return tuple(
        name
        for name, (
            registered_category,
            problem_types,
        ) in _custom_component_specs.items()
        if registered_category == category and problem_type in problem_types
    )


def get_component(name: str) -> Callable:
    if name in _custom_components:
        return _custom_components[name]
    if name not in _cache:
        module_name = _COMPONENT_MODULES.get(name)
        if module_name is None:
            raise KeyError(f"Component {name!r} not registered")
        module = import_module(f".{module_name}", package=__name__)
        _cache[name] = getattr(module, name)
    return _cache[name]


__all__ = ["get_component", "list_components", "register_component"]

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
    "choose_random": "choose_random",
    "choose_tournament": "choose_tournament",
    "choose_stochastic_tournament": "choose_stochastic_tournament",
    "choose_fastga_stochastic_tournament": "choose_fastga_stochastic_tournament",
    "choose_roulette_wheel": "choose_roulette_wheel",
    "choose_fastga_proportional": "choose_fastga_proportional",
    "choose_nich": "choose_nich",
    "choose_elite_fraction": "choose_elite_fraction",
    "update_always": "update_always",
    "update_greedy": "update_greedy",
    "update_fastga_plus": "update_fastga_plus",
    "update_iterated_local_search": "update_iterated_local_search",
    "update_round_robin": "update_round_robin",
    "update_pairwise": "update_pairwise",
    "update_simulated_annealing": "update_simulated_annealing",
    "update_ssga_worst": "update_ssga_worst",
    "update_ssga_fastga_worst": "update_ssga_fastga_worst",
    "update_ssga_stochastic_tournament": "update_ssga_stochastic_tournament",
    "update_ssga_fastga_stochastic_tournament": "update_ssga_fastga_stochastic_tournament",
    "update_ssga_deterministic_tournament": "update_ssga_deterministic_tournament",
    "update_ssga_fastga_deterministic_tournament": "update_ssga_fastga_deterministic_tournament",
    "cross_point_one": "cross_point_one",
    "cross_point_two": "cross_point_two",
    "cross_point_uniform": "cross_point_uniform",
    "cross_point_n": "cross_point_n",
    "cross_point_three": "cross_point_three",
    "cross_point_five": "cross_point_five",
    "cross_fastga_uniform": "cross_fastga_uniform",
    "cross_fastga_one": "cross_fastga_one",
    "cross_fastga_three": "cross_fastga_three",
    "cross_fastga_five": "cross_fastga_five",
    "cross_arithmetic": "cross_arithmetic",
    "cross_sim_binary": "cross_sim_binary",
    "search_de_current": "search_de_current",
    "search_de_current_best": "search_de_current_best",
    "search_de_random": "search_de_random",
    "search_de_current_to_pbest": "search_de_current_to_pbest",
    "search_fastga_de_current_to_pbest": "search_fastga_de_current_to_pbest",
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
    "search_reset_n": "search_reset_n",
    "search_reset_rand": "search_reset_rand",
    "search_reset_creep": "search_reset_creep",
    "search_bit_uniform": "search_bit_uniform",
    "search_bit_standard": "search_bit_standard",
    "search_bit_conditional": "search_bit_conditional",
    "search_bit_shifted": "search_bit_shifted",
    "search_bit_normal": "search_bit_normal",
    "search_bit_fast": "search_bit_fast",
    "search_bit_one": "search_bit_one",
    "search_bit_three": "search_bit_three",
    "search_bit_five": "search_bit_five",
    "search_fastga_bit_uniform": "_fastga_bit_mutation",
    "search_fastga_bit_standard": "_fastga_bit_mutation",
    "search_fastga_bit_conditional": "_fastga_bit_mutation",
    "search_fastga_bit_shifted": "_fastga_bit_mutation",
    "search_fastga_bit_normal": "_fastga_bit_mutation",
    "search_fastga_bit_fast": "_fastga_bit_mutation",
    "search_fastga_bit_one": "_fastga_bit_mutation",
    "search_fastga_bit_three": "_fastga_bit_mutation",
    "search_fastga_bit_five": "_fastga_bit_mutation",
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


def component_category(name: str) -> str | None:
    """Return the registered execution category for a component name."""

    if name in _custom_component_specs:
        return _custom_component_specs[name][0]
    if name not in _COMPONENT_MODULES:
        return None
    inferred = _infer_category(name)
    if inferred is not None:
        return inferred
    if str(name).startswith("archive_"):
        return "archive"
    if str(name).startswith("para_"):
        return "parameter"
    return None


def component_parent_arity(name: str) -> int:
    """Return the number of parents consumed per requested offspring.

    Search and mutation components consume one input solution. Every
    crossover component consumes a pair. Keeping this contract beside the
    registry prevents ``lambda=1`` from silently becoming self-crossover.
    Custom components following the public ``cross_*`` naming convention get
    the same behaviour automatically.
    """

    return 2 if str(name).startswith("cross_") else 1


def _custom_component_snapshot() -> tuple[
    tuple[str, Callable, str, tuple[str, ...]], ...
]:
    """Return process-transferable runtime registrations."""

    return tuple(
        (
            name,
            component,
            _custom_component_specs[name][0],
            tuple(sorted(_custom_component_specs[name][1])),
        )
        for name, component in _custom_components.items()
    )


def _restore_custom_components(
    snapshot: Iterable[tuple[str, Callable, str, Iterable[str]]],
) -> None:
    for name, component, category, problem_types in snapshot:
        register_component(
            name,
            component,
            category=category,
            problem_types=problem_types,
            replace=True,
        )


__all__ = [
    "component_category",
    "component_parent_arity",
    "get_component",
    "list_components",
    "register_component",
]

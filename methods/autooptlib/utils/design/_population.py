"""Algorithm-level population and offspring-size design variables."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ._helpers import get_flex

DEFAULT_POPULATION_SIZE_SPACE = tuple(range(4, 101))
DEFAULT_OFFSPRING_SIZE_SPACE = tuple(range(1, 101))
BOUNDARY_HANDLING_SPACE = ("clip", "reflect", "resample")
_MUTATION_PREFIXES = (
    "search_mu_",
    "search_reset_",
    "search_bit_",
    "search_fastga_de_",
    "search_fastga_bit_",
    "search_swap",
    "search_scramble",
    "search_insert",
)


def is_mutation_component(name: Any) -> bool:
    return str(name).startswith(_MUTATION_PREFIXES)


def _positive_integer_space(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a non-empty sequence of positive integers.")
    result: list[int] = []
    for raw in value:
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{name} must contain only positive integers.")
        number = int(raw)
        if float(raw) != number or number <= 0:
            raise ValueError(f"{name} must contain only positive integers.")
        if number not in result:
            result.append(number)
    if not result:
        raise ValueError(f"{name} cannot be empty.")
    return tuple(sorted(result))


def population_size_space(setting: Any) -> tuple[int, ...]:
    value = get_flex(
        setting,
        "PopulationSizeSpace",
        get_flex(setting, "population_size_space", DEFAULT_POPULATION_SIZE_SPACE),
    )
    values = list(_positive_integer_space(value, name="PopulationSizeSpace"))
    fallback = int(get_flex(setting, "ProbN", get_flex(setting, "prob_n", 20)))
    if (
        tuple(values) == DEFAULT_POPULATION_SIZE_SPACE
        and fallback > 0
        and fallback not in values
    ):
        values.append(fallback)
    return _bounded_space(values, setting)


def offspring_size_space(setting: Any) -> tuple[int, ...]:
    value = get_flex(
        setting,
        "OffspringSizeSpace",
        get_flex(setting, "offspring_size_space", DEFAULT_OFFSPRING_SIZE_SPACE),
    )
    values = list(_positive_integer_space(value, name="OffspringSizeSpace"))
    fallback = int(get_flex(setting, "ProbN", get_flex(setting, "prob_n", 20)))
    if (
        tuple(values) == DEFAULT_OFFSPRING_SIZE_SPACE
        and fallback > 0
        and fallback not in values
    ):
        values.append(fallback)
    return _bounded_space(values, setting)


def _bounded_space(values: Sequence[int], setting: Any) -> tuple[int, ...]:
    budget = get_flex(setting, "ProbFE", get_flex(setting, "prob_fe", None))
    if budget is None:
        return tuple(sorted(set(int(value) for value in values)))
    maximum = int(budget)
    bounded = sorted({int(value) for value in values if int(value) <= maximum})
    if bounded:
        return tuple(bounded)
    if maximum <= 0:
        raise ValueError("ProbFE must be a positive integer.")
    raise ValueError("The configured size space has no value within ProbFE.")


def boundary_handling_space(setting: Any) -> tuple[str, ...]:
    """Return active boundary strategies for the current problem domain."""

    problem_type = str(get_flex(setting, "_autoopt_problem_type", "")).lower()
    if problem_type in {"discrete", "permutation"}:
        return ("clip",)
    # Standalone solve/load paths need not pass through ``space()``.  Keep all
    # strategies available when the type is unknown; a prepared noncontinuous
    # design space is explicitly tagged above and therefore canonicalized.
    return BOUNDARY_HANDLING_SPACE


def _component_constraints(candidate: Any | None) -> tuple[bool, bool]:
    pathways_groups = getattr(candidate, "operator_pheno", None) or []
    pathways = pathways_groups[0] if pathways_groups else []
    uses_ssga = any(
        str(getattr(path, "update", "")).startswith("update_ssga_") for path in pathways
    )
    uses_pso = any(
        any(
            str(getattr(step, "primary", "")) == "search_pso"
            for step in getattr(path, "search", ())
        )
        for path in pathways
    )
    return uses_ssga, uses_pso


def default_configuration(
    setting: Any, candidate: Any | None = None
) -> dict[str, int | float | str]:
    requested = int(get_flex(setting, "ProbN", get_flex(setting, "prob_n", 20)))
    population_values = population_size_space(setting)
    population = min(
        population_values, key=lambda value: (abs(value - requested), value)
    )
    offspring_values = offspring_size_space(setting)
    uses_ssga, uses_pso = _component_constraints(candidate)
    if uses_ssga or uses_pso:
        legal_pairs = [
            (population_value, offspring_value)
            for population_value in population_values
            for offspring_value in offspring_values
            if (not uses_ssga or offspring_value <= population_value)
            and (not uses_pso or offspring_value == population_value)
        ]
        if not legal_pairs:
            raise ValueError(
                "PopulationSizeSpace and OffspringSizeSpace contain no legal "
                "pair for the selected stateful population components."
            )
        population, offspring = min(
            legal_pairs,
            key=lambda pair: (
                abs(pair[0] - requested) + abs(pair[1] - requested),
                abs(pair[0] - requested),
                abs(pair[1] - requested),
                pair,
            ),
        )
    else:
        offspring = min(
            offspring_values, key=lambda value: (abs(value - requested), value)
        )
    return {
        "population_size": population,
        "offspring_size": offspring,
        "crossover_rate": float(
            get_flex(setting, "CrossoverRate", get_flex(setting, "crossover_rate", 1.0))
        ),
        "mutation_rate": float(
            get_flex(setting, "MutationRate", get_flex(setting, "mutation_rate", 1.0))
        ),
        "boundary_handling": (
            str(
                get_flex(
                    setting,
                    "BoundaryHandling",
                    get_flex(setting, "boundary_handling", "clip"),
                )
            ).lower()
            if len(boundary_handling_space(setting)) > 1
            else "clip"
        ),
    }


def normalize_configuration(
    configuration: Mapping[str, Any] | None,
    setting: Any | None = None,
) -> dict[str, int | float | str]:
    fallback = default_configuration(setting) if setting is not None else None
    source = dict(configuration or {})
    population = source.get(
        "population_size", None if fallback is None else fallback["population_size"]
    )
    if population is None:
        raise ValueError("Algorithm configuration requires population_size.")
    offspring = source.get("offspring_size", population)
    crossover_rate = source.get(
        "crossover_rate", 1.0 if fallback is None else fallback["crossover_rate"]
    )
    mutation_rate = source.get(
        "mutation_rate", 1.0 if fallback is None else fallback["mutation_rate"]
    )
    boundary_handling = str(
        source.get(
            "boundary_handling",
            "clip" if fallback is None else fallback["boundary_handling"],
        )
    ).lower()
    if setting is not None and len(boundary_handling_space(setting)) == 1:
        boundary_handling = "clip"
    for name, raw in (
        ("population_size", population),
        ("offspring_size", offspring),
    ):
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{name} must be a positive integer.")
        value = int(raw)
        if float(raw) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
        source[name] = value
    for name, raw in (
        ("crossover_rate", crossover_rate),
        ("mutation_rate", mutation_rate),
    ):
        if isinstance(raw, (bool, np.bool_)):
            raise ValueError(f"{name} must be a real number in [0, 1].")
        value = float(raw)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be a finite real number in [0, 1].")
        source[name] = value
    if boundary_handling not in BOUNDARY_HANDLING_SPACE:
        choices = ", ".join(BOUNDARY_HANDLING_SPACE)
        raise ValueError(f"boundary_handling must be one of: {choices}.")
    source["boundary_handling"] = boundary_handling
    return {
        "population_size": int(source["population_size"]),
        "offspring_size": int(source["offspring_size"]),
        "crossover_rate": float(source["crossover_rate"]),
        "mutation_rate": float(source["mutation_rate"]),
        "boundary_handling": str(source["boundary_handling"]),
    }


def configuration_activity(candidate: Any) -> tuple[bool, bool]:
    """Return whether crossover and paired-mutation rates affect execution."""

    pathways_groups = getattr(candidate, "operator_pheno", None) or []
    pathways = pathways_groups[0] if pathways_groups else []
    stream_graph = bool(pathways) and all(hasattr(path, "stages") for path in pathways)
    if stream_graph:
        path_has_crossover = [
            any(
                str(getattr(stage.search, "primary", "")).startswith("cross_")
                for stage in path.stages
            )
            for path in pathways
        ]
        crossover = len(pathways) == 2 and sum(path_has_crossover) == 1
        mutation = False
        for path in pathways:
            crossed = False
            for stage in path.stages:
                primary = str(getattr(stage.search, "primary", ""))
                if crossed and is_mutation_component(primary):
                    mutation = True
                crossed = crossed or primary.startswith("cross_")
        return crossover, mutation
    crossover = any(
        str(getattr(step, "primary", "")).startswith("cross_")
        for path in pathways
        for step in getattr(path, "search", ())
    )
    mutation = any(
        getattr(step, "secondary", None) is not None
        or is_mutation_component(getattr(step, "primary", ""))
        for path in pathways
        for step in getattr(path, "search", ())
    )
    return crossover, mutation


def sample_configuration(
    setting: Any, rng: np.random.Generator, candidate: Any | None = None
) -> dict[str, int | float | str]:
    population_values = population_size_space(setting)
    offspring_values = offspring_size_space(setting)
    crossover_active, mutation_active = (
        configuration_activity(candidate) if candidate is not None else (True, True)
    )
    _, uses_pso = _component_constraints(candidate)
    if uses_pso:
        shared_sizes = tuple(sorted(set(population_values) & set(offspring_values)))
        if not shared_sizes:
            raise ValueError(
                "PSO requires a shared population/offspring size in the configured domains."
            )
        population_size = offspring_size = int(rng.choice(shared_sizes))
    else:
        population_size = int(rng.choice(population_values))
        offspring_size = int(rng.choice(offspring_values))
    return {
        "population_size": population_size,
        "offspring_size": offspring_size,
        "crossover_rate": float(rng.random()) if crossover_active else 1.0,
        "mutation_rate": float(rng.random()) if mutation_active else 1.0,
        "boundary_handling": str(rng.choice(boundary_handling_space(setting))),
    }


def copy_configuration(
    candidate: Any, setting: Any | None = None
) -> dict[str, int | float | str]:
    configuration = getattr(candidate, "configuration", None)
    if configuration is None:
        population = getattr(candidate, "population_size", None)
        offspring = getattr(candidate, "offspring_size", None)
        if population is not None:
            configuration = {
                "population_size": population,
                "offspring_size": population if offspring is None else offspring,
                "crossover_rate": getattr(candidate, "crossover_rate", 1.0),
                "mutation_rate": getattr(candidate, "mutation_rate", 1.0),
                "boundary_handling": getattr(candidate, "boundary_handling", "clip"),
            }
    return normalize_configuration(configuration, setting)


def set_configuration(candidate: Any, configuration: Mapping[str, Any]) -> None:
    normalized = normalize_configuration(configuration)
    # Rates for absent operators cannot affect execution.  Canonicalizing them
    # here prevents phenotype-identical algorithms from evading Search dedup.
    if hasattr(candidate, "operator_pheno"):
        crossover_active, mutation_active = configuration_activity(candidate)
        if not crossover_active:
            normalized["crossover_rate"] = 1.0
        if not mutation_active:
            normalized["mutation_rate"] = 1.0
    candidate.configuration = dict(normalized)
    candidate.population_size = normalized["population_size"]
    candidate.offspring_size = normalized["offspring_size"]
    candidate.crossover_rate = normalized["crossover_rate"]
    candidate.mutation_rate = normalized["mutation_rate"]
    candidate.boundary_handling = normalized["boundary_handling"]


def configuration_is_valid(candidate: Any, configuration: Mapping[str, Any]) -> bool:
    """Return whether algorithm-level sizes satisfy component constraints."""

    try:
        normalized = normalize_configuration(configuration)
    except (TypeError, ValueError, OverflowError):
        return False
    uses_ssga, uses_pso = _component_constraints(candidate)
    return (
        not uses_ssga or normalized["offspring_size"] <= normalized["population_size"]
    ) and (
        not uses_pso or normalized["offspring_size"] == normalized["population_size"]
    )


def repair_configuration(
    candidate: Any,
    configuration: Mapping[str, Any] | None,
    setting: Any,
) -> dict[str, int | float | str]:
    """Project an inherited configuration onto this graph's legal size domain.

    Structural mutation can activate an update operator whose population-size
    contract differs from its parent's (notably SSGA or PSO).  Preserve every
    still-valid algorithm-level value, and choose the nearest legal population /
    offspring pair deterministically instead of allowing an invalid child to
    reach the solver.
    """

    normalized = normalize_configuration(configuration, setting)
    population_values = population_size_space(setting)
    offspring_values = offspring_size_space(setting)
    uses_ssga, uses_pso = _component_constraints(candidate)
    legal_pairs = [
        (population, offspring)
        for population in population_values
        for offspring in offspring_values
        if (not uses_ssga or offspring <= population)
        and (not uses_pso or offspring == population)
    ]
    if not legal_pairs:
        raise ValueError(
            "PopulationSizeSpace and OffspringSizeSpace contain no legal pair "
            "for the selected stateful population components."
        )
    requested_population = int(normalized["population_size"])
    requested_offspring = int(normalized["offspring_size"])
    population, offspring = min(
        legal_pairs,
        key=lambda pair: (
            abs(pair[0] - requested_population) + abs(pair[1] - requested_offspring),
            abs(pair[0] - requested_population),
            abs(pair[1] - requested_offspring),
            pair,
        ),
    )
    normalized["population_size"] = population
    normalized["offspring_size"] = offspring
    return normalized


__all__ = [
    "DEFAULT_OFFSPRING_SIZE_SPACE",
    "DEFAULT_POPULATION_SIZE_SPACE",
    "BOUNDARY_HANDLING_SPACE",
    "boundary_handling_space",
    "copy_configuration",
    "configuration_activity",
    "configuration_is_valid",
    "is_mutation_component",
    "default_configuration",
    "normalize_configuration",
    "offspring_size_space",
    "population_size_space",
    "repair_configuration",
    "sample_configuration",
    "set_configuration",
]

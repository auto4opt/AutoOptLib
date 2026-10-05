"""Search v6.0 stream-graph helpers and exact eoFastGA topology injection.

This module does not launch an experiment.  It supplies a versioned bridge
from the common external eoFastGA configuration to AutoOptLib's unchanged
edge-list genotype under ``GraphSemantics=stream_graph_v2``.  The bridge is
used for representation and behaviour-parity tests before v6.0 is allowed to
consume a formal design budget.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

import numpy as np
from autooptlib.utils.design import Design
from autooptlib.utils.design._helpers import get_flex
from autooptlib.utils.design._stream_graph import (
    SEARCH_VERSION,
    STREAM_GRAPH_SEMANTICS,
    STREAM_IMPLEMENTATION_REVISION,
    named_initial_genotype,
)

from comparisons.shared.platform_target import (
    configuration_is_valid,
    normalize_configuration,
)

IMPLEMENTATION_REVISION = STREAM_IMPLEMENTATION_REVISION

SELECTORS = {
    0: "choose_random",
    1: "choose_fastga_stochastic_tournament",
    2: "choose_traverse",
    3: "choose_fastga_proportional",
    4: "choose_tournament",
    5: "choose_elite_fraction",
}
PBO_CROSSOVERS = {
    0: "cross_fastga_uniform",
    1: "cross_fastga_one",
    2: "cross_fastga_three",
    3: "cross_fastga_five",
}
PBO_MUTATIONS = {
    0: "search_fastga_bit_uniform",
    1: "search_fastga_bit_standard",
    2: "search_fastga_bit_conditional",
    3: "search_fastga_bit_shifted",
    4: "search_fastga_bit_normal",
    5: "search_fastga_bit_fast",
    6: "search_fastga_bit_one",
    7: "search_fastga_bit_three",
    8: "search_fastga_bit_five",
}
REPLACEMENTS = {
    0: "update_fastga_plus",
    1: "update_ssga_fastga_worst",
    2: "update_ssga_fastga_stochastic_tournament",
    3: "update_ssga_fastga_deterministic_tournament",
}


def _mapped_operators(configuration: Mapping[str, Any], suite: str) -> tuple[str, str]:
    mutation_index = int(configuration["mutation"])
    crossover_index = int(configuration["crossover"])
    if suite == "pbo":
        return PBO_CROSSOVERS[crossover_index], PBO_MUTATIONS[mutation_index]
    if suite != "bbob":
        raise ValueError(f"Unsupported suite {suite!r}.")
    if mutation_index != 3:
        raise ValueError(
            "Search v6.0 fixes FastGA's selector topology but intentionally does "
            "not claim exact semantics for the non-DE BBOB mutation operators."
        )
    # Canonical current-to-pbest disables the external crossover branch.  The
    # injected stream graph therefore contains only the native mutation event;
    # retaining an inactive crossover path would create an illegal second-stage
    # DE target even when its routing probability is zero.
    return "cross_point_uniform", "search_fastga_de_current_to_pbest"


def fastga_genotype(
    source_configuration: Mapping[str, Any], suite: str, setting: Any
) -> tuple[list[np.ndarray], list[list[Any]], dict[str, Any]]:
    """Return a v6.0 edge genotype with all three eoFastGA selectors intact."""

    semantics = str(get_flex(setting, "GraphSemantics", "")).lower()
    if semantics != STREAM_GRAPH_SEMANTICS:
        raise ValueError("FastGA injection requires GraphSemantics='stream_graph_v2'.")
    if (
        int(get_flex(setting, "alg_p", 0)) != 2
        or int(get_flex(setting, "alg_q", 0)) != 2
    ):
        raise ValueError("FastGA injection requires AlgP=2 and AlgQ=2.")
    specification = fastga_initial_spec(source_configuration, suite)
    return named_initial_genotype(specification, setting)


def fastga_initial_spec(
    source_configuration: Mapping[str, Any],
    suite: str,
    *,
    source_method: str | None = None,
) -> dict[str, Any]:
    """Return a portable v6.0 initial-design specification for one FastGA."""

    configuration = normalize_configuration(dict(source_configuration), suite)
    if not configuration_is_valid(suite, configuration):
        raise ValueError(
            "The source configuration is outside the executable eoFastGA "
            f"space for suite {suite!r}."
        )
    crossover, mutation = _mapped_operators(configuration, suite)
    update = REPLACEMENTS[int(configuration["replacement"])]
    crossover_choose = SELECTORS[int(configuration["crossover_selector"])]
    aftercross_choose = SELECTORS[int(configuration["aftercross_selector"])]
    mutation_choose = SELECTORS[int(configuration["mutation_selector"])]

    parameters: dict[str, list[float]] = {}
    if mutation == "search_fastga_de_current_to_pbest":
        parameters[mutation] = [
            configuration["de_f"],
            configuration["de_cr"],
            configuration["de_p"],
        ]
    if any(
        int(configuration[field]) == 5
        for field in (
            "crossover_selector",
            "aftercross_selector",
            "mutation_selector",
        )
    ):
        parameters["choose_elite_fraction"] = [configuration["elite_fraction"]]
    algorithm_configuration = {
        "population_size": int(configuration["population_size"]),
        "offspring_size": int(configuration["offspring_size"]),
        "crossover_rate": float(configuration["crossover_rate"]),
        "mutation_rate": float(configuration["mutation_rate"]),
        "boundary_handling": (
            ("clip", "reflect", "resample")[int(configuration["boundary_handling"])]
            if suite == "bbob"
            else "clip"
        ),
    }
    if mutation == "search_fastga_de_current_to_pbest":
        # The native mutation matches its target against the *current*
        # population.  It must consequently be the first stream stage.  Keep
        # AlgP is an upper bound, so the canonical mutation-only FastGA is one
        # pathway.  Duplicating it would create a second serialization of the
        # same executable algorithm and waste Search candidate identity.
        pathways = [
            {"stages": [{"choose": mutation_choose, "search": mutation}]},
        ]
    else:
        pathways = [
            {
                "stages": [
                    {"choose": crossover_choose, "search": crossover},
                    {"choose": aftercross_choose, "search": mutation},
                ]
            },
            {"stages": [{"choose": mutation_choose, "search": mutation}]},
        ]
    return {
        "schema": "autooptlib.search-initial-design",
        "schema_version": 1,
        "pathways": pathways,
        "update": update,
        "parameters": parameters,
        "configuration": algorithm_configuration,
        "metadata": {
            "initialization_source": "external_fastga_incumbent",
            "source_method": source_method,
            "source_configuration": dict(configuration),
            "search_version": SEARCH_VERSION,
            "implementation_revision": IMPLEMENTATION_REVISION,
        },
    }


def inject_fastga(
    source_configuration: Mapping[str, Any], suite: str, problem: Any, setting: Any
) -> Design:
    """Build a decoded Search v6.0 design without executing the algorithm."""

    operators, parameters, configuration = fastga_genotype(
        source_configuration, suite, setting
    )
    design = Design.from_genotype(
        operators,
        deepcopy(parameters),
        problem,
        setting,
        configuration=configuration,
    )
    design.metadata = {
        "designer": "search_v6_fastga_injection",
        "search_version": SEARCH_VERSION,
        "implementation_revision": IMPLEMENTATION_REVISION,
        "execution_semantics": STREAM_GRAPH_SEMANTICS,
        "semantic_equivalence": "python_reference_audited_native_trace_pending",
        "source_configuration": dict(source_configuration),
    }
    return design


__all__ = [
    "IMPLEMENTATION_REVISION",
    "fastga_genotype",
    "fastga_initial_spec",
    "inject_fastga",
]

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from ..components import compatible_custom_components, get_component
from .design._helpers import get_flex, get_problem_type
from .design._population import offspring_size_space, population_size_space


def _to_namespace(setting: Any) -> SimpleNamespace:
    if isinstance(setting, SimpleNamespace):
        return setting
    if isinstance(setting, dict):
        return SimpleNamespace(**setting)
    # Fallback: copy attributes
    data = {
        k: getattr(setting, k)
        for k in dir(setting)
        if not k.startswith("__") and not callable(getattr(setting, k))
    }
    return SimpleNamespace(**data)


def _to_behavior_matrix(behavior: Any) -> list[list[Any]] | None:
    """Convert component behavior metadata to a list-of-lists structure."""
    if behavior is None:
        return None
    if isinstance(behavior, np.ndarray):
        behavior = behavior.tolist()
    elif isinstance(behavior, tuple):
        behavior = list(behavior)
    elif not isinstance(behavior, list):
        return None

    matrix: list[list[Any]] = []
    for row in behavior:
        if isinstance(row, np.ndarray):
            matrix.append(row.tolist())
        elif isinstance(row, (list, tuple)):
            matrix.append(list(row))
        elif row is None:
            matrix.append([])
        else:
            matrix.append([row])
    return matrix


def _parameter_types(component: Any, problem: Any, count: int) -> tuple[str, ...]:
    """Return explicit continuous/integer parameter-domain metadata."""
    try:
        raw, _ = component(problem, "parameter_type")
    except NotImplementedError:
        raw = None
    except ValueError as exc:
        message = str(exc).strip().lower()
        # Legacy/custom components commonly implement their mode switch as
        # ``raise ValueError(mode)``.  Treat that exact sentinel, and explicit
        # unsupported-mode messages, as absence of optional metadata while
        # still surfacing real exceptions raised inside the implementation.
        unsupported = message == "parameter_type" or (
            "mode" in message and "parameter_type" in message
        )
        if not unsupported:
            raise
        raw = None
    if raw is None:
        return ("continuous",) * count
    if isinstance(raw, str):
        raw = [raw] * count
    kinds = tuple(str(value).lower() for value in raw)
    if len(kinds) != count:
        raise ValueError(
            f"Parameter type metadata has {len(kinds)} entries for {count} parameters"
        )
    kinds = tuple("integer" if value == "discrete" else value for value in kinds)
    invalid = sorted(set(kinds) - {"continuous", "integer"})
    if invalid:
        raise ValueError(f"Unsupported parameter type(s): {', '.join(invalid)}")
    return kinds


def _binary_dimensions(problem: Any) -> tuple[int, ...]:
    problems = problem if isinstance(problem, (list, tuple)) else [problem]
    dimensions: list[int] = []
    for item in problems:
        bounds = np.asarray(getattr(item, "bound", None))
        if (
            bounds.ndim != 2
            or bounds.shape[0] != 2
            or bounds.shape[1] == 0
            or not np.all(bounds[0] == 0)
            or not np.all(bounds[1] == 1)
        ):
            return ()
        dimensions.append(int(bounds.shape[1]))
    return tuple(dimensions)


def _decision_dimensions(problem: Any) -> tuple[int, ...]:
    problems = problem if isinstance(problem, (list, tuple)) else [problem]
    dimensions = []
    for item in problems:
        bounds = np.asarray(getattr(item, "bound", None))
        if bounds.ndim != 2 or bounds.shape[0] != 2 or bounds.shape[1] == 0:
            return ()
        dimensions.append(int(bounds.shape[1]))
    return tuple(dimensions)


def space(problem: Any, setting: Any) -> SimpleNamespace:
    """Define the type-specific operator and parameter design spaces."""
    set_obj = _to_namespace(setting)
    ptype = get_problem_type(problem)

    if ptype == "continuous":
        choose = [
            "choose_traverse",
            "choose_random",
            "choose_tournament",
            "choose_stochastic_tournament",
            "choose_nich",
            "choose_elite_fraction",
        ]
        search = [
            "search_de_current",
            "search_de_current_best",
            "search_de_random",
            "search_de_current_to_pbest",
            "cross_arithmetic",
            "cross_sim_binary",
            "cross_point_one",
            "cross_point_two",
            "cross_point_n",
            "cross_point_uniform",
            "search_mu_gaussian",
            "search_mu_cauchy",
            "search_mu_polynomial",
            "search_mu_uniform",
            "search_eda",
            "search_cma",
            "reinit_continuous",
        ]
        if (
            str(get_flex(set_obj, "GraphSemantics", "legacy_pathway_v1")).lower()
            == "stream_graph_v2"
        ):
            search.insert(
                search.index("search_de_current_to_pbest") + 1,
                "search_fastga_de_current_to_pbest",
            )
        dimensions = _decision_dimensions(problem)
        if dimensions and min(dimensions) < 2:
            search.remove("cross_point_two")
            search.remove("cross_point_n")
        if dimensions and min(dimensions) > 3:
            search.insert(search.index("cross_point_uniform"), "cross_point_three")
        if dimensions and min(dimensions) > 5:
            search.insert(search.index("cross_point_uniform"), "cross_point_five")
        update = [
            "update_greedy",
            "update_round_robin",
            "update_pairwise",
            "update_always",
            "update_simulated_annealing",
            "update_ssga_worst",
            "update_ssga_stochastic_tournament",
            "update_ssga_deterministic_tournament",
        ]
    elif ptype == "discrete":
        choose = [
            "choose_traverse",
            "choose_random",
            "choose_tournament",
            "choose_stochastic_tournament",
            "choose_roulette_wheel",
            "choose_nich",
        ]
        search = [
            "cross_point_one",
            "cross_point_two",
            "cross_point_uniform",
            "cross_point_n",
            "search_reset_one",
            "search_reset_n",
            "search_reset_rand",
            "reinit_discrete",
        ]
        binary_dimensions = _binary_dimensions(problem)
        dimensions = _decision_dimensions(problem)
        if dimensions and min(dimensions) < 2:
            search.remove("cross_point_two")
            search.remove("cross_point_n")
        if dimensions and min(dimensions) > 3:
            search.insert(search.index("search_reset_one"), "cross_point_three")
        if dimensions and min(dimensions) > 5:
            search.insert(search.index("search_reset_one"), "cross_point_five")
        if binary_dimensions:
            search.extend(
                [
                    "search_bit_uniform",
                    "search_bit_standard",
                    "search_bit_conditional",
                    "search_bit_shifted",
                    "search_bit_normal",
                    "search_bit_fast",
                    "search_bit_one",
                ]
            )
            if min(binary_dimensions) >= 3:
                search.append("search_bit_three")
            if min(binary_dimensions) >= 5:
                search.append("search_bit_five")
        update = [
            "update_greedy",
            "update_round_robin",
            "update_pairwise",
            "update_always",
            "update_simulated_annealing",
            "update_ssga_worst",
            "update_ssga_stochastic_tournament",
            "update_ssga_deterministic_tournament",
        ]
    elif ptype == "permutation":
        choose = [
            "choose_traverse",
            "choose_random",
            "choose_tournament",
            "choose_stochastic_tournament",
            "choose_roulette_wheel",
            "choose_nich",
        ]
        search = [
            "cross_order_two",
            "cross_order_n",
            "search_swap",
            "search_swap_multi",
            "search_scramble",
            "search_insert",
            "reinit_permutation",
        ]
        dimensions = _decision_dimensions(problem)
        if dimensions and min(dimensions) < 2:
            # Every positional permutation move needs two distinct locations.
            # Reinitialization is the only meaningful, total operator on the
            # singleton permutation domain.
            search = ["reinit_permutation"]
        update = [
            "update_greedy",
            "update_round_robin",
            "update_pairwise",
            "update_always",
            "update_simulated_annealing",
            "update_ssga_worst",
            "update_ssga_stochastic_tournament",
            "update_ssga_deterministic_tournament",
        ]
    else:
        raise NotImplementedError(
            "space() currently supports continuous, discrete, and permutation problems"
        )

    if (
        str(get_flex(set_obj, "GraphSemantics", "legacy_pathway_v1")).lower()
        == "stream_graph_v2"
    ):
        # eoFastGA exposes this selector at all three selection stages. Keep it
        # scoped to Search v6.0 so activating the new grammar cannot silently
        # change a legacy Search experiment's component space.
        if "choose_fastga_stochastic_tournament" not in choose:
            choose.append("choose_fastga_stochastic_tournament")
        if "choose_elite_fraction" not in choose:
            choose.append("choose_elite_fraction")
        if ptype == "discrete" and _binary_dimensions(problem):
            # Dedicated compatibility components preserve legacy crossover
            # behaviour while making the eoFastGA family representable in the
            # v6.0 stream grammar.
            insert_at = search.index("search_reset_one")
            exact_crossovers = ["cross_fastga_uniform", "cross_fastga_one"]
            if min(_binary_dimensions(problem)) >= 2:
                exact_crossovers.extend(["cross_fastga_three", "cross_fastga_five"])
            search[insert_at:insert_at] = exact_crossovers
            exact_mutations = [
                "search_fastga_bit_uniform",
                "search_fastga_bit_standard",
                "search_fastga_bit_conditional",
                "search_fastga_bit_shifted",
                "search_fastga_bit_normal",
                "search_fastga_bit_fast",
                "search_fastga_bit_one",
            ]
            if min(_binary_dimensions(problem)) >= 3:
                exact_mutations.append("search_fastga_bit_three")
            if min(_binary_dimensions(problem)) >= 5:
                exact_mutations.append("search_fastga_bit_five")
            search.extend(exact_mutations)
            if "choose_fastga_proportional" not in choose:
                choose.append("choose_fastga_proportional")
        for exact_update in (
            "update_fastga_plus",
            "update_ssga_fastga_worst",
            "update_ssga_fastga_stochastic_tournament",
            "update_ssga_fastga_deterministic_tournament",
        ):
            if exact_update not in update:
                update.append(exact_update)

    choose.extend(
        name
        for name in compatible_custom_components("choose", ptype)
        if name not in choose
    )
    search.extend(
        name
        for name in compatible_custom_components("search", ptype)
        if name not in search
    )
    update.extend(
        name
        for name in compatible_custom_components("update", ptype)
        if name not in update
    )
    population_values = population_size_space(set_obj)
    offspring_values = offspring_size_space(set_obj)
    if not any(
        offspring <= population
        for population in population_values
        for offspring in offspring_values
    ):
        # SSGA replacement has the hard lambda <= mu contract.  If the declared
        # configuration domains contain no legal pair, the components are not
        # part of this run's executable graph space.
        update = [name for name in update if not name.startswith("update_ssga_")]

    all_op = choose + search + update
    op_space = np.array(
        [
            [1, len(choose)],
            [len(choose) + 1, len(choose) + len(search)],
            [len(choose) + len(search) + 1, len(all_op)],
        ],
        dtype=int,
    )

    para_space = []
    behav_space = []
    para_type_space = []

    for name in all_op:
        arr = None
        try:
            comp = get_component(name)
        except KeyError:
            para_space.append(None)
            behav_space.append(None)
            para_type_space.append(())
            continue

        # Every registered choose/search/update component follows the public
        # mode protocol. Do not turn a TypeError raised *inside* a component
        # into a parameter-free operator; that silently changes its search
        # space and makes implementation bugs look like valid configurations.
        params, _ = comp(problem, "parameter")
        behavior, _ = comp("behavior")

        if params is None:
            arr = None
        else:
            arr = np.asarray(params, dtype=float)
            if arr.ndim == 1:
                if arr.size % 2 != 0:
                    raise ValueError(f"Parameter bounds for {name} must contain pairs")
                arr = arr.reshape(-1, 2)
            elif arr.ndim > 1 and arr.shape[1] != 2:
                total = arr.size
                if total % 2 != 0:
                    raise ValueError(f"Parameter bounds for {name} must contain pairs")
                arr = arr.reshape(-1, 2)

        behav_matrix = _to_behavior_matrix(behavior)
        para_space.append(arr)
        behav_space.append(behav_matrix if behav_matrix is not None else behavior)
        para_type_space.append(
            () if arr is None else _parameter_types(comp, problem, arr.shape[0])
        )

    set_obj.AllOp = all_op
    set_obj.OpSpace = op_space
    set_obj.ParaSpace = para_space
    set_obj.ParaTypeSpace = para_type_space
    set_obj.BehavSpace = behav_space
    # Algorithm-level variables can have conditional activity just like
    # component parameters. Keep the inferred type private and immutable for
    # this prepared design space; in particular, boundary handling is active
    # only for continuous domains.
    set_obj._autoopt_problem_type = ptype

    # Provide defaults for downstream code
    set_obj.alg_p = getattr(set_obj, "alg_p", getattr(set_obj, "AlgP", 1))
    set_obj.alg_q = getattr(set_obj, "alg_q", getattr(set_obj, "AlgQ", 3))
    set_obj.alg_n = getattr(set_obj, "alg_n", getattr(set_obj, "AlgN", 1))
    return set_obj

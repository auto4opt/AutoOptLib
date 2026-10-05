"""Search v6.0 stream-graph genotype support.

The legacy graph grammar stores every pathway in the same edge-list matrix but
interprets it as one shared chooser, one or more search operators, and one
shared update.  Search v6.0 deliberately keeps that edge-list representation
while relaxing its grammar to an optional selector before each legal stage::

    [choose ->] search -> update
    [choose ->] crossover [[choose ->] mutation] -> update

Two pathways may be used at the same time.  Every pathway has its own terminal
update, while selection before either search is optional.  An omitted choose
means that the complete incoming population or stream participates.  FastGA is
one reachable graph in this bounded general evolutionary-algorithm grammar; it
is not a privileged template or atomic mutation.

This module owns only genotype construction, validation, repair, decoding and
mutation.  Execution lives in :mod:`autooptlib.utils.solve` so legacy graph
semantics remain untouched unless ``GraphSemantics=stream_graph_v2`` is set.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from ._helpers import SearchParam, SearchStep, ensure_rng, get_flex
from ._population import is_mutation_component

SEARCH_VERSION = "search-v6.0"
STREAM_GRAPH_SEMANTICS = "stream_graph_v2"
STREAM_IMPLEMENTATION_REVISION = "search-v6.0-uniform-bit-mutation-dev7"


@dataclass
class StreamStage:
    """One search, optionally preceded by stage-local selection."""

    choose: str | None
    search: SearchStep


@dataclass
class StreamStageParam:
    choose: np.ndarray | None
    search: SearchParam


@dataclass
class StreamPathway:
    """A typed stream pathway with independent selection at every stage."""

    stages: list[StreamStage]
    update: str
    archive: list[str]
    execution_semantics: str = STREAM_GRAPH_SEMANTICS

    @property
    def choose(self) -> str | None:
        """Compatibility view used by diagnostics written for legacy paths."""

        return self.stages[0].choose if self.stages else None

    @property
    def search(self) -> list[SearchStep]:
        return [stage.search for stage in self.stages]


@dataclass
class StreamPathwayParam:
    stages: list[StreamStageParam]
    update: np.ndarray | None

    @property
    def choose(self) -> np.ndarray | None:
        return self.stages[0].choose if self.stages else None

    @property
    def search(self) -> list[SearchParam]:
        return [stage.search for stage in self.stages]


def is_stream_graph(setting: Any) -> bool:
    value = str(get_flex(setting, "GraphSemantics", "legacy_pathway_v1")).lower()
    return value == STREAM_GRAPH_SEMANTICS


def _validated_limits(setting: Any) -> tuple[int, int]:
    """Return the bounded v6.0 grammar limits or reject a widened setting."""

    maximum_pathways = int(get_flex(setting, "alg_p", required=True))
    maximum_searches = int(get_flex(setting, "alg_q", required=True))
    if not 1 <= maximum_pathways <= 2 or not 1 <= maximum_searches <= 2:
        raise ValueError("Search v6.0 requires 1<=AlgP<=2 and 1<=AlgQ<=2.")
    return maximum_pathways, maximum_searches


def _operator_category(index: int, op_space: np.ndarray) -> str:
    if int(op_space[0, 0]) <= index <= int(op_space[0, 1]):
        return "choose"
    if int(op_space[1, 0]) <= index <= int(op_space[1, 1]):
        return "search"
    if int(op_space[2, 0]) <= index <= int(op_space[2, 1]):
        return "update"
    raise ValueError(f"Operator index {index} is outside OpSpace.")


def path_nodes(path: Any) -> list[int]:
    """Return the ordered node IDs from one unchanged edge-list matrix."""

    matrix = np.asarray(path, dtype=int)
    if matrix.ndim != 2 or matrix.shape[1] != 2 or matrix.shape[0] == 0:
        raise ValueError("A stream pathway must be a non-empty N x 2 edge matrix.")
    if matrix.shape[0] > 1 and not np.array_equal(matrix[:-1, 1], matrix[1:, 0]):
        raise ValueError("A stream pathway edge matrix must form one continuous path.")
    return [int(matrix[0, 0]), *[int(value) for value in matrix[:, 1]]]


def nodes_to_path(nodes: Sequence[int]) -> np.ndarray:
    values = [int(value) for value in nodes]
    if len(values) < 2:
        raise ValueError("A stream pathway needs at least search and update nodes.")
    return np.asarray(list(zip(values[:-1], values[1:])), dtype=int)


def named_initial_genotype(
    specification: Mapping[str, Any], setting: Any
) -> tuple[list[np.ndarray], list[list[Any]], dict[str, Any]]:
    """Build one v6.0 genotype from a portable component-named specification.

    Warm starts remain ordinary initial candidates: after this conversion they
    use the same repair, evaluation, logging, selection and FE accounting as a
    randomly initialized design.
    """

    if not is_stream_graph(setting):
        raise ValueError(
            "Named Search initial designs require GraphSemantics='stream_graph_v2'."
        )
    if not isinstance(specification, Mapping):
        raise TypeError("Each Search initial design must be a mapping.")
    all_op = list(get_flex(setting, "all_op", required=True))
    if len(all_op) != len(set(all_op)):
        raise ValueError("Search v6.0 component names must be unique.")
    index_by_name = {name: index + 1 for index, name in enumerate(all_op)}
    op_space = np.asarray(get_flex(setting, "op_space", required=True), dtype=int)

    def component_index(name: Any, expected: str) -> int:
        if not isinstance(name, str) or name not in index_by_name:
            raise ValueError(f"Unknown Search v6.0 {expected} component {name!r}.")
        index = index_by_name[name]
        if _operator_category(index, op_space) != expected:
            raise ValueError(f"Component {name!r} is not a {expected} component.")
        return index

    raw_paths = specification.get("pathways")
    if (
        isinstance(raw_paths, (str, bytes))
        or not isinstance(raw_paths, Sequence)
        or not raw_paths
    ):
        raise ValueError("A Search initial design needs one or two pathways.")
    paths: list[np.ndarray] = []
    for path_index, raw_path in enumerate(raw_paths):
        if not isinstance(raw_path, Mapping):
            raise TypeError(f"Initial pathway {path_index} must be a mapping.")
        stages = raw_path.get("stages")
        if (
            isinstance(stages, (str, bytes))
            or not isinstance(stages, Sequence)
            or not stages
        ):
            raise ValueError(f"Initial pathway {path_index} needs one or two stages.")
        nodes: list[int] = []
        for stage_index, stage in enumerate(stages):
            if not isinstance(stage, Mapping):
                raise TypeError(
                    f"Initial pathway {path_index} stage {stage_index} must be a mapping."
                )
            choose_name = stage.get("choose")
            if choose_name is not None:
                nodes.append(component_index(choose_name, "choose"))
            nodes.append(component_index(stage.get("search"), "search"))
        update_name = raw_path.get("update", specification.get("update"))
        nodes.append(component_index(update_name, "update"))
        paths.append(nodes_to_path(nodes))
    validate_paths(paths, setting)

    parameters: list[list[Any]] = [[None, None] for _ in all_op]
    raw_parameters = specification.get("parameters", {})
    if not isinstance(raw_parameters, Mapping):
        raise TypeError("Search initial-design parameters must be a mapping.")
    active = set(active_operator_indices(paths))
    para_space = list(get_flex(setting, "para_space", required=True))
    for component, raw_values in raw_parameters.items():
        if not isinstance(component, str) or component not in index_by_name:
            raise ValueError(f"Unknown parameter component {component!r}.")
        index = index_by_name[component]
        if index not in active:
            raise ValueError(
                f"Initial-design parameter component {component!r} is inactive."
            )
        values = np.asarray(raw_values, dtype=float).reshape(-1)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError(
                f"Initial-design parameters for {component!r} must be finite and non-empty."
            )
        bounds = para_space[index - 1]
        if bounds is None:
            raise ValueError(f"Component {component!r} has no parameters.")
        bounds_array = np.asarray(bounds, dtype=float).reshape(-1, 2)
        if values.size != bounds_array.shape[0] or np.any(
            (values < bounds_array[:, 0]) | (values > bounds_array[:, 1])
        ):
            raise ValueError(
                f"Initial-design parameters for {component!r} are outside its space."
            )
        parameters[index - 1][0] = values.copy()

    configuration = specification.get("configuration")
    if not isinstance(configuration, Mapping):
        raise TypeError("A Search initial design needs an algorithm configuration.")
    from ._population import (
        normalize_configuration,
        offspring_size_space,
        population_size_space,
    )

    normalized_configuration = normalize_configuration(configuration, setting)
    if normalized_configuration["population_size"] not in population_size_space(
        setting
    ):
        raise ValueError("Initial-design population_size is outside its space.")
    if normalized_configuration["offspring_size"] not in offspring_size_space(setting):
        raise ValueError("Initial-design offspring_size is outside its space.")
    return paths, parameters, normalized_configuration


def validate_paths(paths: Sequence[Any], setting: Any) -> list[list[int]]:
    """Feasibility-check the bounded conditional grammar without modifying it."""

    op_space = np.asarray(get_flex(setting, "op_space", required=True), dtype=int)
    all_op = list(get_flex(setting, "all_op", required=True))
    maximum_pathways, maximum_searches = _validated_limits(setting)
    if not 1 <= len(paths) <= maximum_pathways:
        raise ValueError("Stream graph pathway count is outside AlgP.")
    decoded: list[list[int]] = []
    for path_index, path in enumerate(paths):
        nodes = path_nodes(path)
        categories = [_operator_category(index, op_space) for index in nodes]
        if categories[-1] != "update" or "update" in categories[:-1]:
            raise ValueError(
                f"Stream pathway {path_index} must end at its only update node."
            )
        body = categories[:-1]
        search_count = 0
        position = 0
        while position < len(body):
            if body[position] == "choose":
                position += 1
                if position >= len(body) or body[position] != "search":
                    raise ValueError(
                        f"Stream pathway {path_index} contains a choose that is "
                        "not immediately followed by a search."
                    )
            elif body[position] != "search":
                raise ValueError(
                    f"Stream pathway {path_index} contains an invalid node order."
                )
            search_count += 1
            position += 1
        if not 1 <= search_count <= maximum_searches:
            raise ValueError(
                f"Stream pathway {path_index} requires one to AlgQ="
                f"{maximum_searches} search nodes."
            )
        search_indices = [
            node
            for node in nodes[:-1]
            if _operator_category(node, op_space) == "search"
        ]
        if len(search_indices) == 2:
            first_name = str(all_op[search_indices[0] - 1])
            second_name = str(all_op[search_indices[1] - 1])
            if not first_name.startswith("cross_"):
                raise ValueError(
                    f"Stream pathway {path_index} may contain a second search "
                    "only when its first search is a crossover."
                )
            if not is_mutation_component(second_name):
                raise ValueError(
                    f"Stream pathway {path_index} requires its second search "
                    "to be a mutation component."
                )
        decoded.append(nodes)
    return decoded


def validate_phenotype(
    pathways: Sequence[StreamPathway], parameters: Sequence[StreamPathwayParam]
) -> None:
    """Validate decoded stream objects at serialization and execution boundaries."""

    if not 1 <= len(pathways) <= 2 or len(pathways) != len(parameters):
        raise ValueError("Search v6.0 requires one or two matching pathways.")
    if not all(isinstance(pathway, StreamPathway) for pathway in pathways) or not all(
        isinstance(parameter, StreamPathwayParam) for parameter in parameters
    ):
        raise TypeError("Search v6.0 requires stream pathway objects.")
    shared_archive = list(pathways[0].archive)
    for index, (pathway, path_parameters) in enumerate(zip(pathways, parameters)):
        if pathway.execution_semantics != STREAM_GRAPH_SEMANTICS:
            raise ValueError(
                f"Search v6.0 pathway {index} has incompatible execution semantics."
            )
        if not 1 <= len(pathway.stages) <= 2:
            raise ValueError(f"Search v6.0 pathway {index} requires one or two stages.")
        if len(pathway.stages) != len(path_parameters.stages):
            raise ValueError(
                f"Search v6.0 pathway {index} stage parameters do not match."
            )
        if not all(
            isinstance(stage, StreamStage) for stage in pathway.stages
        ) or not all(
            isinstance(stage_parameter, StreamStageParam)
            for stage_parameter in path_parameters.stages
        ):
            raise TypeError("Search v6.0 requires stream stage objects.")
        if list(pathway.archive) != shared_archive:
            raise ValueError("Search v6.0 pathways must share one archive policy.")
        for stage in pathway.stages:
            if stage.search.secondary is not None:
                raise ValueError(
                    "Search v6.0 stages do not support secondary search components."
                )
        if len(pathway.stages) == 2:
            first_name = str(pathway.stages[0].search.primary)
            second_name = str(pathway.stages[1].search.primary)
            if not first_name.startswith("cross_"):
                raise ValueError(
                    f"Search v6.0 pathway {index} may contain a second stage "
                    "only after a crossover."
                )
            if not is_mutation_component(second_name):
                raise ValueError(
                    f"Search v6.0 pathway {index} requires a mutation in its "
                    "second stage."
                )


def _random_index(rng: np.random.Generator, bounds: Sequence[int]) -> int:
    return int(rng.integers(int(bounds[0]), int(bounds[1]) + 1))


def _random_search_index(
    rng: np.random.Generator,
    setting: Any,
) -> int:
    op_space = np.asarray(get_flex(setting, "op_space", required=True), dtype=int)
    pool = list(range(int(op_space[1, 0]), int(op_space[1, 1]) + 1))
    if not pool:
        raise ValueError("Search v6.0 requires at least one search component.")
    return int(rng.choice(pool))


def _random_mutation_index(rng: np.random.Generator, setting: Any) -> int:
    op_space = np.asarray(get_flex(setting, "op_space", required=True), dtype=int)
    all_op = list(get_flex(setting, "all_op", required=True))
    pool = [
        index
        for index in range(int(op_space[1, 0]), int(op_space[1, 1]) + 1)
        if is_mutation_component(all_op[index - 1])
    ]
    if not pool:
        raise ValueError("Search v6.0 requires at least one mutation component.")
    return int(rng.choice(pool))


def _random_path_nodes(
    rng: np.random.Generator, setting: Any, maximum_searches: int
) -> list[int]:
    op_space = np.asarray(get_flex(setting, "op_space", required=True), dtype=int)
    all_op = list(get_flex(setting, "all_op", required=True))
    nodes: list[int] = []
    if bool(rng.integers(0, 2)):
        nodes.append(_random_index(rng, op_space[0]))
    first_search = _random_search_index(rng, setting)
    nodes.append(first_search)
    if (
        maximum_searches >= 2
        and str(all_op[first_search - 1]).startswith("cross_")
        and bool(rng.integers(0, 2))
    ):
        if bool(rng.integers(0, 2)):
            nodes.append(_random_index(rng, op_space[0]))
        nodes.append(_random_mutation_index(rng, setting))
    nodes.append(_random_index(rng, op_space[2]))
    return nodes


def _initial_parameters(
    rng: np.random.Generator,
    para_space: Sequence[Any],
    para_type_space: Sequence[Any],
) -> list[list[Any]]:
    result: list[list[Any]] = [[None, None] for _ in para_space]
    for index, bounds_value in enumerate(para_space):
        if bounds_value is None or len(bounds_value) == 0:
            continue
        bounds = np.asarray(bounds_value, dtype=float).reshape(-1, 2)
        values = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * rng.random(
            bounds.shape[0]
        )
        kinds = (
            para_type_space[index]
            if index < len(para_type_space)
            else ("continuous",) * len(values)
        )
        for value_index, kind in enumerate(kinds):
            if kind == "integer":
                lower = int(np.ceil(bounds[value_index, 0]))
                upper = int(np.floor(bounds[value_index, 1]))
                values[value_index] = rng.integers(lower, upper + 1)
        result[index][0] = values
    return result


def initialize(setting: Any, n: int):
    """Initialize v6.0 genotypes with independent choose/search stages."""

    count = int(n)
    if count <= 0:
        return [], []
    rng = ensure_rng(setting)
    alg_p, alg_q = _validated_limits(setting)
    para_space = list(get_flex(setting, "para_space", required=True))
    para_types = list(get_flex(setting, "para_type_space", [()] * len(para_space)))
    operators: list[list[np.ndarray]] = []
    parameters: list[list[list[Any]]] = []
    for _ in range(count):
        paths: list[np.ndarray] = []
        path_count = int(rng.integers(1, alg_p + 1))
        for _path_index in range(path_count):
            paths.append(nodes_to_path(_random_path_nodes(rng, setting, alg_q)))
        operators.append(paths)
        parameters.append(_initial_parameters(rng, para_space, para_types))
    return operators, parameters


def repair(operators: Any, parameters: Any, problem: Any, setting: Any):
    """Strictly validate and copy Search v6.0 genotypes without changing them."""

    del problem  # Component-domain filtering already happens in space().
    repaired_operators: list[list[np.ndarray]] = []
    repaired_parameters: list[list[list[Any]]] = []
    for algorithm_index, algorithm_paths in enumerate(operators):
        paths = [np.asarray(path, dtype=int).copy() for path in algorithm_paths]
        validate_paths(paths, setting)
        repaired_operators.append(paths)
        if algorithm_index >= len(parameters):
            raise ValueError("Every Search v6.0 graph requires a parameter bank.")
        entries = parameters[algorithm_index]
        normalized_entries: list[list[Any]] = []
        all_op = list(get_flex(setting, "all_op", required=True))
        if len(entries) != len(all_op):
            raise ValueError(
                "Search v6.0 parameter-bank length must match the component space."
            )
        for index in range(len(all_op)):
            entry = entries[index] if index < len(entries) else None
            if entry is None:
                normalized_entries.append([None, None])
            else:
                values = entry[0] if len(entry) else None
                behavior = entry[1] if len(entry) > 1 else None
                normalized_entries.append(
                    [None if values is None else np.asarray(values).copy(), behavior]
                )
        repaired_parameters.append(normalized_entries)
    return repaired_operators, repaired_parameters


def _parameter_for(entries: Sequence[Any], index: int) -> np.ndarray | None:
    if not 0 < index <= len(entries):
        return None
    entry = entries[index - 1]
    if not entry or entry[0] is None:
        return None
    return np.asarray(entry[0]).copy()


def decode(operators: Any, parameters: Any, problem: Any, setting: Any):
    """Decode unchanged edge matrices into the v6.0 stream execution plan."""

    del problem
    all_op = list(get_flex(setting, "all_op", required=True))
    op_space = np.asarray(get_flex(setting, "op_space", required=True), dtype=int)
    rate = float(get_flex(setting, "inc_rate", 0.0))
    inner_fe = float(get_flex(setting, "inner_fe", 1.0))
    prob_n = max(1.0, float(get_flex(setting, "prob_n", 1.0)))
    termination = np.asarray([rate, max(1, int(np.ceil(inner_fe / prob_n)))])
    archive = list(get_flex(setting, "archive", []))
    decoded_operators: list[list[StreamPathway]] = []
    decoded_parameters: list[list[StreamPathwayParam]] = []
    for algorithm_index, paths in enumerate(operators):
        node_paths = validate_paths(paths, setting)
        entries = parameters[algorithm_index]
        path_objects: list[StreamPathway] = []
        path_params: list[StreamPathwayParam] = []
        for nodes in node_paths:
            stages: list[StreamStage] = []
            stage_params: list[StreamStageParam] = []
            position = 0
            while position < len(nodes) - 1:
                choose_index: int | None = None
                if _operator_category(nodes[position], op_space) == "choose":
                    choose_index = nodes[position]
                    position += 1
                search_index = nodes[position]
                position += 1
                stages.append(
                    StreamStage(
                        choose=(
                            None if choose_index is None else all_op[choose_index - 1]
                        ),
                        search=SearchStep(
                            primary=all_op[search_index - 1],
                            secondary=None,
                            termination=termination.copy(),
                        ),
                    )
                )
                stage_params.append(
                    StreamStageParam(
                        choose=(
                            None
                            if choose_index is None
                            else _parameter_for(entries, choose_index)
                        ),
                        search=SearchParam(
                            primary=_parameter_for(entries, search_index),
                            secondary=None,
                        ),
                    )
                )
            update_index = nodes[-1]
            path_objects.append(
                StreamPathway(stages, all_op[update_index - 1], list(archive))
            )
            path_params.append(
                StreamPathwayParam(
                    stages=stage_params,
                    update=_parameter_for(entries, update_index),
                )
            )
        decoded_operators.append(path_objects)
        decoded_parameters.append(path_params)
    return decoded_operators, decoded_parameters


def active_operator_indices(paths: Iterable[Any]) -> list[int]:
    values: list[int] = []
    for path in paths:
        values.extend(path_nodes(path))
    return values


def mutate_structure(
    algorithm: Any, setting: Any, aux: Any = None
) -> tuple[list[np.ndarray], list[list[Any]], dict[str, Any]]:
    """Apply feasibility-preserving node and grammar mutations."""

    rng = ensure_rng(setting)
    op_space = np.asarray(get_flex(setting, "op_space", required=True), dtype=int)
    alg_p, alg_q = _validated_limits(setting)
    rate = float(get_flex(setting, "structure_mutation_rate", 0.30))
    paths = [np.asarray(path, dtype=int).copy() for path in algorithm.operator]
    entries = [
        [None, None]
        if entry is None
        else [
            None if not entry or entry[0] is None else np.asarray(entry[0]).copy(),
            entry[1] if len(entry) > 1 else None,
        ]
        for entry in algorithm.parameter
    ]
    old_active = set(active_operator_indices(paths))
    node_paths = validate_paths(paths, setting)
    all_op = list(get_flex(setting, "all_op", required=True))
    locations: list[tuple[int, int, str]] = []
    for path_index, nodes in enumerate(node_paths):
        for node_index, value in enumerate(nodes):
            locations.append(
                (path_index, node_index, _operator_category(value, op_space))
            )
    requested = max(1, int(np.floor(rate * len(locations) + 0.5)))
    selected = np.asarray(
        rng.choice(len(locations), size=min(requested, len(locations)), replace=False),
        dtype=int,
    ).reshape(-1)
    edits = 0
    for selected_index in selected:
        path_index, node_index, category = locations[int(selected_index)]
        nodes = node_paths[path_index]
        row = {"choose": 0, "search": 1, "update": 2}[category]
        pool = np.arange(op_space[row, 0], op_space[row, 1] + 1, dtype=int)
        if category == "search":
            search_positions = [
                position
                for position, value in enumerate(nodes[:-1])
                if _operator_category(value, op_space) == "search"
            ]
            if len(search_positions) == 2:
                if node_index == search_positions[0]:
                    pool = np.asarray(
                        [
                            value
                            for value in pool
                            if str(all_op[int(value) - 1]).startswith("cross_")
                        ],
                        dtype=int,
                    )
                else:
                    pool = np.asarray(
                        [
                            value
                            for value in pool
                            if is_mutation_component(all_op[int(value) - 1])
                        ],
                        dtype=int,
                    )
        current = nodes[node_index]
        alternatives = pool[pool != current]
        if alternatives.size:
            replacement = int(rng.choice(alternatives))
            nodes[node_index] = replacement
            edits += 1

    # A topology edit is a production-rule rewrite of the conditional grammar:
    #     [choose] search update
    #     [choose] crossover [[choose] mutation] update
    # Every individual edit therefore remains feasible without repair.
    if rng.random() < rate:
        topology_moves: list[str] = []
        choose_locations: list[tuple[int, int]] = []
        unchosen_searches: list[tuple[int, int]] = []
        expandable_paths: list[int] = []
        reducible_paths: list[int] = []
        for path_index, nodes in enumerate(node_paths):
            search_count = 0
            position = 0
            while position < len(nodes) - 1:
                if _operator_category(nodes[position], op_space) == "choose":
                    choose_locations.append((path_index, position))
                    position += 1
                else:
                    unchosen_searches.append((path_index, position))
                search_count += 1
                position += 1
            first_search = next(
                value
                for value in nodes[:-1]
                if _operator_category(value, op_space) == "search"
            )
            if search_count < alg_q and str(all_op[first_search - 1]).startswith(
                "cross_"
            ):
                expandable_paths.append(path_index)
            if search_count > 1:
                reducible_paths.append(path_index)
        if choose_locations:
            topology_moves.append("delete_choose")
        if unchosen_searches:
            topology_moves.append("add_choose")
        if expandable_paths:
            topology_moves.append("add_search")
        if reducible_paths:
            topology_moves.append("delete_search")
        if len(node_paths) < alg_p or len(node_paths) > 1:
            topology_moves.append("pathway")
        topology_move = str(rng.choice(topology_moves))
        if topology_move == "delete_choose":
            path_index, position = choose_locations[
                int(rng.integers(0, len(choose_locations)))
            ]
            del node_paths[path_index][position]
            edits += 1
        elif topology_move == "add_choose":
            path_index, position = unchosen_searches[
                int(rng.integers(0, len(unchosen_searches)))
            ]
            node_paths[path_index].insert(position, _random_index(rng, op_space[0]))
            edits += 1
        elif topology_move == "add_search":
            path_index = int(rng.choice(expandable_paths))
            nodes = node_paths[path_index]
            nodes.insert(len(nodes) - 1, _random_mutation_index(rng, setting))
            edits += 1
        elif topology_move == "delete_search":
            path_index = int(rng.choice(reducible_paths))
            nodes = node_paths[path_index]
            search_positions = [
                position
                for position, value in enumerate(nodes[:-1])
                if _operator_category(value, op_space) == "search"
            ]
            position = int(rng.choice(search_positions))
            del nodes[position]
            if (
                position > 0
                and _operator_category(nodes[position - 1], op_space) == "choose"
            ):
                del nodes[position - 1]
            edits += 1
        elif topology_move == "pathway":
            if len(node_paths) < alg_p:
                node_paths.append(_random_path_nodes(rng, setting, alg_q))
            elif len(node_paths) > 1:
                del node_paths[int(rng.integers(0, len(node_paths)))]
            edits += 1
    paths = [nodes_to_path(nodes) for nodes in node_paths]
    # This is a feasibility check, not a repair.  Any failure identifies a bug
    # in one of the grammar-preserving mutations above and must stop evaluation.
    validate_paths(paths, setting)
    new_active = set(active_operator_indices(paths))
    parameter_bank = get_flex(setting, "_component_parameter_bank", None)
    if parameter_bank is not None:
        for operator_index in new_active - old_active:
            if 0 < operator_index <= len(entries) and operator_index <= len(
                parameter_bank
            ):
                entries[operator_index - 1] = deepcopy(
                    parameter_bank[operator_index - 1]
                )
    state = dict(aux) if isinstance(aux, dict) else {}
    state = {
        key: value
        for key, value in state.items()
        if not str(key).startswith(("cma_", "ils_"))
    }
    state.update(
        last_move_kind="structure",
        last_structure_edits=edits,
        structure_mutation_rate=rate,
        graph_semantics=STREAM_GRAPH_SEMANTICS,
    )
    return paths, entries, state


__all__ = [
    "SEARCH_VERSION",
    "STREAM_GRAPH_SEMANTICS",
    "STREAM_IMPLEMENTATION_REVISION",
    "StreamPathway",
    "StreamPathwayParam",
    "StreamStage",
    "StreamStageParam",
    "active_operator_indices",
    "decode",
    "initialize",
    "is_stream_graph",
    "mutate_structure",
    "named_initial_genotype",
    "nodes_to_path",
    "path_nodes",
    "repair",
    "validate_paths",
    "validate_phenotype",
]

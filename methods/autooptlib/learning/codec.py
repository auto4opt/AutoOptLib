"""Bidirectional codec between AutoOptLib graphs and learning sequences."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..utils.design import Design
from ..utils.design._helpers import (
    Pathway,
    PathwayParam,
    SearchParam,
    SearchStep,
    get_problem_type,
    problem_list,
)
from ..utils.design._population import (
    configuration_activity,
    configuration_is_valid,
    copy_configuration,
    offspring_size_space,
    population_size_space,
    set_configuration,
)
from ..utils.design._stream_graph import (
    STREAM_GRAPH_SEMANTICS,
    StreamPathway,
    StreamPathwayParam,
    is_stream_graph,
    nodes_to_path,
    validate_paths,
    validate_phenotype,
)
from ..utils.space import space
from .grammar import LearningGrammar
from .vocabulary import LearningVocabulary


@dataclass(frozen=True)
class LearningCodec:
    """A problem-aware, grammar-constrained graph/sequence conversion."""

    vocabulary: LearningVocabulary
    grammar: LearningGrammar
    problems: tuple[Any, ...]
    setting: Any

    @classmethod
    def from_problem(
        cls,
        problem: Any,
        setting: Any,
        *,
        vocabulary: LearningVocabulary | None = None,
    ) -> "LearningCodec":
        problems = tuple(problem_list(problem))
        if not problems:
            raise ValueError("At least one constructed problem is required.")
        if not hasattr(setting, "AllOp") and not hasattr(setting, "all_op"):
            setting = space(problems, setting)
        problem_type = get_problem_type(problems)
        if problem_type is None:
            raise ValueError("The constructed problem does not declare a type.")
        local_vocabulary = LearningVocabulary.from_space(
            setting,
            problem_type,
        )
        if vocabulary is None:
            vocabulary = local_vocabulary
        elif vocabulary != local_vocabulary:
            merged = LearningVocabulary.merge((vocabulary, local_vocabulary))
            if merged != vocabulary:
                raise ValueError(
                    "Provided vocabulary does not cover the target design space."
                )
        population_values = population_size_space(setting)
        offspring_values = offspring_size_space(setting)
        grammar = LearningGrammar(
            vocabulary,
            max_pathways=int(getattr(setting, "AlgP", getattr(setting, "alg_p", 1))),
            max_search_components=int(
                getattr(setting, "AlgQ", getattr(setting, "alg_q", 4))
            ),
            population_size_bins=len(population_values),
            offspring_size_bins=len(offspring_values),
            encode_boundary_handling=True,
            population_size_values=tuple(population_values),
            offspring_size_values=tuple(offspring_values),
            graph_semantics=(
                STREAM_GRAPH_SEMANTICS
                if is_stream_graph(setting)
                else "legacy_pathway_v1"
            ),
        )
        return cls(vocabulary, grammar, problems, setting)

    def _bounds(self, component: str) -> np.ndarray | None:
        all_op = list(
            getattr(self.setting, "AllOp", getattr(self.setting, "all_op", ()))
        )
        spaces = list(
            getattr(self.setting, "ParaSpace", getattr(self.setting, "para_space", ()))
        )
        try:
            value = spaces[all_op.index(component)]
        except (ValueError, IndexError) as exc:
            raise KeyError(
                f"Component {component!r} is outside the design space."
            ) from exc
        if value is None:
            return None
        array = np.asarray(value, dtype=float)
        return array.reshape(-1, 2)

    def _local_termination(self) -> np.ndarray:
        improvement_rate = float(
            getattr(self.setting, "IncRate", getattr(self.setting, "inc_rate", 0.05))
        )
        inner_evaluations = float(
            getattr(self.setting, "InnerFE", getattr(self.setting, "inner_fe", 1))
        )
        population_size = float(
            getattr(self.setting, "ProbN", getattr(self.setting, "prob_n", 1))
        )
        return np.asarray(
            [improvement_rate, float(math.ceil(inner_evaluations / population_size))],
            dtype=float,
        )

    def _configuration_spaces(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        population = population_size_space(self.setting)
        offspring = offspring_size_space(self.setting)
        maximum = 10**self.grammar.configuration_digits
        if len(population) > maximum or len(offspring) > maximum:
            raise ValueError(
                "Learning-based design requires PopulationSizeSpace and "
                f"OffspringSizeSpace to contain at most {maximum} values."
            )
        return population, offspring

    def _encode_size_configuration(self, design: Design) -> list[int]:
        population_space, offspring_space = self._configuration_spaces()
        configuration = copy_configuration(design, self.setting)
        result: list[int] = []
        for name, choices in (
            ("population_size", population_space),
            ("offspring_size", offspring_space),
        ):
            selected = min(
                range(len(choices)),
                key=lambda index: abs(choices[index] - configuration[name]),
            )
            encoded = f"{selected:0{self.grammar.configuration_digits}d}"
            result.extend(
                self.vocabulary.parameter_indices[int(digit)] for digit in encoded
            )
        return result

    def _encode_rates(self, design: Design, pathways: Sequence[Pathway]) -> list[int]:
        configuration = copy_configuration(design, self.setting)
        crossover_used, mutation_used = configuration_activity(design)
        result = []
        for name, active in (
            ("crossover_rate", crossover_used),
            ("mutation_rate", mutation_used),
        ):
            bin_index = int(
                np.clip(
                    np.rint(9 * float(configuration[name] if active else 1.0)),
                    0,
                    9,
                )
            )
            result.append(self.vocabulary.parameter_indices[bin_index])
        return result

    def _decode_configuration(
        self, values: list[int], position: int
    ) -> tuple[dict[str, int | float], int]:
        population_space, offspring_space = self._configuration_spaces()
        width = self.grammar.configuration_digits

        def decode_index(start: int) -> int:
            digits = [
                self.vocabulary.parameter_indices.index(values[start + offset])
                for offset in range(width)
            ]
            return int("".join(str(digit) for digit in digits))

        population_bin = decode_index(position)
        position += width
        offspring_bin = decode_index(position)
        position += width
        return {
            "population_size": population_space[population_bin],
            "offspring_size": offspring_space[offspring_bin],
            "crossover_rate": 1.0,
            "mutation_rate": 1.0,
        }, position

    def _decode_rates(
        self, values: list[int], position: int, configuration: dict[str, int | float]
    ) -> int:
        crossover_bin = self.vocabulary.parameter_indices.index(values[position])
        mutation_bin = self.vocabulary.parameter_indices.index(values[position + 1])
        configuration["crossover_rate"] = crossover_bin / 9.0
        configuration["mutation_rate"] = mutation_bin / 9.0
        return position + 2

    @staticmethod
    def _same_parameter(left: Any, right: Any) -> bool:
        if left is None or right is None:
            return left is None and right is None
        return bool(
            np.array_equal(
                np.asarray(left, dtype=float), np.asarray(right, dtype=float)
            )
        )

    @staticmethod
    def _is_global_termination(value: Any) -> bool:
        array = np.asarray(value, dtype=float).reshape(-1)
        return bool(array.shape == (2,) and np.isneginf(array[0]) and array[1] == 1.0)

    def _encode_component(
        self,
        component: str,
        parameters: np.ndarray | None,
        required_behavior: str | None = None,
    ) -> list[int]:
        result = [self.vocabulary.index(component)]
        bounds = self._bounds(component)
        expected = self.vocabulary.parameter_count(component)
        mode = self.vocabulary.behavior_mode(component)
        if required_behavior == "local" and not self.vocabulary.has_local_choice(
            component
        ):
            raise ValueError(f"Component {component!r} cannot perform local search.")
        if required_behavior == "global" and not self.vocabulary.has_global_choice(
            component
        ):
            raise ValueError(f"Component {component!r} cannot perform global search.")
        if expected == 0:
            return result
        if parameters is None:
            raise ValueError(f"Component {component!r} requires {expected} parameters.")
        values = np.asarray(parameters, dtype=float).reshape(-1)
        if values.size != expected or bounds is None or len(bounds) != expected:
            raise ValueError(
                f"Component {component!r} requires exactly {expected} parameters."
            )
        selected_bins = []
        for parameter_index, (value, (lower, upper)) in enumerate(zip(values, bounds)):
            if not np.isfinite(value):
                raise ValueError(f"Component {component!r} has a non-finite parameter.")
            if upper == lower:
                bin_index = 0
            else:
                normalized = (float(value) - lower) / (upper - lower)
                bin_index = int(np.clip(np.rint(9 * normalized), 0, 9))
            if required_behavior == "local" and mode == "parameter":
                mask = self.vocabulary.local_masks(component)[parameter_index]
                choices = [
                    candidate for candidate in range(10) if mask & (1 << candidate)
                ]
                bin_index = min(
                    choices, key=lambda candidate: abs(candidate - bin_index)
                )
            selected_bins.append(bin_index)
        if required_behavior == "global" and mode == "parameter":
            if self.vocabulary.behavior(component, selected_bins) != "global":
                best: tuple[float, int, int] | None = None
                for parameter_index, current in enumerate(selected_bins):
                    local_mask = self.vocabulary.local_masks(component)[parameter_index]
                    for candidate in range(10):
                        if local_mask & (1 << candidate):
                            continue
                        distance = abs(candidate - current)
                        option = (float(distance), parameter_index, candidate)
                        if best is None or option < best:
                            best = option
                if best is None:
                    raise ValueError(
                        f"Component {component!r} has no global parameter bin."
                    )
                selected_bins[best[1]] = best[2]
        for bin_index in selected_bins:
            result.append(self.vocabulary.parameter_indices[bin_index])
        return result

    def _decode_component(
        self, values: list[int], position: int
    ) -> tuple[str, np.ndarray | None, int]:
        component = self.vocabulary.name(values[position])
        position += 1
        count = self.vocabulary.parameter_count(component)
        if count == 0:
            return component, None, position
        bounds = self._bounds(component)
        if bounds is None or len(bounds) != count:
            raise ValueError(f"Missing parameter bounds for component {component!r}.")
        decoded = []
        for lower, upper in bounds:
            token = values[position]
            position += 1
            bin_index = self.vocabulary.parameter_indices.index(token)
            decoded.append(lower + (upper - lower) * (bin_index / 9.0))
        return component, np.asarray(decoded, dtype=float), position

    def _encode_stream_path(
        self,
        path: StreamPathway,
        params: StreamPathwayParam,
    ) -> list[int]:
        if len(path.stages) != len(params.stages):
            raise ValueError("Stream stage operators and parameters do not match.")
        values = [self.vocabulary.index("path_begin")]
        local_termination = self._local_termination()
        for stage, stage_params in zip(path.stages, params.stages):
            if stage.choose is not None:
                values.extend(self._encode_component(stage.choose, stage_params.choose))
            elif stage_params.choose is not None:
                raise ValueError("A missing stream choose cannot have parameters.")
            if (
                stage.search.secondary is not None
                or stage_params.search.secondary is not None
            ):
                raise ValueError("stream_graph_v2 does not support secondary searches.")
            termination = np.asarray(stage.search.termination, dtype=float).reshape(-1)
            if termination.shape != local_termination.shape or not np.allclose(
                termination, local_termination
            ):
                raise ValueError(
                    "stream_graph_v2 stages require the canonical termination."
                )
            values.extend(
                self._encode_component(
                    stage.search.primary,
                    stage_params.search.primary,
                )
            )
            values.append(self.vocabulary.index("terminate_local"))
        values.extend(self._encode_component(path.update, params.update))
        values.append(self.vocabulary.index("path_end"))
        return values

    def _encode_stream(self, design: Design) -> list[int]:
        operators = getattr(design, "operator_pheno", None)
        parameters = getattr(design, "parameter_pheno", None)
        if not operators or not parameters or not operators[0] or not parameters[0]:
            raise ValueError("The algorithm must contain a decoded graph phenotype.")
        pathways = list(operators[0])
        pathway_parameters = list(parameters[0])
        if not 1 <= len(pathways) <= self.grammar.max_pathways:
            raise ValueError("Stream pathway count is outside AlgP.")
        if len(pathways) != len(pathway_parameters):
            raise ValueError("Pathway operator and parameter counts do not match.")
        if not all(isinstance(path, StreamPathway) for path in pathways) or not all(
            isinstance(item, StreamPathwayParam) for item in pathway_parameters
        ):
            raise TypeError("stream_graph_v2 learning requires typed stream pathways.")
        validate_phenotype(pathways, pathway_parameters)
        expected_archive = list(self.vocabulary.archives)
        if any(list(path.archive) != expected_archive for path in pathways):
            raise ValueError(
                "Graph archives must match the codec's fixed Archive setting."
            )

        problem_type = get_problem_type(self.problems)
        values = [
            self.vocabulary.begin_index,
            self.vocabulary.index(f"problem_{problem_type}"),
        ]
        values.extend(self._encode_size_configuration(design))
        configuration = copy_configuration(design, self.setting)
        boundary = str(configuration["boundary_handling"])
        if self.grammar.encode_boundary_handling:
            values.append(
                self.vocabulary.parameter_indices[
                    ("clip", "reflect", "resample").index(boundary)
                ]
            )
        elif boundary != "clip":
            raise ValueError("This learning grammar only represents clip boundaries.")

        blocks = [
            self._encode_stream_path(path, params)
            for path, params in zip(pathways, pathway_parameters)
        ]
        shared_sink = (
            len(
                {
                    (
                        path.update,
                        None
                        if params.update is None
                        else tuple(np.asarray(params.update, dtype=float).reshape(-1)),
                    )
                    for path, params in zip(pathways, pathway_parameters)
                }
            )
            == 1
        )
        if shared_sink:
            blocks.sort(key=tuple)
        for block in blocks:
            values.extend(block)
        values.extend(self._encode_rates(design, pathways))
        values.append(self.vocabulary.index("archive_begin"))
        values.extend(self.vocabulary.index(name) for name in expected_archive)
        values.append(self.vocabulary.index("archive_end"))
        values.append(self.vocabulary.end_index)
        return self.grammar.validate(values)

    @staticmethod
    def _store_stream_parameter(
        bank: list[list[Any]],
        all_op: Sequence[str],
        component: str,
        parameter: np.ndarray | None,
    ) -> None:
        index = list(all_op).index(component)
        existing = bank[index][0]
        if (
            existing is not None
            and parameter is not None
            and not np.array_equal(np.asarray(existing), np.asarray(parameter))
        ):
            raise ValueError(
                f"Repeated component {component!r} has inconsistent parameters."
            )
        if existing is None and parameter is not None:
            bank[index] = [np.asarray(parameter, dtype=float).copy(), None]

    def _decode_stream(self, sequence: Sequence[int] | np.ndarray) -> Design:
        values = self.grammar.validate(sequence)
        problem_type = get_problem_type(self.problems)
        if self.vocabulary.name(values[1]) != f"problem_{problem_type}":
            raise ValueError("Sequence problem type does not match the codec target.")
        position = 2
        configuration, position = self._decode_configuration(values, position)
        if self.grammar.encode_boundary_handling:
            boundary_index = self.vocabulary.parameter_indices.index(values[position])
            configuration["boundary_handling"] = ("clip", "reflect", "resample")[
                boundary_index
            ]
            position += 1

        all_op = list(
            getattr(self.setting, "AllOp", getattr(self.setting, "all_op", ()))
        )
        parameter_bank: list[list[Any]] = [[None, None] for _ in all_op]
        matrices: list[np.ndarray] = []
        while self.vocabulary.name(values[position]) == "path_begin":
            position += 1
            nodes: list[int] = []
            search_count = 0
            while self.vocabulary.name(values[position]) not in self.vocabulary.update:
                if self.vocabulary.name(values[position]) in self.vocabulary.choose:
                    choose, choose_parameter, position = self._decode_component(
                        values, position
                    )
                    nodes.append(all_op.index(choose) + 1)
                    self._store_stream_parameter(
                        parameter_bank, all_op, choose, choose_parameter
                    )
                search, search_parameter, position = self._decode_component(
                    values, position
                )
                if search not in self.vocabulary.search:
                    raise ValueError("Validated stream sequence lost search alignment.")
                nodes.append(all_op.index(search) + 1)
                self._store_stream_parameter(
                    parameter_bank, all_op, search, search_parameter
                )
                if self.vocabulary.name(values[position]) != "terminate_local":
                    raise ValueError(
                        "Validated stream sequence lost stage termination."
                    )
                position += 1
                search_count += 1
            if search_count == 0:
                raise ValueError("Every stream pathway requires a search stage.")
            update, update_parameter, position = self._decode_component(
                values, position
            )
            nodes.append(all_op.index(update) + 1)
            self._store_stream_parameter(
                parameter_bank, all_op, update, update_parameter
            )
            if self.vocabulary.name(values[position]) != "path_end":
                raise ValueError("Validated stream sequence lost path_end alignment.")
            position += 1
            matrices.append(nodes_to_path(nodes))

        position = self._decode_rates(values, position, configuration)
        if self.vocabulary.name(values[position]) != "archive_begin":
            raise ValueError("Validated stream sequence lost archive_begin alignment.")
        position += 1
        archive = []
        while self.vocabulary.name(values[position]) != "archive_end":
            archive.append(self.vocabulary.name(values[position]))
            position += 1
        if archive != list(self.vocabulary.archives):
            raise ValueError("Validated stream sequence changed the archive policy.")
        position += 1
        if values[position] != self.vocabulary.end_index:
            raise ValueError("Validated stream sequence lost end alignment.")

        validate_paths(matrices, self.setting)
        design = Design.from_genotype(
            matrices,
            parameter_bank,
            self.problems,
            self.setting,
            configuration=configuration,
        )
        if not configuration_is_valid(design, configuration):
            raise ValueError(
                "Invalid population/offspring sizes for the selected components."
            )
        design.learning_sequence = self._encode_stream(design)
        return design

    def canonicalize(self, sequence: Sequence[int] | np.ndarray) -> list[int]:
        """Return the unique token form of one executable algorithm."""

        design = self.decode(sequence)
        return self.encode(design)

    def encode(self, design: Design) -> list[int]:
        """Encode a graph, quantizing parameters while preserving its performance."""

        if self.grammar.stream_graph:
            return self._encode_stream(design)

        operators = getattr(design, "operator_pheno", None)
        parameters = getattr(design, "parameter_pheno", None)
        if not operators or not parameters or not operators[0] or not parameters[0]:
            raise ValueError("The algorithm must contain a decoded graph phenotype.")
        pathways = list(operators[0])
        pathway_parameters = list(parameters[0])
        if len(pathways) != self.grammar.max_pathways:
            raise ValueError(
                f"Expected exactly {self.grammar.max_pathways} pathways, "
                f"got {len(pathways)}."
            )
        if len(pathways) != len(pathway_parameters):
            raise ValueError("Pathway operator and parameter counts do not match.")

        first_path = pathways[0]
        first_params = pathway_parameters[0]
        expected_archive = list(self.vocabulary.archives)
        local_termination = self._local_termination()
        problem_type = get_problem_type(self.problems)
        values = [
            self.vocabulary.begin_index,
            self.vocabulary.index(f"problem_{problem_type}"),
        ]
        values.extend(self._encode_size_configuration(design))
        configuration = copy_configuration(design, self.setting)
        boundary = str(configuration["boundary_handling"])
        if self.grammar.encode_boundary_handling:
            values.append(
                self.vocabulary.parameter_indices[
                    ("clip", "reflect", "resample").index(boundary)
                ]
            )
        elif boundary != "clip":
            raise ValueError(
                "This legacy learning grammar only represents clip boundaries."
            )
        values.extend(self._encode_component(first_path.choose, first_params.choose))
        for path, params in zip(pathways, pathway_parameters):
            if path.choose != first_path.choose or path.update != first_path.update:
                raise ValueError(
                    "All AutoOptLib pathways must share choose and update."
                )
            if not self._same_parameter(params.choose, first_params.choose) or not (
                self._same_parameter(params.update, first_params.update)
            ):
                raise ValueError(
                    "All AutoOptLib pathways must share choose/update parameters."
                )
            if list(path.archive) != expected_archive:
                raise ValueError(
                    "Graph archives must match the codec's fixed Archive setting."
                )
            if len(path.search) != len(params.search):
                raise ValueError(
                    "Search-step operator and parameter counts do not match."
                )
            values.append(self.vocabulary.index("path_begin"))
            component_count = 0
            global_count = 0
            for step, step_params in zip(path.search, params.search):
                termination = np.asarray(step.termination, dtype=float).reshape(-1)
                is_global = self._is_global_termination(termination)
                primary_requirement = "local"
                secondary_requirement = "local"
                if is_global:
                    # Preserve the component that actually makes this paired step
                    # global. Merely being global-capable does not imply that the
                    # primary's current parameters describe global search.
                    primary_tokens = self._encode_component(
                        step.primary, step_params.primary
                    )
                    primary_bins = [
                        self.vocabulary.parameter_indices.index(token)
                        for token in primary_tokens[1:]
                    ]
                    secondary_tokens = (
                        self._encode_component(step.secondary, step_params.secondary)
                        if step.secondary is not None
                        else []
                    )
                    secondary_bins = [
                        self.vocabulary.parameter_indices.index(token)
                        for token in secondary_tokens[1:]
                    ]
                    if self.vocabulary.behavior(step.primary, primary_bins) == "global":
                        primary_requirement, secondary_requirement = "global", None
                    elif (
                        step.secondary is not None
                        and self.vocabulary.behavior(step.secondary, secondary_bins)
                        == "global"
                    ):
                        primary_requirement, secondary_requirement = None, "global"
                    elif self.vocabulary.has_global_choice(step.primary):
                        primary_requirement, secondary_requirement = "global", None
                    elif (
                        step.secondary is not None
                        and self.vocabulary.has_global_choice(step.secondary)
                    ):
                        primary_requirement, secondary_requirement = None, "global"
                    else:
                        raise ValueError(
                            "A global search step needs a global-capable component."
                        )
                values.extend(
                    self._encode_component(
                        step.primary, step_params.primary, primary_requirement
                    )
                )
                component_count += 1
                if step.secondary is not None:
                    values.append(self.vocabulary.index("secondary"))
                    values.extend(
                        self._encode_component(
                            step.secondary,
                            step_params.secondary,
                            secondary_requirement,
                        )
                    )
                    component_count += 1
                expected_termination = (
                    np.asarray([-math.inf, 1.0]) if is_global else local_termination
                )
                if termination.shape != expected_termination.shape or not np.allclose(
                    termination, expected_termination
                ):
                    raise ValueError(
                        "Search termination is outside the canonical AutoOptLib "
                        "learning grammar."
                    )
                if is_global:
                    global_count += 1
                values.append(
                    self.vocabulary.index(
                        "terminate_global" if is_global else "terminate_local"
                    )
                )
            if not path.search:
                raise ValueError("Every pathway must contain at least one search step.")
            if component_count > self.grammar.max_search_components:
                raise ValueError("A pathway contains more search components than AlgQ.")
            if global_count > 1:
                raise ValueError(
                    "A pathway may contain at most one global search step."
                )
            values.append(self.vocabulary.index("path_end"))
        values.extend(self._encode_rates(design, pathways))
        values.extend(self._encode_component(first_path.update, first_params.update))
        values.append(self.vocabulary.index("archive_begin"))
        values.extend(self.vocabulary.index(name) for name in expected_archive)
        values.append(self.vocabulary.index("archive_end"))
        values.append(self.vocabulary.end_index)
        return self.grammar.validate(values)

    def decode(self, sequence: Sequence[int] | np.ndarray) -> Design:
        """Decode one valid sequence into an executable AutoOptLib graph."""

        if self.grammar.stream_graph:
            return self._decode_stream(sequence)

        values = self.grammar.validate(sequence)
        problem_type = get_problem_type(self.problems)
        if self.vocabulary.name(values[1]) != f"problem_{problem_type}":
            raise ValueError("Sequence problem type does not match the codec target.")
        position = 2
        configuration, position = self._decode_configuration(values, position)
        if self.grammar.encode_boundary_handling:
            boundary_index = self.vocabulary.parameter_indices.index(values[position])
            configuration["boundary_handling"] = ("clip", "reflect", "resample")[
                boundary_index
            ]
            position += 1
        choose, choose_parameter, position = self._decode_component(values, position)
        pathways: list[Pathway] = []
        pathway_parameters: list[PathwayParam] = []
        path_searches: list[list[SearchStep]] = []
        path_parameters: list[list[SearchParam]] = []
        local_termination = self._local_termination()
        for _ in range(self.grammar.max_pathways):
            if self.vocabulary.name(values[position]) != "path_begin":
                raise ValueError("Validated sequence lost path_begin alignment.")
            position += 1
            searches: list[SearchStep] = []
            search_parameters: list[SearchParam] = []
            while self.vocabulary.name(values[position]) != "path_end":
                primary, primary_parameter, position = self._decode_component(
                    values, position
                )
                secondary = None
                secondary_parameter = None
                if self.vocabulary.name(values[position]) == "secondary":
                    position += 1
                    secondary, secondary_parameter, position = self._decode_component(
                        values, position
                    )
                termination_name = self.vocabulary.name(values[position])
                position += 1
                termination = (
                    np.asarray([-math.inf, 1.0])
                    if termination_name == "terminate_global"
                    else local_termination.copy()
                )
                searches.append(SearchStep(primary, termination, secondary))
                search_parameters.append(
                    SearchParam(primary_parameter, secondary_parameter)
                )
            position += 1
            path_searches.append(searches)
            path_parameters.append(search_parameters)

        position = self._decode_rates(values, position, configuration)
        update, update_parameter, position = self._decode_component(values, position)
        if self.vocabulary.name(values[position]) != "archive_begin":
            raise ValueError("Validated sequence lost archive_begin alignment.")
        position += 1
        archive = []
        while self.vocabulary.name(values[position]) != "archive_end":
            archive.append(self.vocabulary.name(values[position]))
            position += 1
        position += 1
        if values[position] != self.vocabulary.end_index:
            raise ValueError("Validated sequence lost end alignment.")
        for searches, search_parameters in zip(path_searches, path_parameters):
            pathways.append(Pathway(choose, searches, update, list(archive)))
            pathway_parameters.append(
                PathwayParam(choose_parameter, search_parameters, update_parameter)
            )

        all_op = list(
            getattr(self.setting, "AllOp", getattr(self.setting, "all_op", ()))
        )
        stream_graph = is_stream_graph(self.setting)
        matrices = []
        encoded_parameters: list[list[Any]] = [[None, None] for _ in all_op]
        for path, params in zip(pathways, pathway_parameters):
            raw_names = [] if stream_graph else [choose]
            for step, step_params in zip(path.search, params.search):
                if stream_graph:
                    if step.secondary is not None:
                        raise ValueError(
                            "Search v6.0 learning graphs do not support secondary "
                            "search components."
                        )
                    raw_names.append(choose)
                raw_names.append(step.primary)
                encoded_parameters[all_op.index(step.primary)] = [
                    step_params.primary,
                    "GS" if self._is_global_termination(step.termination) else "LS",
                ]
                if step.secondary is not None:
                    raw_names.append(step.secondary)
                    encoded_parameters[all_op.index(step.secondary)] = [
                        step_params.secondary,
                        "LS",
                    ]
            raw_names.append(update)
            indices = [all_op.index(name) + 1 for name in raw_names]
            matrices.append(np.asarray(list(zip(indices[:-1], indices[1:])), dtype=int))
        encoded_parameters[all_op.index(choose)] = [choose_parameter, None]
        encoded_parameters[all_op.index(update)] = [update_parameter, None]

        if stream_graph:
            # Search- and learning-generated v6 graphs must pass through the
            # same typed edge decoder.  The current learning vocabulary has a
            # shared choose/update token, so it spans a legal subset of the
            # expanded graph rather than inventing a second execution path.
            validate_paths(matrices, self.setting)
            design = Design.from_genotype(
                matrices,
                encoded_parameters,
                self.problems,
                self.setting,
                configuration=configuration,
            )
        else:
            design = Design()
            design.operator = matrices
            design.parameter = encoded_parameters
            design.construct([pathways], [pathway_parameters])
            set_configuration(design, configuration)
        if not configuration_is_valid(design, configuration):
            raise ValueError(
                "Invalid population/offspring sizes for the selected components."
            )
        runs = int(
            getattr(self.setting, "AlgRuns", getattr(self.setting, "alg_runs", 1))
        )
        design.performance = np.zeros((len(self.problems), runs))
        design.performance_approx = np.zeros((len(self.problems), runs))
        design.last_runs = {index: [None] * runs for index in range(len(self.problems))}
        design.learning_sequence = values
        return design


def encode_design(
    design: Design,
    problem: Any,
    setting: Any,
    *,
    vocabulary: LearningVocabulary | None = None,
) -> list[int]:
    """Encode an AutoOptLib graph into its canonical quantized sequence."""

    return LearningCodec.from_problem(problem, setting, vocabulary=vocabulary).encode(
        design
    )


def decode_sequence(
    sequence: Sequence[int] | np.ndarray,
    problem: Any,
    setting: Any,
    *,
    vocabulary: LearningVocabulary | None = None,
) -> Design:
    """Decode a learning sequence into an executable AutoOptLib design."""

    return LearningCodec.from_problem(problem, setting, vocabulary=vocabulary).decode(
        sequence
    )


__all__ = ["LearningCodec", "decode_sequence", "encode_design"]

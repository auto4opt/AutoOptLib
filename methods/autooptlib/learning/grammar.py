"""Grammar masks for canonical AutoOptLib graph-construction sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..utils.design._population import is_mutation_component
from .vocabulary import LearningVocabulary

_MUTATION_PREFIXES = (
    "search_mu_",
    "search_reset_",
    "search_bit_",
    "search_swap",
    "search_scramble",
    "search_insert",
)


class SequenceValidationError(ValueError):
    """Raised when a learning sequence is not a valid graph construction."""


@dataclass
class _State:
    expected: str = "problem"
    active_problem_type: str | None = None
    after_parameters: str | None = None
    remaining_parameters: int = 0
    current_component: str | None = None
    current_parameter_bins: list[int] = field(default_factory=list)
    component_parameter_bins: dict[str, tuple[int, ...]] = field(default_factory=dict)
    pathways: int = 0
    search_components: int = 0
    primary_is_crossover: bool = False
    step_global: bool = False
    global_used: bool = False
    archive_position: int = 0
    configuration_digits: list[int] = field(default_factory=list)
    population_size_index: int | None = None
    offspring_size_index: int | None = None
    crossover_used: bool = False
    mutation_used: bool = False
    crossover_pathways: int = 0
    paired_mutation_used: bool = False
    ended: bool = False


@dataclass(frozen=True)
class LearningGrammar:
    """A bounded grammar matching AutoOptLib's repaired pathway graphs."""

    vocabulary: LearningVocabulary
    max_pathways: int = 1
    max_search_components: int = 4
    population_size_bins: int = 10
    offspring_size_bins: int = 10
    configuration_digits: int = 3
    encode_boundary_handling: bool = False
    population_size_values: tuple[int, ...] = ()
    offspring_size_values: tuple[int, ...] = ()
    graph_semantics: str = "legacy_pathway_v1"

    def __post_init__(self) -> None:
        if type(self.encode_boundary_handling) is not bool:
            raise ValueError("encode_boundary_handling must be boolean.")
        if self.graph_semantics not in {"legacy_pathway_v1", "stream_graph_v2"}:
            raise ValueError("Unsupported learning graph semantics.")
        for name in (
            "max_pathways",
            "max_search_components",
            "population_size_bins",
            "offspring_size_bins",
            "configuration_digits",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if self.stream_graph and (
            self.max_pathways > 2 or self.max_search_components > 2
        ):
            raise ValueError("stream_graph_v2 requires AlgP<=2 and AlgQ<=2.")
        maximum = 10**self.configuration_digits
        if not 1 <= self.population_size_bins <= maximum:
            raise ValueError("population_size_bins is outside the vocabulary.")
        if not 1 <= self.offspring_size_bins <= maximum:
            raise ValueError("offspring_size_bins is outside the vocabulary.")
        population_values = self.population_size_values or tuple(
            range(self.population_size_bins)
        )
        offspring_values = self.offspring_size_values or tuple(
            range(self.offspring_size_bins)
        )
        try:
            normalized_population = tuple(int(value) for value in population_values)
            normalized_offspring = tuple(int(value) for value in offspring_values)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "Population and offspring sizes must be integers."
            ) from exc
        if any(
            isinstance(raw, (bool, np.bool_)) or raw != value
            for raw, value in zip(population_values, normalized_population)
        ) or any(
            isinstance(raw, (bool, np.bool_)) or raw != value
            for raw, value in zip(offspring_values, normalized_offspring)
        ):
            raise ValueError("Population and offspring sizes must be integers.")
        population_values = normalized_population
        offspring_values = normalized_offspring
        if len(population_values) != self.population_size_bins:
            raise ValueError("population_size_values must match population_size_bins.")
        if len(offspring_values) != self.offspring_size_bins:
            raise ValueError("offspring_size_values must match offspring_size_bins.")
        if len(set(population_values)) != len(population_values):
            raise ValueError("population_size_values must be unique.")
        if len(set(offspring_values)) != len(offspring_values):
            raise ValueError("offspring_size_values must be unique.")
        object.__setattr__(
            self,
            "population_size_values",
            population_values,
        )
        object.__setattr__(
            self,
            "offspring_size_values",
            offspring_values,
        )

    @property
    def stream_graph(self) -> bool:
        return self.graph_semantics == "stream_graph_v2"

    def _finish_component(self, state: _State) -> None:
        component = state.current_component
        after = state.after_parameters
        if component is None or after is None:
            raise RuntimeError("Incomplete learning-grammar component state.")
        if not self.stream_graph and after in {"after_primary", "termination"}:
            behavior = self.vocabulary.behavior(component, state.current_parameter_bins)
            if behavior == "global":
                state.step_global = True
        if self.stream_graph:
            bins = tuple(state.current_parameter_bins)
            previous = state.component_parameter_bins.get(component)
            if previous is not None and previous != bins:
                raise SequenceValidationError(
                    f"Repeated component {component!r} must reuse its parameters."
                )
            state.component_parameter_bins[component] = bins
        state.current_component = None
        state.current_parameter_bins = []
        state.expected = after

    def _component(self, state: _State, name: str, after: str) -> None:
        state.current_component = name
        state.current_parameter_bins = []
        state.remaining_parameters = self.vocabulary.parameter_count(name)
        state.after_parameters = after
        if state.remaining_parameters:
            state.expected = "parameters"
        else:
            self._finish_component(state)

    def _component_allowed(self, state: _State, component: str) -> bool:
        compatible = (
            state.active_problem_type is not None
            and self.vocabulary.compatible(component, state.active_problem_type)
        )
        valid_ssga_sizes = True
        if component.startswith("update_ssga_"):
            if (
                state.population_size_index is None
                or state.offspring_size_index is None
            ):
                valid_ssga_sizes = False
            else:
                valid_ssga_sizes = (
                    self.offspring_size_values[state.offspring_size_index]
                    <= self.population_size_values[state.population_size_index]
                )
        valid_pso_sizes = component != "search_pso" or (
            state.population_size_index is not None
            and state.offspring_size_index is not None
            and self.population_size_values[state.population_size_index]
            == self.offspring_size_values[state.offspring_size_index]
        )
        return (
            compatible
            and valid_pso_sizes
            and valid_ssga_sizes
            and (
                self.stream_graph
                or not state.global_used
                or self.vocabulary.has_local_choice(component)
            )
        )

    def _stream_mutations(self, state: _State) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.vocabulary.search
            if is_mutation_component(name) and self._component_allowed(state, name)
        )

    def _start_stream_search(self, state: _State, name: str) -> None:
        if name not in self.vocabulary.search or not self._component_allowed(
            state, name
        ):
            raise SequenceValidationError("Expected a compatible stream search.")
        if state.search_components == 0:
            state.primary_is_crossover = name.startswith("cross_")
            if state.primary_is_crossover:
                state.crossover_pathways += 1
        elif state.search_components == 1:
            if not state.primary_is_crossover or not is_mutation_component(name):
                raise SequenceValidationError(
                    "A second stream stage must be a mutation after a crossover."
                )
            state.paired_mutation_used = True
        else:
            raise SequenceValidationError("The stream pathway exceeds AlgQ.")
        state.search_components += 1
        self._component(state, name, "stream_termination")

    def _rate_active(self, state: _State, name: str) -> bool:
        if not self.stream_graph:
            return state.crossover_used if name == "crossover" else state.mutation_used
        if name == "crossover":
            return state.pathways == 2 and state.crossover_pathways == 1
        return state.paired_mutation_used

    def _consume_rate(self, state: _State, token: int, name: str) -> None:
        vocabulary = self.vocabulary
        if token not in vocabulary.parameter_indices:
            raise SequenceValidationError(f"Expected a {name}-rate parameter token.")
        if (
            not self._rate_active(state, name)
            and token != vocabulary.parameter_indices[-1]
        ):
            raise SequenceValidationError(
                f"Unused {name}_rate must use its canonical value 1."
            )
        state.expected = (
            "mutation_rate"
            if name == "crossover"
            else ("archive_begin" if self.stream_graph else "update")
        )

    def _consume(self, state: _State, token: int) -> None:
        vocabulary = self.vocabulary
        try:
            name = vocabulary.name(token)
        except KeyError as exc:
            raise SequenceValidationError(str(exc)) from exc
        if state.ended:
            if token != vocabulary.end_index:
                raise SequenceValidationError("Only end padding may follow end.")
            return
        expected = state.expected
        if expected == "problem":
            if not name.startswith("problem_"):
                raise SequenceValidationError("Expected a problem-type token.")
            problem_type = name.removeprefix("problem_")
            if problem_type not in vocabulary.supported_problem_types:
                raise SequenceValidationError("Problem type is outside the vocabulary.")
            state.active_problem_type = problem_type
            state.expected = "population_size"
        elif expected == "population_size":
            if token not in vocabulary.parameter_indices:
                raise SequenceValidationError("Expected a population-size digit token.")
            state.configuration_digits.append(vocabulary.parameter_indices.index(token))
            if len(state.configuration_digits) == self.configuration_digits:
                value = self._configuration_value(state.configuration_digits)
                if value >= self.population_size_bins:
                    raise SequenceValidationError(
                        "Population-size index is outside the configured space."
                    )
                state.population_size_index = value
                state.configuration_digits = []
                state.expected = "offspring_size"
        elif expected == "offspring_size":
            if token not in vocabulary.parameter_indices:
                raise SequenceValidationError("Expected an offspring-size digit token.")
            state.configuration_digits.append(vocabulary.parameter_indices.index(token))
            if len(state.configuration_digits) == self.configuration_digits:
                value = self._configuration_value(state.configuration_digits)
                if value >= self.offspring_size_bins:
                    raise SequenceValidationError(
                        "Offspring-size index is outside the configured space."
                    )
                state.offspring_size_index = value
                state.configuration_digits = []
                state.expected = (
                    "boundary_handling"
                    if self.encode_boundary_handling
                    else ("path_begin" if self.stream_graph else "choose")
                )
        elif expected == "boundary_handling":
            count = 3 if state.active_problem_type == "continuous" else 1
            if token not in vocabulary.parameter_indices[:count]:
                raise SequenceValidationError(
                    "Expected a compatible boundary-handling token."
                )
            state.expected = "path_begin" if self.stream_graph else "choose"
        elif expected == "choose":
            if name not in vocabulary.choose or not self._component_allowed(
                state, name
            ):
                raise SequenceValidationError(
                    "A sequence must begin with a choose component."
                )
            self._component(state, name, "path_begin")
        elif expected == "parameters":
            if token not in vocabulary.parameter_indices:
                raise SequenceValidationError(
                    "A component parameter requires a parameter-bin token."
                )
            bin_index = vocabulary.parameter_indices.index(token)
            component = str(state.current_component)
            mode = vocabulary.behavior_mode(component)
            parameter_position = len(state.current_parameter_bins)
            if not self.stream_graph and state.global_used and mode == "parameter":
                masks = vocabulary.local_masks(component)
                if not masks[parameter_position] & (1 << bin_index):
                    raise SequenceValidationError(
                        "A second global-search component is not permitted."
                    )
            state.current_parameter_bins.append(bin_index)
            previous = (
                state.component_parameter_bins.get(component)
                if self.stream_graph
                else None
            )
            if (
                previous is not None
                and previous[len(state.current_parameter_bins) - 1] != bin_index
            ):
                raise SequenceValidationError(
                    f"Repeated component {component!r} must reuse its parameters."
                )
            state.remaining_parameters -= 1
            if state.remaining_parameters == 0:
                self._finish_component(state)
        elif expected == "path_begin":
            if name != "path_begin":
                raise SequenceValidationError("Expected path_begin.")
            if state.pathways >= self.max_pathways:
                raise SequenceValidationError("The graph exceeds AlgP.")
            state.pathways += 1
            state.search_components = 0
            state.global_used = False
            state.primary_is_crossover = False
            state.expected = "stream_stage_start" if self.stream_graph else "primary"
        elif expected == "stream_stage_start":
            if name in vocabulary.choose and self._component_allowed(state, name):
                self._component(state, name, "stream_search")
            else:
                self._start_stream_search(state, name)
        elif expected == "stream_search":
            self._start_stream_search(state, name)
        elif expected == "stream_termination":
            if name != "terminate_local":
                raise SequenceValidationError(
                    "stream_graph_v2 stages require the canonical termination token."
                )
            if state.search_components == 1 and state.primary_is_crossover:
                state.expected = "stream_after_first"
            else:
                state.expected = "stream_update"
        elif expected == "stream_after_first":
            if name in vocabulary.update and self._component_allowed(state, name):
                self._component(state, name, "stream_path_end")
            elif (
                state.search_components < self.max_search_components
                and name in vocabulary.choose
                and self._component_allowed(state, name)
                and self._stream_mutations(state)
            ):
                self._component(state, name, "stream_second_search")
            elif (
                state.search_components < self.max_search_components
                and name in self._stream_mutations(state)
            ):
                self._start_stream_search(state, name)
            else:
                raise SequenceValidationError(
                    "Expected an update or an optional choose/mutation stage."
                )
        elif expected == "stream_second_search":
            if name not in self._stream_mutations(state):
                raise SequenceValidationError(
                    "A crossover's second stream stage must be a mutation."
                )
            self._start_stream_search(state, name)
        elif expected == "stream_update":
            if name not in vocabulary.update or not self._component_allowed(
                state, name
            ):
                raise SequenceValidationError("Expected a stream update component.")
            self._component(state, name, "stream_path_end")
        elif expected == "stream_path_end":
            if name != "path_end":
                raise SequenceValidationError("Expected path_end after stream update.")
            state.expected = (
                "crossover_rate"
                if state.pathways == self.max_pathways
                else "stream_path_or_rate"
            )
        elif expected == "stream_path_or_rate":
            if name == "path_begin":
                if state.pathways >= self.max_pathways:
                    raise SequenceValidationError("The graph exceeds AlgP.")
                state.pathways += 1
                state.search_components = 0
                state.global_used = False
                state.primary_is_crossover = False
                state.expected = "stream_stage_start"
            else:
                self._consume_rate(state, token, "crossover")
        elif expected == "primary":
            if name not in vocabulary.search or not self._component_allowed(
                state, name
            ):
                raise SequenceValidationError("Expected a compatible primary search.")
            state.search_components += 1
            state.primary_is_crossover = name.startswith("cross_")
            state.crossover_used |= state.primary_is_crossover
            state.mutation_used |= name.startswith(_MUTATION_PREFIXES)
            state.step_global = False
            self._component(state, name, "after_primary")
        elif expected == "after_primary":
            if name == "secondary":
                if not state.primary_is_crossover:
                    raise SequenceValidationError(
                        "Only a crossover may introduce a secondary search."
                    )
                if state.search_components >= self.max_search_components:
                    raise SequenceValidationError(
                        "The path has no room for a secondary search."
                    )
                state.expected = "secondary"
            elif name in {"terminate_local", "terminate_global"}:
                self._termination(state, name)
            else:
                raise SequenceValidationError(
                    "Expected secondary or a termination token."
                )
        elif expected == "secondary":
            if name not in vocabulary.secondary_search(
                str(state.active_problem_type)
            ) or not self._component_allowed(state, name):
                raise SequenceValidationError(
                    "Expected a compatible secondary search component."
                )
            state.search_components += 1
            state.mutation_used = True
            self._component(state, name, "termination")
        elif expected == "termination":
            if name not in {"terminate_local", "terminate_global"}:
                raise SequenceValidationError("Expected a termination token.")
            self._termination(state, name)
        elif expected == "after_step":
            if name == "path_end":
                state.expected = (
                    "path_begin"
                    if state.pathways < self.max_pathways
                    else "crossover_rate"
                )
            elif name in vocabulary.search and self._component_allowed(state, name):
                if state.search_components >= self.max_search_components:
                    raise SequenceValidationError("The path exceeds AlgQ.")
                state.search_components += 1
                state.primary_is_crossover = name.startswith("cross_")
                state.crossover_used |= state.primary_is_crossover
                state.mutation_used |= name.startswith(_MUTATION_PREFIXES)
                state.step_global = False
                self._component(state, name, "after_primary")
            else:
                raise SequenceValidationError("Expected another search or path_end.")
        elif expected == "crossover_rate":
            self._consume_rate(state, token, "crossover")
        elif expected == "mutation_rate":
            self._consume_rate(state, token, "mutation")
        elif expected == "update":
            if name not in vocabulary.update or not self._component_allowed(
                state, name
            ):
                raise SequenceValidationError("Expected an update component.")
            self._component(state, name, "archive_begin")
        elif expected == "archive_begin":
            if name != "archive_begin":
                raise SequenceValidationError("Expected archive_begin.")
            state.archive_position = 0
            state.expected = "archive_item" if vocabulary.archives else "archive_end"
        elif expected == "archive_item":
            required = vocabulary.archives[state.archive_position]
            if name != required:
                raise SequenceValidationError(
                    f"Expected archive component {required!r}."
                )
            state.archive_position += 1
            if state.archive_position == len(vocabulary.archives):
                state.expected = "archive_end"
        elif expected == "archive_end":
            if name != "archive_end":
                raise SequenceValidationError("Expected archive_end.")
            state.expected = "end"
        elif expected == "end":
            if name != "end":
                raise SequenceValidationError("Expected end.")
            state.ended = True
        else:  # pragma: no cover - defensive state-machine guard
            raise RuntimeError(f"Unknown learning grammar state {expected!r}.")

    @staticmethod
    def _termination(state: _State, name: str) -> None:
        expected = "terminate_global" if state.step_global else "terminate_local"
        if name != expected:
            raise SequenceValidationError(
                f"Component behavior requires {expected}, got {name}."
            )
        if state.step_global:
            if state.global_used:
                raise SequenceValidationError(
                    "An AutoOptLib pathway may contain at most one global search."
                )
            state.global_used = True
        state.expected = "after_step"

    @staticmethod
    def _configuration_value(digits: Sequence[int]) -> int:
        return int("".join(str(int(digit)) for digit in digits))

    def _configuration_digit_indices(
        self, prefix: Sequence[int], number_of_values: int
    ) -> list[int]:
        """Return decimal digits which can still complete a valid index."""

        remaining = self.configuration_digits - len(prefix) - 1
        allowed = []
        for digit in range(10):
            candidate = [*prefix, digit]
            head = self._configuration_value(candidate)
            lower = head * (10**remaining)
            if lower < number_of_values:
                allowed.append(self.vocabulary.parameter_indices[digit])
        return allowed

    def _state(self, sequence: Sequence[int] | np.ndarray) -> _State:
        values = self._integer_tokens(sequence)
        if not values or values[0] != self.vocabulary.begin_index:
            raise SequenceValidationError("A learning sequence must start with begin.")
        state = _State()
        for token in values[1:]:
            self._consume(state, token)
        return state

    @staticmethod
    def _integer_tokens(sequence: Sequence[int] | np.ndarray) -> list[int]:
        # Object dtype preserves booleans and fractional/string inputs instead
        # of silently coercing a heterogeneous Python sequence to integers.
        raw_values = np.asarray(sequence, dtype=object).reshape(-1).tolist()
        values = []
        for raw in raw_values:
            try:
                token = int(raw)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SequenceValidationError(
                    "Learning sequences must contain integers."
                ) from exc
            if isinstance(raw, (bool, np.bool_)) or raw != token:
                raise SequenceValidationError(
                    "Learning sequences must contain integers."
                )
            values.append(token)
        return values

    def _enable_components(
        self, allowed: np.ndarray, state: _State, components: Sequence[str]
    ) -> None:
        for component in components:
            if self._component_allowed(state, component):
                allowed[self.vocabulary.index(component)] = True

    def allowed_next_tokens(
        self,
        sequence: Sequence[int] | np.ndarray,
        *,
        required_problem_type: str | None = None,
    ) -> np.ndarray:
        state = self._state(sequence)
        vocabulary = self.vocabulary
        allowed = np.zeros(vocabulary.size, dtype=bool)

        def enable(names: Sequence[str]) -> None:
            for name in names:
                allowed[vocabulary.index(name)] = True

        if state.ended:
            allowed[vocabulary.end_index] = True
        elif state.expected == "problem":
            problem_types = vocabulary.supported_problem_types
            if required_problem_type is not None:
                if required_problem_type not in problem_types:
                    raise ValueError(
                        "Required problem type is outside the learning vocabulary."
                    )
                problem_types = (required_problem_type,)
            enable(tuple(f"problem_{problem_type}" for problem_type in problem_types))
        elif state.expected == "population_size":
            allowed[
                self._configuration_digit_indices(
                    state.configuration_digits, self.population_size_bins
                )
            ] = True
        elif state.expected == "offspring_size":
            allowed[
                self._configuration_digit_indices(
                    state.configuration_digits, self.offspring_size_bins
                )
            ] = True
        elif state.expected == "crossover_rate":
            indices = (
                vocabulary.parameter_indices
                if self._rate_active(state, "crossover")
                else (vocabulary.parameter_indices[-1],)
            )
            allowed[list(indices)] = True
        elif state.expected == "mutation_rate":
            indices = (
                vocabulary.parameter_indices
                if self._rate_active(state, "mutation")
                else (vocabulary.parameter_indices[-1],)
            )
            allowed[list(indices)] = True
        elif state.expected == "boundary_handling":
            count = 3 if state.active_problem_type == "continuous" else 1
            allowed[list(vocabulary.parameter_indices[:count])] = True
        elif state.expected == "choose":
            self._enable_components(allowed, state, vocabulary.choose)
        elif state.expected == "parameters":
            indices = list(vocabulary.parameter_indices)
            component = str(state.current_component)
            previous = (
                state.component_parameter_bins.get(component)
                if self.stream_graph
                else None
            )
            if previous is not None:
                position = len(state.current_parameter_bins)
                indices = [vocabulary.parameter_indices[previous[position]]]
            if (
                not self.stream_graph
                and state.global_used
                and vocabulary.behavior_mode(component) == "parameter"
            ):
                position = len(state.current_parameter_bins)
                mask = vocabulary.local_masks(component)[position]
                indices = [
                    index
                    for bin_value, index in enumerate(indices)
                    if mask & (1 << vocabulary.parameter_indices.index(index))
                ]
            allowed[indices] = True
        elif state.expected == "path_begin":
            enable(("path_begin",))
        elif state.expected == "stream_stage_start":
            self._enable_components(allowed, state, vocabulary.choose)
            self._enable_components(allowed, state, vocabulary.search)
        elif state.expected == "stream_search":
            components = (
                vocabulary.search
                if state.search_components == 0
                else self._stream_mutations(state)
            )
            self._enable_components(allowed, state, components)
        elif state.expected == "stream_termination":
            enable(("terminate_local",))
        elif state.expected == "stream_after_first":
            self._enable_components(allowed, state, vocabulary.update)
            mutations = self._stream_mutations(state)
            if state.search_components < self.max_search_components and mutations:
                self._enable_components(allowed, state, vocabulary.choose)
                self._enable_components(allowed, state, mutations)
        elif state.expected == "stream_second_search":
            self._enable_components(allowed, state, self._stream_mutations(state))
        elif state.expected == "stream_update":
            self._enable_components(allowed, state, vocabulary.update)
        elif state.expected == "stream_path_end":
            enable(("path_end",))
        elif state.expected == "stream_path_or_rate":
            if state.pathways < self.max_pathways:
                enable(("path_begin",))
            indices = (
                vocabulary.parameter_indices
                if self._rate_active(state, "crossover")
                else (vocabulary.parameter_indices[-1],)
            )
            allowed[list(indices)] = True
        elif state.expected == "primary":
            self._enable_components(allowed, state, vocabulary.search)
        elif state.expected == "after_primary":
            enable(("terminate_global" if state.step_global else "terminate_local",))
            if (
                state.primary_is_crossover
                and state.search_components < self.max_search_components
                and vocabulary.secondary_search(str(state.active_problem_type))
            ):
                enable(("secondary",))
        elif state.expected == "secondary":
            self._enable_components(
                allowed,
                state,
                vocabulary.secondary_search(str(state.active_problem_type)),
            )
        elif state.expected == "termination":
            enable(("terminate_global" if state.step_global else "terminate_local",))
        elif state.expected == "after_step":
            enable(("path_end",))
            if state.search_components < self.max_search_components:
                self._enable_components(allowed, state, vocabulary.search)
        elif state.expected == "update":
            self._enable_components(allowed, state, vocabulary.update)
        elif state.expected == "archive_begin":
            enable(("archive_begin",))
        elif state.expected == "archive_item":
            enable((vocabulary.archives[state.archive_position],))
        elif state.expected == "archive_end":
            enable(("archive_end",))
        elif state.expected == "end":
            allowed[vocabulary.end_index] = True
        return allowed

    def normalize(self, sequence: Sequence[int] | np.ndarray) -> list[int]:
        values = self._integer_tokens(sequence)
        if not values:
            raise SequenceValidationError("Learning sequence cannot be empty.")
        if values[0] != self.vocabulary.begin_index:
            values.insert(0, self.vocabulary.begin_index)
        while len(values) > 1 and values[-1] == self.vocabulary.end_index:
            values.pop()
        values.append(self.vocabulary.end_index)
        return values

    def validate(self, sequence: Sequence[int] | np.ndarray) -> list[int]:
        values = self.normalize(sequence)
        state = self._state(values)
        if not state.ended:
            raise SequenceValidationError("Learning sequence is incomplete.")
        return values

    def maximum_length(self) -> int:
        """Return a conservative maximum token length for this grammar."""

        choose_parameters = max(
            (self.vocabulary.parameter_count(name) for name in self.vocabulary.choose),
            default=0,
        )
        search_parameters = max(
            (self.vocabulary.parameter_count(name) for name in self.vocabulary.search),
            default=0,
        )
        update_parameters = max(
            (self.vocabulary.parameter_count(name) for name in self.vocabulary.update),
            default=0,
        )
        if self.stream_graph:
            # Every stage may carry its own choose. Every path owns its update.
            return (
                1
                + 1
                + 2 * self.configuration_digits
                + int(self.encode_boundary_handling)
                + self.max_pathways
                * (
                    2
                    + self.max_search_components
                    * (
                        3
                        + choose_parameters * self.vocabulary.parameter_token_count
                        + search_parameters * self.vocabulary.parameter_token_count
                    )
                    + 1
                    + update_parameters * self.vocabulary.parameter_token_count
                )
                + 2
                + 2
                + len(self.vocabulary.archives)
                + 1
            )
        # begin, choose, paths, update, archives, and end. Each raw search
        # component receives one component token, its parameters, and at most
        # one structural/termination token.
        return (
            1
            + 1
            + 2 * self.configuration_digits
            + int(self.encode_boundary_handling)
            + 2
            + 1
            + choose_parameters * self.vocabulary.parameter_token_count
            + self.max_pathways
            * (
                2
                + self.max_search_components
                * (2 + search_parameters * self.vocabulary.parameter_token_count)
            )
            + 1
            + update_parameters * self.vocabulary.parameter_token_count
            + 2
            + len(self.vocabulary.archives)
            + 1
        )


__all__ = ["LearningGrammar", "SequenceValidationError"]

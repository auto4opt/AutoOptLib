"""Structure and unified mixed-integer CMA-ES actions for search design."""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from . import Design
from ._disturb import active_parameter_counts, active_parameter_loci, mutate_structure
from ._helpers import ensure_rng, get_flex
from ._population import (
    boundary_handling_space,
    configuration_activity,
    configuration_is_valid,
    copy_configuration,
    offspring_size_space,
    population_size_space,
    set_configuration,
)
from ._search_control import representation_key

STRUCTURE = "structure"
PARAMETER = "parameter"
STRUCTURE_NOVELTY_ATTEMPTS = 5
_CMA_INHERITABLE_STATE_KEYS = (
    "cma_parameter_signature",
    "cma_parameter_coordinates",
    "cma_parameter_mean",
    "cma_parameter_covariance",
    "cma_parameter_sigma",
    "cma_parameter_pc",
    "cma_parameter_ps",
    "cma_parameter_generation",
)


class StructureNoveltyExhausted(RuntimeError):
    """Raised when a structure action cannot discover a new graph."""

    def __init__(self, attempts: int) -> None:
        self.attempts = int(attempts)
        super().__init__(
            f"Could not produce a globally novel structure within {self.attempts} attempts."
        )


class ParameterSamplingExhausted(RuntimeError):
    """Raised when CMA cannot produce any unique graph-preserving candidate."""


SEARCH_ACTIONS = (STRUCTURE, PARAMETER)


def _relative_improvement(parent_cost: float, child_cost: float) -> float:
    if not np.isfinite(parent_cost) or not np.isfinite(child_cost):
        return 0.0
    scale = max(abs(parent_cost), abs(child_cost), np.finfo(float).eps)
    return max(0.0, float(parent_cost - child_cost) / scale)


@dataclass
class ActionSelector:
    """Track cost-normalized rewards and adapt the action budget split."""

    quality: dict[str, float] = field(
        default_factory=lambda: {action: 0.0 for action in SEARCH_ACTIONS}
    )
    attempts: dict[str, int] = field(
        default_factory=lambda: {action: 0 for action in SEARCH_ACTIONS}
    )
    evaluations: dict[str, float] = field(
        default_factory=lambda: {action: 0 for action in SEARCH_ACTIONS}
    )
    improvements: dict[str, int] = field(
        default_factory=lambda: {action: 0 for action in SEARCH_ACTIONS}
    )
    reward_ewma: float = 0.2
    probability_gain: float = 0.3

    @classmethod
    def from_setting(
        cls, setting: Any, payload: dict[str, Any] | None = None
    ) -> "ActionSelector":
        selector = cls(
            reward_ewma=float(get_flex(setting, "search_action_reward_ewma", 0.2)),
            probability_gain=float(
                get_flex(setting, "search_action_probability_gain", 0.3)
            ),
        )
        if not payload:
            return selector
        for name in ("attempts", "evaluations", "improvements"):
            values = payload.get(name)
            if not isinstance(values, dict):
                continue
            for action in SEARCH_ACTIONS:
                getattr(selector, name)[action] = (
                    float(values.get(action, 0))
                    if name == "evaluations"
                    else int(values.get(action, 0))
                )
        values = payload.get("quality")
        if isinstance(values, dict):
            for action in SEARCH_ACTIONS:
                selector.quality[action] = float(values.get(action, 0.0))
        return selector

    def observe(
        self,
        action: str,
        *,
        parent_cost: float,
        child_cost: float,
        evaluation_cost: float,
    ) -> dict[str, float]:
        cost = max(np.finfo(float).eps, float(evaluation_cost))
        improvement = _relative_improvement(parent_cost, child_cost)
        reward = improvement / cost
        self.attempts[action] += 1
        self.evaluations[action] += cost
        if improvement > 0:
            self.improvements[action] += 1
        return {"relative_improvement": float(improvement), "reward": float(reward)}

    def update_generation(
        self, rewards: dict[str, Sequence[float]]
    ) -> dict[str, float | None]:
        """Update each action quality once from its generation-mean reward."""
        means: dict[str, float | None] = {}
        for action in SEARCH_ACTIONS:
            values = [float(value) for value in rewards.get(action, ())]
            if not values:
                means[action] = None
                continue
            mean_reward = float(np.mean(values))
            self.quality[action] = (1.0 - self.reward_ewma) * self.quality[
                action
            ] + self.reward_ewma * mean_reward
            means[action] = mean_reward
        return means

    def parameter_probability(self, initial: float) -> float:
        """Return the reward-adaptive parameter-action probability."""
        baseline = float(initial)
        parameter_quality = float(self.quality[PARAMETER])
        structure_quality = float(self.quality[STRUCTURE])
        scale = abs(parameter_quality) + abs(structure_quality)
        if scale <= np.finfo(float).eps:
            return baseline
        probability = baseline + self.probability_gain * (
            (parameter_quality - structure_quality) / scale
        )
        if not 0.0 <= probability <= 1.0:
            raise ValueError(
                "The adaptive action probability left [0, 1]; reduce SearchActionProbabilityGain or move the baseline away from a bound."
            )
        return float(probability)

    def payload(self) -> dict[str, Any]:
        return {
            "quality": dict(self.quality),
            "attempts": dict(self.attempts),
            "evaluations": dict(self.evaluations),
            "improvements": dict(self.improvements),
            "reward_ewma": self.reward_ewma,
            "probability_gain": self.probability_gain,
        }


def choose_action(
    available: Sequence[str],
    rng: np.random.Generator,
    *,
    parameter_budget_fraction: float,
    parameter_evaluation_cost: int = 1,
    structure_evaluation_cost: int = 1,
) -> str:
    """Sample an action so the controller probability denotes budget share."""
    choices = tuple(dict.fromkeys(available))
    if not choices:
        raise ValueError("At least one search action must be available.")
    target_budget_fraction = float(parameter_budget_fraction)
    if not 0.0 <= target_budget_fraction <= 1.0:
        raise ValueError("parameter_budget_fraction must be between 0 and 1.")
    if PARAMETER in choices and STRUCTURE in choices:
        parameter_cost = int(parameter_evaluation_cost)
        structure_cost = int(structure_evaluation_cost)
        if parameter_cost <= 0 or structure_cost <= 0:
            raise ValueError("Action evaluation costs must be positive integers.")
        denominator = (
            target_budget_fraction * structure_cost
            + (1.0 - target_budget_fraction) * parameter_cost
        )
        probability = (
            target_budget_fraction * structure_cost / denominator
            if denominator > 0.0
            else target_budget_fraction
        )
        return PARAMETER if float(rng.random()) < probability else STRUCTURE
    return choices[0]


@dataclass
class ActionResult:
    action: str
    representative: Design
    evaluated: list[Design]
    evaluation_cost: int
    candidate_actions: tuple[str, ...] = ()
    structure_baseline: Design | None = None
    structure_candidate_evaluations: int = 0
    post_structure_parameter_evaluations: int = 0
    bootstrap_skip_reason: str | None = None


@dataclass(frozen=True)
class ActionCredit:
    """One reward observation, used either for control or diagnostics."""

    action: str
    parent: Design
    child: Design
    evaluation_cost: int
    context: str


def controller_action_credit(result: ActionResult, parent: Design) -> ActionCredit:
    """Return the reward for the actual outer action selected by Search.

    A structure action is one indivisible outer decision.  Its controller
    reward therefore covers every evaluated baseline and any completed
    bootstrap candidates.  Stream v6 may omit the bootstrap when the graph has
    no tunable variables or the exact budget tail is too short.
    """
    if result.action not in SEARCH_ACTIONS:
        raise ValueError(f"Unknown search action: {result.action}")
    return ActionCredit(
        result.action,
        parent,
        result.representative,
        int(result.evaluation_cost),
        "optional_parameter_block"
        if result.action == PARAMETER
        else "structure_only_block"
        if result.bootstrap_skip_reason is not None
        else "structure_bootstrap_block",
    )


def semantic_action_credits(
    result: ActionResult, parent: Design
) -> tuple[ActionCredit, ...]:
    """Describe structure and parameter contributions without steering control."""
    if result.action == PARAMETER:
        return (
            ActionCredit(
                PARAMETER,
                parent,
                result.representative,
                int(result.evaluation_cost),
                "optional_parameter_block",
            ),
        )
    if result.action != STRUCTURE:
        raise ValueError(f"Unknown search action: {result.action}")
    baseline = result.structure_baseline
    if baseline is None:
        raise ValueError("A structure result must retain its evaluated baseline.")
    credits = [
        ActionCredit(
            STRUCTURE,
            parent,
            baseline,
            max(1, int(result.structure_candidate_evaluations)),
            "structure_proposal_batch",
        )
    ]
    parameter_evaluations = int(result.post_structure_parameter_evaluations)
    if parameter_evaluations:
        credits.append(
            ActionCredit(
                PARAMETER,
                baseline,
                result.representative,
                parameter_evaluations,
                "post_structure_bootstrap",
            )
        )
    if sum((item.evaluation_cost for item in credits)) != int(result.evaluation_cost):
        raise ValueError(
            "Semantic credit costs must exactly cover evaluated candidates."
        )
    return tuple(credits)


def supports_parameter_action(candidate: Design, setting: Any) -> bool:
    """Return whether this algorithm exposes at least one tunable variable."""
    (discrete, continuous) = active_parameter_counts(candidate, setting)
    return bool(
        discrete
        or continuous
        or len(population_size_space(setting)) > 1
        or (len(offspring_size_space(setting)) > 1)
    )


def structure_action_evaluation_cost(setting: Any) -> int:
    """Return the configured size of one complete structure macro-action."""
    proposals = max(1, int(get_flex(setting, "structure_candidates_per_action", 3)))
    offspring = max(1, int(get_flex(setting, "parameter_cma_offspring", 5)))
    generations = max(1, int(get_flex(setting, "post_structure_cma_generations", 1)))
    return proposals + offspring * generations


def parameter_action_evaluation_cost(setting: Any) -> int:
    """Return the configured size of one complete parameter macro-action."""
    offspring = max(1, int(get_flex(setting, "parameter_cma_offspring", 5)))
    generations = max(1, int(get_flex(setting, "parameter_cma_block_generations", 3)))
    return offspring * generations


def available_actions(candidate: Design, setting: Any) -> tuple[str, ...]:
    parameter_patience = 2 * int(
        get_flex(setting, "parameter_cma_block_generations", 3)
    )
    state = candidate.design_aux or {}
    if int(state.get("cma_parameter_stagnation", 0)) >= parameter_patience:
        return (STRUCTURE,)
    if supports_parameter_action(candidate, setting):
        return (STRUCTURE, PARAMETER)
    return (STRUCTURE,)


def _candidate(
    operators: Any,
    parameters: Any,
    problems: Any,
    setting: Any,
    state: dict[str, Any],
    configuration: Any = None,
) -> Design:
    return Design.from_genotype(
        deepcopy(operators),
        deepcopy(parameters),
        problems,
        setting,
        design_aux=deepcopy(state),
        configuration=deepcopy(configuration),
    )


def _algorithm_cma_spec(candidate: Design, setting: Any) -> list[tuple[str, Any]]:
    """Return ordered integer and continuous algorithm-level variables."""
    spec: list[tuple[str, Any]] = []
    population = population_size_space(setting)
    offspring = offspring_size_space(setting)
    if len(population) > 1:
        spec.append(("population_size", population))
    if len(offspring) > 1:
        spec.append(("offspring_size", offspring))
    (crossover_active, mutation_active) = configuration_activity(candidate)
    if crossover_active:
        spec.append(("crossover_rate", None))
    if mutation_active:
        spec.append(("mutation_rate", None))
    boundary_domain = boundary_handling_space(setting)
    if len(boundary_domain) > 1:
        spec.append(("boundary_handling", boundary_domain))
    return spec


def _mixed_cma_spec(candidate: Design, setting: Any) -> list[tuple[str, Any]]:
    """Return every active component and algorithm-level hyperparameter."""
    spec: list[tuple[str, Any]] = [
        ("component", locus) for locus in active_parameter_loci(candidate, setting)
    ]
    spec.extend(
        (("configuration", item) for item in _algorithm_cma_spec(candidate, setting))
    )
    return spec


def _uses_ssga(candidate: Design) -> bool:
    pathways_groups = getattr(candidate, "operator_pheno", None)
    pathways = pathways_groups[0] if pathways_groups else []
    return any(
        (
            str(getattr(path, "update", "")).startswith("update_ssga_")
            for path in pathways
        )
    )


def _mixed_cma_coordinates(
    spec: Sequence[tuple[str, Any]], candidate: Design, setting: Any
) -> tuple[tuple[Any, ...], ...]:
    """Return stable semantic identities for the latent CMA coordinates."""
    component_names = tuple((str(name) for name in get_flex(setting, "all_op", ())))
    coordinates: list[tuple[Any, ...]] = []
    for source, payload in spec:
        if source == "component":
            (kind, parameter_index, value_index, lower, upper) = payload
            parameter_index = int(parameter_index)
            component = (
                component_names[parameter_index]
                if 0 <= parameter_index < len(component_names)
                else f"operator-{parameter_index + 1}"
            )
            coordinates.append(
                (
                    "component",
                    component,
                    int(value_index),
                    str(kind),
                    float(lower),
                    float(upper),
                )
            )
            continue
        (name, domain) = payload
        constraint = (
            "offspring_le_population"
            if name == "offspring_size" and _uses_ssga(candidate)
            else "unconstrained"
        )
        coordinates.append(
            (
                "configuration",
                str(name),
                None if domain is None else tuple(domain),
                constraint,
            )
        )
    if len(set(coordinates)) != len(coordinates):
        raise ValueError("Mixed CMA parameter coordinates must be unique.")
    return tuple(coordinates)


def _carry_cma_state_across_structure(
    parent_state: Any, mutation_state: Any
) -> dict[str, Any]:
    """Carry only reusable CMA geometry into a structural child."""
    result = dict(mutation_state) if isinstance(mutation_state, dict) else {}
    if not isinstance(parent_state, dict):
        return result
    for key in _CMA_INHERITABLE_STATE_KEYS:
        if key in parent_state:
            result[key] = deepcopy(parent_state[key])
    return result


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _logit(value: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError("A CMA rate coordinate must be finite.")
    if number <= 0.0:
        bounded = 0.02
    elif number >= 1.0:
        bounded = 0.98
    else:
        bounded = number
    return math.log(bounded / (1.0 - bounded))


def _discrete_index(coordinate: float, count: int) -> int:
    """Decode an unbounded latent coordinate through equal-width bins."""
    if count <= 1:
        return 0
    unit = _sigmoid(float(coordinate))
    return int(np.clip(math.floor(unit * count), 0, count - 1))


def _discrete_coordinate(index: int, count: int) -> float:
    if count <= 1:
        return 0.0
    selected = int(index)
    if not 0 <= selected < count:
        raise ValueError(f"Discrete index {selected} is outside a domain of {count}.")
    unit = (selected + 0.5) / count
    return math.log(unit / (1.0 - unit))


def _configuration_domain(
    name: str, domain: Sequence[Any], configuration: dict[str, Any], candidate: Design
) -> tuple[Any, ...]:
    values = tuple(domain)
    if name == "offspring_size" and _uses_ssga(candidate):
        population = int(configuration["population_size"])
        values = tuple((int(value) for value in values if int(value) <= population))
    if not values:
        raise ValueError(f"No legal values remain for {name}.")
    return values


def _encode_mixed_parameters(
    parameters: Any,
    configuration: dict[str, Any],
    spec: Sequence[tuple[str, Any]],
    candidate: Design,
) -> np.ndarray:
    encoded: list[float] = []
    for source, payload in spec:
        if source == "component":
            (_, parameter_index, value_index, lower, upper) = payload
            value = float(np.asarray(parameters[parameter_index][0])[value_index])
            if payload[0] == "integer":
                count = int(round(upper - lower)) + 1
                encoded.append(_discrete_coordinate(int(round(value - lower)), count))
            else:
                encoded.append(_logit((value - lower) / (upper - lower)))
            continue
        (name, domain) = payload
        value = configuration[name]
        if domain is None:
            encoded.append(_logit(float(value)))
            continue
        values = _configuration_domain(name, domain, configuration, candidate)
        index = values.index(value)
        encoded.append(_discrete_coordinate(index, len(values)))
    return np.asarray(encoded, dtype=float)


def _decode_mixed_parameters(
    vector: np.ndarray,
    base_parameters: Any,
    base_configuration: dict[str, Any],
    spec: Sequence[tuple[str, Any]],
    candidate: Design,
) -> tuple[Any, dict[str, Any]]:
    parameters = deepcopy(base_parameters)
    configuration = dict(base_configuration)
    for coordinate, (source, payload) in zip(vector, spec):
        if source == "component":
            (kind, parameter_index, value_index, lower, upper) = payload
            values = np.asarray(parameters[parameter_index][0], dtype=float).copy()
            if kind == "integer":
                count = int(round(upper - lower)) + 1
                decoded = lower + _discrete_index(float(coordinate), count)
            else:
                decoded = lower + _sigmoid(float(coordinate)) * (upper - lower)
            if kind == "integer":
                decoded = int(np.rint(decoded))
            values[value_index] = decoded
            parameters[parameter_index][0] = values
            continue
        (name, domain) = payload
        if domain is None:
            configuration[name] = _sigmoid(float(coordinate))
            continue
        values = _configuration_domain(name, domain, configuration, candidate)
        index = _discrete_index(float(coordinate), len(values))
        configuration[name] = values[index]
    return (parameters, configuration)


def _cma_offspring_count(candidate: Design, setting: Any) -> int:
    del candidate
    return int(get_flex(setting, "parameter_cma_offspring", 5))


def _symmetric_eigen_reconstruct(
    vectors: np.ndarray, diagonal: np.ndarray
) -> np.ndarray:
    """Reconstruct V diag(d) V^T without an overflow-prone BLAS matmul."""
    dimension = int(vectors.shape[0])
    result = np.zeros((dimension, dimension), dtype=float)
    for value, vector in zip(np.asarray(diagonal, dtype=float), vectors.T):
        result += float(value) * np.outer(vector, vector)
    return result


def _regularize_cma_covariance(matrix: Any, dimension: int) -> np.ndarray:
    covariance = np.asarray(matrix, dtype=float)
    if covariance.shape != (dimension, dimension):
        covariance = np.eye(dimension)
    covariance = np.nan_to_num(
        covariance, nan=0.0, posinf=1000000000000.0, neginf=-1000000000000.0
    )
    covariance = np.clip(covariance, -1000000000000.0, 1000000000000.0)
    covariance = 0.5 * (covariance + covariance.T)
    (values, vectors) = np.linalg.eigh(covariance)
    values = np.clip(
        np.nan_to_num(values, nan=1.0, posinf=1000000000000.0, neginf=1e-12),
        1e-12,
        1000000000000.0,
    )
    if not np.all(np.isfinite(vectors)):
        vectors = np.eye(dimension)
    regularized = _symmetric_eigen_reconstruct(vectors, values)
    regularized = np.nan_to_num(
        regularized, nan=0.0, posinf=1000000000000.0, neginf=-1000000000000.0
    )
    regularized = np.clip(regularized, -1000000000000.0, 1000000000000.0)
    return 0.5 * (regularized + regularized.T)


def _freeze_coordinate(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return tuple((_freeze_coordinate(item) for item in value.tolist()))
    if isinstance(value, (list, tuple)):
        return tuple((_freeze_coordinate(item) for item in value))
    return value


def _transferable_cma_snapshot(state: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and copy the CMA fields that are safe to remap."""
    raw_coordinates = state.get("cma_parameter_coordinates")
    if not isinstance(raw_coordinates, (list, tuple)) or not raw_coordinates:
        return None
    coordinates = tuple((_freeze_coordinate(value) for value in raw_coordinates))
    if len(set(coordinates)) != len(coordinates):
        return None
    dimension = len(coordinates)
    try:
        mean = np.asarray(state["cma_parameter_mean"], dtype=float)
        covariance = np.asarray(state["cma_parameter_covariance"], dtype=float)
        pc = np.asarray(state["cma_parameter_pc"], dtype=float)
        ps = np.asarray(state["cma_parameter_ps"], dtype=float)
        sigma = float(state["cma_parameter_sigma"])
        generation = int(state["cma_parameter_generation"])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (
        mean.shape != (dimension,)
        or covariance.shape != (dimension, dimension)
        or pc.shape != (dimension,)
        or (ps.shape != (dimension,))
        or (not np.all(np.isfinite(mean)))
        or (not np.all(np.isfinite(pc)))
        or (not np.all(np.isfinite(ps)))
        or (not np.isfinite(sigma))
        or (sigma <= 0.0)
        or (generation < 0)
    ):
        return None
    return {
        "coordinates": coordinates,
        "mean": mean.copy(),
        "covariance": _regularize_cma_covariance(covariance, dimension),
        "sigma": float(np.clip(sigma, 0.001, 5.0)),
        "pc": pc.copy(),
        "ps": ps.copy(),
        "generation": generation,
    }


def _reconcile_cma_state(
    state: dict[str, Any],
    *,
    signature: tuple[Any, ...],
    coordinates: tuple[tuple[Any, ...], ...],
    initial_mean: np.ndarray,
    initial_sigma: float,
    current_cost: float,
) -> dict[str, Any]:
    """Initialize CMA or remap learned state after a graph-space change."""
    dimension = len(coordinates)
    encoded = np.asarray(initial_mean, dtype=float)
    if encoded.shape != (dimension,) or not np.all(np.isfinite(encoded)):
        raise ValueError("The initial CMA mean does not match its coordinate space.")
    snapshot = _transferable_cma_snapshot(state)
    if (
        state.get("cma_parameter_signature") == signature
        and snapshot is not None
        and (snapshot["coordinates"] == coordinates)
    ):
        return state
    for key in tuple(state):
        if str(key).startswith(("cma_algorithm_", "cma_parameter_", "ils_")):
            state.pop(key, None)
    mean = encoded.copy()
    covariance = np.eye(dimension)
    pc = np.zeros(dimension)
    ps = np.zeros(dimension)
    sigma = float(initial_sigma)
    generation = 0
    old_dimension = 0
    shared: list[tuple[int, int]] = []
    if snapshot is not None:
        old_coordinates = snapshot["coordinates"]
        old_dimension = len(old_coordinates)
        old_indices = {
            coordinate: index for (index, coordinate) in enumerate(old_coordinates)
        }
        shared = [
            (new_index, old_indices[coordinate])
            for (new_index, coordinate) in enumerate(coordinates)
            if coordinate in old_indices
        ]
        if shared:
            for new_index, old_index in shared:
                mean[new_index] = snapshot["mean"][old_index]
                pc[new_index] = snapshot["pc"][old_index]
                ps[new_index] = snapshot["ps"][old_index]
            for new_row, old_row in shared:
                for new_column, old_column in shared:
                    covariance[new_row, new_column] = snapshot["covariance"][
                        old_row, old_column
                    ]
            covariance = _regularize_cma_covariance(covariance, dimension)
            sigma = snapshot["sigma"]
            generation = snapshot["generation"]
    inherited = len(shared)
    state["cma_parameter_signature"] = signature
    state["cma_parameter_coordinates"] = coordinates
    state["cma_parameter_mean"] = mean
    state["cma_parameter_covariance"] = covariance
    state["cma_parameter_sigma"] = sigma
    state["cma_parameter_pc"] = pc
    state["cma_parameter_ps"] = ps
    state["cma_parameter_generation"] = generation
    state["cma_parameter_best_cost"] = float(current_cost)
    state["cma_parameter_stagnation"] = 0
    state["cma_parameter_transfer_mode"] = "transferred" if inherited else "initialized"
    state["cma_parameter_inherited_coordinates"] = inherited
    state["cma_parameter_initialized_coordinates"] = dimension - inherited
    state["cma_parameter_dropped_coordinates"] = old_dimension - inherited
    return state


def _parameter_cma_candidates(
    parent: Design,
    problems: Any,
    setting: Any,
    evaluate: Callable[[Sequence[Design]], None],
    cost: Callable[[Design], float],
    state: dict[str, Any],
    seen_representations: set[str] | None,
    *,
    maximum: int,
    maximum_attempts: int,
) -> tuple[Design, list[Design], dict[str, Any]]:
    """Run one generation of CMA-ES over all active hyperparameters.

    Component and algorithm-level integer variables remain continuous in the
    latent CMA vector and are rounded only during decoding.  Component
    continuous parameters, integer component parameters, population size,
    offspring size, crossover rate, and mutation rate therefore share one
    stateful covariance model.
    """
    spec = _mixed_cma_spec(parent, setting)
    if not spec or maximum <= 0:
        return (parent, [], state)
    dimension = len(spec)
    parameters = deepcopy(parent.parameter)
    configuration = copy_configuration(parent, setting)
    coordinates = _mixed_cma_coordinates(spec, parent, setting)
    signature = (
        representation_key(parent, include_continuous_parameters=False),
        coordinates,
    )
    state = _reconcile_cma_state(
        state,
        signature=signature,
        coordinates=coordinates,
        initial_mean=_encode_mixed_parameters(parameters, configuration, spec, parent),
        initial_sigma=float(get_flex(setting, "parameter_cma_initial_sigma", 1.0)),
        current_cost=float(cost(parent)),
    )
    mean = np.asarray(state["cma_parameter_mean"], dtype=float)
    covariance = _regularize_cma_covariance(
        state["cma_parameter_covariance"], dimension
    )
    sigma = float(np.clip(state.get("cma_parameter_sigma", 1.0), 0.001, 5.0))
    (eigenvalues, eigenvectors) = np.linalg.eigh(covariance)
    eigenvalues = np.clip(
        np.nan_to_num(eigenvalues, nan=1.0, posinf=1000000000000.0, neginf=1e-12),
        1e-12,
        1000000000000.0,
    )
    if not np.all(np.isfinite(eigenvectors)):
        eigenvectors = np.eye(dimension)
    square_root = _symmetric_eigen_reconstruct(eigenvectors, np.sqrt(eigenvalues))
    rng = ensure_rng(setting)
    evaluated: list[Design] = []
    encoded: list[np.ndarray] = []
    local_keys: set[str] = set()
    attempt_limit = max(maximum * 1000, maximum * max(1, maximum_attempts) * 20)
    attempts = 0
    structure_rejections = 0
    duplicate_rejections = 0
    while len(evaluated) < maximum and attempts < attempt_limit:
        attempts += 1
        latent = mean + sigma * (square_root @ rng.standard_normal(dimension))
        (trial_parameters, trial_configuration) = _decode_mixed_parameters(
            latent, parameters, configuration, spec, parent
        )
        if not configuration_is_valid(parent, trial_configuration):
            continue
        child = _candidate(
            parent.operator,
            trial_parameters,
            problems,
            setting,
            state,
            trial_configuration,
        )
        if (
            representation_key(child, include_continuous_parameters=False)
            != signature[0]
        ):
            structure_rejections += 1
            continue
        key = representation_key(child)
        if key in local_keys or (
            seen_representations is not None and key in seen_representations
        ):
            duplicate_rejections += 1
            continue
        local_keys.add(key)
        evaluated.append(child)
        encoded.append(latent)
    sampling_exhausted = len(evaluated) < maximum
    state["cma_parameter_last_sampling_attempts"] = attempts
    state["cma_parameter_last_structure_rejections"] = structure_rejections
    state["cma_parameter_last_duplicate_rejections"] = duplicate_rejections
    state["cma_parameter_sampling_exhausted"] = sampling_exhausted
    if not evaluated:
        raise ParameterSamplingExhausted(
            f"Mixed-integer CMA-ES could not produce any unique, graph-preserving hyperparameter configuration after {attempts} internal draws ({structure_rejections} changed structure; {duplicate_rejections} were duplicates)."
        )
    if seen_representations is not None:
        seen_representations.update(local_keys)
    selected_count = max(1, len(evaluated) // 2)
    racing = str(get_flex(setting, "evaluate", "exact")).lower() == "racing"
    if racing:
        for candidate in evaluated:
            candidate.metadata = dict(getattr(candidate, "metadata", {}) or {})
            candidate.metadata["racing_keep"] = selected_count
            candidate.metadata["racing_role"] = "parameter"
    evaluate(evaluated)
    old_mean = mean.copy()
    eligible = [
        index
        for (index, candidate) in enumerate(evaluated)
        if not getattr(candidate, "metadata", {})
        .get("racing", {})
        .get("eliminated", False)
    ]
    adaptation_candidates = [evaluated[index] for index in eligible]
    adaptation_points = [encoded[index] for index in eligible]
    ps = np.asarray(state.get("cma_parameter_ps", np.zeros(dimension)), dtype=float)
    pc = np.asarray(state.get("cma_parameter_pc", np.zeros(dimension)), dtype=float)
    generation = int(state.get("cma_parameter_generation", 0)) + 1
    adaptation_costs = np.asarray(
        [cost(candidate) for candidate in adaptation_candidates], dtype=float
    )
    no_ranking_signal = bool(
        adaptation_costs.size
        and np.all(
            np.isclose(adaptation_costs, adaptation_costs[0], rtol=0.0, atol=1e-15)
        )
    )
    ranking_signal = not no_ranking_signal or len(eligible) < len(evaluated)
    if (
        len(eligible) >= selected_count
        and np.count_nonzero(np.isfinite(adaptation_costs)) >= selected_count
        and ranking_signal
    ):
        order = np.argsort(adaptation_costs, kind="stable")
        selected = order[:selected_count]
        raw_weights = np.log(selected_count + 0.5) - np.log(
            np.arange(1, selected_count + 1)
        )
        weights = raw_weights / np.sum(raw_weights)
        effective = 1.0 / np.sum(weights**2)
        points = np.asarray([adaptation_points[int(index)] for index in selected])
        mean = np.sum(weights[:, None] * points, axis=0)
        displacement = (mean - old_mean) / max(sigma, 1e-12)
        cc = (4.0 + effective / dimension) / (
            dimension + 4.0 + 2.0 * effective / dimension
        )
        cs = (effective + 2.0) / (dimension + effective + 5.0)
        c1 = 2.0 / ((dimension + 1.3) ** 2 + effective)
        cmu = min(
            1.0 - c1,
            2.0
            * (effective - 2.0 + 1.0 / effective)
            / ((dimension + 2.0) ** 2 + effective),
        )
        damping = (
            1.0
            + 2.0 * max(0.0, np.sqrt((effective - 1.0) / (dimension + 1.0)) - 1.0)
            + cs
        )
        inverse_root = _symmetric_eigen_reconstruct(
            eigenvectors, 1.0 / np.sqrt(eigenvalues)
        )
        ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * effective) * (
            inverse_root @ displacement
        )
        chi = np.sqrt(dimension) * (
            1.0 - 1.0 / (4.0 * dimension) + 1.0 / (21.0 * dimension**2)
        )
        path_scale = np.sqrt(max(1e-12, 1.0 - (1.0 - cs) ** (2 * generation)))
        hsig = float(
            np.linalg.norm(ps) / path_scale < (1.4 + 2.0 / (dimension + 1)) * chi
        )
        pc = (1.0 - cc) * pc + hsig * np.sqrt(
            cc * (2.0 - cc) * effective
        ) * displacement
        normalized = (points - old_mean) / max(sigma, 1e-12)
        rank_mu = np.zeros_like(covariance)
        for weight, vector in zip(weights, normalized):
            rank_mu += float(weight) * np.outer(vector, vector)
        covariance = (
            (1.0 - c1 - cmu) * covariance
            + c1 * (np.outer(pc, pc) + (1.0 - hsig) * cc * (2.0 - cc) * covariance)
            + cmu * rank_mu
        )
        covariance = _regularize_cma_covariance(covariance, dimension)
        exponent = float(cs / damping * (np.linalg.norm(ps) / chi - 1.0))
        sigma *= float(np.exp(np.clip(exponent, -20.0, 20.0)))
    state["cma_parameter_mean"] = mean
    state["cma_parameter_covariance"] = covariance
    state["cma_parameter_sigma"] = float(np.clip(sigma, 0.001, 5.0))
    state["cma_parameter_pc"] = pc
    state["cma_parameter_ps"] = ps
    state["cma_parameter_generation"] = generation
    state["cma_parameter_last_samples"] = len(evaluated)
    state["cma_parameter_last_generation_size"] = len(adaptation_candidates)
    state["cma_parameter_last_ranking_signal"] = ranking_signal
    state["cma_parameter_last_mu"] = selected_count
    state["cma_parameter_sampling_exhausted"] = sampling_exhausted
    representative = min([parent, *evaluated], key=cost)
    previous_best = float(state.get("cma_parameter_best_cost", cost(parent)))
    current_best = float(cost(representative))
    if current_best < previous_best and (
        not np.isclose(current_best, previous_best, rtol=0.0, atol=1e-15)
    ):
        state["cma_parameter_stagnation"] = 0
    else:
        state["cma_parameter_stagnation"] = (
            int(state.get("cma_parameter_stagnation", 0)) + 1
        )
    state["cma_parameter_best_cost"] = min(previous_best, current_best)
    return (representative, evaluated, state)


def _run_parameter_cma_block(
    parent: Design,
    problems: Any,
    setting: Any,
    evaluate: Callable[[Sequence[Design]], None],
    cost: Callable[[Design], float],
    state: dict[str, Any],
    seen_representations: set[str] | None,
    *,
    generations: int,
    remaining_evaluations: int,
    maximum_attempts: int,
) -> tuple[Design, list[Design], dict[str, Any]]:
    incumbent = parent
    evaluated: list[Design] = []
    for _ in range(max(1, int(generations))):
        if remaining_evaluations <= 0:
            break
        requested_offspring = _cma_offspring_count(incumbent, setting)
        offspring = min(remaining_evaluations, max(0, requested_offspring))
        if offspring <= 0:
            break
        try:
            (incumbent, children, state) = _parameter_cma_candidates(
                incumbent,
                problems,
                setting,
                evaluate,
                cost,
                state,
                seen_representations,
                maximum=offspring,
                maximum_attempts=maximum_attempts,
            )
        except ParameterSamplingExhausted:
            if not evaluated:
                raise
            state["cma_parameter_sampling_exhausted"] = True
            break
        evaluated.extend(children)
        remaining_evaluations -= len(children)
        incumbent.design_aux = deepcopy(state)
        for child in children:
            child.design_aux = deepcopy(state)
        if bool(state.get("cma_parameter_sampling_exhausted", False)):
            break
    return (incumbent, evaluated, state)


def parameter_action(
    parent: Design,
    problems: Any,
    setting: Any,
    evaluate: Callable[[Sequence[Design]], None],
    cost: Callable[[Design], float],
    *,
    seen_representations: set[str] | None = None,
    maximum_attempts: int = 1,
    remaining_evaluations: int | None = None,
) -> ActionResult:
    """Run a persistent CMA-ES block over every active hyperparameter."""
    if not _mixed_cma_spec(parent, setting):
        raise RuntimeError("CMA-ES requires at least one active hyperparameter.")
    generations = int(get_flex(setting, "parameter_cma_block_generations", 3))
    required = _cma_offspring_count(parent, setting) * max(1, generations)
    remaining = (
        required if remaining_evaluations is None else int(remaining_evaluations)
    )
    if remaining <= 0:
        raise RuntimeError("No candidate budget remains for a parameter action.")
    remaining = min(remaining, required)
    (representative, evaluated, state) = _run_parameter_cma_block(
        parent,
        problems,
        setting,
        evaluate,
        cost,
        dict(parent.design_aux or {}),
        seen_representations,
        generations=generations,
        remaining_evaluations=remaining,
        maximum_attempts=maximum_attempts,
    )
    if not evaluated:
        raise RuntimeError("CMA-ES generation or stagnation limit was reached.")
    state["last_move_kind"] = PARAMETER
    state["last_parameter_kind"] = "mixed_integer_cma_es"
    state["last_parameter_locus"] = "all_active_hyperparameters"
    state["last_parameter_evaluations"] = len(evaluated)
    representative.design_aux = deepcopy(state)
    for child in evaluated:
        child.design_aux = deepcopy(state)
    return ActionResult(
        PARAMETER,
        representative,
        evaluated,
        len(evaluated),
        candidate_actions=(PARAMETER,) * len(evaluated),
    )


def tune_structure_candidates(
    bases: Sequence[Design],
    problems: Any,
    setting: Any,
    evaluate: Callable[[Sequence[Design]], None],
    cost: Callable[[Design], float],
    *,
    seen_representations: set[str],
    remaining_evaluations: int,
) -> ActionResult:
    """Evaluate a novel graph batch, then CMA-tune only its best baseline."""
    proposals = list(bases)
    if not proposals:
        raise RuntimeError("A structure action requires at least one proposal.")
    allowed = int(remaining_evaluations)
    stream_partial = (
        str(get_flex(setting, "graph_semantics", "legacy_pathway_v1")).lower()
        == "stream_graph_v2"
    )
    bootstrap_generations = max(
        1, int(get_flex(setting, "post_structure_cma_generations", 1))
    )
    required = len(proposals)
    if not stream_partial:
        required += _cma_offspring_count(proposals[0], setting) * bootstrap_generations
    if allowed < required:
        raise RuntimeError(
            f"A structure macro-action requires {required} candidates, but only {allowed} remain."
        )
    proposal_keys = [representation_key(base) for base in proposals]
    if len(set(proposal_keys)) != len(proposal_keys) or any(
        (key in seen_representations for key in proposal_keys)
    ):
        raise RuntimeError(
            "A structure batch contains a previously evaluated algorithm."
        )
    seen_representations.update(proposal_keys)
    if str(get_flex(setting, "evaluate", "exact")).lower() == "racing":
        for candidate in proposals:
            candidate.metadata = dict(getattr(candidate, "metadata", {}) or {})
            candidate.metadata["racing_keep"] = 1
            candidate.metadata["racing_role"] = "structure"
    evaluate(proposals)
    evaluated = list(proposals)
    selected_baseline = min(proposals, key=cost)
    cma_spec = _mixed_cma_spec(selected_baseline, setting)
    cma_state = dict(selected_baseline.design_aux or {})
    cma_budget = allowed - len(proposals)
    cma_required = (
        _cma_offspring_count(selected_baseline, setting) * bootstrap_generations
    )
    bootstrap_skip_reason = None
    if not cma_spec:
        if not stream_partial:
            raise RuntimeError(
                "A structure action requires tunable hyperparameters for its mandatory CMA bootstrap."
            )
        cma_state["cma_bootstrap_skipped_no_tunable_parameters"] = True
        bootstrap_skip_reason = "no_tunable_parameters"
        incumbent = selected_baseline
        cma_children = []
    elif cma_budget < cma_required:
        if not stream_partial:
            raise RuntimeError(
                "A structure action requires enough budget for its mandatory CMA bootstrap."
            )
        cma_state["cma_bootstrap_skipped_insufficient_budget"] = True
        bootstrap_skip_reason = "insufficient_budget"
        incumbent = selected_baseline
        cma_children = []
    else:
        try:
            (incumbent, cma_children, cma_state) = _run_parameter_cma_block(
                selected_baseline,
                problems,
                setting,
                evaluate,
                cost,
                cma_state,
                seen_representations,
                generations=bootstrap_generations,
                remaining_evaluations=cma_budget,
                maximum_attempts=20,
            )
        except ParameterSamplingExhausted:
            cma_state["cma_parameter_sampling_exhausted"] = True
            bootstrap_skip_reason = "parameter_sampling_exhausted"
            incumbent = selected_baseline
            cma_children = []
    cma_evaluations = len(cma_children)
    evaluated.extend(cma_children)
    incumbent.design_aux = deepcopy(cma_state)
    for child in cma_children:
        child.design_aux = deepcopy(cma_state)
    state = dict(incumbent.design_aux or {})
    state["last_move_kind"] = STRUCTURE
    state["post_structure_evaluations"] = len(evaluated)
    state["post_structure_cma_evaluations"] = cma_evaluations
    state["structure_proposal_count"] = len(proposals)
    state["post_structure_parameter_count"] = len(cma_spec)
    state.pop("conditional_operator_indices", None)
    incumbent.design_aux = state
    return ActionResult(
        STRUCTURE,
        incumbent,
        evaluated,
        len(evaluated),
        candidate_actions=(STRUCTURE,) * len(proposals)
        + (PARAMETER,) * cma_evaluations,
        structure_baseline=selected_baseline,
        structure_candidate_evaluations=len(proposals),
        post_structure_parameter_evaluations=cma_evaluations,
        bootstrap_skip_reason=bootstrap_skip_reason,
    )


def tune_structure_candidate(
    base: Design,
    problems: Any,
    setting: Any,
    evaluate: Callable[[Sequence[Design]], None],
    cost: Callable[[Design], float],
    *,
    seen_representations: set[str],
    remaining_evaluations: int,
) -> ActionResult:
    """Compatibility wrapper for one graph plus its CMA bootstrap."""
    return tune_structure_candidates(
        [base],
        problems,
        setting,
        evaluate,
        cost,
        seen_representations=seen_representations,
        remaining_evaluations=remaining_evaluations,
    )


def structure_action(
    parent: Design,
    problems: Any,
    setting: Any,
    evaluate: Callable[[Sequence[Design]], None],
    cost: Callable[[Design], float],
    *,
    seen_structures: set[str],
    seen_representations: set[str],
    maximum_attempts: int,
    remaining_evaluations: int,
) -> ActionResult:
    """Evaluate three novel graph-ILS proposals and CMA-tune their winner."""
    del maximum_attempts
    configured = max(1, int(get_flex(setting, "structure_candidates_per_action", 3)))
    stream_partial = (
        str(get_flex(setting, "graph_semantics", "legacy_pathway_v1")).lower()
        == "stream_graph_v2"
    )
    requested = (
        min(configured, max(0, int(remaining_evaluations)))
        if stream_partial
        else configured
    )
    if requested <= 0:
        raise RuntimeError("A structure action has no remaining candidate budget.")
    proposals: list[Design] = []
    proposal_structure_keys: set[str] = set()
    proposal_representation_keys: set[str] = set()
    for _ in range(requested):
        proposal: Design | None = None
        for _ in range(STRUCTURE_NOVELTY_ATTEMPTS):
            (operators, parameters, state) = mutate_structure(
                parent, setting, deepcopy(parent.design_aux)
            )
            state = _carry_cma_state_across_structure(parent.design_aux, state)
            trial = _candidate(operators, parameters, problems, setting, state)
            configuration = copy_configuration(parent, setting)
            if not configuration_is_valid(trial, configuration):
                continue
            set_configuration(trial, configuration)
            structure_key = representation_key(
                trial, include_continuous_parameters=False
            )
            if (
                structure_key in seen_structures
                or structure_key in proposal_structure_keys
            ):
                continue
            executable_key = representation_key(trial)
            if (
                executable_key in seen_representations
                or executable_key in proposal_representation_keys
            ):
                continue
            proposal_structure_keys.add(structure_key)
            proposal_representation_keys.add(executable_key)
            proposal = trial
            break
        if proposal is None:
            raise StructureNoveltyExhausted(STRUCTURE_NOVELTY_ATTEMPTS)
        proposals.append(proposal)
    seen_structures.update(proposal_structure_keys)
    return tune_structure_candidates(
        proposals,
        problems,
        setting,
        evaluate,
        cost,
        seen_representations=seen_representations,
        remaining_evaluations=remaining_evaluations,
    )


def execute_action(
    action: str,
    parent: Design,
    problems: Any,
    setting: Any,
    evaluate: Callable[[Sequence[Design]], None],
    cost: Callable[[Design], float],
    *,
    seen_structures: set[str],
    seen_representations: set[str],
    maximum_attempts: int,
    remaining_evaluations: int,
) -> ActionResult:
    if action == STRUCTURE:
        return structure_action(
            parent,
            problems,
            setting,
            evaluate,
            cost,
            seen_structures=seen_structures,
            seen_representations=seen_representations,
            maximum_attempts=maximum_attempts,
            remaining_evaluations=remaining_evaluations,
        )
    if action == PARAMETER:
        return parameter_action(
            parent,
            problems,
            setting,
            evaluate,
            cost,
            seen_representations=seen_representations,
            maximum_attempts=maximum_attempts,
            remaining_evaluations=remaining_evaluations,
        )
    raise ValueError(f"Unknown search action: {action}")


__all__ = [
    "ActionCredit",
    "ActionResult",
    "ParameterSamplingExhausted",
    "StructureNoveltyExhausted",
    "STRUCTURE_NOVELTY_ATTEMPTS",
    "ActionSelector",
    "PARAMETER",
    "SEARCH_ACTIONS",
    "STRUCTURE",
    "available_actions",
    "choose_action",
    "controller_action_credit",
    "execute_action",
    "parameter_action",
    "parameter_action_evaluation_cost",
    "semantic_action_credits",
    "structure_action_evaluation_cost",
    "supports_parameter_action",
    "tune_structure_candidate",
    "tune_structure_candidates",
]

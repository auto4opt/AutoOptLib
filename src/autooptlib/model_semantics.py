"""Versioned graph semantics and contextual pre-execution validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

MODEL_VERSION = 1
BATCH_SEMANTICS = "population_batch_v1"
EVENT_SEMANTICS = "event_v1"


@dataclass(frozen=True)
class ExecutionSpec:
    execution: str
    version: int = MODEL_VERSION
    graph_grammar: str = "stream_graph_v2"
    state_scope: str = "run/pathway/stage"
    evaluation: str = "ordered_stage_boundary"
    random: str = "one_authoritative_generator_per_run"
    update: str = "ordered_shared_sink_groups"

    def __post_init__(self):
        if (
            self.execution not in {BATCH_SEMANTICS, EVENT_SEMANTICS}
            or self.version != MODEL_VERSION
        ):
            raise ValueError("Unsupported graph execution contract")

    def to_dict(self):
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, record):
        result = cls(**record)
        if result.to_dict() != cls(result.execution).to_dict():
            raise ValueError("Execution rules differ from the supported contract")
        return result


class GraphValidationError(ValueError):
    """An error at a stable path/stage/role location."""

    def __init__(self, location, message):
        self.location = location
        super().__init__(f"{location}: {message}")


def execution_spec(backend="python", stream_backend=None):
    stream = stream_backend or ("batch" if backend == "native" else "python")
    if backend not in {"python", "native"} or stream not in {
        "python",
        "direct",
        "graph",
        "batch",
    }:
        raise ValueError("Unknown execution backend")
    if stream == "batch" and backend != "native":
        raise ValueError("Batch execution requires the native component backend")
    return ExecutionSpec(BATCH_SEMANTICS if stream == "batch" else EVENT_SEMANTICS)


def validate_design(
    design,
    problem: Any = None,
    *,
    backend="python",
    stream_backend=None,
    require_installed=False,
):
    from .components.contracts import get_component_contract
    from .utils.design._population import configuration_is_valid
    from .utils.design._stream_graph import validate_phenotype

    spec = execution_spec(backend, stream_backend)
    declared = getattr(design, "program_contract", None)
    if declared and declared.get("execution") is not None:
        expected = ExecutionSpec.from_dict(declared)
        if expected != spec:
            raise GraphValidationError(
                "execution", f"requires {expected.execution}, got {spec.execution}"
            )
    paths, parameters = design.operator_pheno[0], design.parameter_pheno[0]
    validate_phenotype(paths, parameters)
    if design.configuration is not None and not configuration_is_valid(
        design, design.configuration
    ):
        raise GraphValidationError(
            "configuration", "invalid population/offspring constraints"
        )
    problems = problem if isinstance(problem, (list, tuple)) else [problem]
    for instance, record in enumerate(problems):
        domain = dimension = None
        if record is not None:
            value = getattr(record, "type", None)
            domain = str(
                value[0] if isinstance(value, (list, tuple)) else value
            ).lower()
            bounds = np.asarray(getattr(record, "bound", None))
            if bounds.ndim != 2 or bounds.shape[0] != 2 or not bounds.shape[1]:
                raise GraphValidationError(
                    f"problem[{instance}]", "requires bounds of shape (2, D)"
                )
            if (
                bounds.dtype.kind not in "fiu"
                or not np.all(np.isfinite(bounds))
                or np.any(bounds[0] > bounds[1])
            ):
                raise GraphValidationError(
                    f"problem[{instance}]",
                    "bounds must be finite ordered numeric intervals",
                )
            dimension = bounds.shape[1]
        calls = []
        for pi, (path, parameter) in enumerate(zip(paths, parameters)):
            for si, (stage, values) in enumerate(zip(path.stages, parameter.stages)):
                if stage.choose:
                    calls.append(
                        (
                            f"pathways[{pi}].stages[{si}].choose",
                            stage.choose,
                            values.choose,
                        )
                    )
                calls.append(
                    (
                        f"pathways[{pi}].stages[{si}].search",
                        stage.search.primary,
                        values.search.primary,
                    )
                )
            calls.append((f"pathways[{pi}].update", path.update, parameter.update))
        calls.extend(
            (f"archive[{i}]", name, None) for i, name in enumerate(paths[0].archive)
        )
        for location, name, values in calls:
            try:
                contract = get_component_contract(name, record)
                contract.validate_parameters(values, dimension)
                if backend not in contract.backends:
                    raise ValueError(f"{name} has no {backend} implementation")
                if domain is not None:
                    if domain not in contract.problem_types:
                        raise ValueError(f"{name} does not support {domain} problems")
                    if dimension < contract.min_dimension:
                        raise ValueError(
                            f"{name} requires dimension >= {contract.min_dimension}"
                        )
                    if contract.binary and (
                        np.any(bounds[0] != 0) or np.any(bounds[1] != 1)
                    ):
                        raise ValueError(f"{name} requires binary [0, 1] bounds")
                    if values is not None:
                        for parameter, value in zip(contract.parameters, values):
                            if parameter.name in {
                                "points",
                                "selected_positions",
                                "reset_count",
                            }:
                                maximum = (
                                    dimension - 1
                                    if parameter.name == "points"
                                    else dimension
                                )
                                if value > maximum:
                                    raise ValueError(
                                        f"{parameter.name} exceeds dimension limit {maximum}"
                                    )
            except (ValueError, TypeError) as error:
                raise GraphValidationError(location, str(error)) from error
    if require_installed and backend == "native":
        from .runtime.component_library import load_component_library

        try:
            load_component_library()
        except ImportError as error:
            raise GraphValidationError(
                "backend", "native extension is not installed"
            ) from error
    return spec

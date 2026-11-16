"""Public API for AutoOptLib."""

from __future__ import annotations

from autooptlib.problems.applications import (
    MaterialStackingInstance,
    RISBeamformingInstance,
    StackingWeights,
    generate_ris_instance,
    generate_stacking_instance,
    load_ris_matlab,
    make_material_stacking_problem,
    make_ris_beamforming_problem,
)
from autooptlib.problems.base import ProblemDefinition, make_problem
from autooptlib.problems.cpu import CPUObjective
from autooptlib.problems.ioh import (
    IOHInstance,
    make_bbob_problem,
    make_ioh_problem,
    make_pbo_problem,
)

from ._version import __version__
from .autoopt import autoopt
from .components import get_component, list_components, register_component
from .components.contracts import (
    ComponentContract,
    ParameterSpec,
    get_component_contract,
)
from .components.extensions import ComponentRequest, ComponentResult, register_operator
from .model_semantics import ExecutionSpec, GraphValidationError
from .programming import (
    AlgorithmGraph,
    DesignSpace,
    NodeSpace,
    Operator,
    Pathway,
    Stage,
    describe_component,
)
from .runtime import (
    EvaluationWorkerEnvironment,
    ScheduledTask,
    TaskResources,
    WorkerContext,
)
from .runtime.population import BatchResult, PopulationBatch
from .serialization import (
    algorithm_from_dict,
    algorithm_to_dict,
    load_algorithm,
    save_algorithm,
)
from .utils.design import Design
from .utils.solve import ObjectiveEvaluationError, Solution, SolutionSet

__all__ = [
    "CPUObjective",
    "ComponentContract",
    "Solution",
    "SolutionSet",
    "ParameterSpec",
    "get_component_contract",
    "ComponentRequest",
    "ComponentResult",
    "register_operator",
    "ExecutionSpec",
    "GraphValidationError",
    "AlgorithmGraph",
    "NodeSpace",
    "DesignSpace",
    "Operator",
    "Pathway",
    "Stage",
    "describe_component",
    "list_components",
    "algorithm_from_dict",
    "algorithm_to_dict",
    "BatchResult",
    "PopulationBatch",
    "Design",
    "IOHInstance",
    "EvaluationWorkerEnvironment",
    "MaterialStackingInstance",
    "RISBeamformingInstance",
    "ScheduledTask",
    "StackingWeights",
    "TaskResources",
    "WorkerContext",
    "autoopt",
    "get_component",
    "generate_ris_instance",
    "generate_stacking_instance",
    "load_ris_matlab",
    "make_material_stacking_problem",
    "make_bbob_problem",
    "make_ioh_problem",
    "make_pbo_problem",
    "make_problem",
    "make_ris_beamforming_problem",
    "ObjectiveEvaluationError",
    "ProblemDefinition",
    "load_algorithm",
    "save_algorithm",
    "register_component",
    "__version__",
]

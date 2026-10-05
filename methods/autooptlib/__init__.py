"""Public API for AutoOptLib."""

from __future__ import annotations

from problems.applications import (
    MaterialStackingInstance,
    RISBeamformingInstance,
    StackingWeights,
    generate_ris_instance,
    generate_stacking_instance,
    load_ris_matlab,
    make_material_stacking_problem,
    make_ris_beamforming_problem,
)
from problems.base import ProblemDefinition, make_problem
from problems.cec2013 import cec2013_f1
from problems.ioh import (
    IOHInstance,
    make_bbob_problem,
    make_ioh_problem,
    make_pbo_problem,
)

from . import learning
from ._version import __version__
from .autoopt import autoopt
from .components import get_component, register_component
from .runtime import (
    EvaluationWorkerEnvironment,
    ScheduledTask,
    TaskResources,
    WorkerContext,
)
from .serialization import load_algorithm, save_algorithm
from .utils.design import Design
from .utils.solve import ObjectiveEvaluationError

__all__ = [
    "Design",
    "IOHInstance",
    "EvaluationWorkerEnvironment",
    "learning",
    "MaterialStackingInstance",
    "RISBeamformingInstance",
    "ScheduledTask",
    "StackingWeights",
    "TaskResources",
    "WorkerContext",
    "autoopt",
    "cec2013_f1",
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

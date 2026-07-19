"""Public API for AutoOptLib."""

from __future__ import annotations

from ._version import __version__
from .applications import (
    MaterialStackingInstance,
    RISBeamformingInstance,
    StackingWeights,
    generate_ris_instance,
    generate_stacking_instance,
    load_ris_matlab,
    make_material_stacking_problem,
    make_ris_beamforming_problem,
)
from .autoopt import autoopt
from .components import get_component, register_component
from .problems.base import ProblemDefinition, make_problem
from .problems.cec2013 import cec2013_f1
from .serialization import load_algorithm, save_algorithm
from .utils.design import Design
from .utils.solve import ObjectiveEvaluationError

__all__ = [
    "Design",
    "MaterialStackingInstance",
    "RISBeamformingInstance",
    "StackingWeights",
    "autoopt",
    "cec2013_f1",
    "get_component",
    "generate_ris_instance",
    "generate_stacking_instance",
    "load_ris_matlab",
    "make_material_stacking_problem",
    "make_problem",
    "make_ris_beamforming_problem",
    "ObjectiveEvaluationError",
    "ProblemDefinition",
    "load_algorithm",
    "save_algorithm",
    "register_component",
    "__version__",
]

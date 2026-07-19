"""Reference application problem definitions used by the documentation."""

from .beamforming import (
    RISBeamformingInstance,
    generate_ris_instance,
    load_ris_matlab,
    make_ris_beamforming_problem,
    reciprocal_sum_rate,
)
from .stacking import (
    MaterialStackingInstance,
    StackingWeights,
    generate_stacking_instance,
    make_material_stacking_problem,
    stacking_constraints,
    stacking_objective,
)

__all__ = [
    "MaterialStackingInstance",
    "RISBeamformingInstance",
    "StackingWeights",
    "generate_ris_instance",
    "generate_stacking_instance",
    "load_ris_matlab",
    "make_material_stacking_problem",
    "make_ris_beamforming_problem",
    "reciprocal_sum_rate",
    "stacking_constraints",
    "stacking_objective",
]

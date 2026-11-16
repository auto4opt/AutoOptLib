"""Problem interfaces and optional adapters for AutoOptLib."""
from .base import ProblemDefinition, make_problem
from .cpu import CPUObjective

__all__ = ["ProblemDefinition", "make_problem", "CPUObjective"]

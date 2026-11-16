"""Binary mutation laws used by ParadisEO's ``eoFastGA`` foundry.

The laws mirror the operators registered by the paper experiment's pinned
ParadisEO revision.  NumPy supplies AutoOptLib's seeded random stream; the
probability distributions and default parameters follow the C++ definitions.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ._utils import ensure_rng, flex_get

MutationStrength = Callable[[np.random.Generator, int], int]


def _extract_binary_matrix(solution: Any, *, component: str) -> np.ndarray:
    decisions = flex_get(solution, "decs")
    matrix = np.asarray(decisions if decisions is not None else solution, dtype=int)
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError(f"Solution must be a non-empty 2-D matrix for {component}.")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError(f"{component} requires binary decisions in {{0, 1}}.")
    return matrix.copy()


def _require_binary_bounds(problem: Any, dimension: int, *, component: str) -> None:
    bounds = np.asarray(flex_get(problem, "bound"))
    if (
        bounds.shape != (2, dimension)
        or not np.all(bounds[0] == 0)
        or not np.all(bounds[1] == 1)
    ):
        raise ValueError(f"{component} requires binary [0, 1] problem bounds.")


def uniform_strength(rng: np.random.Generator, dimension: int) -> int:
    """ParadisEO ``eoUniformBitMutation``: uniform integer in ``[0, n)``."""

    return int(rng.integers(0, dimension))


def standard_strength(rng: np.random.Generator, dimension: int) -> int:
    """ParadisEO standard mutation: ``Bin(n, 1/n)``."""

    return int(rng.binomial(dimension, 1.0 / dimension))


def conditional_strength(rng: np.random.Generator, dimension: int) -> int:
    """Standard mutation conditioned on flipping at least one bit."""

    count = 0
    while count == 0:
        count = standard_strength(rng, dimension)
    return count


def shifted_strength(rng: np.random.Generator, dimension: int) -> int:
    """ParadisEO's default shifted law: ``max(1, Bin(n, 0.5))``."""

    return max(1, int(rng.binomial(dimension, 0.5)))


def normal_strength(rng: np.random.Generator, dimension: int) -> int:
    """ParadisEO default normal mutation strength with its range fallback."""

    sampled = int(np.trunc(rng.normal(1.0 / dimension, np.log(dimension))))
    if sampled < 0 or sampled >= dimension:
        return int(rng.integers(0, dimension))
    return sampled


def fast_strength(rng: np.random.Generator, dimension: int) -> int:
    """ParadisEO fast mutation using the default power-law exponent 1.5."""

    half = dimension // 2
    if half == 0:
        rate = 1.0
    else:
        values = np.arange(1, half + 1, dtype=float)
        weights = np.power(values, -1.5)
        index = int(rng.choice(half, p=weights / np.sum(weights)))
        rate = float(values[index]) / dimension
    return int(rng.binomial(dimension, rate))


def fixed_strength(count: int) -> MutationStrength:
    fixed = int(count)

    def sample(_rng: np.random.Generator, dimension: int) -> int:
        if fixed > dimension:
            raise ValueError(
                f"Cannot flip {fixed} distinct bits in dimension {dimension}."
            )
        return fixed

    return sample


def execute_bit_mutation(
    args: tuple[Any, ...],
    *,
    component: str,
    strength: MutationStrength,
    behavior: list[list[str]],
):
    """Execute one mode-based AutoOptLib binary mutation component."""

    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1]
        auxiliary = args[3] if len(args) > 3 else None
        rng = ensure_rng(auxiliary, problem)
        offspring = _extract_binary_matrix(solution, component=component)
        _require_binary_bounds(problem, offspring.shape[1], component=component)
        for row in range(offspring.shape[0]):
            count = int(strength(rng, offspring.shape[1]))
            if not 0 <= count <= offspring.shape[1]:
                raise RuntimeError(f"{component} sampled invalid strength {count}.")
            if count:
                indices = rng.choice(offspring.shape[1], size=count, replace=False)
                offspring[row, np.asarray(indices, dtype=int)] ^= 1
        return offspring, auxiliary
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return behavior, None
    raise ValueError(f"Unsupported mode: {mode}")


__all__ = [
    "conditional_strength",
    "execute_bit_mutation",
    "fast_strength",
    "fixed_strength",
    "normal_strength",
    "shifted_strength",
    "standard_strength",
    "uniform_strength",
]

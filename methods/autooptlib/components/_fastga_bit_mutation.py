"""PBO bit mutations with one-shot uniform distinct-bit sampling."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from ._bit_mutation import (
    _extract_binary_matrix,
    _require_binary_bounds,
    fixed_strength,
    normal_strength,
    uniform_strength,
)
from ._fastga_bit_crossover import STREAM_INVALIDATION_KEY
from ._utils import ensure_rng


def _native_binomial(
    rng: np.random.Generator, dimension: int, probability: float
) -> int:
    """Match eoRng::binomial's explicit sum of biased coins."""

    return int(np.count_nonzero(rng.random(dimension) < probability))


def _standard_strength(rng: np.random.Generator, dimension: int) -> int:
    return _native_binomial(rng, dimension, 1.0 / dimension)


def _conditional_strength(rng: np.random.Generator, dimension: int) -> int:
    count = 0
    while count == 0:
        count = _standard_strength(rng, dimension)
    return count


def _shifted_strength(rng: np.random.Generator, dimension: int) -> int:
    return max(1, _native_binomial(rng, dimension, 0.5))


def _fast_strength(rng: np.random.Generator, dimension: int) -> int:
    half = dimension // 2
    rate = 1.0
    # The native implementation draws its trigger even when n/2 is zero and
    # the loop below is empty (the one-bit edge case).
    trigger = float(rng.random())
    if half:
        normalizer = sum(float(index) ** -1.5 for index in range(1, half + 1))
        cumulative = 0.0
        for index in range(1, half + 1):
            cumulative += float(index) ** -1.5 / normalizer
            if cumulative >= trigger:
                rate = float(index) / dimension
                break
    return _native_binomial(rng, dimension, rate)


def _execute(
    args: tuple[Any, ...],
    *,
    component: str,
    strength: Callable[[np.random.Generator, int], int],
    behavior: list[list[str]],
):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1]
        auxiliary = args[3] if len(args) > 3 else None
        rng = ensure_rng(auxiliary, problem)
        offspring = _extract_binary_matrix(solution, component=component)
        dimension = offspring.shape[1]
        _require_binary_bounds(problem, dimension, component=component)
        invalidated = np.zeros(offspring.shape[0], dtype=bool)
        for row in range(offspring.shape[0]):
            count = int(strength(rng, dimension))
            if not 0 <= count <= dimension:
                raise RuntimeError(f"{component} sampled invalid strength {count}.")
            if count:
                indices = rng.choice(dimension, size=count, replace=False)
                offspring[row, np.asarray(indices, dtype=int)] ^= 1
                invalidated[row] = True
        state = auxiliary if isinstance(auxiliary, dict) else {}
        state[STREAM_INVALIDATION_KEY] = invalidated
        return offspring, state
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return behavior, None
    raise ValueError(f"Unsupported mode: {mode}")


def search_fastga_bit_uniform(*args):
    return _execute(
        args,
        component="search_fastga_bit_uniform",
        strength=uniform_strength,
        behavior=[[], ["GS", "large"]],
    )


def search_fastga_bit_standard(*args):
    return _execute(
        args,
        component="search_fastga_bit_standard",
        strength=_standard_strength,
        behavior=[["LS", "small"], []],
    )


def search_fastga_bit_conditional(*args):
    return _execute(
        args,
        component="search_fastga_bit_conditional",
        strength=_conditional_strength,
        behavior=[["LS", "small"], []],
    )


def search_fastga_bit_shifted(*args):
    return _execute(
        args,
        component="search_fastga_bit_shifted",
        strength=_shifted_strength,
        behavior=[[], ["GS", "large"]],
    )


def search_fastga_bit_normal(*args):
    return _execute(
        args,
        component="search_fastga_bit_normal",
        strength=normal_strength,
        behavior=[[], ["GS", "large"]],
    )


def search_fastga_bit_fast(*args):
    return _execute(
        args,
        component="search_fastga_bit_fast",
        strength=_fast_strength,
        behavior=[[], ["GS", "large"]],
    )


def search_fastga_bit_one(*args):
    return _execute(
        args,
        component="search_fastga_bit_one",
        strength=fixed_strength(1),
        behavior=[["LS", "small"], []],
    )


def search_fastga_bit_three(*args):
    return _execute(
        args,
        component="search_fastga_bit_three",
        strength=fixed_strength(3),
        behavior=[["LS", "small"], []],
    )


def search_fastga_bit_five(*args):
    return _execute(
        args,
        component="search_fastga_bit_five",
        strength=fixed_strength(5),
        behavior=[["LS", "small"], []],
    )

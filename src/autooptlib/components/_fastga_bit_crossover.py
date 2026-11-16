"""Bit crossovers with the execution semantics used by ParadisEO eoFastGA.

These operators are deliberately separate from Search's legacy crossover
components.  In particular, eo1PtBitXover exchanges a prefix (and permits an
empty prefix), while eoNPtsBitXover reports both children invalid even when
their bit strings happen to be unchanged.  Keeping those details local to the
v6.0 component names preserves all legacy Search experiments.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._utils import ensure_rng

STREAM_INVALIDATION_KEY = "_stream_invalidated_mask"


def _parents_and_state(args: tuple[Any, ...]):
    parent = args[0]
    auxiliary = args[3] if len(args) > 3 else None
    rng = ensure_rng(auxiliary, args[1] if len(args) > 1 else None)
    decisions = getattr(parent, "decs", None)
    decisions = decisions() if callable(decisions) else decisions
    matrix = np.asarray(parent if decisions is None else decisions)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("FastGA crossover requires at least two parent solutions.")
    if matrix.shape[1] == 0:
        raise ValueError("FastGA crossover requires a non-empty bit string.")
    if not np.all((matrix == 0) | (matrix == 1)):
        raise ValueError("FastGA bit crossover requires binary decisions in {0, 1}.")
    state = auxiliary if isinstance(auxiliary, dict) else {}
    return matrix, state, rng


def _finish(children: np.ndarray, state: dict[str, Any], invalidated: np.ndarray):
    # eoFastGA invalidates both members of each pair when its quad operator
    # returns true.  A graph without an explicit Choose may supply the whole
    # population, so preserve that pairwise rule for every batch member.
    state[STREAM_INVALIDATION_KEY] = np.asarray(invalidated, dtype=bool)
    return children, state


def execute_fastga_bit_crossover(
    args: tuple[Any, ...], kind: Literal["uniform", "one", "three", "five"]
):
    mode = args[-1]
    if mode == "execute":
        parents, state, rng = _parents_and_state(args)
        count, dimension = parents.shape
        half = (count + 1) // 2
        first = parents[:half].copy()
        second = parents[count - half :].copy()
        invalid_first = np.zeros(half, dtype=bool)
        invalid_second = np.zeros(half, dtype=bool)

        if kind == "uniform":
            for pair_index in range(half):
                # eoUBitXover short-circuits the RNG coin when two bits are
                # equal.  Keep that exact rule independently for every pair.
                different = np.flatnonzero(first[pair_index] != second[pair_index])
                exchange = np.zeros(dimension, dtype=bool)
                if different.size:
                    exchange[different] = rng.random(different.size) < 0.5
                if np.any(exchange):
                    temporary = first[pair_index, exchange].copy()
                    first[pair_index, exchange] = second[pair_index, exchange]
                    second[pair_index, exchange] = temporary
                    invalid_first[pair_index] = True
                    invalid_second[pair_index] = True

        elif kind == "one":
            for pair_index in range(half):
                site = int(rng.integers(0, dimension))
                changed = not np.array_equal(
                    first[pair_index, :site], second[pair_index, :site]
                )
                if changed:
                    temporary = first[pair_index, :site].copy()
                    first[pair_index, :site] = second[pair_index, :site]
                    second[pair_index, :site] = temporary
                    invalid_first[pair_index] = True
                    invalid_second[pair_index] = True

        else:
            requested_points = 3 if kind == "three" else 5
            if dimension < 2:
                raise ValueError("FastGA n-point crossover requires at least two bits.")
            for pair_index in range(half):
                remaining = min(dimension - 1, requested_points)
                points = np.zeros(dimension, dtype=bool)
                # Match eoNPtsBitXover's repeated uniform draws instead of
                # replacing it with choice(replace=False).
                while remaining:
                    bit = int(rng.integers(0, dimension))
                    if not points[bit]:
                        points[bit] = True
                        remaining -= 1
                exchange = False
                for bit in range(1, dimension):
                    if points[bit]:
                        exchange = not exchange
                    if exchange:
                        temporary = first[pair_index, bit]
                        first[pair_index, bit] = second[pair_index, bit]
                        second[pair_index, bit] = temporary
                # eoNPtsBitXover returns true unconditionally.
                invalid_first[pair_index] = True
                invalid_second[pair_index] = True

        children = np.vstack([first, second])[:count]
        invalidated = np.concatenate([invalid_first, invalid_second])[:count]
        return _finish(children, state, invalidated)

    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", "GS"], None
    raise ValueError(f"Unsupported mode: {mode}")

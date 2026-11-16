"""Exact segment-based n-point crossover shared by fixed-point components."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import ensure_rng


def execute_n_point_crossover(args: tuple[Any, ...], points: int):
    mode = args[-1]
    if mode == "execute":
        parent = args[0]
        auxiliary = args[3] if len(args) > 3 else None
        rng = ensure_rng(auxiliary, args[1] if len(args) > 1 else None)
        decisions = getattr(parent, "decs", None)
        decisions = decisions() if callable(decisions) else decisions
        matrix = np.asarray(parent if decisions is None else decisions)
        if matrix.ndim != 2:
            raise ValueError("N-point crossover requires a two-dimensional population.")
        count, dimension = matrix.shape
        if dimension <= points:
            raise ValueError(f"{points}-point crossover requires dimension > {points}.")
        first = matrix[: (count + 1) // 2]
        second = matrix[count // 2 :]
        pair_count = first.shape[0]
        child_first = first.copy()
        child_second = second.copy()
        for index in range(pair_count):
            cuts = np.sort(
                rng.choice(np.arange(1, dimension), size=points, replace=False)
            )
            toggles = np.zeros(dimension, dtype=bool)
            toggles[cuts] = True
            exchange = np.logical_xor.accumulate(toggles)
            child_first[index, exchange] = second[index, exchange]
            child_second[index, exchange] = first[index, exchange]
        return np.vstack([child_first, child_second])[:count], auxiliary
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", "GS"], None
    raise ValueError(f"Unsupported mode: {mode}")

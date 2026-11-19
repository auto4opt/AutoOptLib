"""Exact two-distinct-parent SSGA tournament replacement used by eoFastGA."""

from __future__ import annotations

import numpy as np

from ..utils._constraint_order import solution_key
from ._ssga_update import replace_parents_fastga


def _distinct_deterministic_victim(parents: list, rng: np.random.Generator) -> int:
    first = int(rng.integers(0, len(parents)))
    if len(parents) == 1:
        return first
    second = first
    while second == first:
        second = int(rng.integers(0, len(parents)))
    return first if solution_key(parents[first]) >= solution_key(parents[second]) else second


def update_ssga_fastga_deterministic_tournament(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        auxiliary = args[3] if len(args) > 3 else None
        return (
            replace_parents_fastga(
                solution, problem, auxiliary, _distinct_deterministic_victim
            ),
            None,
        )
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

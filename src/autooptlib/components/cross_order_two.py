"""Two-order crossover for permutations."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import ensure_rng, flex_get


def _extract_matrix(parent: Any) -> np.ndarray:
    decs = flex_get(parent, "decs")
    if callable(decs):
        arr = np.asarray(decs(), dtype=int)
    elif decs is not None:
        arr = np.asarray(decs, dtype=int)
    else:
        arr = np.asarray(parent, dtype=int)
    if arr.ndim != 2:
        raise ValueError("Parent must be 2-D for cross_order_two")
    return arr


def _order_crossover(
    p1: np.ndarray, p2: np.ndarray, start: int, end: int
) -> np.ndarray:
    segment = p1[start : end + 1]
    remainder = [x for x in p2 if x not in segment]
    mask = np.ones(p1.shape[0], dtype=bool)
    mask[start : end + 1] = False
    offspring = p1.copy()
    offspring[mask] = remainder
    return offspring


def cross_order_two(*args):
    mode = args[-1]
    if mode == "execute":
        parent_obj = args[0]
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        parent = _extract_matrix(parent_obj)
        n, d = parent.shape
        half = (n + 1) // 2
        parent1 = parent[:half]
        parent2 = parent[n - half :]
        off1 = parent1.copy()
        off2 = parent2.copy()
        for i in range(half):
            idx = np.sort(rng.choice(d, size=2, replace=False))
            off1[i] = _order_crossover(parent1[i], parent2[i], int(idx[0]), int(idx[1]))
            idx = np.sort(rng.choice(d, size=2, replace=False))
            off2[i] = _order_crossover(parent2[i], parent1[i], int(idx[0]), int(idx[1]))
        offspring = np.vstack([off1, off2])
        return offspring[:n], aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

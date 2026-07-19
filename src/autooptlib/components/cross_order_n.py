"""N-order crossover for permutations."""

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
        raise ValueError("Parent must be 2-D for cross_order_n")
    return arr


def _stable_setdiff(seq: np.ndarray, remove: np.ndarray) -> list[int]:
    remove_set = set(int(x) for x in remove)
    return [int(x) for x in seq if int(x) not in remove_set]


def cross_order_n(*args):
    mode = args[-1]
    if mode == "execute":
        parent_obj = args[0]
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        n_points = (
            int(round(float(np.asarray(para).reshape(-1)[0])))
            if para is not None
            else 1
        )
        parent = _extract_matrix(parent_obj)
        n, d = parent.shape
        half = (n + 1) // 2
        parent1 = parent[:half]
        parent2 = parent[n - half :]
        off1 = parent1.copy()
        off2 = parent2.copy()
        for i in range(half):
            k = rng.choice(d, size=min(max(1, n_points), d), replace=False)
            k.sort()
            complement = [idx for idx in range(d) if idx not in k]
            temp = _stable_setdiff(parent2[i], parent1[i, k])
            off1[i, complement] = temp

            complement2 = [idx for idx in range(d) if idx not in k]
            temp2 = _stable_setdiff(parent1[i], parent2[i, k])
            off2[i, complement2] = temp2
        offspring = np.vstack([off1, off2])
        return offspring[:n], aux

    if mode == "parameter":
        problem = args[0]
        if isinstance(problem, (list, tuple)):
            problem = problem[0]
        bounds = np.asarray(flex_get(problem, "bound"))
        d = bounds.shape[1] if bounds.ndim == 2 else bounds.size
        n_max = max(1, int(round(d * 0.5)))
        return [1, n_max], None

    if mode == "behavior":
        return [["LS", "small"], ["GS", "large"]], None

    raise ValueError(f"Unsupported mode: {mode}")

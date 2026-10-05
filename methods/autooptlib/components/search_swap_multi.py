"""Multi swap between segments."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import ensure_rng, flex_get


def _extract_matrix(solution: Any) -> np.ndarray:
    decs = flex_get(solution, "decs")
    if callable(decs):
        arr = np.asarray(decs(), dtype=int)
    elif decs is not None:
        arr = np.asarray(decs, dtype=int)
    else:
        arr = np.asarray(solution, dtype=int)
    if arr.ndim != 2:
        raise ValueError("Solution must be 2-D for search_swap_multi")
    return arr.copy()


def search_swap_multi(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        new = _extract_matrix(solution)
        n, d = new.shape
        for i in range(n):
            idx = np.sort(rng.choice(d, size=2, replace=False))
            start, end = int(idx[0]), int(idx[1])
            length = (end - start + 1) // 2
            for j in range(length):
                a = start + j
                b = end - j
                new[i, a], new[i, b] = new[i, b], new[i, a]
        return new, aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

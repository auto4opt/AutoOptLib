"""Reset a single element of each solution."""

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
        raise ValueError("Solution must be 2-D for search_reset_one")
    return arr.copy()


def search_reset_one(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1]
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        new = _extract_matrix(solution)
        bound = np.asarray(flex_get(problem, "bound"), dtype=int)
        lower = bound[0]
        upper = bound[1]
        n, _ = new.shape
        mutable = np.flatnonzero(upper > lower)
        if mutable.size == 0:
            return new, aux
        idx = rng.choice(mutable, size=n, replace=True)
        for i in range(n):
            col = int(idx[i])
            current = new[i, col]
            candidate = rng.integers(lower[col], upper[col] + 1)
            while candidate == current:
                candidate = rng.integers(lower[col], upper[col] + 1)
            new[i, col] = candidate
        return new, aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["LS", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

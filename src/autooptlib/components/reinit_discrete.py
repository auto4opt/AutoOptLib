"""Reinitialization for discrete problems."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import ensure_rng, flex_get


def _extract_shape(solution: Any) -> tuple[int, int]:
    decs = flex_get(solution, "decs")
    if callable(decs):
        arr = np.asarray(decs())
    elif decs is not None:
        arr = np.asarray(decs)
    else:
        arr = np.asarray(solution)
    if arr.ndim != 2:
        raise ValueError("Solution must be 2-D for reinit_discrete")
    return arr.shape


def reinit_discrete(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1]
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        n, d = _extract_shape(solution)
        bound = np.asarray(flex_get(problem, "bound"), dtype=int)
        lower = bound[0]
        upper = bound[1]
        new = np.empty((n, d), dtype=int)
        for j in range(d):
            new[:, j] = rng.integers(lower[j], upper[j] + 1, size=n)
        return new, aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

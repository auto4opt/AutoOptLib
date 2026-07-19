"""Random reset operator for discrete variables."""

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
        raise ValueError("Solution must be 2-D for search_reset_rand")
    return arr


def search_reset_rand(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1]
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        prob = float(np.asarray(para).reshape(-1)[0]) if para is not None else 0.1
        new = _extract_matrix(solution)
        bound = np.asarray(flex_get(problem, "bound"), dtype=int)
        lower = bound[0]
        upper = bound[1]
        n, d = new.shape
        for j in range(d):
            mask = rng.random(n) < prob
            if not np.any(mask):
                continue
            new[mask, j] = rng.integers(lower[j], upper[j] + 1, size=mask.sum())
        return new, aux

    if mode == "parameter":
        return [0, 0.5], None

    if mode == "behavior":
        return [["LS", "small"], ["GS", "large"]], None

    raise ValueError(f"Unsupported mode: {mode}")

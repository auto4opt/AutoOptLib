"""Creep mutation for discrete ordinal variables."""

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
        raise ValueError("Solution must be 2-D for search_reset_creep")
    return arr


def search_reset_creep(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1]
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        params = (
            np.asarray(para).reshape(-1) if para is not None else np.array([0.1, 0.1])
        )
        prob = float(params[0]) if params.size > 0 else 0.1
        amp_ratio = float(params[1]) if params.size > 1 else 0.1

        new = _extract_matrix(solution)
        bound = np.asarray(flex_get(problem, "bound"), dtype=int)
        lower = bound[0]
        upper = bound[1]
        n, d = new.shape

        amp = np.maximum(1, np.round((upper - lower + 1) * amp_ratio).astype(int))
        step = np.zeros((n, d), dtype=int)
        for j in range(d):
            step[:, j] = rng.integers(1, amp[j] + 1, size=n)
        sign = rng.random((n, d)) < 0.5
        step[sign] *= -1
        candidate = new + step
        candidate = np.maximum(candidate, lower)
        candidate = np.minimum(candidate, upper)

        mask = rng.random((n, d)) < prob
        new[mask] = candidate[mask]
        return new, aux

    if mode == "parameter":
        return [0, 0.5, 0, 0.5], None

    if mode == "behavior":
        return [["LS", "small", "small"], ["GS", "large", "large"]], None

    raise ValueError(f"Unsupported mode: {mode}")

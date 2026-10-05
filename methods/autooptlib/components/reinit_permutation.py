"""Reinitialization for permutation problems."""

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
        raise ValueError("Solution must be 2-D for reinit_permutation")
    return arr.shape


def reinit_permutation(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        n, d = _extract_shape(solution)
        rand = rng.random((n, d))
        offspring = np.argsort(rand, axis=1) + 1
        return offspring, aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

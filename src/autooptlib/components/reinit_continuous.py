"""Python translation of reinit_continuous."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng


def reinit_continuous(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, problem)

        try:
            n = len(solution)
        except TypeError:
            n = int(getattr(problem, "N", 1))
        bound = getattr(problem, "bound")
        lower = np.asarray(bound[0], dtype=float)
        upper = np.asarray(bound[1], dtype=float)
        dec = lower + (upper - lower) * rng.random((n, lower.shape[-1]))
        return dec, aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

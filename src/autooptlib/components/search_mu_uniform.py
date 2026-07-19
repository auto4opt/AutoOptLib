"""Python translation of search_mu_uniform."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng


def search_mu_uniform(*args):
    mode = args[-1]
    if mode == "execute":
        parent = args[0]
        problem = args[1] if len(args) > 1 else None
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, problem)

        if not isinstance(parent, (np.ndarray, list, tuple)):
            decs = getattr(parent, "decs", None)
            offspring = (
                decs() if callable(decs) else (decs if decs is not None else parent)
            )
        else:
            offspring = parent
        offspring = np.asarray(offspring, dtype=float)
        prob = float(np.asarray(para).reshape(-1)[0]) if para is not None else 0.1
        n, d = offspring.shape
        lower = np.asarray(getattr(problem, "bound")[0], dtype=float)
        upper = np.asarray(getattr(problem, "bound")[1], dtype=float)
        mask = rng.random((n, d)) < prob
        rand_vals = lower + (upper - lower) * rng.random((n, d))
        offspring[mask] = rand_vals[mask]
        return offspring, aux

    if mode == "parameter":
        return [0, 0.3], None

    if mode == "behavior":
        return [["LS", "small"], ["GS", "large"]], None

    raise ValueError(f"Unsupported mode: {mode}")

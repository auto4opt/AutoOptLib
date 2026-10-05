"""Python translation of search_de_current."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng


def search_de_current(*args):
    mode = args[-1]
    if mode == "execute":
        parent_obj = args[0]
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, args[1] if len(args) > 1 else None)
        decs = getattr(parent_obj, "decs", None)
        if callable(decs):
            parent = decs()
        else:
            parent = decs if decs is not None else parent_obj
        targets = np.asarray(parent, dtype=float)
        context = aux.get("_population_context") if isinstance(aux, dict) else None
        if context is None:
            population = targets
        else:
            context_decs = getattr(context, "decs", None)
            population = np.asarray(
                context_decs() if callable(context_decs) else context, dtype=float
            )
        n, d = targets.shape
        if para is None:
            f, cr = 0.5, 0.5
        else:
            arr = np.asarray(para).reshape(-1)
            f = float(arr[0])
            cr = float(arr[1]) if arr.size > 1 else 0.5
        if context is None:
            p2 = population[rng.permutation(n)]
            p3 = population[rng.permutation(n)]
        else:
            p2 = population[rng.integers(0, len(population), size=n)]
            p3 = population[rng.integers(0, len(population), size=n)]
        mask = rng.random((n, d)) < cr
        offspring = targets.copy()
        donor = targets + f * (p2 - p3)
        offspring[mask] = donor[mask]
        return offspring, aux
    if mode == "parameter":
        return [[0, 1], [0, 1]], None
    if mode == "behavior":
        return [["LS", "small", "small"], ["GS", "large", "large"]], None
    raise ValueError(f"Unsupported mode: {mode}")

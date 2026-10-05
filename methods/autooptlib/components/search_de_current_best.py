"""Python translation of search_de_current_best."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng


def search_de_current_best(*args):
    mode = args[-1]
    if mode == "execute":
        parent_obj = args[0]
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, args[1] if len(args) > 1 else None)
        context = aux.get("_population_context") if isinstance(aux, dict) else None
        population_obj = parent_obj if context is None else context
        fits = getattr(population_obj, "fits", None)
        if callable(fits):
            fit_vals = np.asarray(fits()).reshape(-1)
        else:
            fit_vals = np.asarray(
                fits if fits is not None else getattr(population_obj, "fit", None)
            ).reshape(-1)
        best_idx = int(np.argmin(fit_vals))
        best_dec = getattr(population_obj[best_idx], "dec", None)
        if callable(best_dec):
            gbest = np.asarray(best_dec())
        else:
            gbest = np.asarray(
                best_dec
                if best_dec is not None
                else getattr(population_obj, "decs")[best_idx]
            )
        decs = getattr(parent_obj, "decs", None)
        if callable(decs):
            parent = decs()
        else:
            parent = decs if decs is not None else parent_obj
        targets = np.asarray(parent, dtype=float)
        context_decs = getattr(population_obj, "decs", None)
        population = np.asarray(
            context_decs() if callable(context_decs) else population_obj,
            dtype=float,
        )
        n, d = targets.shape
        if para is None:
            f, cr = 0.5, 0.5
        else:
            arr = np.asarray(para).reshape(-1)
            f = float(arr[0])
            cr = float(arr[1]) if arr.size > 1 else 0.5
        p2 = np.repeat(gbest.reshape(1, -1), n, axis=0)
        if context is None:
            p3 = population[rng.permutation(n)]
        else:
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

"""Estimation of Distribution Algorithm (EDA) operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import ensure_rng


def search_eda(*args: Any):
    mode = args[-1]
    if mode == "execute":
        parent = args[0]
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, args[1] if len(args) > 1 else None)

        decs = getattr(parent, "decs", None)
        if callable(decs):
            parent = decs()
        else:
            parent = decs if decs is not None else parent
        targets = np.asarray(parent, dtype=float)
        context = aux.get("_population_context") if isinstance(aux, dict) else None
        if context is None:
            population = targets
        else:
            context_decs = getattr(context, "decs", None)
            population = np.asarray(
                context_decs() if callable(context_decs) else context, dtype=float
            )
        if population.ndim != 2 or targets.ndim != 2:
            raise ValueError("Parent decision variables must form a 2D array")
        n = targets.shape[0]
        model_n = population.shape[0]
        mean = population.mean(axis=0)
        # A chooser may legitimately feed EDA a single selected parent even
        # when the algorithm population itself is larger. ``ddof=1`` on one
        # row yields NaNs and previously poisoned the generated offspring.
        std = population.std(axis=0, ddof=1 if model_n > 1 else 0)
        std = np.where(
            np.isfinite(std) & (std > 0),
            std,
            population.std(axis=0, ddof=0),
        )
        std = np.where(std > 0, std, 1e-12)
        offspring = rng.normal(loc=mean, scale=std, size=(n, targets.shape[1]))
        return offspring, aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

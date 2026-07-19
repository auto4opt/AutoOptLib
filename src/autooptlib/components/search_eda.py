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
        parent = np.asarray(parent, dtype=float)
        if parent.ndim != 2:
            raise ValueError("Parent decision variables must form a 2D array")
        n, _ = parent.shape
        mean = parent.mean(axis=0)
        std = parent.std(axis=0, ddof=1)
        std = np.where(np.isfinite(std) & (std > 0), std, parent.std(axis=0, ddof=0))
        std = np.where(std > 0, std, 1e-12)
        offspring = rng.normal(loc=mean, scale=std, size=(n, parent.shape[1]))
        return offspring, aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

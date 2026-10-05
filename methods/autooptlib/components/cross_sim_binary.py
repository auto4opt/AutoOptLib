"""Simulated binary crossover (SBX) operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import ensure_rng


def cross_sim_binary(*args: Any):
    mode = args[-1]
    if mode == "execute":
        parent = args[0]
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, args[1] if len(args) > 1 else None)

        decs = getattr(parent, "decs", None)
        if callable(decs):
            parent = decs()
        else:
            parent = decs if decs is not None else parent
        parent = np.asarray(parent, dtype=float)
        n, d = parent.shape
        first = parent[: (n + 1) // 2]
        second = parent[n // 2 :]
        n_half = first.shape[0]

        eta = (
            float(np.asarray(para, dtype=float).reshape(-1)[0])
            if para is not None
            else 20.0
        )
        mu = rng.random((n_half, d))
        beta = np.empty_like(mu)
        mask = mu <= 0.5
        beta[mask] = (2.0 * mu[mask]) ** (1.0 / (1.0 + eta))
        beta[~mask] = (2.0 - 2.0 * mu[~mask]) ** (-1.0 / (1.0 + eta))

        offspring = np.vstack(
            (
                0.5 * ((1.0 + beta) * first + (1.0 - beta) * second),
                0.5 * ((1.0 - beta) * first + (1.0 + beta) * second),
            )
        )
        return offspring[:n], aux

    if mode == "parameter":
        return np.array([20.0, 40.0], dtype=float), None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

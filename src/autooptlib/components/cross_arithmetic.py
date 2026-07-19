"""Whole arithmetic crossover operator."""

from __future__ import annotations

from typing import Any

import numpy as np


def cross_arithmetic(*args: Any):
    mode = args[-1]
    if mode == "execute":
        parent = args[0]
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None  # noqa: F841

        decs = getattr(parent, "decs", None)
        if callable(decs):
            parent = decs()
        else:
            parent = decs if decs is not None else parent
        parent = np.asarray(parent, dtype=float)
        n, d = parent.shape
        first = parent[: (n + 1) // 2]
        second = parent[n // 2 :]

        prob = (
            np.asarray(para, dtype=float)
            if para is not None
            else np.array(0.5, dtype=float)
        )
        if prob.ndim == 0:
            ratio = prob
        else:
            ratio = prob.reshape(1, -1)

        offspring1 = ratio * first + (1.0 - ratio) * second
        offspring2 = ratio * second + (1.0 - ratio) * first
        offspring = np.vstack((offspring1, offspring2))
        return offspring[:n], aux

    if mode == "parameter":
        return np.array([0.0, 0.3], dtype=float), None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

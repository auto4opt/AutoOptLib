"""Imperialist Competitive Algorithm (ICA) search operator."""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from ._utils import ensure_rng, flex_get


def _ensure_aux(aux: Any) -> Dict[str, Any]:
    if isinstance(aux, dict):
        return aux
    if aux is None:
        return {}
    return dict(aux)


def _get_bounds(problem: Any, decs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    bound = flex_get(problem, "bound", None)
    if bound is None:
        lower = np.min(decs, axis=0)
        upper = np.max(decs, axis=0)
        if np.allclose(lower, upper):
            lower = lower - 1.0
            upper = upper + 1.0
        return lower, upper
    arr = np.asarray(bound, dtype=float)
    return arr[0], arr[1]


def search_ica(*args: Any):
    mode = args[-1]
    if mode == "execute":
        raw_parent = args[0]
        problem = args[1] if len(args) > 1 else None
        para = args[2] if len(args) > 2 else None
        aux = _ensure_aux(args[3] if len(args) > 3 else None)
        rng = ensure_rng(aux, problem)

        if not isinstance(raw_parent, (np.ndarray, list, tuple)):
            decs_attr = getattr(raw_parent, "decs", None)
            parent = (
                decs_attr()
                if callable(decs_attr)
                else (decs_attr if decs_attr is not None else raw_parent)
            )
        else:
            parent = raw_parent
        parent = np.asarray(parent, dtype=float)
        if parent.ndim != 2:
            parent = parent.reshape(1, -1)
        n, d = parent.shape
        if n == 0:
            return parent, aux

        assimilation_coeff = 0.6
        revolution_prob = 0.5
        if para is not None:
            arr = np.asarray(para).reshape(-1)
            if arr.size > 0:
                assimilation_coeff = float(arr[0])
            if arr.size > 1:
                revolution_prob = float(arr[1])

        imperialist_count = max(1, min(int(np.sqrt(n)), n))

        fits = flex_get(raw_parent, "fits", None)
        if fits is None:
            fitness = np.zeros(n)
        else:
            fitness = np.asarray(fits, dtype=float).reshape(-1)
            if fitness.size != n:
                fitness = np.resize(fitness, n)

        order = np.argsort(fitness)
        imperialists = parent[order[:imperialist_count]]
        colonies = parent[order[imperialist_count:]]

        lower, upper = _get_bounds(problem, parent)
        range_span = upper - lower

        offspring = parent.copy()
        if colonies.size:
            colony_indices = order[imperialist_count:]
            assign = rng.integers(0, imperialist_count, size=colonies.shape[0])
            rand_scale = rng.random((colonies.shape[0], d))
            target = imperialists[assign]
            moved = colonies + assimilation_coeff * rand_scale * (target - colonies)
            moved = np.clip(moved, lower, upper)
            offspring[colony_indices] = moved

        revol_mask = rng.random((n, d)) < revolution_prob
        if np.any(revol_mask):
            random_values = lower + range_span * rng.random((n, d))
            offspring[revol_mask] = random_values[revol_mask]

        return offspring, aux

    if mode == "parameter":
        return np.array([0.6, 0.5]), None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

"""Python translation of search_de_random."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng


def search_de_random(*args):
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
        parent = np.asarray(parent, dtype=float)
        n, d = parent.shape
        if para is None:
            f, cr = 0.5, 0.5
        else:
            arr = np.asarray(para).reshape(-1)
            f = float(arr[0])
            cr = float(arr[1]) if arr.size > 1 else 0.5
        p1 = parent[rng.permutation(n)]
        p2 = parent[rng.permutation(n)]
        p3 = parent[rng.permutation(n)]
        mask = rng.random((n, d)) < cr
        offspring = parent.copy()
        offspring[mask] = p1[mask] + f * (p2[mask] - p3[mask])
        return offspring, aux
    if mode == "parameter":
        return [[0, 1], [0, 1]], None
    if mode == "behavior":
        return [["LS", "small", "small"], ["GS", "large", "large"]], None
    raise ValueError(f"Unsupported mode: {mode}")

"""Python translation of cross_point_n."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, flex_get


def _minimum_dimension(problem):
    problems = problem if isinstance(problem, (list, tuple)) else [problem]
    dimensions = []
    for item in problems:
        bound = np.asarray(flex_get(item, "bound", np.empty((2, 1))))
        if bound.ndim == 2:
            dimensions.append(int(bound.shape[1]))
    return max(1, min(dimensions, default=1))


def cross_point_n(*args):
    mode = args[-1]
    if mode == "execute":
        parent = args[0]
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, args[1] if len(args) > 1 else None)
        n_points = (
            int(round(float(np.asarray(para).reshape(-1)[0])))
            if para is not None
            else 1
        )
        decs = getattr(parent, "decs", None)
        if callable(decs):
            parent = decs()
        else:
            parent = decs if decs is not None else parent
        parent = np.asarray(parent)
        n, d = parent.shape
        p1 = parent[: (n + 1) // 2]
        p2 = parent[n // 2 :]
        nh = p1.shape[0]
        off1 = p1.copy()
        off2 = p2.copy()
        for i in range(nh):
            k = rng.choice(d, size=min(max(1, n_points), d), replace=False)
            off1[i, k] = p2[i, k]
            off2[i, k] = p1[i, k]
        offspring = np.vstack([off1, off2])
        return offspring[:n, :], aux
    if mode == "parameter":
        problem = args[0] if len(args) > 1 else None
        return [1, _minimum_dimension(problem)], None
    if mode == "behavior":
        return [["LS", "small"], ["GS", "large"]], None
    raise ValueError(f"Unsupported mode: {mode}")

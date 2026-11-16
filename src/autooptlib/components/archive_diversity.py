"""Select the most diversified solutions for the archive."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ._utils import ensure_rng, flex_get


def _ensure_solution_list(sol: Any) -> list:
    if isinstance(sol, Sequence):
        return list(sol)
    return [sol]


def _stack_decs(sol_list: list) -> np.ndarray:
    decs = [
        np.asarray(flex_get(s, "dec"), dtype=float).reshape(1, -1) for s in sol_list
    ]
    return np.vstack(decs)


def archive_diversity(*args):
    mode = args[-1]
    if mode == "execute":
        solutions = _ensure_solution_list(args[0])
        archive = _ensure_solution_list(args[1]) if len(args) > 1 else []
        problem = args[2] if len(args) > 2 else None
        rng = ensure_rng(problem)

        combined = solutions + archive
        if not combined:
            return [], None
        decs = _stack_decs(combined)
        ptype = flex_get(problem, "type", ["continuous"])
        if isinstance(ptype, Sequence):
            ptype = ptype[0]
        if ptype == "continuous":
            dist = np.linalg.norm(decs[:, None, :] - decs[None, :, :], axis=-1)
        else:
            dist = np.mean(decs[:, None, :] != decs[None, :, :], axis=-1)
        n = int(flex_get(problem, "N", len(solutions)))
        chosen = [int(rng.integers(len(combined)))]
        while len(chosen) < n and len(chosen) < len(combined):
            sums = np.sum(dist[chosen], axis=0)
            order = np.argsort(sums)[::-1]
            for idx in order:
                if idx not in chosen:
                    chosen.append(int(idx))
                    break
        chosen = chosen[: min(n, len(combined))]
        return [combined[i] for i in chosen], None

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

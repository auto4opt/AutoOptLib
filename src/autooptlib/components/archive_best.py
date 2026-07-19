"""Python translation of archive_best."""

from __future__ import annotations

import numpy as np


def _extract_fits(solution) -> np.ndarray:
    fits = getattr(solution, "fits", None)
    if callable(fits):
        return np.asarray(fits()).reshape(-1)
    if fits is not None:
        return np.asarray(fits).reshape(-1)
    # fallback: build from sequence of items with `.fit` or `.obj`/`.con`
    try:
        seq = list(solution)
    except TypeError as exc:  # not iterable
        raise ValueError(
            "Solution must expose fits() or be iterable of items with fitness"
        ) from exc
    vals = []
    for it in seq:
        val = getattr(it, "fit", None)
        if val is None:
            obj = getattr(it, "obj", None)
            con = getattr(it, "con", None)
            if obj is None:
                raise ValueError(
                    "Item lacks fit/obj attributes for fitness computation"
                )
            if con is None:
                val = float(obj)
            else:
                con_sum = float(np.sum(np.maximum(0.0, np.asarray(con))))
                feasible = con_sum <= 0.0
                val = float(obj) if feasible else (con_sum + 1e8)
        vals.append(float(val))
    return np.asarray(vals, dtype=float)


def archive_best(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        curr_archive = args[1] if len(args) > 1 else []
        fitness = _extract_fits(solution)
        best = int(np.argmin(fitness))
        try:
            # index into sequence of Solution-like objects
            best_item = solution[best]
        except TypeError:
            # if solution is a custom container, try to recover by getattr
            seq = list(solution)
            best_item = seq[best]
        return list(curr_archive) + [best_item], None

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

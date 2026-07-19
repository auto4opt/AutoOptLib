"""Python translation of choose_traverse."""

from __future__ import annotations

import numpy as np

from ._utils import flex_get


def choose_traverse(*args):
    mode = args[-1]
    if mode == "execute":
        problem = args[1] if len(args) > 1 else None
        size = flex_get(problem, "N")
        if size is None:
            # fall back to population size
            solution = args[0] if args else None
            size = flex_get(solution, "__len__", 0)
            size = size() if callable(size) else size
        index = np.arange(int(size))
        return index, None
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

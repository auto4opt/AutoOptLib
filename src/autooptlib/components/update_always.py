"""Python translation of update_always."""

from __future__ import annotations

from typing import Sequence

from ._utils import flex_get


def update_always(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        n = int(flex_get(problem, "N", len(solution)))
        if isinstance(solution, Sequence):
            return list(solution[-n:]), None
        raise TypeError("Solution collection must be a sequence for update_always")

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

"""Python translation of choose_traverse."""

from __future__ import annotations

import numpy as np

from ._utils import flex_get


def choose_traverse(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0] if args else None
        try:
            size = len(solution)
        except (TypeError, AttributeError):
            decisions = flex_get(solution, "decs")
            decisions = decisions() if callable(decisions) else decisions
            size = np.asarray(decisions).shape[0] if decisions is not None else 0
        if not size:
            problem = args[1] if len(args) > 1 else None
            size = flex_get(problem, "N", 0)
        size = int(size or 0)
        if size <= 0:
            return np.empty(0, dtype=int), None

        aux = args[3] if len(args) > 3 else None
        requested = flex_get(aux, "_selection_count")
        if requested is None:
            # Direct component calls retain the historical public contract.
            return np.arange(size, dtype=int), None

        requested = max(0, int(requested))
        state = dict(aux) if isinstance(aux, dict) else {}
        previous_size = int(state.get("choose_traverse_population_size", size))
        cursor = int(state.get("choose_traverse_cursor", 0))
        if previous_size != size:
            cursor = 0
        index = (cursor + np.arange(requested, dtype=int)) % size
        state["choose_traverse_cursor"] = int((cursor + requested) % size)
        state["choose_traverse_population_size"] = size
        state.pop("_selection_count", None)
        return index, state
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

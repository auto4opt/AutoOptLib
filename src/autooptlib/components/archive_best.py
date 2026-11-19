"""Python translation of archive_best."""

from __future__ import annotations

import numpy as np

from ._utils import extract_fits


def archive_best(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        curr_archive = args[1] if len(args) > 1 else []
        fitness = extract_fits(solution)
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

"""ParadisEO eoSSGAStochTournamentReplacement compatibility component."""

from __future__ import annotations

import numpy as np

from ._ssga_update import replace_parents_fastga


def _fastga_stochastic_victim(parents: list, rng: np.random.Generator) -> int:
    first = int(rng.integers(0, len(parents)))
    second = int(rng.integers(0, len(parents)))
    return_worse = bool(rng.random() < 0.75)
    if parents[first].fit > parents[second].fit:
        return first if return_worse else second
    return second if return_worse else first


def update_ssga_fastga_stochastic_tournament(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        auxiliary = args[3] if len(args) > 3 else None
        return (
            replace_parents_fastga(
                solution, problem, auxiliary, _fastga_stochastic_victim
            ),
            None,
        )
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

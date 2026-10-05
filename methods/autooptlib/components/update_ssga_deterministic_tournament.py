"""Steady-state replacement using a reverse deterministic tournament."""

from ._ssga_update import deterministic_tournament_victim, replace_parents


def update_ssga_deterministic_tournament(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        auxiliary = args[3] if len(args) > 3 else None
        return (
            replace_parents(
                solution, problem, auxiliary, deterministic_tournament_victim
            ),
            None,
        )
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

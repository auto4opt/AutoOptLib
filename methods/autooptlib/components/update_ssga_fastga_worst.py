"""ParadisEO eoSSGAWorseReplacement compatibility component."""

from ._ssga_update import replace_parents_fastga, worst_victim


def update_ssga_fastga_worst(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        auxiliary = args[3] if len(args) > 3 else None
        return replace_parents_fastga(solution, problem, auxiliary, worst_victim), None
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", ""], None
    raise ValueError(f"Unsupported mode: {mode}")

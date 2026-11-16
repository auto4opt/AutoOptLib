"""Python translation of cross_point_one."""

from __future__ import annotations

from ._point_crossover import prepare, draw_one, splice_one, result, specification


def cross_point_one(*args):
    mode = args[-1]
    if mode == "execute":
        rng, parent, n, d, p1, p2, probability = prepare(args, "one")
        cuts = draw_one(rng, p1, d, probability)
        return result(splice_one(p1, p2, cuts), n, args)
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", "GS"], None
    raise ValueError(f"Unsupported mode: {mode}")


cross_point_one.__execution_computation__ = specification(cross_point_one, "one", draw_one, splice_one)

"""Python translation of cross_point_two."""

from __future__ import annotations

from ._point_crossover import prepare, draw_two, splice_two, result, specification


def cross_point_two(*args):
    mode = args[-1]
    if mode == "execute":
        rng, parent, n, d, p1, p2, probability = prepare(args, "two")
        cuts = draw_two(rng, p1, d, probability)
        return result(splice_two(p1, p2, cuts), n, args)
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", "GS"], None
    raise ValueError(f"Unsupported mode: {mode}")


cross_point_two.__execution_computation__ = specification(cross_point_two, "two", draw_two, splice_two)

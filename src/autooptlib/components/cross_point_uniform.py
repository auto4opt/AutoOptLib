"""Python translation of cross_point_uniform."""

from __future__ import annotations

from ._point_crossover import prepare, draw_uniform, splice_uniform, result, specification


def cross_point_uniform(*args):
    mode = args[-1]
    if mode == "execute":
        rng, parent, n, d, p1, p2, probability = prepare(args, "uniform")
        cuts = draw_uniform(rng, p1, d, probability)
        return result(splice_uniform(p1, p2, cuts), n, args)
    if mode == "parameter":
        return [0.05, 0.5], None
    if mode == "behavior":
        return [["LS", "small"], ["GS", "large"]], None
    raise ValueError(f"Unsupported mode: {mode}")


cross_point_uniform.__execution_computation__ = specification(cross_point_uniform, "uniform", draw_uniform, splice_uniform)

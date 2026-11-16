"""Shared reference primitives for one/two-point and uniform crossover."""

import numpy as np

from ._execution import Computation, DependencyGuard, NativePattern, Primitive
from ._utils import decision_matrix, ensure_rng, rng_method


def prepare(args, kind):
    aux = args[3] if len(args) > 3 else None
    rng = ensure_rng(aux, args[1] if len(args) > 1 else None)
    probability = None
    if kind == "uniform":
        para = args[2] if len(args) > 2 else None
        probability = (
            float(np.asarray(para).reshape(-1)[0]) if para is not None else 0.5
        )
    parent = decision_matrix(args[0])
    n, d = parent.shape
    if kind == "two" and d < 2:
        raise ValueError("Two-point crossover requires D >= 2")
    return rng, parent, n, d, parent[: (n + 1) // 2], parent[n // 2 :], probability


def draw_one(rng, first, d, probability):
    return rng_method(rng, "integers")(0, d, size=first.shape[0])


def draw_two(rng, first, d, probability):
    return np.vstack(
        [
            np.sort(rng_method(rng, "choice")(d, size=2, replace=False))
            for _ in range(first.shape[0])
        ]
    )


def draw_uniform(rng, first, d, probability):
    return rng_method(rng, "random")((first.shape[0], d)) < probability


def splice_one(first, second, cuts):
    off1, off2 = first.copy(), second.copy()
    for i in range(first.shape[0]):
        off1[i, cuts[i] :] = second[i, cuts[i] :]
        off2[i, cuts[i] :] = first[i, cuts[i] :]
    return np.vstack([off1, off2])


def splice_two(first, second, cuts):
    off1, off2 = first.copy(), second.copy()
    for i in range(first.shape[0]):
        a, b = cuts[i]
        off1[i, a : b + 1] = second[i, a : b + 1]
        off2[i, a : b + 1] = first[i, a : b + 1]
    return np.vstack([off1, off2])


def splice_uniform(first, second, mask):
    off1, off2 = first.copy(), second.copy()
    off1[mask] = second[mask]
    off2[mask] = first[mask]
    return np.vstack([off1, off2])


def result(offspring, n, args):
    return offspring[:n, :], args[3] if len(args) > 3 else None


def specification(reference, kind, draw, splice):
    return Computation(
        reference,
        (
            Primitive(
                "bind-rng-parameter-and-parents",
                prepare,
                ("args",),
                ("rng", "parent", "n", "d", "first", "second", "probability"),
                ("external-read", "parameter-read"),
                reads=("parents.decisions", "parameters", "auxiliary.rng"),
                positional_tail=(kind,),
            ),
            Primitive(
                "draw-reference-cuts",
                draw,
                ("rng", "first", "d", "probability"),
                ("cuts",),
                ("rng",),
                writes=("rng",),
            ),
            Primitive(
                "splice-ordered-pairs",
                splice,
                ("first", "second", "cuts"),
                ("offspring",),
            ),
            Primitive(
                "return-output-view",
                result,
                ("offspring", "n", "args"),
                ("result",),
                aliases=(("result", "offspring"),),
                binding_point=2,
            ),
        ),
        native_patterns=(
            NativePattern(
                "splice_" + kind,
                ("splice-ordered-pairs",),
                ("first", "second", "cuts"),
                ("offspring",),
            ),
        ),
        native_dependencies=DependencyGuard((splice,)),
    )

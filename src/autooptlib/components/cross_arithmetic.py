"""Whole arithmetic crossover operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import decision_matrix
from ._execution import Computation, Primitive, NativePattern


def _read(args):
    parent = decision_matrix(args[0], dtype=float)
    n, d = parent.shape
    first = parent[: (n + 1) // 2]
    second = parent[n // 2 :]
    return parent, n, first, second


def _ratio(args):
    para = args[2] if len(args) > 2 else None
    prob = np.asarray(para, dtype=float) if para is not None else np.array(0.5, dtype=float)
    return prob if prob.ndim == 0 else prob.reshape(1, -1)


def _mix(first, second, ratio):
    offspring1 = ratio * first + (1.0 - ratio) * second
    offspring2 = ratio * second + (1.0 - ratio) * first
    return np.vstack((offspring1, offspring2))


def _result(offspring, n, args):
    return offspring[:n], args[3] if len(args) > 3 else None


def cross_arithmetic(*args: Any):
    mode = args[-1]
    if mode == "execute":
        parent, n, first, second = _read(args)
        ratio = _ratio(args)
        return _result(_mix(first, second, ratio), n, args)

    if mode == "parameter":
        return np.array([0.0, 0.3], dtype=float), None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")


cross_arithmetic.__execution_computation__ = Computation(cross_arithmetic, (
    Primitive("read-ordered-parents", _read, ("args",), ("parent", "n", "first", "second"),
              ("external-read",), reads=("parents.decisions",)),
    Primitive("bind-current-ratio", _ratio, ("args",), ("ratio",),
              ("parameter-read",), reads=("parameters",)),
    Primitive("arithmetic-mix", _mix, ("first", "second", "ratio"), ("offspring",)),
    Primitive("return-owning-base-view", _result, ("offspring", "n", "args"), ("result",),
              aliases=(("result", "offspring"),), binding_point=2),
), native_patterns=(NativePattern("arithmetic", ("arithmetic-mix",),
    ("first", "second", "ratio"), ("offspring",)),))

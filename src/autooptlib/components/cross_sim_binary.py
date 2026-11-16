"""Simulated binary crossover (SBX) operator."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import decision_matrix, ensure_rng, rng_method
from ._execution import Computation, Primitive


def _read(args):
    aux = args[3] if len(args) > 3 else None
    rng = ensure_rng(aux, args[1] if len(args) > 1 else None)
    parent = decision_matrix(args[0], dtype=float)
    n, d = parent.shape
    first = parent[: (n + 1) // 2]
    second = parent[n // 2 :]
    return rng, parent, n, d, first, second


def _parameter(args):
    para = args[2] if len(args) > 2 else None
    return float(np.asarray(para, dtype=float).reshape(-1)[0]) if para is not None else 20.0


def _draw(rng, first, d):
    return rng_method(rng, "random")((first.shape[0], d))


def _beta(mu, eta):
    beta = np.empty_like(mu)
    mask = mu <= 0.5
    beta[mask] = (2.0 * mu[mask]) ** (1.0 / (1.0 + eta))
    beta[~mask] = (2.0 - 2.0 * mu[~mask]) ** (-1.0 / (1.0 + eta))
    return beta


def _mix(first, second, beta):
    return np.vstack((
        0.5 * ((1.0 + beta) * first + (1.0 - beta) * second),
        0.5 * ((1.0 - beta) * first + (1.0 + beta) * second),
    ))


def _result(offspring, n, args):
    return offspring[:n], args[3] if len(args) > 3 else None


def cross_sim_binary(*args: Any):
    mode = args[-1]
    if mode == "execute":
        rng, parent, n, d, first, second = _read(args)
        eta = _parameter(args)
        mu = _draw(rng, first, d)
        beta = _beta(mu, eta)
        return _result(_mix(first, second, beta), n, args)

    if mode == "parameter":
        return np.array([20.0, 40.0], dtype=float), None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")


cross_sim_binary.__execution_computation__ = Computation(cross_sim_binary, (
    Primitive("read-ordered-parents", _read, ("args",), ("rng", "parent", "n", "d", "first", "second"),
              ("external-read",), reads=("parents.decisions", "auxiliary.rng")),
    Primitive("bind-current-eta", _parameter, ("args",), ("eta",), ("parameter-read",), reads=("parameters",)),
    Primitive("draw-pair-coordinate-coins", _draw, ("rng", "first", "d"), ("mu",), ("rng",), writes=("rng",)),
    Primitive("reference-power", _beta, ("mu", "eta"), ("beta",)),
    Primitive("simulated-binary-mix", _mix, ("first", "second", "beta"), ("offspring",)),
    Primitive("return-output-view", _result, ("offspring", "n", "args"), ("result",),
              aliases=(("result", "offspring"),), binding_point=4),
))

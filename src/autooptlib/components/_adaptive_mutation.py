"""Shared self-adaptation for Gaussian and Cauchy mutation."""

from __future__ import annotations

import numpy as np

from ._execution import Computation, Primitive
from ._utils import decision_matrix, ensure_rng, compact_rows, finalize_rows, input_boundary, write_aux, rng_method


def _set_eta(aux, key, eta):
    write_aux(aux, key, eta)


def _read(args):
    parent = args[0]
    aux = args[3] if len(args) > 3 else None
    rng = ensure_rng(aux, args[1] if len(args) > 1 else None)
    inner_g = int(args[5]) if len(args) > 5 else 1
    if not isinstance(parent, (np.ndarray, list, tuple)):
        parent = decision_matrix(parent, dtype=float)
    parent = np.asarray(parent, dtype=float)
    n, d = parent.shape
    input_boundary("input-prefix-end")
    return parent, aux, rng, inner_g, n, d


def _eta(aux, rng, inner_g, n, d, key):
    eta = None
    if inner_g != 1:
        eta = aux.get(key) if isinstance(aux, dict) else getattr(aux, key, None)
        if eta is not None and np.shape(eta) != (n, d):
            eta = None
    if eta is None:
        eta = rng_method(rng, "random")((n, d))
        _set_eta(aux, key, eta)
    return eta


def _disturb(eta, rng, n, d, distribution):
    return eta * rng_method(rng, distribution)(size=(n, d))


def _offspring(parent, disturb):
    return parent + disturb


def _scales(d):
    return 1.0 / np.sqrt(2.0 * np.sqrt(d)), 1.0 / np.sqrt(2.0 * d)


def _row_normal(rng, n, d):
    return compact_rows(rng_method(rng, "standard_normal")((n, 1)), d)


def _coordinate_normal(rng, n, d):
    return rng_method(rng, "standard_normal")((n, d))


def _adapt(eta, tau1, tau2, normal, normal_j, d):
    normal = finalize_rows(normal, normal_j, d)
    return eta * np.exp(tau2 * normal + tau1 * normal_j)


def _commit(aux, eta, offspring, key):
    _set_eta(aux, key, eta)
    return offspring, aux


def specification(reference, key, distribution):
    return Computation(reference, (
        Primitive("read-parent-and-rng", _read, ("args",), ("parent", "aux", "rng", "inner_g", "n", "d"),
                  ("external-read",), reads=("parents.decisions", "auxiliary.rng", "inner_g")),
        Primitive("read-or-initialize-eta", _eta, ("aux", "rng", "inner_g", "n", "d"), ("eta",),
                  ("state-read", "conditional-rng", "state-write"), reads=(key,), writes=(key, "rng"), positional_tail=(key,)),
        Primitive("draw-and-scale-disturbance", _disturb,
                  ("eta", "rng", "n", "d"), ("disturb",), ("rng",), writes=("rng",), positional_tail=(distribution,)),
        Primitive("add-disturbance", _offspring, ("parent", "disturb"), ("offspring",)),
        Primitive("adaptation-scales", _scales, ("d",), ("tau1", "tau2")),
        Primitive("draw-row-normal", _row_normal, ("rng", "n", "d"), ("normal",), ("rng",), writes=("rng",)),
        Primitive("draw-coordinate-normal", _coordinate_normal, ("rng", "n", "d"), ("normal_j",), ("rng",), writes=("rng",)),
        Primitive("reference-exponential", _adapt, ("eta", "tau1", "tau2", "normal", "normal_j", "d"), ("next_eta",)),
        Primitive("commit-current-eta", _commit, ("aux", "next_eta", "offspring"), ("result",),
                  ("state-write",), writes=(key,), positional_tail=(key,)),
    ), dispatcher=execute_adaptive_mutation)


def execute_adaptive_mutation(args, key, distribution):
    mode = args[-1]
    if mode == "execute":
        parent, aux, rng, inner_g, n, d = _read(args)
        eta = _eta(aux, rng, inner_g, n, d, key)
        disturb = _disturb(eta, rng, n, d, distribution)
        offspring = _offspring(parent, disturb)
        tau1, tau2 = _scales(d)
        normal = _row_normal(rng, n, d)
        normal_j = _coordinate_normal(rng, n, d)
        eta = _adapt(eta, tau1, tau2, normal, normal_j, d)
        return _commit(aux, eta, offspring, key)
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", "GS"], None
    raise ValueError(f"Unsupported mode: {mode}")

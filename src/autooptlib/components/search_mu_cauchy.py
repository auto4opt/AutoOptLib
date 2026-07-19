"""Python translation of search_mu_cauchy."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng

_KEY = "cauchy_eta"


def _get_eta(aux: object):
    if aux is None:
        return None
    if isinstance(aux, dict):
        return aux.get(_KEY)
    return getattr(aux, _KEY, None)


def _set_eta(aux: object, value: np.ndarray) -> None:
    if aux is None:
        return
    if isinstance(aux, dict):
        aux[_KEY] = value
    else:
        setattr(aux, _KEY, value)


def search_mu_cauchy(*args):
    mode = args[-1]
    if mode == "execute":
        parent = args[0]
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, args[1] if len(args) > 1 else None)
        inner_g = int(args[5]) if len(args) > 5 else 1

        if not isinstance(parent, (np.ndarray, list, tuple)):
            decs = getattr(parent, "decs", None)
            parent = (
                decs() if callable(decs) else (decs if decs is not None else parent)
            )
        parent = np.asarray(parent, dtype=float)
        n, d = parent.shape

        eta = None
        if inner_g != 1:
            eta = _get_eta(aux)
            if eta is not None and np.shape(eta) != (n, d):
                eta = None
        if eta is None:
            eta = rng.random((n, d))
            _set_eta(aux, eta)

        disturb = eta * rng.standard_cauchy(size=(n, d))
        offspring = parent + disturb

        tau1 = 1.0 / np.sqrt(2.0 * np.sqrt(d))
        tau2 = 1.0 / np.sqrt(2.0 * d)
        normal = rng.standard_normal((n, 1)).repeat(d, axis=1)
        normal_j = rng.standard_normal((n, d))
        eta = eta * np.exp(tau2 * normal + tau1 * normal_j)
        _set_eta(aux, eta)
        return offspring, aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", "GS"], None

    raise ValueError(f"Unsupported mode: {mode}")

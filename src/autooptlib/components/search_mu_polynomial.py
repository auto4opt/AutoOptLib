"""Python translation of search_mu_polynomial."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng


def search_mu_polynomial(*args):
    mode = args[-1]
    if mode == "execute":
        parent = args[0]
        problem = args[1] if len(args) > 1 else None
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux, problem)

        if not isinstance(parent, (np.ndarray, list, tuple)):
            decs = getattr(parent, "decs", None)
            offspring = (
                decs() if callable(decs) else (decs if decs is not None else parent)
            )
        else:
            offspring = parent
        offspring = np.asarray(offspring, dtype=float)
        n, d = offspring.shape

        prob_m = 0.1
        dis_m = 30.0
        if para is not None:
            arr = np.asarray(para).reshape(-1)
            if arr.size > 0:
                prob_m = float(arr[0])
            if arr.size > 1:
                dis_m = float(arr[1])

        bound = getattr(problem, "bound")
        lower = np.asarray(bound[0], dtype=float)
        upper = np.asarray(bound[1], dtype=float)
        lower = np.repeat(lower.reshape(1, -1), n, axis=0)
        upper = np.repeat(upper.reshape(1, -1), n, axis=0)

        site = rng.random((n, d)) < prob_m
        mu = rng.random((n, d))
        temp = site & (mu <= 0.5)
        if np.any(temp):
            offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (
                (
                    2.0 * mu[temp]
                    + (1.0 - 2.0 * mu[temp])
                    * (
                        1.0
                        - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])
                    )
                    ** (dis_m + 1.0)
                )
                ** (1.0 / (dis_m + 1.0))
                - 1.0
            )
        temp = site & (mu > 0.5)
        if np.any(temp):
            offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (
                1.0
                - (
                    2.0 * (1.0 - mu[temp])
                    + 2.0
                    * (mu[temp] - 0.5)
                    * (
                        1.0
                        - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])
                    )
                    ** (dis_m + 1.0)
                )
                ** (1.0 / (dis_m + 1.0))
            )
        return offspring, aux

    if mode == "parameter":
        return [[0, 0.3], [20, 40]], None

    if mode == "behavior":
        return [["LS", "small", "large"], ["GS", "large", "small"]], None

    if mode == "algorithm":
        parent = args[0]
        bound = np.asarray(args[1], dtype=float)
        prob_m = 1.0
        dis_m = 30.0
        offspring = np.asarray(parent, dtype=float)
        rng = ensure_rng(args[2] if len(args) > 3 else None)
        n, d = offspring.shape
        lower = np.repeat(bound[0].reshape(1, -1), n, axis=0)
        upper = np.repeat(bound[1].reshape(1, -1), n, axis=0)
        site = rng.random((n, d)) < prob_m
        mu = rng.random((n, d))
        temp = site & (mu <= 0.5)
        if np.any(temp):
            offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (
                (
                    2.0 * mu[temp]
                    + (1.0 - 2.0 * mu[temp])
                    * (
                        1.0
                        - (offspring[temp] - lower[temp]) / (upper[temp] - lower[temp])
                    )
                    ** (dis_m + 1.0)
                )
                ** (1.0 / (dis_m + 1.0))
                - 1.0
            )
        temp = site & (mu > 0.5)
        if np.any(temp):
            offspring[temp] = offspring[temp] + (upper[temp] - lower[temp]) * (
                1.0
                - (
                    2.0 * (1.0 - mu[temp])
                    + 2.0
                    * (mu[temp] - 0.5)
                    * (
                        1.0
                        - (upper[temp] - offspring[temp]) / (upper[temp] - lower[temp])
                    )
                    ** (dis_m + 1.0)
                )
                ** (1.0 / (dis_m + 1.0))
            )
        return offspring, None

    raise ValueError(f"Unsupported mode: {mode}")

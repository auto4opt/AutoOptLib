"""Python translation of search_cma."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from ._utils import ensure_rng, flex_get


def _extract_decs(parent_obj: Any) -> np.ndarray:
    if isinstance(parent_obj, np.ndarray):
        return parent_obj
    decs = flex_get(parent_obj, "decs")
    if callable(decs):
        return np.asarray(decs(), dtype=float)
    if decs is not None:
        return np.asarray(decs, dtype=float)
    # assume sequence of Solution objects
    return np.vstack(
        [np.asarray(flex_get(item, "dec"), dtype=float) for item in parent_obj]
    )


def _ensure_aux(aux: Any) -> dict:
    if aux is None or not isinstance(aux, dict):
        return {}
    return aux


def _regularize_covariance(matrix: np.ndarray) -> np.ndarray:
    """Return a finite, symmetric positive-definite covariance matrix."""
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-12, 1e6)
    # Some macOS Accelerate builds emit spurious floating-point warnings for
    # otherwise finite matrix products.  The result is validated immediately.
    with np.errstate(all="ignore"):
        result = (eigvecs * eigvals) @ eigvecs.T
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)


def _init_cma(
    aux: dict,
    n: int,
    d: int,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
) -> None:
    half_n = max(1, int(np.floor(n / 2 + 0.5)))
    w = np.log(half_n + 0.5) - np.log(np.arange(1, half_n + 1))
    w = w / np.sum(w)
    better_n = 1.0 / np.sum(w**2)

    aux.setdefault("cma_halfN", half_n)
    aux.setdefault("cma_w", w)
    aux.setdefault("cma_betterN", better_n)
    aux.setdefault("cma_csigma", (better_n + 2) / (d + better_n + 5))
    aux.setdefault(
        "cma_dsigma",
        aux["cma_csigma"] + 2 * max(np.sqrt((better_n - 1) / (d + 1)) - 1, 0) + 1,
    )
    aux.setdefault("cma_chiN", np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d * d)))
    aux.setdefault("cma_cc", (4 + better_n / d) / (4 + d + 2 * better_n / d))
    aux.setdefault("cma_ccov", 2 / ((d + 1.3) ** 2 + better_n))
    aux.setdefault(
        "cma_cmu",
        min(
            1 - aux["cma_ccov"],
            2 * (better_n - 2 + 1 / better_n) / ((d + 2) ** 2 + 2 * better_n / 2),
        ),
    )
    aux.setdefault("cma_hth", (1.4 + 2 / (d + 1)) * aux["cma_chiN"])
    aux.setdefault("cma_mean", lower + (upper - lower) * rng.random(d))
    aux.setdefault("cma_ps", np.zeros(d))
    aux.setdefault("cma_pc", np.zeros(d))
    aux.setdefault("cma_C", np.eye(d))
    aux.setdefault("cma_sigma", 0.1 * (upper - lower))


def _sample(
    aux: dict, n: int, d: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(aux["cma_mean"], dtype=float)
    sigma = np.asarray(aux["cma_sigma"], dtype=float)
    C = _regularize_covariance(aux["cma_C"])
    sigma = np.clip(
        np.nan_to_num(sigma, nan=1e-3, posinf=1e6, neginf=1e-12), 1e-12, 1e6
    )
    aux["cma_C"] = C
    aux["cma_sigma"] = sigma
    with np.errstate(all="ignore"):
        disturbance = rng.multivariate_normal(np.zeros(d), C, size=n)
    offspring = mean + sigma * disturbance
    aux["cma_Disturb"] = disturbance
    return offspring, aux


def search_cma(*args):
    mode = args[-1]
    if mode == "execute":
        parent_obj = args[0]
        problem = args[1]
        aux = _ensure_aux(args[3] if len(args) > 3 else None)
        rng = ensure_rng(aux, problem)
        parent = _extract_decs(parent_obj)
        n, d = parent.shape
        bound = flex_get(problem, "bound")
        lower = np.asarray(bound[0], dtype=float)
        upper = np.asarray(bound[1], dtype=float)
        _init_cma(aux, n, d, lower, upper, rng)
        offspring, aux = _sample(aux, n, d, rng)
        return offspring, aux

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", "GS"], None

    if mode == "algorithm":
        parent = np.asarray(args[0], dtype=float)
        bound = np.asarray(args[1], dtype=float)
        aux = _ensure_aux(args[2] if len(args) > 2 else None)
        rng = ensure_rng(aux)
        n, d = parent.shape
        lower = bound[0]
        upper = bound[1]
        _init_cma(aux, n, d, lower, upper, rng)
        offspring, aux = _sample(aux, n, d, rng)
        return offspring, aux

    raise ValueError(f"Unsupported mode: {mode}")

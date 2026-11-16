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
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("CMA covariance must be a non-empty square matrix.")
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    # Constrain large but finite state too; otherwise ``matrix + matrix.T``
    # can overflow before the eigenspectrum is clipped.
    matrix = np.clip(matrix, -1e6, 1e6)
    matrix = 0.5 * (matrix + matrix.T)
    try:
        eigvals, eigvecs = np.linalg.eigh(matrix)
    except np.linalg.LinAlgError:
        # A single numerically damaged algorithm state must not abort the
        # outer algorithm-design experiment. Reset only that CMA covariance;
        # all other component and Search state remains intact.
        return np.eye(matrix.shape[0], dtype=float)
    eigvals = np.nan_to_num(eigvals, nan=1.0, posinf=1e6, neginf=1e-12)
    largest = float(np.clip(np.max(eigvals), 1e-12, 1e6))
    # Keep the condition number within the range where double-precision
    # eigendecomposition and triangular solves remain reliable.
    floor = max(1e-12, largest * 1e-12)
    eigvals = np.clip(eigvals, floor, 1e6)
    # Some macOS Accelerate builds emit spurious floating-point warnings for
    # otherwise finite matrix products.  The result is validated immediately.
    with np.errstate(all="ignore"):
        result = (eigvecs * eigvals) @ eigvecs.T
    result = np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)
    result = 0.5 * (result + result.T)
    if not np.all(np.isfinite(result)):
        return np.eye(matrix.shape[0], dtype=float)
    return result


def _init_cma(
    aux: dict,
    n: int,
    d: int,
    lower: np.ndarray,
    upper: np.ndarray,
    rng: np.random.Generator,
    initial_mean: np.ndarray | None = None,
) -> None:
    expected_shapes = {
        "cma_mean": (d,),
        "cma_sigma": (d,),
        "cma_ps": (d,),
        "cma_pc": (d,),
        "cma_C": (d, d),
        "cma_lower": (d,),
        "cma_upper": (d,),
    }
    incompatible = False
    for name, shape in expected_shapes.items():
        if name in aux and np.asarray(aux[name]).shape != shape:
            incompatible = True
            break
    if "cma_Disturb" in aux:
        disturbance = np.asarray(aux["cma_Disturb"])
        incompatible = (
            incompatible or disturbance.ndim != 2 or disturbance.shape[1] != d
        )
    incompatible = incompatible or (
        "cma_lambda" in aux and int(aux["cma_lambda"]) != int(n)
    )
    if incompatible:
        # Graph mutations and survivor reordering can change the number and
        # meaning of tunable parameters. A CMA distribution is meaningful only
        # for the exact parameter vector that created it.
        for name in tuple(aux):
            if str(name).startswith("cma_"):
                aux.pop(name, None)

    half_n = max(1, int(np.floor(n / 2 + 0.5)))
    w = np.log(half_n + 0.5) - np.log(np.arange(1, half_n + 1))
    w = w / np.sum(w)
    better_n = 1.0 / np.sum(w**2)

    aux.setdefault("cma_lambda", int(n))
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
    if initial_mean is None:
        initial_mean = lower + (upper - lower) * rng.random(d)
    initial_mean = np.asarray(initial_mean, dtype=float).reshape(d)
    midpoint = lower + 0.5 * (upper - lower)
    initial_mean = np.where(np.isfinite(initial_mean), initial_mean, midpoint)
    initial_mean = np.clip(initial_mean, lower, upper)
    aux.setdefault("cma_mean", initial_mean)
    aux.setdefault("cma_ps", np.zeros(d))
    aux.setdefault("cma_pc", np.zeros(d))
    aux.setdefault("cma_C", np.eye(d))
    aux.setdefault("cma_sigma", 0.1 * (upper - lower))
    aux.setdefault("cma_generation", 0)
    # Bounds are refreshed even for a compatible persisted state because a
    # problem instance may retain the dimension while changing its domain.
    aux["cma_lower"] = np.asarray(lower, dtype=float).copy()
    aux["cma_upper"] = np.asarray(upper, dtype=float).copy()


def _sample(
    aux: dict, n: int, d: int, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(aux["cma_mean"], dtype=float)
    sigma = np.asarray(aux["cma_sigma"], dtype=float)
    C = _regularize_covariance(aux["cma_C"])
    lower = np.asarray(aux.get("cma_lower", np.full(d, -1e6)), dtype=float)
    upper = np.asarray(aux.get("cma_upper", np.full(d, 1e6)), dtype=float)
    midpoint = lower + 0.5 * (upper - lower)
    mean = np.where(np.isfinite(mean), mean, midpoint)
    mean = np.clip(mean, lower, upper)
    sigma = np.clip(
        np.nan_to_num(sigma, nan=1e-3, posinf=1e6, neginf=1e-12), 1e-12, 1e6
    )
    aux["cma_C"] = C
    aux["cma_mean"] = mean
    aux["cma_sigma"] = sigma
    # NumPy's multivariate_normal defaults to an SVD. Very ill-conditioned
    # but already regularized CMA matrices can still make that SVD fail. Draw
    # standard normals and apply a clipped symmetric eigen factor directly.
    try:
        eigvals, eigvecs = np.linalg.eigh(C)
    except np.linalg.LinAlgError:
        C = np.eye(d, dtype=float)
        eigvals = np.ones(d, dtype=float)
        eigvecs = np.eye(d, dtype=float)
        aux["cma_C"] = C
    eigvals = np.clip(np.nan_to_num(eigvals, nan=1.0), 1e-12, 1e6)
    factor = eigvecs * np.sqrt(eigvals)
    with np.errstate(all="ignore"):
        disturbance = rng.standard_normal(size=(n, d)) @ factor.T
    if not np.all(np.isfinite(disturbance)):
        # This is a last-resort, local CMA restart. It preserves the candidate
        # evaluation ledger instead of turning a recoverable component state
        # into a failed outer Search run.
        aux["cma_C"] = np.eye(d, dtype=float)
        aux["cma_ps"] = np.zeros(d, dtype=float)
        aux["cma_pc"] = np.zeros(d, dtype=float)
        disturbance = rng.standard_normal(size=(n, d))
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
        context = aux.get("_population_context")
        model_parent = parent if context is None else _extract_decs(context)
        n, d = parent.shape
        model_n = int(aux.get("_stream_generation_target", n))
        bound = flex_get(problem, "bound")
        lower = np.asarray(bound[0], dtype=float)
        upper = np.asarray(bound[1], dtype=float)
        _init_cma(
            aux,
            model_n,
            d,
            lower,
            upper,
            rng,
            initial_mean=np.mean(model_parent, axis=0) if context is not None else None,
        )
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
        _init_cma(
            aux,
            n,
            d,
            lower,
            upper,
            rng,
            initial_mean=np.mean(parent, axis=0),
        )
        offspring, aux = _sample(aux, n, d, rng)
        return offspring, aux

    raise ValueError(f"Unsupported mode: {mode}")

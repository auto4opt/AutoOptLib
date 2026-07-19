"""Python translation of para_cma."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import flex_get


def _regularize_covariance(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=1e6, neginf=-1e6)
    matrix = 0.5 * (matrix + matrix.T)
    eigvals, eigvecs = np.linalg.eigh(matrix)
    eigvals = np.clip(eigvals, 1e-12, 1e6)
    with np.errstate(all="ignore"):
        result = (eigvecs * eigvals) @ eigvecs.T
    return np.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6)


def _extract_fitness(obj: Any, mode: str) -> np.ndarray:
    if mode == "solution":
        fits = flex_get(obj, "fits")
        if callable(fits):
            return np.asarray(fits()).reshape(-1)
        if fits is not None:
            return np.asarray(fits).reshape(-1)
        raise ValueError("Solution must expose fits() for para_cma")
    if mode == "algorithm":
        vals = flex_get(obj, "avePerformAll")
        if callable(vals):
            return np.asarray(vals()).reshape(-1)
        if vals is not None:
            return np.asarray(vals).reshape(-1)
        raise ValueError("Algorithm must expose avePerformAll for para_cma")
    raise ValueError(f"Unknown type {mode}")


def para_cma(*args):
    solution = args[0]
    problem = args[1]
    aux = args[2] if len(args) > 2 else {}
    mode = args[3] if len(args) > 3 else "solution"

    if aux is None:
        aux = {}
    disturbance = np.asarray(aux.get("cma_Disturb"))
    if disturbance.size == 0:
        return aux

    half_n = int(aux.get("cma_halfN"))
    w = np.asarray(aux.get("cma_w"))
    better_n = float(aux.get("cma_betterN"))
    mean = np.asarray(aux.get("cma_mean"), dtype=float)
    sigma = np.asarray(aux.get("cma_sigma"), dtype=float)
    csigma = float(aux.get("cma_csigma"))
    dsigma = float(aux.get("cma_dsigma"))
    chi_n = float(aux.get("cma_chiN"))
    cc = float(aux.get("cma_cc"))
    ccov = float(aux.get("cma_ccov"))
    cmu = float(aux.get("cma_cmu"))
    hth = float(aux.get("cma_hth"))
    ps = np.asarray(aux.get("cma_ps"), dtype=float)
    pc = np.asarray(aux.get("cma_pc"), dtype=float)
    C = _regularize_covariance(aux.get("cma_C"))

    fitness = _extract_fitness(solution, mode)
    rank = np.argsort(fitness)
    disturbance = disturbance[rank]
    selected_n = min(half_n, len(w), len(disturbance))
    if selected_n == 0:
        return aux
    selected_w = np.asarray(w[:selected_n], dtype=float)
    selected_w /= np.sum(selected_w)
    # The last generation may be deliberately truncated to respect ProbFE.
    # Recompute the effective selection mass for that smaller sample.
    better_n = 1.0 / np.sum(selected_w**2)
    with np.errstate(all="ignore"):
        disturbance_w = selected_w @ disturbance[:selected_n]
    disturbance_w = np.nan_to_num(disturbance_w, nan=0.0, posinf=1e6, neginf=-1e6)
    mean = mean + sigma * disturbance_w

    try:
        chol_c = np.linalg.cholesky(C)
        inv_chol_T = np.linalg.solve(chol_c.T, disturbance_w)
    except np.linalg.LinAlgError:
        chol_c = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]))
        inv_chol_T = np.linalg.solve(chol_c.T, disturbance_w)

    ps = (1 - csigma) * ps + np.sqrt(csigma * (2 - csigma) * better_n) * inv_chol_T
    gmax = int(
        flex_get(
            problem[0] if isinstance(problem, (list, tuple)) else problem, "Gmax", 1
        )
    )
    denom = np.sqrt(1 - (1 - csigma) ** (2 * (gmax + 1)))
    hs = float(np.linalg.norm(ps) / denom < hth)
    pc = (1 - cc) * pc + hs * np.sqrt(cc * (2 - cc) * better_n) * disturbance_w
    delta = (1 - hs) * cc * (2 - cc)
    C = (1 - ccov - cmu) * C + ccov * (np.outer(pc, pc) + delta * C)
    for i in range(selected_n):
        C += cmu * selected_w[i] * np.outer(disturbance[i], disturbance[i])
    C = _regularize_covariance(C)

    exponent = 0.3 * csigma / dsigma * (np.linalg.norm(ps) / chi_n - 1)
    sigma = sigma * np.exp(np.clip(exponent, -20.0, 20.0))
    sigma = np.clip(
        np.nan_to_num(sigma, nan=1e-3, posinf=1e6, neginf=1e-12), 1e-12, 1e6
    )

    target = problem[0] if isinstance(problem, (list, tuple)) else problem
    bound = flex_get(target, "bound", None)
    if bound is not None and mode == "solution":
        bound = np.asarray(bound, dtype=float)
        mean = np.clip(mean, bound[0], bound[1])

    aux.update(
        {
            "cma_mean": mean,
            "cma_ps": ps,
            "cma_pc": pc,
            "cma_C": C,
            "cma_sigma": sigma,
        }
    )
    return aux

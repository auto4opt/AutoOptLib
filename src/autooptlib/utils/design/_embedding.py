"""Algorithm embedding utilities (translation of MATLAB Embedding.m)."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.nan_to_num(np.asarray(vec, dtype=float), nan=0.0, posinf=1e6, neginf=-1e6)
    vmin = np.min(vec)
    vmax = np.max(vec)
    if np.isclose(vmax, vmin):
        return np.zeros_like(vec, dtype=float)
    return (vec - vmin) / (vmax - vmin)


def _extract_parameter_values(parameter_entry: Any, max_len: int = 2) -> np.ndarray:
    if parameter_entry is None:
        return np.zeros(max_len, dtype=float)
    if isinstance(parameter_entry, (list, tuple)):
        values = parameter_entry[0] if parameter_entry else None
    else:
        values = parameter_entry
    if values is None:
        return np.zeros(max_len, dtype=float)
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size >= max_len:
        return arr[:max_len]
    result = np.zeros(max_len, dtype=float)
    result[: arr.size] = arr
    return result


def _vectorize_single_path(alg: Any, setting: Any, rand_seed: np.ndarray) -> np.ndarray:
    alg_q = int(getattr(setting, "AlgQ", getattr(setting, "alg_q", 0)))
    matrix = np.asarray(alg.operator[0], dtype=int)
    operator = np.zeros(alg_q + 2, dtype=float)
    operator[0] = matrix[0, 0] if matrix.size else 0
    if matrix.shape[0] > 1:
        operator[1 : matrix.shape[0]] = matrix[1:, 0]
    operator[-1] = matrix[-1, 1] if matrix.size else 0

    parameter = np.zeros((2, alg_q + 2), dtype=float)
    for idx, op_idx in enumerate(operator.astype(int)):
        if op_idx <= 0:
            continue
        if op_idx - 1 < len(alg.parameter):
            entry = alg.parameter[op_idx - 1]
        else:
            entry = None
        parameter[:, idx] = _extract_parameter_values(entry, max_len=2)

    # MATLAB linearizes matrices column-by-column.  The embedding learned for
    # the reference implementation therefore depends on Fortran ordering.
    parameter_flat = parameter.reshape(1, -1, order="F")
    operator_norm = _normalize(operator)
    vector = np.concatenate([operator_norm, parameter_flat.ravel()])
    shuffled = vector[rand_seed]
    return shuffled.reshape(len(rand_seed) // 3, 3)


def _vectorize_multi_path(alg: Any, setting: Any, rand_seed: np.ndarray) -> np.ndarray:
    alg_p = int(getattr(setting, "AlgP", getattr(setting, "alg_p", 0)))
    operator = np.zeros(alg_p + 2, dtype=float)
    first_path = np.asarray(alg.operator[0], dtype=int)
    operator[0] = first_path[0, 0] if first_path.size else 0
    for j in range(alg_p):
        path_matrix = np.asarray(alg.operator[j], dtype=int)
        if path_matrix.shape[0] >= 2:
            operator[j + 1] = path_matrix[1, 0]
    operator[-1] = first_path[-1, 1] if first_path.size else 0

    parameter = np.zeros((2, alg_p + 2), dtype=float)
    choose_idx = int(operator[0])
    if choose_idx > 0 and choose_idx - 1 < len(alg.parameter):
        parameter[:, 0] = _extract_parameter_values(
            alg.parameter[choose_idx - 1], max_len=2
        )
    for j in range(alg_p):
        op_idx = int(operator[j + 1])
        if op_idx > 0 and op_idx - 1 < len(alg.parameter):
            parameter[:, j + 1] = _extract_parameter_values(
                alg.parameter[op_idx - 1], max_len=2
            )
    update_idx = int(operator[-1])
    if update_idx > 0 and update_idx - 1 < len(alg.parameter):
        parameter[:, -1] = _extract_parameter_values(
            alg.parameter[update_idx - 1], max_len=2
        )

    parameter_flat = parameter.reshape(1, -1, order="F")
    lower = 1.0
    upper = float(max(len(alg.parameter), 1))
    if np.isclose(upper, lower):
        operator_norm = np.zeros_like(operator, dtype=float)
    else:
        operator_norm = (operator - lower) / (upper - lower)
    vector = np.concatenate([operator_norm, parameter_flat.ravel()])
    shuffled = vector[rand_seed]
    return shuffled.reshape(len(rand_seed) // 3, 3)


def _ensure_rand_seed(seed: Sequence[int]) -> np.ndarray:
    arr = np.asarray(seed, dtype=int)
    if arr.min() < 0:
        raise ValueError("rand_seed indices must be non-negative")
    return arr


def embedding(algs: Sequence[Any], setting: Any, surrogate: Any, mode: str) -> Any:
    """Embed algorithms via marginalized stacked denoising autoencoder."""
    alg_list = list(algs)
    if not alg_list:
        return np.zeros((0, 0))

    rand_seed = _ensure_rand_seed(getattr(surrogate, "rand_seed"))
    layer = 2
    p = 0.3

    vectors = []
    if int(getattr(setting, "AlgP", getattr(setting, "alg_p", 0))) == 1:
        for alg in alg_list:
            vectors.append(_vectorize_single_path(alg, setting, rand_seed))
    else:
        for alg in alg_list:
            vectors.append(_vectorize_multi_path(alg, setting, rand_seed))

    VectorAlgs = np.hstack(vectors)

    if mode == "get":
        embed_map, _ = _msda(VectorAlgs, p, layer)
        return embed_map
    if mode == "use":
        embed_map = getattr(surrogate, "embedding", None)
        if embed_map is None:
            raise ValueError("Surrogate does not provide a precomputed embedding map")
        temp = np.vstack([VectorAlgs, np.ones((1, VectorAlgs.shape[1]))])
        with np.errstate(all="ignore"):
            wx = embed_map[:, :, 0] @ temp
        wx = np.nan_to_num(wx, nan=0.0, posinf=1e6, neginf=-1e6)
        rows = len(rand_seed) // 3
        argslist = np.zeros((rows, len(alg_list)))
        for i in range(len(alg_list)):
            block = wx[:, 3 * i : 3 * i + 3]
            argslist[:, i] = np.sum(block, axis=1) / 3.0
        wx = np.tanh(argslist)
        for i in range(1, embed_map.shape[2]):
            wx = np.vstack([wx, np.ones((1, wx.shape[1]))])
            with np.errstate(all="ignore"):
                wx = embed_map[:, :, i] @ wx
            wx = np.nan_to_num(wx, nan=0.0, posinf=1e6, neginf=-1e6)
            wx = np.tanh(wx)
        return wx.T
    raise ValueError(f"Unsupported mode: {mode}")


def _msda(x: np.ndarray, p: float, layers: int):
    d, n = x.shape
    ws = np.zeros((d, d + 1, layers))
    hs = np.zeros((d, n // 3, layers + 1))
    w, h = _mda_de(x, p)
    ws[:, :, 0] = w
    hs[:, :, 1] = h
    last_h = h
    for t in range(1, layers):
        w_t, h_t = _mda(last_h, p)
        ws[:, :, t] = w_t
        hs[:, :, t + 1] = h_t
        last_h = h_t
    return ws, hs


def _mda_de(x: np.ndarray, p: float):
    x_aug = np.vstack([x, np.ones((1, x.shape[1]))])
    d_aug = x_aug.shape[0]
    q = np.concatenate([np.full(d_aug - 1, 1 - p), [1.0]])
    with np.errstate(all="ignore"):
        s = x_aug @ x_aug.T
    s = np.nan_to_num(s, nan=0.0, posinf=1e12, neginf=-1e12)
    q_outer = s * (q[:, None] * q[None, :])
    diag_idx = np.arange(d_aug)
    q_outer[diag_idx, diag_idx] = q * np.diag(s)
    p_mat = s * q[np.newaxis, :]
    reg = q_outer + 1e-5 * np.eye(d_aug)
    with np.errstate(all="ignore"):
        w = p_mat[:-1, :] @ np.linalg.pinv(reg)
    w = np.nan_to_num(w, nan=0.0, posinf=1e6, neginf=-1e6)

    length = x_aug.shape[1] // 3
    h = np.zeros((d_aug - 1, length))
    with np.errstate(all="ignore"):
        wx = w @ x_aug
    wx = np.nan_to_num(wx, nan=0.0, posinf=1e6, neginf=-1e6)
    for i in range(length):
        block = wx[:, 3 * i : 3 * i + 3]
        h[:, i] = np.sum(block, axis=1) / 3.0
    return w, np.tanh(h)


def _mda(x: np.ndarray, p: float):
    x_aug = np.vstack([x, np.ones((1, x.shape[1]))])
    d_aug = x_aug.shape[0]
    q = np.concatenate([np.full(d_aug - 1, 1 - p), [1.0]])
    with np.errstate(all="ignore"):
        s = x_aug @ x_aug.T
    s = np.nan_to_num(s, nan=0.0, posinf=1e12, neginf=-1e12)
    q_outer = s * (q[:, None] * q[None, :])
    diag_idx = np.arange(d_aug)
    q_outer[diag_idx, diag_idx] = q * np.diag(s)
    p_mat = s * q[np.newaxis, :]
    reg = q_outer + 1e-5 * np.eye(d_aug)
    with np.errstate(all="ignore"):
        w = p_mat[:-1, :] @ np.linalg.pinv(reg)
    w = np.nan_to_num(w, nan=0.0, posinf=1e6, neginf=-1e6)
    with np.errstate(all="ignore"):
        projected = w @ x_aug
    projected = np.nan_to_num(projected, nan=0.0, posinf=1e6, neginf=-1e6)
    h = np.tanh(projected)
    return w, h

from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from ...components import get_component
from ._helpers import ensure_rng, get_flex


def _as_int(value: Any) -> int:
    arr = np.asarray(value)
    if arr.size == 0:
        raise ValueError("Cannot convert empty array to int")
    return int(arr.reshape(-1)[0])


def _ensure_alg_list(algs: Any) -> List[Any]:
    if isinstance(algs, Sequence) and not isinstance(algs, (str, bytes)):
        return list(algs)
    return [algs]


def _copy_parameter_structure(parameter: Sequence[Any]) -> List[List[Any]]:
    copied: List[List[Any]] = []
    for entry in parameter:
        if entry is None:
            copied.append([None, None])
            continue
        values = entry[0] if len(entry) > 0 else None
        behavior = entry[1] if len(entry) > 1 else None
        values_copy = None if values is None else np.array(values, copy=True)
        copied.append([values_copy, behavior])
    return copied


def _gather_operator_positions(
    paths: List[np.ndarray],
) -> Tuple[List[Tuple[str, int, int]], List[int]]:
    positions: List[Tuple[str, int, int]] = []
    values: List[int] = []
    if not paths or paths[0].size == 0:
        return positions, values
    # Choose and update are shared by all pathways in the reference encoding;
    # only search operators are pathway-specific.
    positions.append(("choose", 0, 0))
    values.append(_as_int(paths[0][0, 0]))
    for path_idx, path in enumerate(paths):
        if path.size == 0:
            continue
        for row in range(1, path.shape[0]):
            positions.append(("search", path_idx, row))
            values.append(_as_int(path[row, 0]))
    positions.append(("update", 0, paths[0].shape[0] - 1))
    values.append(_as_int(paths[0][-1, -1]))
    return positions, values


def _disturb_search_single(
    path: np.ndarray,
    row_idx: int,
    rng: np.random.Generator,
    op_space: np.ndarray,
    alg_q: int,
) -> np.ndarray:
    pool = np.arange(op_space[1, 0], op_space[1, 1] + 1)
    current = _as_int(path[row_idx, 0])
    pool = pool[pool != current]
    num_search = path.shape[0] - 1
    if pool.size == 0:
        ind_new = current
    else:
        ind_new = _as_int(rng.choice(pool))
    if num_search == 1 and num_search < alg_q:
        sample_pool = list(pool) + [np.inf]
    elif num_search > 1 and num_search < alg_q:
        sample_pool = list(pool) + [np.inf, -np.inf]
    elif num_search > 1 and num_search == alg_q:
        sample_pool = list(pool) + [-np.inf]
    else:
        sample_pool = list(pool)
    sample_choice = rng.choice(sample_pool) if sample_pool else ind_new
    if sample_choice == np.inf:
        insert_row = np.zeros((1, 2), dtype=int)
        insert_row[0, 0] = ind_new
        insert_row[0, 1] = path[row_idx, 1]
        path = np.insert(path, row_idx + 1, insert_row, axis=0)
        path[row_idx, 1] = ind_new
    elif sample_choice == -np.inf and path.shape[0] > 2:
        path[row_idx - 1, 1] = path[row_idx, 1]
        path = np.delete(path, row_idx, axis=0)
    else:
        path[row_idx, 0] = ind_new
        path[row_idx - 1, 1] = ind_new
    return path


def _disturb_search_multi(
    path: np.ndarray, rng: np.random.Generator, op_space: np.ndarray, alg_q: int
) -> np.ndarray:
    choose_idx = _as_int(path[0, 0])
    update_idx = _as_int(path[-1, -1])
    curr_q = int(rng.integers(1, alg_q + 1))
    new_path = np.zeros((curr_q + 1, 2), dtype=int)
    start = int(rng.integers(op_space[1, 0], op_space[1, 1] + 1))
    new_path[0, 0] = choose_idx
    new_path[0, 1] = start
    prev = start
    for row in range(1, curr_q):
        nxt = int(rng.integers(op_space[1, 0], op_space[1, 1] + 1))
        new_path[row, 0] = prev
        new_path[row, 1] = nxt
        prev = nxt
    new_path[-1, 0] = prev
    new_path[-1, 1] = update_idx
    return new_path


def disturb(
    algs: Any, setting: Any, inner_g: int, aux: Any
) -> Tuple[List[List[np.ndarray]], List[List[List[Any]]], Any]:
    alg_list = _ensure_alg_list(algs)
    rng = ensure_rng(setting)

    op_space = np.asarray(get_flex(setting, "op_space", required=True), dtype=int)
    para_space = list(get_flex(setting, "para_space", required=True))
    para_local_space = list(
        get_flex(setting, "para_local_space", default=[None] * len(para_space))
    )
    alg_p = int(get_flex(setting, "alg_p", required=True))
    alg_q = int(get_flex(setting, "alg_q", required=True))
    alg_n = int(get_flex(setting, "alg_n", 1))
    tune_para = bool(get_flex(setting, "tune_para", False))

    ind_non_empty_para = [
        idx
        for idx, space in enumerate(para_space)
        if space is not None and len(space) > 0
    ]

    aux_list: List[Any]
    if aux is None:
        aux_list = [{} for _ in alg_list]
    elif isinstance(aux, Sequence):
        aux_list = list(aux)
        while len(aux_list) < len(alg_list):
            aux_list.append({})
    else:
        aux_list = [aux] + [{} for _ in range(len(alg_list) - 1)]

    new_ops: List[List[np.ndarray]] = []
    new_paras: List[List[List[Any]]] = []

    for idx, alg in enumerate(alg_list):
        this_op_raw = getattr(alg, "operator", None)
        this_para_raw = getattr(alg, "parameter", None)
        if this_op_raw is None or this_para_raw is None:
            raise ValueError(
                "Algorithm must provide operator and parameter fields for disturb()."
            )

        paths = [np.array(path, copy=True) for path in this_op_raw]
        this_para = _copy_parameter_structure(this_para_raw)

        op_positions, op_values = _gather_operator_positions(paths)
        total_ops = len(op_positions)
        ind_para = [p for p in ind_non_empty_para if (p + 1) in op_values]

        aux_entry = aux_list[idx] if isinstance(aux_list[idx], dict) else {}

        if inner_g == 1:
            num_choices = total_ops + 1
            if not tune_para and num_choices > 0:
                unit = 1.0 / max(1, total_ops + len(ind_para))
                probs = np.concatenate(
                    [np.full(total_ops, unit), np.array([unit * len(ind_para)])]
                )
                if op_space[0, 0] == op_space[0, 1] and total_ops:
                    probs[0] = 0.0
                if op_space[1, 0] == op_space[1, 1] and total_ops > 2:
                    probs[1:-1] = 0.0
                if op_space[2, 0] == op_space[2, 1] and total_ops:
                    probs[total_ops - 1] = 0.0
                if probs.sum() > 0:
                    probs /= probs.sum()
                    seed = int(rng.choice(num_choices, p=probs)) + 1
                else:
                    seed = total_ops + 1
            elif tune_para and ind_para:
                seed = total_ops + 1
            else:
                seed = total_ops + 1
            aux_entry["seed"] = seed
        else:
            seed = int(aux_entry.get("seed", total_ops + 1))

        if seed <= total_ops and total_ops > 0:
            category, path_idx, row_idx = op_positions[seed - 1]
            path = paths[path_idx]
            if category == "choose":
                pool = np.arange(op_space[0, 0], op_space[0, 1] + 1)
                current = _as_int(path[0, 0])
                pool = pool[pool != current]
                if pool.size > 0:
                    new_idx = _as_int(rng.choice(pool))
                    for shared_path in paths:
                        shared_path[0, 0] = new_idx
            elif category == "update":
                pool = np.arange(op_space[2, 0], op_space[2, 1] + 1)
                current = _as_int(path[-1, -1])
                pool = pool[pool != current]
                if pool.size > 0:
                    new_idx = _as_int(rng.choice(pool))
                    for shared_path in paths:
                        shared_path[-1, -1] = new_idx
            else:  # search
                if alg_p == 1:
                    path = _disturb_search_single(path, row_idx, rng, op_space, alg_q)
                else:
                    path = _disturb_search_multi(path, rng, op_space, alg_q)
                paths[path_idx] = path
        else:
            if not ind_para:
                seed = total_ops + 1
            active_parameters: list[tuple[int, np.ndarray, np.ndarray]] = []
            for para_idx in ind_para:
                entry = this_para[para_idx]
                values = entry[0]
                if values is None:
                    continue
                behavior = entry[1]
                space = (
                    para_local_space[para_idx]
                    if (behavior != "GS" and para_local_space[para_idx] is not None)
                    else para_space[para_idx]
                )
                if space is None:
                    continue
                bounds = np.asarray(space, dtype=float)
                if bounds.ndim == 1:
                    bounds = bounds.reshape(-1, 2)
                values = np.asarray(values, dtype=float).reshape(-1)
                if bounds.shape[0] != values.size:
                    raise ValueError(
                        f"Parameter {para_idx + 1} has {values.size} values but "
                        f"{bounds.shape[0]} bound rows"
                    )
                active_parameters.append((para_idx, values, bounds))

            if active_parameters:
                current = np.concatenate(
                    [item[1] for item in active_parameters]
                ).reshape(1, -1)
                lower = np.concatenate([item[2][:, 0] for item in active_parameters])
                upper = np.concatenate([item[2][:, 1] for item in active_parameters])
                bounds = np.vstack((lower, upper))

                # Match the original bi-level design procedure: CMA-ES keeps
                # state while tuning one algorithm; population-level design
                # uses polynomial mutation for independent proposals.
                component_aux = dict(aux_entry)
                component_aux["rng"] = rng
                if alg_n == 1:
                    candidate, component_aux = get_component("search_cma")(
                        current, bounds, component_aux, "algorithm"
                    )
                    aux_entry.update(component_aux)
                else:
                    candidate, _ = get_component("search_mu_polynomial")(
                        current, bounds, component_aux, "algorithm"
                    )

                candidate = np.clip(
                    np.asarray(candidate, dtype=float).reshape(-1), lower, upper
                )
                offset = 0
                for para_idx, values, _ in active_parameters:
                    width = values.size
                    this_para[para_idx][0] = candidate[offset : offset + width].copy()
                    offset += width

        new_ops.append(paths)
        new_paras.append(this_para)
        aux_list[idx] = aux_entry

    return new_ops, new_paras, aux_list

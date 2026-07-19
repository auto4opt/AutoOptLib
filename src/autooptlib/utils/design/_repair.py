"""Python port of the MATLAB Repair routine."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._helpers import (
    ensure_rng,
    get_flex,
    get_problem_type,
    has_global_behavior,
    has_local_behavior,
    reinit_parameters,
    set_behavior,
)


def repair(
    operators: list[list[np.ndarray]],
    paras: list[list[list[Any]]],
    problem: Any,
    setting: Any,
):
    """Ensure the designed algorithm(s) remain reasonable."""
    rng = ensure_rng(setting)
    all_op = list(get_flex(setting, "all_op", required=True))
    para_local_space = list(
        get_flex(setting, "para_local_space", default=[None] * len(all_op))
    )
    behav_space = list(get_flex(setting, "behav_space", default=[None] * len(all_op)))

    while len(para_local_space) < len(all_op):
        para_local_space.append(None)
    while len(behav_space) < len(all_op):
        behav_space.append(None)

    problem_type = get_problem_type(problem) or ""
    if problem_type == "continuous":
        ind_mu = [idx + 1 for idx, name in enumerate(all_op) if "search_mu" in name]
    elif problem_type in {"discrete", "permutation"}:
        ind_mu = [idx + 1 for idx, name in enumerate(all_op) if "search" in name]
    else:
        ind_mu = []
    ind_cross = [idx + 1 for idx, name in enumerate(all_op) if "cross" in name]

    repaired_ops: list[list[np.ndarray]] = []
    repaired_paras: list[list[list[Any]]] = []

    for algo_idx, algo_ops in enumerate(operators):
        curr_ops = [np.array(path, dtype=int, copy=True) for path in algo_ops]
        curr_paras = [
            list(entry) if entry is not None else [None, None]
            for entry in paras[algo_idx]
        ]

        for path_idx, matrix in enumerate(curr_ops):
            if matrix.size == 0:
                continue

            for row in range(1, matrix.shape[0] - 1):
                if (
                    ind_cross
                    and matrix[row, 0] in ind_cross
                    and matrix[row, 1] not in ind_mu
                    and ind_mu
                ):
                    ind_new = int(rng.choice(ind_mu))
                    matrix[row, 1] = ind_new
                    matrix[row + 1, 0] = ind_new

            if matrix[-1, 0] in ind_cross and ind_mu:
                ind_new = int(rng.choice(ind_mu))
                matrix[-1, 0] = ind_new
                matrix[-2, 1] = ind_new

            mask = matrix[:, 0] != matrix[:, 1]
            matrix = matrix[mask]

            ind_pso = next(
                (idx + 1 for idx, name in enumerate(all_op) if name == "search_pso"),
                None,
            )
            if ind_pso is not None and np.any(matrix == ind_pso):
                ind_choose = next(
                    (
                        idx + 1
                        for idx, name in enumerate(all_op)
                        if name == "choose_traverse"
                    ),
                    None,
                )
                ind_update = next(
                    (
                        idx + 1
                        for idx, name in enumerate(all_op)
                        if name == "update_always"
                    ),
                    None,
                )
                if ind_choose and ind_update:
                    first_matrix = np.array(curr_ops[0], dtype=int, copy=True)
                    if first_matrix.size == 0:
                        first_matrix = np.zeros((2, 2), dtype=int)
                    first_matrix[0, 0] = ind_choose
                    first_matrix[-1, 1] = ind_update
                    curr_ops[0] = first_matrix
                    matrix = np.array(
                        [[ind_choose, ind_pso], [ind_pso, ind_update]], dtype=int
                    )

            ind_search = matrix[1:, 0].tolist()
            for idx in ind_search:
                set_behavior(curr_paras[idx - 1], "LS")

            ind_search = [
                idx
                for idx in ind_search
                if behav_space[idx - 1] is not None
                and has_global_behavior(behav_space[idx - 1])
            ]

            ind_gs: list[int] = []
            for idx in ind_search:
                behav_entry = behav_space[idx - 1]
                entry = curr_paras[idx - 1]
                if behav_entry is None or not has_local_behavior(behav_entry):
                    ind_gs.append(idx)
                    continue
                values = entry[0]
                bounds = para_local_space[idx - 1]
                if values is None or bounds is None:
                    ind_gs.append(idx)
                    continue
                bounds = np.asarray(bounds, dtype=float)
                lower = bounds[:, 0]
                upper = bounds[:, 1]
                if np.any(values < lower) or np.any(values > upper):
                    ind_gs.append(idx)

            if len(ind_gs) == 1:
                set_behavior(curr_paras[ind_gs[0] - 1], "GS")
            elif len(ind_gs) > 1:
                retain = int(rng.choice(ind_gs))
                rows_retain = np.where(matrix[:, 0] == retain)[0]
                if rows_retain.size > 1:
                    rows_to_delete = rng.choice(
                        rows_retain, size=rows_retain.size - 1, replace=False
                    )
                    rows_to_delete = np.sort(rows_to_delete)
                    for row in rows_to_delete[::-1]:
                        if row > 0:
                            matrix[row - 1, 1] = matrix[row, 1]
                        matrix = np.delete(matrix, row, axis=0)
                row_retain = np.where(matrix[:, 0] == retain)[0]
                retained = {retain}
                if (
                    retain in ind_cross
                    and row_retain.size > 0
                    and matrix[row_retain[0], 1] in ind_mu
                    and matrix[row_retain[0], 1] in ind_gs
                ):
                    retained.add(int(matrix[row_retain[0], 1]))
                for idx in retained:
                    set_behavior(curr_paras[idx - 1], "GS")
                    if idx in ind_gs:
                        ind_gs.remove(idx)

                for idx in list(ind_gs):
                    bounds = para_local_space[idx - 1]
                    if bounds is not None and len(bounds) > 0:
                        curr_paras[idx - 1][0] = reinit_parameters(
                            np.asarray(bounds, dtype=float), rng
                        )
                        set_behavior(curr_paras[idx - 1], "LS")
                    else:
                        rows_to_remove = np.where(matrix[:, 0] == idx)[0]
                        for row in rows_to_remove[::-1]:
                            if row > 0:
                                matrix[row - 1, 1] = matrix[row, 1]
                            matrix = np.delete(matrix, row, axis=0)

            curr_ops[path_idx] = matrix

        repaired_ops.append(curr_ops)
        repaired_paras.append(curr_paras)

    return repaired_ops, repaired_paras

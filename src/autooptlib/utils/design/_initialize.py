"""Python port of the MATLAB Initialize routine."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._helpers import ensure_rng, get_flex


def initialize(setting: Any, n: int):
    """Initialize the designed algorithm(s)."""
    alg_count = int(n)
    if alg_count <= 0:
        return [], []

    rng = ensure_rng(setting)
    op_space = np.array(get_flex(setting, "op_space", required=True), dtype=int)
    para_space = list(get_flex(setting, "para_space", required=True))
    alg_p = int(get_flex(setting, "alg_p", required=True))
    alg_q = int(get_flex(setting, "alg_q", required=True))

    operators = [[None for _ in range(alg_p)] for _ in range(alg_count)]
    paras: list[list[list[Any]]] = [
        [None for _ in range(len(para_space))] for _ in range(alg_count)
    ]

    for i in range(alg_count):
        ind_choose = int(rng.integers(op_space[0, 0], op_space[0, 1] + 1))
        ind_update = int(rng.integers(op_space[2, 0], op_space[2, 1] + 1))
        for j in range(alg_p):
            curr_q = int(rng.integers(1, alg_q + 1))
            matrix = np.zeros((curr_q + 1, 2), dtype=int)
            ind_search = int(rng.integers(op_space[1, 0], op_space[1, 1] + 1))
            matrix[0, :] = (ind_choose, ind_search)
            ind_start = ind_search
            for k in range(1, curr_q):
                ind_end = int(rng.integers(op_space[1, 0], op_space[1, 1] + 1))
                matrix[k, :] = (ind_start, ind_end)
                ind_start = ind_end
            matrix[-1, :] = (ind_start, ind_update)
            operators[i][j] = matrix

        ind_non_empty_para = [
            idx + 1
            for idx, space in enumerate(para_space)
            if space is not None and len(space) > 0
        ]
        temp_para = [[None, None] for _ in range(len(para_space))]
        for idx in ind_non_empty_para:
            bounds = np.asarray(para_space[idx - 1], dtype=float)
            lower = bounds[:, 0]
            upper = bounds[:, 1]
            values = lower + (upper - lower) * rng.random(lower.shape)
            temp_para[idx - 1][0] = values
        paras[i] = temp_para

    return operators, paras

"""Python port of the MATLAB Decode routine."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ._helpers import (
    Pathway,
    PathwayParam,
    SearchParam,
    SearchStep,
    ceil_divide,
    copy_if_array,
    get_flex,
    get_problem_type,
)


def decode(operators, paras, problem: Any, setting: Any):
    """Decode the designed algorithm from the graph representation."""
    all_op = list(get_flex(setting, "all_op", required=True))
    rate = float(get_flex(setting, "inc_rate", default=0.0))
    inner_fe = float(get_flex(setting, "inner_fe", default=1.0))
    prob_n = float(get_flex(setting, "prob_n", default=1.0))
    inner_g_max = ceil_divide(inner_fe, prob_n)
    alg_p = int(get_flex(setting, "alg_p", required=True))
    archive = list(get_flex(setting, "archive", default=[]))

    problem_type = get_problem_type(problem) or ""
    if problem_type == "continuous":
        ind_mu = [idx + 1 for idx, name in enumerate(all_op) if "search_mu" in name]
    elif problem_type in {"discrete", "permutation"}:
        ind_mu = [idx + 1 for idx, name in enumerate(all_op) if "search" in name]
    else:
        ind_mu = []
    ind_cross = [idx + 1 for idx, name in enumerate(all_op) if "cross" in name]

    op_pheno: list[list[Pathway]] = []
    para_pheno: list[list[PathwayParam]] = []

    for algo_idx, algo_ops in enumerate(operators):
        pathways: list[Pathway] = []
        pathway_params: list[PathwayParam] = []
        for path_idx in range(alg_p):
            matrix = np.array(algo_ops[path_idx], dtype=int)
            if matrix.size == 0:
                pathways.append(
                    Pathway(choose="", search=[], update="", archive=list(archive))
                )
                pathway_params.append(PathwayParam(choose=None, search=[], update=None))
                continue

            choose_idx = int(matrix[0, 0])
            update_idx = int(matrix[-1, 1])
            choose_name = all_op[choose_idx - 1]
            update_name = all_op[update_idx - 1]

            choose_param = None
            if choose_idx - 1 < len(paras[algo_idx]):
                entry = paras[algo_idx][choose_idx - 1]
                if entry:
                    choose_param = copy_if_array(entry[0])

            update_param = None
            if update_idx - 1 < len(paras[algo_idx]):
                entry = paras[algo_idx][update_idx - 1]
                if entry:
                    update_param = copy_if_array(entry[0])

            search_steps: list[SearchStep] = []
            search_params: list[SearchParam] = []

            row = 1
            while row < matrix.shape[0]:
                primary_idx = int(matrix[row, 0])
                primary_name = all_op[primary_idx - 1]
                primary_param = None
                if primary_idx - 1 < len(paras[algo_idx]):
                    entry = paras[algo_idx][primary_idx - 1]
                    if entry:
                        primary_param = copy_if_array(entry[0])
                secondary_name = None
                secondary_param = None
                termination = np.array([rate, inner_g_max], dtype=float)

                if (
                    primary_idx in ind_cross
                    and row < matrix.shape[0]
                    and matrix[row, 1] in ind_mu
                ):
                    secondary_idx = int(matrix[row, 1])
                    secondary_name = all_op[secondary_idx - 1]
                    if secondary_idx - 1 < len(paras[algo_idx]):
                        entry = paras[algo_idx][secondary_idx - 1]
                        if entry:
                            secondary_param = copy_if_array(entry[0])
                    primary_behav = None
                    if primary_idx - 1 < len(paras[algo_idx]):
                        entry = paras[algo_idx][primary_idx - 1]
                        if entry:
                            primary_behav = entry[1]
                    secondary_behav = None
                    if secondary_idx - 1 < len(paras[algo_idx]):
                        entry = paras[algo_idx][secondary_idx - 1]
                        if entry:
                            secondary_behav = entry[1]
                    if primary_behav == "GS" or secondary_behav == "GS":
                        termination = np.array([-math.inf, 1], dtype=float)
                    row += 2
                else:
                    primary_behav = None
                    if primary_idx - 1 < len(paras[algo_idx]):
                        entry = paras[algo_idx][primary_idx - 1]
                        if entry:
                            primary_behav = entry[1]
                    if primary_behav == "GS":
                        termination = np.array([-math.inf, 1], dtype=float)
                    row += 1

                search_steps.append(
                    SearchStep(
                        primary=primary_name,
                        secondary=secondary_name,
                        termination=termination,
                    )
                )
                search_params.append(
                    SearchParam(primary=primary_param, secondary=secondary_param)
                )

            pathways.append(
                Pathway(
                    choose=choose_name,
                    search=search_steps,
                    update=update_name,
                    archive=list(archive),
                )
            )
            pathway_params.append(
                PathwayParam(
                    choose=choose_param,
                    search=search_params,
                    update=update_param,
                )
            )

        op_pheno.append(pathways)
        para_pheno.append(pathway_params)

    return op_pheno, para_pheno

"""Python translation of choose_nich."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from ._utils import ensure_rng, flex_get, pairwise_distances


@dataclass
class Species:
    seed: int
    idx: np.ndarray


def _extract_decisions(solution) -> np.ndarray:
    decs = flex_get(solution, "decs")
    if decs is None:
        raise ValueError("Solution object must provide decision vectors")
    return np.asarray(decs, dtype=float)


def _nearest_better_clustering(
    distance: np.ndarray, fai: float, min_size: int, g: int, gmax: int
) -> List[Species]:
    if min_size == -1:
        # MATLAB rounds positive .5 values away from zero; Python uses bankers'
        # rounding, which changes species splitting at exact half steps.
        min_size = int(np.floor(3 + g / max(gmax, 1) * 7 + 0.5))

    npop = distance.shape[0]
    nearest = np.full(npop, -1, dtype=int)
    dist_val = np.zeros(npop, dtype=float)

    for i in range(1, npop):
        segment = distance[i, :i]
        idx = int(np.argmin(segment))
        nearest[i] = idx
        dist_val[i] = segment[idx]

    follow = np.ones(npop, dtype=int)
    for i in range(npop - 1, 0, -1):
        parent = nearest[i]
        if parent >= 0:
            follow[parent] += follow[i]

    meandis = fai * dist_val[1:].mean() if npop > 1 else 0.0
    seeds: List[int] = [0]

    order = np.argsort(dist_val)[::-1]
    for inf_index in order:
        if dist_val[inf_index] <= meandis:
            continue
        sup_index = nearest[inf_index]
        if sup_index < 0:
            continue
        chain = [sup_index]
        top_index = sup_index
        while nearest[top_index] != -1:
            top_index = nearest[top_index]
            chain.append(top_index)
        if (
            follow[inf_index] >= min_size
            and follow[top_index] - follow[inf_index] >= min_size
        ):
            nearest[inf_index] = -1
            dist_val[inf_index] = 0.0
            seeds.append(inf_index)
            for node in chain:
                follow[node] -= follow[inf_index]

    roots = np.zeros(npop, dtype=int)
    for i in range(npop):
        node = i
        parent = nearest[node]
        while parent != -1:
            node = parent
            parent = nearest[node]
        roots[i] = node if node != -1 else i

    species = []
    for seed in seeds:
        members = np.where(roots == seed)[0]
        species.append(Species(seed=seed, idx=members))
    return species


def choose_nich(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        aux = args[3] if len(args) > 3 else None
        g = int(args[4]) if len(args) > 4 else 0
        gmax = int(flex_get(problem, "Gmax", g + 1))
        rng = ensure_rng(aux)

        decs = _extract_decisions(solution)
        distance = pairwise_distances(decs)
        species = _nearest_better_clustering(distance, 1.0, -1, g, gmax)

        index_list: List[int] = []
        for specie in species:
            if specie.idx.size == 0:
                continue
            permuted = rng.permutation(specie.idx)
            index_list.extend(permuted.tolist())

        if not index_list:
            return np.array([], dtype=int), None

        if len(index_list) % 2 == 1:
            odd = True
            index_list.append(index_list[-1])
        else:
            odd = False

        pairs = np.asarray(index_list, dtype=int).reshape(-1, 2)
        ordered = np.concatenate((pairs[:, 0], pairs[:, 1]))
        if odd:
            ordered = ordered[:-1]
        return ordered.astype(int), None

    if mode == "parameter":
        return None, None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

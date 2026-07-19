"""Brainstorm-based selection operator (continuous problems)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ._utils import ensure_rng, flex_get


def _extract_decs(solution: Any) -> np.ndarray:
    decs = flex_get(solution, "decs")
    if decs is None:
        raise ValueError("Solution object must provide decision variables via decs")
    return np.asarray(decs, dtype=float)


def _extract_objs(solution: Any) -> np.ndarray:
    objs = flex_get(solution, "objs")
    if objs is None:
        raise ValueError("Solution object must provide objective values via objs")
    arr = np.asarray(objs, dtype=float)
    if arr.ndim > 1:
        arr = arr[:, 0]
    return arr.reshape(-1)


def _kmeans_labels(
    data: np.ndarray, k: int, rng: np.random.Generator, max_iter: int = 50
) -> np.ndarray:
    n_samples = data.shape[0]
    if n_samples == 0 or k <= 0:
        return np.empty(0, dtype=int)
    k = min(k, n_samples)
    initial_idx = rng.choice(n_samples, size=k, replace=False)
    centers = data[initial_idx].copy()
    labels = np.zeros(n_samples, dtype=int)

    for _ in range(max_iter):
        distances = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for idx in range(k):
            members = data[labels == idx]
            if members.size == 0:
                centers[idx] = data[rng.integers(0, n_samples)]
            else:
                centers[idx] = members.mean(axis=0)
    return labels


def choose_brainstorm(*args: Any):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        rng = ensure_rng(aux)

        decs = _extract_decs(solution)
        pop_size = decs.shape[0]
        if pop_size == 0:
            return np.array([], dtype=int), None

        ptype = flex_get(problem, "type", ["continuous"])
        ptype0 = ptype[0] if isinstance(ptype, (list, tuple)) else ptype
        if "continuous" not in str(ptype0).lower():
            raise ValueError(
                "choose_brainstorm is only available for continuous problems"
            )

        para_arr = (
            np.asarray(para, dtype=float) if para is not None else np.array([2.0, 0.5])
        )
        if para_arr.size < 2:
            para_arr = np.pad(
                para_arr, (0, 2 - para_arr.size), mode="constant", constant_values=0.5
            )
        k = max(2, int(round(float(para_arr[0]))))
        gamma = float(para_arr[1])
        k = min(k, pop_size)
        if k <= 0:
            return np.array([], dtype=int), None

        labels = _kmeans_labels(decs, k, rng)
        objs = _extract_objs(solution)

        centers: list[int] = []
        probabilities: list[float] = []
        cluster_members: dict[int, np.ndarray] = {}

        for cluster_id in range(k):
            members = np.where(labels == cluster_id)[0]
            if members.size == 0:
                members = np.array([rng.integers(0, pop_size)], dtype=int)
            costs = objs[members]
            best_local = int(members[int(np.argmin(costs))])
            centers.append(best_local)
            probabilities.append(float(len(members)) / float(pop_size))
            remaining = members[members != best_local]
            cluster_members[best_local] = remaining

        probabilities = np.asarray(probabilities, dtype=float)
        total = probabilities.sum()
        if total <= 0:
            probabilities = np.full(len(probabilities), 1.0 / len(probabilities))
        else:
            probabilities /= total

        n_select = int(flex_get(problem, "N", pop_size))
        chosen_centers = rng.choice(
            centers, size=n_select, replace=True, p=probabilities
        )

        result = []
        for center_idx in chosen_centers:
            members = cluster_members.get(int(center_idx))
            if members is not None and members.size > 0 and rng.random() <= gamma:
                result.append(int(rng.choice(members)))
            else:
                result.append(int(center_idx))

        return np.asarray(result, dtype=int), None

    if mode == "parameter":
        problem = args[0]
        pop_size = int(flex_get(problem, "N", 1))
        k_max = max(1, round(pop_size / 5))
        return np.array([[1, k_max], [0, 1]], dtype=float), None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

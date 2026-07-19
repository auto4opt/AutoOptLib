"""Python translation of update_simulated_annealing."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, extract_fits, flex_get, solution_as_list


def update_simulated_annealing(*args):
    mode = args[-1]
    if mode == "execute":
        solution = args[0]
        problem = args[1] if len(args) > 1 else None
        para = args[2] if len(args) > 2 else None
        aux = args[3] if len(args) > 3 else None
        g = int(args[4]) if len(args) > 4 else 0
        rng = ensure_rng(aux)

        sol_list = solution_as_list(solution)
        fitness = extract_fits(solution)
        n = int(flex_get(problem, "N", len(sol_list) // 2))
        if len(sol_list) < 2 * n:
            raise ValueError(
                "update_simulated_annealing expects old and new populations of size N"
            )

        try:
            t_initial = float(np.asarray(para).reshape(-1)[0])
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(
                "Parameter for simulated annealing must be convertible to float"
            ) from exc
        t_final = 0.01
        gmax = max(int(flex_get(problem, "Gmax", g + 1)), 1)
        rate = (t_final / t_initial) ** (1.0 / gmax)
        temperature = t_initial * (rate**g)

        old_list = sol_list[:n]
        new_list = sol_list[n : 2 * n]
        old_fits = fitness[:n]
        new_fits = fitness[n : n + n]

        denom = np.abs(old_fits + 1e-6) * max(temperature, 1e-8)
        exponent = (old_fits - new_fits) / denom
        exponent = np.clip(exponent, -700.0, 700.0)
        acceptance_prob = np.exp(exponent)
        acceptance = rng.random(n) < acceptance_prob
        acceptance = acceptance | (old_fits > new_fits)

        updated = list(old_list)
        for idx, accept in enumerate(acceptance):
            if accept:
                updated[idx] = new_list[idx]
        return updated, None

    if mode == "parameter":
        return np.array([0.1, 1.0]), None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

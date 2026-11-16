"""Python translation of update_simulated_annealing."""

from __future__ import annotations

import numpy as np

from ._utils import ensure_rng, flex_get, solution_as_list


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
        n = int(flex_get(problem, "N", len(sol_list) // 2))
        if len(sol_list) < n:
            raise ValueError("update_simulated_annealing requires an old population")

        try:
            t_initial = float(np.asarray(para).reshape(-1)[0])
        except Exception as exc:  # pylint: disable=broad-except
            raise ValueError(
                "Parameter for simulated annealing must be convertible to float"
            ) from exc
        if not np.isfinite(t_initial) or t_initial <= 0.0:
            raise ValueError(
                "The simulated-annealing initial temperature must be finite and positive."
            )
        t_final = 0.01
        gmax = max(int(flex_get(problem, "Gmax", g + 1)), 1)
        rate = (t_final / t_initial) ** (1.0 / gmax)
        temperature = t_initial * (rate**g)

        old_list = sol_list[:n]
        new_list = sol_list[n:]
        updated = list(old_list)
        for offspring_index, child in enumerate(new_list):
            idx = offspring_index % n
            old_fit = float(updated[idx].fit)
            new_fit = float(child.fit)
            denom = (abs(old_fit) + 1e-6) * max(temperature, 1e-8)
            exponent = float(np.clip((old_fit - new_fit) / denom, -700.0, 700.0))
            if old_fit > new_fit or float(rng.random()) < float(np.exp(exponent)):
                updated[idx] = child
        return updated, None

    if mode == "parameter":
        return np.array([0.1, 1.0]), None

    if mode == "behavior":
        return ["", ""], None

    raise ValueError(f"Unsupported mode: {mode}")

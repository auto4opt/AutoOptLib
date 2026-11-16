"""Current hand-designed algorithm presets used by the shared solver."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence

import numpy as np
from autooptlib.utils.design._helpers import (
    Pathway,
    PathwayParam,
    SearchParam,
    SearchStep,
)

if TYPE_CHECKING:
    from autooptlib.utils.design import Design


def _to_array(value: Any) -> Optional[np.ndarray]:
    if value is None or (isinstance(value, (list, tuple)) and len(value) == 0):
        return None
    arr = np.asarray(value, dtype=float)
    return arr.reshape(-1) if arr.ndim > 0 else arr


def _make_search_step(
    primary: str, secondary: Optional[str], termination: Sequence[float]
) -> SearchStep:
    term = np.asarray(termination, dtype=float).reshape(-1)
    return SearchStep(
        primary=primary, secondary=secondary if secondary else None, termination=term
    )


def _make_search_param(primary: Any, secondary: Any) -> SearchParam:
    return SearchParam(primary=_to_array(primary), secondary=_to_array(secondary))


def _build_default_algorithm(name: str, setting: Any) -> "Design":
    name = name.strip().lower()
    tmpl: dict[str, list[Any]] = {
        "continuous genetic algorithm": [
            (
                "choose_tournament",
                [
                    (
                        "cross_sim_binary",
                        "search_mu_polynomial",
                        (-np.inf, 1),
                        [20.0],
                        [0.2, 20.0],
                    ),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "evolutionary programming": [
            (
                "choose_tournament",
                [
                    ("search_mu_gaussian", None, (-np.inf, 1), None, None),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "fast evolutionary programming": [
            (
                "choose_tournament",
                [
                    ("search_mu_cauchy", None, (-np.inf, 1), None, None),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "cma-es": [
            (
                "choose_traverse",
                [
                    ("search_cma", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "estimation of distribution": [
            (
                "choose_traverse",
                [
                    ("search_eda", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "particle swarm optimization": [
            (
                "choose_traverse",
                [
                    ("search_pso", None, (-np.inf, 1), [0.9], None),
                ],
                "update_pairwise",
                [],
            ),
        ],
        "differential evolution": [
            (
                "choose_traverse",
                [
                    ("search_de_current", None, (-np.inf, 1), [0.9, 0.1], None),
                ],
                "update_pairwise",
                [],
            ),
        ],
        "continuous random search": [
            (
                "choose_traverse",
                [
                    ("reinit_continuous", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "ica": [
            (
                "choose_ica",
                [
                    ("search_ica", None, (-np.inf, 1), [0.6, 0.5], None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "discrete genetic algorithm": [
            (
                "choose_tournament",
                [
                    (
                        "cross_point_uniform",
                        "search_reset_rand",
                        (-np.inf, 1),
                        [0.2],
                        [0.2],
                    ),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "discrete iterative local search": [
            (
                "choose_traverse",
                [
                    ("search_reset_one", None, (0.05, 10), None, None),
                    ("reinit_discrete", None, (-np.inf, 1), None, None),
                ],
                "update_iterated_local_search",
                [],
            ),
        ],
        "discrete simulated annealing": [
            (
                "choose_traverse",
                [
                    ("search_reset_one", None, (-np.inf, 1), None, None),
                ],
                "update_simulated_annealing",
                [],
                np.array([0.1]),
            ),
        ],
        "discrete random search": [
            (
                "choose_traverse",
                [
                    ("reinit_discrete", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "permutation genetic algorithm": [
            (
                "choose_tournament",
                [
                    ("cross_order_two", "search_swap", (-np.inf, 1), None, None),
                ],
                "update_round_robin",
                [],
            ),
        ],
        "permutation iterative local search": [
            (
                "choose_traverse",
                [
                    ("search_insert", None, (0.05, 10), None, None),
                    ("reinit_permutation", None, (-np.inf, 1), None, None),
                ],
                "update_iterated_local_search",
                [],
            ),
        ],
        "permutation simulated annealing": [
            (
                "choose_traverse",
                [
                    ("search_insert", None, (-np.inf, 1), None, None),
                ],
                "update_simulated_annealing",
                [],
                np.array([0.1]),
            ),
        ],
        "permutation variable neighborhood search": [
            (
                "choose_traverse",
                [
                    ("search_swap", None, (0.05, 10), None, None),
                    ("search_scramble", None, (0.05, 10), None, None),
                    ("search_insert", None, (0.05, 10), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
        "permutation random search": [
            (
                "choose_traverse",
                [
                    ("reinit_permutation", None, (-np.inf, 1), None, None),
                ],
                "update_greedy",
                [],
            ),
        ],
    }

    if name not in tmpl:
        raise NotImplementedError(
            f"Preset algorithm {name!r} is not available in the Python translation."
        )

    pathways_cfg = tmpl[name]
    pathways: List[Pathway] = []
    params: List[PathwayParam] = []
    for item in pathways_cfg:
        choose = item[0]
        search_entries = item[1]
        update = item[2]
        archive = item[3] if len(item) > 3 else []
        update_param = item[4] if len(item) > 4 else None

        search_steps = []
        search_params = []
        for entry in search_entries:
            primary, secondary, termination, primary_param, secondary_param = entry
            search_steps.append(_make_search_step(primary, secondary, termination))
            search_params.append(_make_search_param(primary_param, secondary_param))

        pathways.append(
            Pathway(choose=choose, search=search_steps, update=update, archive=archive)
        )
        params.append(
            PathwayParam(
                choose=None,
                search=search_params,
                update=_to_array(update_param),
            )
        )

    from autooptlib.utils.design import Design

    design = Design()
    design.construct([pathways], [params])
    return design

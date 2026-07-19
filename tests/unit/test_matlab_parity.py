"""Golden contracts for intentional MATLAB/Python semantic parity."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from autooptlib import ObjectiveEvaluationError
from autooptlib.utils.design._disturb import disturb
from autooptlib.utils.design._evaluate import _auc_score
from autooptlib.utils.design._helpers import (
    Pathway,
    PathwayParam,
    SearchParam,
    SearchStep,
)
from autooptlib.utils.select import select
from autooptlib.utils.solve import make_solutions, run_design


def _encoded_algorithm():
    return SimpleNamespace(
        operator=[
            np.array([[1, 3], [3, 5]]),
            np.array([[1, 4], [4, 5]]),
        ],
        parameter=[[None, None] for _ in range(6)],
    )


def _disturb_setting():
    return SimpleNamespace(
        op_space=np.array([[1, 2], [3, 4], [5, 6]]),
        para_space=[None] * 6,
        para_local_space=[None] * 6,
        alg_p=2,
        alg_q=1,
        alg_n=2,
        tune_para=False,
        rng=np.random.default_rng(8),
    )


def test_choose_and_update_disturbance_are_shared_across_pathways():
    choose_ops, _, _ = disturb(
        [_encoded_algorithm()], _disturb_setting(), 2, [{"seed": 1}]
    )
    assert [path[0, 0] for path in choose_ops[0]] == [2, 2]

    # The encoded positions are choose, two searches, update.
    update_ops, _, _ = disturb(
        [_encoded_algorithm()], _disturb_setting(), 2, [{"seed": 4}]
    )
    assert [path[-1, -1] for path in update_ops[0]] == [6, 6]


class _Performance:
    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)
        self.performance = np.ones((1, len(self.values)))

    def get_performance(self, setting, seeds):
        return self.values


def test_racing_keeps_all_significant_survivors(monkeypatch):
    select_module = importlib.import_module("autooptlib.utils.select")
    monkeypatch.setattr(
        select_module, "_statistic_wins", lambda matrix, alpha: np.array([2, 1, 0])
    )
    algorithms = [_Performance([1, 1]), _Performance([2, 2]), _Performance([3, 3])]
    setting = SimpleNamespace(
        Compare="statistic",
        Evaluate="racing",
        AlgN=1,
        AlgRuns=2,
        Seed=4,
    )
    assert select(algorithms, None, None, setting, [0]) == algorithms[:2]


def test_auc_uses_fe_checkpoints_and_reciprocal_success_fraction():
    assert _auc_score([5.0, 3.0, 2.0], [2, 4, 6], [5, 3, 1], 2) == pytest.approx(1.5)
    assert np.isinf(_auc_score([], [2, 4], [2, 1], 2))


def test_problem_repair_uncertainty_and_accessory_data_are_retained():
    repaired_problem = SimpleNamespace(
        type=["continuous", "static", "certain"],
        bound=np.array([[-10.0], [10.0]]),
        repair=lambda data, decision: np.zeros_like(decision),
        evaluate=lambda data, decision: (4.0, 0.0, {"source": "simulator"}),
    )
    repaired = make_solutions(np.array([[7.0]]), repaired_problem, None)[0]
    np.testing.assert_array_equal(repaired.dec, [0.0])
    assert repaired.acc == {"source": "simulator"}

    samples = iter([1.0, 3.0, 5.0])
    uncertain = SimpleNamespace(
        type=["continuous", "static", "uncertain"],
        bound=np.array([[-1.0], [1.0]]),
        sampleN=3,
        setting="uncertain_average",
        evaluate=lambda data, decision: (next(samples), 0.0, None),
    )
    assert make_solutions(np.array([[0.0]]), uncertain, None)[0].obj == 3.0

    worst_samples = iter([1.0, 3.0, 5.0])
    uncertain.setting = "uncertain_worst"
    uncertain.evaluate = lambda data, decision: (next(worst_samples), 0.0, None)
    assert make_solutions(np.array([[0.0]]), uncertain, None)[0].obj == 5.0

    uncertain.sampleN = 0
    with pytest.raises(ObjectiveEvaluationError, match="positive sampleN"):
        make_solutions(np.array([[0.0]]), uncertain, None)
    uncertain.sampleN = 1
    uncertain.setting = ""
    uncertain.evaluate = lambda data, decision: (1.0, 0.0, None)
    with pytest.raises(ObjectiveEvaluationError, match="uncertain_average"):
        make_solutions(np.array([[0.0]]), uncertain, None)


def test_execution_tracks_historical_best_and_uses_embedded_archive(monkeypatch):
    solve_module = importlib.import_module("autooptlib.utils.solve")
    seen = []

    def choose_all(population, problem, parameter, aux, *args):
        aux["choose"] = True
        return np.arange(len(population)), aux

    def make_worse(parent, problem, parameter, aux, *args):
        assert aux["choose"]
        aux["search"] = True
        return parent.decs() + 1.0, aux

    def replace_with_new(combined, problem, parameter, aux, *args):
        assert aux["search"]
        seen.append(aux)
        return combined[-problem.N :], None

    components = {
        "choose_parity": choose_all,
        "search_parity": make_worse,
        "update_parity": replace_with_new,
    }
    original = solve_module.get_component

    def lookup(name):
        return components[name] if name in components else original(name)

    monkeypatch.setattr(solve_module, "get_component", lookup)

    path = Pathway(
        "choose_parity",
        [SearchStep("search_parity", np.array([-np.inf, 1.0]))],
        "update_parity",
        ["archive_best"],
    )
    params = PathwayParam(None, [SearchParam(None, None)], None)
    problem = SimpleNamespace(
        type=["continuous", "static", "certain"],
        bound=np.array([[0.0], [10.0]]),
        N=2,
        Gmax=1,
        evaluate=lambda data, decision: (float(decision[0]), 0.0, None),
    )
    result = run_design(
        [path],
        [params],
        problem,
        None,
        SimpleNamespace(ProbN=2, ProbFE=4, Seed=2),
    )
    assert seen and seen[0]["choose"] and seen[0]["search"]
    assert result["fit_history"] == sorted(result["fit_history"], reverse=True)
    assert result["best_solution"].fit == min(result["fit_history"])
    assert len(result["archives"]) == 1
    assert result["archives"][0]

"""Unit tests for utility functions and problem definitions."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from autooptlib.problems.cec2013 import cec2013_f1
from autooptlib.utils.design import Design
from autooptlib.utils.space import space


class DummyProblem:
    """Continuous problem with quadratic objective."""

    def __init__(self):
        self.bound = np.array([[-1.0, -1.0], [1.0, 1.0]])
        self.type = ["continuous", "static", "certain"]
        self.N = 4
        self.Gmax = 5

    def evaluate(self, data, dec):
        arr = np.asarray(dec, dtype=float)
        value = float(np.sum(arr**2))
        return value, 0.0, None


def test_cec2013_f1_construct_and_evaluate():
    problems, data, _ = cec2013_f1([SimpleNamespace()], [2], "construct")
    assert len(problems) == 1
    problem = problems[0]
    datum = data[0]
    assert problem.type[0] == "continuous"
    fit, con, _ = cec2013_f1(datum, np.array([[0.0, 0.0]]), "evaluate")
    assert con == 0.0
    assert fit.shape == (1,)


def test_design_initialize_and_evaluate_small_problem():
    problem = DummyProblem()
    setting = SimpleNamespace(
        Mode="design",
        archive=["archive_best"],
        AlgP=1,
        AlgQ=2,
        IncRate=0.05,
        ProbN=4,
        ProbFE=40,
        AlgN=1,
        AlgFE=20,
        AlgRuns=1,
        Evaluate="exact",
        Compare="average",
    )
    setting = space([problem], setting)
    design = Design([problem], setting)
    assert design.performance.shape == (1, 1)
    design.evaluate([problem], [None], setting, [0])
    assert design.performance[0, 0] >= 0.0


def test_space_supports_discrete_and_permutation():
    discrete = SimpleNamespace(
        type=["discrete", "static"], bound=np.array([[0, 0], [3, 3]]), N=3, Gmax=5
    )
    permutation = SimpleNamespace(
        type=["permutation", "static"],
        bound=np.array([[1, 2, 3], [1, 2, 3]]),
        N=3,
        Gmax=5,
    )
    base_kwargs = dict(
        Mode="design",
        archive=["archive_best"],
        AlgP=1,
        AlgQ=2,
        IncRate=0.05,
        ProbN=3,
        ProbFE=30,
        AlgN=1,
        AlgFE=10,
        AlgRuns=1,
        Evaluate="exact",
        Compare="average",
    )
    discrete_setting = space([discrete], SimpleNamespace(**base_kwargs))
    permutation_setting = space([permutation], SimpleNamespace(**base_kwargs))
    assert "reinit_discrete" in discrete_setting.AllOp
    assert "reinit_permutation" in permutation_setting.AllOp


def test_continuous_space_contains_paper_search_components():
    result = space([DummyProblem()], SimpleNamespace())
    for component in ("cross_arithmetic", "cross_sim_binary", "search_eda"):
        assert component in result.AllOp


def test_space_preserves_matlab_style_algorithm_settings():
    setting = SimpleNamespace(AlgP=2, AlgQ=4, AlgN=3)
    result = space([DummyProblem()], setting)
    assert result.alg_p == 2
    assert result.alg_q == 4
    assert result.alg_n == 3


def test_design_get_new_preserves_dimensions():
    problem = DummyProblem()
    setting = SimpleNamespace(
        Mode="design",
        archive=["archive_best"],
        AlgP=1,
        AlgQ=2,
        IncRate=0.05,
        ProbN=4,
        ProbFE=40,
        AlgN=1,
        AlgFE=20,
        AlgRuns=1,
        Evaluate="exact",
        Compare="average",
    )
    setting = space([problem], setting)
    design = Design([problem], setting)
    new_design, aux = design.get_new([problem], setting, inner_g=1, aux=None)
    assert new_design.performance.shape == design.performance.shape
    assert aux is None or isinstance(aux, dict)


def test_parameter_tuning_uses_stateful_cma_for_one_algorithm():
    problem = DummyProblem()
    setting = SimpleNamespace(
        Mode="design",
        archive=["archive_best"],
        AlgP=1,
        AlgQ=2,
        IncRate=0.05,
        ProbN=4,
        ProbFE=40,
        AlgN=1,
        AlgFE=20,
        AlgRuns=1,
        Evaluate="exact",
        Compare="average",
        TunePara=True,
        rng=np.random.default_rng(7),
    )
    setting = space([problem], setting)
    design = Design([problem], setting)
    new_design, aux = design.get_new([problem], setting, inner_g=1, aux=None)

    assert new_design.performance.shape == design.performance.shape
    assert isinstance(aux, dict)
    assert aux["seed"] > sum(path.shape[0] + 1 for path in design.operator)
    active = [
        int(value)
        for path in design.operator
        for value in np.concatenate(([path[0, 0]], path[1:, 0], [path[-1, -1]]))
    ]
    has_tunable_parameter = any(
        0 < index <= len(design.parameter)
        and design.parameter[index - 1][0] is not None
        for index in active
    )
    assert ("cma_Disturb" in aux) == has_tunable_parameter

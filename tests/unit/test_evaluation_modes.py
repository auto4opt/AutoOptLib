"""Focused tests for exact, approximate, and sequential design evaluation."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from autooptlib.utils.design import Design
from autooptlib.utils.solve import Solution

evaluation_module = importlib.import_module("autooptlib.utils.design._evaluate")


def _design():
    design = Design()
    design.construct([[object()]], [[object()]])
    return design


def _problem(behavior="static"):
    return SimpleNamespace(
        type=["continuous", behavior, "certain"],
        N=2,
        bound=np.array([[-1.0], [1.0]]),
    )


def _setting(metric="quality", evaluate="exact"):
    return SimpleNamespace(
        Metric=metric,
        Evaluate=evaluate,
        AlgRuns=1,
        ProbN=2,
        ProbFE=6,
        Tmax=[2, 4, 6] if metric == "auc" else None,
        Thres=[5, 3, 1] if metric == "auc" else None,
    )


def test_matrix_resize_preserves_existing_values():
    matrix = np.array([[3.0]])
    assert evaluation_module._ensure_matrix(matrix, 1, 1) is matrix
    resized = evaluation_module._ensure_matrix(matrix, 2, 2)
    assert resized.shape == (2, 2)
    assert resized[0, 0] == 3.0
    assert evaluation_module._ensure_matrix(resized, 1, 1) is resized


def test_design_performance_aggregates_one_value_per_algorithm():
    design = Design()
    design.performance = np.array([[1.0, 3.0], [5.0, 7.0]])
    design.performance_approx = np.array([[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_array_equal(design.ave_perform_all(), [4.0])
    np.testing.assert_array_equal(design.ave_perform_approx_all(), [5.0])
    np.testing.assert_array_equal(design.ave_perform_per([1]), [6.0])
    np.testing.assert_array_equal(design.ave_perform_approx_per([0]), [3.0])


def test_sequential_update_supports_method_module_and_fallback(monkeypatch):
    marker = object()
    direct = SimpleNamespace(advance_sequence=lambda best, data: (marker, data + 1))
    assert evaluation_module._update_sequential(direct, 1, object()) == (marker, 2)

    module = ModuleType("autooptlib_test_sequence")
    module.advance = lambda problem, data, best, mode: (marker, mode)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    plugin = SimpleNamespace(name="autooptlib_test_sequence.advance")
    assert evaluation_module._update_sequential(plugin, None, None) == (
        marker,
        "sequence",
    )
    broken = SimpleNamespace(name="missing.module")
    assert evaluation_module._update_sequential(broken, 1, None) == (broken, 1)
    plain = SimpleNamespace(name=None)
    assert evaluation_module._update_sequential(plain, 2, None) == (plain, 2)


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("quality", 2.0),
        ("auc", 1.5),
        ("runtimeFE", 6.0),
        ("runtimeSec", 0.25),
        ("custom", 2.0),
    ],
)
def test_static_evaluation_metrics(monkeypatch, metric, expected):
    monkeypatch.setattr(
        evaluation_module,
        "run_design",
        lambda *args: {
            "fit_history": [5.0, 3.0, 2.0],
            "evaluations": 6,
            "elapsed": 0.25,
        },
    )
    design = _design()
    design.evaluate([_problem()], [None], _setting(metric), [0])
    assert design.performance[0, 0] == pytest.approx(expected)


def test_static_approximate_and_empty_auc(monkeypatch):
    monkeypatch.setattr(
        evaluation_module,
        "run_design",
        lambda *args: {"fit_history": [], "evaluations": 0, "elapsed": 0.0},
    )
    design = _design()
    design.evaluate(_problem(), None, _setting("auc", "approximate"), [1])
    assert design.performance_approx.shape == (2, 1)
    assert np.isinf(design.performance_approx[1, 0])


def _sequential_problem(next_stage=True):
    problem = _problem("sequential")

    def advance(best, data):
        next_data = SimpleNamespace(stage=data.stage + 1)
        setattr(next_data, "continue", next_stage and next_data.stage < 1)
        return problem, next_data

    problem.advance_sequence = advance
    return problem


@pytest.mark.parametrize(
    ("metric", "expected"),
    [
        ("quality", 2.0),
        ("auc", 1.5),
        ("runtimeFE", 2.0),
        ("runtimeSec", 0.5),
        ("custom", 2.0),
    ],
)
def test_sequential_evaluation_metrics(monkeypatch, metric, expected):
    best = Solution(np.array([0.0]), 2.0, 0.0, 2.0)
    monkeypatch.setattr(
        evaluation_module,
        "run_design",
        lambda *args: {
            "fit_history": [5.0, 3.0, 2.0],
            "evaluations": 2,
            "elapsed": 0.5,
            "best_solution": best,
        },
    )
    data = SimpleNamespace(stage=0)
    setattr(data, "continue", True)
    design = _design()
    design.evaluate([_sequential_problem()], [data], _setting(metric), [0])
    assert design.performance[0, 0] == pytest.approx(expected)


def test_sequential_empty_history_and_missing_best_stop(monkeypatch):
    monkeypatch.setattr(
        evaluation_module,
        "run_design",
        lambda *args: {
            "fit_history": [],
            "evaluations": 2,
            "elapsed": 0.0,
            "best_solution": None,
        },
    )
    data = SimpleNamespace()
    setattr(data, "continue", True)
    design = _design()
    design.evaluate([_sequential_problem()], [data], _setting("auc"), [0])
    assert np.isinf(design.performance[0, 0])


def test_evaluation_no_phenotype_and_unknown_behavior():
    empty = Design()
    assert empty.evaluate([], [], _setting(), []) is empty
    design = _design()
    with pytest.raises(NotImplementedError, match="Unsupported problem type behavior"):
        design.evaluate([_problem("dynamic")], [None], _setting(), [0])

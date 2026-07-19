"""Behavioral tests for improvement tracking and algorithm selection."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from autooptlib.utils.design import Design
from autooptlib.utils.general.improve_rate import improve_rate
from autooptlib.utils.select import (
    _ensure_design_list,
    _require_performance,
    _statistic_wins,
    select,
)

select_module = importlib.import_module("autooptlib.utils.select")


def test_improvement_rate_supports_solution_interfaces_and_constraints():
    callable_fits = SimpleNamespace(fits=lambda: np.array([3.0, 1.0]))
    assert improve_rate(callable_fits, None, 1, "solution")[-1] == 1.0
    property_fits = SimpleNamespace(fits=np.array([4.0, 2.0]))
    assert improve_rate(property_fits, None, 1, "solution")[-1] == 2.0

    feasible = SimpleNamespace(objs=lambda: [3.0], cons=lambda: [-1.0])
    infeasible = SimpleNamespace(obj=1.0, cons=[2.0])
    result = improve_rate([feasible, infeasible], None, 1, "solution")
    assert result[-1] == 3.0
    single = improve_rate(SimpleNamespace(obj=4.0), None, 1, "solution")
    assert single[-1] == 4.0


def test_improvement_rate_window_algorithm_and_errors():
    algorithm = SimpleNamespace(avePerformAll=lambda: [3.0, 1.0])
    result = improve_rate(algorithm, [1.0, 4.0], 3, "algorithm")
    assert len(result) == 4
    assert np.isfinite(result[0])
    empty = improve_rate(SimpleNamespace(), None, 1, "algorithm")
    assert np.isinf(empty[-1])
    zeros = improve_rate(
        SimpleNamespace(avePerformAll=lambda: [0.0]),
        [1.0, 0.0, 0.0, 0.0],
        3,
        "algorithm",
    )
    assert zeros[0] == 0.0
    with pytest.raises(ValueError, match="typ must be"):
        improve_rate([], None, 1, "other")


class FakeDesign:
    def __init__(self, values, performance=None):
        self.values = np.asarray(values, dtype=float)
        self.performance = performance
        self.calls = []

    def get_performance(self, setting, seeds):
        return self.values

    def evaluate(self, problem, data, setting, seeds):
        self.calls.append(list(seeds))


def _setting(compare="average", evaluate="exact", alg_n=1):
    return SimpleNamespace(
        Compare=compare, Evaluate=evaluate, AlgN=alg_n, AlgRuns=2, alpha=0.05
    )


def test_racing_requires_missing_or_zero_performance():
    none = FakeDesign([1, 1], None)
    scalar = FakeDesign([1, 1], np.array(0.0))
    short = FakeDesign([1, 1], np.zeros((1, 2)))
    ready = FakeDesign([1, 1], np.ones((2, 2)))
    _require_performance(
        [none, scalar, short, ready], None, None, _setting(evaluate="racing"), [1]
    )
    assert none.calls == []
    assert scalar.calls == [[1]]
    assert short.calls == [[1]]
    assert ready.calls == []
    _require_performance([ready], None, None, _setting(), [0])
    assert ready.calls == []


def test_statistic_wins_fallback_and_significant_ranks(monkeypatch):
    monkeypatch.setattr(
        select_module,
        "friedman_nemenyi",
        lambda matrix: (_ for _ in ()).throw(ValueError("too few")),
    )
    np.testing.assert_array_equal(
        _statistic_wins(np.array([[1, 3], [2, 4]]), 0.05), [1, 0]
    )

    monkeypatch.setattr(
        select_module,
        "friedman_nemenyi",
        lambda matrix: (
            np.array([1.0, 2.0]),
            np.array([[np.nan, 0.01], [0.01, np.nan]]),
        ),
    )
    np.testing.assert_array_equal(_statistic_wins(np.ones((2, 2)), 0.05), [1, 0])


def test_selection_modes_and_failures(monkeypatch):
    better = FakeDesign([1.0, 1.0], np.ones((1, 2)))
    worse = FakeDesign([3.0, 3.0], np.ones((1, 2)))
    assert _ensure_design_list(iter([better, worse])) == [better, worse]
    assert select([], None, None, _setting(), [0]) == []
    assert select([worse, better], None, None, _setting(), [0]) == [better]
    assert select(
        [worse, better], None, None, _setting(evaluate="approximate"), [0]
    ) == [better]
    assert (
        select([better, worse], None, None, _setting(evaluate="intensification"), [0])
        == []
    )
    with pytest.raises(NotImplementedError, match="evaluate mode"):
        select([better], None, None, _setting(evaluate="other"), [0])

    monkeypatch.setattr(
        select_module, "_statistic_wins", lambda matrix, alpha: np.array([0, 1])
    )
    monkeypatch.setattr(
        select_module,
        "_new_algorithm_wins",
        lambda matrix, old, alpha: np.array([0, 1]),
    )
    assert select([better, worse], None, None, _setting("statistic", "exact"), [0]) == [
        worse
    ]
    assert select(
        [better, worse], None, None, _setting("statistic", "intensification"), [0]
    ) == [worse]
    with pytest.raises(NotImplementedError, match="evaluate mode"):
        select([better], None, None, _setting("statistic", "other"), [0])
    with pytest.raises(NotImplementedError, match="compare mode"):
        select([better], None, None, _setting("other"), [0])


def test_design_empty_averages_and_approximate_performance():
    design = Design()
    assert design.ave_perform_all().size == 0
    assert design.ave_perform_approx_all().size == 0
    assert design.ave_perform_per([0]).size == 0
    assert design.ave_perform_approx_per([0]).size == 0
    design.performance_approx = np.array([[1.0, 3.0]])
    setting = SimpleNamespace(Evaluate="approximate", AlgRuns=2)
    np.testing.assert_array_equal(design.get_performance(setting, [0]), [[1.0], [3.0]])
    np.testing.assert_array_equal(design.ave_perform_approx_all(), [2.0])
    np.testing.assert_array_equal(design.ave_perform_approx_per([0]), [2.0])

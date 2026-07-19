"""Focused contracts for solution repair, evaluation, and solve helpers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from autooptlib import ObjectiveEvaluationError
from autooptlib.utils.design._helpers import (
    Pathway,
    PathwayParam,
    SearchParam,
    SearchStep,
)
from autooptlib.utils.solve import (
    Solution,
    SolutionSet,
    _call_evaluator,
    _ensure_solution_instance,
    _extract_mode,
    _init_population,
    _normalize_setting,
    _setting_with_budget,
    _split_indices,
    _to_array,
    input_algorithm,
    make_solutions,
    repair_sol,
    run_algorithm,
    run_design,
)

solve_module = importlib.import_module("autooptlib.utils.solve")


def _problem(problem_type="continuous", bound=None, **overrides):
    if bound is None:
        bound = np.array([[-1.0, -1.0], [1.0, 1.0]])
    values = dict(
        type=[problem_type, "static", "certain"],
        bound=np.asarray(bound),
        N=2,
        Gmax=2,
        evaluate=lambda data, decision: (float(np.sum(decision**2)), 0.0, None),
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_solution_set_aggregations_and_empty_decisions():
    empty = SolutionSet([])
    assert empty.decs().shape == (0, 0)
    solutions = SolutionSet(
        [
            Solution(np.array([1.0]), 2.0, 0.0, 2.0),
            Solution(np.array([2.0]), 4.0, 1.0, 100000001.0),
        ]
    )
    np.testing.assert_array_equal(solutions.decs(), [[1.0], [2.0]])
    np.testing.assert_array_equal(solutions.objs(), [[2.0], [4.0]])
    np.testing.assert_array_equal(solutions.cons(), [[0.0], [1.0]])
    np.testing.assert_array_equal(solutions.fits(), [[2.0], [100000001.0]])


def test_repair_supports_all_spaces_and_distinct_discrete_values():
    raw = np.array([2.0])
    assert repair_sol(raw, SimpleNamespace(type=["continuous"])) is raw
    np.testing.assert_array_equal(
        repair_sol(np.array([-2.0, 2.0]), _problem()), [-1.0, 1.0]
    )
    discrete = _problem(
        "discrete",
        [[0, 0, 0], [2, 2, 2]],
        setting="dec_diff",
        rng=np.random.default_rng(2),
    )
    repaired = repair_sol(np.array([[1.2, 1.2, 2.8]]), discrete)
    assert len(set(repaired[0])) == 3
    np.testing.assert_array_equal(
        repair_sol(np.array([[0.2, 3.0]]), _problem("discrete", [[0, 0], [2, 2]])),
        [[0, 2]],
    )
    permutation = _problem("permutation", [[1, 2, 3], [1, 2, 3]])
    np.testing.assert_array_equal(
        repair_sol(np.array([1, 1, 3]), permutation), [1, 2, 3]
    )
    repaired_rows = repair_sol(np.array([[1, 1, 2], [3, 3, 1]]), permutation)
    assert all(sorted(row.tolist()) == [1, 2, 3] for row in repaired_rows)
    unknown = SimpleNamespace(type=["graph"], bound=[[0], [1]])
    np.testing.assert_array_equal(repair_sol(np.array([2]), unknown), [2])


def test_evaluator_normalizes_result_shapes_and_callable_name():
    decision = np.array([1.0])
    assert _call_evaluator(
        SimpleNamespace(evaluate=lambda d, x: (2.0,)), None, decision
    )[:2] == (2.0, 0.0)
    assert _call_evaluator(
        SimpleNamespace(evaluate=lambda d, x: (2.0, [-1, 3])), None, decision
    )[:2] == (2.0, 3.0)
    assert _call_evaluator(SimpleNamespace(evaluate=lambda d, x: 2.0), None, decision)[
        :2
    ] == (2.0, 0.0)
    assert _call_evaluator(SimpleNamespace(name=lambda d, x: 3.0), None, decision)[
        :2
    ] == (3.0, 0.0)
    with pytest.raises(ObjectiveEvaluationError, match="must provide"):
        _call_evaluator(SimpleNamespace(), None, decision)
    with pytest.raises(ObjectiveEvaluationError, match="Constraint did not return"):
        _call_evaluator(
            SimpleNamespace(evaluate=lambda d, x: (1.0, "bad")), None, decision
        )
    with pytest.raises(ObjectiveEvaluationError, match="non-finite"):
        _call_evaluator(
            SimpleNamespace(evaluate=lambda d, x: (1.0, [np.nan])), None, decision
        )


def test_make_solutions_applies_constraint_penalty():
    problem = _problem(evaluate=lambda data, decision: (1.0, [2.0], None))
    solution = make_solutions(np.array([[0.0, 0.0]]), problem, None)[0]
    assert solution.fit == 100000002.0


def test_population_initialization_errors_and_index_splitting():
    rng = np.random.default_rng(3)
    malformed = _problem("permutation", np.empty((1, 0)), dimension=0, D=0)
    with pytest.raises(ValueError, match="requires explicit dimension"):
        _init_population(rng, malformed, None, SimpleNamespace(ProbN=2))
    with pytest.raises(NotImplementedError, match="Unsupported problem type"):
        _init_population(
            rng,
            _problem("graph"),
            None,
            SimpleNamespace(ProbN=2),
        )
    assert _split_indices([], 1, 3) == [[0, 1, 2]]
    assert _split_indices([0, 1], 3, 2) == [[0], [1], [0, 1]]


def test_run_design_rejects_budget_smaller_than_initial_population():
    algorithm, _ = input_algorithm(SimpleNamespace(AlgName="Continuous Random Search"))
    with pytest.raises(ValueError, match="initial population size"):
        run_design(
            algorithm.operator_pheno[0],
            algorithm.parameter_pheno[0],
            _problem(N=3),
            None,
            SimpleNamespace(ProbN=3, ProbFE=2),
        )


def test_serial_path_updates_population_after_each_search_step(monkeypatch):
    updates = []

    def choose_all(population, *args):
        return np.arange(len(population)), None

    def add_one(parent, *args):
        return parent.decs() + 1.0, args[2]

    def add_ten(parent, *args):
        return parent.decs() + 10.0, args[2]

    def update_to_new(combined, problem, *args):
        updates.append(1)
        return combined[-problem.N :], None

    components = {
        "choose_test": choose_all,
        "search_add_one": add_one,
        "search_add_ten": add_ten,
        "update_test": update_to_new,
    }
    original_get_component = solve_module.get_component

    def lookup_component(name):
        if name in components:
            return components[name]
        return original_get_component(name)

    monkeypatch.setattr(solve_module, "get_component", lookup_component)
    path = Pathway(
        "choose_test",
        [
            SearchStep("search_add_one", np.array([-np.inf, 1.0])),
            SearchStep("search_add_ten", np.array([-np.inf, 1.0])),
        ],
        "update_test",
        [],
    )
    parameters = PathwayParam(
        None,
        [SearchParam(None, None), SearchParam(None, None)],
        None,
    )
    result = run_design(
        [path],
        [parameters],
        _problem(bound=[[-100.0, -100.0], [100.0, 100.0]], Gmax=2),
        None,
        SimpleNamespace(ProbN=2, ProbFE=6, Seed=3),
    )
    assert len(updates) == 2
    assert result["evaluations"] == 6


def test_parallel_path_helper_repairs_secondary_and_truncates(monkeypatch):
    def primary(parent, *args):
        return parent.decs() + 2.0, args[2]

    def secondary(parent, *args):
        return np.asarray(parent) + 1.0, args[2]

    components = {"primary_test": primary, "secondary_test": secondary}
    original_get_component = solve_module.get_component

    def lookup_component(name):
        if name in components:
            return components[name]
        return original_get_component(name)

    monkeypatch.setattr(solve_module, "get_component", lookup_component)
    path = Pathway(
        "choose_traverse",
        [
            SearchStep(
                "primary_test",
                np.array([-np.inf, 1.0]),
                "secondary_test",
            )
        ],
        "update_greedy",
        [],
    )
    parameters = PathwayParam(None, [SearchParam(None, None)], None)
    parent = SolutionSet(
        [
            Solution(np.array([0.0, 0.0]), 0.0, 0.0, 0.0),
            Solution(np.array([1.0, 1.0]), 2.0, 0.0, 2.0),
        ]
    )
    produced, aux, evaluations = solve_module._execute_path(
        path,
        parameters,
        parent,
        [],
        _problem(bound=[[-10.0, -10.0], [10.0, 10.0]]),
        None,
        SimpleNamespace(Seed=4),
        1,
        1,
    )
    assert evaluations == 1
    assert len(produced) == 1
    assert len(aux) == 1


def test_solve_helper_normalization_and_algorithm_errors(tmp_path):
    normalized = _normalize_setting({"AlgName": "Continuous Random Search"})
    assert normalized.AlgName == "Continuous Random Search"
    setting = SimpleNamespace(ProbFE=3)
    _normalize_setting(setting)
    assert setting.probfe == 3
    assert _to_array(None) is None
    assert _to_array([]) is None
    assert np.asarray(_to_array(2.0)).shape == ()
    np.testing.assert_array_equal(_to_array([[1, 2]]), [1, 2])

    solution = Solution(np.array([1]), 1, 0, 1)
    assert _ensure_solution_instance(solution) is solution
    assert np.isinf(_ensure_solution_instance(None).fit)
    converted = _ensure_solution_instance(SimpleNamespace(dec=[1], fit=2))
    assert converted.fit == 2
    assert np.isinf(_ensure_solution_instance(object()).fit)
    assert _extract_mode(SimpleNamespace(type="continuous")) == "static"
    assert _setting_with_budget({}, 7) == {"ProbFE": 7, "prob_fe": 7}

    with pytest.raises(FileNotFoundError):
        input_algorithm(SimpleNamespace(AlgFile=tmp_path / "missing.json"))
    with pytest.raises(ValueError, match="specify AlgFile"):
        input_algorithm(SimpleNamespace(AlgName=""))
    with pytest.raises(NotImplementedError, match="not available"):
        input_algorithm(SimpleNamespace(AlgName="Unknown Optimizer"))


def test_run_algorithm_static_fallback_progress_and_unknown_mode(monkeypatch):
    algorithm, _ = input_algorithm(SimpleNamespace(AlgName="Continuous Random Search"))
    solution = Solution(np.array([0.0]), 1.0, 0.0, 1.0)
    monkeypatch.setattr(
        solve_module,
        "run_design",
        lambda *args: {"history": [solution], "best_solution": None},
    )
    app = SimpleNamespace(TextArea=SimpleNamespace(Value=""))
    best, histories = run_algorithm(
        algorithm,
        [_problem()],
        [],
        app,
        SimpleNamespace(AlgRuns=1),
    )
    assert best[0][0] is solution
    assert histories[0] == [solution]
    assert app.TextArea.Value == "Solving... 100.0%"

    with pytest.raises(NotImplementedError, match="Unsupported problem mode"):
        run_algorithm(
            algorithm,
            [_problem(type=["continuous", "dynamic"])],
            [None],
            None,
            SimpleNamespace(AlgRuns=1),
        )

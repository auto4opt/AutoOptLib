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
    input_algorithm,
    make_solutions,
    repair_sol,
    run_algorithm,
    run_design,
)

from autooptlib.presets import _to_array

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


def test_continuous_boundary_repair_supports_reflect_and_resample() -> None:
    reflected = _problem(
        _AlgorithmBoundaryHandling="reflect",
        rng=np.random.default_rng(4),
    )
    np.testing.assert_allclose(repair_sol(np.array([-3.5, 4.5]), reflected), [0.5, 0.5])

    resampled = _problem(
        _AlgorithmBoundaryHandling="resample",
        rng=np.random.default_rng(7),
    )
    original = np.array([0.25, 4.0])
    repaired = repair_sol(original, resampled)
    assert repaired[0] == pytest.approx(0.25)
    assert -1.0 <= repaired[1] <= 1.0
    assert repaired[1] != pytest.approx(1.0)


def test_continuous_boundary_repair_eliminates_nonfinite_coordinates() -> None:
    values = np.asarray([np.nan, np.inf, -np.inf, 12.0])
    for strategy in ("clip", "reflect", "resample"):
        problem = _problem(
            bound=np.asarray([[0.0] * 4, [10.0] * 4]),
            _AlgorithmBoundaryHandling=strategy,
            rng=np.random.default_rng(17),
        )
        repaired = repair_sol(values, problem)
        assert np.all(np.isfinite(repaired))
        assert np.all((repaired >= 0.0) & (repaired <= 10.0))


def test_continuous_boundary_repair_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported continuous boundary"):
        repair_sol(
            np.array([2.0, 0.0]),
            _problem(_AlgorithmBoundaryHandling="wrap"),
        )


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


def test_problem_repair_hook_arity_is_resolved_without_masking_errors():
    one_argument = _problem(repair=lambda decision: np.zeros_like(decision))
    repaired = make_solutions(np.array([[0.5, -0.5]]), one_argument, None)[0]
    np.testing.assert_array_equal(repaired.dec, [0.0, 0.0])

    def broken_repair(data, decision):
        raise TypeError("bug inside repair hook")

    broken = _problem(repair=broken_repair)
    with pytest.raises(TypeError, match="bug inside repair hook"):
        make_solutions(np.array([[0.5, -0.5]]), broken, None)


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
    assert _split_indices([0, 1], 3, 2) == [[0], [1], []]


def test_run_design_honors_population_and_offspring_sizes(monkeypatch):
    parent_counts = []

    def choose_all(population, *args):
        return np.arange(len(population)), None

    def record_parents(parent, *args):
        parent_counts.append(len(parent))
        return parent.decs(), args[2]

    components = {"choose_test": choose_all, "search_test": record_parents}
    original_get_component = solve_module.get_component

    def lookup_component(name):
        if name in components:
            return components[name]
        return original_get_component(name)

    monkeypatch.setattr(solve_module, "get_component", lookup_component)
    path = Pathway(
        "choose_test",
        [SearchStep("search_test", np.array([-np.inf, 1.0]))],
        "update_greedy",
        [],
    )
    result = run_design(
        [path],
        [PathwayParam(None, [SearchParam(None, None)], None)],
        _problem(N=50, Gmax=1),
        None,
        SimpleNamespace(ProbN=50, ProbFE=7, Seed=3),
        algorithm_configuration={"population_size": 3, "offspring_size": 2},
    )

    assert parent_counts == [2, 2]
    assert len(result["solutions"]) == 3
    assert result["evaluations"] == 7


def test_new_continuous_components_execute_through_graph_runtime() -> None:
    path = Pathway(
        "choose_elite_fraction",
        [
            SearchStep(
                "search_de_current_to_pbest",
                np.array([-np.inf, 1.0]),
            )
        ],
        "update_pairwise",
        [],
    )
    parameters = PathwayParam(
        np.array([0.5]),
        [SearchParam(np.array([0.5, 0.7, 0.25]), None)],
        None,
    )
    result = run_design(
        [path],
        [parameters],
        _problem(N=6, Gmax=3),
        None,
        SimpleNamespace(ProbN=6, ProbFE=18, Seed=29),
        algorithm_configuration={
            "population_size": 6,
            "offspring_size": 6,
            "crossover_rate": 1.0,
            "mutation_rate": 1.0,
            "boundary_handling": "reflect",
        },
    )

    assert result["evaluations"] == 18
    assert len(result["solutions"]) == 6
    assert all(np.all(np.isfinite(solution.dec)) for solution in result["solutions"])


def test_negative_objective_improvements_do_not_stop_local_search(monkeypatch):
    original_get_component = solve_module.get_component

    def improving_search(*args):
        parent = args[0]
        auxiliary = args[3]
        return parent.decs() + 1.0, auxiliary

    def lookup_component(name):
        if name == "search_negative_improvement_test":
            return improving_search
        return original_get_component(name)

    monkeypatch.setattr(solve_module, "get_component", lookup_component)
    path = Pathway(
        "choose_traverse",
        [
            SearchStep(
                "search_negative_improvement_test",
                np.array([0.05, 10.0]),
            )
        ],
        "update_greedy",
        [],
    )
    problem = SimpleNamespace(
        type=["continuous", "static", "certain"],
        bound=np.asarray([[-100.0], [100.0]]),
        N=1,
        Gmax=20,
        evaluate=lambda data, decision: (-float(decision[0]), 0.0, None),
    )
    result = run_design(
        [path],
        [PathwayParam(None, [SearchParam(None, None)], None)],
        problem,
        None,
        SimpleNamespace(
            ProbN=1,
            ProbFE=5,
            Seed=3,
            _InitialPopulation=np.asarray([[0.0]]),
        ),
    )

    assert result["evaluations"] == 5
    assert result["fit_history"][-1] == -4.0


def test_solve_checkpoint_rejects_changed_execution_semantics(tmp_path):
    checkpoint = tmp_path / "solve.pkl"
    path = Pathway(
        "choose_traverse",
        [SearchStep("reinit_continuous", np.array([-np.inf, 1.0]))],
        "update_greedy",
        [],
    )
    params = [PathwayParam(None, [SearchParam(None, None)], None)]
    setting = SimpleNamespace(
        ProbN=2,
        ProbFE=4,
        Seed=9,
        _checkpoint_path=str(checkpoint),
        CheckpointEvery=1,
    )
    problem = _problem(N=2, Gmax=2)
    run_design([path], params, problem, None, setting)

    changed_metric = SimpleNamespace(
        **vars(setting), Resume=True, Metric="runtimeFE", Tmax=4
    )
    with pytest.warns(UserWarning, match="trusted run"):
        with pytest.raises(ValueError, match="checkpoint_signature"):
            run_design([path], params, problem, None, changed_metric)

    changed_problem = _problem(
        bound=[[-2.0, -2.0], [2.0, 2.0]],
        N=2,
        Gmax=2,
    )
    with pytest.warns(UserWarning, match="trusted run"):
        with pytest.raises(ValueError, match="checkpoint_signature"):
            run_design(
                [path],
                params,
                changed_problem,
                None,
                SimpleNamespace(**vars(setting), Resume=True),
            )

    changed_path = Pathway(
        "choose_traverse",
        [SearchStep("search_mu_uniform", np.array([-np.inf, 1.0]))],
        "update_greedy",
        [],
    )
    changed_params = [PathwayParam(None, [SearchParam(np.array([0.2]), None)], None)]
    with pytest.warns(UserWarning, match="trusted run"):
        with pytest.raises(ValueError, match="checkpoint_signature"):
            run_design(
                [changed_path],
                changed_params,
                problem,
                None,
                SimpleNamespace(**vars(setting), Resume=True),
            )


def test_run_design_rejects_misaligned_pso_population_state():
    path = Pathway(
        "choose_traverse",
        [SearchStep("search_pso", np.array([-np.inf, 1.0]))],
        "update_always",
        [],
    )
    with pytest.raises(ValueError, match="offspring_size == population_size"):
        run_design(
            [path],
            [PathwayParam(None, [SearchParam(np.array([0.5]), None)], None)],
            _problem(N=3, Gmax=2),
            None,
            SimpleNamespace(
                ProbFE=6,
                Seed=2,
                _AlgorithmPopulationSize=3,
                _AlgorithmOffspringSize=2,
            ),
        )


def test_sequential_solve_scores_runtime_fe_and_uses_stage_budgets():
    problem = _problem(N=2, Gmax=10)

    def advance(_best, data):
        stage = data.stage + 1
        next_data = SimpleNamespace(
            stage=stage,
            task=SimpleNamespace(budget=4),
        )
        setattr(next_data, "continue", stage < 2)
        return problem, next_data

    problem.type = ["continuous", "sequential", "certain"]
    problem.advance_sequence = advance
    data = SimpleNamespace(stage=0, task=SimpleNamespace(budget=3))
    setattr(data, "continue", True)
    path = Pathway(
        "choose_traverse",
        [SearchStep("reinit_continuous", np.array([-np.inf, 1.0]))],
        "update_greedy",
        [],
    )
    session = solve_module._SolveSession(
        pathways=[path],
        params=[PathwayParam(None, [SearchParam(None, None)], None)],
        problems=[problem],
        data=[data],
        setting=SimpleNamespace(
            ProbN=2,
            ProbFE=10,
            Metric="runtimeFE",
            Tmax=np.inf,
            Thres=-np.inf,
            Seed=4,
        ),
        custom_components=(),
    )
    result = solve_module._run_solve_task(
        session,
        solve_module._SolveTask(instance=0, run=0, random_seed=19),
    )

    assert len(result[3]) == 2
    assert result[-1] == 7.0


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


def test_parallel_path_helper_executes_complete_search_sequence(monkeypatch):
    calls = []

    def primary(parent, *args):
        calls.append("primary")
        return parent.decs() + 2.0, args[2]

    def secondary(parent, *args):
        calls.append("secondary")
        return np.asarray(parent) + 1.0, args[2]

    def later(parent, *args):
        calls.append("later")
        return parent.decs() + 10.0, args[2]

    components = {
        "primary_test": primary,
        "secondary_test": secondary,
        "later_test": later,
    }
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
                np.array([-np.inf, 5.0]),
                "secondary_test",
            ),
            SearchStep("later_test", np.array([-np.inf, 5.0])),
        ],
        "update_greedy",
        [],
    )
    parameters = PathwayParam(
        None, [SearchParam(None, None), SearchParam(None, None)], None
    )
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
        4,
    )
    assert evaluations == 4
    assert len(produced) == 2
    assert len(aux) == 1
    assert calls == ["primary", "secondary", "later"]


def test_crossover_and_mutation_rates_independently_control_effects(monkeypatch):
    def crossover(parent, *args):
        return parent.decs() + 10.0, args[2]

    def mutation(decisions, *args):
        values = decisions.decs() if hasattr(decisions, "decs") else decisions
        return np.asarray(values) + 1.0, args[2]

    original_get_component = solve_module.get_component

    def lookup_component(name):
        if name == "cross_test":
            return crossover
        if name == "search_mu_test":
            return mutation
        return original_get_component(name)

    monkeypatch.setattr(solve_module, "get_component", lookup_component)
    parent = SolutionSet(
        [
            Solution(np.array([0.0, 0.0]), 0.0, 0.0, 0.0),
            Solution(np.array([1.0, 1.0]), 2.0, 0.0, 2.0),
        ]
    )
    step = SearchStep("cross_test", np.array([-np.inf, 1.0]), "search_mu_test")
    params = SearchParam(None, None)
    problem = _problem(bound=[[-100.0, -100.0], [100.0, 100.0]])

    unchanged, _ = solve_module._execute_search_operators(
        parent,
        step,
        params,
        {},
        problem,
        None,
        SimpleNamespace(
            Seed=2, _AlgorithmCrossoverRate=0.0, _AlgorithmMutationRate=0.0
        ),
        1,
        1,
    )
    crossed, _ = solve_module._execute_search_operators(
        parent,
        step,
        params,
        {},
        problem,
        None,
        SimpleNamespace(
            Seed=2, _AlgorithmCrossoverRate=1.0, _AlgorithmMutationRate=0.0
        ),
        1,
        1,
    )
    mutated, _ = solve_module._execute_search_operators(
        parent,
        step,
        params,
        {},
        problem,
        None,
        SimpleNamespace(
            Seed=2, _AlgorithmCrossoverRate=0.0, _AlgorithmMutationRate=1.0
        ),
        1,
        1,
    )
    direct_mutation, _ = solve_module._execute_search_operators(
        parent,
        SearchStep("search_mu_test", np.array([-np.inf, 1.0])),
        SearchParam(None, None),
        {},
        problem,
        None,
        SimpleNamespace(Seed=2, _AlgorithmMutationRate=0.0),
        1,
        1,
    )

    np.testing.assert_array_equal(unchanged, parent.decs())
    np.testing.assert_array_equal(crossed, parent.decs() + 10.0)
    np.testing.assert_array_equal(mutated, parent.decs() + 1.0)
    np.testing.assert_array_equal(direct_mutation, parent.decs())


def test_lambda_one_crossover_receives_two_distinct_parents(monkeypatch):
    captured = []

    def crossover(parent, *args):
        decisions = parent.decs()
        captured.append(decisions.copy())
        child = np.mean(decisions, axis=0, keepdims=True)
        return np.vstack((child, child)), args[2]

    original_get_component = solve_module.get_component

    def lookup_component(name):
        if name == "cross_probe":
            return crossover
        return original_get_component(name)

    monkeypatch.setattr(solve_module, "get_component", lookup_component)
    path = Pathway(
        "choose_traverse",
        [SearchStep("cross_probe", np.array([-np.inf, 1.0]))],
        "update_greedy",
        [],
    )
    parameters = PathwayParam(None, [SearchParam(None, None)], None)
    population = SolutionSet(
        [
            Solution(np.array([float(index)]), float(index), 0.0, float(index))
            for index in range(4)
        ]
    )
    problem = _problem(bound=[[-10.0], [10.0]], N=4, Gmax=1)

    _, state, evaluations, _, _, _ = solve_module._execute_single_path(
        path,
        parameters,
        population,
        {},
        problem,
        None,
        SimpleNamespace(
            Seed=4,
            _AlgorithmOffspringSize=1,
            _AlgorithmCrossoverRate=1.0,
            _AlgorithmMutationRate=1.0,
        ),
        1,
        1,
        [],
        [],
    )

    assert evaluations == 1
    assert captured[0].shape == (2, 1)
    assert captured[0][0, 0] != captured[0][1, 0]
    assert state["choose_traverse_cursor"] == 2


def test_multi_path_execution_rejects_nonshared_choose_or_update():
    first = Pathway(
        "choose_traverse",
        [SearchStep("reinit_continuous", np.array([-np.inf, 1.0]))],
        "update_greedy",
        [],
    )
    second = Pathway(
        "choose_random",
        [SearchStep("reinit_continuous", np.array([-np.inf, 1.0]))],
        "update_greedy",
        [],
    )
    parameters = PathwayParam(None, [SearchParam(None, None)], None)

    with pytest.raises(ValueError, match="must share choose and update"):
        run_design(
            [first, second],
            [parameters, parameters],
            _problem(N=2, Gmax=1),
            None,
            SimpleNamespace(ProbN=2, ProbFE=4, Seed=3),
        )


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

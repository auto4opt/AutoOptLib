"""Reliability tests for public configuration, budgets, and serialization."""

from __future__ import annotations

import importlib
import json
import multiprocessing
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from autooptlib import (
    ObjectiveEvaluationError,
    autoopt,
    load_algorithm,
    make_problem,
    save_algorithm,
)
from autooptlib.serialization import algorithm_to_dict
from autooptlib.utils.solve import (
    _configure_evaluation_runtime,
    input_algorithm,
    make_solutions,
    run_design,
)


def test_unknown_public_option_is_rejected():
    with pytest.raises(TypeError, match="Unknown AutoOpt option.*ProbFee"):
        autoopt(Mode="solve", ProbFee=10)


def test_public_options_accept_snake_case_and_output_directory(tmp_path):
    problem = make_problem(
        lambda decision, dimension: float(np.sum(decision**2)),
        bounds=(-5.0, 5.0),
    )
    algorithms, _ = autoopt(
        mode="design",
        problem=problem,
        instance_train=[3],
        instance_test=[3],
        alg_n=2,
        alg_fe=4,
        prob_n=5,
        prob_fe=10,
        archive=["archive_best"],
        tune_para=True,
        seed=8,
        output_dir=tmp_path,
    )
    assert algorithms[0].operator_pheno[0][0].archive == ["archive_best"]
    assert (tmp_path / "Algorithm_1.json").is_file()


def test_problem_factory_and_design_budgets_count_every_objective_call(tmp_path):
    calls = {"count": 0}

    def objective(decision, dimension):
        calls["count"] += 1
        return float(np.sum(decision**2))

    problem = make_problem(objective, bounds=(-5.0, 5.0), name="counted_sphere")
    algorithms, _ = autoopt(
        Mode="design",
        Problem=problem,
        InstanceTrain=[3],
        InstanceTest=[4],
        AlgN=2,
        AlgFE=4,
        ProbN=5,
        ProbFE=10,
        AlgRuns=1,
        Seed=9,
        OutputDir=tmp_path,
    )
    # Initialization evaluates AlgN incumbents; AlgFE then counts four new
    # proposals. Two selected algorithms are finally run on held-out data.
    assert calls["count"] == (2 + 4 + 2) * 10
    assert len(algorithms) == 2


@pytest.mark.parametrize(
    ("algorithm_name", "problem_type", "bounds"),
    [
        ("Discrete Iterative Local Search", "discrete", (0, 5)),
        ("Permutation Variable Neighborhood Search", "permutation", None),
    ],
)
def test_multistep_algorithms_never_exceed_probfe(algorithm_name, problem_type, bounds):
    calls = {"count": 0}

    def objective(decision, instance):
        calls["count"] += 1
        return float(np.sum(np.asarray(decision, dtype=float) ** 2))

    definition = make_problem(objective, bounds=bounds, problem_type=problem_type)
    records = [SimpleNamespace(N=10, Gmax=10)]
    problems, data, _ = definition(records, [5], "construct")
    setting = SimpleNamespace(
        AlgName=algorithm_name,
        AlgRuns=1,
        ProbN=10,
        ProbFE=100,
        Metric="quality",
        rng=np.random.default_rng(1),
    )
    algorithm, setting = input_algorithm(setting)
    result = run_design(
        algorithm.operator_pheno[0],
        algorithm.parameter_pheno[0],
        problems[0],
        data[0],
        setting,
    )
    assert result["evaluations"] == 100
    assert calls["count"] == 100


@pytest.mark.parametrize(
    ("algorithm_name", "problem_type"),
    [
        ("Continuous Genetic Algorithm", "continuous"),
        ("Evolutionary Programming", "continuous"),
        ("Fast Evolutionary Programming", "continuous"),
        ("CMA-ES", "continuous"),
        ("Estimation of Distribution", "continuous"),
        ("Particle Swarm Optimization", "continuous"),
        ("Differential Evolution", "continuous"),
        ("Continuous Random Search", "continuous"),
        ("ICA", "continuous"),
        ("Discrete Genetic Algorithm", "discrete"),
        ("Discrete Iterative Local Search", "discrete"),
        ("Discrete Simulated Annealing", "discrete"),
        ("Discrete Random Search", "discrete"),
        ("Permutation Genetic Algorithm", "permutation"),
        ("Permutation Iterative Local Search", "permutation"),
        ("Permutation Simulated Annealing", "permutation"),
        ("Permutation Variable Neighborhood Search", "permutation"),
        ("Permutation Random Search", "permutation"),
    ],
)
def test_every_preset_handles_a_partial_final_population(algorithm_name, problem_type):
    calls = {"count": 0}

    def objective(decision, instance):
        calls["count"] += 1
        return float(np.sum(np.asarray(decision, dtype=float) ** 2))

    if problem_type == "permutation":
        bounds = None
    elif problem_type == "discrete":
        bounds = (0, 5)
    else:
        bounds = (-5.0, 5.0)
    definition = make_problem(objective, bounds=bounds, problem_type=problem_type)
    problems, data, _ = definition([SimpleNamespace(N=7, Gmax=10)], [6], "construct")
    setting = SimpleNamespace(
        AlgName=algorithm_name,
        ProbN=7,
        ProbFE=23,
        Metric="quality",
        rng=np.random.default_rng(3),
    )
    algorithm, setting = input_algorithm(setting)
    result = run_design(
        algorithm.operator_pheno[0],
        algorithm.parameter_pheno[0],
        problems[0],
        data[0],
        setting,
    )
    assert result["evaluations"] == 23
    assert calls["count"] == 23


def test_json_algorithm_round_trip_and_schema_validation(tmp_path):
    setting = SimpleNamespace(AlgName="Continuous Random Search")
    algorithm, _ = input_algorithm(setting)
    target = save_algorithm(
        algorithm, tmp_path / "algorithm.json", metadata={"purpose": "test"}
    )
    document = json.loads(target.read_text(encoding="utf-8"))
    assert document["schema"] == "autooptlib.algorithm"
    assert document["schema_version"] == 1
    assert document["metadata"] == {"purpose": "test"}

    restored = load_algorithm(target)
    assert restored.operator_pheno[0][0].search[0].primary == "reinit_continuous"
    assert restored.parameter_pheno[0][0].search[0].primary is None

    document["schema_version"] = 999
    target.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported algorithm schema version"):
        load_algorithm(target)


def test_invalid_constructed_problem_fails_before_optimization(tmp_path):
    invalid = make_problem(lambda decision, data: 0.0, bounds=[[1.0], [-1.0]])
    with pytest.raises(ValueError, match="lower bounds greater"):
        autoopt(
            Mode="solve",
            Problem=invalid,
            InstanceSolve=[1],
            AlgName="Continuous Random Search",
            ProbN=5,
            ProbFE=10,
            OutputDir=tmp_path,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (np.nan, "non-finite"),
        ([1.0, 2.0], "one scalar objective"),
        ("not-a-number", "numeric scalar"),
        ((), "one scalar objective"),
    ],
)
def test_invalid_objective_results_fail_with_context(value, message):
    definition = make_problem(lambda decision, instance: value, bounds=(-1.0, 1.0))
    records = [SimpleNamespace(N=2, Gmax=1)]
    problems, data, _ = definition(records, [2], "construct")
    algorithm, _ = input_algorithm(SimpleNamespace(AlgName="Continuous Random Search"))
    with pytest.raises(ObjectiveEvaluationError, match=message):
        run_design(
            algorithm.operator_pheno[0],
            algorithm.parameter_pheno[0],
            problems[0],
            data[0],
            SimpleNamespace(ProbN=2, ProbFE=2),
        )


def test_objective_exception_is_wrapped_with_decision_context():
    def broken_objective(decision, instance):
        raise TimeoutError("simulator timed out")

    definition = make_problem(broken_objective, bounds=(-1.0, 1.0))
    problems, data, _ = definition([SimpleNamespace(N=2, Gmax=1)], [2], "construct")
    algorithm, _ = input_algorithm(SimpleNamespace(AlgName="Continuous Random Search"))
    with pytest.raises(
        ObjectiveEvaluationError,
        match="Objective evaluation failed.*simulator timed out",
    ):
        run_design(
            algorithm.operator_pheno[0],
            algorithm.parameter_pheno[0],
            problems[0],
            data[0],
            SimpleNamespace(ProbN=2, ProbFE=2),
        )


def test_failed_objective_is_retried_then_succeeds(tmp_path):
    calls = {"count": 0}

    def flaky(decision, instance):
        calls["count"] += 1
        if calls["count"] <= 2:
            raise ConnectionError("temporary simulator failure")
        return float(np.sum(decision**2))

    problem = make_problem(flaky, bounds=(-1.0, 1.0))
    best, _ = autoopt(
        Mode="solve",
        Problem=problem,
        InstanceSolve=[1],
        AlgName="Continuous Random Search",
        AlgRuns=1,
        ProbN=2,
        ProbFE=2,
        EvalRetries=2,
        OutputDir=tmp_path,
    )
    assert calls["count"] == 4
    assert np.isfinite(best[0][0].fit)


def test_penalty_log_and_cache_are_opt_in(tmp_path):
    calls = {"count": 0}

    def broken(decision, instance):
        calls["count"] += 1
        raise RuntimeError("unavailable")

    log_path = tmp_path / "events.jsonl"
    problem = make_problem(broken, bounds=(0.0, 0.0))
    best, _ = autoopt(
        Mode="solve",
        Problem=problem,
        InstanceSolve=[1],
        AlgName="Continuous Random Search",
        ProbN=2,
        ProbFE=2,
        AlgRuns=1,
        EvalRetries=1,
        EvalFailure="penalize",
        EvalPenalty=1234.0,
        EvalLog=log_path,
        OutputDir=tmp_path / "penalty",
    )
    assert best[0][0].fit == 1234.0
    assert calls["count"] == 4
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [event["status"] for event in events] == ["failure"] * 4
    assert all(len(event["decision_sha256"]) == 64 for event in events)

    cached_calls = {"count": 0}

    def constant(decision, instance):
        cached_calls["count"] += 1
        return 1.0

    cached = make_problem(constant, bounds=(0.0, 0.0))
    autoopt(
        Mode="solve",
        Problem=cached,
        InstanceSolve=[1],
        AlgName="Continuous Random Search",
        ProbN=3,
        ProbFE=6,
        EvalCache=True,
        OutputDir=tmp_path / "cache",
    )
    assert cached_calls["count"] == 1


def test_candidate_evaluations_can_run_in_parallel(tmp_path):
    rendezvous = threading.Barrier(2)
    thread_ids = set()
    lock = threading.Lock()

    def synchronized_objective(decision, instance):
        with lock:
            thread_ids.add(threading.get_ident())
        rendezvous.wait(timeout=2)
        return float(np.sum(decision**2))

    problem = make_problem(synchronized_objective, bounds=(-1.0, 1.0))
    best, _ = autoopt(
        Mode="solve",
        Problem=problem,
        InstanceSolve=[1],
        AlgName="Continuous Random Search",
        AlgRuns=1,
        ProbN=2,
        ProbFE=2,
        EvalWorkers=2,
        OutputDir=tmp_path,
    )
    assert len(thread_ids) == 2
    assert np.isfinite(best[0][0].fit)


def test_parallel_evaluation_preserves_order_cache_logs_and_retries(tmp_path):
    calls = {}
    call_lock = threading.Lock()

    def objective(data, decision):
        value = int(np.asarray(decision).reshape(-1)[0])
        with call_lock:
            calls[value] = calls.get(value, 0) + 1
            attempt = calls[value]
        time.sleep(0.005 * (5 - value))
        if value == 2 and attempt == 1:
            raise ConnectionError("transient")
        if value == 4:
            raise RuntimeError("permanent")
        return float(value)

    problem = SimpleNamespace(
        type=["continuous", "static", "certain"],
        bound=np.array([[0.0], [4.0]]),
        evaluate=objective,
        name="parallel_policy_test",
    )
    log_path = tmp_path / "parallel.jsonl"
    _configure_evaluation_runtime(
        problem,
        SimpleNamespace(
            EvalWorkers=4,
            EvalRetries=1,
            EvalFailure="penalize",
            EvalPenalty=99.0,
            EvalCache=True,
            EvalLog=log_path,
        ),
    )
    solutions = make_solutions(
        np.array([[3.0], [1.0], [2.0], [1.0], [4.0]]), problem, None
    )

    assert [solution.obj for solution in solutions] == [3.0, 1.0, 2.0, 1.0, 99.0]
    assert calls == {1: 1, 2: 2, 3: 1, 4: 2}
    events = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert len(events) == 7
    assert sum(event["status"] == "cache_hit" for event in events) == 1
    assert sum(event["status"] == "failure" for event in events) == 3
    assert sum(event["status"] == "success" for event in events) == 3


def test_parallel_seeded_runs_are_deterministic(tmp_path):
    problem = make_problem(
        lambda decision, dimension: float(np.sum(decision**2)),
        bounds=(-5.0, 5.0),
    )
    common = dict(
        Mode="solve",
        Problem=problem,
        InstanceSolve=[4],
        AlgName="Continuous Random Search",
        ProbN=8,
        ProbFE=32,
        Seed=2026,
        EvalWorkers=4,
    )
    first_best, first_history = autoopt(**common, OutputDir=tmp_path / "parallel-first")
    second_best, second_history = autoopt(
        **common, OutputDir=tmp_path / "parallel-second"
    )
    np.testing.assert_array_equal(first_best[0][0].dec, second_best[0][0].dec)
    assert [solution.fit for solution in first_history[0]] == [
        solution.fit for solution in second_history[0]
    ]


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(),
    reason="closure-based timeout test requires the fork start method",
)
def test_objective_timeout_can_be_penalized(tmp_path):
    def slow(decision, instance):
        time.sleep(0.2)
        return 1.0

    problem = make_problem(slow, bounds=(-1.0, 1.0))
    best, _ = autoopt(
        Mode="solve",
        Problem=problem,
        InstanceSolve=[1],
        AlgName="Continuous Random Search",
        ProbN=1,
        ProbFE=1,
        EvalTimeoutSec=0.02,
        EvalFailure="penalize",
        EvalPenalty=999.0,
        OutputDir=tmp_path,
    )
    assert best[0][0].fit == 999.0


def test_experiment_manifest_records_environment_and_options(tmp_path):
    problem = make_problem(lambda decision, instance: 0.0, bounds=(0.0, 0.0))
    autoopt(
        Mode="solve",
        Problem=problem,
        InstanceSolve=[2],
        AlgName="Continuous Random Search",
        ProbN=1,
        ProbFE=1,
        Seed=42,
        OutputDir=tmp_path,
    )
    manifest = json.loads((tmp_path / "experiment.json").read_text())
    assert manifest["schema"] == "autooptlib.experiment"
    assert manifest["software"]["autooptlib"] == "1.3.0"
    assert manifest["options"]["Seed"] == 42
    assert manifest["options"]["Problem"].endswith(":make_problem.<locals>.definition")


def test_completed_solve_can_resume_from_atomic_checkpoint_without_evaluation(tmp_path):
    calls = {"count": 0}

    def objective(decision, instance):
        calls["count"] += 1
        return float(np.sum(decision**2))

    problem = make_problem(objective, bounds=(-1.0, 1.0))
    common = dict(
        Mode="solve",
        Problem=problem,
        InstanceSolve=[2],
        AlgName="Continuous Random Search",
        ProbN=3,
        ProbFE=6,
        Seed=10,
        CheckpointDir=tmp_path / "checkpoints",
    )
    first, _ = autoopt(**common, OutputDir=tmp_path / "first")
    calls_after_first = calls["count"]
    checkpoint = tmp_path / "checkpoints" / "instance_1_run_1.pkl"
    assert checkpoint.is_file()
    assert not checkpoint.with_suffix(".pkl.tmp").exists()

    with pytest.warns(UserWarning, match="trusted run"):
        second, _ = autoopt(**common, Resume=True, OutputDir=tmp_path / "second")
    assert calls["count"] == calls_after_first
    assert second[0][0].fit == first[0][0].fit


def test_completed_design_can_resume_without_objective_evaluation(tmp_path):
    calls = {"count": 0}

    def objective(decision, instance):
        calls["count"] += 1
        return float(np.sum(decision**2))

    problem = make_problem(objective, bounds=(-1.0, 1.0))
    common = dict(
        Mode="design",
        Problem=problem,
        InstanceTrain=[2],
        InstanceTest=[2],
        ProbN=2,
        ProbFE=4,
        AlgN=1,
        AlgFE=2,
        AlgRuns=1,
        Seed=12,
        CheckpointDir=tmp_path / "checkpoints",
    )
    first, _ = autoopt(**common, OutputDir=tmp_path / "first")
    calls_after_first = calls["count"]
    checkpoint = tmp_path / "checkpoints" / "design.pkl"
    assert checkpoint.is_file()
    assert not checkpoint.with_suffix(".pkl.tmp").exists()

    with pytest.warns(UserWarning, match="trusted run"):
        second, _ = autoopt(**common, Resume=True, OutputDir=tmp_path / "second")
    assert calls["count"] == calls_after_first
    assert second[0].ave_perform_all().tolist() == first[0].ave_perform_all().tolist()


def test_interrupted_static_solve_resumes_identically(monkeypatch, tmp_path):
    solve_module = importlib.import_module("autooptlib.utils.solve")
    interrupted_calls = {"count": 0}

    def interrupted_objective(decision, instance):
        interrupted_calls["count"] += 1
        return float(np.sum(decision**2))

    problem = make_problem(interrupted_objective, bounds=(-1.0, 1.0))
    checkpoint_dir = tmp_path / "static-checkpoints"
    common = dict(
        Mode="solve",
        Problem=problem,
        InstanceSolve=[3],
        AlgName="Continuous Random Search",
        ProbN=3,
        ProbFE=9,
        AlgRuns=1,
        Seed=41,
        CheckpointDir=checkpoint_dir,
        CheckpointEvery=1,
    )
    original_writer = solve_module._write_checkpoint
    crashed = {"value": False}

    def crash_after_checkpoint(path, payload):
        original_writer(path, payload)
        if (
            not crashed["value"]
            and payload.get("schema") == "autooptlib.checkpoint"
            and not payload.get("complete")
            and payload.get("evaluations", 0) > 3
        ):
            crashed["value"] = True
            raise RuntimeError("injected static interruption")

    monkeypatch.setattr(solve_module, "_write_checkpoint", crash_after_checkpoint)
    with pytest.raises(RuntimeError, match="injected static interruption"):
        autoopt(**common, OutputDir=tmp_path / "static-interrupted")
    assert crashed["value"]
    assert (checkpoint_dir / "instance_1_run_1.pkl").is_file()

    monkeypatch.setattr(solve_module, "_write_checkpoint", original_writer)
    with pytest.warns(UserWarning, match="trusted run"):
        resumed_best, resumed_history = autoopt(
            **common,
            Resume=True,
            OutputDir=tmp_path / "static-resumed",
        )

    baseline_calls = {"count": 0}

    def baseline_objective(decision, instance):
        baseline_calls["count"] += 1
        return float(np.sum(decision**2))

    baseline_problem = make_problem(baseline_objective, bounds=(-1.0, 1.0))
    baseline_best, baseline_history = autoopt(
        **{**common, "Problem": baseline_problem, "CheckpointDir": None},
        OutputDir=tmp_path / "static-baseline",
    )
    assert interrupted_calls["count"] == baseline_calls["count"] == 9
    np.testing.assert_array_equal(resumed_best[0][0].dec, baseline_best[0][0].dec)
    assert [solution.fit for solution in resumed_history[0]] == [
        solution.fit for solution in baseline_history[0]
    ]


def test_interrupted_design_resumes_identically(monkeypatch, tmp_path):
    process_module = importlib.import_module("autooptlib.utils.general.process")
    interrupted_calls = {"count": 0}

    def interrupted_objective(decision, instance):
        interrupted_calls["count"] += 1
        return float(np.sum(decision**2))

    problem = make_problem(interrupted_objective, bounds=(-1.0, 1.0))
    checkpoint_dir = tmp_path / "design-checkpoints"
    common = dict(
        Mode="design",
        Problem=problem,
        InstanceTrain=[3],
        InstanceTest=[3],
        ProbN=3,
        ProbFE=6,
        AlgN=1,
        AlgFE=3,
        AlgRuns=1,
        Seed=52,
        CheckpointDir=checkpoint_dir,
        CheckpointEvery=1,
    )
    original_writer = process_module._write_design_checkpoint
    crashed = {"value": False}

    def crash_after_checkpoint(path, payload):
        original_writer(path, payload)
        if (
            not crashed["value"]
            and not payload.get("complete")
            and payload.get("evaluated_count", 0) > 1
        ):
            crashed["value"] = True
            raise RuntimeError("injected design interruption")

    monkeypatch.setattr(
        process_module, "_write_design_checkpoint", crash_after_checkpoint
    )
    with pytest.raises(RuntimeError, match="injected design interruption"):
        autoopt(**common, OutputDir=tmp_path / "design-interrupted")
    assert crashed["value"]
    assert (checkpoint_dir / "design.pkl").is_file()

    monkeypatch.setattr(process_module, "_write_design_checkpoint", original_writer)
    with pytest.warns(UserWarning, match="trusted run"):
        resumed, resumed_trace = autoopt(
            **common,
            Resume=True,
            OutputDir=tmp_path / "design-resumed",
        )

    baseline_calls = {"count": 0}

    def baseline_objective(decision, instance):
        baseline_calls["count"] += 1
        return float(np.sum(decision**2))

    baseline_problem = make_problem(baseline_objective, bounds=(-1.0, 1.0))
    baseline, baseline_trace = autoopt(
        **{**common, "Problem": baseline_problem, "CheckpointDir": None},
        OutputDir=tmp_path / "design-baseline",
    )
    assert interrupted_calls["count"] == baseline_calls["count"]
    assert algorithm_to_dict(resumed[0]) == algorithm_to_dict(baseline[0])
    assert [algorithm_to_dict(algorithm) for algorithm in resumed_trace] == [
        algorithm_to_dict(algorithm) for algorithm in baseline_trace
    ]


def test_interrupted_sequential_solve_resumes_identically(monkeypatch, tmp_path):
    solve_module = importlib.import_module("autooptlib.utils.solve")

    def sequential_definition(counter):
        def definition(records, instances, mode):
            if mode != "construct":
                raise ValueError(mode)
            data_entries = []
            for record, dimension in zip(records, instances):
                dimension = int(dimension)
                record.type = ["continuous", "sequential", "certain"]
                record.bound = np.vstack(
                    (np.full(dimension, -1.0), np.full(dimension, 1.0))
                )

                def evaluate(data, decision):
                    counter["count"] += 1
                    return float(np.sum(np.asarray(decision) ** 2) + data.stage)

                def advance(best, data, problem=record):
                    next_stage = data.stage + 1
                    next_data = SimpleNamespace(stage=next_stage)
                    setattr(next_data, "continue", next_stage < 3)
                    return problem, next_data

                record.evaluate = evaluate
                record.advance_sequence = advance
                first_data = SimpleNamespace(stage=0)
                setattr(first_data, "continue", True)
                data_entries.append(first_data)
            return list(records), data_entries, None

        return definition

    interrupted_calls = {"count": 0}
    checkpoint_dir = tmp_path / "sequential-checkpoints"
    common = dict(
        Mode="solve",
        Problem=sequential_definition(interrupted_calls),
        InstanceSolve=[3],
        AlgName="Continuous Random Search",
        ProbN=3,
        ProbFE=18,
        AlgRuns=1,
        Metric="runtimeFE",
        Tmax=18,
        Thres=1e99,
        Seed=63,
        CheckpointDir=checkpoint_dir,
    )
    original_writer = solve_module._write_checkpoint
    crashed = {"value": False}

    def crash_after_checkpoint(path, payload):
        original_writer(path, payload)
        if (
            not crashed["value"]
            and payload.get("schema") == "autooptlib.sequence-checkpoint"
            and not payload.get("complete")
            and len(payload.get("solutions", [])) == 1
        ):
            crashed["value"] = True
            raise RuntimeError("injected sequential interruption")

    monkeypatch.setattr(solve_module, "_write_checkpoint", crash_after_checkpoint)
    with pytest.raises(RuntimeError, match="injected sequential interruption"):
        autoopt(**common, OutputDir=tmp_path / "sequential-interrupted")
    assert crashed["value"]

    monkeypatch.setattr(solve_module, "_write_checkpoint", original_writer)
    with pytest.warns(UserWarning, match="trusted run"):
        resumed_best, resumed_history = autoopt(
            **common,
            Resume=True,
            OutputDir=tmp_path / "sequential-resumed",
        )

    baseline_calls = {"count": 0}
    baseline_problem = sequential_definition(baseline_calls)
    baseline_best, baseline_history = autoopt(
        **{**common, "Problem": baseline_problem, "CheckpointDir": None},
        OutputDir=tmp_path / "sequential-baseline",
    )
    assert interrupted_calls["count"] == baseline_calls["count"] == 9
    np.testing.assert_array_equal(resumed_best[0][0].dec, baseline_best[0][0].dec)
    assert [solution.fit for solution in resumed_history[0]] == [
        solution.fit for solution in baseline_history[0]
    ]

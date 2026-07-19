"""System-level tests for design and solve workflows."""

from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from autooptlib import autoopt, save_algorithm
from autooptlib.utils.solve import input_algorithm


@pytest.fixture()
def chdir_tmp(tmp_path):
    """Temporarily change working directory to an isolated tmp path."""
    prev = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(prev)


@contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def test_autoopt_design_creates_outputs(chdir_tmp):
    final_algs, alg_trace = autoopt(
        Mode="design",
        Problem="cec2013_f1",
        InstanceTrain=[10],
        InstanceTest=[10],
        AlgN=2,
        AlgFE=60,
        AlgRuns=2,
        ProbN=15,
        ProbFE=800,
        Evaluate="racing",
        Compare="statistic",
        RacingK=1,
        archive=["archive_best"],
        Seed=2024,
    )

    # basic structural checks
    assert final_algs, "No algorithms returned from design workflow."
    assert final_algs[0].performance.shape[0] == 2  # train + test instances
    assert final_algs[0].performance.shape[1] == 2
    assert alg_trace, "Algorithm trace should not be empty."

    # output files generated
    expected = [
        "Algs.pkl",
        "Algs_best_algs_iter.csv",
        "Algs_perf_final_algs.csv",
        "ConvergenceCurve.csv",
        "experiment.json",
    ]
    for name in expected:
        assert Path(name).exists(), f"Missing expected design output {name}"


def test_autoopt_solve_generates_solution_logs(chdir_tmp):
    best_solutions, all_solutions = autoopt(
        Mode="solve",
        Problem="cec2013_f1",
        InstanceSolve=[10],
        AlgName="ICA",
        AlgRuns=1,
        ProbN=20,
        ProbFE=1000,
        Metric="quality",
        Seed=2025,
    )

    assert len(best_solutions) == 1
    assert len(best_solutions[0]) == 1
    assert len(all_solutions) == 1
    assert all(all_solutions[0]), "History for best run should not be empty."
    assert np.isfinite(best_solutions[0][0].fit)

    expected = [
        "Solutions.pkl",
        "Solutions.csv",
        "Fitness_all_runs.csv",
        "experiment.json",
    ]
    for name in expected:
        assert Path(name).exists(), f"Missing expected solve output {name}"


def test_seed_reproduces_end_to_end_solve(tmp_path):
    kwargs = dict(
        Mode="solve",
        Problem="cec2013_f1",
        InstanceSolve=[10],
        AlgName="Continuous Random Search",
        AlgRuns=2,
        ProbN=12,
        ProbFE=360,
        Metric="quality",
        Seed=8675309,
    )

    results = []
    for name in ("first", "second"):
        with working_directory(tmp_path / name):
            best, histories = autoopt(**kwargs)
        results.append(
            (
                np.array([solution.fit for solution in best[0]]),
                np.array([solution.dec for solution in best[0]]),
                np.array([solution.fit for solution in histories[0]]),
            )
        )

    for left, right in zip(results[0], results[1]):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize(
    ("evaluation_mode", "extra"),
    [
        ("intensification", {}),
        ("approximate", {"Surro": 1}),
    ],
)
def test_less_common_design_modes_complete_with_exact_final_test(
    tmp_path, evaluation_mode, extra
):
    output_dir = tmp_path / evaluation_mode
    algorithms, trace = autoopt(
        Mode="design",
        Problem="cec2013_f1",
        InstanceTrain=[2],
        InstanceTest=[2],
        AlgN=2,
        AlgFE=4,
        AlgRuns=1,
        ProbN=5,
        ProbFE=10,
        Evaluate=evaluation_mode,
        Compare="average",
        Seed=44,
        OutputDir=output_dir,
        **extra,
    )
    assert len(algorithms) == 2
    assert trace
    assert np.all(np.isfinite(algorithms[0].performance))
    assert (output_dir / "Algorithm_1.json").is_file()


def test_exported_json_algorithm_can_be_solved_without_pickle(tmp_path):
    preset, _ = input_algorithm(SimpleNamespace(AlgName="Continuous Random Search"))
    algorithm_file = save_algorithm(preset, tmp_path / "portable.json")
    best, history = autoopt(
        Mode="solve",
        Problem="cec2013_f1",
        InstanceSolve=[2],
        AlgFile=algorithm_file,
        AlgRuns=1,
        ProbN=5,
        ProbFE=15,
        Seed=7,
        OutputDir=tmp_path / "solve",
    )
    assert np.isfinite(best[0][0].fit)
    assert history[0]


@pytest.mark.parametrize("problem", ["cec2013_f6", "cec2013_f13", "cec2013_f21"])
def test_published_cec_algorithm_profiles_execute(tmp_path, problem):
    algorithm_file = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "reproducibility"
        / "algorithms"
        / f"{problem}_alg_star.json"
    )
    best, history = autoopt(
        Mode="solve",
        Problem=problem,
        InstanceSolve=[10],
        AlgFile=algorithm_file,
        AlgRuns=1,
        ProbN=10,
        ProbFE=100,
        Seed=19,
        OutputDir=tmp_path / problem,
    )
    assert np.isfinite(best[0][0].fit)
    assert history[0]


def test_sequential_problem_allocates_budget_per_stage(tmp_path):
    calls = {"count": 0}

    def sequential_problem(records, instances, mode):
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
                calls["count"] += 1
                return float(np.sum(np.asarray(decision) ** 2) + data.stage), 0.0, None

            def advance(best, data, problem=record):
                next_stage = data.stage + 1
                return problem, SimpleNamespace(
                    stage=next_stage, continue_=next_stage < 3
                )

            record.evaluate = evaluate
            record.advance_sequence = advance
            data_entries.append(SimpleNamespace(stage=0, continue_=True))
        # The runtime uses the attribute name "continue"; assign it without
        # using a reserved word in constructor syntax.
        for data in data_entries:
            setattr(data, "continue", data.continue_)
        original_advances = [record.advance_sequence for record in records]
        for record, original in zip(records, original_advances):

            def wrapped(best, data, original=original):
                problem, next_data = original(best, data)
                setattr(next_data, "continue", next_data.continue_)
                return problem, next_data

            record.advance_sequence = wrapped
        return list(records), data_entries, None

    common = dict(
        Mode="solve",
        Problem=sequential_problem,
        InstanceSolve=[3],
        AlgName="Continuous Random Search",
        AlgRuns=1,
        ProbN=5,
        ProbFE=30,
        Metric="runtimeFE",
        Tmax=30,
        Thres=1e99,
        Seed=2,
        CheckpointDir=tmp_path / "checkpoints",
    )
    best, histories = autoopt(**common, OutputDir=tmp_path / "sequential")
    assert len(histories[0]) == 3
    assert len(best[0]) == 1
    assert calls["count"] == 15

    calls_after_first = calls["count"]
    with pytest.warns(UserWarning, match="trusted run"):
        resumed_best, resumed_histories = autoopt(
            **common,
            Resume=True,
            OutputDir=tmp_path / "sequential-resumed",
        )
    assert calls["count"] == calls_after_first
    assert resumed_best[0][0].fit == best[0][0].fit
    assert [solution.fit for solution in resumed_histories[0]] == [
        solution.fit for solution in histories[0]
    ]

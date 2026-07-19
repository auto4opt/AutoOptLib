"""Smoke tests for autoopt entry point (lightweight settings)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from autooptlib import autoopt


@pytest.fixture()
def tmp_workdir(tmp_path):
    prev = Path.cwd()
    os.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        os.chdir(prev)


def test_autoopt_design_smoke(tmp_workdir):
    final_algs, alg_trace = autoopt(
        Mode="design",
        Problem="cec2013_f1",
        InstanceTrain=[10],
        InstanceTest=[10],
        AlgN=2,
        AlgFE=40,
        ProbN=10,
        ProbFE=500,
        Evaluate="exact",
        Compare="average",
        archive=["archive_best"],
        Seed=123,
    )
    assert final_algs
    assert alg_trace
    assert Path("Algs.pkl").exists()
    assert Path("Algs_perf_final_algs.csv").exists()


def test_autoopt_solve_smoke(tmp_workdir):
    best_solutions, all_solutions = autoopt(
        Mode="solve",
        Problem="cec2013_f1",
        InstanceSolve=[10],
        AlgName="Continuous Random Search",
        AlgRuns=1,
        ProbN=15,
        ProbFE=600,
        Metric="quality",
        Seed=321,
    )
    assert len(best_solutions) == 1
    assert Path("Solutions.pkl").exists()
    assert Path("Fitness_all_runs.csv").exists()

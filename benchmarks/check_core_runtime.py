"""Small, dependency-free performance regression gate for the solve engine."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import numpy as np

from autooptlib import autoopt, make_problem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-seconds", type=float, default=10.0)
    args = parser.parse_args()

    calls = 0

    def sphere(decision, dimension):
        nonlocal calls
        calls += 1
        return float(np.dot(decision, decision))

    problem = make_problem(sphere, bounds=(-5.0, 5.0), name="benchmark_sphere")
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as directory:
        autoopt(
            Mode="solve",
            Problem=problem,
            InstanceSolve=[20],
            AlgName="Continuous Random Search",
            AlgRuns=1,
            ProbN=64,
            ProbFE=2048,
            Seed=2026,
            OutputDir=Path(directory),
        )
    elapsed = time.perf_counter() - started
    report = {
        "schema": "autooptlib.performance-smoke",
        "objective_evaluations": calls,
        "elapsed_seconds": elapsed,
        "maximum_seconds": args.max_seconds,
    }
    print(json.dumps(report, sort_keys=True))
    if calls != 2048:
        raise SystemExit(f"expected 2048 objective evaluations, observed {calls}")
    if elapsed > args.max_seconds:
        raise SystemExit(
            f"core runtime {elapsed:.3f}s exceeded ceiling {args.max_seconds:.3f}s"
        )


if __name__ == "__main__":
    main()

"""Run lightweight design and solve workflows for every bundled problem."""

from __future__ import annotations

import argparse
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Sequence

from autooptlib import autoopt
from autooptlib.problems import __all__ as PROBLEM_NAMES


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _require_files(paths: Iterable[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise RuntimeError(f"expected output files missing: {', '.join(missing)}")


def run_design_smoke(problem: str, base_dir: Path) -> None:
    workdir = base_dir / problem / "design"
    with _working_directory(workdir):
        autoopt(
            Mode="design",
            Problem=problem,
            InstanceTrain=[10],
            InstanceTest=[10],
            AlgN=2,
            AlgFE=40,
            AlgRuns=1,
            ProbN=10,
            ProbFE=500,
            Evaluate="exact",
            Compare="average",
            archive=["archive_best"],
            Seed=42,
        )
        _require_files(
            [
                Path("Algs.pkl"),
                Path("Algs_perf_final_algs.csv"),
                Path("Algs_final_algs.csv"),
            ]
        )


def run_solve_smoke(problem: str, base_dir: Path) -> None:
    workdir = base_dir / problem / "solve"
    with _working_directory(workdir):
        autoopt(
            Mode="solve",
            Problem=problem,
            InstanceSolve=[10],
            AlgName="Continuous Random Search",
            AlgRuns=1,
            ProbN=15,
            ProbFE=600,
            Metric="quality",
            Seed=321,
        )
        _require_files(
            [
                Path("Solutions.pkl"),
                Path("Fitness_all_runs.csv"),
                Path("Solutions.csv"),
            ]
        )


def _select_problems(selected: Sequence[str] | None) -> list[str]:
    if not selected:
        return list(PROBLEM_NAMES)
    lower = {name.lower(): name for name in PROBLEM_NAMES}
    resolved = []
    for item in selected:
        key = item.lower()
        if key not in lower:
            available = ", ".join(PROBLEM_NAMES)
            raise ValueError(f"unknown problem {item!r}. Available: {available}")
        resolved.append(lower[key])
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Lightweight smoke test for all bundled AutoOpt problems."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("smoke_outputs"),
        help="directory where per-problem results will be stored (default: %(default)s)",
    )
    parser.add_argument(
        "--problems",
        nargs="*",
        help="optional subset of problems to run (default: all available)",
    )
    args = parser.parse_args()

    problems = _select_problems(args.problems)
    output_root = args.output_root.resolve()

    failures: list[tuple[str, Exception]] = []
    for problem in problems:
        try:
            run_design_smoke(problem, output_root)
            run_solve_smoke(problem, output_root)
            print(f"[ok] {problem}")
        except Exception as exc:
            failures.append((problem, exc))
            print(f"[fail] {problem}: {exc}")

    if failures:
        print("\nSome problems failed:")
        for problem, exc in failures:
            print(f"  - {problem}: {exc}")
        return 1

    print("\nAll requested problems completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

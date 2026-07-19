"""Time-bounded execution parity check against the software-paper CEC table.

This benchmark evaluates the three Alg* structures fully disclosed in the
paper.  It deliberately tests all three possible held-out dimensions because
the paper does not disclose which dimension was selected for each function.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from autooptlib import __version__, autoopt

PAPER_RESULTS = {
    6: {"mean": -868.49, "std": 8.87, "optimum": -900.0},
    13: {"mean": 5.90, "std": 98.8, "optimum": -200.0},
    21: {"mean": 1113.42, "std": 4.15, "optimum": 900.0},
}


def _atomic_json(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = [
        "function",
        "dimension",
        "runs",
        "prob_n",
        "prob_fe",
        "seed",
        "python_mean",
        "python_std_sample",
        "python_std_population",
        "paper_mean",
        "paper_std",
        "python_gap_to_optimum",
        "paper_gap_to_optimum",
        "gap_ratio_python_over_paper",
        "mean_absolute_difference",
        "elapsed_seconds",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(
    output: Path, runs: int, seed: int, max_seconds: float
) -> dict[str, Any]:
    repository = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    document: dict[str, Any] = {
        "schema": "autooptlib.paper-cec-parity",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "software": {
            "autooptlib": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "method": {
            "functions": [6, 13, 21],
            "dimensions": [10, 30, 50],
            "runs": runs,
            "prob_n": 50,
            "prob_fe": "1000 * dimension",
            "seed_base": seed,
            "max_seconds": max_seconds,
            "paper_test_dimension_disclosed": False,
            "algorithm_source": "software-paper pseudocode JSON profiles",
        },
        "paper_results": PAPER_RESULTS,
        "results": [],
        "complete": False,
    }
    json_path = output / "results.json"

    for function in (6, 13, 21):
        profile = (
            repository
            / "examples"
            / "reproducibility"
            / "algorithms"
            / f"cec2013_f{function}_alg_star.json"
        )
        for dimension in (10, 30, 50):
            if perf_counter() - started >= max_seconds:
                document["stopped_reason"] = "time limit reached between cases"
                document["elapsed_seconds"] = perf_counter() - started
                _atomic_json(json_path, document)
                _write_csv(output / "summary.csv", document["results"])
                return document

            case_seed = seed + function * 1000 + dimension
            case_output = output / f"f{function}_d{dimension}"
            case_started = perf_counter()
            best, _ = autoopt(
                Mode="solve",
                Problem=f"cec2013_f{function}",
                InstanceSolve=[dimension],
                AlgFile=profile,
                AlgRuns=runs,
                ProbN=50,
                ProbFE=1000 * dimension,
                Seed=case_seed,
                OutputDir=case_output,
            )
            fits = np.asarray([solution.fit for solution in best[0]], dtype=float)
            paper = PAPER_RESULTS[function]
            python_mean = float(np.mean(fits))
            paper_gap = float(paper["mean"] - paper["optimum"])
            python_gap = float(python_mean - paper["optimum"])
            row = {
                "function": function,
                "dimension": dimension,
                "runs": runs,
                "prob_n": 50,
                "prob_fe": 1000 * dimension,
                "seed": case_seed,
                "python_mean": python_mean,
                "python_std_sample": float(np.std(fits, ddof=1)) if runs > 1 else 0.0,
                "python_std_population": float(np.std(fits)),
                "paper_mean": float(paper["mean"]),
                "paper_std": float(paper["std"]),
                "python_gap_to_optimum": python_gap,
                "paper_gap_to_optimum": paper_gap,
                "gap_ratio_python_over_paper": (
                    python_gap / paper_gap if paper_gap > 0 else float("inf")
                ),
                "mean_absolute_difference": abs(python_mean - float(paper["mean"])),
                "elapsed_seconds": perf_counter() - case_started,
            }
            document["results"].append(row)
            document["elapsed_seconds"] = perf_counter() - started
            _atomic_json(json_path, document)
            _write_csv(output / "summary.csv", document["results"])
            print(
                f"f{function} D={dimension}: {python_mean:.8g} "
                f"± {row['python_std_sample']:.4g} "
                f"({row['elapsed_seconds']:.1f}s)",
                flush=True,
            )

    document["complete"] = True
    document["elapsed_seconds"] = perf_counter() - started
    _atomic_json(json_path, document)
    _write_csv(output / "summary.csv", document["results"])
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/paper_cec_parity_20260719"),
    )
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--max-seconds", type=float, default=3300.0)
    args = parser.parse_args()
    result = run_benchmark(args.output, args.runs, args.seed, args.max_seconds)
    print(
        f"complete={result['complete']} elapsed={result['elapsed_seconds']:.1f}s "
        f"cases={len(result['results'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()

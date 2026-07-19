"""Python translation of Utilities/General/Output.m."""

from __future__ import annotations

import csv
import json
import pickle
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import numpy as np

from ..._version import __version__
from ...serialization import save_algorithm
from ..design import Design


def _manifest_value(value: Any) -> Any:
    """Convert public options to stable, bounded JSON metadata."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.size <= 100:
            return value.tolist()
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if isinstance(value, dict):
        return {str(key): _manifest_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_manifest_value(item) for item in value]
    if callable(value):
        module = getattr(value, "__module__", "")
        name = getattr(
            value,
            "__qualname__",
            getattr(value, "__name__", type(value).__name__),
        )
        return f"{module}:{name}" if module else name
    return {"type": type(value).__name__, "repr": repr(value)[:500]}


def write_experiment_manifest(
    directory: str | Path | None, *, mode: str, options: dict[str, Any]
) -> Path:
    """Atomically write software and normalized invocation provenance."""
    target_directory = _ensure_path(directory or Path.cwd())
    target_directory.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "autooptlib.experiment",
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "software": {
            "autooptlib": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "platform": platform.platform(),
            "executable": sys.executable,
        },
        "options": {
            key: _manifest_value(value) for key, value in sorted(options.items())
        },
    }
    target = target_directory / "experiment.json"
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    return target


def _ensure_path(path: str | Path) -> Path:
    return Path(path).resolve()


def _remove_if_exists(paths: Iterable[Path]) -> None:
    for p in paths:
        if p.exists():
            p.unlink()


def _format_array(arr: np.ndarray | None) -> str:
    if arr is None:
        return ""
    arr = np.asarray(arr)
    if arr.size == 0:
        return ""
    return np.array2string(arr, precision=4, separator=",", suppress_small=True)


def _generate_pseudocode(design_obj: Design, alg_index: int, stage: str) -> List[str]:
    lines: List[str] = [f"Algorithm {alg_index} ({stage})"]
    pathways = getattr(design_obj, "operator_pheno", []) or []
    params = getattr(design_obj, "parameter_pheno", []) or []
    if not pathways:
        lines.append("  <empty>")
        return lines
    pathways = pathways[0]
    params = params[0] if params else []
    for idx, path in enumerate(pathways, 1):
        lines.append(f"  Pathway {idx}:")
        lines.append(f"    Choose: {path.choose or '<none>'}")
        if params and idx - 1 < len(params):
            choose_param = params[idx - 1].choose
            if choose_param is not None:
                lines.append(f"      Choose param: {_format_array(choose_param)}")
        for step_idx, step in enumerate(path.search, 1):
            desc = f"    Search {step_idx}: {step.primary}"
            if step.secondary:
                desc += f" -> {step.secondary}"
            term = (
                np.asarray(step.termination).reshape(-1)
                if step.termination is not None
                else []
            )
            if term.size:
                desc += f" | termination={term.tolist()}"
            lines.append(desc)
            if (
                params
                and idx - 1 < len(params)
                and step_idx - 1 < len(params[idx - 1].search)
            ):
                primary_param = params[idx - 1].search[step_idx - 1].primary
                secondary_param = params[idx - 1].search[step_idx - 1].secondary
                if primary_param is not None:
                    lines.append(f"      Primary param: {_format_array(primary_param)}")
                if secondary_param is not None:
                    lines.append(
                        f"      Secondary param: {_format_array(secondary_param)}"
                    )
        lines.append(f"    Update: {path.update or '<none>'}")
        if params and idx - 1 < len(params):
            update_param = params[idx - 1].update
            if update_param is not None:
                lines.append(f"      Update param: {_format_array(update_param)}")
        if path.archive:
            lines.append(f"    Archive: {', '.join(path.archive)}")
        lines.append("")
    return lines


def _write_matrix_csv(
    path: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)


def _output_design(
    algs: Sequence[Design],
    alg_trace: Sequence[Design],
    instance_train: Sequence[Any],
    instance_test: Sequence[Any],
    setting: Any,
    directory: Path,
) -> None:
    algs = list(algs)
    alg_trace = list(alg_trace)
    directory.mkdir(parents=True, exist_ok=True)

    files_to_remove = [
        directory / "Algs.pkl",
        directory / "Algs_final_algs.csv",
        directory / "Algs_perf_final_algs.csv",
        directory / "Algs_best_algs_iter.csv",
        directory / "Algs_perf_best_algs_iter.csv",
        directory / "ConvergenceCurve.csv",
    ]
    _remove_if_exists(files_to_remove)
    _remove_if_exists(directory.glob("Algorithm_*.json"))

    with (directory / "Algs.pkl").open("wb") as handle:
        pickle.dump({"algs": algs, "trace": alg_trace}, handle)

    for index, algorithm in enumerate(algs, start=1):
        save_algorithm(
            algorithm,
            directory / f"Algorithm_{index}.json",
            metadata={"algorithm_index": index},
        )

    codes = [
        _generate_pseudocode(alg, idx + 1, "final") for idx, alg in enumerate(algs)
    ]
    max_len = max(len(code) for code in codes) if codes else 0
    header = [f"Algorithm {idx + 1}" for idx in range(len(codes))]
    rows = []
    for line_idx in range(max_len):
        row = []
        for code in codes:
            row.append(code[line_idx] if line_idx < len(code) else "")
        rows.append(row)
    _write_matrix_csv(directory / "Algs_final_algs.csv", header, rows)

    alg_runs = int(getattr(setting, "AlgRuns", getattr(setting, "alg_runs", 1)))
    train_len = len(instance_train)
    test_len = len(instance_test)
    test_indices = list(range(train_len, train_len + test_len))

    perf_rows = []
    for run in range(alg_runs):
        for idx, instance in enumerate(instance_test):
            row = [instance]
            for alg in algs:
                matrix = np.asarray(alg.performance)
                if matrix.size == 0:
                    value = ""
                else:
                    value = matrix[test_indices[idx], run]
                row.append(value)
            perf_rows.append(row)
    perf_header = ["Instance Index"] + [f"Algorithm {i + 1}" for i in range(len(algs))]
    _write_matrix_csv(directory / "Algs_perf_final_algs.csv", perf_header, perf_rows)

    trace_codes = [
        _generate_pseudocode(alg, idx + 1, "iteration")
        for idx, alg in enumerate(alg_trace)
    ]
    if trace_codes:
        max_len = max(len(code) for code in trace_codes)
        header = [f"Iteration {idx + 1}" for idx in range(len(trace_codes))]
        rows = []
        for line_idx in range(max_len):
            row = []
            for code in trace_codes:
                row.append(code[line_idx] if line_idx < len(code) else "")
            rows.append(row)
        _write_matrix_csv(directory / "Algs_best_algs_iter.csv", header, rows)

    trace_rows = []
    trace_header = ["Instance Index"] + [
        f"Iteration {i + 1}" for i in range(len(alg_trace))
    ]
    for idx, instance in enumerate(instance_train):
        row = [instance]
        for alg in alg_trace:
            matrix = np.asarray(alg.performance)
            if matrix.size == 0:
                value = ""
            else:
                value = np.mean(matrix[idx, :])
            row.append(value)
        trace_rows.append(row)
    _write_matrix_csv(
        directory / "Algs_perf_best_algs_iter.csv", trace_header, trace_rows
    )

    convergence = []
    for idx, alg in enumerate(alg_trace):
        matrix = np.asarray(alg.performance)
        if matrix.size == 0:
            convergence.append((idx + 1, np.nan))
        else:
            train_indices = list(range(train_len))
            sub = matrix[train_indices, :]
            convergence.append((idx + 1, float(np.mean(sub))))
    _write_matrix_csv(
        directory / "ConvergenceCurve.csv", ["Iteration", "Performance"], convergence
    )


def _output_solve(
    best_solutions: Any,
    all_solutions: Any,
    instance: Sequence[Any],
    setting: Any,
    directory: Path,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    files_to_remove = [
        directory / "Solutions.pkl",
        directory / "Solutions.csv",
        directory / "Fitness.csv",
        directory / "ConstraintViolation.csv",
        directory / "Fitness_all_runs.csv",
    ]
    _remove_if_exists(files_to_remove)

    with (directory / "Solutions.pkl").open("wb") as handle:
        pickle.dump({"best": best_solutions, "all": all_solutions}, handle)

    solutions_rows = []
    fitness_rows = []
    constraint_rows = []

    num_iterations = len(all_solutions[0]) if all_solutions else 0
    solution_header = ["Instance Index", "Best Solution"] + [
        f"Iteration {i + 1}" for i in range(num_iterations)
    ]
    for idx, inst in enumerate(instance):
        best_row = [inst]
        fit_row = [inst]
        con_row = [inst]
        best_iteration = None
        best_value = float("inf")
        for iter_idx in range(num_iterations):
            sol = all_solutions[idx][iter_idx]
            best_row.append(json.dumps(np.asarray(sol.dec).tolist()))
            fit_row.append(sol.fit)
            con_row.append(sol.con)
            if sol.fit < best_value:
                best_value = sol.fit
                best_iteration = sol
        if best_iteration is not None:
            best_row.insert(1, json.dumps(np.asarray(best_iteration.dec).tolist()))
            fit_row.insert(1, best_iteration.fit)
            con_row.insert(1, best_iteration.con)
        else:
            best_row.insert(1, "")
            fit_row.insert(1, "")
            con_row.insert(1, "")
        solutions_rows.append(best_row)
        fitness_rows.append(fit_row)
        constraint_rows.append(con_row)

    _write_matrix_csv(directory / "Solutions.csv", solution_header, solutions_rows)
    _write_matrix_csv(directory / "Fitness.csv", solution_header, fitness_rows)
    _write_matrix_csv(
        directory / "ConstraintViolation.csv", solution_header, constraint_rows
    )

    alg_runs = int(getattr(setting, "AlgRuns", getattr(setting, "alg_runs", 1)))
    fitness_all_header = ["Instance Index", "Mean", "Std"] + [
        f"Run {i + 1}" for i in range(alg_runs)
    ]
    fitness_all_rows = []
    for idx, inst in enumerate(instance):
        row = [inst]
        fits = np.asarray(
            [best_solutions[idx][run].fit for run in range(alg_runs)], dtype=float
        )
        finite = fits[np.isfinite(fits)]
        if finite.size == fits.size:
            mean = float(np.mean(finite))
            std = float(np.std(finite))
        elif finite.size == 0:
            mean = float("inf")
            std = 0.0
        else:
            mean = float("inf")
            std = float("inf")
        row.extend([mean, std])
        row.extend(fits.tolist())
        fitness_all_rows.append(row)
    _write_matrix_csv(
        directory / "Fitness_all_runs.csv", fitness_all_header, fitness_all_rows
    )


def output(
    *args, setting: Any, app: Any | None = None, directory: str | Path | None = None
):
    """Main entry replicating MATLAB Output.m behaviour."""
    mode = str(getattr(setting, "Mode", getattr(setting, "mode", "design"))).lower()
    if directory is None:
        directory = Path.cwd()
    directory = _ensure_path(directory)

    if mode == "design":
        if len(args) < 4:
            raise ValueError(
                "Design mode output requires algs, alg_trace, instance_train, instance_test."
            )
        algs, alg_trace, instance_train, instance_test = args[:4]
        _output_design(
            algs, alg_trace, instance_train, instance_test, setting, directory
        )
        return

    if mode == "solve":
        if len(args) < 3:
            raise ValueError(
                "Solve mode output requires best_solutions, all_solutions, instance."
            )
        best_solutions, all_solutions, instance = args[:3]
        _output_solve(best_solutions, all_solutions, instance, setting, directory)
        return

    raise ValueError("Mode must be 'design' or 'solve' for output.")


__all__ = ["output", "write_experiment_manifest"]

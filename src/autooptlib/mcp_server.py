"""Local stdio tools for running AutoOptLib experiments."""

import argparse
import json
import math
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import mkdtemp
from typing import Annotated

from . import (
    AlgorithmGraph,
    DesignSpace,
    NodeSpace,
    Operator,
    Pathway,
    Stage,
    algorithm_from_dict,
    autoopt,
    load_algorithm,
    save_algorithm,
)
from .utils.general import normalize_options

_MANAGED_OPTIONS = {
    "Mode",
    "Problem",
    "InstanceTrain",
    "InstanceTest",
    "InstanceSolve",
    "OutputDir",
    "DesignSpace",
    "Algorithm",
    "AlgFile",
    "AlgName",
}
_LOCAL_OPTIONS = {
    "CheckpointDir",
    "EvalLog",
    "CandidateLogPath",
    "EvalConcurrencyProfilePath",
    "EvalTrainingScorer",
    "EvalResourceEstimator",
    "EvalWorkerInitializer",
    "EvalWorkerFinalizer",
    "EvalWorkerConfig",
}


def _run_directory(prefix: str) -> Path:
    runs = Path.home() / ".autooptlib" / "mcp" / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return Path(mkdtemp(prefix=prefix + "-", dir=runs))


def _operator(value):
    return Operator(**value) if isinstance(value, dict) else value


def _pathway(document):
    data = dict(document)
    data["stages"] = [
        Stage(
            **{
                **stage,
                "search": _operator(stage["search"]),
                "choose": _operator(stage.get("choose")),
            }
        )
        for stage in document["stages"]
    ]
    data["update"] = _operator(document.get("update", "update_greedy"))
    return Pathway(**data)


def _graph(document):
    if "schema" in document:
        return algorithm_from_dict(document)
    data = dict(document)
    data["pathways"] = [_pathway(path) for path in document["pathways"]]
    return AlgorithmGraph(**data)


def _design_space(document):
    if "schema" in document:
        return DesignSpace.from_dict(document)
    data = dict(document)
    data["nodes"] = [NodeSpace(**node) for node in data.get("nodes", ())]
    return DesignSpace(**data)


def _json_algorithm(path: str):
    file = Path(path).expanduser()
    if file.suffix.lower() != ".json":
        raise ValueError("Use an AutoOptLib algorithm JSON file.")
    return load_algorithm(file)


def _options(catalog, supplied, profile):
    profile_options = normalize_options(catalog.run_options(profile))
    supplied_options = normalize_options(supplied or {})
    for key in _MANAGED_OPTIONS & (profile_options.keys() | supplied_options.keys()):
        raise ValueError(f"Pass {key} through its MCP tool input.")
    for key in _LOCAL_OPTIONS & supplied_options.keys():
        raise ValueError(f"Configure {key} in a local run profile.")
    options = {**profile_options, **supplied_options}
    if options.get("SearchInitialDesigns"):
        options["SearchInitialDesigns"] = [
            AlgorithmGraph.from_design(_graph(item))
            if isinstance(item, dict) and "schema" in item
            else _graph(item)
            if isinstance(item, dict) and "pathways" in item
            else item
            for item in options["SearchInitialDesigns"]
        ]
    return options


def _finite(value):
    return float(value) if math.isfinite(value) else None


def _write_problem(directory, problem_id, custom_problem, **instances):
    (directory / "mcp_problem.json").write_text(
        json.dumps(
            {
                "problem_id": problem_id,
                **instances,
                "custom_problem": custom_problem.model_dump(mode="json")
                if custom_problem
                else None,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Launch the MCP server with the optional dependencies installed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        metavar="MODULE",
        help="Import a trusted Python module with local problem and run profiles.",
    )
    args = parser.parse_args()
    # Reserve the original stream for protocol messages. Native objectives and
    # external simulators may write to file descriptor 1 independently of Python.
    stdout_fd = sys.stdout.fileno()
    protocol_fd = os.dup(stdout_fd)
    sys.stdout.flush()
    os.dup2(sys.stderr.fileno(), stdout_fd)
    try:
        with os.fdopen(
            protocol_fd, "w", encoding="utf-8", buffering=1, closefd=False
        ) as protocol:
            _serve(args.extension, protocol)
    finally:
        os.dup2(protocol_fd, stdout_fd)
        os.close(protocol_fd)


def _serve(extensions, protocol) -> None:
    try:
        from mcp.server.fastmcp import FastMCP
        from mcp.types import ToolAnnotations
    except ModuleNotFoundError as exc:
        if exc.name != "mcp":
            raise
        raise SystemExit(
            "Install the MCP extra with python -m pip install 'autooptlib[mcp]'."
        ) from None

    from pydantic import Field, JsonValue

    from .mcp_artifacts import write_graph, write_performance_chart
    from .mcp_components import ComponentPopulation, ComponentProblem, ComponentSession
    from .mcp_problems import CustomProblem, ProblemCatalog

    catalog = ProblemCatalog(extensions)
    components = ComponentSession()
    server = FastMCP(
        "AutoOptLib",
        instructions=(
            "Design or run optimization algorithms for a target problem. Translate "
            "a user's problem into bounds, a decision vector, a numerical objective "
            "and optional constraints and instance data. Ask the user for missing "
            "information that affects the mathematics. For a simulator, select a "
            "configured local problem ID. Use distinct representative training and "
            "test instances. File paths are local to the server."
        ),
    )
    creates_files = ToolAnnotations(destructiveHint=False, openWorldHint=False)
    positive = Field(strict=True, gt=0)
    nonnegative = Field(strict=True, ge=0)

    @server.tool(annotations=creates_files)
    def design_algorithm(
        problem_id: str = "sphere",
        train_dimension: Annotated[int, positive] = 10,
        test_dimension: Annotated[int, positive] = 20,
        candidate_budget: Annotated[int, positive] = 20,
        evaluations_per_run: Annotated[int, positive] = 200,
        seed: Annotated[int, nonnegative] = 2026,
        train_instances: list[JsonValue] | None = None,
        test_instances: list[JsonValue] | None = None,
        custom_problem: CustomProblem | None = None,
        design_space: dict[str, JsonValue] | None = None,
        run_options: dict[str, JsonValue] | None = None,
        run_profile: str | None = None,
    ) -> dict[str, JsonValue]:
        """Run AutoOptLib design and return algorithm JSON paths and held-out scores.

        Use sphere, bbob_f1..bbob_f24, pbo_f1..pbo_f23,
        custom, or a configured local problem ID. IOH instances accept a dimension
        or an object with dimension, instance, repeat and optional budget.
        For custom problems supply objective, bounds, problem_type, optional sense
        and constraints. Expressions see x and the JSON instance as data.
        Design space accepts the versioned AutoOptLib design-space JSON or its
        constructor fields. run_options uses AutoOptLib keyword names, for example
        AlgN, AlgRuns, ProbN, Metric, Designer, DesignFEs, EvalWorkers or
        SearchInitialDesigns. Values there override the convenience budget defaults.
        """
        if (train_instances is None) != (test_instances is None):
            raise ValueError("Supply both train_instances and test_instances.")
        train = [train_dimension] if train_instances is None else train_instances
        test = [test_dimension] if test_instances is None else test_instances
        if not train or not test:
            raise ValueError("Training and held-out test sets must both be nonempty.")
        with redirect_stdout(sys.stderr):
            problem, parse = catalog.resolve(problem_id, custom_problem)
            training = [parse(item) for item in train]
            testing = [parse(item) for item in test]
            if problem_id == "sphere" and train == test:
                raise ValueError("Training and test dimensions must differ for Sphere.")
            options = {
                "AlgN": 2,
                "AlgFE": candidate_budget,
                "AlgRuns": 2,
                "ProbN": 10,
                "ProbFE": evaluations_per_run,
                "Seed": seed,
                **_options(catalog, run_options, run_profile),
            }
            if design_space is not None:
                options["DesignSpace"] = _design_space(design_space)
            directory = _run_directory("design")
            _write_problem(
                directory,
                problem_id,
                custom_problem,
                train_instances=train,
                test_instances=test,
            )
            algorithms, _ = autoopt(
                Mode="design",
                Problem=problem,
                InstanceTrain=training,
                InstanceTest=testing,
                OutputDir=directory,
                **options,
            )
        return {
            "problem_id": problem_id,
            "train_dimension": train[0]
            if len(train) == 1 and type(train[0]) is int
            else None,
            "test_dimension": test[0]
            if len(test) == 1 and type(test[0]) is int
            else None,
            "seed": options["Seed"],
            "metric": options.get("Metric", "quality"),
            "output_directory": str(directory),
            "algorithms": [
                {
                    "algorithm_path": str(directory / f"Algorithm_{index}.json"),
                    "test_objectives": [
                        _finite(value)
                        for row in algorithm.performance[len(train) :]
                        for value in row
                    ],
                    "held_out": [
                        {
                            "instance": instance,
                            "scores": [_finite(value) for value in row],
                        }
                        for instance, row in zip(
                            test, algorithm.performance[len(train) :]
                        )
                    ],
                }
                for index, algorithm in enumerate(algorithms, start=1)
            ],
        }

    @server.tool(annotations=creates_files)
    def solve(
        instances: list[JsonValue],
        problem_id: str = "sphere",
        algorithm_path: str | None = None,
        graph: dict[str, JsonValue] | None = None,
        algorithm_name: str | None = None,
        custom_problem: CustomProblem | None = None,
        evaluations_per_run: Annotated[int, positive] = 200,
        seed: Annotated[int, nonnegative] = 2026,
        run_options: dict[str, JsonValue] | None = None,
        run_profile: str | None = None,
    ) -> dict[str, JsonValue]:
        """Run a saved algorithm JSON, an inline graph, or an AutoOptLib named preset.

        Supply exactly one algorithm source. Problem IDs and instances follow
        design_algorithm. Results include decisions, objective values and violations.
        For maximization the returned objective uses the engine's minimization sign.
        """
        if not instances:
            raise ValueError("Supply at least one solve instance.")
        if (
            sum(item is not None for item in (algorithm_path, graph, algorithm_name))
            != 1
        ):
            raise ValueError(
                "Supply exactly one of algorithm_path, graph, or algorithm_name."
            )
        with redirect_stdout(sys.stderr):
            problem, parse = catalog.resolve(problem_id, custom_problem)
            targets = [parse(item) for item in instances]
            options = {
                "ProbFE": evaluations_per_run,
                "AlgRuns": 1,
                "Seed": seed,
                **_options(catalog, run_options, run_profile),
            }
            if graph is not None:
                options["Algorithm"] = _graph(graph)
            elif algorithm_path is not None:
                options["Algorithm"] = _json_algorithm(algorithm_path)
            else:
                options["AlgName"] = algorithm_name
            directory = _run_directory("solve")
            _write_problem(directory, problem_id, custom_problem, instances=instances)
            solutions, _ = autoopt(
                Mode="solve",
                Problem=problem,
                InstanceSolve=targets,
                OutputDir=directory,
                **options,
            )
        return {
            "problem_id": problem_id,
            "output_directory": str(directory),
            "solutions": [
                {
                    "instance": instance,
                    "runs": [
                        {
                            "decision": solution.dec.tolist(),
                            "objective": _finite(solution.obj),
                            "constraint_violation": _finite(solution.con),
                            "fitness": _finite(solution.fit),
                        }
                        for solution in row
                    ],
                }
                for instance, row in zip(instances, solutions)
            ],
        }

    @server.tool(annotations=creates_files)
    def build_algorithm(
        graph: dict[str, JsonValue],
    ) -> dict[str, str]:
        """Build and save an executable AutoOptLib graph from registered components.

        Provide pathways, each with stages and update. Each stage has search and
        optional choose. An operator is a component name or an object with name
        and parameters. Graph settings include population_size, offspring_size,
        crossover_rate, mutation_rate, boundary_handling and archive.
        Returns portable algorithm JSON and a Mermaid diagram.
        """
        with redirect_stdout(sys.stderr):
            algorithm = _graph(graph)
            directory = _run_directory("algorithm")
            path = directory / "Algorithm.json"
            save_algorithm(algorithm, path)
            diagram = write_graph(algorithm, directory / "Algorithm.mmd")
        return {"algorithm_path": str(path), "diagram_path": diagram}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    def apply_component(
        name: str,
        population: ComponentPopulation,
        problem: ComponentProblem,
        parameters: list[float] | None = None,
        state_id: str | None = None,
        seed: int = 2026,
        generation: int = 0,
        inner_generation: int = 1,
        count: int | None = None,
        archive: ComponentPopulation | list[list[float]] | None = None,
        active_search: str | None = None,
        context: ComponentPopulation | None = None,
        targets: list[int] | None = None,
        data: JsonValue = None,
        backend: str = "python",
    ) -> dict[str, JsonValue]:
        """Apply a component to an explicit population in its existing execution role.

        Selectors return row indices, searches return decisions and updates return
        survivors. Supply fitness for fitness-dependent components. An archive
        accepts prior rows, or prior statistics for archive_statistic. To continue
        a stateful operation, pass the preceding state_id in the same MCP process.
        CMA state is initialized by search_cma. The optional native backend
        requires AutoOptLib's compiled component library.
        """
        if backend not in {"python", "native"}:
            raise ValueError("backend must be python or native.")
        with redirect_stdout(sys.stderr):
            return components.apply(
                name,
                population,
                problem,
                parameters,
                backend,
                seed,
                state_id,
                generation,
                inner_generation,
                count,
                archive,
                active_search,
                context,
                targets,
                data,
            )

    @server.tool(annotations=creates_files)
    def generate_charts(
        run_directory: str | None = None,
        algorithm_path: str | None = None,
    ) -> dict[str, str]:
        """Generate a performance SVG from design/solve results, or a Mermaid graph.

        Supply a run_directory containing AutoOptLib CSV files, an algorithm_path
        to a portable algorithm JSON, or both.
        """
        if run_directory is None and algorithm_path is None:
            raise ValueError("Supply run_directory or algorithm_path.")
        directory = _run_directory("charts")
        result = {}
        with redirect_stdout(sys.stderr):
            if run_directory is not None:
                result["chart_path"] = write_performance_chart(
                    Path(run_directory).expanduser(), directory / "Performance.svg"
                )
            if algorithm_path is not None:
                result["diagram_path"] = write_graph(
                    _json_algorithm(algorithm_path), directory / "Algorithm.mmd"
                )
        return result

    with redirect_stdout(protocol):
        server.run(transport="stdio")


if __name__ == "__main__":
    main()

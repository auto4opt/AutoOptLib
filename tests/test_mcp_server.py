import asyncio
import math
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
from autooptlib import AlgorithmGraph, load_algorithm


def test_stdio_operations(tmp_path):
    pytest.importorskip("mcp")
    pytest.importorskip("ioh")
    pytest.importorskip("matplotlib")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    project = Path(__file__).resolve().parents[1]
    (tmp_path / "target_extension.py").write_text(
        "import os\n"
        "from autooptlib import make_problem\n"
        "def objective(x, data):\n"
        "    os.write(1, b'evaluating instance\\n')\n"
        "    return float(((x - data['target']) ** 2).sum())\n"
        "PROBLEMS = {'local_target': make_problem(objective, "
        "bounds=([-2, -2, -2], [2, 2, 2]))}\n"
        "RUN_OPTIONS = {'quick': {'ProbFE': 10}}\n",
        encoding="utf-8",
    )
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "autooptlib.mcp_server", "--extension", "target_extension"],
        env={
            "HOME": str(tmp_path),
            "USERPROFILE": str(tmp_path),
            "PYTHONPATH": os.pathsep.join(
                map(str, (project / "src", tmp_path))
            ),
        },
        cwd=tmp_path,
    )
    runs = tmp_path / ".autooptlib" / "mcp" / "runs"
    transport_errors = []
    budget = {"candidate_budget": 2, "evaluations_per_run": 40}
    custom = {
        "sense": "min",
        "objective": 'sum((x - data["target"]) ** 2)',
        "constraints": ["sum(x) - 4"],
        "bounds": [[-5, -5, -5], [5, 5, 5]],
    }
    space = {
        "search": ["search_mu_uniform"],
        "max_pathways": 1,
        "max_stages": 1,
        "population_sizes": [4],
        "offspring_sizes": [4],
    }

    async def record_transport_errors(message):
        if isinstance(message, Exception):
            transport_errors.append(message)

    async def exercise_tools():
        async with stdio_client(server) as (read, write):
            async with ClientSession(
                read,
                write,
                read_timeout_seconds=timedelta(seconds=30),
                message_handler=record_transport_errors,
            ) as client:
                await client.initialize()
                tools = await client.list_tools()
                assert {tool.name for tool in tools.tools} == {
                    "design_algorithm",
                    "solve",
                    "build_algorithm",
                    "apply_component",
                    "generate_charts",
                }

                async def call(tool, **arguments):
                    result = await client.call_tool(tool, arguments)
                    assert not result.isError, result.content
                    return result.structuredContent

                invalid = await client.call_tool(
                    "design_algorithm", {"candidate_budget": 0}
                )
                assert invalid.isError
                invalid = await client.call_tool(
                    "design_algorithm",
                    {
                        "problem_id": "custom",
                        "custom_problem": {**custom, "objective": 'open("test")'},
                    },
                )
                assert invalid.isError
                assert not runs.exists()

                for name in ("sphere", "bbob_f1", "pbo_f1"):
                    design = await call("design_algorithm", problem_id=name, **budget)
                    _verify_design(design, runs, 20)

                design = await call(
                    "design_algorithm",
                    problem_id="custom",
                    custom_problem=custom,
                    train_instances=[{"target": [0, 0, 0]}],
                    test_instances=[{"target": [1, 1, 1]}],
                    design_space=space,
                    **budget,
                )
                _verify_design(design, runs, {"target": [1, 1, 1]})
                for result in design["algorithms"]:
                    graph = AlgorithmGraph.from_design(
                        load_algorithm(result["algorithm_path"])
                    )
                    for path in graph.pathways:
                        assert path.stages[0].search.name == "search_mu_uniform"

                algorithm = await call(
                    "build_algorithm",
                    graph={
                        "population_size": 4,
                        "pathways": [
                            {
                                "stages": [
                                    {
                                        "search": {
                                            "name": "search_mu_uniform",
                                            "parameters": [0.1],
                                        }
                                    }
                                ]
                            }
                        ],
                    },
                )
                assert (
                    Path(algorithm["diagram_path"]).read_text().startswith("flowchart")
                )
                solved = await call(
                    "solve",
                    algorithm_path=algorithm["algorithm_path"],
                    problem_id="local_target",
                    run_profile="quick",
                    instances=[{"target": [0.5, 0.5, 0.5]}],
                    evaluations_per_run=40,
                )
                result = solved["solutions"][0]["runs"][0]
                assert math.isfinite(result["objective"])
                assert len(result["decision"]) == 3
                import json

                manifest = json.loads(
                    (Path(solved["output_directory"]) / "experiment.json").read_text()
                )
                assert manifest["options"]["ProbFE"] == 10

                population = {"decisions": [[1, 0], [0, 2]], "fitness": [1, 4]}
                problem = {"bounds": [[-5, -5], [5, 5]]}
                selected = await call(
                    "apply_component",
                    name="choose_traverse",
                    population=population,
                    problem=problem,
                    count=1,
                )
                continued = await call(
                    "apply_component",
                    name="choose_traverse",
                    population=population,
                    problem=problem,
                    count=1,
                    state_id=selected["state_id"],
                )
                assert (selected["indices"], continued["indices"]) == ([0], [1])

                for run in (design, solved):
                    chart = await call(
                        "generate_charts",
                        run_directory=run["output_directory"],
                        algorithm_path=algorithm["algorithm_path"],
                    )
                    assert "<svg " in Path(chart["chart_path"]).read_text()
                    assert (
                        "search_mu_uniform" in Path(chart["diagram_path"]).read_text()
                    )

    asyncio.run(exercise_tools())
    assert not transport_errors


def _verify_design(result, runs, test_instance):
    output_directory = Path(result["output_directory"])
    assert output_directory.parent == runs.resolve()
    assert (output_directory / "experiment.json").is_file()
    assert (output_directory / "mcp_problem.json").is_file()
    assert len(result["algorithms"]) == 2
    for algorithm in result["algorithms"]:
        path = Path(algorithm["algorithm_path"])
        assert path.parent == output_directory
        load_algorithm(path)
        assert algorithm["held_out"][0]["instance"] == test_instance
        assert len(algorithm["test_objectives"]) == 2
        assert all(math.isfinite(value) for value in algorithm["test_objectives"])

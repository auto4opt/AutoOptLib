# Local MCP usage

Use the [README setup](../README.md#local-mcp) to install and configure the
stdio server. The client launches AutoOptLib on the user's machine. With Codex,
set a suitable tool timeout for long experiments in the server's
`mcp_servers.autooptlib` entry in `config.toml` using `tool_timeout_sec`.

| Tool | What it runs |
| --- | --- |
| `design_algorithm` | Search V6, followed by held-out evaluation. Saves algorithm JSON, metrics, convergence CSV and provenance. |
| `solve` | Executes a saved algorithm JSON, an inline algorithm graph, or a named AutoOptLib preset. Returns the best decision and fitness for each instance and repetition. |
| `build_algorithm` | Builds a runnable graph using registered components and saves algorithm JSON and a Mermaid diagram. |
| `apply_component` | Applies a registered selector, search, update, archive, or parameter component to a population. |
| `generate_charts` | Uses design or solve CSV files to create a performance SVG and uses an algorithm JSON to create a Mermaid graph. |

## Use a built-in benchmark

Use `sphere` with a dimension instance. Use `bbob_f1` through `bbob_f24` and
`pbo_f1` through `pbo_f23` with integers for dimensions or with IOH instance
objects. For example, ask the client to design on BBOB function 1 at dimension
10, instance 1, and evaluate at the same dimension on instance 2.

```json
{
  "problem_id": "bbob_f1",
  "train_instances": [{"dimension": 10, "instance": 1}],
  "test_instances": [{"dimension": 10, "instance": 2}],
  "candidate_budget": 20,
  "evaluations_per_run": 200,
  "seed": 2026
}
```

IOH instance objects also accept `repeat` and `budget`. PBO scores use the
engine's minimization convention, which negates the original objective.

## Design for a problem described in a conversation or document

Ask the model to use AutoOptLib for the target problem after it has read the
description and any necessary data. To run an experiment, it needs the decision
variables, their type and bounds, a numerical objective, any constraints, and
representative training and testing instances. The application that hosts the
model handles text, audio transcription and access to documents.

For a static mathematical objective, the client sends `problem_id: "custom"`
and an inline problem specification. This example searches for an optimizer
that minimizes squared error subject to a total allocation limit:

```json
{
  "problem_id": "custom",
  "custom_problem": {
    "name": "allocation",
    "problem_type": "continuous",
    "bounds": [[0, 0, 0], [6, 6, 6]],
    "objective": "sum((x - data['target']) ** 2)",
    "constraints": ["sum(x) - data['capacity']"]
  },
  "train_instances": [{"target": [1, 2, 3], "capacity": 5}],
  "test_instances": [{"target": [2, 3, 4], "capacity": 7}],
  "candidate_budget": 20,
  "evaluations_per_run": 200
}
```

Expressions use `x` for the decision vector and `data` for the supplied
instance. They support arithmetic, indexing, slices, comparisons and calls
to `abs`, `sum`, `mean`, `min`, `max`, `dot`, `matmul`, `sqrt`, `exp`,
`log`, `sin`, `cos`, `minimum`, `maximum` and `where`. Objectives must
return a finite scalar. Positive constraint values represent violations.
Set `sense` to `maximize` to have AutoOptLib minimize the negated objective.
The same problem specification can be sent to `solve`.

### Local data loaders and simulators

Use an importable Python module for a simulator, custom evaluator, or dynamic
problem. Export named `ProblemDefinition` callables in `PROBLEMS`. A module
can construct those callables with `make_problem`, the RIS and stacking
factories, or the advanced [problem contract](custom-problems.md). Its
`RUN_OPTIONS` mapping can define named profiles for Python callbacks and
paths such as `CheckpointDir`. The client selects a profile with `run_profile`.

```python
from autooptlib import make_problem
from my_simulator import evaluate, read_case

PROBLEMS = {
    "plant": make_problem(
        evaluate,
        bounds=([0.0, 0.0], [1.0, 10.0]),
        data_factory=read_case,
        name="plant",
    ),
}
RUN_OPTIONS = {"plant_validation": {"AlgRuns": 3, "ProbFE": 500}}
```

Add `--extension my_package.mcp_extension` to the local server's startup
arguments and make the module importable in that environment. Each extension
is explicitly selected in the startup configuration and runs with the local
account's permissions. A module can also register custom operators through
the existing Python API.

## Use a design space, algorithm graph, or individual component

`design_space` in `design_algorithm` accepts the versioned JSON produced by
`DesignSpace.to_dict()` or the constructor fields in the
[graph programming model](graph-programming-model.md). For example:

```json
{
  "search": ["search_mu_uniform"],
  "max_pathways": 1,
  "max_stages": 1,
  "population_sizes": [4, 8],
  "offspring_sizes": [4, 8]
}
```

Use `run_options` to pass JSON-compatible [AutoOptLib options](api.md) such as
`AlgN`, `AlgRuns`, `Metric`, `EvalWorkers`, `DesignFEs`, and
`SearchInitialDesigns`. Options here take precedence over convenience
arguments and defaults. Pass an algorithm JSON or an inline graph to
`solve`. Use the same graph input with `build_algorithm` to save it without
evaluating an objective.

For a standalone component, `apply_component` accepts its name, a matrix
of decisions, optional fitness, and problem bounds. Use `state_id` to continue
an operation within the same server process. For example, use the state from
`search_cma` when applying the CMA parameter update to the evaluated
offspring. See the [component catalogue](component-catalogue.md) for names
and roles. Use `generate_charts` with a returned run directory or algorithm
path to create artifacts under `~/.autooptlib/mcp/runs/`.

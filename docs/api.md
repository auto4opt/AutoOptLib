# Public API

AutoOptLib's supported entry points are imported from `autooptlib`.

## `autoopt(**options)`

Run automated design (`Mode="design"`) or execute an algorithm
(`Mode="solve"`). Keyword names are case-insensitive and underscore-insensitive;
unknown names raise `TypeError`.

Common options:

- `Problem`: a `ProblemDefinition` callable or a bundled problem name.
- `InstanceTrain`, `InstanceTest`, `InstanceSolve`: instance identifiers.
- `ProbN`: solution population size.
- `ProbFE`: hard objective-evaluation limit for each algorithm run, including
  initialization.
- `AlgN`: number of final algorithms retained in design mode.
- `AlgFE`: candidate-algorithm proposal limit during design, excluding the
  initial `AlgN` incumbents and held-out final evaluation.
- `AlgRuns`: repeated runs per instance.
- `Evaluate`: `exact`, `racing`, `intensification`, or `approximate`.
- `Metric`: `quality`, `runtimeFE`, `runtimeSec`, or `auc`.
- `Seed`: integer random seed.
- `OutputDir`: artifact directory.
- `EvalRetries`: retries after each failed objective attempt (default `0`).
- `EvalTimeoutSec`: optional hard timeout per attempt.
- `EvalFailure`: `raise` or `penalize` after attempts are exhausted.
- `EvalPenalty`: finite objective used by `penalize`.
- `EvalCache`: cache identical decisions for deterministic objectives.
- `EvalLog`: optional JSON Lines evaluation-event path.
- `EvalWorkers`: ordered candidate-evaluation workers (default `1`).
- `CheckpointDir`: optional Solve or Design checkpoint directory.
- `CheckpointEvery`: generations or candidate evaluations between atomic writes.
- `Resume`: resume matching completed or interrupted checkpoints.

The mode-specific defaults match the reference MATLAB package: Design uses
`AlgQ=4`, `ProbN=20`, `ProbFE=5000`, `InnerFE=500`, `AlgN=10`, `AlgFE=5000`,
and `AlgRuns=5`; Solve uses `ProbN=50`, `ProbFE=50000`, and `AlgRuns=5`.
Install `autooptlib[surrogate]` to use the MATLAB-equivalent 1000-tree random
forest in approximate mode; the lightweight core installation uses a
deterministic k-nearest-neighbor fallback.

Design returns `(algorithms, trace)`. Solve returns
`(best_solutions, histories)`. Design also evaluates the final `AlgN`
algorithms on held-out instances after the `AlgFE` search budget.
Every completed call also writes `experiment.json` to `OutputDir`.

## `make_problem(...)`

Wrap an ordinary scalar minimization objective. See
[custom problems](custom-problems.md).

## `save_algorithm(design, path, metadata=None)`

Write a decoded designed algorithm using the versioned
`autooptlib.algorithm` JSON schema.

## `load_algorithm(path)`

Validate and load an algorithm JSON document. Unknown schema versions and
unregistered component names are rejected.

## `register_component(...)`

Register a custom component for the current Python process. The component is
then included in compatible design spaces. Component names stored in JSON must
be registered before loading that JSON.

## Errors and constraints

AutoOptLib currently supports scalar minimization. A failed evaluation,
non-scalar objective, or non-finite objective/constraint raises
`ObjectiveEvaluationError` with decision context. Constraint callables return
one or more values; positive entries are summed as violation and non-positive
entries are feasible.

# Reliable objective evaluation

External simulators commonly fail transiently, hang, or return invalid values.
AutoOptLib provides opt-in controls:

- `EvalRetries`: retry count after a failed attempt;
- `EvalTimeoutSec`: hard per-attempt timeout using an isolated worker process;
- `EvalFailure`: `"raise"` or `"penalize"` after all attempts fail;
- `EvalPenalty`: finite minimization penalty used by `"penalize"`;
- `EvalCache`: reuse evaluations of identical decisions within one problem
  object;
- `EvalLog`: JSON Lines file recording attempts, failures, time, and cache hits;
- `EvalWorkers`: ordered thread workers for evaluating one candidate batch
  (default `1`).

```python
best, history = autoopt(
    Mode="solve",
    Problem=my_simulator_problem,
    InstanceSolve=["plant-17"],
    AlgName="Continuous Random Search",
    ProbN=20,
    ProbFE=2000,
    EvalRetries=2,
    EvalTimeoutSec=30,
    EvalFailure="penalize",
    EvalPenalty=1e30,
    EvalWorkers=4,
    EvalLog="results/evaluations.jsonl",
    OutputDir="results/run-17",
)
```

Timeout isolation adds process-start overhead and requires the problem and data
to be transferable to a worker process. On spawn-based platforms, prefer
module-level problem callables over lambdas and closures. For command-line
simulators, the most robust architecture is still for the objective itself to
launch a subprocess with its own resource limits and capture its stdout,
stderr, and input checksum.

`EvalWorkers>1` is intended for thread-safe objectives, especially external
simulators and NumPy/SciPy workloads that release the GIL. Repaired decisions
are submitted concurrently but returned in their original order, so the
optimizer's population ordering remains stable. `EvalWorkers>1` and
`EvalTimeoutSec` are deliberately mutually exclusive because forking timeout
workers from evaluation threads is unsafe on several platforms.

Caching is disabled by default. Enable it only for deterministic objectives;
otherwise it changes the statistical meaning of repeated evaluations.

## Checkpoint and resume

Solve and Design workflows can write atomic checkpoints:

```python
autoopt(..., CheckpointDir="results/checkpoints", CheckpointEvery=5)
autoopt(..., CheckpointDir="results/checkpoints", CheckpointEvery=5, Resume=True)
```

Resume requires matching run-defining budgets and instances. Static Solve
checkpoints preserve solution, component, archive, history, and random-generator
state. Sequential Solve uses a manifest plus one checkpoint per stage; completed
stages are not evaluated again. Design checkpoints preserve candidate
algorithms, trace, surrogate, inner-search state, held-out split, and random
state. They use Python pickle because they contain internal runtime objects:
load only checkpoints produced by a trusted AutoOptLib run. Portable designed
algorithms continue to use the validated JSON format.

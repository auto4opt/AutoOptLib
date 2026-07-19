# Custom problems

## Static problems

`make_problem` is the recommended interface for static continuous, discrete,
and permutation problems.

```python
import numpy as np
from autooptlib import make_problem

def load_instance(instance_id):
    return np.load(f"instances/{instance_id}.npy")

def objective(decision, target):
    return float(np.sum((decision - target) ** 2))

problem = make_problem(
    objective,
    bounds=lambda instance_id: np.vstack((
        np.full(20, -10.0),
        np.full(20, 10.0),
    )),
    data_factory=load_instance,
    name="target_matching",
)
```

`objective(decision, data)` must return one finite scalar to minimize. An
optional `constraint(decision, data)` returns values whose positive parts are
constraint violations.

Bounds can be a `(2, dimension)` array, a `(lower, upper)` pair expanded using
an integer instance identifier, or a callable of the instance identifier.
Permutation problems may omit bounds when each instance identifier is the
permutation length.

## Evaluation budgets

Every call to the objective counts toward `ProbFE`, including the initial
population. When the remaining budget cannot fit a full offspring population,
AutoOptLib evaluates only the remaining decisions and uses a deterministic
elitist merge for that final partial batch.

## Advanced definitions

Dynamic or sequential problems may implement `ProblemDefinition` directly. In
`construct` mode the callable receives problem records and instance identifiers
and returns `(problems, data, auxiliary)`. Every constructed problem must define:

- `type`: `[search_space, behavior, uncertainty]`, where the search space is
  `continuous`, `discrete`, or `permutation`;
- `bound`: a finite `(2, dimension)` array;
- `N` and `Gmax`: positive integers supplied by AutoOptLib;
- `evaluate(data, decision)`: scalar objective, optionally with constraint and
  auxiliary outputs.

A sequential problem sets behavior to `sequential`, provides data with a
boolean attribute named `continue`, and defines
`advance_sequence(best_solution, data) -> (next_problem, next_data)`. Each stage
receives the configured `ProbFE` limit, matching the reference MATLAB protocol.

For uncertain problems, set the third type entry to `uncertain`, provide a
positive `sampleN`, and include either `uncertain_average` or
`uncertain_worst` in `setting`. Auxiliary evaluator output is retained on each
`Solution` as `acc`. A problem-specific `repair(data, decision)` hook can be
provided in addition to the generic bound/type repair.

For external simulators, configure `EvalRetries`, `EvalTimeoutSec`,
`EvalFailure`, `EvalPenalty`, and `EvalLog` as described in
[reliability](reliability.md). AutoOptLib defaults to raising failures rather
than silently replacing them with a score. A simulator wrapper should still
capture its own inputs, stdout/stderr, executable version, and resource limits.

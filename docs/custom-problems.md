# Custom problems

Use `make_problem` for a scalar minimization objective:

```python
import numpy as np
from autooptlib import make_problem

problem = make_problem(
    lambda x, data: float(np.sum(x * x)),
    bounds=(-5.0, 5.0),
)
```

For a bounds pair, an integer instance identifier supplies the dimension.
Bounds can also be a `(2, dimension)` array or a callable of the instance
identifier. `data_factory(instance_id)` can load problem-specific input data.
The objective receives the decision vector and that instance data.

An optional `constraint(decision, data)` returns constraint values. Positive
parts are violations, so feasible values are nonpositive. Return finite values.
Every admitted objective evaluation counts toward `ProbFE`, including the
initial population. Remaining partial batches respect this limit.

AutoOptLib sums the positive parts and uses feasibility-first ordering across
its execution backends. A feasible solution ranks ahead of an infeasible
solution at any objective scale. The engine compares feasible solutions by
objective value and infeasible solutions by total violation. The returned
`Solution` keeps both values in `obj` and `con`. Its scalar `fit` reports the
objective for a feasible solution and total violation for an infeasible
solution. For a custom operator comparing candidates, use the pair
`(solution.con > 0, solution.fit)` as the key, with smaller keys preferred.

The quality metric reports the best feasible objective and uses infinity when
the run has yet to find a feasible solution. An objective target is reached
only by a feasible solution. Start a fresh run to use the new ordering with
an experiment that was checkpointed under the fixed-penalty policy.

`ProblemDefinition` supports richer construction, evaluation and repair hooks.
A custom repair hook is part of the problem definition, separate from generic
bound/type repair. For independent numerical batch objectives, see
[CPU execution](cpu-execution.md).

## Failure handling

The default is to raise evaluation failures. `EvalRetries`, `EvalTimeoutSec`,
`EvalFailure="penalize"` and `EvalPenalty` configure explicit alternatives.
Use `EvalLog` when individual evaluation records are required. External
simulators should manage their own process cleanup and resource limits.

## Optional IOH adapter

Install `.[benchmarks]` to connect to IOHexperimenter through
`make_bbob_problem`, `make_pbo_problem` or `make_ioh_problem`.
Only the adapter source is included; no benchmark campaigns, comparison
configurations, benchmark data archives or experimental results are bundled.

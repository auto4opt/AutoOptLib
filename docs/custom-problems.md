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

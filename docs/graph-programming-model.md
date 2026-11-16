# Graph programming model

AutoOptLib expresses a concrete optimizer with `Operator`, `Stage`, `Pathway`,
and `AlgorithmGraph`, and describes an admissible family with `DesignSpace`.
These declarations use the existing Search V6 grammar, JSON format, and solve
engines. They do not introduce a second optimizer, numerical implementation,
or execution-state store.

The public lifecycle is:

```text
register/discover components
    → declare an AlgorithmGraph or a DesignSpace
    → solve directly or run Search
    → save the selected Design as algorithm JSON
    → load, inspect with AlgorithmGraph.from_design, and solve again
```

## 1. Concrete algorithms

```python
from autooptlib import AlgorithmGraph, Operator, Pathway, Stage

algorithm = AlgorithmGraph(
    pathways=(Pathway(
        stages=(
            Stage("cross_point_one", choose="choose_tournament"),
            Stage(Operator("search_mu_uniform", parameters=(0.1,))),
        ),
        update="update_greedy",
    ),),
    population_size=20,
    offspring_size=20,
    mutation_rate=0.5,
    archive=("archive_best",),
)
```

| Declaration | Meaning |
| --- | --- |
| `Operator(name, parameters=None)` | One registered component and its numeric parameter vector; None preserves that component's existing default behavior |
| `Stage(search, choose=None)` | Optional selection, then one search/crossover operation; strings mean parameter-free/default component calls |
| `Pathway(stages, update=...)` | An ordered sequence ending in a replacement operation |
| `AlgorithmGraph(pathways, ...)` | A concrete optimizer, including population/offspring sizes, rates, repair policy and shared archives |
| `DesignSpace(...)` | Allowed components, narrower parameter domains, size choices and topology limits for automatic design |
| `NodeSpace(pathway, stage, role, choices, parameters=...)` | Required occurrence, its component choices and independent parameter intervals; updates use `stage=-1` |
| `ComponentContract`, `ParameterSpec` | Machine-readable execution domains, data flow, arity, state, effects and backend support |
| `ExecutionSpec` | Versioned execution and state-ownership rules persisted with executable algorithms |

Declarations take independent immutable copies of sequence inputs. Every
`to_design()` creates fresh mutable phenotype and parameter arrays; RNG,
velocity, covariance, populations, archives and counters are created by each
run, not stored in the declaration.

The current grammar is explicitly bounded:

- One or two pathways, each with one or two stages.
- A second stage is allowed only after crossover and must be a mutation.
- Each stage may have its own selector; None means no explicit selection.
- One shared archive policy, with a separate terminal update per pathway.
- Arbitrary loops, general DAG branching, recursive graphs and secondary search
  operators are outside this public grammar.

Construction reuses the existing phenotype and configuration validators.
SSGA requires offspring size no greater than population size; PSO requires
equal sizes. Inactive rate fields are canonicalized by the existing rules.
Dimension-dependent and problem-dependent component restrictions are checked
before the first objective call. `algorithm.validate(problem_record,
backend="python")` also performs this check without evaluation or random draws.
`GraphValidationError.location` identifies the pathway, stage and role; messages
include the offending component or parameter. Backend support, binary bounds,
minimum dimensions and dimension-dependent parameter limits are checked together.

`Stage.termination` retains the historical search-step metadata for lossless
conversion of existing Search results. It defaults to `(1.0, 1.0)` for hand
declarations. It does not replace `ProbFE`, rates, or the batch stage boundaries.

## 2. Run without an intermediate algorithm file

```python
from autooptlib import autoopt, make_problem

problem = make_problem(
    lambda decision, data: float((decision ** 2).sum()),
    bounds=(-5.0, 5.0),
)
best, history = autoopt(
    Mode="solve",
    Problem=problem,
    InstanceSolve=[8],
    Algorithm=algorithm,
    ProbFE=100,
    AlgRuns=1,
    Seed=2026,
    OutputDir="results/model-solve",
)
```

`Algorithm` accepts an `AlgorithmGraph` or a decoded `Design`, including a
Search result. It is exclusive with `AlgName` and `AlgFile`. A validated
snapshot is taken before solving, so the caller's Design is not used as a
mutable runtime object. Its stored population configuration remains authoritative;
if `ProbN` is omitted, the declaration's population size supplies the input default.

The default Python backend remains available without a compiler. Add
`ComponentBackend="native", StreamBackend="batch"` to use the existing C++
batch core when the native extension is installed. An unbound hand declaration
uses the requested backend. `execution_semantics="population_batch_v1"` pins a
graph to native batch; `"event_v1"` pins event semantics. A saved Search result
preserves its execution contract, selects its required batch default on load,
and rejects an explicitly incompatible backend.

## 3. Automatic design spaces

```python
from autooptlib import DesignSpace

domain = DesignSpace(
    choose=("choose_tournament", "choose_random"),
    search=("cross_point_one", "search_mu_uniform", "search_de_random"),
    update=("update_greedy", "update_pairwise"),
    parameters={"search_mu_uniform": ((0.05, 0.2),)},
    max_pathways=2,
    max_stages=2,
    population_sizes=(10, 20),
    offspring_sizes=(10, 20),
)

selected, trace = autoopt(
    Mode="design",
    Problem=problem,
    InstanceTrain=[8],
    InstanceTest=[12],
    DesignSpace=domain,
    ProbN=20,
    ProbFE=100,
    AlgN=1,
    AlgFE=4,
    AlgRuns=1,
    Seed=2026,
    OutputDir="results/model-design",
)
```

The small budgets above demonstrate the API, not scientific performance.

`choose`, `search`, and `update` are allowed catalogues, not fixed stage
positions. Search still chooses path count, stage placement and optional
selection according to the existing grammar. An explicit catalogue must be
non-empty; None keeps the existing problem-compatible catalogue.

Parameter intervals must narrow the component's declared bounds, preserve
parameter count, and contain a legal integer when its metadata requires one.
Equal endpoints fix a parameter. By default every occurrence has independent
parameters, keyed by `(pathway, stage, role, component)` through initialization,
decoding, mutation, CMA coordinates, workers and checkpoints. Set
`parameter_scope="component"` only to request legacy shared values.

```python
from autooptlib import NodeSpace

two_nodes = DesignSpace(
    search=("search_mu_uniform",), max_stages=1,
    population_sizes=(4,), offspring_sizes=(4,),
    nodes=(
        NodeSpace(0, 0, "search", ("search_mu_uniform",),
                  {"search_mu_uniform": ((0.05, 0.15),)}),
        NodeSpace(1, 0, "search", ("search_mu_uniform",),
                  {"search_mu_uniform": ((0.20, 0.30),)}),
    ),
)
```

Node constraints require those positions to exist and restrict choices at those
positions, including `choices=(None,)` for an absent selector. Initialization
constructs an admissible topology; structural mutation retains required nodes.
Unconstrained positions remain searchable. Local intervals may use the legal
execution range beyond a component's default Search range; if a global component
interval is explicitly supplied, local intervals must narrow it. Execution-valid
parameters and recommended Search intervals are deliberately different concepts.

The exact narrowed component, parameter and integer domains are used by both
initialization and subsequent Search actions through the original `space()`
pipeline. Random Design also consumes those domains. Explicit incompatible
components fail before candidate evaluation; they are not silently dropped.
`domain.resolve(constructed_problem)` lists the actual admissible component
metadata without running objectives. A constructed problem is a problem record
with `type` and `bound`, not a `make_problem` factory.

`AlgP`, `AlgQ`, `GraphSemantics` and size-space options cannot contradict a
declaration. `ProbN` remains an initialization/default hint and must fit `ProbFE`.
Design-space topology maxima are at most two. If crossover is allowed with
two-stage paths, at least one mutation must also be allowed; otherwise set
`max_stages=1`.

Search retains its existing novelty requirements and bounded attempt limits.
A very narrow or fully fixed declared space can stop before the requested budget.
After bounded attempts produce no admissible new candidate, Search returns the
evaluated incumbents and records
`no_admissible_action_within_attempt_limit` in controller statistics. This is a
bounded-search stopping reason, not a proof that all graphs were enumerated.

The existing problem-filtered catalogue intentionally differs from all 87
callable components: archives and state updates are managed by their own
execution roles, and not every callable is a Search choice for every problem.
For example, PSO may be hand-constructed even when absent from the current
automatic-design catalogue. No declaration widens these existing limits.

## 4. Manual seeds, exchange and inspection

Pass `SearchInitialDesigns=[algorithm]` to seed Search with a hand-constructed
graph. Its components, parameters and population configuration must belong to
the chosen space; its archive policy must match `Archive`. Warm-start count
cannot exceed `AlgN`. Initial candidates are evaluated by the ordinary Search
pipeline and do not constitute fixed templates.

Concrete graphs and node-scoped Search preserve different parameter values at
different occurrences. `DesignSpace.from_graph(algorithm)` fixes its component
positions, sizes and explicit numeric values; replace selected `NodeSpace`
intervals to optimize those values. Unspecified/default component parameters
retain their searchable domains. Legacy component scope rejects conflicting
occurrence values rather than silently collapsing them.

```python
import json
from autooptlib import (
    AlgorithmGraph, DesignSpace, algorithm_to_dict,
    load_algorithm, save_algorithm,
)

save_algorithm(selected[0], "selected.json")
loaded = load_algorithm("selected.json")       # existing Design return type
readable = AlgorithmGraph.from_design(loaded)
document = algorithm_to_dict(readable)          # existing algorithm schema 2
space_document = domain.to_dict()              # design-space schema 2
restored_space = DesignSpace.from_dict(json.loads(json.dumps(space_document)))
```

There is one algorithm JSON format and one lowered Design representation.
Schema-1 legacy pathway algorithms continue to load and run as Design objects;
`AlgorithmGraph.from_design` deliberately requires schema-2 stream graphs.
Graph declarations represent executable structure/configuration, not benchmark
performance or training history; keep Design/JSON metadata separately when needed.

Experiment manifests store full algorithm/space declarations rather than repr
strings. Existing checkpoint signatures include the resolved component and
parameter domains; changing the declared space cannot silently resume an
incompatible design run. JSON contains component names, not custom code.
Schema-1 design spaces retain component-scoped parameter semantics on import.
New checkpoint signatures additionally bind the execution contract, component
contracts and node constraints. Changing any of those rejects incompatible resume.

## 5. Execution semantics and state ownership

Graph grammar `stream_graph_v2` and runtime `population_batch_v1` name different
things. A declaration preserves the existing grammar; the selected backend
determines execution semantics. Python event and native batch runs need not
consume random numbers in the same order, so changing backend does not promise
identical whole-run trajectories or interchangeable checkpoints.
Execution contract version 1 fixes the grammar, state scope, evaluation boundary,
RNG ownership and ordered update groups. Serialized rule changes or unsupported
versions fail on import instead of silently changing a saved program's meaning.

For native batch execution:

1. A generation routes the requested offspring to pathways, preserving route order.
2. Each pathway executes its stages in order. Selection returns indices; crossover
   requests paired parents, including odd-output handling.
3. Stage output is repaired/evaluated before a downstream fitness-dependent stage.
   The second-stage mutation mask follows the configured mutation rate.
4. Path outputs return to route order. Matching update sinks are grouped using
   existing rules, then update groups execute in their established order.
5. CMA/PSO state updates, archives and termination follow their existing boundaries.

`population_size` controls survivors; `offspring_size` controls requested children.
`crossover_rate` affects route assignment when exactly one of two pathways contains
crossover; it is inactive for a single crossover pathway. `mutation_rate` controls
activity in the second stage of crossover-then-mutation paths. Component mutation
probabilities remain independent numeric parameters of their Operator.

The runtime owns one authoritative RNG per run, per-stage auxiliary state, FE
counters and logical row identities. Passing a graph never shares mutable state
between independent runs. PSO/CMA state is handled by the existing runtime;
arbitrary user auxiliary objects retain the existing owner/serialization rules.

Objective evaluations count toward `ProbFE`, including initialization, under the
existing evaluation/retry accounting policy. Native batch behavior at the final FE preserves observed
best results and its existing interruption/commit rule. Custom callbacks receive
independently owned decision arrays; the graph declaration does not make a black-box
objective pure or vectorized.

See [CPU execution](cpu-execution.md) for backend and objective-kernel details.

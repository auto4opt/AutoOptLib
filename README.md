# AutoOptLib

[![Version](https://img.shields.io/badge/version-1.2.0-blue.svg)](https://github.com/auto4opt/AutoOptLib/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Tests](https://github.com/auto4opt/AutoOptLib/actions/workflows/tests.yml/badge.svg)](https://github.com/auto4opt/AutoOptLib/actions/workflows/tests.yml)
[![Documentation](https://readthedocs.org/projects/autooptlib/badge/?version=latest)](https://autooptlib.readthedocs.io/)

AutoOptLib is a Python framework for component-based automated design of
metaheuristic optimizers. It searches over typed operator pathways containing
selection, search, and update components, and evaluates candidate algorithms on
user-provided training instances before testing them on held-out instances.

Version 1.2.0 is the first release packaged as a self-contained Python project.
The earlier MATLAB implementation remains available from the repository's
historical releases.

## Features

- Components for continuous, discrete, and permutation search spaces.
- Joint search over operator composition and component parameters.
- Exact, racing, intensification, and surrogate-assisted evaluation modes.
- Quality, evaluation-count, wall-clock, and anytime objectives.
- Deterministic runs through a single user-provided random seed.
- Strict `ProbFE` objective budgets and MATLAB-compatible `AlgFE` proposal budgets.
- Versioned JSON export for designed algorithms.
- A small public factory for ordinary Python objective functions.
- Runtime registration of custom components without editing package internals.
- Bundled CEC 2013 data that works from source and installed wheels.
- Executable reference models for material stacking and RIS beamforming.
- Retry, hard-timeout, failure-penalty, evaluation-cache, and JSONL logging
  controls for external objectives.
- Automatic experiment manifests with software and invocation provenance.

## Installation

AutoOptLib requires Python 3.9 or newer.

Install the latest release from PyPI:

```bash
python -m pip install autooptlib
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
```

## Quick start

The following example designs a small optimizer for the bundled CEC 2013
shifted-sphere problem. The small budgets are intended only as a smoke test.

```python
from autooptlib import autoopt, make_problem

sphere = make_problem(
    lambda decision, dimension: float((decision**2).sum()),
    bounds=(-5.0, 5.0),
    name="sphere",
)

algorithms, trace = autoopt(
    Mode="design",
    Problem=sphere,
    InstanceTrain=[10],
    InstanceTest=[20],
    AlgN=2,
    AlgFE=40,
    AlgRuns=2,
    ProbN=10,
    ProbFE=500,
    Evaluate="exact",
    Compare="average",
    Seed=2026,
    OutputDir="results/design",
)

print(algorithms[0].operator_pheno)
```

To run a built-in optimizer directly:

```python
from autooptlib import autoopt

best, history = autoopt(
    Mode="solve",
    Problem="cec2013_f1",
    InstanceSolve=[10],
    AlgName="CMA-ES",
    AlgRuns=3,
    ProbN=20,
    ProbFE=2000,
    Metric="quality",
    Seed=2026,
)
```

The design call writes portable files such as `Algorithm_1.json` to
`OutputDir`. Use one output directory per experiment. Designed algorithms can
later be loaded without executing pickle code:

```python
best, history = autoopt(
    Mode="solve",
    Problem=sphere,
    InstanceSolve=[30],
    AlgFile="results/design/Algorithm_1.json",
    ProbN=20,
    ProbFE=2000,
    Seed=2026,
    OutputDir="results/solve",
)
```

All public keyword names are case-insensitive and accept snake case. Unknown
keywords raise an error. `ProbFE` includes initial-population evaluations and
is a hard upper bound per algorithm run. For compatibility with the published
MATLAB procedure, `AlgFE` counts newly proposed algorithms after the initial
`AlgN` incumbents; held-out evaluation of the final algorithms is separate.
Sequential problems receive a fresh `ProbFE` budget at every stage.

## Reproducibility

Pass `Seed=<integer>` to `autoopt`. Version 1.2.0 routes this seed through
algorithm construction, component execution, selection, archiving, and
surrogate sampling. Repeating the same call with the same software version,
inputs, and platform produces the same stochastic sequence.

For expensive thread-safe objectives, set `EvalWorkers` above one to evaluate
candidate batches concurrently while preserving result order. Static and
sequential Solve runs and the outer Design search support atomic
`CheckpointDir`/`Resume` recovery; see the
[reliability guide](docs/reliability.md).

For archival experiments, record the AutoOptLib version, Python and NumPy
versions, operating system, complete call arguments, and the generated output
files.

## Extending the component library

Components use a mode-based callable interface. A custom component can be
registered at runtime:

```python
from autooptlib import register_component

def search_example(*args):
    mode = args[-1]
    if mode == "execute":
        parent = args[0]
        aux = args[3]
        return parent.decs(), aux
    if mode == "parameter":
        return None, None
    if mode == "behavior":
        return ["", "GS"], None
    raise ValueError(f"Unsupported mode: {mode}")

register_component(
    "search_example",
    search_example,
    category="search",
    problem_types=["continuous"],
)
```

Registered components are automatically added to compatible design spaces
created later in the same Python process.

For most applications, use `make_problem` as in the quick start. Advanced
dynamic or sequential problems can implement the `ProblemDefinition` protocol.
See [the user-problem guide](docs/custom-problems.md) and
[the API guide](docs/api.md). The canonical rendered documentation is hosted
on [Read the Docs](https://autooptlib.readthedocs.io/).

## Application examples

The repository contains runnable examples for the material-stacking and RIS
passive-beamforming problem classes discussed in the paper. The included
instances are synthetic and public; they validate the software workflow but do
not reproduce the paper's proprietary stacking records or original wireless
channel files. See [the application guide](docs/applications.md) and
[`examples/applications`](examples/applications/README.md).

## Scope and limitations

AutoOptLib performs offline algorithm design and can require many objective
function evaluations. It is most appropriate when the design cost can be
amortized over repeated solution of related instances. The default component
space is curated rather than exhaustive, and the current graph representation
encodes one or more bounded operator pathways rather than arbitrary programs.
It is a research-oriented beta: application-specific validation, complete
constraint modelling, simulator resource isolation, and deployment monitoring
remain the user's responsibility, although opt-in retry, timeout, penalty,
caching, and evaluation logging controls are available. AutoOptLib is not a
universal replacement for expert optimizer selection on a one-off problem.

## Contributing and support

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow. Report
bugs and request features through the
[GitHub issue tracker](https://github.com/auto4opt/AutoOptLib/issues).

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). Until the
software paper is published, please cite:

```bibtex
@article{zhao2023autooptlib,
  title   = {AutoOptLib: Tailoring Metaheuristic Optimizers via Automated Algorithm Design},
  author  = {Zhao, Qi and Yan, Bai and Hu, Taiwei and Chen, Xianglong and Duan, Qiqi and Yang, Jian and Shi, Yuhui},
  journal = {arXiv preprint arXiv:2303.06536},
  year    = {2023}
}
```

## License

AutoOptLib is licensed under the [Apache License 2.0](LICENSE).

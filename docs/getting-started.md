# Getting started

## Installation

AutoOptLib requires Python 3.9 or later. Until version 1.2.0 is available on
PyPI, install it from a checkout:

```bash
python -m pip install .
```

For development and documentation builds:

```bash
python -m pip install -e ".[dev]"
python -W error -m pytest
sphinx-build -W --keep-going -b html docs docs/_build/html
```

## Design and held-out evaluation

```python
from autooptlib import autoopt, make_problem

sphere = make_problem(
    lambda decision, dimension: float((decision**2).sum()),
    bounds=(-5.0, 5.0),
)

algorithms, trace = autoopt(
    Mode="design",
    Problem=sphere,
    InstanceTrain=[10],
    InstanceTest=[20],
    AlgN=2,
    AlgFE=20,
    AlgRuns=2,
    ProbN=10,
    ProbFE=200,
    Seed=2026,
    OutputDir="results/design",
)
```

The output directory contains a portable JSON description for each selected
algorithm, tabular performance data, a convergence trace, and an
`experiment.json` environment manifest.

Use realistic design budgets for scientific conclusions. The values above are
only suitable for checking that an installation works.

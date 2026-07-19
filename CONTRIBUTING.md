# Contributing to AutoOptLib

Contributions are welcome through GitHub issues and pull requests.

## Development setup

```bash
git clone https://github.com/auto4opt/AutoOptLib.git
cd AutoOptLib
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m ruff check .
python -m ruff format --check .
python -m pytest -W error
python -m pytest --cov=autooptlib --cov-report=term-missing --cov-fail-under=90
python -m build
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Pull requests

1. Open an issue for substantial behavior or API changes.
2. Keep changes focused and add tests for new behavior.
3. Preserve seeded reproducibility: stochastic code must use the generator
   supplied through component auxiliary state or the problem/setting object.
4. Update user documentation and `CHANGELOG.md` when behavior changes.
5. Confirm that the full test suite and wheel build pass locally.
6. Treat `ProbFE` and `AlgFE` as inclusive hard limits in new execution paths.

## Adding a component

A component must implement each mode relevant to its category:

- `execute`: run the component and return `(result, auxiliary_state)`;
- `parameter`: describe tunable parameter bounds;
- `behavior`: describe local/global-search behavior.

Use `register_component` for external components. Components proposed for the
built-in library should be added to the registry, documented, and covered by
deterministic unit tests.

## Benchmark changes

Changes to a bundled benchmark require numerical comparisons with an
independent official implementation. Do not approve a benchmark change solely
because it reaches the documented optimum; transformations can be wrong while
still passing that test.

## Reporting bugs

Include the AutoOptLib, Python, NumPy, and operating-system versions; the full
call arguments; the traceback; and a minimal reproducible example. Do not post
confidential application data in a public issue.

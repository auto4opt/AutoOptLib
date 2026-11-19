# AutoOptLib

**Version 3.1.0** · Python · Apache License 2.0

AutoOptLib designs metaheuristic optimization algorithms for user-defined problems.
It represents algorithms as executable graphs, searches their structures, operator
combinations and parameters, and runs the resulting algorithms through a shared
Python or native C++ execution engine.

- **Graph programming model:** construct algorithms with `Operator`, `Stage`,
  `Pathway` and `AlgorithmGraph`, or specify an admissible `DesignSpace`.
- **87 reusable operators:** selection, crossover, search, replacement, archives
  and state updates for continuous, discrete and permutation optimization.
- **Native CPU execution:** shared compiled operators, batch computation, SIMD
  where supported, and memory reuse. A Python backend remains available.

The supported graph grammar has one or two pathways and one or two stages per
pathway; a second stage follows crossover with mutation. It is not an arbitrary
control-flow graph. See the [programming model](docs/graph-programming-model.md).

## Install

Python 3.9 or later is required. Install this release directly from GitHub:

```sh
python -m pip install "git+https://github.com/auto4opt/AutoOptLib.git@v3.1.0"
```

Or clone the repository and install locally:

```sh
git clone https://github.com/auto4opt/AutoOptLib.git
cd AutoOptLib
python -m pip install .
```

Installation attempts to build the optional native extensions using a C++17
compiler. On Linux/macOS, use `AUTOOPTLIB_BUILD_NATIVE=0 python -m pip install .`
to install only the Python backend. Explicit native execution requires a
successful native build. See [CPU execution](docs/cpu-execution.md).

Optional dependencies: `.[cpu]` for Numba objective kernels, `.[applications]`
for MATLAB application-data import, and `.[benchmarks]` for the optional IOH
problem adapter. No external benchmark implementation is bundled.

## Local MCP

The optional local MCP server lets a compatible client design, run and chart
algorithms on your computer. It requires Python 3.10 or later. From a fresh
GitHub checkout, run these commands on macOS or Linux:

```sh
git clone --branch v3.1.0 https://github.com/auto4opt/AutoOptLib.git
cd AutoOptLib
python3 -m venv .venv-mcp
AUTOOPTLIB_BUILD_NATIVE=0 .venv-mcp/bin/python -m pip install '.[mcp]'
.venv-mcp/bin/python examples/local_mcp.py
```

The example starts the MCP server, designs an algorithm on Sphere, and uses
the resulting JSON to solve a new instance. It prints the objective and the
local artifact paths. On Windows, use `py -3 -m venv .venv-mcp` and
`.venv-mcp\Scripts\python.exe` for installation and the example. The server
command for MCP clients is `.venv-mcp\Scripts\autooptlib-mcp.exe`.

With Codex, register the server and reload the client's MCP configuration:

```sh
codex mcp add autooptlib -- "$PWD/.venv-mcp/bin/autooptlib-mcp"
```

Other stdio MCP clients can use the same absolute command path. Try asking
the model: “Use AutoOptLib to design an optimizer on 10-dimensional Sphere and
test it on 20-dimensional Sphere.” The [MCP usage guide](docs/local-mcp.md)
shows how to use BBOB, PBO, custom problems, algorithm graphs, components and
charts. The [custom problem guide](docs/custom-problems.md) describes the
default feasibility-first constraint handling.

## Design an algorithm

```python
from autooptlib import DesignSpace, autoopt, make_problem

problem = make_problem(
    lambda x, data: float((x * x).sum()),
    bounds=(-5.0, 5.0),
)
space = DesignSpace(
    choose=("choose_tournament", "choose_random"),
    search=("cross_point_one", "search_mu_uniform", "search_de_random"),
    update=("update_greedy", "update_pairwise"),
    max_pathways=2,
    max_stages=2,
    population_sizes=(10, 20),
    offspring_sizes=(10, 20),
)
algorithms, trace = autoopt(
    Mode="design",
    Problem=problem,
    InstanceTrain=[8],
    InstanceTest=[12],
    DesignSpace=space,
    ProbN=20,
    ProbFE=100,
    AlgN=1,
    AlgFE=4,
    AlgRuns=1,
    Seed=2026,
    OutputDir="outputs/design",
)
```

These small budgets demonstrate the API. Set budgets appropriate to your own
problem before assessing performance. `ProbFE` limits objective evaluations per
algorithm run. `AlgFE` limits candidate proposals during design.

## Execute a designed algorithm

```python
from autooptlib import autoopt, load_algorithm, save_algorithm

save_algorithm(algorithms[0], "algorithm.json")
algorithm = load_algorithm("algorithm.json")
best, histories = autoopt(
    Mode="solve",
    Problem=problem,
    InstanceSolve=[12],
    Algorithm=algorithm,
    ProbFE=200,
    AlgRuns=1,
    Seed=2026,
    OutputDir="outputs/solve",
)
```

For native CPU execution, add `ComponentBackend="native"` and
`StreamBackend="batch"` to the call that designs or executes the algorithm.
Saved designs preserve their execution contract; load them with the compatible
backend. Python/event and native/batch trajectories need not be identical.

## Source layout

| Path | Contents |
| --- | --- |
| `src/autooptlib/` | Public API, graph model, design and solve engines |
| `src/autooptlib/components/` | Reusable operator library and extension contracts |
| `src/autooptlib/runtime/` | Python runtime and optional native C++ source |
| `src/autooptlib/problems/` | Custom-problem interfaces and optional problem adapters |
| `examples/` | Small runnable usage examples |
| `tests/` | Software regression tests |
| `docs/` | User documentation |

The repository and distribution packages contain software source only: no paper
manuscripts, experiment campaigns, result archives, competitor implementations,
or comparison configurations.

## Documentation and development

- [Public API](docs/api.md)
- [Graph programming model](docs/graph-programming-model.md)
- [Operator catalogue](docs/component-catalogue.md)
- [Custom problems](docs/custom-problems.md)
- [CPU execution](docs/cpu-execution.md)
- [Application adapters](docs/applications.md)

```sh
python -m pip install -e '.[test]'
python -m pytest
```

Version 3 uses a single `autooptlib` namespace. Import problem helpers from
`autooptlib` or `autooptlib.problems`; the former top-level `problems`, `methods`
and `comparisons` packages are no longer installed. Use `Algorithm`, `AlgFile`
or a built-in `AlgName` to solve, and Search to design. Experimental comparison
entry points and bundled CEC data are not part of this release.

Licensed under [Apache License 2.0](LICENSE). The native source retains the
[NumPy license notice](src/autooptlib/runtime/_native/NUMPY_LICENSE.txt) required
for its reused numerical routines.

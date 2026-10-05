# AutoOptLib 2.0

AutoOptLib designs optimization algorithms as graphs of reusable components.
Version **2.0.0** provides a focused codebase with Search V6, the current Learning
method, benchmark problems, and the main comparison methods.

| Directory | Contents |
| --- | --- |
| [`methods/`](methods/) | Search V6, the Learning implementation frozen on 2026-09-30, shared components, and the execution engine |
| [`problems/`](problems/) | BBOB, PBO, CEC2013, material stacking, and RIS beamforming problems |
| [`comparisons/`](comparisons/) | Sparkle, ParadisEO/irace, Random Design, and hand-designed baselines |

The hand-designed baselines include BIPOP-CMA-ES, SHADE, PSO, GA, ILS, and SA.
Sparkle and ParadisEO integrations invoke their official implementations;
external source revisions are recorded in
[`comparisons/sources.lock.json`](comparisons/sources.lock.json).

This release contains the algorithm library and shared evaluation utilities.
Experiment campaigns, experimental results, and the additional standalone
SMAC/GOMEA designers are excluded. The proposed 2026-09-29 experiment protocol
is not shipped as an implemented campaign.

## Installation

Clone or download this release, then install it with Python 3.9 or later:

```sh
python -m pip install .
```

For Learning, IOH benchmarks, and comparison dependencies:

```sh
python -m pip install '.[experiments]'
```

Sparkle, ParadisEO, and irace require separate installation and building; see
[`comparisons/shared/README.md`](comparisons/shared/README.md).

## API and migration

The core API remains `from autooptlib import autoopt, make_problem`.
Problem modules now use `problems` instead of `autooptlib.problems`, and
application modules use `problems.applications`. Comparison implementations
are grouped under `comparisons.sparkle`, `comparisons.paradiseo`,
`comparisons.random_design`, and `comparisons.manual`.

Version 2.0 removes older method versions and changes module paths. Update
imports and entry points when migrating from 1.x. The project version is
**2.0.0**; the retained search algorithm remains **Search V6**.

Licensed under [Apache License 2.0](LICENSE).

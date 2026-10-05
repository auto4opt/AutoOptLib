# Comparison methods

| Directory | Implementation |
| --- | --- |
| `sparkle/` | Official Sparkle/SMAC3 integration; configuration entry point: `configure.py` |
| `paradiseo/` | Official ParadisEO/irace integration; configuration entry point: `configure.py` |
| `random_design/` | Random Design core in `search.py`, design workflow in `workflow.py` |
| `manual/` | BIPOP-CMA-ES, SHADE, and ILS in `algorithms.py`; PSO, GA, SA, and other presets in `presets.py` |
| `shared/` | Native runner, dependency setup, task definitions, scoring, records, and final evaluation |

Official external repositories are pinned in [sources.lock.json](sources.lock.json).
The repository contains integration code; official platform dependencies must
be installed separately. See [shared/README.md](shared/README.md) for setup.

Common final evaluation is implemented in `shared/evaluation.py`. The public
2.0 release excludes the additional standalone SMAC/GOMEA designers and
experiment campaign scripts.

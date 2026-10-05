# Official external methods and shared evaluation

Sparkle and ParadisEO/irace use their official implementations over the common
native `eoFastGA` runner. Exact upstream revisions are listed in
[../sources.lock.json](../sources.lock.json). Shared task definitions, scoring,
and records are in `protocol.py`, `objectives.py`, `scoring.py`, and `store.py`.

## Install external dependencies

Keep dependency checkouts, build products, and machine-specific configuration
outside the repository. For example, set `AOL_EXTERNAL` to a writable directory:

```sh
export AOL_EXTERNAL="$HOME/.cache/autooptlib-external"
python -m comparisons.shared.bootstrap --root "$AOL_EXTERNAL/sources"
```

Install the pinned irace checkout into R and the pinned Sparkle checkout into a
separate Python environment. Make `Rscript` and `sparkle` available on `PATH`,
or pass their paths with `--rscript` and `--sparkle` below. CMake and a C++
compiler are also required.

```sh
python -m comparisons.shared.setup_native \
  --source-root "$AOL_EXTERNAL/sources" \
  --output-root "$AOL_EXTERNAL/build" \
  --manifest "$AOL_EXTERNAL/external-methods.json" \
  --jobs 8 --workers 8
python -m comparisons.shared.preflight \
  --manifest "$AOL_EXTERNAL/external-methods.json"
```

The preflight verifies native BBOB/PBO execution, FE accounting, IOH optimum
metadata, the irace R package, and the Sparkle CLI. Use `--native-only` to check
the native runner before installing R and Sparkle.

## Configuration adapters

The manifest registers `comparisons.paradiseo.configure` and
`comparisons.sparkle.configure`. Each accepts an explicit design request,
invokes the official configurator, and writes a selected algorithm artifact.
Run either module with `--help` for its arguments.

The adapters share a durable FE ledger. Repeated configurations reuse completed
records; candidate counts and FE use are audited. Resume requires the same
request identity, native template, tasks, and seeds. irace supports parallel
target calls; each Sparkle configuration job uses its official sequential CLI.

Sparkle compatibility workarounds are scoped to its subprocess workspace.
They prevent blocked logging pipes and scheduler waits, restore the official
incumbent-update callback, and synchronize explicit seeds. They do not modify
the pinned upstream checkouts.

Experiment schedules and campaign scripts are not included in the public 2.0
release. Callers must supply their own explicit task and budget configuration;
these utilities do not implement the proposed 2026-09-29 campaign.

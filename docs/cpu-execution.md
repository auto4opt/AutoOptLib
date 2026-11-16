# CPU execution

AutoOptLib has a Python component backend and an optional C++17 native backend.
Both use the public graph model and problem interface.

## Build and check

```sh
python -m pip install .
python -c "from autooptlib.runtime._native import _native_stream, _native_compatibility"
```

A C++17 compiler and the declared build dependencies are required for native
extensions. Their build is optional: installation may succeed with only the
Python backend when a compiler is unavailable. The import check above verifies
that both native extensions actually built.

To build against an existing NumPy version:

```sh
python -m pip install 'setuptools>=77' wheel 'pybind11>=2.12,<4' 'numpy>=1.23,<3'
python -m pip install --no-build-isolation .
```

For an explicit Python-only installation, set `AUTOOPTLIB_BUILD_NATIVE=0`
before running pip. In PowerShell use `$env:AUTOOPTLIB_BUILD_NATIVE="0"`.

## Select the backend

Pass `ComponentBackend="native", StreamBackend="batch"` to `autoopt`.
The native engine uses shared compiled operator routines, population batches,
SIMD for supported kernels and CPU capabilities, and reusable memory buffers.
It retains Python callbacks where a problem has no compatible native evaluator.

The default Python backend works without the native extension. Native batch
and Python/event execution have different execution semantics and may follow
different random trajectories. Saved algorithms record their execution
contract. Use the compatible backend when reloading them.

## Objective evaluation

An arbitrary Python objective remains usable. For an independent, deterministic
numerical objective, `CPUObjective` can provide an explicit batch kernel:

```python
from autooptlib import CPUObjective, make_problem

cpu = CPUObjective(
    evaluate_into=kernel,
    n_constraints=0,
    capacity=64,
    backend="python",
    independent=True,
)
problem = make_problem(reference_objective, bounds=(-5, 5), cpu=cpu)
```

`kernel(X, prepared, workspace, objectives, constraints)` writes into the
provided output buffers. It must not modify input decisions, retain temporary
buffers, draw random numbers or depend on evaluation order. A matching scalar
reference remains required. Optional `.[cpu]` dependencies enable Numba kernels.

`CPUObjective.from_native(...)` accepts a compatible external native evaluator;
see its docstring for the function-pointer and buffer contract.

## Parallel evaluation

`EvalWorkers=1` is the conservative default. Multiple workers use separate
problem/run state. Control native-library threads with `EvalCoresPerWorker`
and avoid oversubscribing a machine with nested process or BLAS pools.
Worker-local simulator initialization is available through
`EvaluationWorkerEnvironment`, `WorkerContext` and `EvalWorkerInitializer`.

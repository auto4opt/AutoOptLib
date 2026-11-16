"""Explicit, initialization-bound CPU objective adapters; no scheduler or online selector."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache, cached_property
from typing import Any, Callable
import ctypes
import hashlib
from pathlib import Path

import numpy as np


@lru_cache(maxsize=64)
def _compiled_kernel(function):
    try:
        from numba import njit
    except ImportError as exc:
        raise ImportError("Install autooptlib[cpu] to use backend='numba'") from exc
    return njit(cache=True, fastmath=False, parallel=False)(function)


@lru_cache(maxsize=64)
def _jitable_scalar(function):
    from numba.extending import register_jitable
    return register_jitable(inline="always")(function)


@dataclass(frozen=True)
class _ScalarKernel:
    objective: Callable
    constraint: Callable | None
    vector_constraint: bool

    def resolve(self, backend):
        objective = self.objective
        constraint = self.constraint
        if backend == "numba":
            objective = _jitable_scalar(objective)
            if constraint is not None:
                constraint = _jitable_scalar(constraint)
        # Select the implementation here, never in the per-individual loop.
        if constraint is None:
            def into(X, data, workspace, y, c):
                for i in range(len(X)):
                    y[i] = objective(X[i], data)
        elif self.vector_constraint:
            def into(X, data, workspace, y, c):
                for i in range(len(X)):
                    y[i] = objective(X[i], data)
                    c[i, :] = constraint(X[i], data)
        else:
            def into(X, data, workspace, y, c):
                for i in range(len(X)):
                    y[i] = objective(X[i], data)
                    c[i, 0] = constraint(X[i], data)
        return into


@dataclass(frozen=True)
class _NativeKernel:
    library: str
    symbol: str
    digest: str
    dtype: str
    builtin: bool = False

    def resolve(self, prepared):
        if self.builtin and self.symbol == "ordered_sum_squares":
            prepared = None
        return _NativeCall(self, prepared)


class _NativeCall:
    def __init__(self, spec, prepared):
        if spec.builtin:
            from autooptlib.runtime.component_library import load_component_library
            self.owner = load_component_library()
            info = self.owner.cpu_objective_kernel(spec.symbol)
            self.address, self.isa = info["address"], info["isa"]
            self.n_constraints = info["constraints"]
        else:
            if hashlib.sha256(Path(spec.library).read_bytes()).hexdigest() != spec.digest:
                raise ValueError("CPU objective library changed since construction")
            self.owner = ctypes.CDLL(spec.library)
            self.address = ctypes.cast(getattr(self.owner, spec.symbol), ctypes.c_void_p).value
            self.isa = "external"
        if prepared is not None and (not isinstance(prepared, np.ndarray)
                or not prepared.flags.c_contiguous or prepared.dtype.hasobject):
            raise TypeError("Native objective context must be a contiguous non-object ndarray or None")
        self.context_owner = prepared
        self.context = 0 if prepared is None else prepared.ctypes.data
        signature = ctypes.PYFUNCTYPE(None, ctypes.c_int64, ctypes.c_int64,
                                     ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)
        self.function = signature(self.address)

    def __call__(self, X, data, workspace, y, c):
        self.function(len(X), X.shape[1], X.ctypes.data, y.ctypes.data, c.ctypes.data, self.context)


def _borrow_immutable(data):
    return data


@dataclass(frozen=True)
class CPUObjective:
    """Pure, independent kernel on the repaired domain, with explicit numerical equivalence.

    evaluate_into(X, prepared, workspace, objectives, raw_constraints) overwrites
    all outputs and retains/mutates no input or output references. No RNG or
    auxiliary objects. Compilation does not imply NumPy reduction equivalence.
    """
    evaluate_into: Callable | _ScalarKernel | _NativeKernel
    prepare: Callable[[Any], Any] = deepcopy
    workspace_factory: Callable[[Any, int], Any] | None = None
    n_constraints: int = 0
    backend: str = "python"
    capacity: int = 64
    independent: bool = False

    def __post_init__(self):
        if not self.independent:
            raise ValueError("CPUObjective requires explicit independent=True")
        if self.backend not in ("python", "numba", "native"):
            raise ValueError("CPUObjective backend must be python, numba or native")
        if self.capacity < 1 or self.n_constraints < 0:
            raise ValueError("CPUObjective requires positive capacity and nonnegative n_constraints")
        if not (callable(self.evaluate_into) or isinstance(self.evaluate_into, (_ScalarKernel, _NativeKernel))):
            raise TypeError("CPUObjective kernel must be callable or an explicit adapter")
        if not callable(self.prepare):
            raise TypeError("CPUObjective prepare must be callable")
        if self.backend == "native" and not isinstance(self.evaluate_into, _NativeKernel):
            raise TypeError("Use from_native or from_builtin for backend='native'")
        if self.workspace_factory is not None and not callable(self.workspace_factory):
            raise TypeError("CPUObjective workspace_factory must be callable")

    @classmethod
    def borrowed(cls, evaluate_into, **kwargs):
        """Borrow caller-owned immutable prepared data with a retained owner.

        The caller guarantees lifetime-long immutability, including all aliases;
        no scanning, flag probing or runtime change detection is performed.
        """
        return cls(evaluate_into, prepare=_borrow_immutable, **kwargs)

    @classmethod
    def from_scalar(cls, objective, *, constraint=None, n_constraints=0, backend="numba", **kwargs):
        """Lift a compatible scalar implementation; no speculative evaluations.

        Specify n_constraints=1 for a scalar constraint, >1 for a fixed vector.
        Preserve the reference arithmetic: Numba's np.sum need not equal NumPy's.
        """
        if not callable(objective) or (constraint is not None and not callable(constraint)):
            raise TypeError("Scalar objective and constraint must be callable")
        if (constraint is None) != (n_constraints == 0):
            raise ValueError("Declare the scalar/vector constraint count explicitly")
        return cls(_ScalarKernel(objective, constraint, n_constraints > 1),
                   n_constraints=n_constraints, backend=backend, **kwargs)

    @classmethod
    def from_native(cls, library, symbol, *, decision_dtype="float64", **kwargs):
        """C ABI: void f(int64 n,int64 d,const void* X,double* y,double* c,const void* ctx).

        Keep the library immutable. Context is an owned numeric ndarray (or None),
        and its layout, input dtype and constraint count are the library contract.
        """
        path = str(Path(library).resolve())
        spec = _NativeKernel(path, symbol, hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                             np.dtype(decision_dtype).str)
        return cls(spec, backend="native", **kwargs)

    @classmethod
    def from_builtin(cls, name, *, decision_dtype="float64", **kwargs):
        return cls(_NativeKernel("", name, "", np.dtype(decision_dtype).str, True),
                   backend="native", **kwargs)

    def bind(self, data, dimension, dtype):
        return _CPUObjective(self, data, dimension, dtype)


class _CPUObjective:
    def __init__(self, spec, data, dimension, dtype):
        self.spec = spec
        # Construction-time snapshot is an existing observable CPUObjective
        # contract. No template workspace or compilation is needed for it.
        self.prepared = spec.prepare(data)
        self.dimension, self.dtype = dimension, np.dtype(dtype)
        if isinstance(spec.evaluate_into, _NativeKernel) and self.dtype.str != spec.evaluate_into.dtype:
            raise TypeError("Native objective decision dtype does not match bounds")
        self.kernel = spec.evaluate_into
        self._resolved = None
        self._free = []
        self._scalar_entry, self._rows_entry = self._first_scalar, self._first_rows

    def __getstate__(self):
        return dict(spec=self.spec,prepared=self.prepared,dimension=self.dimension,
                    dtype=self.dtype,runtime=hasattr(self,"_owner"))

    def __setstate__(self, state):
        runtime=state.pop("runtime")
        self.__dict__.update(state)
        self.kernel=self.spec.evaluate_into
        self._resolved=None;self._free=[]
        self._scalar_entry,self._rows_entry=self._first_scalar,self._first_rows
        if runtime:
            owner=object.__new__(type(self));owner.__dict__.update(self.__dict__)
            ready=owner.fresh();self.__dict__.update(ready.__dict__)

    def _runtime(self):
        result = object.__new__(_CPURuntime)
        result.spec, result.prepared = self.spec, self.prepared
        result.dimension, result.dtype = self.dimension, self.dtype
        result.workspace = (None if self.spec.workspace_factory is None else
                            self.spec.workspace_factory(self.prepared, self.spec.capacity))
        if self._resolved is None:
            kernel = self.spec.evaluate_into
            if isinstance(kernel, _ScalarKernel):
                kernel = kernel.resolve(self.spec.backend)
            if isinstance(kernel, _NativeKernel):
                kernel = kernel.resolve(self.prepared)
            elif self.spec.backend == "numba":
                from numba import typeof, types, from_dtype
                kernel = _compiled_kernel(kernel)
                kernel.compile((types.Array(from_dtype(self.dtype),2,'C'),typeof(self.prepared),
                    typeof(result.workspace),types.Array(types.float64,1,'C'),types.Array(types.float64,2,'C')))
            if isinstance(kernel, _NativeCall) and hasattr(kernel, "n_constraints") and kernel.n_constraints >= 0 and kernel.n_constraints != self.spec.n_constraints:
                raise ValueError("Built-in objective constraint count mismatch")
            self._resolved = self.kernel = kernel
        result.kernel = self._resolved
        result.normalize = _normalize_constrained if self.spec.n_constraints else _normalize_unconstrained
        result.scalar, result.rows, result.arrays = result._scalar_run, result._rows_run, result._arrays_run
        result._owner = self
        return result

    def fresh(self):
        result = self._free.pop() if self._free else self._runtime()
        result._leased = True
        return result

    def release(self):
        # A run-boundary operation, never a candidate or kernel admission check.
        if self._leased and self._owner is not None:
            self._leased = False
            self._owner._free.append(self)

    def _standalone(self):
        runtime = self._runtime()
        runtime._owner = None
        self._scalar_entry, self._rows_entry = runtime.scalar, runtime.rows
        return runtime

    def _first_scalar(self, data, decision):
        return self._standalone().scalar(data, decision)

    def _first_rows(self, decisions):
        return self._standalone().rows(decisions)

    def scalar(self, data, decision):
        return self._scalar_entry(data, decision)

    def rows(self, decisions):
        return self._rows_entry(decisions)

    def _scalar_run(self, _data, decision):
        self.inputs[0] = decision
        self.kernel(self.inputs[:1], self.prepared, self.workspace,
                    self.objectives[:1], self.constraints[:1])
        return float(self.objectives[0]), self.constraints[0].copy(), None

    def _arrays_run(self, X, objectives, constraints):
        # Native repaired storage is borrowed directly. The existing fixed
        # kernel capacity bounds scratch, not the size of the admitted batch.
        for start in range(0, len(X), self.spec.capacity):
            stop = start + self.spec.capacity
            try:
                self.kernel(X[start:stop], self.prepared, self.workspace,
                            objectives[start:stop], constraints[start:stop])
            except Exception as error:
                from autooptlib.utils.solve import _native_evaluator_error
                _native_evaluator_error(error, X[start])

    def _rows_run(self, decisions):
        # Explicit legacy-list compatibility. Native execution never enters here.
        results = []
        for start in range(0, len(decisions), self.spec.capacity):
            rows = decisions[start:start + self.spec.capacity]
            n = len(rows)
            x, y, c = self.inputs[:n], self.objectives[:n], self.constraints[:n]
            np.stack(rows, out=x)
            self._arrays_run(x, y, c)
            results.extend(self.normalize(y, c))
        return results


class _CPURuntime(_CPUObjective):
    @cached_property
    def inputs(self):
        # Only the scalar/list compatibility API needs staging arrays. The
        # non-data descriptor replaces itself; subsequent reads are plain attrs.
        self.objectives=np.empty(self.spec.capacity,dtype=np.float64)
        self.constraints=np.empty((self.spec.capacity,self.spec.n_constraints),dtype=np.float64)
        return np.empty((self.spec.capacity,self.dimension),dtype=self.dtype)


def _normalize_constrained(y, c):
    from autooptlib.utils.solve import _normalize_evaluator_output
    return [_normalize_evaluator_output((y[i], c[i], None), []) for i in range(len(y))]


def _normalize_unconstrained(y, _c):
    from autooptlib.utils.solve import _normalize_evaluator_output
    return [_normalize_evaluator_output((value, 0.0, None), []) for value in y]


def bind_runtime(problem, setting):
    source = getattr(problem, "_autoopt_cpu_objective", None)
    if source is None:
        return
    current = problem.evaluate
    if (not isinstance(getattr(current, "__self__", None), _CPUObjective)
            and current != problem._autoopt_cpu_reference_evaluate):
        problem._autoopt_cpu_runtime = None
        return
    enabled = (getattr(problem, "_autoopt_eval_cache", None) is None
               and problem._autoopt_eval_log is None
               and problem._autoopt_eval_timeout is None
               and problem._autoopt_eval_retries == 0
               and problem._autoopt_eval_failure == "raise"
               and problem.type[2] == "certain")
    if enabled:
        bound = source.fresh()
        from autooptlib.utils import solve
        norm = next(value for name,value,_code in solve._NATIVE_EVALUATOR_POLICY
                    if name == '_normalize_evaluator_output')
        expected_code = next(code for name,_value,code in solve._NATIVE_EVALUATOR_POLICY
                             if name == '_normalize_evaluator_output')
        bound.array_results = (solve._normalize_evaluator_output is norm
            and norm.__code__ is expected_code
            and all(vars(np).get(name) is value for name,value in solve._NORMALIZE_NUMPY))
        problem.evaluate = bound.scalar
        problem._autoopt_cpu_runtime = bound
    else:
        problem.evaluate = problem._autoopt_cpu_reference_evaluate
        problem._autoopt_cpu_runtime = None

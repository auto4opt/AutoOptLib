"""Python boundary services for the C++ population search session.

The reference PopulationExecution stays in population_graph.py. This module
converts state only at run/checkpoint/public boundaries, never per component.
"""
from __future__ import annotations

from dataclasses import dataclass

from .component_library import load_component_library
from .population import PopulationBatch


@dataclass(frozen=True)
class NativeRunRequest:
    """One complete solve request for ``solve.run_design_many``.

    Settings supply each run's authoritative RNG. External owners deliberately
    shared by requests retain their ordered semantics.
    """
    pathways: object
    parameters: object
    problem: object
    data: object
    setting: object
    algorithm_configuration: object = None


def checkpoint_observable(payload, solve):
    """Unknown persistence code can mutate records without retaining references.

    Only the original writer and ordinary pickle data can use private-row
    provenance after saving. This is a cold boundary, never a stage cache.
    """
    import numpy as np

    if (solve._write_checkpoint is not solve._CHECKPOINT_WRITER
            or solve._write_checkpoint.__code__ is not solve._CHECKPOINT_WRITER_CODE):
        return True
    pending, seen = [payload], set()
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        kind = type(value)
        if value is None or kind in (bool, int, float, str, bytes):
            continue
        if kind is dict:
            pending.extend(value.keys())
            pending.extend(value.values())
        elif kind in (list, tuple):
            pending.extend(value)
        elif kind in (solve.Solution, PopulationBatch):
            pending.append(vars(value))
        elif kind is np.ndarray:
            if value.dtype.hasobject or value.dtype.metadata is not None:
                return True
        elif kind is np.random.Generator:
            if type(value.bit_generator) not in (np.random.PCG64, np.random.MT19937,
                                                  np.random.Philox, np.random.SFC64):
                return True
        elif kind.__module__ == 'numpy' and isinstance(value, np.generic):
            if value.dtype.hasobject or value.dtype.metadata is not None:
                return True
        else:
            return True
    return False


def convert_state(value, *, native, memo=None, _library=None):
    """Preserve aliasing and the authoritative Generator across cold boundaries."""
    from ..utils import solve

    library = load_component_library() if _library is None else _library
    memo = {} if memo is None else memo
    key = id(value)
    if key in memo:
        return memo[key]
    if native and isinstance(value, PopulationBatch):
        result = library.PopulationStorage.from_population(value)
    elif not native and isinstance(value, library.PopulationStorage):
        result = value.to_population(PopulationBatch, solve.Solution)
    elif isinstance(value, dict):
        result = {}
        memo[key] = result
        result.update((k, convert_state(v, native=native, memo=memo, _library=library)) for k, v in value.items())
    elif isinstance(value, list):
        result = []
        memo[key] = result
        result.extend(convert_state(v, native=native, memo=memo, _library=library) for v in value)
    elif isinstance(value, tuple):
        result = tuple(convert_state(v, native=native, memo=memo, _library=library) for v in value)
    else:
        return value
    memo[key] = result
    return result


class NativePopulationExecution:
    mode = 'batch'

    def __init__(self, pathways):
        from ..utils import solve

        library = load_component_library()
        self.session = library.RunSession(solve, pathways)
        from .data_planning import configure_session

        configure_session(self.session, library)

    def generation(self, pathways, parameters, solutions, auxiliary, problem, data,
                   setting, generation, remaining, archive_names, archives, event_budget):
        """One-generation compatibility boundary used by differential tests."""
        aux, archive = convert_state((auxiliary, archives), native=True)
        result = self.session.generation(pathways, parameters, solutions, aux, problem,
                                         data, setting, generation, remaining,
                                         archive_names, archive, event_budget)
        public_aux, public_archive = convert_state((result[1], result[3]), native=False)
        return result[0], public_aux, result[2], public_archive, result[4], result[5]

    def _prepare_context(self, context):
        from ..utils import solve

        if self.session.closed:
            raise RuntimeError('RunSession is closed')

        cpu = getattr(context['problem'], '_autoopt_cpu_runtime', None)
        arrays = cpu is not None and cpu.array_results
        context['cpu_rows'] = None if cpu is None else cpu.arrays if arrays else cpu.rows
        context['cpu_arrays'] = arrays
        context['cpu_constraints'] = 0 if cpu is None else cpu.spec.n_constraints
        context['cpu_address'] = 0 if not arrays else getattr(cpu.kernel, 'address', 0)
        context['cpu_context'] = 0 if not arrays else getattr(cpu.kernel, 'context', 0)

        path = context['checkpoint_path']
        def save(snapshot):
            if not isinstance(snapshot, dict):
                snapshot = snapshot._legacy_live(solve)
                snapshot.pop('rng_state', None)
            snapshot['aux_cache'], snapshot['archives'] = convert_state(
                (snapshot['aux_cache'], snapshot['archives']), native=False)
            payload = solve._checkpoint_payload(
                **snapshot, population_size=context['population_size'],
                evaluation_budget=context['evaluation_budget'],
                checkpoint_signature=context['checkpoint_signature'])
            observable = checkpoint_observable(payload, solve)
            solve._write_checkpoint(path, payload)
            return observable
        context['checkpoint'] = save if path is not None else None
        context['native_checkpoint'] = True
        if 'resume_state' in context:
            context['resume_state'] = convert_state(context['resume_state'], native=True)
        else:
            context['auxiliary'], context['archives'] = convert_state(
                (context['auxiliary'], context['archives']), native=True)
        context['clock'] = getattr(context['setting'], '_NativeClock', None)
        return context

    def run(self, context):
        cpu = getattr(context['problem'], '_autoopt_cpu_runtime', None)
        try:
            result = self.session.run(self._prepare_context(context))
            result['archives'] = convert_state(result['archives'], native=False)
            return result
        finally:
            if cpu is not None:
                cpu.release()

    def run_many(self, contexts, *, on_result=None, collect_results=True):
        """Execute prepared independent runs in one native session.

        Contexts supply their own RNG and problem instance, as in ``run``.
        Cancellation returns the completed prefix. An error stops submission,
        releases the lifecycle guard, and preserves completed-run progress.
        """
        def prepare():
            for context in contexts:
                cpu = getattr(context['problem'], '_autoopt_cpu_runtime', None)
                try:
                    yield self._prepare_context(context)
                finally:
                    if cpu is not None:
                        cpu.release()
        prepared = prepare()

        def consume(result, index):
            result['archives'] = convert_state(result['archives'], native=False)
            on_result(result, index)

        try:
            results = self.session.run_many(prepared, consume if on_result is not None else None, collect_results)
        finally:
            prepared.close()
        if on_result is None:
            for result in results:
                result['archives'] = convert_state(result['archives'], native=False)
        return results

    def snapshot(self):
        return self.session.snapshot()

    def restore(self, context, snapshot):
        cpu = getattr(context['problem'], '_autoopt_cpu_runtime', None)
        try:
            result = self.session.restore(self._prepare_context(context), snapshot)
            result['archives'] = convert_state(result['archives'], native=False)
            return result
        finally:
            if cpu is not None:
                cpu.release()

    @property
    def control(self):
        return self.session.control

    def close(self):
        self.session.close()

    def __enter__(self):
        self.session.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

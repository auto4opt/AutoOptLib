"""Dense population data shared by all native batch components.

Individuals are constructed only at evaluation/public API boundaries. Components
consume X/F arrays; selection and replacement return row indices in BatchResult.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _same_scalar(left, right):
    return np.float64(left).tobytes() == np.float64(right).tobytes()


class PopulationBatch:
    def __init__(self, decisions, fitness=None, objectives=None, constraints=None,
                 *, valid=None, row_ids=None, records=None, auxiliary=None):
        x = np.asarray(decisions)
        if x.ndim != 2 or x.dtype not in (np.dtype('float64'), np.dtype('int64')):
            raise ValueError('Population decisions must be a 2-D float64 or int64 matrix')
        self._batch_X = np.ascontiguousarray(x)
        self._batch_F = self._column(fitness, len(x), 'fitness')
        self.objectives = self._column(objectives, len(x), 'objectives')
        self.constraints = self._column(constraints, len(x), 'constraints')
        self.valid = np.full(len(x), fitness is not None, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
        self.row_ids = np.arange(len(x), dtype=np.int64) if row_ids is None else np.asarray(row_ids, dtype=np.int64)
        if self.valid.shape != (len(x),) or self.row_ids.shape != (len(x),):
            raise ValueError('Population validity and row IDs must have one entry per row')
        self.auxiliary = (None,) * len(x) if auxiliary is None else tuple(auxiliary)
        if len(self.auxiliary) != len(x):
            raise ValueError('Population auxiliary values must match its rows')
        self._records = None if records is None else tuple(records)
        if self._records is not None and len(self._records) != len(x):
            raise ValueError('Population boundary records must match its rows')

    @staticmethod
    def _column(value, n, name):
        if value is None:
            return None
        result = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
        if result.shape != (n,):
            raise ValueError(f'Population {name} must have one entry per row')
        return result

    def __getstate__(self):
        state = dict(vars(self))
        state.pop('_records', None)
        return state

    def __setstate__(self, state):
        vars(self).update(state)
        self._records = None

    @property
    def X(self):
        return self._batch_X

    @property
    def F(self):
        return self._batch_F

    def __len__(self):
        return self.X.shape[0]

    def require_evaluated(self):
        if self.F is None or not self.valid.all():
            raise ValueError('This component requires evaluated population rows')

    def take(self, indices):
        i = np.asarray(indices, dtype=np.int64).reshape(-1)
        return PopulationBatch(self.X[i], None if self.F is None else self.F[i],
                               None if self.objectives is None else self.objectives[i],
                               None if self.constraints is None else self.constraints[i],
                               valid=self.valid[i], row_ids=self.row_ids[i],
                               records=None if self._records is None else [self._records[int(j)] for j in i],
                               auxiliary=[self.auxiliary[int(j)] for j in i])

    @classmethod
    def concatenate(cls, batches):
        parts = list(batches)
        if not parts:
            raise ValueError('At least one population batch is required')
        if len(parts) == 1:
            return parts[0]
        if any(p.X.shape[1] != parts[0].X.shape[1] for p in parts):
            raise ValueError('Population dimensions must match')
        def join(name):
            values = [getattr(p, name) for p in parts]
            return None if any(v is None for v in values) else np.concatenate(values)
        records = None if any(p._records is None for p in parts) else [record for snapshot in (list(p._records) for p in parts) for record in snapshot]
        return cls(np.concatenate([p.X for p in parts]), join('F'), join('objectives'), join('constraints'),
                   valid=join('valid'), row_ids=join('row_ids'), records=records,
                   auxiliary=[value for snapshot in (list(p.auxiliary) for p in parts) for value in snapshot])

    @classmethod
    def from_solutions(cls, solutions, *, dimension=None, dtype=float):
        rows = list(solutions)
        x = np.stack([np.asarray(s.dec) for s in rows]) if rows else np.empty((0, dimension or 0), dtype=dtype)
        return cls(x, [s.fit for s in rows], [s.obj for s in rows], [s.con for s in rows],
                   records=rows, auxiliary=[s.acc for s in rows])

    def to_solutions(self):
        from ..utils.solve import Solution, SolutionSet
        self.require_evaluated()
        result = []
        for i in range(len(self)):
            prior = None if self._records is None else self._records[i]
            obj = self.F[i] if self.objectives is None else self.objectives[i]
            con = 0.0 if self.constraints is None else self.constraints[i]
            if (prior is not None and np.asarray(prior.dec).dtype == self.X.dtype
                    and np.asarray(prior.dec).shape == self.X[i].shape
                    and np.asarray(prior.dec).tobytes() == self.X[i].tobytes()
                    and _same_scalar(prior.fit, self.F[i]) and _same_scalar(prior.obj, obj)
                    and _same_scalar(prior.con, con) and prior.acc is self.auxiliary[i]):
                result.append(prior)
            else:
                result.append(Solution(self.X[i].copy(), float(obj), float(con), float(self.F[i]), self.auxiliary[i]))
        return SolutionSet(result)


@dataclass
class BatchResult:
    population: PopulationBatch
    state: dict
    indices: np.ndarray | None = None
    statistics: np.ndarray | None = None
    invalidated: np.ndarray | None = None


class BatchComponent:
    """One prebound native entry for every name in the component registry."""
    def __init__(self, name, native):
        self.name = name
        self.native = native

    def __call__(self, population, problem, parameters=None, state=None, *,
                 generation=0, inner_generation=1, data=None, count=None,
                 context=None, targets=None, archive=None, mode='solution'):
        if not isinstance(population, PopulationBatch):
            raise TypeError('Batch components require PopulationBatch input')
        name = self.name
        state = {} if state is None else state
        if name.startswith(('choose_', 'update_', 'para_')) or name in ('archive_best', 'archive_statistic'):
            if name not in ('choose_random', 'choose_traverse'):
                population.require_evaluated()
        if name.startswith('para_'):
            value = self.native(population, problem, state, mode)
            if name == 'para_pso' and isinstance(value, tuple):
                indices, best = value
                combined = PopulationBatch.concatenate([state['Pbest'], population])
                state['Pbest'] = combined.take(indices)
                state['Gbest'] = state['Pbest'].X[int(best)].copy()
            else:
                state = value
            return BatchResult(population, state)
        if name.startswith('archive_'):
            if name == 'archive_diversity':
                combined = population if archive is None else PopulationBatch.concatenate([population, archive])
                raw, _ = self.native(combined, None, problem, None, generation, 1, data, 'execute')
                return BatchResult(combined.take(raw), state, indices=raw)
            raw, _ = self.native(population, archive, None, None, generation, 1, data, 'execute')
            if name == 'archive_statistic':
                return BatchResult(population, state, statistics=raw)
            if name == 'archive_tabu':
                return BatchResult(population, state)
            best = population.take(raw)
            return BatchResult(best if archive is None else PopulationBatch.concatenate([archive, best]), state, indices=raw)
        if count is not None:
            state['_selection_count'] = int(count)
        if context is not None:
            state['_population_context'] = context
        if targets is not None:
            state['_target_population_indices'] = np.asarray(targets, dtype=np.int64)
        try:
            raw, returned_state = self.native(population, problem, parameters, state,
                                              generation, inner_generation, data, 'execute')
        finally:
            for key in ('_selection_count', '_population_context', '_target_population_indices'):
                state.pop(key, None)
        if isinstance(returned_state, dict):
            if 'rng' in state:
                returned_state.setdefault('rng', state['rng'])
            state = returned_state
            for key in ('_selection_count', '_population_context', '_target_population_indices'):
                state.pop(key, None)
        if name.startswith('choose_'):
            return BatchResult(population, state, indices=raw)
        if name.startswith('update_'):
            return BatchResult(population.take(raw), state, indices=raw)
        invalidated = state.pop('_stream_invalidated_mask', None)
        return BatchResult(PopulationBatch(raw), state, invalidated=invalidated)

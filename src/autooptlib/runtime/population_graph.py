"""Population graph execution: one native component call per stage batch.

This is an explicit generational execution contract. The old stream executor
remains available when exact event-by-event trajectories are required.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from .population import PopulationBatch

POPULATION_EXECUTION_SEMANTICS = 'population_batch_v1'


class PopulationExecution:
    mode = 'batch'

    def __init__(self, pathways):
        from ..components import get_component
        self.entries = {}
        for path in pathways:
            for stage in path.stages:
                for name in (stage.choose, stage.search.primary):
                    if name is not None:
                        self.entries[name] = get_component(name, batch=True)
            self.entries[path.update] = get_component(path.update, batch=True)
        from ..utils import solve
        from .native import load_native
        self.engine = load_native().Engine(solve)
        self.engine.configure_metrics(False)
        self.current = None

    def component(self, name):
        from ..components import get_component
        if name not in self.entries:
            self.entries[name] = get_component(name, batch=True)
        return self.entries[name]

    def generation(self, pathways, parameters, solutions, auxiliary, problem, data,
                   setting, generation, remaining, archive_names, archives, event_budget):
        from ..components import component_parent_arity
        from ..utils.design._helpers import ensure_rng, get_flex
        from ..utils.solve import (
            _resize_parent_indices,
            _same_optional_array,
        )

        rng = ensure_rng(setting)
        parent = PopulationBatch.from_solutions(solutions) if self.current is None else self.current
        target = int(get_flex(setting, '_AlgorithmOffspringSize', len(parent)))
        if event_budget is not None:
            target = min(target, max(0, int(event_budget)))
        if target <= 0 or remaining <= 0:
            return parent.to_solutions(), auxiliary, 0, archives, [], 0
        cross_paths = [i for i, p in enumerate(pathways)
                       if any(s.search.primary.startswith('cross_') for s in p.stages)]
        plain_paths = [i for i in range(len(pathways)) if i not in cross_paths]
        if len(cross_paths) == len(plain_paths) == 1:
            route = np.where(rng.random(target) < float(get_flex(setting, '_AlgorithmCrossoverRate', 1.0)),
                             cross_paths[0], plain_paths[0])
        else:
            route = rng.integers(0, len(pathways), size=target)
        owners = list(dict.fromkeys(int(i) for i in route))
        outputs = []
        observed = []
        used = 0
        interrupted = False
        for owner in owners:
            path, parameter = pathways[owner], parameters[owner]
            count = int(np.count_nonzero(route == owner))
            path_state = dict(auxiliary[owner]) if isinstance(auxiliary[owner], dict) else {}
            states = path_state.setdefault('_stream_stage_aux', [])
            while len(states) < len(path.stages):
                states.append({})
            pool = parent
            for stage_index, (stage, stage_parameter) in enumerate(zip(path.stages, parameter.stages)):
                name = stage.search.primary
                state = states[stage_index]
                state['rng'] = rng
                state['_stream_generation_target'] = int(get_flex(setting, '_AlgorithmOffspringSize', len(parent)))
                required = 2 * ((count + 1) // 2) if component_parent_arity(name) == 2 else count
                if stage.choose is None:
                    indices = np.arange(len(pool), dtype=np.int64)
                else:
                    chosen = self.component(stage.choose)(pool, problem, stage_parameter.choose,
                                                          state, generation=generation + stage_index,
                                                          data=data, count=required)
                    state = chosen.state
                    indices = chosen.indices
                indices = _resize_parent_indices(indices, population_size=len(pool),
                                                  offspring_size=required, rng=rng)
                selected = pool if len(indices) == len(pool) and np.array_equal(indices, np.arange(len(pool))) else pool.take(indices)
                mask = np.ones(len(selected), dtype=bool)
                if stage_index:
                    mask = rng.random(len(selected)) < float(get_flex(setting, '_AlgorithmMutationRate', 1.0))
                active = np.flatnonzero(mask)
                decisions = selected.X
                force = np.zeros(len(selected), dtype=bool)
                if len(active):
                    target_rows = indices[active] if pool is parent else None
                    varied = self.component(name)(selected if len(active) == len(selected) else selected.take(active), problem,
                                                   stage_parameter.search.primary, state,
                                                   generation=generation + stage_index, data=data,
                                                   context=parent, targets=target_rows)
                    state = varied.state
                    if varied.population.X.shape != decisions[active].shape:
                        raise ValueError(f'{name} returned incompatible batch dimensions')
                    if len(active) == len(selected):
                        decisions = varied.population.X
                    else:
                        decisions = decisions.astype(np.result_type(decisions.dtype, varied.population.X.dtype), copy=True)
                        decisions[active] = varied.population.X
                    force[active] = True if varied.invalidated is None else np.asarray(varied.invalidated, dtype=bool)
                if stage_index == len(path.stages) - 1:
                    decisions, force = decisions[:count], force[:count]
                # Evaluation is a real boundary. Its scalar callback policy,
                # repair, clones, owning inputs and FE cutoff remain centralized.
                evaluated, consumed, newly_evaluated = self.engine.changed(
                    decisions, selected.to_solutions(), problem, data, remaining - used, force,
                    aligned_sources=True)
                used += consumed
                observed.extend(newly_evaluated)
                states[stage_index] = state
                if evaluated is None:
                    interrupted = True
                    break
                pool = PopulationBatch.from_solutions(evaluated, dimension=parent.X.shape[1], dtype=parent.X.dtype)
                if name == 'search_cma':
                    self.component('para_cma')(pool, problem, state=state)
                elif name == 'search_pso':
                    state['_batch_pso_targets'] = state.pop('_stream_pso_target_indices', indices.tolist())
                    self.component('para_pso')(pool, problem, state=state)
                    state.pop('_batch_pso_targets', None)
            auxiliary[owner] = path_state
            path_state['_stream_stage_aux'] = states
            if interrupted:
                break
            outputs.append((owner, pool))
        if interrupted:
            best_rows = [min([*observed, *list(solutions)], key=lambda s: s.fit)] if observed else []
            return parent.to_solutions(), auxiliary, used, archives, best_rows, target

        # Restore route order before applying terminal update groups. Each sink
        # receives one combined matrix; sequential replacement stays inside it.
        combined = PopulationBatch.concatenate([p for _, p in outputs])
        offsets, offset = {}, 0
        for owner, p in outputs:
            offsets[owner] = offset
            offset += len(p)
        order = []
        for owner in route:
            order.append(offsets[int(owner)])
            offsets[int(owner)] += 1
        children = combined.take(order)
        groups: list[list[Any]] = []
        for owner in owners:
            path, parameter = pathways[owner], parameters[owner]
            group = next((g for g in groups if g[0] == path.update
                          and _same_optional_array(g[1], parameter.update)
                          and (path.update != 'update_iterated_local_search'
                               or g[2] == path.stages[-1].search.primary)), None)
            rows = np.flatnonzero(route == owner).tolist()
            if group is None:
                groups.append([path.update, parameter.update, path.stages[-1].search.primary, owner, rows])
            else:
                group[4].extend(rows)
        current = parent
        for name, parameter, active_search, owner, rows in groups:
            state = auxiliary[owner]
            state['rng'] = rng
            state['_active_search_operator'] = active_search
            merged = PopulationBatch.concatenate([current, children.take(sorted(rows))])
            updated = self.component(name)(merged, problem, parameter, state,
                                            generation=generation, data=data)
            state.pop('_active_search_operator', None)
            current = updated.population
            if len(current) != len(parent):
                current = current.take(np.argsort(current.F)[:len(parent)])
        self.current = current
        for i, name in enumerate(archive_names):
            archive = archives[i]
            if name != 'archive_statistic' and archive is not None and not isinstance(archive, PopulationBatch):
                archive = PopulationBatch.from_solutions(archive, dimension=current.X.shape[1], dtype=current.X.dtype) if len(archive) else None
            result = self.component(name)(current, problem, state={'rng': rng}, generation=generation,
                                           data=data, archive=archive)
            archives[i] = result.statistics if name == 'archive_statistic' else result.population
        public = current.to_solutions()
        best = min([*observed, *list(public)], key=lambda s: s.fit)
        return public, auxiliary, used, archives, [best], target

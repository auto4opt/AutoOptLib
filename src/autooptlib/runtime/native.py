"""Per-run native execution of fixed Search stream graphs.

The Python reference remains available for custom components and repair contracts.
Plans are derived execution data and are never serialized into algorithms.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ..utils.solve import SolutionSet
from .regions import plain_path

_MODULE_NAME = "autooptlib.runtime._native._native_stream"
_native_api = None


def _remember_native(module):
    global _native_api
    if _native_api is None:
        _native_api = (module, module.gather_decisions, module.gather_scalar)
    return module


def load_native():
    if _MODULE_NAME in sys.modules:
        return _remember_native(sys.modules[_MODULE_NAME])
    directory = os.environ.get("AOL_NATIVE_STREAM_BUILD")
    if directory:
        candidates = list(Path(directory).glob("_native_stream*.so")) + list(
            Path(directory).glob("_native_stream*.pyd")
        )
        if len(candidates) != 1:
            raise ImportError(
                "AOL_NATIVE_STREAM_BUILD must contain one native stream extension."
            )
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, candidates[0])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[_MODULE_NAME] = module
        return _remember_native(module)
    try:
        return _remember_native(importlib.import_module(_MODULE_NAME))
    except ModuleNotFoundError as error:
        if error.name != _MODULE_NAME:
            raise
        raise ImportError(
            "StreamBackend requires the optional C++ extension. Reinstall with a C++17 "
            'compiler, or use StreamBackend="python".'
        ) from error


_native_loader = load_native
_native_loader_code = load_native.__code__


def canonical_native_reader():
    # Metadata only: admission must not import/execute a replaced loader.
    return (
        load_native is _native_loader
        and load_native.__code__ is _native_loader_code
        and _native_api is not None
        and sys.modules.get(_MODULE_NAME) is _native_api[0]
        and vars(_native_api[0]).get("gather_decisions") is _native_api[1]
        and vars(_native_api[0]).get("gather_scalar") is _native_api[2]
    )


class NativeSolutionSet(SolutionSet):
    """Source objects stay authoritative; native aggregation returns owning copies."""

    def decs(self):
        result = load_native().gather_decisions(self._items)
        return super().decs() if result is None else result

    def fits(self):
        result = load_native().gather_scalar(self._items, "fit")
        return super().fits() if result is None else result

    def objs(self):
        result = load_native().gather_scalar(self._items, "obj")
        return super().objs() if result is None else result

    def cons(self):
        result = load_native().gather_scalar(self._items, "con")
        return super().cons() if result is None else result

    def __reduce__(self):
        # Checkpoints and deep copies restore the ordinary, backend-neutral type.
        return SolutionSet, (self._items,)


_native_decision_reader = NativeSolutionSet.decs
_native_decision_code = NativeSolutionSet.decs.__code__
_native_fitness_reader = NativeSolutionSet.fits
_native_fitness_code = NativeSolutionSet.fits.__code__


class NativeExecution:
    def __init__(self, mode, solve):
        self.mode = mode
        self.engine = load_native().Engine(solve)
        self.engine.configure_metrics(False)
        from .. import components

        if mode == "graph" and components._native_execution.get():
            self.engine.bind_components(components)
        # The measured input-prefix/broadcast transforms have no established
        # whole-Search benefit. Keep them behind the research adapter until
        # their preparation and guard costs are justified.
        self.engine.configure_dataflow(False)
        self.plans = {}
        self.pathways = None
        self.update_plan = None
        self.update_sources = None
        self.native_updates = mode == "graph"
        if mode == "graph":
            self.engine.set_population_type(NativeSolutionSet)

    def population(self, solutions):
        if self.mode == "graph" and not isinstance(solutions, NativeSolutionSet):
            return NativeSolutionSet(solutions)
        return solutions

    def plan(self, path, parameters, population, setting):
        if self.mode != "graph" or not plain_path(path, parameters):
            return None
        key = (
            next((i for i, item in enumerate(self.pathways) if item is path), None)
            if self.pathways is not None else 0
        )
        if key is None:
            return None
        if key not in self.plans or not self.plans[key].matches(path, parameters):
            self.plans[key] = self.engine.compile(path, parameters, setting, len(population), key)
        return self.plans[key]

    def event(self, *args):
        return self.engine.event(*args, self.plan(args[0], args[1], args[2], args[6]))

    def collect(self, pathways, parameters, population, auxiliary, problem, data,
                setting, generation, remaining, event_budget):
        batch = self.matrix_collect(pathways, parameters, population, auxiliary,
                                    problem, setting, remaining, event_budget)
        if batch is not None:
            return batch
        return self.engine.collect(
            pathways, parameters, population, auxiliary, problem, data, setting,
            generation, remaining, event_budget, self.plan,
        )

    def matrix_collect(self, pathways, parameters, population, auxiliary,
                       problem, setting, remaining, event_budget):
        """One matrix for independent fixed-bit events before deferred evaluation.

        No arbitrary event regrouping: routing/selection/mutation draws retain
        event order. Unknown layouts, state or callbacks use ordered collection.
        """
        if len(pathways) != 1:
            return None
        path, parameter = pathways[0], parameters[0]
        if not plain_path(path, parameter) or len(path.stages) != 1:
            return None
        stage = path.stages[0]
        name = stage.search.primary
        strengths = {"search_fastga_bit_one": 1, "search_fastga_bit_three": 3,
                     "search_fastga_bit_five": 5}
        if stage.choose != "choose_random" or name not in strengths:
            return None

        from .. import components
        from ..utils import solve
        from ..utils.solve import Solution, _PendingStreamOffspring
        from .component_library import load_component_library

        if (not components._native_execution.get() or solve.get_component is not components.get_component
                or type(population) is not NativeSolutionSet
                or type(population._items) is not list
                or type(problem) is not SimpleNamespace or type(setting) is not SimpleNamespace
                or type(setting.rng) is not np.random.Generator or remaining <= 0
                or name in components._custom_components
                or stage.choose in components._custom_components
                or parameter.stages[0].choose is not None
                or parameter.stages[0].search.primary is not None):
            return None
        target = getattr(setting, "_AlgorithmOffspringSize", len(population))
        if type(target) is not int or target <= 0 or type(getattr(problem, "Gmax", 0)) is not int:
            return None
        if any(type(getattr(setting, key, 1.0)) not in (int, float)
               for key in ("_AlgorithmCrossoverRate", "_AlgorithmMutationRate")):
            return None
        count = target if event_budget is None else min(target, max(0, int(event_budget)))
        if not count or type(getattr(problem, "bound", None)) is not np.ndarray:
            return None
        old = auxiliary[0]
        if type(old) is not dict:
            return None
        raw_states = old.get("_stream_stage_aux", [])
        if type(raw_states) is not list or any(type(state) is not dict for state in raw_states):
            return None
        state = raw_states[0] if raw_states else {}
        if state.get("rng", setting.rng) is not setting.rng:
            return None
        # Bind at the same run-local plan and registration boundary as ordered events.
        self.plan(path, parameter, population, setting)
        result = load_component_library().fixed_bit_events(
            population._items, problem.bound, setting.rng, Solution, count, strengths[name],
        )
        if result is None:
            return None
        decisions, parents = result
        path_state = dict(old)
        path_state.setdefault("rng", setting.rng)
        states = [dict(value) for value in raw_states] or [{}]
        for value in states:
            value.setdefault("rng", setting.rng)
        states[0]["_stream_generation_target"] = target
        for key in ("_selection_count", "_population_context", "_target_population_indices",
                    "_stream_invalidated_mask"):
            states[0].pop(key, None)
        path_state["_stream_stage_aux"] = states
        auxiliary[0] = path_state
        offspring = [
            (0, _PendingStreamOffspring(decisions[i:i + 1],
                                       [population[int(parent)]], np.array([True])))
            for i, parent in enumerate(parents)
        ]
        return offspring, [], 0, count, False

    def finish_updates(
        self,
        pathways,
        parameters,
        offspring,
        solutions,
        auxiliary,
        problem,
        data,
        rng,
        generation,
    ):
        if (
            self.update_sources is None
            or self.update_sources[0] is not pathways
            or self.update_sources[1] is not parameters
        ):
            self.update_plan = self.engine.compile_updates(pathways, parameters)
            self.update_sources = (pathways, parameters)
        return self.engine.finish_updates(
            self.update_plan,
            offspring,
            solutions,
            auxiliary,
            problem,
            data,
            rng,
            generation,
        )


def create_execution(mode, pathways, setting):
    if mode not in {"python", "direct", "graph", "batch"}:
        raise ValueError("StreamBackend must be python, direct, graph, or batch.")
    if mode == "python":
        return None
    if mode == "batch":
        if str(getattr(setting, "ComponentBackend", "python")).lower() != "native":
            raise ValueError("Population batch execution requires ComponentBackend='native'.")
        if getattr(setting, "_NativeBatchReference", False):
            from .population_graph import PopulationExecution
            return PopulationExecution(pathways)
        from .native_batch import NativePopulationExecution
        return NativePopulationExecution(pathways)
    if any(not plain_path(path) for path in pathways):
        return None
    from ..components import _COMPONENT_MODULES, _custom_components

    names = [
        name
        for path in pathways
        for stage in path.stages
        for name in (stage.choose, stage.search.primary)
        if name is not None
    ]
    names.extend(path.update for path in pathways)
    names.extend(name for path in pathways for name in path.archive)
    if not isinstance(setting.rng, np.random.Generator) or any(
        name not in _COMPONENT_MODULES or name in _custom_components for name in names
    ):
        return None
    solve = importlib.import_module("autooptlib.utils.solve")
    execution = NativeExecution(mode, solve)
    execution.pathways = pathways
    return execution

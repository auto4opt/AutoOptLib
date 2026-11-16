"""Public graph programming model over the existing Search and solve contracts.

These immutable declarations lower to Design/stream_graph_v2. They do not own
populations, random generators, execution state, or a second interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any, Mapping

import numpy as np

from .components import component_category, component_parent_arity
from .utils.design import Design


def _names(values, category):
    if isinstance(values, str):
        raise TypeError(f"{category} components must be a sequence, not a string")
    result = tuple(values)
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{category} components must be non-empty and unique")
    for name in result:
        _require_category(name, category)
    return result


def _require_category(name, category):
    if not isinstance(name, str) or component_category(name) != category:
        raise ValueError(f"Expected a registered {category} component, got {name!r}")


def _positive(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class Operator:
    """A registered component call and its optional numeric parameter vector."""

    name: str
    parameters: tuple[float, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or component_category(self.name) is None:
            raise ValueError(f"Unknown component {self.name!r}")
        if self.parameters is not None:
            values = np.asarray(self.parameters, dtype=float)
            if values.ndim != 1 or not np.all(np.isfinite(values)):
                raise ValueError("Operator parameters must be a finite numeric vector")
            object.__setattr__(self, "parameters", tuple(float(x) for x in values))
        from .components.contracts import get_component_contract

        get_component_contract(self.name).validate_parameters(self.parameters)


def _operator(value, category):
    result = Operator(value) if isinstance(value, str) else value
    if not isinstance(result, Operator):
        raise TypeError(f"Expected an Operator or component name for {category}")
    _require_category(result.name, category)
    return result


@dataclass(frozen=True)
class Stage:
    """Optional selection followed by one search/crossover component."""

    search: Operator | str
    choose: Operator | str | None = None
    termination: tuple[float, ...] = (1.0, 1.0)

    def __post_init__(self):
        object.__setattr__(self, "search", _operator(self.search, "search"))
        if self.choose is not None:
            object.__setattr__(self, "choose", _operator(self.choose, "choose"))
        termination = np.asarray(self.termination, dtype=float)
        if (
            termination.ndim != 1
            or not termination.size
            or np.any(np.isnan(termination))
        ):
            raise ValueError(
                "Stage termination metadata must be a non-empty numeric vector"
            )
        object.__setattr__(self, "termination", tuple(float(x) for x in termination))


@dataclass(frozen=True)
class Pathway:
    """An ordered stage sequence and its terminal replacement component."""

    stages: tuple[Stage, ...]
    update: Operator | str = "update_greedy"

    def __post_init__(self):
        stages = tuple(self.stages)
        if not 1 <= len(stages) <= 2 or not all(isinstance(x, Stage) for x in stages):
            raise ValueError("A pathway requires one or two Stage objects")
        object.__setattr__(self, "stages", stages)
        object.__setattr__(self, "update", _operator(self.update, "update"))


@dataclass(frozen=True)
class AlgorithmGraph:
    """Concrete graph, using the existing bounded Search v6 grammar.

    A graph has one or two pathways. A two-stage pathway is crossover followed
    by mutation. Archive policy is shared. Runtime state is created per solve.
    """

    pathways: tuple[Pathway, ...]
    population_size: int = 50
    offspring_size: int | None = None
    crossover_rate: float = 1.0
    mutation_rate: float = 1.0
    boundary_handling: str = "clip"
    archive: tuple[str, ...] = ()
    execution_semantics: str | None = None

    def __post_init__(self):
        pathways = tuple(self.pathways)
        if not 1 <= len(pathways) <= 2 or not all(
            isinstance(x, Pathway) for x in pathways
        ):
            raise ValueError("An AlgorithmGraph requires one or two Pathway objects")
        object.__setattr__(self, "pathways", pathways)
        archive = tuple(self.archive)
        for name in archive:
            _require_category(name, "archive")
        object.__setattr__(self, "archive", archive)
        if self.execution_semantics is not None:
            from .model_semantics import ExecutionSpec

            ExecutionSpec(self.execution_semantics)
        population = _positive(self.population_size, "population_size")
        offspring = (
            population
            if self.offspring_size is None
            else _positive(self.offspring_size, "offspring_size")
        )
        object.__setattr__(self, "population_size", population)
        object.__setattr__(self, "offspring_size", offspring)
        # Reuse the same grammar/configuration validation as JSON and execution.
        design = self.to_design()
        for name, value in design.configuration.items():
            object.__setattr__(self, name, value)

    def to_design(self) -> Design:
        """Lower to the existing executable Design, with fresh mutable state."""
        from .serialization import _validate_stream_phenotype
        from .utils.design import Design
        from .utils.design._helpers import SearchParam, SearchStep
        from .utils.design._population import configuration_is_valid, set_configuration
        from .utils.design._stream_graph import (
            StreamPathway,
            StreamPathwayParam,
            StreamStage,
            StreamStageParam,
        )

        def parameters(op):
            return (
                None if op is None or op.parameters is None else np.array(op.parameters)
            )

        paths, params = [], []
        for path in self.pathways:
            paths.append(
                StreamPathway(
                    [
                        StreamStage(
                            stage.choose.name if stage.choose else None,
                            SearchStep(
                                stage.search.name, np.array(stage.termination), None
                            ),
                        )
                        for stage in path.stages
                    ],
                    path.update.name,
                    list(self.archive),
                )
            )
            params.append(
                StreamPathwayParam(
                    [
                        StreamStageParam(
                            parameters(stage.choose),
                            SearchParam(parameters(stage.search), None),
                        )
                        for stage in path.stages
                    ],
                    parameters(path.update),
                )
            )
        _validate_stream_phenotype(paths, params)
        result = Design()
        result.construct([paths], [params])
        set_configuration(
            result,
            {
                "population_size": self.population_size,
                "offspring_size": self.offspring_size,
                "crossover_rate": self.crossover_rate,
                "mutation_rate": self.mutation_rate,
                "boundary_handling": self.boundary_handling,
            },
        )
        if not configuration_is_valid(result, result.configuration):
            raise ValueError("Graph sizes violate PSO or SSGA population constraints")
        from .model_semantics import MODEL_VERSION, ExecutionSpec

        result.program_contract = (
            {"version": MODEL_VERSION, "execution": None}
            if self.execution_semantics is None
            else ExecutionSpec(self.execution_semantics).to_dict()
        )
        return result

    def validate(
        self,
        problem=None,
        *,
        backend="python",
        stream_backend=None,
        require_installed=False,
    ):
        """Validate graph, component parameters, problem and backend before evaluation."""
        from .model_semantics import validate_design

        return validate_design(
            self.to_design(),
            problem,
            backend=backend,
            stream_backend=stream_backend,
            require_installed=require_installed,
        )

    @classmethod
    def from_design(cls, design: Design) -> AlgorithmGraph:
        """Read a decoded Search graph; legacy pathway graphs remain Design objects."""
        from .serialization import algorithm_to_dict
        from .utils.design._stream_graph import STREAM_GRAPH_SEMANTICS

        document = algorithm_to_dict(design)
        if document.get("execution_semantics") != STREAM_GRAPH_SEMANTICS:
            raise ValueError("AlgorithmGraph requires a stream_graph_v2 algorithm")
        configuration = document.get("configuration")
        if configuration is None:
            raise ValueError(
                "AlgorithmGraph requires explicit population configuration"
            )
        paths = []
        for path, params in zip(design.operator_pheno[0], design.parameter_pheno[0]):
            stages = tuple(
                Stage(
                    Operator(stage.search.primary, p.search.primary),
                    Operator(stage.choose, p.choose) if stage.choose else None,
                    tuple(stage.search.termination),
                )
                for stage, p in zip(path.stages, params.stages)
            )
            paths.append(Pathway(stages, Operator(path.update, params.update)))
        return cls(
            tuple(paths),
            archive=tuple(design.operator_pheno[0][0].archive),
            execution_semantics=getattr(design, "program_contract", {}).get(
                "execution"
            ),
            **configuration,
        )

    def to_initial_design(self, *, parameter_scope="node") -> dict[str, Any]:
        """Encode a Search warm start using its existing named-genotype protocol.

        Search shares parameters by component name. A concrete graph may have
        different values per occurrence; such a graph cannot be a lossless warm
        start and is rejected rather than silently changing parameters.
        """
        parameters = {}
        node_parameters = []
        seen = {}
        paths = []
        for path in self.pathways:
            stages = []
            for si, stage in enumerate(path.stages):
                stages.append(
                    {
                        "choose": stage.choose.name if stage.choose else None,
                        "search": stage.search.name,
                    }
                )
            paths.append({"stages": stages, "update": path.update.name})
            pi = len(paths) - 1
            calls = [(-1, "update", path.update)]
            for si, stage in enumerate(path.stages):
                calls.append((si, "search", stage.search))
                if stage.choose:
                    calls.append((si, "choose", stage.choose))
            for si, role, call in calls:
                node_parameters.append(
                    {
                        "pathway": pi,
                        "stage": si,
                        "role": role,
                        "component": call.name,
                        "values": None
                        if call.parameters is None
                        else list(call.parameters),
                    }
                )
                if call.name in seen and seen[call.name] != call.parameters:
                    if parameter_scope == "component":
                        raise ValueError(
                            f"Search requires shared parameters for {call.name}"
                        )
                seen[call.name] = call.parameters
                if call.parameters:
                    parameters[call.name] = list(call.parameters)
        return {
            "pathways": paths,
            "parameters": parameters,
            "node_parameters": node_parameters if parameter_scope == "node" else [],
            "parameter_scope": parameter_scope,
            "metadata": {},
            "configuration": dict(self.to_design().configuration),
        }


@dataclass(frozen=True)
class NodeSpace:
    """Component choices and local parameter domains at a graph occurrence."""

    pathway: int
    stage: int
    role: str
    choices: tuple
    parameters: tuple = ()

    def __post_init__(self):
        if self.pathway not in (0, 1) or self.role not in {
            "choose",
            "search",
            "update",
        }:
            raise ValueError("Invalid node location")
        if self.stage not in ((-1,) if self.role == "update" else (0, 1)):
            raise ValueError("Updates use stage=-1; other roles use stage=0/1")
        choices = tuple(self.choices)
        if not choices or len(set(choices)) != len(choices):
            raise ValueError("Node choices must be nonempty and unique")
        for name in choices:
            if name is None and self.role == "choose":
                continue
            _require_category(name, self.role)
        object.__setattr__(self, "choices", choices)
        raw = (
            self.parameters.items()
            if isinstance(self.parameters, Mapping)
            else self.parameters
        )
        normalized = []
        for name, value in raw:
            if name not in choices:
                raise ValueError("Node parameter component is outside its choices")
            a = np.asarray(value, dtype=float)
            if (
                a.ndim != 2
                or a.shape[1] != 2
                or not np.all(np.isfinite(a))
                or np.any(a[:, 0] > a[:, 1])
            ):
                raise ValueError("Node parameter intervals must be finite (n,2)")
            normalized.append((name, tuple(tuple(float(v) for v in row) for row in a)))
        if len({name for name, _ in normalized}) != len(normalized):
            raise ValueError("Duplicate node parameter domains")
        object.__setattr__(self, "parameters", tuple(sorted(normalized)))

    @property
    def location(self):
        return self.pathway, self.stage, self.role


@dataclass(frozen=True)
class DesignSpace:
    """Restrictions on the existing graph grammar, not a separate search engine.

    None keeps the problem-compatible component catalogue. Explicit choices
    must all be compatible; they are never silently dropped. Parameters default
    to independent node occurrences; component scope preserves legacy sharing.
    Component and node intervals may fix a value [x,x].
    """

    choose: tuple[str, ...] | None = None
    search: tuple[str, ...] | None = None
    update: tuple[str, ...] | None = None
    parameters: tuple = ()
    max_pathways: int = 2
    max_stages: int = 2
    population_sizes: tuple[int, ...] | None = None
    offspring_sizes: tuple[int, ...] | None = None
    parameter_scope: str = "node"
    nodes: tuple[NodeSpace, ...] = ()

    def __post_init__(self):
        if self.parameter_scope not in {"node", "component"}:
            raise ValueError("parameter_scope must be node or component")
        nodes = tuple(self.nodes)
        if not all(isinstance(n, NodeSpace) for n in nodes) or len(
            {n.location for n in nodes}
        ) != len(nodes):
            raise ValueError("Node spaces require unique NodeSpace locations")
        if self.parameter_scope == "component" and any(n.parameters for n in nodes):
            raise ValueError("Local parameter domains require node scope")
        object.__setattr__(self, "nodes", nodes)
        for category in ("choose", "search", "update"):
            value = getattr(self, category)
            if value is not None:
                object.__setattr__(self, category, _names(value, category))
        for name in ("max_pathways", "max_stages"):
            value = _positive(getattr(self, name), name)
            if value > 2:
                raise ValueError(f"{name} cannot exceed the existing grammar limit 2")
            object.__setattr__(self, name, value)
        if any(
            n.pathway >= self.max_pathways or n.stage >= self.max_stages for n in nodes
        ):
            raise ValueError("Node location exceeds topology limits")
        for name in ("population_sizes", "offspring_sizes"):
            values = getattr(self, name)
            if values is not None:
                values = tuple(sorted({_positive(x, name) for x in values}))
                if not values:
                    raise ValueError(f"{name} cannot be empty")
                object.__setattr__(self, name, values)
        items = (
            self.parameters.items()
            if isinstance(self.parameters, Mapping)
            else self.parameters
        )
        parameters = []
        for name, bounds in items:
            if component_category(name) not in {"choose", "search", "update"}:
                raise ValueError(f"Unknown design-space component {name!r}")
            a = np.asarray(bounds, dtype=float)
            if (
                a.ndim != 2
                or a.shape[1] != 2
                or not np.all(np.isfinite(a))
                or np.any(a[:, 0] > a[:, 1])
            ):
                raise ValueError(
                    f"Parameter bounds for {name} must be finite (n, 2) intervals"
                )
            parameters.append((name, tuple(tuple(float(x) for x in row) for row in a)))
        if len({name for name, _ in parameters}) != len(parameters):
            raise ValueError("Duplicate component parameter domains")
        object.__setattr__(self, "parameters", tuple(sorted(parameters)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible, versioned design-space declaration."""
        return {
            "schema": "autooptlib.design-space",
            "schema_version": 2,
            "parameter_scope": self.parameter_scope,
            "nodes": [
                dict(
                    pathway=n.pathway,
                    stage=n.stage,
                    role=n.role,
                    choices=list(n.choices),
                    parameters={
                        name: [list(row) for row in bounds]
                        for name, bounds in n.parameters
                    },
                )
                for n in self.nodes
            ],
            "choose": None if self.choose is None else list(self.choose),
            "search": None if self.search is None else list(self.search),
            "update": None if self.update is None else list(self.update),
            "parameters": {
                name: [list(row) for row in bounds] for name, bounds in self.parameters
            },
            "max_pathways": self.max_pathways,
            "max_stages": self.max_stages,
            "population_sizes": None
            if self.population_sizes is None
            else list(self.population_sizes),
            "offspring_sizes": None
            if self.offspring_sizes is None
            else list(self.offspring_sizes),
        }

    @classmethod
    def from_dict(cls, document: Mapping[str, Any]) -> DesignSpace:
        data = dict(document)
        version = data.pop("schema_version", None)
        if data.pop("schema", None) != "autooptlib.design-space" or version not in (
            1,
            2,
        ):
            raise ValueError("Unsupported design-space schema")
        if version == 1:
            data.setdefault("parameter_scope", "component")
        data["nodes"] = tuple(NodeSpace(**n) for n in data.get("nodes", ()))
        return cls(**data)

    @classmethod
    def from_graph(cls, graph):
        """Freeze graph structure and explicit values as a node-scoped space."""
        nodes = []
        names = {"choose": set(), "search": set(), "update": set()}
        for pi, path in enumerate(graph.pathways):
            calls = [(-1, "update", path.update)]
            for si, stage in enumerate(path.stages):
                calls.extend(
                    [(si, "choose", stage.choose), (si, "search", stage.search)]
                )
            for si, role, op in calls:
                local = (
                    {}
                    if op is None or op.parameters is None
                    else {op.name: [(v, v) for v in op.parameters]}
                )
                nodes.append(
                    NodeSpace(pi, si, role, (None if op is None else op.name,), local)
                )
                if op is not None:
                    names[role].add(op.name)
        return cls(
            choose=tuple(sorted(names["choose"])) or None,
            search=tuple(sorted(names["search"])),
            update=tuple(sorted(names["update"])),
            nodes=tuple(nodes),
            max_pathways=len(graph.pathways),
            max_stages=max(len(p.stages) for p in graph.pathways),
            population_sizes=(graph.population_size,),
            offspring_sizes=(graph.offspring_size,),
        )

    def _restrict_components(self, choose, search, update):
        result = []
        for category, available in zip(
            ("choose", "search", "update"), (choose, search, update)
        ):
            requested = getattr(self, category)
            if requested is None:
                result.append(available)
                continue
            missing = set(requested) - set(available)
            if missing:
                raise ValueError(
                    f"Incompatible {category} components for this problem: {sorted(missing)}"
                )
            result.append(list(requested))
        if self.max_stages == 2 and any(x.startswith("cross_") for x in result[1]):
            from .utils.design._population import is_mutation_component

            if not any(is_mutation_component(x) for x in result[1]):
                raise ValueError(
                    "Two-stage crossover spaces require a mutation component; use max_stages=1"
                )
        return result

    def resolve(self, problem: Any) -> list[dict[str, Any]]:
        """Inspect admissible components and domains for constructed problem(s).

        This calls component metadata modes, not objectives or the designer.
        Problem records must have the same type/bound fields as construct mode.
        """
        from types import SimpleNamespace

        from .utils.space import space

        setting = SimpleNamespace(
            GraphSemantics="stream_graph_v2",
            DesignSpace=self,
            AlgP=self.max_pathways,
            AlgQ=self.max_stages,
        )
        if self.population_sizes is not None:
            setting.PopulationSizeSpace = self.population_sizes
        if self.offspring_sizes is not None:
            setting.OffspringSizeSpace = self.offspring_sizes
        prepared = space(problem, setting)
        if self.nodes:
            from .utils.design._parameter_bank import (
                bind_bank,
                sample_constrained_paths,
            )

            setting.rng = np.random.default_rng(0)
            setting.ParameterScope = self.parameter_scope
            paths = sample_constrained_paths(
                setting, setting.rng, self.max_pathways, self.max_stages
            )
            bind_bank([[None, None] for _ in prepared.AllOp], paths, setting)
        return [
            dict(
                describe_component(name),
                parameters=None if bounds is None else bounds.tolist(),
                parameter_types=list(kinds),
            )
            for name, bounds, kinds in zip(
                prepared.AllOp, prepared.ParaSpace, prepared.ParaTypeSpace
            )
        ]

    def _restrict_parameters(self, names, bounds, kinds):
        for name, intervals in self.parameters:
            if name not in names:
                raise ValueError(
                    f"Parameter domain refers to inactive component {name!r}"
                )
            index = names.index(name)
            original = bounds[index]
            proposed = np.asarray(intervals, dtype=float)
            if (
                original is None
                or proposed.shape != original.shape
                or np.any(proposed[:, 0] < original[:, 0])
                or np.any(proposed[:, 1] > original[:, 1])
            ):
                raise ValueError(
                    f"Parameter domain for {name} must narrow its declared bounds"
                )
            for row, kind in zip(proposed, kinds[index]):
                if kind == "integer" and np.ceil(row[0]) > np.floor(row[1]):
                    raise ValueError(f"Parameter domain for {name} contains no integer")
            bounds[index] = proposed.copy()


def _apply_saved_execution_defaults(design, options):
    from .model_semantics import BATCH_SEMANTICS, GraphValidationError, execution_spec

    contract = getattr(design, "program_contract", None)
    if not contract or contract.get("execution") is None:
        return
    options["ModelContractVersion"] = 1
    expected = contract["execution"]
    if expected == BATCH_SEMANTICS:
        for name, value in (("ComponentBackend", "native"), ("StreamBackend", "batch")):
            if name in options and options[name] != value:
                raise GraphValidationError(
                    "execution", f"requires {expected}: {name}={value}"
                )
            options.setdefault(name, value)
    actual = execution_spec(
        options.get("ComponentBackend", "python"), options.get("StreamBackend")
    )
    if actual.execution != expected:
        raise GraphValidationError(
            "execution", f"requires {expected}, got {actual.execution}"
        )


def _configure_model_options(options):
    """Expand declarations before the existing option validation pipeline."""
    from .utils.design import Design

    graph = options.get("Algorithm")
    domain = options.get("DesignSpace")
    mode = options.get("Mode")
    if (
        mode == "solve"
        and graph is None
        and options.get("AlgFile")
        and str(options["AlgFile"]).lower().endswith(".json")
    ):
        # Population/backend defaults are needed before generic option checks,
        # not only when process() later decodes the executable algorithm.
        from .serialization import load_algorithm

        saved = load_algorithm(options["AlgFile"])
        if saved.configuration is not None:
            options.setdefault("ProbN", saved.population_size)
        _apply_saved_execution_defaults(saved, options)
    if graph is not None or domain is not None:
        options["ModelContractVersion"] = 1
    if options.get("SearchInitialDesigns"):
        initial = []
        archives = None
        for item in options["SearchInitialDesigns"]:
            if isinstance(item, AlgorithmGraph):
                _apply_saved_execution_defaults(item.to_design(), options)
                if archives is not None and archives != item.archive:
                    raise ValueError("Warm-start graphs must share the Archive policy")
                archives = item.archive
                item = item.to_initial_design(
                    parameter_scope=domain.parameter_scope
                    if domain is not None
                    else "node"
                )
            initial.append(item)
        options["SearchInitialDesigns"] = initial
        if any(
            i.get("parameter_scope") == "node" for i in initial if isinstance(i, dict)
        ):
            options["ParameterScope"] = "node"
        if archives is not None:
            if "Archive" in options and tuple(options["Archive"]) != archives:
                raise ValueError("Archive conflicts with warm-start graphs")
            options["Archive"] = list(archives)
    if graph is not None:
        if mode != "solve":
            raise ValueError("Algorithm is only accepted in solve mode")
        if options.get("AlgFile") or options.get("AlgName"):
            raise ValueError("Use exactly one of Algorithm, AlgFile, or AlgName")
        if isinstance(graph, AlgorithmGraph):
            graph = graph.to_design()
        if not isinstance(graph, Design):
            raise TypeError("Algorithm must be an AlgorithmGraph or decoded Design")
        from .serialization import algorithm_from_dict, algorithm_to_dict

        # Take an independent validated snapshot; do not mutate the user's Design.
        graph = algorithm_from_dict(algorithm_to_dict(graph))
        options["Algorithm"] = graph
        _apply_saved_execution_defaults(graph, options)
        if graph.configuration is not None:
            options.setdefault("ProbN", graph.population_size)
    if domain is not None:
        if mode != "design" or not isinstance(domain, DesignSpace):
            raise ValueError(
                "DesignSpace requires design mode and a DesignSpace object"
            )
        expanded = {
            "GraphSemantics": "stream_graph_v2",
            "AlgP": domain.max_pathways,
            "AlgQ": domain.max_stages,
            "ParameterScope": domain.parameter_scope,
        }
        for field, key in (
            ("population_sizes", "PopulationSizeSpace"),
            ("offspring_sizes", "OffspringSizeSpace"),
        ):
            if getattr(domain, field) is not None:
                expanded[key] = list(getattr(domain, field))
        for key, value in expanded.items():
            if key in options:
                supplied = options[key]
                if isinstance(value, list) and not isinstance(supplied, (str, bytes)):
                    supplied = list(supplied)
                if supplied != value:
                    raise ValueError(f"{key} conflicts with DesignSpace")
            options[key] = value


def describe_component(name: str) -> dict[str, Any]:
    """Inspect registry facts; execution/parameter modes remain on get_component."""
    from .components import _COMPONENT_MODULES, _custom_components

    if component_category(name) is None:
        raise ValueError(f"Unknown component {name!r}")
    native = name in _COMPONENT_MODULES and name not in _custom_components
    from .components.contracts import get_component_contract

    return {
        "name": name,
        "category": component_category(name),
        "parent_arity": component_parent_arity(name),
        "native": native,
        "batch": native,
        "contract": get_component_contract(name).to_dict(),
    }


__all__ = [
    "AlgorithmGraph",
    "DesignSpace",
    "Operator",
    "Pathway",
    "Stage",
    "describe_component",
]

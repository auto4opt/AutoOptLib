"""Component contracts shared by declarations, validation and extensions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ParameterSpec:
    name: str
    lower: float = -float("inf")
    upper: float = float("inf")
    kind: str = "continuous"
    default: float | None = None

    def __post_init__(self):
        if not self.name or self.kind not in {"continuous", "integer"}:
            raise ValueError("Parameters need a name and continuous/integer kind")
        if np.isnan(self.lower) or np.isnan(self.upper) or self.lower > self.upper:
            raise ValueError("Invalid execution parameter interval")
        if self.default is not None:
            self.validate(self.default)

    def validate(self, value):
        if not np.isfinite(value) or not self.lower <= value <= self.upper:
            raise ValueError(
                f"parameter {self.name} must be in [{self.lower}, {self.upper}]"
            )
        if self.kind == "integer" and float(value) != int(value):
            raise ValueError(f"parameter {self.name} must be an integer")


@dataclass(frozen=True)
class ComponentContract:
    name: str
    category: str
    parameters: tuple[ParameterSpec, ...] = ()
    problem_types: tuple[str, ...] = ("continuous", "discrete", "permutation")
    parent_arity: int = 1
    input_kind: str = "population"
    output_kind: str = "auto"
    needs_fitness: bool = False
    state_fields: tuple[str, ...] = ()
    effects: tuple[str, ...] = ("rng",)
    backends: tuple[str, ...] = ("python",)
    binary: bool = False
    min_dimension: int = 1
    dimension_parameters: bool = False
    version: int = 1
    require_parameters: bool = False

    def __post_init__(self):
        for field in (
            "parameters",
            "problem_types",
            "state_fields",
            "effects",
            "backends",
        ):
            object.__setattr__(self, field, tuple(getattr(self, field)))
        if self.category not in {"choose", "search", "update", "archive", "parameter"}:
            raise ValueError("Invalid component category")
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("Component names must be identifiers")
        if (
            self.version != 1
            or not isinstance(self.min_dimension, int)
            or self.min_dimension < 1
        ):
            raise ValueError("Invalid contract version or minimum dimension")
        if self.dimension_parameters and len(self.parameters) != 1:
            raise ValueError(
                "Dimension parameters require one broadcast parameter spec"
            )
        expected_output = {
            "choose": "indices",
            "search": "decisions",
            "update": "survivors",
            "archive": "archive",
            "parameter": "state",
        }[self.category]
        if self.output_kind == "auto":
            object.__setattr__(self, "output_kind", expected_output)
        elif self.output_kind != expected_output:
            raise ValueError("Output kind conflicts with component execution category")
        if self.input_kind not in {"population", "decisions"}:
            raise ValueError("Unsupported component input kind")
        if any(
            not isinstance(s, str) or not s.isidentifier() for s in self.state_fields
        ):
            raise ValueError("State fields must be identifiers")
        if self.parent_arity not in (1, 2) or not self.problem_types:
            raise ValueError("Invalid component arity or problem types")
        if not set(self.problem_types) <= {"continuous", "discrete", "permutation"}:
            raise ValueError("Unknown component problem domain")
        if not set(self.backends) <= {"python", "native"} or not self.backends:
            raise ValueError("Unknown component backend")
        if len({p.name for p in self.parameters}) != len(self.parameters):
            raise ValueError("Duplicate parameter names")

    def validate_parameters(self, values, dimension=None):
        if values is None:
            if self.require_parameters:
                raise ValueError(f"{self.name} requires explicit parameters")
            return
        a = np.asarray(values, dtype=float)
        if a.ndim != 1 or not np.all(np.isfinite(a)):
            raise ValueError("parameters must be a finite vector")
        expected = len(self.parameters)
        if self.dimension_parameters and (
            a.size == 1 or (dimension is None and a.size > 0) or a.size == dimension
        ):
            for value in a:
                self.parameters[0].validate(float(value))
            return
        if a.size != expected:
            raise ValueError(f"{self.name} expects {expected} parameters, got {a.size}")
        for parameter, value in zip(self.parameters, a):
            parameter.validate(float(value))

    def to_dict(self):
        result = asdict(self)
        for p in result["parameters"]:
            for key in ("lower", "upper"):
                if not np.isfinite(p[key]):
                    p[key] = None
        return result

    @classmethod
    def from_dict(cls, record):
        """Restore the strict JSON form; null interval edges mean unbounded."""
        data = dict(record)
        parameters = []
        for parameter in data.get("parameters", ()):
            values = dict(parameter)
            for key, unbounded in (("lower", -float("inf")), ("upper", float("inf"))):
                if values.get(key) is None:
                    values[key] = unbounded
            parameters.append(ParameterSpec(**values))
        data["parameters"] = tuple(parameters)
        return cls(**data)

    def validate_search_bounds(self, bounds, dimension=None):
        """Validate reachable values, allowing fractional integer interval edges."""
        a = np.asarray(bounds, dtype=float)
        if (
            a.ndim != 2
            or a.shape[1] != 2
            or not np.all(np.isfinite(a))
            or np.any(a[:, 0] > a[:, 1])
        ):
            raise ValueError("Search intervals must be finite ordered (n, 2) bounds")
        values = a.copy()
        for i, p in enumerate(self.parameters):
            if i >= len(values):
                break
            if p.kind == "integer":
                values[i] = [np.ceil(a[i, 0]), np.floor(a[i, 1])]
                if values[i, 0] > values[i, 1]:
                    raise ValueError(
                        f"Search interval for {p.name} contains no integer"
                    )
        self.validate_parameters(values[:, 0], dimension)
        self.validate_parameters(values[:, 1], dimension)


_NAMES = {
    "choose_brainstorm": ("clusters", "selection_probability"),
    "choose_ica": ("imperialists",),
    "choose_elite_fraction": ("elite_fraction",),
    "cross_arithmetic": ("ratio",),
    "cross_point_uniform": ("swap_probability",),
    "cross_point_n": ("points",),
    "cross_order_n": ("selected_positions",),
    "cross_sim_binary": ("distribution_index",),
    "search_de_current": ("scale", "crossover_probability"),
    "search_de_current_best": ("scale", "crossover_probability"),
    "search_de_random": ("scale", "crossover_probability"),
    "search_de_current_to_pbest": ("scale", "crossover_probability", "elite_fraction"),
    "search_fastga_de_current_to_pbest": (
        "scale",
        "crossover_probability",
        "elite_fraction",
    ),
    "search_mu_uniform": ("mutation_probability",),
    "search_mu_polynomial": ("mutation_probability", "distribution_index"),
    "search_pso": ("inertia",),
    "search_ica": ("assimilation",),
    "search_reset_n": ("reset_count",),
    "search_reset_rand": ("reset_probability",),
    "search_reset_creep": ("mutation_probability", "step_ratio"),
    "update_simulated_annealing": ("temperature",),
}


def _parameter_metadata(name, problem):
    from ..utils.space import _parameter_types
    from . import get_component

    if name.startswith(("archive_", "para_")):
        return (), None
    problem = problem or SimpleNamespace(
        type=["continuous", "static", "certain"],
        N=8,
        Gmax=10,
        bound=np.array([[-5.0] * 8, [5.0] * 8]),
    )
    function = get_component(name, native=False)
    raw, _ = function(problem, "parameter")
    if raw is None:
        return (), None
    bounds = np.asarray(raw, dtype=float).reshape(-1, 2)
    return _parameter_types(function, problem, len(bounds)), bounds


def get_component_contract(name: str, problem: Any = None) -> ComponentContract:
    from . import _custom_component_specs, _custom_components, component_category

    category = component_category(name)
    if category is None:
        raise ValueError(f"Unknown component {name!r}")
    if name in _custom_components:
        declared = getattr(_custom_components[name], "contract", None)
        if isinstance(declared, ComponentContract):
            return declared
    kinds, bounds = _parameter_metadata(name, problem)
    specs = []
    for index, kind in enumerate(kinds):
        label = (
            _NAMES.get(name, ())[index]
            if index < len(_NAMES.get(name, ()))
            else f"parameter_{index}"
        )
        lo, hi = -float("inf"), float("inf")
        if "probability" in label or "fraction" in label or "ratio" in label:
            lo, hi = 0.0, 1.0
        if kind == "integer":
            lo = 1.0
        if label in {"temperature", "distribution_index"}:
            lo = np.nextafter(0.0, 1.0)
        specs.append(ParameterSpec(label, lo, hi, kind))
    types = ("continuous", "discrete", "permutation")
    binary = "_bit_" in name or name.startswith("cross_fastga_")
    if binary or name.startswith("search_reset_") or name == "reinit_discrete":
        types = ("discrete",)
    elif name.startswith("cross_order_") or name in {
        "search_swap",
        "search_swap_multi",
        "search_scramble",
        "search_insert",
        "reinit_permutation",
    }:
        types = ("permutation",)
    elif category == "search" and not name.startswith("cross_point_"):
        types = ("continuous",)
    elif name.startswith("cross_point_"):
        types = ("continuous", "discrete")
    if name in _custom_component_specs:
        types = tuple(sorted(_custom_component_specs[name][1]))
    elif name == "choose_brainstorm":
        types = ("continuous",)
    minimum = 1
    if (
        types == ("permutation",)
        and name not in {"reinit_permutation", "cross_order_n"}
        and name not in _custom_components
    ):
        minimum = 2
    if name == "cross_point_n":
        minimum = 2
    for word, count in (("two", 2), ("three", 4), ("five", 6)):
        if name == "cross_point_" + word:
            minimum = count
    if name.startswith("cross_fastga_") and name.endswith(("three", "five")):
        minimum = 2
    if name.endswith("bit_three"):
        minimum = 3
    if name.endswith("bit_five"):
        minimum = 5
    fields = ()
    if name in {"search_pso", "para_pso"}:
        fields = ("V", "Pbest", "Gbest")
    elif name in {"search_cma", "para_cma", "para_cmaes"}:
        fields = ("cma_C", "cma_mean", "cma_pc", "cma_ps", "cma_sigma")
    elif name == "search_mu_gaussian":
        fields = ("gaussian_eta",)
    elif name == "search_mu_cauchy":
        fields = ("cauchy_eta",)
    return ComponentContract(
        name,
        category,
        tuple(specs),
        types,
        parent_arity=2 if name.startswith("cross_") else 1,
        output_kind={
            "choose": "indices",
            "update": "survivors",
            "archive": "archive",
            "parameter": "state",
        }.get(category, "decisions"),
        needs_fitness=(
            category in {"choose", "update", "archive", "parameter"}
            and name not in {"choose_random", "choose_traverse"}
        )
        or name
        in {
            "search_cma",
            "search_ica",
            "search_de_current_best",
            "search_de_current_to_pbest",
            "search_fastga_de_current_to_pbest",
        },
        state_fields=fields,
        effects=("rng", "state") if fields else ("rng",),
        backends=("python",) if name in _custom_components else ("python", "native"),
        binary=binary,
        min_dimension=minimum,
        dimension_parameters=name == "cross_arithmetic",
    )


def parameter_domains(name, problem):
    """Search bounds remain separate from execution validity intervals."""
    contract = get_component_contract(name, problem)
    kinds, bounds = _parameter_metadata(name, problem)
    return contract, bounds, kinds

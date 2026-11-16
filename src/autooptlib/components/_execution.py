"""Audited execution effects of built-ins, shared by planning and native dispatch.

Unknown or overridden components have no contract. These declarations never
authorize reordering RNG, evaluation, exception or state-commit effects.
"""

from __future__ import annotations

import builtins
import sys
from dataclasses import dataclass, field
from functools import partial
from importlib import import_module
from types import CodeType, FunctionType, ModuleType

import numpy as np

from . import _COMPONENT_MODULES, _custom_components

# NumPy 1.x permits replacing ndarray arithmetic without changing np.multiply
# itself. A no-argument call only reads the operator table; NumPy 2 removed it.
_NUMERIC_OPERATOR_READER = vars(np).get("set_numeric_ops")
_NUMERIC_OPERATORS = _NUMERIC_OPERATOR_READER() if _NUMERIC_OPERATOR_READER else {}


@dataclass(frozen=True)
class ExecutionContract:
    kernel: str
    random_rule: str
    reads: tuple[str, ...]
    writes: tuple[str, ...]
    private_output: bool = True
    deferred_final: bool = False
    output_domain: str = "numeric"
    matrix_prefix: tuple[str, ...] = ()
    row_broadcast: bool = False
    computation: bool = False


@dataclass(frozen=True)
class Primitive:
    """A reference-ordered computation, with explicit register/effect edges.

    A primitive can contain a reference loop or conditional. Its internal RNG
    calls stay in that shared function; the planner never expands or reorders
    those calls. One output stores the return object; multiple outputs unpack it.
    """

    name: str
    function: object
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    effects: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    aliases: tuple[tuple[str, str], ...] = ()
    requirements: tuple = ()
    positional_tail: tuple = ()
    binding_point: int | None = None
    preparation: str = ""


class DependencyGuard:
    """Frozen code/global dependencies of an explicitly pure builtin template.

    Capture at template declaration, never after a callback may have replaced
    a helper. Attribute lookup here only uses ordinary module dictionaries.
    Numerical input/alias/environment admission is a separate runtime check.
    """

    def __init__(self, functions):
        self.bindings, self.functions, self.classes, self.cells = [], [], [], []
        self.builtin_lookups = []
        self.complete = True
        self.native_check = None
        seen = set()

        def dependency(value):
            if type(value) is FunctionType:
                visit(value)
            elif type(value) is type and value.__module__.startswith("autooptlib."):
                if id(value) in seen:
                    return
                seen.add(id(value))
                attrs = tuple(vars(value).items())
                self.classes.append((value, attrs))
                for _, method in attrs:
                    if type(method) is FunctionType:
                        visit(method)
            elif (
                type(value).__module__.startswith("numpy")
                and type(value).__name__ == "_ArrayFunctionDispatcher"
            ):
                # NumPy 2 wraps Python implementations in a C dispatcher.
                # Guard the implementation's helpers too (e.g. _wrapfunc),
                # rather than checking only the public np.argsort object.
                wrapped = value.__wrapped__
                self.bindings.append((vars(value), "__wrapped__", wrapped))
                visit(wrapped)

        def visit(function):
            if type(function) is not FunctionType or id(function) in seen:
                return
            seen.add(id(function))
            if len(seen) > 4096:
                self.complete = False
                return
            self.functions.append(
                (
                    function,
                    function.__code__,
                    function.__defaults__,
                    tuple((function.__kwdefaults__ or {}).items()),
                )
            )
            for cell in function.__closure__ or ():
                try:
                    value = cell.cell_contents
                except ValueError:
                    self.complete = False
                    continue
                self.cells.append((cell, value))
                dependency(value)
            names = set()

            def collect(code):
                names.update(code.co_names)
                for value in code.co_consts:
                    if type(value) is CodeType:
                        collect(value)

            collect(function.__code__)
            namespace = function.__globals__
            modules = []
            for name in names:
                if name in namespace:
                    value = namespace[name]
                    self.bindings.append((namespace, name, value))
                    dependency(value)
                    if type(value) is ModuleType:
                        modules.append(value)
                elif name in vars(builtins):
                    # Resolve the same global-then-builtin lookup as Python.
                    self.builtin_lookups.append((namespace, name, vars(builtins)[name]))
            for module in modules:
                for name in names:
                    if name in vars(module):
                        value = vars(module)[name]
                        self.bindings.append((vars(module), name, value))
                        dependency(value)

        for function in functions:
            visit(function)

    def trusted(self):
        if self.native_check is not None:
            return self.native_check.trusted()

        def same_cell(cell, value):
            try:
                return cell.cell_contents is value
            except ValueError:
                return False

        if vars(np).get("set_numeric_ops") is not _NUMERIC_OPERATOR_READER:
            return False
        if _NUMERIC_OPERATOR_READER is not None:
            current = _NUMERIC_OPERATOR_READER()
            if len(current) != len(_NUMERIC_OPERATORS) or any(
                current.get(k) is not v for k, v in _NUMERIC_OPERATORS.items()
            ):
                return False
        return (
            self.complete
            and all(
                namespace.get(name, vars(builtins).get(name)) is value
                for namespace, name, value in self.builtin_lookups
            )
            and all(same_cell(cell, value) for cell, value in self.cells)
            and all(
                namespace.get(name) is value for namespace, name, value in self.bindings
            )
            and all(
                function.__code__ is code
                and function.__defaults__ is defaults
                and len(function.__kwdefaults__ or {}) == len(keywords)
                and all(
                    (function.__kwdefaults__ or {}).get(k) is v for k, v in keywords
                )
                for function, code, defaults, keywords in self.functions
            )
            # Pickle can add unrelated __slotnames__ metadata. Only the
            # captured implementations participate in this dependency.
            and all(
                all(vars(cls).get(name) is value for name, value in attrs)
                for cls, attrs in self.classes
            )
        )


@dataclass(frozen=True)
class NativePattern:
    kernel: str
    operations: tuple[str, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True)
class Access:
    source: str
    field: str
    rows: object = None

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        aliases = {
            "rng": ("auxiliary", "rng"),
            "parameters": ("parameters", "value"),
            "fitness": ("population", "fitness"),
            "inner_g": ("arguments", "inner_g"),
            "generation": ("arguments", "generation"),
            "Gmax": ("problem", "Gmax"),
            "bounds": ("problem", "bound"),
            "target_indices": ("auxiliary", "_target_population_indices"),
            "pso_target_indices": ("auxiliary", "_stream_pso_target_indices"),
        }
        source, field = aliases.get(
            value, value.split(".", 1) if "." in value else ("auxiliary", value)
        )
        return cls(source, field)


@dataclass(frozen=True)
class Requirement:
    source: str
    field: str = "decisions"
    rows: object = "all"
    representation: str = "matrix"
    reader: str = "decision"

    def __post_init__(self):
        valid_rows = (
            type(self.rows) is str
            and self.rows in {"all", "selected"}
            or type(self.rows) is tuple
            and all(type(i) is int for i in self.rows)
        )
        if (
            not valid_rows
            or self.field != "decisions"
            or self.representation not in {"matrix", "indexed", "references"}
        ):
            raise ValueError("Unsupported representation/row requirement")


@dataclass(frozen=True)
class ResolvedCall:
    function: object
    expected: object
    expected_code: object
    prefix: tuple = ()
    keywords: object = None
    tail: tuple = ()

    def trusted(self):
        return (
            self.function is self.expected
            and getattr(self.function, "__code__", None) is self.expected_code
        )

    def __call__(self, *args):
        return self.function(*self.prefix, *args, *self.tail, **(self.keywords or {}))


class CallableBinding:
    """Resolve the SAME global name as the reference at the primitive call."""

    def __init__(self, function, namespace, positional_tail=()):
        self.original = function
        self.positional_tail = positional_tail
        base = function.func if isinstance(function, partial) else function
        self.namespace = namespace
        self.name = next(
            (
                name
                for name, value in namespace.items()
                if value is base and name == getattr(base, "__name__", None)
            ),
            None,
        )
        if self.name is None:
            self.name = next(
                (name for name, value in namespace.items() if value is base), None
            )
        self.base = base
        self.code = getattr(base, "__code__", None)

    def resolve(self):
        current = self.base
        if self.name is not None:
            if self.name in self.namespace:
                current = self.namespace[self.name]
            else:
                builtins = self.namespace.get("__builtins__", {})
                names = builtins if type(builtins) is dict else vars(builtins)
                if self.name not in names:
                    raise NameError(
                        f"name '{self.name}' is not defined", name=self.name
                    )
                current = names[self.name]
        bound = ResolvedCall(
            current,
            self.base,
            self.code,
            self.original.args if isinstance(self.original, partial) else (),
            self.original.keywords if isinstance(self.original, partial) else None,
            self.positional_tail,
        )
        return bound, bound.trusted()


@dataclass(frozen=True)
class Computation:
    reference: object
    operations: tuple[Primitive, ...]
    result: str = "result"
    reference_code: object = field(init=False, repr=False)
    dispatcher: object = None
    dispatcher_code: object = field(init=False, repr=False)
    bindings: tuple = field(init=False, repr=False)
    native_patterns: tuple[NativePattern, ...] = ()
    dependencies: object = field(init=False, repr=False)
    native_dependencies: object = field(default=None, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "reference_code", self.reference.__code__)
        object.__setattr__(
            self, "dispatcher_code", getattr(self.dispatcher, "__code__", None)
        )
        # Capture the global NAME and expected code when the template is
        # declared. Searching for the old object only at first execution would
        # miss a helper replaced before the first plan was compiled.
        namespace = (
            self.dispatcher.__globals__
            if self.dispatcher is not None
            else self.reference.__globals__
        )
        object.__setattr__(
            self,
            "bindings",
            tuple(
                CallableBinding(op.function, namespace, op.positional_tail)
                for op in self.operations
            ),
        )
        object.__setattr__(
            self,
            "dependencies",
            DependencyGuard(
                tuple(
                    op.function.func
                    if isinstance(op.function, partial)
                    else op.function
                    for op in self.operations
                    if not op.effects and op.function is not self.reference
                )
            ),
        )

    def entry_valid(self):
        if self.reference.__code__ is not self.reference_code:
            return False
        if self.dispatcher is None:
            return True
        return (
            self.reference.__globals__.get(self.dispatcher.__name__) is self.dispatcher
            and self.dispatcher.__code__ is self.dispatcher_code
        )

    def bind(self, operation):
        return next(
            binding
            for op, binding in zip(self.operations, self.bindings)
            if op is operation
        )


def computation(name: str, implementation):
    """Bind at the original component read point, never during path analysis."""
    contract = execution_contract(name, implementation)
    if (
        contract is None
        or not contract.computation
        or type(implementation) is not FunctionType
    ):
        return None
    specification = vars(implementation).get("__execution_computation__")
    if (
        type(specification) is not Computation
        or specification.reference is not implementation
        or not specification.entry_valid()
    ):
        return None
    from ..runtime.regions import ComputationPlan

    return ComputationPlan(specification, contract)


_DE_READS = ("parents", "population", "fitness", "target_indices", "parameters", "rng")
_CONTRACTS = {
    "search_de_current_to_pbest": ExecutionContract(
        "de", "elite-donors-vector-coins-forced-index", _DE_READS, ("rng",)
    ),
    "search_fastga_de_current_to_pbest": ExecutionContract(
        "fastga_de",
        "matched-target-forced-index-conditional-coins",
        _DE_READS,
        ("rng", "invalidation"),
        deferred_final=True,
    ),
    "search_mu_polynomial": ExecutionContract(
        "polynomial",
        "sites-matrix-coins-matrix-numpy-power",
        ("parents", "bounds", "parameters", "rng"),
        ("rng",),
    ),
}
for _name in ("search_de_current", "search_de_random", "search_de_current_best"):
    _CONTRACTS[_name] = ExecutionContract(
        "",
        "input-prefix-then-reference-rng",
        _DE_READS,
        ("rng",),
        matrix_prefix=("parents", "population"),
        computation=True,
    )
_CONTRACTS["search_pso"] = ExecutionContract(
    "",
    "guide-index-then-two-particle-coefficient-arrays",
    ("parents", "population", "parameters", "Pbest", "Gbest", "V", "rng"),
    ("Pbest", "Gbest", "V", "rng"),
    matrix_prefix=("parents", "population"),
    computation=True,
)
for _name in ("search_mu_gaussian", "search_mu_cauchy"):
    _CONTRACTS[_name] = ExecutionContract(
        "",
        "eta-then-disturbance-then-row-normal-then-coordinate-normal",
        ("parents", "auxiliary", "rng"),
        ("auxiliary", "rng"),
        row_broadcast=True,
        computation=True,
    )
for _name in (
    "cross_arithmetic",
    "cross_sim_binary",
    "cross_point_one",
    "cross_point_two",
    "cross_point_uniform",
):
    _CONTRACTS[_name] = ExecutionContract(
        "",
        "reference-crossover-draw-order",
        ("parents", "parameters", "rng"),
        ("rng",),
        computation=True,
    )
_CONTRACTS["choose_nich"] = ExecutionContract(
    "",
    "species-order-then-reference-permutations",
    ("population", "decisions", "fitness", "generation", "Gmax", "rng"),
    ("rng",),
    computation=True,
)
for _family in ("search_bit_", "search_fastga_bit_"):
    for _law in (
        "uniform",
        "standard",
        "conditional",
        "shifted",
        "normal",
        "fast",
        "one",
        "three",
        "five",
    ):
        _fastga = _family == "search_fastga_bit_"
        _CONTRACTS[_family + _law] = ExecutionContract(
            "bit",
            f"{'fastga' if _fastga else 'ordinary'}-{_law}-strength-then-sites",
            ("parents", "bounds", "rng"),
            ("rng", "invalidation") if _fastga else ("rng",),
            deferred_final=_fastga,
            output_domain="binary",
        )


def is_builtin_component(name: str, implementation) -> bool:
    if name not in _COMPONENT_MODULES or name in _custom_components:
        return False
    # Observe the current module/function, without re-entering importlib on
    # every event. Registry replacement and module monkeypatching stay visible.
    module_name = __package__ + "." + _COMPONENT_MODULES[name]
    module = sys.modules.get(module_name)
    if module is None:
        module = import_module(module_name)
    return implementation is getattr(module, name, None)


def execution_contract(name: str, implementation=None) -> ExecutionContract | None:
    contract = _CONTRACTS.get(name)
    if contract is None or name not in _COMPONENT_MODULES or name in _custom_components:
        return None
    if implementation is not None and not is_builtin_component(name, implementation):
        return None
    return contract


@dataclass(frozen=True)
class UpdateContract:
    tie_rule: str
    random_rule: str = "none"
    barrier: str = "all-offspring-evaluated"
    grouping: str = "first-arrival-representative-current-parameter"
    escapes: tuple[str, ...] = ("surviving-solutions", "auxiliary-state")


_UPDATE_CONTRACTS = {
    "update_greedy": UpdateContract("numpy-argsort"),
    "update_fastga_plus": UpdateContract("stable-offspring-before-parents"),
    "update_pairwise": UpdateContract("strict-improvement-in-arrival-order"),
    "update_simulated_annealing": UpdateContract(
        "strict-improvement-otherwise-random-acceptance",
        "one-scalar-coin-for-each-non-improvement",
    ),
    "update_round_robin": UpdateContract(
        "numpy-argpartition-then-reversed-argsort", "square-random-matrix"
    ),
}


def update_contract(name: str, implementation=None) -> UpdateContract | None:
    contract = _UPDATE_CONTRACTS.get(name)
    if contract is None or name not in _COMPONENT_MODULES or name in _custom_components:
        return None
    if implementation is not None and not is_builtin_component(name, implementation):
        return None
    return contract

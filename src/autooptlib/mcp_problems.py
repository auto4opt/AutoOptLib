"""Translate MCP problem inputs to AutoOptLib problem definitions."""

from __future__ import annotations

import ast
import importlib
import operator
import re
from collections.abc import Mapping
from importlib.util import find_spec
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from . import IOHInstance, make_bbob_problem, make_pbo_problem, make_problem

SPHERE_BOUNDS = (-5.0, 5.0)
PROBLEM_ID = re.compile(r"(bbob|pbo)_f([1-9][0-9]?)")
SUITE_SIZES = {"bbob": 24, "pbo": 23}

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "abs": np.abs,
    "sum": np.sum,
    "mean": np.mean,
    "min": np.min,
    "max": np.max,
    "dot": np.dot,
    "matmul": np.matmul,
    "sqrt": np.sqrt,
    "exp": np.exp,
    "log": np.log,
    "sin": np.sin,
    "cos": np.cos,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "where": np.where,
}
_COMPARISONS = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}


def _validate_expression(node: ast.AST) -> None:
    if isinstance(node, ast.Constant) and (type(node.value) in (int, float, str)):
        return
    if isinstance(node, ast.Name) and node.id in ("x", "data", "pi"):
        return
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        children = (node.left, node.right)
    elif isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        children = (node.operand,)
    elif (
        isinstance(node, ast.Compare)
        and len(node.ops) == 1
        and type(node.ops[0]) in _COMPARISONS
    ):
        children = (node.left, node.comparators[0])
    elif isinstance(node, ast.Subscript):
        children = (node.value, node.slice)
    elif isinstance(node, ast.Slice):
        children = tuple(
            part for part in (node.lower, node.upper, node.step) if part is not None
        )
    elif isinstance(node, (ast.List, ast.Tuple)):
        children = node.elts
    elif (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        children = node.args
    else:
        raise ValueError(f"Unsupported expression syntax: {type(node).__name__}")
    for child in children:
        _validate_expression(child)


def _evaluate(node: ast.AST, variables: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return (
            float(node.value)
            if type(node.value) is int and abs(node.value) > 2**53
            else node.value
        )
    if isinstance(node, ast.Name):
        return variables[node.id]
    if isinstance(node, ast.BinOp):
        # NumPy arithmetic keeps vector expressions numeric and avoids Python
        # repeating lists or strings, or allocating unbounded integers for powers.
        left = np.asarray(_evaluate(node.left, variables), dtype=float)
        right = np.asarray(_evaluate(node.right, variables), dtype=float)
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp):
        return _UNARY[type(node.op)](_evaluate(node.operand, variables))
    if isinstance(node, ast.Compare):
        return _COMPARISONS[type(node.ops[0])](
            _evaluate(node.left, variables), _evaluate(node.comparators[0], variables)
        )
    if isinstance(node, ast.Subscript):
        return _evaluate(node.value, variables)[_evaluate(node.slice, variables)]
    if isinstance(node, ast.Slice):
        return slice(
            *(
                _evaluate(part, variables) if part is not None else None
                for part in (node.lower, node.upper, node.step)
            )
        )
    if isinstance(node, (ast.List, ast.Tuple)):
        return tuple(_evaluate(item, variables) for item in node.elts)
    if isinstance(node, ast.Call):
        return _FUNCTIONS[node.func.id](
            *(_evaluate(item, variables) for item in node.args)
        )
    raise AssertionError("Unexpected validated expression node")


def _parse_expression(source: str) -> ast.AST:
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ValueError(f"Invalid mathematical expression: {source!r}") from exc
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError(
            "A mathematical expression can contain at most 256 syntax nodes."
        )
    _validate_expression(tree.body)
    return tree.body


class CustomProblem(BaseModel):
    """Static mathematical problem. The expression sees x and the JSON instance as data."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    name: str = Field(default="custom", min_length=1, max_length=80)
    problem_type: Literal["continuous", "discrete", "permutation"] = "continuous"
    sense: Literal["minimize", "maximize", "min", "max"] = "minimize"
    objective: str = Field(
        min_length=1,
        max_length=2000,
        description="Scalar formula in x and data, for example sum((x - data['target']) ** 2).",
    )
    constraints: list[str] = Field(
        default_factory=list,
        max_length=32,
        description="Expressions that must be <= 0; use sum(x) - capacity for a limit.",
    )
    bounds: tuple[float, float] | tuple[list[float], list[float]] | None = Field(
        default=None,
        description=(
            "Use [lower, upper] for integer dimension instances, or two numeric "
            "arrays [all lower bounds, all upper bounds] for fixed dimension, "
            "for example [[0, 0, 0], [6, 6, 6]]."
        ),
    )

    def definition(self):
        objective = _parse_expression(self.objective)
        constraints = tuple(_parse_expression(item) for item in self.constraints)
        bounds = self.bounds
        if bounds is None and self.problem_type != "permutation":
            raise ValueError("Continuous and discrete custom problems require bounds.")
        if bounds is not None:
            array = np.asarray(bounds, dtype=float)
            if array.size < 2 or np.any(array[0] > array[1]):
                raise ValueError(
                    "Bounds require a lower and upper value for each variable."
                )
            if self.problem_type != "continuous" and np.any(array != np.floor(array)):
                raise ValueError("Discrete and permutation bounds must be integers.")
        direction = -1 if self.sense in ("maximize", "max") else 1

        def objective_value(x, data):
            return direction * _evaluate(objective, {"x": x, "data": data, "pi": np.pi})

        def constraint_values(x, data):
            variables = {"x": x, "data": data, "pi": np.pi}
            return np.concatenate(
                [
                    np.asarray(_evaluate(item, variables), dtype=float).reshape(-1)
                    for item in constraints
                ]
            )

        problem = make_problem(
            objective_value,
            bounds=bounds,
            problem_type=self.problem_type,
            constraint=constraint_values if constraints else None,
            name=self.name,
        )
        problem.__autoopt_checkpoint_semantics__ = self.model_dump(mode="json")
        return problem


class ProblemCatalog:
    def __init__(self, modules: list[str]):
        self.problems = {}
        self.options = {}
        for name in modules:
            module = importlib.import_module(name)
            problems = getattr(module, "PROBLEMS", {})
            options = getattr(module, "RUN_OPTIONS", {})
            if not (isinstance(problems, Mapping) and isinstance(options, Mapping)):
                raise ValueError(f"{name}: PROBLEMS and RUN_OPTIONS must be mappings.")
            for problem_id, problem in problems.items():
                if not (
                    re.fullmatch(r"[a-z][a-z0-9_-]{0,79}", problem_id)
                    and callable(problem)
                ):
                    raise ValueError(f"{name}: invalid problem entry {problem_id!r}.")
                if (
                    problem_id == "sphere"
                    or PROBLEM_ID.fullmatch(problem_id)
                    or problem_id in self.problems
                ):
                    raise ValueError(f"{name}: duplicate problem ID {problem_id!r}.")
                self.problems[problem_id] = problem
            for profile, values in options.items():
                if profile in self.options or not isinstance(values, Mapping):
                    raise ValueError(
                        f"{name}: duplicate or invalid run profile {profile!r}."
                    )
                self.options[profile] = dict(values)

    def resolve(self, problem_id: str, custom: CustomProblem | None):
        if custom is not None:
            if problem_id != "custom":
                raise ValueError("Set problem_id='custom' when sending custom_problem.")
            return custom.definition(), lambda item: item
        if problem_id == "custom":
            raise ValueError(
                "Supply custom_problem or use an ID from a configured extension."
            )
        if problem_id == "sphere":
            return make_problem(
                _sphere, bounds=SPHERE_BOUNDS, name="sphere"
            ), _dimension
        match = PROBLEM_ID.fullmatch(problem_id)
        if match:
            suite, number = match.group(1), int(match.group(2))
            if not 1 <= number <= SUITE_SIZES[suite]:
                raise ValueError(f"{suite} function IDs are 1–{SUITE_SIZES[suite]}.")
            if find_spec("ioh") is None:
                raise ValueError(
                    "Install AutoOptLib with the MCP extra to enable BBOB and PBO."
                )
            factory = make_bbob_problem if suite == "bbob" else make_pbo_problem
            return factory(number), _ioh_instance
        if problem_id in self.problems:
            return self.problems[problem_id], lambda item: item
        raise ValueError(f"Unknown problem ID {problem_id!r}.")

    def run_options(self, name: str | None):
        if name is None:
            return {}
        if name not in self.options:
            raise ValueError(f"Unknown run profile {name!r}.")
        return self.options[name]


def _sphere(decision, _dimension):
    return float(np.sum(decision**2))


def _dimension(item):
    if type(item) is not int or item < 2:
        raise ValueError("Supply an integer dimension of at least two.")
    return item


def _ioh_instance(item):
    if type(item) is int:
        return IOHInstance(dimension=_dimension(item))
    if not isinstance(item, dict) or any(
        type(value) is not int and (key != "budget" or value is not None)
        for key, value in item.items()
    ):
        raise ValueError("Supply a dimension or an IOH instance with integer fields.")
    _dimension(item.get("dimension"))
    return IOHInstance(**item)

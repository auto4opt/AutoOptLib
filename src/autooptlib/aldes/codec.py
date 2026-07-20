"""Translate ALDes token programs into AutoOptLib algorithm designs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from ..components import get_component
from ..utils.design import Design
from ..utils.design._helpers import Pathway, PathwayParam, SearchParam, SearchStep
from ..utils.space import space
from .vocabulary import PARAMETER_INDICES, TOKEN_BY_INDEX, TokenKind, validate_sequence


@dataclass(frozen=True)
class ComponentInstruction:
    token: int
    parameter_token: int | None
    pointer: str
    pointer_parameter_token: int | None

    @property
    def name(self) -> str:
        return TOKEN_BY_INDEX[self.token].name

    @property
    def kind(self) -> TokenKind:
        return TOKEN_BY_INDEX[self.token].kind


def _parse(sequence: Sequence[int] | np.ndarray) -> list[ComponentInstruction]:
    values = validate_sequence(sequence)
    instructions: list[ComponentInstruction] = []
    position = 1
    while position < len(values) - 1:
        component = TOKEN_BY_INDEX[values[position]]
        component_index = values[position]
        position += 1
        parameter_token = None
        if component.parameter_count:
            parameter_token = values[position]
            position += 1
        pointer = TOKEN_BY_INDEX[values[position]]
        position += 1
        pointer_parameter_token = None
        if pointer.parameter_count:
            pointer_parameter_token = values[position]
            position += 1
        instructions.append(
            ComponentInstruction(
                token=component_index,
                parameter_token=parameter_token,
                pointer=pointer.name,
                pointer_parameter_token=pointer_parameter_token,
            )
        )
    return instructions


def _parameter_value(
    instruction: ComponentInstruction, problem: Any
) -> np.ndarray | None:
    if instruction.parameter_token is None:
        return None
    component = get_component(instruction.name)
    bounds, _ = component(problem, "parameter")
    if bounds is None:
        return None
    array = np.asarray(bounds, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 2)
    bin_index = PARAMETER_INDICES.index(instruction.parameter_token)
    fraction = bin_index / 9.0
    return array[:, 0] + (array[:, 1] - array[:, 0]) * fraction


def _termination(instruction: ComponentInstruction, setting: Any) -> np.ndarray:
    if instruction.pointer == "forward":
        return np.array([-math.inf, 1.0])
    if instruction.pointer == "iterate":
        condition_values = (0.01, 0.05, 0.10, 0.15, 0.20)
        token = instruction.pointer_parameter_token
        if token is None:
            raise ValueError("Iterate pointer is missing its condition token.")
        fraction = condition_values[PARAMETER_INDICES.index(token)]
        probability_evaluations = int(
            getattr(setting, "ProbFE", getattr(setting, "prob_fe", 5000))
        )
        population_size = int(getattr(setting, "ProbN", getattr(setting, "prob_n", 20)))
        limit = max(1, math.ceil(fraction * probability_evaluations / population_size))
        rate = float(getattr(setting, "IncRate", getattr(setting, "inc_rate", 0.05)))
        return np.array([rate, float(limit)])
    return np.array([-math.inf, 1.0])


def _branches(
    instructions: list[ComponentInstruction],
) -> list[list[ComponentInstruction]]:
    branches = [list(instructions)]
    for fork_position, instruction in enumerate(instructions):
        if instruction.pointer != "fork":
            continue
        token = instruction.pointer_parameter_token
        if token is None:
            raise ValueError("Fork pointer is missing its target token.")
        # ALDes maps parameter bins to one-based operator targets. The
        # constrained grammar retains only targets 2 and 3 because all larger
        # targets clamp to one of those same branch structures.
        target = max(0, PARAMETER_INDICES.index(token) - 1)
        # The legacy generator clamps a fork target to the final search
        # component. Never let a branch jump directly to the update.
        target = min(target, len(instructions) - 2)
        branch = instructions[: fork_position + 1] + instructions[target:]
        # Avoid a duplicated component at the splice boundary.
        compact: list[ComponentInstruction] = []
        for item in branch:
            if compact and compact[-1] is item:
                continue
            compact.append(item)
        if (
            compact
            and compact[0].kind is TokenKind.CHOOSE
            and compact[-1].kind is TokenKind.UPDATE
            and any(item.kind is TokenKind.SEARCH for item in compact)
        ):
            branches.append(compact)
    return branches


def _pathway(
    instructions: list[ComponentInstruction], problem: Any, setting: Any
) -> tuple[Pathway, PathwayParam]:
    choose = instructions[0]
    update = instructions[-1]
    searches = instructions[1:-1]
    steps: list[SearchStep] = []
    parameters: list[SearchParam] = []
    position = 0
    while position < len(searches):
        primary = searches[position]
        secondary = None
        if primary.name.startswith("cross_") and position + 1 < len(searches):
            candidate = searches[position + 1]
            if candidate.name.startswith("search_reset_"):
                secondary = candidate
                position += 1
        termination = _termination(primary, setting)
        if secondary is not None and secondary.pointer == "forward":
            termination = np.array([-math.inf, 1.0])
        steps.append(
            SearchStep(
                primary=primary.name,
                secondary=secondary.name if secondary is not None else None,
                termination=termination,
            )
        )
        parameters.append(
            SearchParam(
                primary=_parameter_value(primary, problem),
                secondary=(
                    _parameter_value(secondary, problem)
                    if secondary is not None
                    else None
                ),
            )
        )
        position += 1

    archive = list(getattr(setting, "Archive", getattr(setting, "archive", [])) or [])
    return (
        Pathway(
            choose=choose.name,
            search=steps,
            update=update.name,
            archive=archive,
        ),
        PathwayParam(
            choose=_parameter_value(choose, problem),
            search=parameters,
            update=_parameter_value(update, problem),
        ),
    )


def decode_sequence(
    sequence: Sequence[int] | np.ndarray, problem: Any, setting: Any
) -> Design:
    """Decode an ALDes action sequence into an executable :class:`Design`."""

    instructions = _parse(sequence)
    problems = list(problem) if isinstance(problem, (list, tuple)) else [problem]
    if not problems:
        raise ValueError("At least one constructed problem is required.")
    if not hasattr(setting, "AllOp") and not hasattr(setting, "all_op"):
        setting = space(problems, setting)

    pathways: list[Pathway] = []
    pathway_parameters: list[PathwayParam] = []
    instruction_branches = _branches(instructions)
    for branch in instruction_branches:
        pathway, parameters = _pathway(branch, problems, setting)
        pathways.append(pathway)
        pathway_parameters.append(parameters)

    all_op = list(getattr(setting, "AllOp", getattr(setting, "all_op", [])))
    matrices: list[np.ndarray] = []
    for branch in instruction_branches:
        indices = [all_op.index(item.name) + 1 for item in branch]
        matrices.append(np.asarray(list(zip(indices[:-1], indices[1:])), dtype=int))

    encoded_parameters: list[list[Any]] = [[None, None] for _ in all_op]
    for instruction in instructions:
        index = all_op.index(instruction.name)
        behavior = None
        if instruction.kind is TokenKind.SEARCH:
            behavior = "GS" if instruction.pointer == "forward" else "LS"
        encoded_parameters[index] = [
            _parameter_value(instruction, problems),
            behavior,
        ]

    design = Design()
    design.operator = matrices
    design.parameter = encoded_parameters
    design.construct([pathways], [pathway_parameters])
    runs = int(getattr(setting, "AlgRuns", getattr(setting, "alg_runs", 1)))
    design.performance = np.zeros((len(problems), runs))
    design.performance_approx = np.zeros((len(problems), runs))
    design.last_runs = {index: [None] * runs for index in range(len(problems))}
    design.aldes_sequence = validate_sequence(sequence)
    return design


__all__ = ["ComponentInstruction", "decode_sequence"]

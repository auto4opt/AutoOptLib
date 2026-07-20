"""ALDes-compatible vocabulary and constrained grammar utilities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Sequence

import numpy as np


class TokenKind(str, Enum):
    CHOOSE = "choose"
    SEARCH = "search"
    UPDATE = "update"
    BEGIN = "begin"
    END = "end"
    PARAMETER = "parameter"
    POINTER = "pointer"


@dataclass(frozen=True)
class Token:
    index: int
    name: str
    kind: TokenKind
    parameter_count: int = 0


_COMPONENTS = (
    ("choose_traverse", TokenKind.CHOOSE, 0),
    ("choose_tournament", TokenKind.CHOOSE, 0),
    ("choose_roulette_wheel", TokenKind.CHOOSE, 0),
    ("choose_nich", TokenKind.CHOOSE, 0),
    ("cross_point_one", TokenKind.SEARCH, 0),
    ("cross_point_two", TokenKind.SEARCH, 0),
    ("cross_point_n", TokenKind.SEARCH, 1),
    ("cross_point_uniform", TokenKind.SEARCH, 1),
    ("search_reset_one", TokenKind.SEARCH, 0),
    ("search_reset_n", TokenKind.SEARCH, 1),
    ("search_reset_rand", TokenKind.SEARCH, 1),
    ("reinit_discrete", TokenKind.SEARCH, 0),
    ("update_greedy", TokenKind.UPDATE, 0),
    ("update_round_robin", TokenKind.UPDATE, 0),
    ("update_pairwise", TokenKind.UPDATE, 0),
    ("update_always", TokenKind.UPDATE, 0),
    ("update_simulated_annealing", TokenKind.UPDATE, 1),
)

TOKENS: tuple[Token, ...] = (
    tuple(
        Token(index, name, kind, parameter_count)
        for index, (name, kind, parameter_count) in enumerate(_COMPONENTS)
    )
    + (
        Token(17, "begin", TokenKind.BEGIN),
        Token(18, "end", TokenKind.END),
    )
    + tuple(
        Token(index, f"{value / 10:.1f}", TokenKind.PARAMETER)
        for index, value in enumerate(range(1, 11), start=19)
    )
    + (
        Token(29, "forward", TokenKind.POINTER),
        Token(30, "iterate", TokenKind.POINTER, 1),
        Token(31, "fork", TokenKind.POINTER, 1),
    )
)

TOKEN_BY_INDEX = {token.index: token for token in TOKENS}
TOKEN_BY_NAME = {token.name: token for token in TOKENS}
VOCABULARY_SIZE = len(TOKENS)
BEGIN_INDEX = TOKEN_BY_NAME["begin"].index
END_INDEX = TOKEN_BY_NAME["end"].index
PARAMETER_INDICES = tuple(range(19, 29))
POINTER_INDICES = tuple(range(29, 32))

# Discrete ALDes permits at most one global-search component.  The first
# group is global for every parameter choice; the second becomes global for
# parameter tokens 0.4--1.0, matching the legacy ``gs_para_begin = 22``.
_ONLY_GLOBAL_SEARCH = frozenset({4, 5, 11})
_PARAMETERIZED_GLOBAL_SEARCH = frozenset({6, 7, 9, 10})
_GLOBAL_PARAMETER_START = 22
# The tightened fork grammar has two distinct branch modes. Token 0.3 keeps
# the complete search row on both branches; token 0.4 lets the second branch
# start at the mutation in a crossover+mutation row. Larger tokens collapsed
# to one of those two structures after target clamping and were aliases.
_FORK_PARAMETER_INDICES = PARAMETER_INDICES[2:4]


class SequenceValidationError(ValueError):
    """Raised when an ALDes token sequence does not follow the grammar."""


def normalize_sequence(sequence: Sequence[int] | np.ndarray) -> list[int]:
    """Return one canonical sequence with one begin and one end token."""

    values = np.asarray(sequence).reshape(-1).tolist()
    try:
        values = [int(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise SequenceValidationError(
            "ALDes sequences must contain integer tokens."
        ) from exc
    if not values:
        raise SequenceValidationError("ALDes sequence cannot be empty.")
    unknown = [value for value in values if value not in TOKEN_BY_INDEX]
    if unknown:
        raise SequenceValidationError(f"Unknown ALDes token index: {unknown[0]}")
    if values[0] != BEGIN_INDEX:
        values.insert(0, BEGIN_INDEX)
    while len(values) > 1 and values[-1] == END_INDEX:
        values.pop()
    values.append(END_INDEX)
    return values


def _consume_component(values: Sequence[int], position: int) -> int:
    token = TOKEN_BY_INDEX[values[position]]
    position += 1
    for _ in range(token.parameter_count):
        if position >= len(values) or values[position] not in PARAMETER_INDICES:
            raise SequenceValidationError(
                f"Component {token.name!r} must be followed by a parameter token."
            )
        position += 1
    return position


def validate_sequence(sequence: Sequence[int] | np.ndarray) -> list[int]:
    """Validate AutoOptLib's constrained ALDes grammar and normalize it."""

    values = normalize_sequence(sequence)
    position = 1
    expected = TokenKind.CHOOSE
    search_count = 0
    search_tokens: list[int] = []
    operator_count = 0
    has_fork = False
    fork_parameter: int | None = None
    components_seen: set[int] = set()
    global_search_count = 0

    while position < len(values) - 1:
        token = TOKEN_BY_INDEX[values[position]]
        if token.kind is not expected:
            raise SequenceValidationError(
                f"Expected a {expected.value} token at position {position}, "
                f"got {token.name!r}."
            )
        operator_count += 1
        if operator_count > 6:
            raise SequenceValidationError("ALDes supports at most six components.")
        if token.index in components_seen:
            raise SequenceValidationError(
                f"Component {token.name!r} cannot appear more than once."
            )
        components_seen.add(token.index)
        if token.kind is TokenKind.SEARCH:
            search_count += 1
            search_tokens.append(token.index)
            is_global = token.index in _ONLY_GLOBAL_SEARCH
            if token.index in _PARAMETERIZED_GLOBAL_SEARCH:
                parameter_position = position + 1
                is_global = (
                    parameter_position < len(values)
                    and values[parameter_position] >= _GLOBAL_PARAMETER_START
                )
            if is_global:
                global_search_count += 1
                if global_search_count > 1:
                    raise SequenceValidationError(
                        "An ALDes sequence permits at most one global search."
                    )
        position = _consume_component(values, position)
        if position >= len(values) - 1:
            raise SequenceValidationError(
                f"Component {token.name!r} must be followed by a pointer."
            )
        pointer = TOKEN_BY_INDEX[values[position]]
        if pointer.kind is not TokenKind.POINTER:
            raise SequenceValidationError(
                f"Expected a pointer after {token.name!r}, got {pointer.name!r}."
            )
        if pointer.name == "fork" and token.kind is not TokenKind.CHOOSE:
            raise SequenceValidationError(
                "A fork pointer may only follow the choose component."
            )
        if token.kind is TokenKind.CHOOSE and pointer.name == "iterate":
            raise SequenceValidationError(
                "The choose component must use forward or fork."
            )
        if token.kind is TokenKind.UPDATE and pointer.name != "forward":
            raise SequenceValidationError(
                "The final update component must use the forward pointer."
            )
        has_fork |= pointer.name == "fork"
        position += 1
        if pointer.parameter_count:
            if position >= len(values) - 1 or values[position] not in PARAMETER_INDICES:
                raise SequenceValidationError(
                    f"Pointer {pointer.name!r} must be followed by a parameter token."
                )
            if pointer.name == "iterate" and values[position] > PARAMETER_INDICES[4]:
                raise SequenceValidationError(
                    "Iterate accepts only the five condition tokens 0.1 through 0.5."
                )
            if pointer.name == "fork":
                if values[position] not in _FORK_PARAMETER_INDICES:
                    raise SequenceValidationError(
                        "Fork accepts only its two distinct branch-mode tokens."
                    )
                fork_parameter = values[position]
            position += 1

        if token.index in {4, 5, 6, 7}:
            if position >= len(values) - 1 or values[position] not in {8, 9, 10}:
                raise SequenceValidationError(
                    "A crossover must be followed by one mutation component."
                )

        if token.kind is TokenKind.CHOOSE:
            expected = TokenKind.SEARCH
        elif token.kind is TokenKind.SEARCH:
            expected = TokenKind.SEARCH
            if position < len(values) - 1:
                next_kind = TOKEN_BY_INDEX[values[position]].kind
                if next_kind is TokenKind.UPDATE:
                    expected = TokenKind.UPDATE
        else:
            if position != len(values) - 1:
                raise SequenceValidationError("The update component must be last.")

    if expected is not TokenKind.UPDATE or search_count == 0:
        raise SequenceValidationError(
            "A complete ALDes sequence needs choose, search, and update components."
        )
    if has_fork:
        paired_search = (
            len(search_tokens) == 2
            and search_tokens[0] in {4, 5, 6, 7}
            and search_tokens[1] in {8, 9, 10}
        )
        single_search = len(search_tokens) == 1 and search_tokens[0] not in {
            4,
            5,
            6,
            7,
        }
        if not (single_search or paired_search):
            raise SequenceValidationError(
                "A fork algorithm permits one search step, optionally a "
                "crossover followed by one mutation."
            )
        if fork_parameter == _FORK_PARAMETER_INDICES[1] and not paired_search:
            raise SequenceValidationError(
                "The mutation-branch fork mode requires crossover followed by mutation."
            )
    return values


def _has_global_search(values: Sequence[int]) -> bool:
    for position, value in enumerate(values):
        if value in _ONLY_GLOBAL_SEARCH:
            return True
        if value in _PARAMETERIZED_GLOBAL_SEARCH and position + 1 < len(values):
            parameter = values[position + 1]
            if parameter in PARAMETER_INDICES and parameter >= _GLOBAL_PARAMETER_START:
                return True
    return False


def allowed_next_tokens(prefix: Sequence[int] | np.ndarray) -> np.ndarray:
    """Return a boolean mask for legal continuations of a partial sequence.

    This is the shared grammar mask used by both random sampling and the
    autoregressive PyTorch generator.  It intentionally accepts an incomplete
    prefix and therefore does not call :func:`validate_sequence`.
    """

    values = np.asarray(prefix).reshape(-1).astype(int).tolist()
    if not values:
        values = [BEGIN_INDEX]
    if values[0] != BEGIN_INDEX:
        values.insert(0, BEGIN_INDEX)
    mask = np.zeros(VOCABULARY_SIZE, dtype=bool)
    last = TOKEN_BY_INDEX.get(values[-1])
    if last is None:
        return mask
    if last.kind is TokenKind.END:
        mask[END_INDEX] = True
        return mask
    if last.kind is TokenKind.BEGIN:
        mask[0:4] = True
        return mask

    component_positions = [
        index
        for index, value in enumerate(values)
        if TOKEN_BY_INDEX.get(value, Token(-1, "", TokenKind.END)).kind
        in {TokenKind.CHOOSE, TokenKind.SEARCH, TokenKind.UPDATE}
    ]
    components = [values[index] for index in component_positions]
    most_recent_component = components[-1] if components else None
    most_recent_kind = (
        TOKEN_BY_INDEX[most_recent_component].kind if components else None
    )
    has_global_search = _has_global_search(values)
    has_fork = TOKEN_BY_NAME["fork"].index in values

    # A component or pointer that owns a parameter must receive it next.
    if last.kind in {
        TokenKind.CHOOSE,
        TokenKind.SEARCH,
        TokenKind.UPDATE,
        TokenKind.POINTER,
    }:
        if last.parameter_count:
            if last.name == "iterate":
                mask[list(PARAMETER_INDICES[:5])] = True
            elif last.name == "fork":
                mask[list(_FORK_PARAMETER_INDICES)] = True
            elif last.index in _PARAMETERIZED_GLOBAL_SEARCH and has_global_search:
                mask[list(PARAMETER_INDICES[:3])] = True
            else:
                mask[list(PARAMETER_INDICES)] = True
            return mask

    # A parameter may belong to the preceding component or pointer.
    owner = None
    for value in reversed(values[:-1] if last.kind is TokenKind.PARAMETER else values):
        candidate = TOKEN_BY_INDEX[value]
        if candidate.kind in {
            TokenKind.CHOOSE,
            TokenKind.SEARCH,
            TokenKind.UPDATE,
            TokenKind.POINTER,
        }:
            owner = candidate
            break
    if last.kind in {TokenKind.CHOOSE, TokenKind.SEARCH, TokenKind.UPDATE} or (
        last.kind is TokenKind.PARAMETER
        and owner is not None
        and owner.kind is not TokenKind.POINTER
    ):
        if most_recent_kind is TokenKind.CHOOSE:
            mask[29] = True
            mask[31] = True
        elif most_recent_kind is TokenKind.SEARCH:
            mask[29:31] = True
        elif most_recent_kind is TokenKind.UPDATE:
            mask[29] = True
        return mask

    # A pointer without a parameter, or the parameter of a pointer, opens the
    # next component position.
    pointer_complete = last.name == "forward" or (
        last.kind is TokenKind.PARAMETER
        and owner is not None
        and owner.kind is TokenKind.POINTER
    )
    if pointer_complete:
        count = len(components)
        if most_recent_kind is TokenKind.UPDATE:
            mask[END_INDEX] = True
        elif count == 1:
            mask[4:12] = True
            if (
                last.kind is TokenKind.PARAMETER
                and owner is not None
                and owner.name == "fork"
                and last.index == _FORK_PARAMETER_INDICES[1]
            ):
                # The heterogeneous fork starts its second branch at the
                # paired mutation, so it requires a crossover search row.
                mask[8:12] = False
        elif has_fork and most_recent_kind is TokenKind.SEARCH:
            mask[12:17] = True
        elif count >= 5:
            mask[12:17] = True
        elif count == 4:
            # A crossover would require a paired mutation and exceed the
            # six-component limit (choose + four searches + update).
            mask[8:17] = True
        else:
            mask[4:17] = True

        # Crossover must be followed by a mutation operator.
        if most_recent_component in {4, 5, 6, 7}:
            mask[:] = False
            mask[8:11] = True
        for component in components:
            if 0 <= component <= 16:
                mask[component] = False
        if has_global_search:
            mask[list(_ONLY_GLOBAL_SEARCH)] = False
        return mask

    return mask


def tokens_to_names(sequence: Iterable[int]) -> list[str]:
    return [TOKEN_BY_INDEX[int(index)].name for index in sequence]


__all__ = [
    "BEGIN_INDEX",
    "END_INDEX",
    "PARAMETER_INDICES",
    "POINTER_INDICES",
    "SequenceValidationError",
    "TOKENS",
    "TOKEN_BY_INDEX",
    "TOKEN_BY_NAME",
    "Token",
    "TokenKind",
    "VOCABULARY_SIZE",
    "allowed_next_tokens",
    "normalize_sequence",
    "tokens_to_names",
    "validate_sequence",
]

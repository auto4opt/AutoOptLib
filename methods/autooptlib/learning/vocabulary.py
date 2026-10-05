"""Discrete vocabulary for learning over AutoOptLib algorithm graphs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

STRUCTURAL_TOKENS = (
    "begin",
    "end",
    "problem_continuous",
    "problem_discrete",
    "problem_permutation",
    "path_begin",
    "path_end",
    "secondary",
    "terminate_local",
    "terminate_global",
    "archive_begin",
    "archive_end",
)
PARAMETER_TOKENS = tuple(f"parameter_{index}" for index in range(10))
_BEHAVIOR_MODES = {"neutral", "local", "global", "parameter"}


def _has_behavior(value: Any, row: int, expected: str) -> bool:
    try:
        entry = value[row]
        if isinstance(entry, np.ndarray):
            entry = entry.tolist()
        if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
            return bool(entry) and entry[0] == expected
        return entry == expected
    except (IndexError, TypeError):
        return False


@dataclass(frozen=True)
class LearningVocabulary:
    """Versioned token specification derived from an AutoOptLib design space."""

    choose: tuple[str, ...]
    search: tuple[str, ...]
    update: tuple[str, ...]
    archives: tuple[str, ...]
    parameter_counts: tuple[int, ...]
    behavior_modes: tuple[str, ...]
    local_bin_masks: tuple[tuple[int, ...], ...]
    component_problem_types: tuple[tuple[str, ...], ...]
    problem_type: str

    def __post_init__(self) -> None:
        components = self.components
        for name, values in (
            ("choose", self.choose),
            ("search", self.search),
            ("update", self.update),
            ("archives", self.archives),
        ):
            if any(not isinstance(value, str) or not value for value in values):
                raise ValueError(f"{name} must contain non-empty strings.")
        token_names = STRUCTURAL_TOKENS + PARAMETER_TOKENS + components + self.archives
        if len(set(token_names)) != len(token_names):
            raise ValueError("Learning vocabulary token names must be unique.")
        if len(self.parameter_counts) != len(components):
            raise ValueError("parameter_counts must contain one entry per component.")
        if len(self.behavior_modes) != len(components):
            raise ValueError("behavior_modes must contain one entry per component.")
        if len(self.local_bin_masks) != len(components):
            raise ValueError("local_bin_masks must contain one entry per component.")
        if len(self.component_problem_types) != len(components):
            raise ValueError(
                "component_problem_types must contain one entry per component."
            )
        if any(type(count) is not int or count < 0 for count in self.parameter_counts):
            raise ValueError("Component parameter counts cannot be negative.")
        if any(mode not in _BEHAVIOR_MODES for mode in self.behavior_modes):
            raise ValueError("Unknown component behavior mode.")
        for count, masks in zip(self.parameter_counts, self.local_bin_masks):
            if len(masks) not in {0, count}:
                raise ValueError("Local-bin masks must match component parameters.")
            if any(
                type(mask) is not int or mask < 0 or mask >= (1 << 10) for mask in masks
            ):
                raise ValueError("Local-bin masks must use ten bits.")
        valid_types = {"continuous", "discrete", "permutation"}
        if any(
            not set(types) <= valid_types or not types
            for types in self.component_problem_types
        ):
            raise ValueError("Invalid component problem-type compatibility.")
        if self.problem_type not in valid_types | {"mixed"}:
            raise ValueError("Unsupported learning-vocabulary problem type.")

    @property
    def components(self) -> tuple[str, ...]:
        return self.choose + self.search + self.update

    @property
    def names(self) -> tuple[str, ...]:
        return STRUCTURAL_TOKENS + PARAMETER_TOKENS + self.components + self.archives

    @property
    def size(self) -> int:
        return len(self.names)

    @property
    def begin_index(self) -> int:
        return 0

    @property
    def end_index(self) -> int:
        return 1

    @property
    def parameter_indices(self) -> tuple[int, ...]:
        start = len(STRUCTURAL_TOKENS)
        return tuple(range(start, start + len(PARAMETER_TOKENS)))

    @property
    def parameter_token_count(self) -> int:
        """Number of tokens used to encode one component parameter."""

        return 1

    def index(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(
                f"Token {name!r} is not in this learning vocabulary."
            ) from exc

    def name(self, index: int) -> str:
        raw_index = index
        try:
            index = int(raw_index)
        except (TypeError, ValueError, OverflowError) as exc:
            raise KeyError(f"Unknown learning token index: {raw_index}") from exc
        if isinstance(raw_index, (bool, np.bool_)) or raw_index != index:
            raise KeyError(f"Unknown learning token index: {raw_index}")
        if not 0 <= index < len(self.names):
            raise KeyError(f"Unknown learning token index: {index}")
        try:
            return self.names[index]
        except (IndexError, ValueError) as exc:
            raise KeyError(f"Unknown learning token index: {index}") from exc

    def component_position(self, component: str) -> int:
        try:
            return self.components.index(component)
        except ValueError as exc:
            raise KeyError(f"Unknown component token {component!r}.") from exc

    def parameter_count(self, component: str) -> int:
        return self.parameter_counts[self.component_position(component)]

    def behavior_mode(self, component: str) -> str:
        return self.behavior_modes[self.component_position(component)]

    def local_masks(self, component: str) -> tuple[int, ...]:
        return self.local_bin_masks[self.component_position(component)]

    def problem_types(self, component: str) -> tuple[str, ...]:
        return self.component_problem_types[self.component_position(component)]

    @property
    def supported_problem_types(self) -> tuple[str, ...]:
        values = {value for types in self.component_problem_types for value in types}
        return tuple(
            name for name in ("continuous", "discrete", "permutation") if name in values
        )

    def compatible(self, component: str, problem_type: str) -> bool:
        return problem_type in self.problem_types(component)

    def category_components(self, category: str, problem_type: str) -> tuple[str, ...]:
        values = getattr(self, category)
        return tuple(name for name in values if self.compatible(name, problem_type))

    def behavior(self, component: str, parameter_bins: Sequence[int]) -> str:
        """Return the repair-equivalent local/global behavior of a component."""

        mode = self.behavior_mode(component)
        if mode in {"local", "global", "neutral"}:
            return mode
        masks = self.local_masks(component)
        if len(parameter_bins) != len(masks):
            raise ValueError(f"Incomplete parameter bins for {component!r}.")
        is_local = all(
            mask & (1 << int(value)) for mask, value in zip(masks, parameter_bins)
        )
        return "local" if is_local else "global"

    def has_local_choice(self, component: str) -> bool:
        mode = self.behavior_mode(component)
        if mode in {"local", "neutral"}:
            return True
        if mode == "global":
            return False
        return all(bool(mask) for mask in self.local_masks(component))

    def has_global_choice(self, component: str) -> bool:
        mode = self.behavior_mode(component)
        if mode == "global":
            return True
        if mode in {"local", "neutral"}:
            return False
        full_mask = (1 << 10) - 1
        return any(mask != full_mask for mask in self.local_masks(component))

    def secondary_search(self, problem_type: str) -> tuple[str, ...]:
        search = self.category_components("search", problem_type)
        if problem_type == "continuous":
            return tuple(name for name in search if name.startswith("search_mu_"))
        return tuple(name for name in search if name.startswith("search_"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "autooptlib.learning.vocabulary",
            "schema_version": 6,
            "choose": list(self.choose),
            "search": list(self.search),
            "update": list(self.update),
            "archives": list(self.archives),
            "parameter_counts": list(self.parameter_counts),
            "behavior_modes": list(self.behavior_modes),
            "local_bin_masks": [list(masks) for masks in self.local_bin_masks],
            "component_problem_types": [
                list(types) for types in self.component_problem_types
            ],
            "problem_type": self.problem_type,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LearningVocabulary":
        if value.get("schema") != "autooptlib.learning.vocabulary":
            raise ValueError("Not an AutoOptLib learning vocabulary.")
        version = value.get("schema_version", -1)
        if version != 6:
            raise ValueError(
                "This checkpoint predates the exact integer population/offspring "
                "encoding and must be retrained with vocabulary schema 6."
            )
        sequence_fields = (
            "choose",
            "search",
            "update",
            "archives",
            "parameter_counts",
            "behavior_modes",
            "local_bin_masks",
            "component_problem_types",
        )
        if any(
            not isinstance(value.get(name), (list, tuple)) for name in sequence_fields
        ):
            raise ValueError("Learning vocabulary fields must be arrays.")
        return cls(
            choose=tuple(value["choose"]),
            search=tuple(value["search"]),
            update=tuple(value["update"]),
            archives=tuple(value["archives"]),
            parameter_counts=tuple(value["parameter_counts"]),
            behavior_modes=tuple(value["behavior_modes"]),
            local_bin_masks=tuple(tuple(masks) for masks in value["local_bin_masks"]),
            component_problem_types=tuple(
                tuple(types) for types in value["component_problem_types"]
            ),
            problem_type=value["problem_type"],
        )

    @classmethod
    def from_space(
        cls,
        setting: Any,
        problem_type: str,
    ) -> "LearningVocabulary":
        all_op = tuple(getattr(setting, "AllOp", getattr(setting, "all_op", ())))
        op_space = np.asarray(
            getattr(setting, "OpSpace", getattr(setting, "op_space", ())), dtype=int
        )
        para_space = list(
            getattr(setting, "ParaSpace", getattr(setting, "para_space", ()))
        )
        behavior_space = list(
            getattr(
                setting,
                "BehavSpace",
                getattr(setting, "behav_space", [None] * len(all_op)),
            )
        )
        archives = tuple(
            str(name)
            for name in (
                getattr(setting, "Archive", getattr(setting, "archive", [])) or []
            )
        )
        if not all_op or op_space.shape != (3, 2):
            raise ValueError("Build the AutoOptLib design space before its vocabulary.")

        def section(row: int) -> tuple[str, ...]:
            lower, upper = op_space[row]
            return all_op[int(lower) - 1 : int(upper)]

        counts: list[int] = []
        modes: list[str] = []
        masks_per_component: list[tuple[int, ...]] = []
        for bounds, behavior in zip(para_space, behavior_space):
            if bounds is None:
                bounds_array = None
                counts.append(0)
            else:
                bounds_array = np.asarray(bounds, dtype=float).reshape(-1, 2)
                counts.append(int(bounds_array.shape[0]))
            has_local = _has_behavior(behavior, 0, "LS")
            has_global = _has_behavior(behavior, 1, "GS")
            if has_local and has_global:
                mode = "parameter"
            elif has_global:
                mode = "global"
            elif has_local:
                mode = "local"
            else:
                mode = "neutral"
            modes.append(mode)
            if mode != "parameter" or bounds_array is None:
                masks_per_component.append(())
                continue
            component_masks = []
            local_row = behavior[0] if behavior and len(behavior) > 0 else []
            global_row = behavior[1] if behavior and len(behavior) > 1 else []
            for parameter_index, _ in enumerate(bounds_array):
                mask = 0
                for bin_index in range(10):
                    normalized = bin_index / 9.0
                    local_trend = (
                        local_row[parameter_index + 1]
                        if len(local_row) > parameter_index + 1
                        else None
                    )
                    global_trend = (
                        global_row[parameter_index + 1]
                        if len(global_row) > parameter_index + 1
                        else None
                    )
                    local_distance = (
                        normalized
                        if local_trend == "small"
                        else 1.0 - normalized
                        if local_trend == "large"
                        else 0.5
                    )
                    global_distance = (
                        normalized
                        if global_trend == "small"
                        else 1.0 - normalized
                        if global_trend == "large"
                        else 0.5
                    )
                    if local_distance <= global_distance:
                        mask |= 1 << bin_index
                component_masks.append(mask)
            masks_per_component.append(tuple(component_masks))
        return cls(
            choose=section(0),
            search=section(1),
            update=section(2),
            archives=archives,
            parameter_counts=tuple(counts),
            behavior_modes=tuple(modes),
            local_bin_masks=tuple(masks_per_component),
            component_problem_types=tuple((str(problem_type),) for _ in all_op),
            problem_type=str(problem_type),
        )

    @classmethod
    def merge(
        cls, vocabularies: Sequence["LearningVocabulary"]
    ) -> "LearningVocabulary":
        """Create one typed vocabulary for compatible typed algorithm datasets."""

        if not vocabularies:
            raise ValueError("At least one vocabulary is required.")
        archives = vocabularies[0].archives
        if any(vocabulary.archives != archives for vocabulary in vocabularies[1:]):
            raise ValueError(
                "Merged vocabularies must use the same archive configuration."
            )
        categories: dict[str, list[str]] = {"choose": [], "search": [], "update": []}
        specs: dict[str, tuple[int, str, tuple[int, ...]]] = {}
        compatibility: dict[str, set[str]] = {}
        for vocabulary in vocabularies:
            for category in categories:
                for component in getattr(vocabulary, category):
                    if component not in categories[category]:
                        categories[category].append(component)
                    position = vocabulary.component_position(component)
                    spec = (
                        vocabulary.parameter_counts[position],
                        vocabulary.behavior_modes[position],
                        vocabulary.local_bin_masks[position],
                    )
                    if component in specs and specs[component] != spec:
                        raise ValueError(
                            f"Component {component!r} has incompatible parameter or behavior metadata."
                        )
                    specs[component] = spec
                    compatibility.setdefault(component, set()).update(
                        vocabulary.component_problem_types[position]
                    )
        components = tuple(
            categories["choose"] + categories["search"] + categories["update"]
        )
        return cls(
            choose=tuple(categories["choose"]),
            search=tuple(categories["search"]),
            update=tuple(categories["update"]),
            archives=archives,
            parameter_counts=tuple(specs[name][0] for name in components),
            behavior_modes=tuple(specs[name][1] for name in components),
            local_bin_masks=tuple(specs[name][2] for name in components),
            component_problem_types=tuple(
                tuple(
                    name
                    for name in ("continuous", "discrete", "permutation")
                    if name in compatibility[component]
                )
                for component in components
            ),
            problem_type="mixed",
        )


__all__ = [
    "LearningVocabulary",
    "PARAMETER_TOKENS",
    "STRUCTURAL_TOKENS",
]

"""Python translation of the MATLAB @DESIGN class."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable, Sequence

import numpy as np

from ._decode import decode as _decode
from ._disturb import disturb as _disturb
from ._evaluate import evaluate as _evaluate
from ._helpers import get_flex, problem_list
from ._initialize import initialize as _initialize
from ._population import (
    copy_configuration,
    default_configuration,
    repair_configuration,
    set_configuration,
)
from ._repair import repair as _repair

__all__ = ["Design"]


class Design:
    """Container for algorithm designs with methods mirroring MATLAB DESIGN."""

    def __init__(self, problem: Any | None = None, setting: Any | None = None):
        self.operator: list[Any] | None = None
        self.parameter: list[Any] | None = None
        self.operator_pheno: list[Any] | None = None
        self.parameter_pheno: list[Any] | None = None
        self.performance: np.ndarray = np.array([])
        self.last_runs: dict[int, list[Any]] = {}
        self.configuration: dict[str, int | float | str] | None = None
        # Search-controller state is deliberately kept outside serializable
        # algorithm metadata. It follows an incumbent through elitist
        # selection and is included in trusted pickle checkpoints.
        self.design_aux: Any = None

        if problem is not None and setting is not None:
            self._initialize_from_problem(problem, setting)

    # Static method wrappers -------------------------------------------------
    initialize = staticmethod(_initialize)
    repair = staticmethod(_repair)
    decode = staticmethod(_decode)

    # Instance methods -------------------------------------------------------
    def _initialize_from_problem(self, problem: Any, setting: Any) -> None:
        operators, paras = _initialize(setting, 1)
        operators, paras = _repair(operators, paras, problem, setting)
        op_pheno, para_pheno = _decode(operators, paras, problem, setting)

        self.operator = operators[0] if operators else []
        self.parameter = paras[0] if paras else []
        self.operator_pheno = op_pheno
        self.parameter_pheno = para_pheno
        set_configuration(self, default_configuration(setting, self))
        self._attach_program_contract(setting)

        self._reset_performance(problem, setting)

    def _reset_performance(self, problem: Any, setting: Any) -> None:
        count = len(problem_list(problem))
        runs = int(get_flex(setting, "AlgRuns", 1))
        self.performance = np.zeros((count, runs))
        self.last_runs = {index: [None] * runs for index in range(count)}

    @property
    def population_size(self) -> int | None:
        return (
            None
            if self.configuration is None
            else self.configuration["population_size"]
        )

    @property
    def offspring_size(self) -> int | None:
        return (
            None if self.configuration is None else self.configuration["offspring_size"]
        )

    @property
    def crossover_rate(self) -> float | None:
        return (
            None if self.configuration is None else self.configuration["crossover_rate"]
        )

    @property
    def mutation_rate(self) -> float | None:
        return (
            None if self.configuration is None else self.configuration["mutation_rate"]
        )

    @property
    def boundary_handling(self) -> str | None:
        return (
            None
            if self.configuration is None
            else self.configuration["boundary_handling"]
        )

    def get_new(self, problem: Any, setting: Any, inner_g: int, aux: Any):
        """Design new algorithms based on the current one (GetNew)."""
        new_ops, new_paras, aux_out = _disturb(self, setting, inner_g, aux)
        new_design = Design()
        operators, paras = _repair(new_ops, new_paras, problem, setting)
        op_pheno, para_pheno = _decode(operators, paras, problem, setting)

        new_design.operator = operators[0] if operators else []
        new_design.parameter = paras[0] if paras else []
        new_design.operator_pheno = op_pheno
        new_design.parameter_pheno = para_pheno
        set_configuration(
            new_design,
            repair_configuration(
                new_design,
                copy_configuration(self, setting),
                setting,
            ),
        )

        new_design._reset_performance(problem, setting)

        # disturb returns lists; for a single algorithm pick the first entry
        aux_result = (
            aux_out[0] if isinstance(aux_out, Sequence) and aux_out else aux_out
        )
        new_design.design_aux = deepcopy(aux_result)
        new_design._attach_program_contract(setting)
        return new_design, aux_result

    @classmethod
    def from_genotype(
        cls,
        operators: Any,
        parameters: Any,
        problem: Any,
        setting: Any,
        *,
        design_aux: Any = None,
        configuration: Any = None,
    ) -> "Design":
        """Build a repaired, decoded design from one graph genotype."""
        instance = cls()
        from ._parameter_bank import copy_bank
        repaired_ops, repaired_parameters = _repair(
            [[np.asarray(path, dtype=int).copy() for path in operators]],
            [copy_bank(parameters)],
            problem,
            setting,
        )
        op_pheno, para_pheno = _decode(
            repaired_ops, repaired_parameters, problem, setting
        )
        instance.operator = repaired_ops[0]
        instance.parameter = repaired_parameters[0]
        instance.operator_pheno = op_pheno
        instance.parameter_pheno = para_pheno
        set_configuration(
            instance,
            default_configuration(setting, instance)
            if configuration is None
            else configuration,
        )
        instance._reset_performance(problem, setting)
        instance.design_aux = deepcopy(design_aux)
        instance._attach_program_contract(setting)
        return instance

    def _attach_program_contract(self, setting):
        if (
            getattr(setting, "DesignSpace", None) is not None
            or getattr(setting, "ParameterScope", "component") == "node"
        ):
            from ...model_semantics import execution_spec

            self.program_contract = execution_spec(
                getattr(setting, "ComponentBackend", "python"),
                getattr(setting, "StreamBackend", None),
            ).to_dict()

    def get_performance(self, setting: Any, seed_instance: Iterable[int]) -> np.ndarray:
        """Return performance vector for given instances (GetPerformance)."""
        seeds = list(seed_instance)
        alg_runs = int(get_flex(setting, "AlgRuns", 1))
        if get_flex(setting, "Evaluate", "exact") != "exact":
            raise ValueError("Design performance supports exact evaluation only.")
        data = self.performance[seeds, :]
        return data.reshape(len(seeds) * alg_runs, 1, order="F")

    def construct(self, operator: Any, parameter: Any) -> None:
        """Set decoded operator/parameter phenotype (Construct)."""
        self.operator_pheno = operator
        self.parameter_pheno = parameter

    def ave_perform_all(self) -> np.ndarray:
        """Average actual performance across instances and runs."""
        if self.performance.size == 0:
            return np.array([])
        return np.array([float(np.mean(self.performance))])

    def ave_perform_per(self, indices: Iterable[int]) -> np.ndarray:
        """Average actual performance over selected instances."""
        if self.performance.size == 0:
            return np.array([])
        idx = np.atleast_1d(list(indices))
        return np.array([float(np.mean(self.performance[idx, :]))])

    # Bind external implementations as instance methods ---------------------
    disturb = _disturb
    evaluate = _evaluate

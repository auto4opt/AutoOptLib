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


def _get_alg_runs(setting: Any) -> int:
    return int(get_flex(setting, "AlgRuns", get_flex(setting, "alg_runs", 1)))


class Design:
    """Container for algorithm designs with methods mirroring MATLAB DESIGN."""

    def __init__(self, problem: Any | None = None, setting: Any | None = None):
        self.operator: list[Any] | None = None
        self.parameter: list[Any] | None = None
        self.operator_pheno: list[Any] | None = None
        self.parameter_pheno: list[Any] | None = None
        self.performance: np.ndarray = np.array([])
        self.performance_approx: np.ndarray = np.array([])
        self.last_runs: dict[int, list[Any]] = {}
        self.configuration: dict[str, int | float | str] | None = None
        self.population_size: int | None = None
        self.offspring_size: int | None = None
        self.crossover_rate: float | None = None
        self.mutation_rate: float | None = None
        self.boundary_handling: str | None = None
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

        problems = problem_list(problem)
        alg_runs = _get_alg_runs(setting)
        self.performance = np.zeros((len(problems), alg_runs))
        self.performance_approx = np.zeros((len(problems), alg_runs))
        self.last_runs = {idx: [None] * alg_runs for idx in range(len(problems))}

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

        problems = problem_list(problem)
        alg_runs = _get_alg_runs(setting)
        new_design.performance = np.zeros((len(problems), alg_runs))
        new_design.performance_approx = np.zeros((len(problems), alg_runs))
        new_design.last_runs = {idx: [None] * alg_runs for idx in range(len(problems))}

        # disturb returns lists; for a single algorithm pick the first entry
        aux_result = (
            aux_out[0] if isinstance(aux_out, Sequence) and aux_out else aux_out
        )
        new_design.design_aux = deepcopy(aux_result)
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
        repaired_ops, repaired_parameters = _repair(
            [[np.asarray(path, dtype=int).copy() for path in operators]],
            [[deepcopy(entry) for entry in parameters]],
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
        problems = problem_list(problem)
        alg_runs = _get_alg_runs(setting)
        instance.performance = np.zeros((len(problems), alg_runs))
        instance.performance_approx = np.zeros((len(problems), alg_runs))
        instance.last_runs = {idx: [None] * alg_runs for idx in range(len(problems))}
        instance.design_aux = deepcopy(design_aux)
        return instance

    def get_performance(self, setting: Any, seed_instance: Iterable[int]) -> np.ndarray:
        """Return performance vector for given instances (GetPerformance)."""
        seeds = list(seed_instance)
        alg_runs = _get_alg_runs(setting)
        if (
            get_flex(setting, "Evaluate", get_flex(setting, "evaluate", "exact"))
            == "approximate"
        ):
            data = self.performance_approx[seeds, :]
        else:
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

    def ave_perform_approx_all(self) -> np.ndarray:
        """Average approximate performance across instances and runs."""
        if self.performance_approx.size == 0:
            return np.array([])
        return np.array([float(np.mean(self.performance_approx))])

    def ave_perform_per(self, indices: Iterable[int]) -> np.ndarray:
        """Average actual performance over selected instances."""
        if self.performance.size == 0:
            return np.array([])
        idx = np.atleast_1d(list(indices))
        return np.array([float(np.mean(self.performance[idx, :]))])

    def ave_perform_approx_per(self, indices: Iterable[int]) -> np.ndarray:
        """Average approximate performance over selected instances."""
        if self.performance_approx.size == 0:
            return np.array([])
        idx = np.atleast_1d(list(indices))
        return np.array([float(np.mean(self.performance_approx[idx, :]))])

    # Bind external implementations as instance methods ---------------------
    disturb = _disturb
    evaluate = _evaluate

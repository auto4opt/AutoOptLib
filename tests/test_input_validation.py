"""Tests for public option parsing and validation failures."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from autooptlib.utils.general.input import (
    _check_setting,
    _ensure_namespace,
    _find_argument,
    _to_list,
    _to_sequence,
    input_handler,
    normalize_options,
)


def _design_setting(**overrides):
    values = dict(
        Mode="design",
        AlgP=1,
        AlgQ=3,
        ProbN=5,
        ProbFE=20,
        InnerFE=1,
        AlgN=2,
        AlgFE=4,
        AlgRuns=2,
        Metric="quality",
        Compare="average",
        Evaluate="exact",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _solve_setting(**overrides):
    values = dict(
        Mode="solve",
        ProbN=5,
        ProbFE=20,
        AlgRuns=1,
        AlgName="Continuous Random Search",
        Metric="quality",
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_option_helpers_cover_aliases_and_container_forms():
    assert normalize_options(
        {"graph_semantics": "stream_graph_v2", "search_initial_designs": []}
    ) == {"GraphSemantics": "stream_graph_v2", "SearchInitialDesigns": []}
    with pytest.raises(TypeError, match="supplied more than once"):
        normalize_options({"ProbFE": 10, "prob_fe": 20})
    with pytest.raises(TypeError, match="Unknown AutoOpt option"):
        normalize_options({1: 2})
    assert _to_sequence("x") == ("x",)
    assert _to_sequence([1, 2]) == [1, 2]
    assert _to_list((1, 2)) == [1, 2]
    assert _to_list(1) == [1]
    assert _ensure_namespace({"x": 1}).x == 1

    class Setting:
        x = 2

    assert _ensure_namespace(Setting()).x == 2
    assert _find_argument([], "x") == (False, None)
    assert _find_argument(["y", 1], "x") == (False, None)
    arguments = ["features", np.zeros(4), "AlgN", 3]
    assert _find_argument(arguments, "AlgN") == (True, 3)
    with pytest.raises(ValueError, match="Missing value"):
        _find_argument(["x"], "x")


def test_input_handler_data_and_defaults():
    with pytest.raises(ValueError, match="targeted problem"):
        input_handler([], SimpleNamespace(Mode="solve"), "data")
    with pytest.raises(ValueError, match="instance indexes"):
        input_handler(["Problem", "p"], SimpleNamespace(Mode="design"), "data")
    with pytest.raises(ValueError, match="instance indexes"):
        input_handler(["Problem", "p"], SimpleNamespace(Mode="solve"), "data")
    with pytest.raises(ValueError, match="mode"):
        input_handler(["Problem", "p"], SimpleNamespace(Mode="other"), "data")
    assert input_handler(
        ["Problem", "p", "InstanceSolve", (1, 2)], SimpleNamespace(Mode="solve"), "data"
    ) == ("p", [1, 2])
    setting = input_handler(["ProbN", 7], SimpleNamespace(Mode="solve"), "parameter")
    assert setting.ProbN == 7
    assert setting.ProbFE == 50000
    assert setting.ProbN == 7
    assert setting.AlgRuns == 5
    design_defaults = input_handler(
        ["InstanceTrain", [1, 2, 3, 4, 5]], SimpleNamespace(Mode="design"), "parameter"
    )
    assert design_defaults.AlgQ == 2
    assert design_defaults.InnerFE == 500
    assert design_defaults.AlgN == 10
    assert design_defaults.AlgFE == 5000
    assert design_defaults.AlgRuns == 5
    assert design_defaults.StructureMutationRate == 0.3
    assert design_defaults.ParameterCMAInitialSigma == 1.0
    assert design_defaults.SearchStagnation == 3
    assert design_defaults.SearchRestartFraction == 0.5
    assert design_defaults.SearchImprovementRate == 0.05
    assert design_defaults.SearchParameterActionProbability == 0.5
    assert design_defaults.ParameterCMAOffspring == 5
    assert design_defaults.ParameterCMABlockGenerations == 3
    assert design_defaults.PostStructureCMAGenerations == 1
    assert design_defaults.PopulationSizeSpace == list(range(4, 101))
    assert design_defaults.OffspringSizeSpace == list(range(1, 101))
    assert design_defaults.SearchMaxAttempts == 50
    assert design_defaults.Alpha == 0.05
    with pytest.raises(ValueError, match="Unsupported mode"):
        input_handler([], setting, "other")


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ProbN": 0}, "positive integer"),
        ({"AlgP": 4, "AlgQ": 3}, "pathways"),
        ({"AlgN": 5, "AlgFE": 4}, "algorithms"),
        ({"AlgRuns": 21}, "runs should not exceed"),
        ({"ProbN": 21}, "ProbFE must be at least"),
        ({"Evaluate": "unknown"}, "Evaluate=exact only"),
        ({"Compare": "unknown"}, "Compare must be"),
        ({"Metric": "unknown"}, "Metric must be"),
        ({"DesignFEs": 0}, "DesignFEs must be"),
        ({"Evaluate": "intensification"}, "Evaluate=exact only"),
        (
            {"Evaluate": "intensification", "AlgN": 1, "Compare": "statistic"},
            "Evaluate=exact only",
        ),
        (
            {"Evaluate": "intensification", "AlgN": 1},
            "Evaluate=exact only",
        ),
        (
            {"Evaluate": "racing", "Compare": "statistic"},
            "Evaluate=exact only",
        ),
        (
            {"Evaluate": "racing", "Compare": "statistic"},
            "Evaluate=exact only",
        ),
        ({"Evaluate": "racing", "Alpha": 1.0}, "Evaluate=exact only"),
        (
            {"Evaluate": "racing"},
            "Evaluate=exact only",
        ),
        ({"Evaluate": "approximate"}, "Evaluate=exact only"),
        ({"Evaluate": "approximate"}, "Evaluate=exact only"),
        (
            {"Evaluate": "approximate", "Compare": "statistic"},
            "Evaluate=exact only",
        ),
        ({"Compare": "statistic", "AlgRuns": 1}, "statistical blocks"),
        ({"Designer": "aldes"}, "Designer must be"),
        ({"Designer": "learning"}, "Designer must be"),
        ({"StructureMutationRate": 0}, "StructureMutationRate"),
        ({"ParameterCMAInitialSigma": 0}, "ParameterCMAInitialSigma"),
        ({"SearchStagnation": 0}, "SearchStagnation"),
        ({"SearchRestartFraction": 0}, "SearchRestartFraction"),
        ({"SearchImprovementRate": 1.1}, "SearchImprovementRate"),
        ({"SearchParameterActionProbability": 1.1}, "SearchParameterActionProbability"),
        ({"ParameterCMAOffspring": 0}, "ParameterCMAOffspring"),
        ({"ParameterCMABlockGenerations": 0}, "ParameterCMABlockGenerations"),
        ({"PostStructureCMAGenerations": 0}, "PostStructureCMAGenerations"),
        ({"SearchMaxAttempts": 0}, "SearchMaxAttempts"),
    ],
)
def test_design_validation_rejects_invalid_combinations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _check_setting(_design_setting(**overrides))


def test_exact_allows_multiple_training_tasks_as_statistical_blocks():
    setting = _design_setting(Evaluate="exact", Compare="statistic", AlgRuns=1)
    setting._DesignStatisticalBlocks = 6
    _check_setting(setting)
    assert setting.AlgRuns == 1


def test_design_validation_normalizes_values_and_warns_on_weak_settings():
    setting = _design_setting(Evaluate="EXACT", Compare="AVERAGE")
    _check_setting(setting)
    assert setting.Evaluate == "exact"
    assert setting.Compare == "average"
    random_setting = _design_setting(Designer="RANDOM")
    with pytest.raises(ValueError, match="Designer must be"):
        _check_setting(random_setting)
    with pytest.warns(UserWarning, match="large population"):
        _check_setting(_design_setting(ProbN=4, AlgP=2, AlgQ=1))
    with pytest.warns(UserWarning, match="recommended"):
        _check_setting(_design_setting(AlgQ=5))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"ProbFE": 0}, "positive integer"),
        ({"ProbN": 21}, "ProbFE must be at least"),
        ({"AlgName": None}, "specify an algorithm"),
        ({"Metric": "unknown"}, "Metric must be"),
        ({"Metric": "runtimeFE", "Thres": None}, "Setting.Thres"),
        ({"Metric": "runtimeSec", "Tmax": None, "Thres": 0}, "Setting.Tmax"),
        ({"Metric": "runtimeSec", "Tmax": 1, "Thres": None}, "Setting.Thres"),
        ({"Metric": "auc", "Tmax": [1], "Thres": [0]}, "multiple time points"),
        ({"Metric": "auc", "Tmax": [1, 2], "Thres": [0]}, "number of thresholds"),
    ],
)
def test_solve_validation_rejects_invalid_combinations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _check_setting(_solve_setting(**overrides))


def test_solve_runtimefe_defaults_tmax_and_accepts_valid_auc():
    runtime = _solve_setting(Metric="runtimeFE", Tmax=None, Thres=0)
    _check_setting(runtime)
    assert runtime.Tmax == runtime.ProbFE
    _check_setting(_solve_setting(Metric="auc", Tmax=[5, 10], Thres=[2, 1]))


def test_check_handler_and_invalid_mode():
    setting = _solve_setting()
    assert input_handler([], setting, "check") is setting
    with pytest.raises(ValueError, match="mode"):
        _check_setting(SimpleNamespace(Mode="other"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"EvalRetries": -1}, "non-negative"),
        ({"EvalTimeoutSec": 0}, "positive finite"),
        ({"EvalFailure": "ignore"}, "raise.*penalize"),
        ({"EvalPenalty": float("inf")}, "finite numeric"),
        ({"StreamEventBudgetMultiplier": 0}, "positive finite"),
        ({"StreamEventBudgetMultiplier": True}, "positive finite"),
        ({"EvalCache": 1}, "boolean"),
        ({"EvalLog": 4}, "filesystem path"),
        ({"EvalWorkers": 0}, "positive integer"),
        ({"EvalBackend": "gpu"}, "EvalBackend"),
        ({"EvalBackend": "thread"}, "EvalBackend"),
        ({"EvalCoresPerWorker": 0}, "positive integer"),
        (
            {"EvalCoresPerWorker": 2, "EvalMaxCoresPerTask": 1},
            "at least EvalCoresPerWorker",
        ),
        ({"EvalAffinity": 1}, "boolean"),
        ({"EvalTaskTimeoutSec": 0}, "positive and finite"),
        ({"EvalAdaptive": 1}, "boolean"),
        ({"EvalMemoryLimitBytes": 0}, "positive or None"),
        ({"EvalMaxOversubscription": 0.5}, "at least 1"),
        ({"EvalIOCPUThreshold": 1}, "between 0 and 1"),
        ({"EvalSharedMemoryThreshold": -1}, "non-negative"),
        ({"EvalResourceEstimator": 1}, "callable"),
        ({"EvalWorkerInitializer": 1}, "callable"),
        ({"EvalWorkerFinalizer": 1}, "callable"),
        (
            {"EvalWorkerFinalizer": lambda *_args: None},
            "requires EvalWorkerInitializer",
        ),
        ({"CheckpointDir": 4}, "filesystem path"),
        ({"CheckpointEvery": 0}, "positive integer"),
        ({"Resume": 1}, "boolean"),
    ],
)
def test_evaluation_policy_validation(overrides, message):
    with pytest.raises(ValueError, match=message):
        _check_setting(_solve_setting(**overrides))


def test_checkpointing_is_accepted_in_design_mode():
    setting = _design_setting(CheckpointDir="checkpoints")
    _check_setting(setting)
    assert setting.CheckpointDir == "checkpoints"


def test_process_workers_can_be_combined_with_objective_timeout():
    setting = _solve_setting(EvalWorkers=2, EvalTimeoutSec=1)
    _check_setting(setting)
    assert setting.EvalWorkers == 2
    assert setting.EvalTimeoutSec == 1


def test_worker_lifecycle_options_are_accepted():

    def initializer(config, context):
        return (config, context)

    def finalizer(environment, context):
        return None

    setting = _solve_setting(
        EvalWorkerInitializer=initializer,
        EvalWorkerFinalizer=finalizer,
        EvalWorkerConfig={"endpoint": "local"},
    )
    _check_setting(setting)
    assert setting.EvalWorkerInitializer is initializer
    assert setting.EvalWorkerFinalizer is finalizer


def test_resource_aware_runtime_options_are_accepted():

    def estimator(task):
        return {"cores": 2, "profile_key": type(task).__name__}

    setting = _solve_setting(
        EvalAdaptive=True,
        EvalMaxCoresPerTask=2,
        EvalMemoryLimitBytes=1000000,
        EvalMaxOversubscription=3,
        EvalIOCPUThreshold=0.4,
        EvalSharedMemoryThreshold=4096,
        EvalResourceEstimator=estimator,
    )
    _check_setting(setting)
    assert setting.EvalResourceEstimator is estimator


@pytest.mark.parametrize(
    "name",
    [
        "SearchArchiveSize",
        "RacingK",
        "StructureRacingMinFraction",
        "IntensificationMaxConfigCalls",
        "Surro",
    ],
)
def test_retired_evaluation_options_are_not_public_configuration(name):
    with pytest.raises(TypeError, match="Unknown AutoOpt option"):
        normalize_options({name: 1})

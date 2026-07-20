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
        ["Problem", "p", "InstanceSolve", (1, 2)],
        SimpleNamespace(Mode="solve"),
        "data",
    ) == ("p", [1, 2])
    setting = input_handler(["ProbN", 7], SimpleNamespace(Mode="solve"), "parameter")
    assert setting.ProbN == 7
    assert setting.ProbFE == 50000
    assert setting.ProbN == 7
    assert setting.AlgRuns == 5

    design_defaults = input_handler(
        ["InstanceTrain", [1, 2, 3, 4, 5]],
        SimpleNamespace(Mode="design"),
        "parameter",
    )
    assert design_defaults.AlgQ == 4
    assert design_defaults.InnerFE == 500
    assert design_defaults.AlgN == 10
    assert design_defaults.AlgFE == 5000
    assert design_defaults.AlgRuns == 5
    assert design_defaults.ALDesMode == "single"
    assert design_defaults.RacingK == 1
    assert design_defaults.Surro == 1500
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
        ({"Evaluate": "unknown"}, "Evaluate must be"),
        ({"Compare": "unknown"}, "Compare must be"),
        ({"Metric": "unknown"}, "Metric must be"),
        ({"Evaluate": "racing"}, "should be used with"),
        (
            {"Evaluate": "racing", "Compare": "statistic", "RacingK": None},
            "Setting.K",
        ),
        ({"Evaluate": "approximate"}, "Setting.Surro"),
        ({"Evaluate": "approximate", "Surro": 5}, "Surro must be"),
        (
            {"Evaluate": "approximate", "Surro": 2, "Compare": "statistic"},
            "not necessary",
        ),
        ({"Compare": "statistic", "AlgRuns": 1}, "run the design multiple"),
        ({"ALDesMode": "unknown"}, "ALDesMode"),
    ],
)
def test_design_validation_rejects_invalid_combinations(overrides, message):
    with pytest.raises(ValueError, match=message):
        _check_setting(_design_setting(**overrides))


def test_design_validation_normalizes_values_and_warns_on_weak_settings():
    setting = _design_setting(Evaluate="EXACT", Compare="AVERAGE")
    _check_setting(setting)
    assert setting.Evaluate == "exact"
    assert setting.Compare == "average"
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
        (
            {"Metric": "auc", "Tmax": [1, 2], "Thres": [0]},
            "number of thresholds",
        ),
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
        ({"EvalCache": 1}, "boolean"),
        ({"EvalLog": 4}, "filesystem path"),
        ({"EvalWorkers": 0}, "positive integer"),
        (
            {"EvalWorkers": 2, "EvalTimeoutSec": 1},
            "cannot currently be combined",
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

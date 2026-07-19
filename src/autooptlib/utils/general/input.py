"""Translation of MATLAB Utilities/General/Input.m."""

from __future__ import annotations

import math
import os
import re
import warnings
from copy import deepcopy
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

_PARAM_KEYS = {
    "AlgP",
    "AlgQ",
    "Archive",
    "LSRange",
    "IncRate",
    "ProbN",
    "ProbFE",
    "InnerFE",
    "AlgN",
    "AlgFE",
    "AlgRuns",
    "Metric",
    "Compare",
    "Evaluate",
    "Tmax",
    "Thres",
    "RacingK",
    "Surro",
    "AlgFile",
    "AlgName",
    "Seed",
    "TunePara",
    "EvalRetries",
    "EvalTimeoutSec",
    "EvalFailure",
    "EvalPenalty",
    "EvalCache",
    "EvalLog",
    "EvalWorkers",
    "CheckpointDir",
    "CheckpointEvery",
    "Resume",
}

_DATA_KEYS = {"Mode", "Problem", "InstanceTrain", "InstanceTest", "InstanceSolve"}
_PUBLIC_KEYS = _DATA_KEYS | _PARAM_KEYS | {"OutputDir"}
_KEY_LOOKUP = {re.sub(r"[^a-z0-9]", "", key.lower()): key for key in _PUBLIC_KEYS}

_COMMON_DEFAULTS = {
    "AlgP": 1,
    "AlgQ": 4,
    "Archive": [],
    "LSRange": 0.25,
    "IncRate": 0.05,
    "Metric": "quality",
    "Compare": "average",
    "Evaluate": "exact",
    "TunePara": False,
    "Seed": None,
    "EvalRetries": 0,
    "EvalTimeoutSec": None,
    "EvalFailure": "raise",
    "EvalPenalty": 1e30,
    "EvalCache": False,
    "EvalLog": None,
    "EvalWorkers": 1,
    "CheckpointDir": None,
    "CheckpointEvery": 1,
    "Resume": False,
}

_DESIGN_DEFAULTS = {
    "ProbN": 20,
    "ProbFE": 5000,
    "InnerFE": 500,
    "AlgN": 10,
    "AlgFE": 5000,
    "AlgRuns": 5,
    "Tmax": None,
    "Thres": None,
}

_SOLVE_DEFAULTS = {
    "ProbN": 50,
    "ProbFE": 50000,
    "AlgRuns": 5,
    "AlgFile": "",
    "AlgName": "",
    "Tmax": None,
    "Thres": None,
}


def normalize_options(options: dict[str, Any]) -> dict[str, Any]:
    """Normalize public keyword spellings and reject unknown options.

    Public keys are case-insensitive and may use underscores, so ``archive``,
    ``Archive``, and ``ARCHIVE`` all resolve to ``Archive``.  Rejecting unknown
    keys prevents experiments from silently running with ignored settings.
    """
    normalized: dict[str, Any] = {}
    unknown: list[str] = []
    for key, value in options.items():
        if not isinstance(key, str):
            unknown.append(repr(key))
            continue
        token = re.sub(r"[^a-z0-9]", "", key.lower())
        canonical = _KEY_LOOKUP.get(token)
        if canonical is None:
            unknown.append(key)
            continue
        if canonical in normalized:
            raise TypeError(
                f"AutoOpt option {canonical!r} was supplied more than once through aliases."
            )
        normalized[canonical] = value
    if unknown:
        supported = ", ".join(sorted(_PUBLIC_KEYS))
        raise TypeError(
            f"Unknown AutoOpt option(s): {', '.join(unknown)}. Supported options: {supported}."
        )
    return normalized


def _to_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return value
    return (value,)


def _ensure_namespace(setting: Any) -> SimpleNamespace:
    if isinstance(setting, SimpleNamespace):
        return setting
    if isinstance(setting, dict):
        return SimpleNamespace(**setting)
    data = {
        name: getattr(setting, name)
        for name in dir(setting)
        if not name.startswith("__") and not callable(getattr(setting, name))
    }
    return SimpleNamespace(**data)


def _find_argument(arguments: Sequence[Any], name: str) -> tuple[bool, Any]:
    if not arguments:
        return False, None
    try:
        idx = arguments.index(name)
    except ValueError:
        return False, None
    if idx + 1 >= len(arguments):
        raise ValueError(f'Missing value for argument "{name}"')
    return True, arguments[idx + 1]


def _to_list(obj: Any) -> list[Any]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, tuple):
        return list(obj)
    return [obj]


def input_handler(arguments: Iterable[Any], setting: Any, mode: str):
    """Reimplementation of the MATLAB Input.m logic."""
    args = list(arguments)
    set_ns = _ensure_namespace(setting)

    if mode == "data":
        if "Problem" not in args:
            raise ValueError("Please set the targeted problem.")
        _, problem = _find_argument(args, "Problem")
        if set_ns.Mode == "design":
            if "InstanceTrain" not in args or "InstanceTest" not in args:
                raise ValueError("Please set the targeted problem instance indexes.")
            _, train = _find_argument(args, "InstanceTrain")
            _, test = _find_argument(args, "InstanceTest")
            return problem, _to_list(train), _to_list(test)

        if set_ns.Mode == "solve":
            if "InstanceSolve" not in args:
                raise ValueError("Please set the targeted problem instance indexes.")
            _, solve = _find_argument(args, "InstanceSolve")
            return problem, _to_list(solve)

        raise ValueError('Please set the mode to "design" or "solve".')

    if mode == "parameter":
        for key in _PARAM_KEYS:
            found, value = _find_argument(args, key)
            if found:
                setattr(set_ns, key, value)
        defaults = dict(_COMMON_DEFAULTS)
        defaults.update(
            _DESIGN_DEFAULTS if set_ns.Mode == "design" else _SOLVE_DEFAULTS
        )
        if set_ns.Mode == "design":
            train_found, train = _find_argument(args, "InstanceTrain")
            train_count = len(_to_list(train)) if train_found else 1
            defaults["RacingK"] = max(1, int(round(train_count * 0.2)))
            prob_fe: Any = getattr(set_ns, "ProbFE", defaults["ProbFE"])
            defaults["Surro"] = max(1, int(math.floor(float(prob_fe) * 0.3 + 0.5)))
        for key, value in defaults.items():
            if not hasattr(set_ns, key):
                setattr(set_ns, key, deepcopy(value))
        return set_ns

    if mode == "check":
        _check_setting(set_ns)
        return set_ns

    raise ValueError(f"Unsupported mode: {mode}")


def _check_setting(setting: SimpleNamespace) -> None:
    mode = getattr(setting, "Mode", None)
    if mode not in {"design", "solve"}:
        raise ValueError('Please set the mode to "design" or "solve".')

    retries = getattr(setting, "EvalRetries", 0)
    if not isinstance(retries, Integral) or isinstance(retries, bool) or retries < 0:
        raise ValueError("EvalRetries must be a non-negative integer.")
    timeout = getattr(setting, "EvalTimeoutSec", None)
    if timeout is not None and (
        not isinstance(timeout, Real)
        or isinstance(timeout, bool)
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise ValueError("EvalTimeoutSec must be a positive finite number or None.")
    failure = str(getattr(setting, "EvalFailure", "raise")).lower()
    if failure not in {"raise", "penalize"}:
        raise ValueError("EvalFailure must be 'raise' or 'penalize'.")
    setting.EvalFailure = failure
    penalty = getattr(setting, "EvalPenalty", 1e30)
    if (
        not isinstance(penalty, Real)
        or isinstance(penalty, bool)
        or not math.isfinite(float(penalty))
    ):
        raise ValueError("EvalPenalty must be a finite numeric scalar.")
    if not isinstance(getattr(setting, "EvalCache", False), bool):
        raise ValueError("EvalCache must be a boolean.")
    eval_log = getattr(setting, "EvalLog", None)
    if eval_log is not None and not isinstance(eval_log, (str, os.PathLike)):
        raise ValueError("EvalLog must be a filesystem path or None.")
    eval_workers = getattr(setting, "EvalWorkers", 1)
    if (
        not isinstance(eval_workers, Integral)
        or isinstance(eval_workers, bool)
        or int(eval_workers) <= 0
    ):
        raise ValueError("EvalWorkers must be a positive integer.")
    if int(eval_workers) > 1 and timeout is not None:
        raise ValueError(
            "EvalWorkers greater than 1 cannot currently be combined with "
            "EvalTimeoutSec; use one isolation mechanism at a time."
        )
    checkpoint_dir = getattr(setting, "CheckpointDir", None)
    if checkpoint_dir is not None and not isinstance(
        checkpoint_dir, (str, os.PathLike)
    ):
        raise ValueError("CheckpointDir must be a filesystem path or None.")
    checkpoint_every = getattr(setting, "CheckpointEvery", 1)
    if (
        not isinstance(checkpoint_every, Integral)
        or isinstance(checkpoint_every, bool)
        or checkpoint_every <= 0
    ):
        raise ValueError("CheckpointEvery must be a positive integer.")
    if not isinstance(getattr(setting, "Resume", False), bool):
        raise ValueError("Resume must be a boolean.")
    if mode == "design":
        for name in (
            "AlgP",
            "AlgQ",
            "ProbN",
            "ProbFE",
            "InnerFE",
            "AlgN",
            "AlgFE",
            "AlgRuns",
        ):
            value = getattr(setting, name, None)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if getattr(setting, "AlgP", 0) > 1 and getattr(setting, "AlgQ", 0) > 1:
            raise ValueError(
                "Use either multiple pathways (AlgP>1) or multiple search "
                "operators per pathway (AlgQ>1), not both."
            )
        if getattr(setting, "AlgN", 0) > getattr(setting, "AlgFE", 0):
            raise ValueError(
                "The number of algorithms should not be larger than the evaluation budget."
            )
        if getattr(setting, "AlgRuns", 0) > getattr(setting, "ProbFE", 0):
            raise ValueError(
                "The number of runs should not exceed problem evaluations."
            )
        if getattr(setting, "ProbN", 0) > getattr(setting, "ProbFE", 0):
            raise ValueError(
                "ProbFE must be at least ProbN so the initial population fits the budget."
            )

        evaluate = str(getattr(setting, "Evaluate", "exact")).lower()
        compare = str(getattr(setting, "Compare", "average")).lower()
        metric_token = str(getattr(setting, "Metric", "quality")).lower()
        metric_names = {
            "quality": "quality",
            "runtimefe": "runtimeFE",
            "runtimesec": "runtimeSec",
            "auc": "auc",
        }
        if evaluate not in {"exact", "racing", "intensification", "approximate"}:
            raise ValueError(
                "Evaluate must be one of: exact, racing, intensification, approximate."
            )
        if compare not in {"average", "statistic"}:
            raise ValueError("Compare must be 'average' or 'statistic'.")
        if metric_token not in metric_names:
            raise ValueError(
                "Metric must be one of: quality, runtimeFE, runtimeSec, auc."
            )
        metric = metric_names[metric_token]
        setting.Evaluate = evaluate
        setting.Compare = compare
        setting.Metric = metric
        if evaluate == "racing" and compare != "statistic":
            raise ValueError(
                'The "racing" evaluation method should be used with the algorithm comparing method of "statistic".'
            )
        if evaluate == "racing" and not getattr(setting, "RacingK", None):
            raise ValueError(
                'Please set "Setting.K" as the number of instances evaluated before the first round of racing.'
            )
        if evaluate == "approximate" and not getattr(setting, "Surro", None):
            raise ValueError(
                'Please set "Setting.Surro" as the number of exact performance evaluations when using surrogate.'
            )
        if evaluate == "approximate":
            surrogate_budget = getattr(setting, "Surro")
            if (
                not isinstance(surrogate_budget, Integral)
                or isinstance(surrogate_budget, bool)
                or surrogate_budget <= 0
                or surrogate_budget > setting.AlgFE
            ):
                raise ValueError(
                    "Surro must be a positive integer no larger than AlgFE."
                )
        if evaluate == "approximate" and compare == "statistic":
            raise ValueError(
                'It is not necessary to use the "statistic" algorithm comparing method when using the "approximate" evaluation method.'
            )
        if compare == "statistic" and getattr(setting, "AlgRuns", 1) == 1:
            raise ValueError(
                'Please run the design multiple times (Setting.AlgRuns>1) when using the "statistic" comparsion method.'
            )
        if getattr(setting, "ProbN", 0) < 5 and getattr(setting, "AlgP", 0) > 1:
            warnings.warn(
                "It is better to have a large population size if involving the EDA operator",
                stacklevel=2,
            )
        if getattr(setting, "AlgQ", 0) > 4:
            warnings.warn(
                "AlgQ is recommended to be larger than 4 for discrete and permutation problems due to the lack of so many search operators",
                stacklevel=2,
            )
    elif mode == "solve":
        for name in ("ProbN", "ProbFE", "AlgRuns"):
            value = getattr(setting, name, None)
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if setting.ProbN > setting.ProbFE:
            raise ValueError(
                "ProbFE must be at least ProbN so the initial population fits the budget."
            )
        if not getattr(setting, "AlgFile", None) and not getattr(
            setting, "AlgName", None
        ):
            raise ValueError(
                "Please specify an algorithm file in Setting.AlgFile or specify an algorithm name in Setting.AlgName."
            )

        metric_token = str(getattr(setting, "Metric", "quality")).lower()
        metric_names = {
            "quality": "quality",
            "runtimefe": "runtimeFE",
            "runtimesec": "runtimeSec",
            "auc": "auc",
        }
        if metric_token not in metric_names:
            raise ValueError(
                "Metric must be one of: quality, runtimeFE, runtimeSec, auc."
            )
        metric = metric_names[metric_token]
        setting.Metric = metric
        if metric == "runtimeFE":
            if getattr(setting, "Tmax", None) in (None, []):
                setting.Tmax = getattr(setting, "ProbFE", None)
            if getattr(setting, "Thres", None) in (None, []):
                raise ValueError(
                    'Please set "Setting.Thres" as the lowest acceptable performance of the design algorithms, '
                    "the performance can be the solution quality."
                )
        if metric == "runtimeSec":
            if getattr(setting, "Tmax", None) in (None, []):
                raise ValueError(
                    'Please set "Setting.Tmax" as the maximum runtime (seconds).'
                )
            if getattr(setting, "Thres", None) in (None, []):
                raise ValueError(
                    'Please set "Setting.Thres" as the lowest acceptable performance of the design algorithms, '
                    "the performance can be the solution quality."
                )
        if metric == "auc":
            tmax = getattr(setting, "Tmax", None)
            thres = getattr(setting, "Thres", None)
            if not isinstance(tmax, Sequence) or len(tmax) <= 1:
                raise ValueError(
                    '"Setting.Tmax" should contain multiple time points. The time points should the numbers of '
                    "function evaluations spent during the alorithm execution."
                )
            if not isinstance(thres, Sequence) or len(thres) != len(tmax):
                raise ValueError(
                    'The number of thresholds in "Setting.Thres" should be equal to the number of time points in '
                    '"Setting.Tmax". "Setting.Thres" refers to the lowest acceptable performance of the design '
                    "algorithms, the performance can be the solution quality."
                )

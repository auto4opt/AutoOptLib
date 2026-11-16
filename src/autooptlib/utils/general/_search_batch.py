"""Ordered Search action evaluation with one owned concurrency lifecycle."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock
from typing import Any, Callable, Sequence

from ..design import Design
from ..design._search_actions import (
    PARAMETER,
    STRUCTURE,
    ParameterSamplingExhausted,
    StructureNoveltyExhausted,
    available_actions,
    choose_action,
    execute_action,
    parameter_action_evaluation_cost,
    structure_action_evaluation_cost,
    supports_parameter_action,
)
from ..design._stream_graph import STREAM_GRAPH_SEMANTICS


@dataclass(frozen=True)
class SearchActionBatch:
    """Share the run context; keep plans, threads, queues and cancellation local."""

    problems: Any
    setting: Any
    evaluate: Callable[[Sequence[Design]], None]
    cost: Callable[[Design], float]
    seen_structures: set[str]
    seen_representations: set[str]

    def execute(
        self,
        parents: Sequence[Design],
        *,
        candidate_budget: int,
        maximum_actions: int,
        parameter_budget_fraction: float,
    ) -> tuple[list[dict[str, Any]], int]:
        """Run independent parent actions in synchronized evaluation waves.

        Action state transitions remain serialized in parent order, which
        keeps RNG use and global novelty reservations deterministic. Every
        time an action reaches an expensive evaluation, it yields through
        a small thread-backed broker. The coordinator combines all yielded
        candidates into one runtime call, so parents at the same CMA stage
        use the worker pool concurrently without sharing mutable CMA state.
        """
        setting_ns = self.setting
        problems = self.problems
        rng = setting_ns.rng
        design_fe_limit = getattr(setting_ns, "DesignFEs", None)
        maximum_attempts = int(getattr(setting_ns, "SearchMaxAttempts", 50))
        seen_structure_keys = self.seen_structures
        seen_keys = self.seen_representations
        evaluate_action_batch = self.evaluate
        parent_values = list(parents)
        limit = min(len(parent_values), max(0, int(maximum_actions)))
        if limit <= 0 or candidate_budget <= 0:
            return ([], 0)
        action_costs = {
            STRUCTURE: structure_action_evaluation_cost(setting_ns),
            PARAMETER: parameter_action_evaluation_cost(setting_ns),
        }
        stream_partial_structure = (
            str(getattr(setting_ns, "GraphSemantics", "")).lower()
            == STREAM_GRAPH_SEMANTICS
            and design_fe_limit is None
        )
        executor = ThreadPoolExecutor(
            max_workers=limit, thread_name_prefix="search-generation-action"
        )
        transition_lock = Lock()
        plans: list[dict[str, Any]] = []
        remaining = int(candidate_budget)

        def action_worker(plan: dict[str, Any]) -> None:
            transition_lock.acquire()
            transition_lock_held = True

            def broker_evaluate(candidates: Sequence[Design]) -> None:
                nonlocal transition_lock_held
                error = plan.get("evaluation_error")
                if error is not None:
                    raise error
                values = list(candidates)
                plan["status"].put(("request", values))
                transition_lock.release()
                transition_lock_held = False
                plan["release"].wait()
                plan["release"].clear()
                transition_lock.acquire()
                transition_lock_held = True
                error = plan.get("evaluation_error")
                if error is not None:
                    raise error
                plan["completed_evaluation_cost"] += len(values)

            try:
                attempted_actions: set[str] = set()
                while True:
                    action = str(plan["action"])
                    attempted_actions.add(action)
                    try:
                        result = execute_action(
                            action,
                            plan["parent"],
                            problems,
                            setting_ns,
                            broker_evaluate,
                            self.cost,
                            seen_structures=seen_structure_keys,
                            seen_representations=seen_keys,
                            maximum_attempts=maximum_attempts,
                            remaining_evaluations=plan["remaining_evaluations"],
                        )
                        break
                    except StructureNoveltyExhausted as error:
                        plan["novelty_attempts"] += error.attempts
                        if plan["completed_evaluation_cost"]:
                            raise RuntimeError(
                                "Structure novelty failed after the macro-action had already evaluated candidates; falling back would corrupt the exact candidate budget."
                            ) from error
                        if (
                            PARAMETER in attempted_actions
                            or not supports_parameter_action(plan["parent"], setting_ns)
                        ):
                            if getattr(setting_ns,"DesignSpace",None) is not None:
                                plan["exhausted"] = True
                                return
                            raise RuntimeError(
                                "Neither Search action could produce a complete candidate block for this parent."
                            ) from error
                        plan["action"] = PARAMETER
                    except ParameterSamplingExhausted as error:
                        if plan["completed_evaluation_cost"]:
                            raise RuntimeError(
                                "CMA sampling failed after the macro-action had already evaluated candidates; falling back would corrupt the exact candidate budget."
                            ) from error
                        if (
                            STRUCTURE in attempted_actions
                            or (
                                not stream_partial_structure
                                and structure_action_evaluation_cost(setting_ns)
                                > plan["remaining_evaluations"]
                            )
                            or plan["remaining_evaluations"] <= 0
                        ):
                            if getattr(setting_ns,"DesignSpace",None) is not None:
                                plan["exhausted"] = True
                                return
                            raise RuntimeError(
                                "Neither Search action could produce a complete candidate block for this parent."
                            ) from error
                        plan["parameter_sampling_fallback"] = True
                        plan["action"] = STRUCTURE
                plan["result"] = result
            except BaseException as error:
                plan["exception"] = error
            finally:
                if transition_lock_held:
                    transition_lock.release()
                plan["status"].put(("done", None))

        def abort_pending(error: BaseException) -> None:
            for plan in plans:
                if plan.get("result") is None and plan.get("exception") is None:
                    plan["evaluation_error"] = error
                if plan.get("waiting", False):
                    plan["release"].set()

        try:
            for parent in parent_values[:limit]:
                if design_fe_limit is not None and remaining < max(
                    action_costs.values()
                ):
                    break
                if remaining <= 0:
                    break
                available = list(available_actions(parent, setting_ns))
                available = [
                    action
                    for action in available
                    if action_costs[action] <= remaining
                    or (stream_partial_structure and action == STRUCTURE)
                ]
                if (
                    not available
                    and supports_parameter_action(parent, setting_ns)
                    and (remaining > 0)
                ):
                    available = [PARAMETER]
                if not available:
                    if getattr(setting_ns,"DesignSpace",None) is not None:
                        plans.append({"parent":parent,"exhausted":True})
                        continue
                    raise RuntimeError(
                        "The remaining Search candidate budget cannot be filled because this parent has no executable action."
                    )
                action = choose_action(
                    available,
                    rng,
                    parameter_budget_fraction=parameter_budget_fraction,
                    parameter_evaluation_cost=action_costs[PARAMETER],
                    structure_evaluation_cost=action_costs[STRUCTURE],
                )
                plan: dict[str, Any] = {
                    "parent": parent,
                    "requested_action": action,
                    "action": action,
                    "remaining_evaluations": remaining,
                    "status": Queue(),
                    "release": Event(),
                    "waiting": False,
                    "novelty_attempts": 0,
                    "parameter_sampling_fallback": False,
                    "completed_evaluation_cost": 0,
                }
                plans.append(plan)
                plan["future"] = executor.submit(action_worker, plan)
                (status, candidates) = plan["status"].get()
                if status != "request":
                    if plan.get("exhausted"):
                        continue
                    exception = plan.get("exception")
                    if exception is None:
                        raise RuntimeError(
                            "A Search action completed without evaluating candidates."
                        )
                    raise exception
                plan["request"] = candidates
                plan["waiting"] = True
                reserved = (
                    (
                        min(action_costs[STRUCTURE], remaining)
                        if stream_partial_structure
                        else action_costs[STRUCTURE]
                    )
                    if plan["action"] == STRUCTURE
                    else min(action_costs[PARAMETER], remaining)
                )
                plan["reserved_cost"] = int(reserved)
                remaining -= int(reserved)
            while any((plan.get("waiting", False) for plan in plans)):
                wave = [plan for plan in plans if plan.get("waiting", False)]
                candidates = [
                    candidate for plan in wave for candidate in plan["request"]
                ]
                try:
                    evaluate_action_batch(candidates)
                except BaseException as error:
                    abort_pending(error)
                    raise
                for plan in wave:
                    plan["waiting"] = False
                    plan["release"].set()
                    (status, next_candidates) = plan["status"].get()
                    if status == "request":
                        plan["request"] = next_candidates
                        plan["waiting"] = True
                    elif status != "done":
                        raise RuntimeError(
                            f"Unknown batched Search action status: {status!r}."
                        )
                failures = [
                    plan["exception"]
                    for plan in plans
                    if plan.get("exception") is not None
                ]
                if failures:
                    batch_error = failures[0]
                    abort_pending(batch_error)
                    raise batch_error
            for plan in plans:
                future = plan.get("future")
                if isinstance(future, Future):
                    future.result()
                result = plan.get("result")
                if plan.get("exhausted"):
                    continue
                if result is None:
                    raise RuntimeError("A batched Search action produced no result.")
                if not 0 < int(result.evaluation_cost) <= int(plan["reserved_cost"]):
                    raise RuntimeError(
                        "A batched Search action reported an invalid candidate cost relative to its reservation."
                    )
                exact_candidate_counts = {
                    int(result.evaluation_cost),
                    len(result.evaluated),
                    int(plan["completed_evaluation_cost"]),
                    len(result.candidate_actions),
                }
                if len(exact_candidate_counts) != 1:
                    raise RuntimeError(
                        "A batched Search action reported inconsistent exact candidate accounting."
                    )
            return (plans, len(plans))
        except BaseException as error:
            abort_pending(error)
            raise
        finally:
            executor.shutdown(wait=True, cancel_futures=True)

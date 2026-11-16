"""Counters for managed execution only; unavailable measurements stay null."""

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from time import perf_counter_ns

_timer = ContextVar("autoopt_execution_timer", default=None)


class ExecutionMetrics:
    def __init__(self):
        self.counts = Counter()
        self.fallbacks = Counter()
        self.invalidations = Counter()
        self.materializations = Counter()
        self.exclusive_ns = Counter()
        self.measure = False
        self.enabled = True

    def configure(self, enabled=True):
        """Disable mechanism recording independently of semantic guards."""
        self.enabled = bool(enabled)
        for name in (
            "counts",
            "fallbacks",
            "invalidations",
            "materializations",
            "exclusive_ns",
        ):
            setattr(self, name, Counter())
        if not self.enabled:
            self.measure = False

    @contextmanager
    def timed(self, category):
        if not self.measure:
            yield
            return
        parent = _timer.get()
        current = [perf_counter_ns(), 0]
        token = _timer.set(current)
        try:
            yield
        finally:
            elapsed = perf_counter_ns() - current[0]
            self.exclusive_ns[category] += elapsed - current[1]
            if parent is not None:
                parent[1] += elapsed
            _timer.reset(token)

    def call(self, category, function, *args):
        if not self.measure:
            return function(*args)
        with self.timed(category):
            return function(*args)

    def native_outputs(self, kernel, outputs):
        if not self.enabled:
            return
        # These templates publish fresh owning ndarrays. Do not infer total
        # NumPy/allocator traffic from these observable output allocations.
        size = sum(value.nbytes for value in outputs)
        count = len(outputs)
        self.counts["native_output_allocations"] += count
        self.counts["native_output_bytes"] += size
        self.counts["allocations"] += count
        self.counts["allocated_bytes"] += size
        if kernel.startswith("splice_"):
            self.counts["native_copy_bytes"] += size
            self.counts["copy_bytes"] += size

    def description(self):
        if not self.enabled:
            return dict(
                enabled=False,
                counters={},
                fallback_by_reason={},
                invalidations_by_effect={},
                materialization_by_reason={},
                allocations=None,
                copy_bytes=None,
                scratch_peak_bytes=None,
                reuse_hits=None,
                exclusive_ns_by_category=None,
            )
        return dict(
            enabled=True,
            scope="per-independent-run; managed bindings/arena/region dispatch",
            denominators={
                k: self.counts[k]
                for k in (
                    "binding_attempts",
                    "program_calls",
                    "region_attempts",
                    "executed_operations",
                    "rng_calls",
                )
            },
            counters=dict(self.counts),
            fallback_by_reason=dict(self.fallbacks),
            invalidations_by_effect=dict(self.invalidations),
            materialization_by_reason=dict(self.materializations),
            allocations=self.counts["allocations"],
            copy_bytes=self.counts["copy_bytes"],
            scratch_peak_bytes=self.counts["scratch_peak_bytes"],
            reuse_hits=self.counts["reuse_hits"],
            exclusive_ns_by_category=dict(self.exclusive_ns) if self.measure else None,
            timing_scope="Python managed regions, excluding instrumented nested work; do not add C++ timers",
            missing={
                "numpy_internal_allocations": None,
                "process_rss": None,
                "native_operator_temporaries": None,
                "hardware_counters": None,
                "whole_search_unclassified_ns": None,
            },
        )

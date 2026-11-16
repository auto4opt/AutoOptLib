"""Bounded, run-owned deterministic preparations, always content-revalidated.

Unknown callbacks never leave borrowed input facts valid. Retained records own
their input bytes; a hit is allowed only at the original pure operation after
the complete ordered inputs, implementation and numerical environment match.
"""

from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
import threadpoolctl

_GETERR = np.geterr
_ISFINITE = np.isfinite
_ABS = np.abs
_MAX = np.max
_ALL = np.all
_THREAD_INFO = threadpoolctl.threadpool_info
_THREAD_CODE = _THREAD_INFO.__code__


@dataclass
class PreparationRecord:
    source: object
    members: tuple
    signature: tuple
    result: object
    size: int
    token: object = field(default_factory=object)
    kind: str = ""


class PreparationStore:
    def __init__(self, metrics, max_bytes=32 * 1024 * 1024):
        self.metrics = metrics
        self.max_bytes = max_bytes
        self.records = OrderedDict()
        self.bytes = 0
        self.enabled = False
        self.max_records = 64
        self.environment_reader = None
        self.threadpools = None

    def clear(self, reason):
        if self.metrics.enabled:
            self.metrics.invalidations["preparation-" + reason] += len(self.records)
        self.records.clear()
        self.bytes = 0

    def boundary(self, reason):
        # Records own their snapshots and every lookup revalidates full inputs.
        # There is no separate direct-validity state to synchronize.
        if self.metrics.enabled:
            self.metrics.invalidations["preparation-boundary-" + reason] += len(
                self.records
            )

    def _environment(self):
        if self.environment_reader is None:
            return None
        control = self.environment_reader()
        if control is None:
            return None
        if (
            np.geterr is not _GETERR
            or np.isfinite is not _ISFINITE
            or np.abs is not _ABS
            or np.max is not _MAX
            or np.all is not _ALL
            or threadpoolctl.threadpool_info is not _THREAD_INFO
            or _THREAD_INFO.__code__ is not _THREAD_CODE
        ):
            return None
        err = _GETERR()
        # Finite bounded input proves absence of over/divide/invalid warnings
        # for this distance template. Tiny products can underflow.
        if err["under"] != "ignore":
            return None
        if self.threadpools is None:
            self.threadpools = threadpoolctl.ThreadpoolController().select(
                user_api="blas"
            )
        pools = self.threadpools.info()
        if not any(p.get("user_api") == "blas" for p in pools) or any(
            type(p.get("num_threads")) is not int or p["num_threads"] != 1
            for p in pools
        ):
            return None
        return (
            control,
            tuple(sorted(err.items())),
            tuple(
                sorted(
                    (
                        p.get("filepath", ""),
                        p.get("internal_api", ""),
                        p.get("num_threads"),
                        p.get("version", ""),
                        p.get("architecture", ""),
                    )
                    for p in pools
                )
            ),
        )

    def _drop(self, key):
        old = self.records.pop(key, None)
        if old is not None:
            self.bytes -= old.size

    def _put(self, key, record):
        self._drop(key)
        if record.size > self.max_bytes:
            if self.metrics.enabled:
                self.metrics.fallbacks["preparation-memory-budget"] += 1
            return
        while self.records and (
            self.bytes + record.size > self.max_bytes
            or len(self.records) >= self.max_records
        ):
            _, old = self.records.popitem(last=False)
            self.bytes -= old.size
            if self.metrics.enabled:
                self.metrics.counts["preparation_evictions"] += 1
        self.records[key] = record
        self.bytes += record.size
        if self.metrics.enabled:
            self.metrics.counts["preparation_peak_bytes"] = max(
                self.metrics.counts["preparation_peak_bytes"], self.bytes
            )

    def _probe(self, invocation, name, operands, source, view, key):
        env = invocation.preparation_environment()
        signature = None
        parent = invocation.preparation_tokens.get("nich-distance")
        if env is not None and name == "nich-distance":
            x, fitness = operands
            if (
                type(x) is np.ndarray
                and type(fitness) is np.ndarray
                and x.dtype == np.dtype(float)
                and fitness.dtype == np.dtype(float)
                and x.ndim == 2
                and fitness.ndim == 1
                and x.flags.c_contiguous
                and fitness.flags.c_contiguous
                and x.shape[0] == fitness.size == len(view.references)
                and x.size
                and x.shape[0]
                and x.nbytes + fitness.nbytes + (x.shape[0] ** 2 + x.shape[0]) * 8
                <= self.max_bytes
            ):
                size = x.nbytes + fitness.nbytes
                if self.metrics.enabled:
                    self.metrics.counts["preparation_validation_input_bytes"] += size
                    self.metrics.counts["preparation_snapshot_copy_bytes"] += size
                    self.metrics.counts["copy_bytes"] += size
                signature = (env, x.shape, x.dtype.str, x.tobytes(), fitness.tobytes())
        elif env is not None and name == "nich-species":
            distance, g, gmax = operands
            if (
                parent is not None
                and distance is parent.result[1]
                and type(g) is int
                and type(gmax) is int
                and abs(g) <= 2**31
                and 1 <= gmax <= 2**31
            ):
                signature = (env, parent.token, g, gmax)

        def matches(record):
            return (
                signature is not None
                and record is not None
                and record.kind == name
                and record.source is source
                and len(record.members) == len(view.references)
                and all(a is b for a, b in zip(record.members, view.references))
                and record.signature == signature
            )

        record = self.records.get(key)
        if matches(record):
            return env, signature, record, key, True
        # One canonical record can serve multiple stages only after the same
        # complete proof. No aliases are inserted or charged a second time.
        if signature is not None:
            for other_key, other in self.records.items():
                if other_key != key and matches(other):
                    return env, signature, other, other_key, True
        if name == "nich-distance" and signature is not None:
            # A matching owned snapshot already proves these facts. Only a
            # new input needs another finite/range traversal.
            if self.metrics.enabled:
                self.metrics.counts["preparation_input_range_checks"] += 1
            if not self._finite_inputs(*operands):
                signature = None
        return env, signature, record, key, False

    @staticmethod
    def _finite_inputs(x, fitness):
        return (
            _ALL(_ISFINITE(x)) and _ALL(_ISFINITE(fitness)) and _MAX(_ABS(x)) <= 1e100
        )

    def invoke(self, invocation, instruction, function, operands):
        name = instruction.preparation
        if not self.enabled or not name:
            return function(*operands)
        if self.metrics.enabled:
            self.metrics.counts["preparation_requests"] += 1
        key = (invocation.cursor.pathway, invocation.cursor.stage, name)
        if (
            name == "nich-species"
            and "nich-distance" not in invocation.preparation_tokens
        ):
            self._drop(key)
            if self.metrics.enabled:
                self.metrics.counts["preparation_recomputations"] += 1
                self.metrics.fallbacks["preparation-parent-unavailable"] += 1
            return self.metrics.call("preparation-compute", function, *operands)
        aux = invocation.slots[invocation.plan.argument_slot][3]
        if (
            invocation.frame.depth != 1
            or invocation.unknown_helper
            or type(aux) is not dict
            or type(aux.get("rng")) is not np.random.Generator
            or not invocation.dependencies_trusted()
        ):
            self.clear("untrusted-call")
            if self.metrics.enabled:
                self.metrics.fallbacks["preparation-dependency"] += 1
            return function(*operands)
        source = invocation.slots[invocation.plan.argument_slot][0]
        view = invocation.bindings.source_view(source)
        if view is None:
            if self.metrics.enabled:
                self.metrics.fallbacks["preparation-source"] += 1
            return function(*operands)
        env, signature, record, record_key, hit = self.metrics.call(
            "preparation-validation",
            self._probe,
            invocation,
            name,
            operands,
            source,
            view,
            key,
        )
        if hit:
            if record_key != key:
                self._drop(key)
                if self.metrics.enabled:
                    self.metrics.counts["preparation_cross_stage_hits"] += 1
            self.records.move_to_end(record_key)
            invocation.preparation_tokens[name] = record
            if self.metrics.enabled:
                self.metrics.counts["preparation_validated_hits"] += 1
                self.metrics.counts["preparation_hits_" + name] += 1
            return record.result
        if record is not None:
            if self.metrics.enabled:
                self.metrics.counts["preparation_validation_failures"] += 1
        if env is None:
            if self.metrics.enabled:
                self.metrics.fallbacks["preparation-environment"] += 1
        elif signature is None:
            if self.metrics.enabled:
                self.metrics.fallbacks["preparation-input-or-budget"] += 1
        elif record is not None:
            if self.metrics.enabled:
                self.metrics.invalidations["preparation-content-or-dependency"] += 1
        self._drop(key)
        if self.metrics.enabled:
            self.metrics.counts["preparation_recomputations"] += 1
        result = self.metrics.call("preparation-compute", function, *operands)
        if signature is not None and invocation.dependencies_trusted():
            if name == "nich-distance":
                size = (
                    sum(a.nbytes for a in result)
                    + len(signature[3])
                    + len(signature[4])
                )
            else:
                size = sum(s.idx.nbytes for s in result)
            record = PreparationRecord(
                source, view.references, signature, result, size, kind=name
            )
            self._put(key, record)
            if self.records.get(key) is record:
                invocation.preparation_tokens[name] = record
        return result

    def remember_elite(self, source, fitness, elite, members):
        if not self.enabled:
            return
        signature = (fitness.shape, fitness.tobytes())
        if self.metrics.enabled:
            self.metrics.counts["preparation_snapshot_copy_bytes"] += fitness.nbytes
            self.metrics.counts["copy_bytes"] += fitness.nbytes
        # Both arrays were created privately by the native kernel; retain the
        # owning sorted indices, never a mutable population or borrowed view.
        self._put(
            ("de-elite", id(source)),
            PreparationRecord(
                source,
                members,
                signature,
                elite,
                fitness.nbytes + elite.nbytes,
                kind="de-elite",
            ),
        )

    def description(self):
        return dict(
            enabled=self.enabled,
            max_bytes=self.max_bytes,
            max_records=self.max_records,
            retained_bytes=self.bytes,
            records=len(self.records),
            hit_proof="full-input-bytes-ordered-members-code-and-environment",
        )

"""Invocation-local representations and typed storage. No scientific shadow state."""

from dataclasses import dataclass
from types import GetSetDescriptorType, SimpleNamespace

import numpy as np


@dataclass
class Lease:
    pool: object
    key: object
    array: np.ndarray
    reusable: bool = True
    released: bool = False

    def release(self):
        if not self.released:
            self.released = True
            self.pool.release(self)


class Workspace:
    """One retained owning allocation per logical slot, never overwrite a live lease."""

    def __init__(self, metrics):
        self.metrics = metrics
        self.slots = {}
        self.active = {}
        self.live_bytes = 0
        self.transient_bytes = 0

    def acquire(self, key, shape, dtype):
        if not self.metrics.measure:
            return self._acquire(key, shape, dtype)
        with self.metrics.timed("allocation"):
            return self._acquire(key, shape, dtype)

    def _acquire(self, key, shape, dtype):
        shape, dtype = tuple(shape), np.dtype(dtype)
        array = self.slots.get(key)
        busy = key in self.active
        if (
            array is None
            or busy
            or array.shape != shape
            or array.dtype != dtype
            or not array.flags.c_contiguous
            or not array.flags.writeable
        ):
            array = np.empty(shape, dtype=dtype)
            if self.metrics.enabled:
                self.metrics.counts["scratch_peak_bytes"] = max(
                    self.metrics.counts["scratch_peak_bytes"],
                    sum(a.nbytes for a in self.slots.values())
                    + self.transient_bytes
                    + array.nbytes,
                )
                self.metrics.counts["allocations"] += 1
                self.metrics.counts["allocated_bytes"] += array.nbytes
            if not busy:
                self.slots[key] = array
        else:
            if self.metrics.enabled:
                self.metrics.counts["storage_reuse_hits"] += 1
        lease = Lease(self, key, array, reusable=not busy)
        if not busy:
            self.active[key] = lease
        else:
            self.transient_bytes += array.nbytes
        self.live_bytes += array.nbytes
        if self.metrics.enabled:
            self.metrics.counts["scratch_active_peak_bytes"] = max(
                self.metrics.counts["scratch_active_peak_bytes"], self.live_bytes
            )
        return lease

    def release(self, lease):
        self.live_bytes -= lease.array.nbytes
        if self.active.get(lease.key) is lease:
            del self.active[lease.key]
        else:
            self.transient_bytes -= lease.array.nbytes
        if not lease.reusable and self.slots.get(lease.key) is lease.array:
            del self.slots[lease.key]

    def description(self):
        return dict(
            retained_bytes=sum(a.nbytes for a in self.slots.values()),
            active_bytes=self.live_bytes,
            transient_bytes=self.transient_bytes,
            logical_slots=len(self.slots),
            active_slots=len(self.active),
        )


@dataclass
class SourceView:
    source: object
    references: tuple
    rows: tuple


@dataclass
class SelectedRows:
    """Private pre-RNG snapshot for a proven single indexed consumer."""

    values: np.ndarray
    rows: tuple

    def reference_matrix(self):
        # The admitted span to this entry contains only ordinary index work
        # and exact Generator draws. Before reference arithmetic can call a
        # floating-point handler, restore its ordinary full-array input.
        result = np.empty((len(self.rows), self.values.shape[1]), dtype=float)
        for i, row in enumerate(self.rows):
            result[i] = row
        return result


@dataclass
class Representation:
    view: SourceView
    field: str
    indices: tuple
    value: object
    stamp: int
    dtype: object
    layout: str


def admitted_source(source):
    """Check before any reader/conversion; do not probe user descriptors."""
    from ..utils.solve import (
        Solution,
        SolutionSet,
        _solution_decision_code,
        _solution_decision_reader,
        _solution_stack,
    )
    from .native import (
        NativeSolutionSet,
        _native_decision_code,
        _native_decision_reader,
        canonical_native_reader,
    )

    if type(source) not in (SolutionSet, NativeSolutionSet):
        return None
    if type(source) is NativeSolutionSet and not canonical_native_reader():
        return None
    expected = (
        _solution_decision_reader
        if type(source) is SolutionSet
        else _native_decision_reader
    )
    code = (
        _solution_decision_code
        if type(source) is SolutionSet
        else _native_decision_code
    )
    if (
        type(source).__getattribute__ is not object.__getattribute__
        or Solution.__getattribute__ is not object.__getattribute__
        or "dec" in vars(Solution)
        or any("_items" in vars(cls) for cls in type(source).__mro__)
        or expected.__code__ is not code
        or np.vstack is not _solution_stack
        or _solution_decision_reader.__globals__.get("np") is not np
        or vars(type(source)).get("decs") is not expected
        or "decs" in vars(source)
    ):
        return None
    items = vars(source).get("_items")
    if type(items) is not list:
        return None
    rows = []
    for item in items:
        if type(item) is not Solution:
            return None
        row = vars(item).get("dec")
        if type(row) is not np.ndarray or row.dtype.kind not in "biuf":
            return None
        rows.append(row)
    return SourceView(source, tuple(items), tuple(rows))


def rows_overlap(a, b):
    return a is None or b is None or bool(set(a).intersection(b))


class InputBindings:
    """The sole store for prefix sharing and planned representations/validity."""

    def __init__(self, frame, region, args=None, key=None):
        self.frame = frame
        self.metrics = frame.metrics
        self.workspace = frame.workspace
        self.args = args
        self.key = key or "legacy-prefix"
        self.records = []
        self.writes = []
        self.leases = []
        self.prefix_enabled = region is None or bool(region.get("inputs"))
        self.broadcast_enabled = bool(region and region.get("row_broadcast"))
        self.compact_value = None
        self.current_requirements = ()
        self.current_writes = ()
        self.reuse_enabled = True
        self.row_map = {}
        self.source_views = {}
        self.unknown_effects = []

    def source_view(self, source):
        if self.reuse_enabled:
            cached = self.source_views.get(id(source))
            if cached is not None:
                return cached
            for record in reversed(self.records):
                if record.view.source is source and self.valid(record):
                    return record.view
        view = admitted_source(source)
        if view is not None and self.reuse_enabled:
            self.source_views[id(source)] = view
        return view

    @staticmethod
    def _row_shape(view):
        if view is None or not view.rows:
            return None
        first = view.rows[0]
        if (
            first.ndim != 1
            or first.dtype != np.dtype(float)
            or any(
                row.ndim != 1 or row.dtype != first.dtype or row.shape != first.shape
                for row in view.rows
            )
        ):
            return None
        return len(view.rows), first.size

    def shape_only(self, source):
        """Prove the shape of an ordinary float64 stack without copying it.

        Every row is still checked, including rows not selected downstream;
        malformed shapes/conversions must retain their reference error point.
        """
        view = self.source_view(source)
        shape = self._row_shape(view)
        if shape is None:
            return None
        if self.metrics.enabled:
            self.metrics.counts["shape_only_reads"] += 1
            self.metrics.counts["shape_only_copy_bytes_avoided"] += (
                shape[0] * shape[1] * 8
            )
        return shape

    def selected_snapshot(self, source, indices, shape):
        if (
            not self.reuse_enabled
            or type(indices) is not np.ndarray
            or indices.dtype != np.dtype(np.int64)
            or indices.ndim != 1
            or indices.size >= shape[0]
        ):
            return None
        view = self.source_view(source)
        if self._row_shape(view) != shape or any(
            int(i) < 0 or int(i) >= shape[0] for i in indices
        ):
            return None
        # Snapshot at the original Pbest reader, before either coefficient RNG.
        # Repeated rows remain repeated, in their original order.
        values = np.empty((indices.size, shape[1]), dtype=float)
        for out, index in enumerate(indices):
            values[out] = view.rows[int(index)]
        if self.metrics.enabled:
            self.metrics.counts["allocations"] += 1
            self.metrics.counts["allocated_bytes"] += values.nbytes
            self.metrics.counts["copy_bytes"] += values.nbytes
            self.metrics.counts["selected_row_snapshots"] += 1
            self.metrics.counts["selected_row_copy_bytes_avoided"] += (
                (shape[0] - indices.size) * shape[1] * 8
            )
        return SelectedRows(values, view.rows)

    def begin(self, instruction):
        self.current_writes = instruction.writes
        self.current_requirements = instruction.requirements

    def source(self, name):
        if self.args is None:
            return None
        if name == "parents":
            return self.args[0]
        if name == "parameters":
            return self.args[2]
        if name == "arguments":
            return self.args
        if name == "problem":
            return self.args[1]
        aux = self.args[3]
        if name == "auxiliary":
            return aux
        if name == "population":
            context = aux.get("_population_context") if type(aux) is dict else None
            return self.args[0] if context is None else context
        if name.startswith("auxiliary.") and type(aux) is dict:
            return aux.get(name.split(".", 1)[1])
        return None

    def _plain_access(self, access):
        source = self.source(access.source)
        if access.source == "arguments":
            index = {"inner_g": 5, "generation": 4}.get(access.field)
            value = source[index] if index is not None and len(source) > index else None
        elif access.source == "problem" and source is not None:
            cls = type(source)
            if cls is SimpleNamespace:
                value = vars(source).get(access.field)
                return (
                    value is None
                    or type(value) in (bool, int, float, str, np.float64, np.int64)
                    or type(value) is np.ndarray
                    and value.dtype.kind in "biuf"
                )
            dictionary = next(
                (
                    vars(base)["__dict__"]
                    for base in cls.__mro__
                    if "__dict__" in vars(base)
                ),
                None,
            )
            if (
                cls.__getattribute__ is not object.__getattribute__
                or any(access.field in vars(base) for base in cls.__mro__)
                or type(dictionary) is not GetSetDescriptorType
            ):
                return False
            value = vars(source).get(access.field)
        elif type(source) is dict and access.source != "parameters":
            value = source.get(access.field)
        else:
            value = source
        if value is None or type(value) in (
            bool,
            int,
            float,
            str,
            np.float64,
            np.int64,
        ):
            return True
        if type(value) is np.ndarray:
            return value.dtype.kind in "biuf"
        if type(value) in (list, tuple):
            return all(
                type(v) in (bool, int, float, np.float64, np.int64) for v in value
            )
        if type(value) is np.random.Generator:
            return True
        from ..utils.solve import Solution

        if type(value) is Solution:
            return (
                Solution.__getattribute__ is object.__getattribute__
                and "dec" not in vars(Solution)
                and type(vars(value).get("dec")) is np.ndarray
                and vars(value)["dec"].dtype.kind in "biuf"
            )
        view = self.source_view(value)
        if view is None or access.field != "fitness":
            return view is not None
        from ..utils.solve import (
            SolutionSet,
            _solution_fitness_code,
            _solution_fitness_reader,
        )
        from .native import _native_fitness_code, _native_fitness_reader

        reader, code = (
            (_solution_fitness_reader, _solution_fitness_code)
            if type(value) is SolutionSet
            else (_native_fitness_reader, _native_fitness_code)
        )
        return (
            type(value).fits is reader
            and reader.__code__ is code
            and "fits" not in vars(value)
            and "fit" not in vars(Solution)
            and all(
                type(vars(item).get("fit")) in (int, float, np.float64, np.int64)
                for item in view.references
            )
        )

    def before(self, instruction, operands):
        # Consume read contracts in admission, not just the diagnostic dump.
        for access in instruction.reads:
            if not self._plain_access(access):
                self.unknown("unknown-read")
                break
        if any("rng" in effect for effect in instruction.effects):
            generators = [v for v in operands if isinstance(v, np.random.Generator)]
            if any(type(v) is not np.random.Generator for v in generators):
                self.unknown("unknown-rng")

    def after(self, instruction, operands):
        # Actual commits and actual RNG calls notify touch at their reference
        # position. A declared conditional write is NOT an observed write.
        if self.metrics.enabled:
            self.metrics.counts["completed_primitives"] += 1

    def requirement(self, source, reader):
        for requirement in self.current_requirements:
            if (
                requirement.reader in (reader, "*")
                and self.source(requirement.source) is source
            ):
                return requirement
        return None

    def valid(self, record):
        for source, field, rows, array in self.writes[record.stamp :]:
            if source is record.view.source and field in (
                record.field,
                "members",
                "layout",
                "*",
            ):
                if rows_overlap(rows, record.indices):
                    return False
            elif type(array) is np.ndarray:
                rows_read = (
                    record.view.rows
                    if type(record.view.source) is np.ndarray
                    else (record.view.rows[i] for i in record.indices)
                )
                if any(np.may_share_memory(array, row) for row in rows_read):
                    return False
        return True

    def find(self, source, field, indices, dtype, layout):
        for record in reversed(self.records):
            if (
                record.view.source is source
                and record.field == field
                and record.indices == indices
                and record.dtype == dtype
                and record.layout == layout
                and self.valid(record)
            ):
                if self.metrics.enabled:
                    self.metrics.counts["reuse_hits"] += 1
                if self.metrics.enabled:
                    self.frame.stats["reuse_hits"] += 1
                    size = getattr(record.value, "nbytes", 0)
                    self.frame.stats["copy_bytes_avoided"] += size
                return record.value
        return None

    def record(self, view, indices, dtype, layout, value, copied=True):
        self.records.append(
            Representation(
                view, "decisions", indices, value, len(self.writes), dtype, layout
            )
        )
        if self.metrics.enabled:
            materialized = layout not in ("references", "indexed") and copied
            size = getattr(value, "nbytes", 0)
            self.metrics.counts["representation_bindings"] += 1
            self.frame.stats["materializations"] += materialized
            self.frame.stats["materialized_bytes"] += size
            self.metrics.counts["materializations"] += materialized
            self.metrics.counts["copy_bytes"] += size if copied else 0
            live = sum(getattr(r.value, "nbytes", 0) for r in self.records)
            self.frame.stats["peak_live_bytes"] = max(
                self.frame.stats["peak_live_bytes"], live
            )
        return value

    def read(self, source, dtype, reader, reference):
        if not self.metrics.measure:
            return self._read(source, dtype, reader, reference)
        with self.metrics.timed("representation-binding"):
            return self._read(source, dtype, reader, reference)

    def _read(self, source, dtype, reader, reference):
        requirement = self.requirement(source, reader)
        if not self.reuse_enabled or (requirement is None and not self.prefix_enabled):
            return reference()
        if (
            type(source) is np.ndarray
            and source.ndim == 2
            and source.dtype.kind in "biuf"
            and (requirement is None or requirement.rows in ("all", "selected"))
        ):
            indices = tuple(range(source.shape[0]))
            cached = self.find(source, "decisions", indices, dtype, "borrowed")
            if cached is not None:
                return cached
            value = reference()  # np.asarray keeps its original cast/ownership rules.
            borrowed = value is source or (
                type(value) is np.ndarray and np.may_share_memory(value, source)
            )
            if self.metrics.enabled:
                self.metrics.materializations[
                    "borrowed-array" if borrowed else "required-array-conversion"
                ] += 1
            return self.record(
                SourceView(source, (), (source,)),
                indices,
                dtype,
                "borrowed",
                value,
                copied=not borrowed,
            )
        view = self.source_view(source)
        if view is None:
            self.unknown("unknown-effect")
            if self.metrics.enabled:
                self.metrics.fallbacks["reader-protocol"] += 1
            return reference()
        indices = tuple(range(len(view.rows)))
        if requirement is not None and type(requirement.rows) is tuple:
            indices = tuple(
                i + len(view.rows) if i < 0 else i for i in requirement.rows
            )
        layout = requirement.representation if requirement is not None else "matrix"
        self.row_map[id(source)] = (source, view.references, indices)
        references = view.references
        cached = self.find(source, "decisions", indices, dtype, layout)
        if cached is not None:
            return cached
        if layout == "references":
            result = tuple(references[i] for i in indices)
            if self.metrics.enabled:
                self.metrics.materializations["reference-sequence"] += 1
        elif layout == "indexed":
            result = tuple(view.rows[i] for i in indices)
            if self.metrics.enabled:
                self.metrics.materializations["indexed-rows"] += 1
        else:
            rows = tuple(view.rows[i] for i in indices)
            # Same-type native rows can be copied directly into a private
            # owning matrix. Other promotion/protocol rules stay in reference.
            supported = bool(rows) and all(
                r.ndim == 1
                and r.dtype == rows[0].dtype
                and r.dtype.isnative
                and (self.frame.materializer is None or r.dtype.byteorder == "=")
                and r.dtype in (np.dtype(float), np.dtype(np.int64))
                and r.shape == rows[0].shape
                for r in rows
            )
            target_dtype = (
                np.dtype(dtype)
                if dtype is not None
                else (rows[0].dtype if rows else None)
            )
            supported = supported and target_dtype == rows[0].dtype
            if supported:
                lease = self.workspace.acquire(
                    (self.key, reader, self._role(source)),
                    (len(rows), rows[0].size),
                    target_dtype,
                )
                self.leases.append(lease)
                result = lease.array
                if self.frame.materializer is not None:
                    result = self.frame.materializer(
                        [references[i] for i in indices], result
                    )
                    if result is None:
                        raise RuntimeError(
                            "Admitted matrix materializer rejected its layout"
                        )
                else:
                    for i, row in enumerate(rows):
                        result[i] = row
                if self.metrics.enabled:
                    self.metrics.materializations["required-contiguous-rows"] += 1
            else:
                result = reference()
                if indices != tuple(range(len(view.rows))):
                    result = result[list(indices)]
                if self.metrics.enabled:
                    self.metrics.materializations["reference-promotion-or-layout"] += 1
        return self.record(view, indices, dtype, layout, result)

    def _role(self, source):
        if self.args is not None:
            if source is self.args[0]:
                return "parents"
            if source is self.source("population"):
                return "population"
            aux = self.args[3]
            if type(aux) is dict and source is aux.get("Pbest"):
                return "Pbest"
        # A single non-authoritative slot for other admitted readers, not an
        # id-keyed population cache that grows with every event.
        return "other"

    def touch(self, source, field, rows=None, array=None):
        if field != "state":
            self.source_views.clear()
        rows = None if rows is None else tuple(int(i) for i in rows)
        self.writes.append((source, field, rows, array))
        if self.metrics.enabled:
            self.metrics.counts["actual_writes"] += 1
        declared = {access.field for access in self.current_writes}
        if (
            self.args is not None
            and type(source) is dict
            and type(self.args[3]) is not dict
        ):
            self.args = (*self.args[:3], source, *self.args[4:])
        if self.args is not None and field not in declared:
            if field != "state":
                self.unknown("undeclared-write")
        retained = [record for record in self.records if self.valid(record)]
        invalidated = len(self.records) - len(retained)
        if invalidated:
            if self.metrics.enabled:
                self.metrics.invalidations["controlled-write"] += invalidated
            if self.frame.metrics.enabled:
                self.frame.invalidations["controlled-write"] += invalidated
        self.records = retained

    def unknown(self, reason):
        self.source_views.clear()
        self.frame.preparations.clear(reason)
        if self.metrics.enabled:
            self.metrics.invalidations[reason] += len(self.records)
        self.records.clear()
        for lease in self.leases:
            lease.reusable = False
        self.reuse_enabled = False
        self.unknown_effects.append(reason)

    def end_prefix(self, reason):
        self.prefix_enabled = False
        if self.args is None:
            self.invalidate(reason)
        elif self.records:
            if self.frame.metrics.enabled:
                self.frame.invalidations[reason] += 1

    def invalidate(self, reason):
        if self.records:
            if self.frame.metrics.enabled:
                self.frame.invalidations[reason] += 1
        self.unknown(reason)
        if reason == "component-exit":
            self.compact_value = None

    def escape(self, result):
        from ..utils.solve import Solution, SolutionSet
        from .native import NativeSolutionSet

        seen = set()

        def visit(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if type(value) is np.ndarray:
                for lease in self.leases:
                    if np.may_share_memory(value, lease.array):
                        lease.reusable = False
            elif type(value) in (tuple, list):
                for child in value:
                    visit(child)
            elif type(value) is dict:
                for child in value.values():
                    visit(child)
            elif type(value) not in (
                type(None),
                bool,
                int,
                float,
                str,
                np.random.Generator,
                Solution,
                SolutionSet,
                NativeSolutionSet,
            ):
                for lease in self.leases:
                    lease.reusable = False

        visit(result)

    def close(self, result=None):
        self.escape(result)
        self.records.clear()
        self.row_map.clear()
        self.source_views.clear()
        for lease in self.leases:
            lease.release()
        self.leases.clear()

    def compact_rows(self, value, columns):
        if (
            self.broadcast_enabled
            and type(value) is np.ndarray
            and value.dtype == np.dtype(float)
            and value.ndim == 2
            and value.shape[0] > 0
            and value.shape[1] == 1
            and columns > 1
        ):
            result = value.copy()
            if self.metrics.enabled:
                self.metrics.counts["allocations"] += 1
                self.metrics.counts["copy_bytes"] += result.nbytes
            self.compact_value = result
            if self.frame.metrics.enabled:
                self.frame.stats["row_broadcasts"] += 1
                self.frame.stats["representation_bytes_avoided"] += (
                    columns - 1
                ) * value.nbytes
            return result
        result = value.repeat(columns, axis=1)
        if type(result) is np.ndarray:
            if self.metrics.enabled:
                self.metrics.counts["allocations"] += bool(result.flags.owndata)
                self.metrics.counts["copy_bytes"] += result.nbytes
        return result

    def finalize_rows(self, value, consumer, columns):
        if self.compact_value is None or value is not self.compact_value:
            return value
        self.compact_value = None
        if (
            type(consumer) is not np.ndarray
            or consumer.dtype != np.dtype(float)
            or consumer.shape != (value.shape[0], columns)
        ):
            if self.frame.metrics.enabled:
                self.frame.stats["row_broadcasts"] -= 1
                self.frame.stats["representation_bytes_avoided"] -= (
                    columns - 1
                ) * value.nbytes
                self.frame.stats["opaque_broadcast_consumers"] += 1
            result = value.repeat(columns, axis=1)
            if self.metrics.enabled:
                self.metrics.counts["allocations"] += 1
                self.metrics.counts["copy_bytes"] += result.nbytes
            return result
        return value

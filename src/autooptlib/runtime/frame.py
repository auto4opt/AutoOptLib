"""Executable regions, exact cursors and invocation-owned bindings/storage."""

from collections import Counter
from dataclasses import dataclass, replace

import numpy as np

from ..components._utils import _input_bindings
from .bindings import InputBindings, SelectedRows, Workspace
from .metrics import ExecutionMetrics
from .preparations import PreparationStore

_GETERR = np.geterr
_NO_PROOF = object()


@dataclass
class Cursor:
    pathway: int = 0
    stage: int = 0
    phase: str = "component"
    event: object = None
    operation: object = 0
    effect: int = 0
    region: object = None


@dataclass
class RegionExit:
    status: str
    next: object
    cursor: Cursor
    writes: tuple
    outputs: tuple
    reason: object = None
    unknown_effects: tuple = ()
    writes_applied: bool = True


def unpack_outputs(value, count):
    # Use Python's own UNPACK_SEQUENCE semantics, including bounded iteration,
    # exceptions and user protocols. These are the finite template arities.
    if count == 2:
        a, b = value
        return a, b
    if count == 3:
        a, b, c = value
        return a, b, c
    if count == 4:
        a, b, c, d = value
        return a, b, c, d
    if count == 5:
        a, b, c, d, e = value
        return a, b, c, d, e
    if count == 6:
        a, b, c, d, e, f = value
        return a, b, c, d, e, f
    if count == 7:
        a, b, c, d, e, f, g = value
        return a, b, c, d, e, f, g
    if count == 8:
        a, b, c, d, e, f, g, h = value
        return a, b, c, d, e, f, g, h
    if count == 9:
        a, b, c, d, e, f, g, h, i = value
        return a, b, c, d, e, f, g, h, i
    if count == 10:
        a, b, c, d, e, f, g, h, i, j = value
        return a, b, c, d, e, f, g, h, i, j
    raise ValueError("Unsupported computation output arity")


class Invocation:
    def __init__(self, frame, plan, args, origin):
        self.frame, self.plan = frame, plan
        self.slots = [None] * plan.slot_count
        self.slots[plan.argument_slot] = args
        self.cursor = Cursor(**(origin or {}))
        self.bindings = InputBindings(
            frame,
            plan.binding_region,
            args,
            key=(
                self.cursor.pathway,
                self.cursor.stage,
                self.cursor.phase,
                plan.description["component"],
            ),
        )
        frame.workspace_scope = (self.cursor.pathway, self.cursor.stage)
        self.retained = []
        self.region_write_start = 0
        self.region_unknown_start = 0
        self.unknown_helper = False
        self.callables = {}
        self.preparation_tokens = {}
        self._dependencies_trusted = None
        self._preparation_environment = _NO_PROOF
        self.frame.active_invocations.append(self)

    def dependencies_trusted(self, refresh=False):
        if refresh or self._dependencies_trusted is None:
            self._dependencies_trusted = self.plan.specification.dependencies.trusted()
        return self._dependencies_trusted

    def preparation_environment(self):
        if self._preparation_environment is _NO_PROOF:
            self._preparation_environment = self.frame.preparations._environment()
        return self._preparation_environment

    def native_region(self, region, receipt=True):
        """Bind once, execute one C++ block, publish only complete live-outs."""
        if not region.kernel or not self.frame.native_blocks:
            return None
        if self.frame.metrics.enabled:
            self.frame.metrics.counts["native_block_attempts"] += 1
        result = self._native_region(region, receipt)
        if result is None:
            if self.frame.metrics.enabled:
                self.frame.metrics.counts["native_block_refusals"] += 1
            self.bindings.unknown("native-block-fallback")
        return result

    def _native_region(self, region, receipt=True):
        if (
            self.frame.native_kernel is None
            or self.unknown_helper
            or not self.bindings.reuse_enabled
        ):
            return None
        if not self.dependencies_trusted():
            if self.frame.metrics.enabled:
                self.frame.metrics.fallbacks["native-block-dependency"] += 1
            return None
        floating = not region.kernel.startswith("splice_")
        if floating and (np.geterr is not _GETERR or _GETERR()["under"] != "ignore"):
            if self.frame.metrics.enabled:
                self.frame.metrics.fallbacks["native-block-float-environment"] += 1
            return None
        for pc in range(region.entry, region.exit):
            instruction = self.plan.instructions[pc]
            resolved = self.callables.get(pc)
            if resolved is None:
                resolved = instruction.binding.resolve()[0]
            if not resolved.trusted():
                if self.frame.metrics.enabled:
                    self.frame.metrics.fallbacks["native-block-helper"] += 1
                return None
        # Preserve LOAD_GLOBAL of outer calls before the inner arithmetic.
        for pc in range(region.entry, region.exit):
            for target in self.plan.instructions[pc].prefetch:
                self.callables[target] = self.plan.instructions[
                    target
                ].binding.resolve()[0]
        operands = tuple(self.slots[s] for s in region.kernel_inputs)
        if region.kernel == "pso" and type(operands[4]) is SelectedRows:
            snapshot = operands[4]
            operands = (*operands[:4], snapshot.values, *operands[5:], True)
        if self.frame.metrics.measure:
            with self.frame.metrics.timed("native-numeric"):
                outputs = self.frame.native_kernel(region.kernel, operands)
        else:
            outputs = self.frame.native_kernel(region.kernel, operands)
        if outputs is None:
            if self.frame.metrics.enabled:
                self.frame.metrics.fallbacks["native-block-layout-or-range"] += 1
            return None
        if type(outputs) is not tuple or len(outputs) != len(region.kernel_outputs):
            raise RuntimeError("Native block returned incompatible live-outs")
        if self.frame.metrics.enabled:
            self.frame.metrics.native_outputs(region.kernel, outputs)
        # All pure inputs are still held by operands. Clear dead slots before
        # assigning outputs, since a later output can reuse an earlier slot.
        for pc in range(region.entry, region.exit):
            instruction = self.plan.instructions[pc]
            self.callables.pop(pc, None)
            for slot in instruction.dead:
                self.slots[slot] = None
        for slot, value in zip(region.kernel_outputs, outputs):
            self.slots[slot] = value
        count = region.exit - region.entry
        if self.frame.metrics.enabled:
            self.frame.stats["executed_operations"] += count
            self.frame.metrics.counts["executed_operations"] += count
            self.frame.metrics.counts["completed_primitives"] += count
            self.frame.metrics.counts["native_block_calls"] += 1
            self.frame.metrics.counts["native_block_" + region.kernel] += 1
        if not receipt:
            self.cursor.operation = region.continuation
            self.frame.position = self.cursor
            return True
        self.cursor = replace(self.cursor, operation=region.continuation)
        self.frame.position = self.cursor
        return self.leave_region(region, region.continuation)

    def region_inputs(self, region):
        # These are actual values/owners, not guessed static shapes.
        values = []
        for slot in region.input_slots:
            value = self.slots[slot]
            descriptor = dict(slot=slot, value=value, owner=value)
            if type(value) is np.ndarray:
                descriptor.update(
                    dtype=value.dtype.str,
                    shape=value.shape,
                    strides=value.strides,
                    data=value.__array_interface__["data"][0],
                )
            descriptor["row_maps"] = tuple(self.bindings.row_map.values())
            values.append(descriptor)
        return values

    def admit_region(self, region, workspace, snapshot=True):
        if self.frame.metrics.enabled:
            self.frame.metrics.counts["region_attempts"] += 1
            self.frame.metrics.counts["region_attempts_" + region.kind] += 1
        self.region_write_start = len(self.bindings.writes)
        self.region_unknown_start = len(self.bindings.unknown_effects)
        if snapshot:
            self.cursor = replace(
                self.cursor, operation=region.entry, region=region.entry
            )
        else:
            self.cursor.operation = region.entry
            self.cursor.region = region.entry
        if workspace is not self.frame.workspace:
            return "workspace-owner"
        if region.kind == "numeric" and not (
            region.kernel and self.frame.native_blocks
        ):
            # Admission consumes values directly. The richer region_inputs
            # description is only for an explicitly requested diagnostic.
            for slot in region.input_slots:
                value = self.slots[slot]
                if type(value) is np.ndarray:
                    if (
                        value.dtype
                        not in (np.dtype(float), np.dtype(np.int64), np.dtype(bool))
                        or not value.dtype.isnative
                        or not value.flags.c_contiguous
                    ):
                        return "layout"
                elif type(value) is SelectedRows:
                    if (
                        value.values.dtype != np.dtype(float)
                        or not value.values.flags.c_contiguous
                    ):
                        return "layout"
                elif type(value) in (list, tuple) and all(
                    type(v) in (int, float, bool, np.int64, np.float64) for v in value
                ):
                    # Pure reference assembly (e.g. niching's ordered index
                    # list) needs no ndarray admission or unknown-effect exit.
                    # Native kernels still enforce their own exact layouts.
                    pass
                elif type(value) not in (
                    int,
                    float,
                    bool,
                    type(None),
                    np.float64,
                    np.int64,
                ):
                    return "opaque-input"
        return None

    def refuse(self, region, reason):
        if self.frame.metrics.enabled:
            self.frame.metrics.fallbacks[reason] += 1
            self.frame.metrics.counts["region_refusals"] += 1
        return RegionExit("refused", region.entry, self.cursor, (), (), reason)

    def reference_region(self, region, workspace):
        # Admission refusal has executed NO primitive. Continue at this entry,
        # preserving all earlier regions and their consumed RNG/effects.
        self.bindings.unknown("region-refusal")
        self.region_write_start = len(self.bindings.writes)
        self.region_unknown_start = len(self.bindings.unknown_effects) - 1
        pc = region.entry
        while pc is not None and region.entry <= pc < region.exit:
            pc = self.step(pc)
        return self.leave_region(region, pc)

    def step(self, pc):
        instruction = self.plan.instructions[pc]
        self.cursor = replace(self.cursor, operation=pc)
        self.frame.position = self.cursor
        self.bindings.begin(instruction)
        # LOAD_GLOBAL for an outer call precedes evaluation of its nested
        # argument call. Keep that function object, even if the inner helper
        # replaces its module name, and recheck the captured object's code.
        for target in instruction.prefetch:
            self.callables[target] = self.plan.instructions[target].binding.resolve()[0]
        function = (
            self.callables.pop(pc)
            if pc in self.callables
            else instruction.binding.resolve()[0]
        )
        unchanged = function.trusted()
        if not unchanged:
            self.unknown_helper = True
            self.bindings.unknown("implementation-dependency")
            if self.frame.metrics.enabled:
                self.frame.metrics.fallbacks["implementation-dependency"] += 1
        operands = tuple(self.slots[i] for i in instruction.inputs)
        if any(type(value) is SelectedRows for value in operands):
            for slot in instruction.inputs:
                value = self.slots[slot]
                if type(value) is SelectedRows:
                    matrix = value.reference_matrix()
                    self.slots[slot] = matrix
                    if self.frame.metrics.enabled:
                        self.frame.metrics.counts["allocations"] += 1
                        self.frame.metrics.counts["allocated_bytes"] += matrix.nbytes
                        self.frame.metrics.counts["copy_bytes"] += matrix.nbytes
                        self.frame.metrics.counts[
                            "selected_row_reference_materializations"
                        ] += 1
            operands = tuple(self.slots[i] for i in instruction.inputs)
        self.bindings.before(instruction, operands)
        # RNG methods are timed at the actual call, including conditional draws.
        # The remaining arithmetic of a draw-and-scale primitive is numeric.
        if self.frame.metrics.measure:
            category = (
                "state"
                if any("state" in e for e in instruction.effects)
                else "binding"
                if any(
                    e in ("external-read", "parameter-read", "reference-read")
                    for e in instruction.effects
                )
                else "numeric"
            )
            with self.frame.metrics.timed(category):
                result = self.frame.preparations.invoke(
                    self, instruction, function, operands
                )
        else:
            result = self.frame.preparations.invoke(
                self, instruction, function, operands
            )
        if len(instruction.outputs) == 1:
            self.slots[instruction.outputs[0]] = result
        else:
            for slot, value in zip(
                instruction.outputs, unpack_outputs(result, len(instruction.outputs))
            ):
                self.slots[slot] = value
        self.bindings.after(instruction, operands)
        # Only an admitted preparation supplies a reusable no-callback proof.
        # Reference NumPy arithmetic may invoke a floating-point handler.
        if (
            instruction.effects
            or instruction.preparation not in self.preparation_tokens
        ):
            self._dependencies_trusted = None
            self._preparation_environment = _NO_PROOF
            self.bindings.source_views.clear()
        if self.frame.metrics.enabled:
            self.frame.stats["executed_operations"] += 1
            self.frame.metrics.counts["executed_operations"] += 1
        if instruction.effects:
            if self.frame.metrics.enabled:
                self.frame.stats["completed_effect_operations"] += 1
        self.cursor = replace(self.cursor, effect=len(self.bindings.writes))
        for slot in instruction.dead:
            value = self.slots[slot]
            if not (
                type(value) in (type(None), bool, int, float, str)
                or (
                    type(value) is np.ndarray
                    and value.flags.owndata
                    and value.dtype.kind in "biuf"
                )
            ):
                self.retained.append(value)
            self.slots[slot] = None
        successor = instruction.successor
        if successor is not None and successor != pc + 1:
            self.bindings.unknown("control-join-or-backedge")
        self.cursor = replace(self.cursor, operation=successor)
        return successor

    def leave_region(self, region, pc):
        if pc != region.continuation:
            self.bindings.unknown("control-join-or-backedge")
        if self.frame.metrics.enabled:
            self.frame.metrics.counts["region_calls"] += 1
            self.frame.metrics.counts["region_calls_" + region.kind] += 1
        return RegionExit(
            "completed",
            pc,
            self.cursor,
            tuple(self.bindings.writes[self.region_write_start :]),
            tuple((i, self.slots[i]) for i in region.output_slots),
            unknown_effects=tuple(
                self.bindings.unknown_effects[self.region_unknown_start :]
            ),
        )

    def apply_exit(self, region, outcome):
        if tuple(slot for slot, value in outcome.outputs) != region.output_slots:
            raise RuntimeError("Region returned a different live-out binding")
        for slot, value in outcome.outputs:
            self.slots[slot] = value
        if not outcome.writes_applied:
            for source, field, rows, array in outcome.writes:
                self.bindings.touch(source, field, rows, array)
        if outcome.unknown_effects:
            self.bindings.unknown("unknown-region-effects")
        self.cursor = outcome.cursor

    def close(self, result=None):
        # Destructors can reenter. Keep physical leases active until all
        # invocation-owned references have been dropped.
        self.slots.clear()
        self.retained.clear()
        self.callables.clear()
        self.preparation_tokens.clear()
        self.bindings.close(result)
        if self.frame.active_invocations and self.frame.active_invocations[-1] is self:
            self.frame.active_invocations.pop()


def execute_interval(region, invocation, workspace):
    """Python counterpart of the C++ region launcher; same refusal contract."""
    reason = invocation.admit_region(region, workspace)
    if reason is not None:
        return invocation.refuse(region, reason)
    outcome = invocation.native_region(region)
    if outcome is not None:
        return outcome
    pc = region.entry
    while pc is not None and region.entry <= pc < region.exit:
        pc = invocation.step(pc)
    return invocation.leave_region(region, pc)


class RunFrame:
    def __init__(self):
        self.stats = Counter()
        self.invalidations = Counter()
        self.depth = 0
        self.position = None
        self.metrics = ExecutionMetrics()
        self.preparations = PreparationStore(self.metrics)
        self.workspace = Workspace(self.metrics)
        self.measure = False
        self.materializer = None
        self.interval_executor = execute_interval
        self.last_exit = None
        self.last_private_output = False
        self.boundary_counts = Counter()
        self.native_blocks = False
        self.native_kernel = None
        self.program_compiler = None
        self.native_interval_executor = None
        self.active_invocations = []
        self.workspace_scope = (0, 0)

    def set_native(
        self, materializer, executor, kernel=None, environment=None, compiler=None
    ):
        self.materializer = materializer
        self.interval_executor = executor
        self.native_kernel = kernel
        self.native_interval_executor = executor
        self.program_compiler = compiler
        self.preparations.environment_reader = environment

    def configure_acceleration(
        self,
        preparations=False,
        native_blocks=False,
        max_preparation_bytes=32 * 1024 * 1024,
    ):
        if type(max_preparation_bytes) is not int or max_preparation_bytes < 0:
            raise ValueError("Preparation byte budget must be a nonnegative integer")
        self.preparations.clear("configuration")
        self.preparations.max_bytes = max_preparation_bytes
        self.preparations.enabled = bool(preparations)
        self.native_blocks = bool(native_blocks)

    def cluster_block(self, kernel, inputs):
        if (
            not self.native_blocks
            or self.native_kernel is None
            or not self.active_invocations
        ):
            return None
        if self.metrics.enabled:
            self.metrics.counts["native_block_attempts"] += 1
        invocation = self.active_invocations[-1]
        if (
            invocation.unknown_helper
            or not invocation.bindings.reuse_enabled
            or not invocation.dependencies_trusted(
                refresh="nich-distance" not in invocation.preparation_tokens
            )
        ):
            if self.metrics.enabled:
                self.metrics.fallbacks["native-cluster-dependency"] += 1
            return None
        if kernel == "nich_finish":
            mean = inputs[3]
            if type(mean) not in (float, np.float64) or type(inputs[5]) is not int:
                return None
            inputs = (*inputs[:3], float(mean), inputs[4], inputs[5])
        outcome = self.metrics.call(
            "native-numeric", self.native_kernel, kernel, inputs
        )
        if outcome is not None:
            if self.metrics.enabled:
                self.metrics.native_outputs(kernel, outcome)
            if self.metrics.enabled:
                self.metrics.counts["native_block_calls"] += 1
                self.metrics.counts["native_block_" + kernel] += 1
        else:
            if self.metrics.enabled:
                self.metrics.fallbacks[
                    "native-cluster-layout-range-or-environment"
                ] += 1
        return outcome

    def selection(self, references, container_type):
        from ..utils.solve import (
            SolutionSet,
            _solution_container_code,
            _solution_container_initializer,
        )
        from .native import NativeSolutionSet

        if (
            container_type in (SolutionSet, NativeSolutionSet)
            and container_type.__init__ is _solution_container_initializer
            and _solution_container_initializer.__code__ is _solution_container_code
            and container_type.__new__ is object.__new__
            and container_type.__setattr__ is object.__setattr__
            and container_type.__getattribute__ is object.__getattribute__
            and not any("_items" in vars(cls) for cls in container_type.__mro__)
        ):
            value = object.__new__(container_type)
            value._items = references
            if self.metrics.enabled:
                self.metrics.counts["reference_bindings"] += 1
            return value
        if self.metrics.enabled:
            self.metrics.fallbacks["container-protocol"] += 1
        return container_type(references)

    def boundary(self, origin, phase):
        # Called by actual stage branches and evaluation/commit exits.
        self.workspace_scope = (origin.get("pathway", 0), origin.get("stage", 0))
        if self.metrics.enabled:
            cursor = Cursor(**dict(origin, phase=phase, operation=None))
            self.boundary_counts[phase] += 1
            self.last_exit = dict(cursor.__dict__, status="boundary")
        if self.metrics.enabled:
            self.preparations.boundary(phase)
        active = _input_bindings.get()
        if active is not None:
            active.unknown(phase)

    def note_binding(self, accepted, reason=None):
        if self.metrics.enabled:
            self.metrics.counts["binding_attempts"] += 1
        if not accepted:
            if self.metrics.enabled:
                self.metrics.fallbacks[reason] += 1

    def repair_workspace(self, integer, rows, columns):
        key = (*self.workspace_scope, "repair")
        return self.workspace.acquire(
            key, (rows, columns), np.int64 if integer else float
        )

    def _run_regions(self, plan, invocation):
        if (
            self.program_compiler is not None
            and not self.metrics.enabled
            and self.interval_executor is self.native_interval_executor
        ):
            if plan.native_program is None:
                plan.native_program = self.program_compiler(plan)
            plan.native_program.execute(invocation)
            return
        pc = 0
        while pc is not None:
            region = plan.region_entries[pc]
            outcome = self.interval_executor(region, invocation, self.workspace)
            if outcome.status == "refused":
                outcome = invocation.reference_region(region, self.workspace)
            if outcome.status != "completed":
                raise RuntimeError("Invalid region exit status")
            invocation.apply_exit(region, outcome)
            pc = outcome.next

    def execute(self, plan, args, origin=None):
        self.last_private_output = False
        if not plan.specification.entry_valid():
            if self.metrics.enabled:
                self.metrics.fallbacks["implementation-entry"] += 1
            return plan.reference(*args)
        parent = _input_bindings.get()
        if parent is not None:
            parent.unknown("reentry")
            self.preparations.clear("reentry")
        invocation = Invocation(self, plan, args, origin)
        token = _input_bindings.set(invocation.bindings)
        previous = self.position
        self.depth += 1
        self.metrics.measure = self.measure and self.metrics.enabled
        if self.metrics.enabled:
            self.stats["program_calls"] += 1
            self.metrics.counts["program_calls"] += 1
        result = None
        try:
            if self.metrics.measure:
                with self.metrics.timed("region-control"):
                    self._run_regions(plan, invocation)
            else:
                self._run_regions(plan, invocation)
            result = invocation.slots[plan.result_slot]
            self.last_private_output = bool(
                plan.private_output
                and not invocation.unknown_helper
                and invocation.bindings.reuse_enabled
                and type(result) is tuple
                and type(result[0]) is np.ndarray
            )
            if self.metrics.enabled:
                self.last_exit = dict(
                    invocation.cursor.__dict__,
                    status="completed",
                    writes=len(invocation.bindings.writes),
                    unknown_effects=tuple(invocation.bindings.unknown_effects),
                )
            return result
        except BaseException:
            if self.metrics.enabled:
                self.stats["program_exceptions"] += 1
            self.stats["last_exception_op"] = invocation.cursor.operation
            invocation.cursor = replace(
                invocation.cursor, effect=len(invocation.bindings.writes)
            )
            self.last_exit = dict(invocation.cursor.__dict__, status="exception")
            invocation.bindings.unknown("exception")
            raise
        finally:
            invocation.close(result)
            self.position = previous
            self.depth -= 1
            _input_bindings.reset(token)

    def invoke(self, function, *args, region=None):
        # Historical prefix-only comparison uses the SAME representation store.
        parent = _input_bindings.get()
        if parent is not None:
            parent.unknown("reentry")
        if type(args[3]) is not dict:
            if self.metrics.enabled:
                self.stats["opaque_auxiliary"] += 1
            token = _input_bindings.set(None)
            try:
                return function(*args)
            finally:
                _input_bindings.reset(token)
        bindings = InputBindings(self, region)
        token = _input_bindings.set(bindings)
        self.depth += 1
        if self.metrics.enabled:
            self.stats["regions"] += 1
        previous = self.position
        self.position = (
            None
            if region is None
            else (region.get("stage"), region.get("entry"), region.get("exit"))
        )
        result = None
        try:
            result = function(*args)
            return result
        finally:
            bindings.invalidate("component-exit")
            bindings.close(result)
            self.depth -= 1
            self.position = previous
            _input_bindings.reset(token)

    def description(self):
        return dict(
            self.stats,
            invalidations_by_effect=dict(self.invalidations),
            active_depth=self.depth,
            cursor=self.last_exit,
            boundaries=dict(self.boundary_counts),
            workspace=self.workspace.description(),
            preparations=self.preparations.description(),
            native_blocks_enabled=self.native_blocks,
            execution_metrics=self.metrics.description(),
        )

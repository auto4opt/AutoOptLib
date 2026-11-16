"""Conservative per-stage regions derived from the candidate execution graph.

Plans describe logical lifetimes, not shadow population state. Evaluation is
always a barrier; final FastGA output escapes into generation-owned pending
offspring and therefore cannot share mutation storage with immediate repair.
"""

from dataclasses import asdict, dataclass

from ..components._execution import Access, Requirement, execution_contract


@dataclass(frozen=True)
class Instruction:
    binding: object
    inputs: tuple
    outputs: tuple
    dead: tuple
    effects: tuple
    reads: tuple
    writes: tuple
    requirements: tuple
    successor: object
    prefetch: tuple
    preparation: str = ""


@dataclass(frozen=True)
class Region:
    entry: int
    exit: int
    kind: str
    input_slots: tuple
    output_slots: tuple
    continuation: object
    kernel: str = ""
    kernel_inputs: tuple = ()
    kernel_outputs: tuple = ()


def plain_path(path, parameters=None):
    """Structural admission without evaluating user descriptors or parameters."""
    from ..utils.design._helpers import SearchParam, SearchStep
    from ..utils.design._stream_graph import (
        StreamPathway,
        StreamPathwayParam,
        StreamStage,
        StreamStageParam,
    )

    def ordinary(value, expected, fields):
        return (
            type(value) is expected
            and vars(expected).get("__getattribute__", object.__getattribute__)
            is object.__getattribute__
            and not any(name in vars(expected) for name in fields)
        )

    if not ordinary(path, StreamPathway, ("stages", "update", "archive")):
        return False
    stages = vars(path).get("stages")
    archive = vars(path).get("archive")
    if (
        type(stages) is not list
        or type(vars(path).get("update")) is not str
        or type(archive) is not list
        or any(type(name) is not str for name in archive)
    ):
        return False
    for stage in stages:
        if not ordinary(stage, StreamStage, ("search", "choose")):
            return False
        search = vars(stage).get("search")
        if not ordinary(search, SearchStep, ("primary",)):
            return False
        if type(vars(search).get("primary")) is not str:
            return False
        selector = vars(stage).get("choose")
        if selector is not None and type(selector) is not str:
            return False
    if parameters is None:
        return True
    if not ordinary(parameters, StreamPathwayParam, ("stages",)):
        return False
    stages = vars(parameters).get("stages")
    return type(stages) is list and all(
        ordinary(stage, StreamStageParam, ("search", "choose"))
        and ordinary(vars(stage).get("search"), SearchParam, ("primary",))
        for stage in stages
    )


class ComputationPlan:
    """Executable primitive schedule; slots contain derived values, not state."""

    def __init__(self, specification, contract=None):
        self.specification = specification
        self.native_program = None
        self.private_output = bool(contract and contract.private_output)
        self.reference = specification.reference
        self.reference_code = specification.reference_code
        self.binding_region = (
            dict(inputs=contract.matrix_prefix, row_broadcast=contract.row_broadcast)
            if contract and (contract.matrix_prefix or contract.row_broadcast)
            else None
        )
        producers = {"args": -1}
        consumers = {"args": []}
        operations = specification.operations
        effects = tuple(
            op.effects or (("reference-read",) if "args" in op.inputs else ())
            for op in operations
        )
        for index, operation in enumerate(operations):
            if not operation.outputs or not callable(operation.function):
                raise ValueError(
                    "Every primitive needs a callable and explicit outputs"
                )
            if len(operation.outputs) > 10:
                raise ValueError("Unsupported computation output arity")
            if (
                operation.binding_point is not None
                and not 0 <= operation.binding_point <= index
            ):
                raise ValueError("Invalid helper binding point")
            for name in operation.inputs:
                if name not in producers:
                    raise ValueError(f"Undefined computation input {name!r}")
                consumers[name].append(index)
            for name in operation.outputs:
                if name in producers:
                    raise ValueError(f"Computation value {name!r} is defined twice")
                producers[name] = index
                consumers[name] = []
        if specification.result not in producers:
            raise ValueError("Computation has no result")
        consumers[specification.result].append(len(operations))
        escaping = {"args", specification.result}
        # An effect may expose a weak reference, subclass or retained RNG
        # return. Preserve its result lifetime through the reference call.
        for index, operation in enumerate(operations):
            if effects[index]:
                escaping.update(operation.outputs)
        changed = True
        while changed:
            changed = False
            for operation in operations:
                for output, source in operation.aliases:
                    if output in escaping and source not in escaping:
                        escaping.add(source)
                        changed = True
        if not escaping <= producers.keys():
            raise ValueError("Computation alias references an undefined value")
        for name in escaping:
            if len(operations) not in consumers[name]:
                consumers[name].append(len(operations))
        for operation in operations:
            for output, source in operation.aliases:
                if output not in producers or source not in producers:
                    raise ValueError("Computation alias references an undefined value")
        values = [
            dict(
                value=name,
                born=born,
                last_use=max(consumers[name], default=born),
                consumers=tuple(consumers[name]),
                source="current-call" if born == -1 else born,
                storage_kind="python-reference",
                representation="argument-references"
                if name == "args"
                else "bound-primitive-result",
                accesses=tuple(
                    asdict(Access.parse(a))
                    for a in (operations[born].reads if born >= 0 else ())
                ),
                physical_binding="owning-arena-or-borrowed-with-owner; checked-at-region-entry",
                escapes=name in escaping,
            )
            for name, born in producers.items()
        ]
        slots = plan_storage(values)
        self.slot_count = max(slots.values()) + 1
        self.argument_slot = slots["args"]
        self.result_slot = slots[specification.result]
        self.instructions = tuple(
            Instruction(
                specification.bind(operation),
                tuple(slots[n] for n in operation.inputs),
                tuple(slots[n] for n in operation.outputs),
                tuple(
                    slots[v["value"]]
                    for v in values
                    if v["last_use"] == index and not v["escapes"]
                ),
                effects[index],
                tuple(Access.parse(a) for a in operation.reads),
                tuple(Access.parse(a) for a in operation.writes),
                operation.requirements
                or tuple(
                    Requirement(
                        a.source,
                        rows="selected" if a.source == "parents" else "all",
                        reader="*",
                    )
                    for a in map(Access.parse, operation.reads)
                    if a.field == "decisions"
                ),
                index + 1 if index + 1 < len(operations) else None,
                tuple(
                    i for i, op in enumerate(operations) if op.binding_point == index
                ),
                operation.preparation,
            )
            for index, operation in enumerate(operations)
        )
        regions, start = [], 0
        for index, operation in enumerate(operations):
            if effects[index]:
                if start < index:
                    regions.append(dict(entry=start, exit=index, kind="numeric"))
                regions.append(
                    dict(
                        entry=index,
                        exit=index + 1,
                        kind="effect",
                        effects=effects[index],
                    )
                )
                start = index + 1
        if start < len(operations):
            regions.append(dict(entry=start, exit=len(operations), kind="numeric"))
        for region in regions:
            entry, exit = region["entry"], region["exit"]
            live_in = tuple(
                name
                for name, born in producers.items()
                if born < entry and any(entry <= use < exit for use in consumers[name])
            )
            live_out = tuple(
                name
                for name, born in producers.items()
                if entry <= born < exit and any(use >= exit for use in consumers[name])
            )
            region.update(
                live_in=live_in,
                live_out=live_out,
                input_slots=tuple(slots[n] for n in live_in),
                output_slots=tuple(slots[n] for n in live_out),
                owner="invocation-registers",
                continuation=exit if exit < len(operations) else None,
                admission="at-current-entry-without-replay",
            )
            names = tuple(op.name for op in operations[entry:exit])
            pattern = next(
                (
                    p
                    for p in specification.native_patterns
                    if region["kind"] == "numeric" and p.operations == names
                ),
                None,
            )
            region.update(
                kernel=pattern.kernel if pattern else "",
                kernel_inputs=tuple(slots[n] for n in pattern.inputs)
                if pattern
                else (),
                kernel_outputs=tuple(slots[n] for n in pattern.outputs)
                if pattern
                else (),
            )
        self.regions = tuple(
            Region(
                r["entry"],
                r["exit"],
                r["kind"],
                r["input_slots"],
                r["output_slots"],
                r["exit"] if r["exit"] < len(operations) else None,
                r["kernel"],
                r["kernel_inputs"],
                r["kernel_outputs"],
            )
            for r in regions
        )
        self.region_entries = {r.entry: r for r in self.regions}
        self.has_preparations = any(op.preparation for op in self.instructions)
        self.has_native = any(r.kernel for r in self.regions) or any(
            op.preparation == "nich-species" for op in self.instructions
        )
        self.slots = slots
        self.values = values
        self.description = dict(
            schema_version=1,
            component=specification.reference.__name__,
            binding="current-component-call",
            slots=slots,
            values=values,
            implementation_guard="entry-code-dispatcher-and-current-helper-at-each-call",
            input_bindings=self.binding_region,
            regions=regions,
            result=specification.result,
            operations=[
                dict(
                    op_id=i,
                    kind=op.name,
                    inputs=op.inputs,
                    outputs=op.outputs,
                    reads=op.reads,
                    writes=op.writes,
                    effects=effects[i],
                    aliases=op.aliases,
                    binding_point=op.binding_point
                    if op.binding_point is not None
                    else i,
                    preparation=op.preparation,
                    control="reference-conditional-or-loop-inside-primitive",
                    successors=(self.instructions[i].successor,)
                    if self.instructions[i].successor is not None
                    else (),
                    requirements=[asdict(r) for r in self.instructions[i].requirements],
                    exception_boundary="propagate-without-replay",
                )
                for i, op in enumerate(operations)
            ],
            fallback="component-before-entry-or-region-before-first-effect; no-replay",
            state="python-authoritative",
        )


def plan_storage(values):
    """Assign logical slots; equal-boundary reuse needs a pointwise proof.

    Physical dtype/size/alignment checks remain native entry guards. Escaping
    values always get independent ownership and never enter the free list.
    """
    assignments, occupied = {}, []
    for value in values:
        slot = len(occupied)
        if not value["escapes"]:
            for index, previous in enumerate(occupied):
                if previous.get("storage_kind", "numeric") != value.get(
                    "storage_kind", "numeric"
                ):
                    continue
                if previous["escapes"] or previous["last_use"] is None:
                    continue
                dead = previous["last_use"] < value["born"]
                overwrite = (
                    previous["last_use"] == value["born"]
                    and value.get("overwrite_from") == previous["value"]
                )
                if dead or overwrite:
                    slot = index
                    break
        assignments[value["value"]] = slot
        if slot == len(occupied):
            occupied.append(value)
        else:
            occupied[slot] = value
    return assignments


def analyze_path(path):
    stages = []
    for index, stage in enumerate(path.stages):
        name = stage.search.primary
        contract = execution_contract(name)
        deferred = bool(
            contract and contract.deferred_final and index == len(path.stages) - 1
        )
        admitted = bool(contract and contract.kernel) and not deferred
        lifetimes = (
            {
                "value": "selected",
                "born": -1,
                "last_use": 3,
                "escapes": not admitted,
                "storage_kind": "solution_refs",
            },
            {
                "value": "operator_output",
                "born": 0,
                "last_use": 1,
                "escapes": deferred or contract is None or not contract.private_output,
            },
            {
                "value": "repaired",
                "born": 1,
                "last_use": 3,
                "escapes": False,
                "overwrite_from": "operator_output" if admitted else None,
            },
            {"value": "objective_inputs", "born": 3, "last_use": None, "escapes": True},
        )
        slots = plan_storage(lifetimes)
        reuse = admitted and slots["operator_output"] == slots["repaired"]
        stages.append(
            {
                "stage": index,
                "component": name,
                "selector": stage.choose,
                "contract": asdict(contract) if contract else None,
                "kernel": contract.kernel if contract else "",
                "native_context": bool(contract and contract.kernel),
                "input_bindings": contract.matrix_prefix if contract else (),
                "row_broadcast": bool(contract and contract.row_broadcast),
                "computation": bool(contract and contract.computation),
                "computation_binding": "at-original-component-read",
                "context_inputs": tuple(
                    name
                    for name in ("population", "target_indices")
                    if contract and name in contract.reads
                ),
                "repair_region": admitted,
                "reuse_output": reuse and contract.private_output,
                "selection_view": admitted and stage.choose is not None,
                "output_domain": contract.output_domain if contract else "opaque",
                "reason": "repair-clone-materialize"
                if admitted
                else (
                    "pending-output-escapes"
                    if deferred
                    else "computation-template-at-call"
                    if contract and contract.computation
                    else "reference-math-input-prefix"
                    if contract and (contract.matrix_prefix or contract.row_broadcast)
                    else "opaque-component"
                ),
                "effects": (
                    "selection",
                    "mutation-gate",
                    "operator-rng",
                    "operator-state",
                    "pending-transfer" if deferred else "repair-all",
                    "ordered-clone-resolution",
                    "budget-truncation",
                    "materialize-owning-inputs",
                    "objective-sequence",
                    "state-commit",
                ),
                "lifetimes": lifetimes,
                "storage_slots": slots,
                "alias": ("operator_output", "repaired") if reuse else (),
                "guards": (
                    "native-kernel-produced-private-output",
                    "built-in-idempotent-repair",
                    "dtype-shape-strides",
                    "writable-storage",
                    "no-evaluation-cache",
                ),
            }
        )
    operations, values, regions = [], [], []
    for stage in stages:
        start = len(operations)
        effects = stage["effects"]
        if stage["reason"] == "pending-output-escapes":
            effects = effects[:4] + (
                "output-limit",
                "ordered-validity-resolution",
                "pending-transfer",
                "state-commit",
            )
        elif stage["stage"] == len(stages) - 1:
            effects = effects[:4] + ("output-limit",) + effects[4:]
        stage["effects"] = effects
        for effect in effects:
            index = len(operations)
            operations.append(
                dict(
                    op_id=index,
                    stage=stage["stage"],
                    kind=effect,
                    next=index + 1,
                    binding="reference-read-point",
                )
            )
        for value in stage["lifetimes"]:
            values.append(dict(value, stage=stage["stage"], value_id=len(values)))
        regions.append(
            dict(
                stage=stage["stage"],
                entry=start,
                exit=len(operations),
                inputs=stage["input_bindings"],
                row_broadcast=stage["row_broadcast"],
                reads=stage["contract"]["reads"] if stage["contract"] else ("unknown",),
                writes=stage["contract"]["writes"]
                if stage["contract"]
                else ("unknown",),
                live_in=("selected", "current-parameters", "auxiliary"),
                live_out=("operator-output", "current-auxiliary"),
                output_ownership=(
                    "private" if stage["contract"]["private_output"] else "reference"
                )
                if stage["contract"]
                else "unknown",
                prefix_end="before-rng-or-auxiliary-conversion",
                persistent_state="python-authoritative",
                fallback=stage["reason"],
                guards=stage["guards"],
            )
        )
    if operations:
        operations[-1]["next"] = None
    return {
        "version": 2,
        "stages": stages,
        "operations": operations,
        "values": values,
        "regions": regions,
        "binding_policy": "current-reference-position",
        "resume_policy": "rebuild-frame-and-plan",
        "barrier": "objective-callback",
        "update": path.update,
        "archives": tuple(path.archive),
    }


def analyze_updates(paths):
    from ..components._execution import update_contract

    endpoints = []
    for i, path in enumerate(paths):
        contract = update_contract(path.update)
        endpoints.append(
            {
                "path": i,
                "component": path.update,
                "contract": asdict(contract) if contract else None,
                "fallback": contract is None,
            }
        )
    return {
        "barrier": "all-offspring-evaluated",
        "grouping": "ordered-first-arrival",
        "parameter_binding": "current-generation-value",
        "endpoints": endpoints,
        "lifetimes": (
            {"value": "fitness-view", "escapes": False},
            {"value": "survivor-indices", "escapes": False},
            {"value": "surviving-solutions", "escapes": True},
        ),
    }

# Architecture

AutoOptLib separates four concerns:

1. A **problem definition** constructs related training and test instances.
2. A **design space** supplies type-compatible selection, search, update, and
   archive components.
3. A **design backend** either searches graph structures and parameters or
   uses ALDes to generate a grammar-constrained token sequence.
4. The **execution engine** applies either a designed JSON algorithm or a
   built-in baseline under a strict objective-evaluation budget.

This separation is important for application studies: the problem code does
not contain optimizer logic, and a selected algorithm can be exported and
executed later without rerunning automated design.

ALDes token programs are decoded into the same pathway representation before
execution. The learning backend and search backend therefore share component,
budget, problem, serialization, and reliability semantics rather than
maintaining separate Python and MATLAB evaluators.

For a multi-path or ALDes fork algorithm, one outer iteration selects and
partitions the population, evaluates only the first primary search step of
each branch, merges the offspring, and applies the shared update. A mutation
paired with a crossover is part of that first step and is also executed. ALDes
fork sequences cannot contain later search rows, keeping generation and
execution consistent. This is AutoOptLib's constrained ALDes dialect, not a
claim that every permissive sequence accepted by the historical generator or
the paper's general pointer notation has identical semantics.
The fork parameter has two non-aliased modes: both branches execute the whole
search row, or—only for crossover plus mutation—the second branch starts at
the mutation.

The 32-token component vocabulary and ten-bin parameter decoder follow the
released ALDes source. They are intentionally versioned separately from the
paper's Appendix A2 table, which lists a different component set and is itself
inconsistent with the paper's `always_select` example. Parameter bins retain
the released implementation's linear interpolation over each component's
bounds; AutoOptLib does not silently reinterpret them as percentages.

ALDes uses two explicit learning modes. Single-problem design is the default
and conditions only on the generated token prefix. Continual design opts into
a problem-feature token, paper-style random-walk feature extraction, and EWC.
The sampled random-walk populations can be supplied to `EvaluationConfig` so
all candidate algorithms are compared from the same initial solutions.

The supported public surface is documented in [Public API](api.md). Internal
mode-based component functions remain implementation details and may evolve
between minor releases.

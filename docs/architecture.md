# Architecture

AutoOptLib separates four concerns:

1. A **problem definition** constructs related training and test instances.
2. A **design space** supplies type-compatible selection, search, update, and
   archive components.
3. The **design engine** searches graph structures and component parameters.
4. The **execution engine** applies either a designed JSON algorithm or a
   built-in baseline under a strict objective-evaluation budget.

This separation is important for application studies: the problem code does
not contain optimizer logic, and a selected algorithm can be exported and
executed later without rerunning automated design.

The supported public surface is documented in [Public API](api.md). Internal
mode-based component functions remain implementation details and may evolve
between minor releases.

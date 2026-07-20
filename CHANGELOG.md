# Changelog

All notable changes to AutoOptLib are documented here.

## 1.3.0 - 2026-07-19

### Added

- Integrated ALDes as an optional learning-based design backend with an
  ALDes-compatible 32-token vocabulary, a deliberately constrained grammar,
  an autoregressive PyTorch generator, PPO trainer, and EWC
  continual-learning penalty.
- Added a pure-Python ALDes sequence codec and evaluator that execute generated
  algorithms through the common AutoOptLib pathway engine.
- Added an optional IOH PBO problem adapter and the missing `search_reset_n`
  component required by the ALDes vocabulary.
- Added `Designer="aldes"` to the high-level `autoopt` workflow, including
  checkpoint loading, problem features, candidate count, temperature, and
  greedy-decoding controls.

### Changed

- Corrected discrete uniform-crossover and random-reset parameter bounds to
  match the reference implementation.
- Defined one consistent multi-path rule: a fork follows choose, every branch
  evaluates one search row (a crossover may include a paired mutation), and
  the branches merge before a shared population update. ALDes no longer
  generates unused later search rows.
- Made single-problem ALDes design the default with no landscape-feature
  extraction or input. Continual design explicitly enables problem-feature
  conditioning and paper-style random-walk feature extraction.
- Matched the paper's 5,000-FE training budget, deterministic PPO likelihood
  calculation, 100-step learning-rate annealing, and EWC weight of 200.
- Made candidate evaluation order-independent through common random streams
  and added reuse of feature-sampling solutions as initial populations.
- Added deterministic candidate-level CPU multiprocessing for ALDes PBO
  evaluation, including duplicate-sequence caching and persistent workers.
- Made option lookup safe when an earlier option value is a NumPy array, which
  is required for in-memory ALDes problem features.
- Kept PyTorch, IOH, pflacco, pandas, and scikit-learn behind the optional
  `autooptlib[aldes]` dependency group.

## 1.2.0 - 2026-07-19

### Added

- Self-contained Python packaging with declared runtime and test dependencies.
- Packaged CEC 2013 shift and rotation data for wheel installations.
- Public runtime API for registering custom components and including them in
  compatible design spaces.
- Cross-platform continuous integration and wheel-installation checks.
- Citation, contribution, conduct, and security documentation.
- End-to-end reproducibility tests for seeded runs.
- `make_problem` and a runtime-checkable problem-definition protocol for user
  objectives.
- Versioned, validated JSON import/export for designed algorithms.
- Frozen numerical regressions against the official CEC 2013 C reference.
- Behavioral tests for public problem contracts, configuration failures,
  evaluation modes, serialization validation, selection, archives, and solve
  helpers, with a 90% CI line-coverage floor.
- Canonical Sphinx/Read the Docs sources in the main repository, with a
  warning-clean documentation CI job.
- Executable synthetic reference models and examples for the paper's material
  stacking and RIS passive-beamforming application classes.
- Opt-in evaluator retries, isolated hard timeouts, failure penalties,
  deterministic caching, and structured JSON Lines event logging.
- Ordered `EvalWorkers` thread parallelism for candidate batches, including
  in-flight duplicate suppression when deterministic caching is enabled.
- Atomic `experiment.json` manifests recording environment and normalized
  invocation provenance.
- Atomic, run-scoped checkpoint/resume for static and sequential Solve
  workflows and the outer Design search, including optimizer, inner-search,
  stage, surrogate, budget, and random-generator state.
- Configuration-driven application repeatability harness retaining every raw
  run and seed, plus trusted-publishing release automation for tagged builds.
- Incremental mypy checks for public/core execution modules and a lightweight
  core-runtime regression gate in continuous integration.
- Ruff formatting and static-analysis checks for source, tests, examples, and
  benchmark utilities.
- Fault-injection regressions proving that interrupted static Solve,
  sequential Solve, and Design runs resume identically to uninterrupted seeded
  baselines, plus parallel ordering, retry, logging, caching, and determinism
  stress tests.
- Portable algorithm profiles for the three CEC2013 Alg* structures fully
  disclosed in the software paper (f6, f13, and f21).
- A faithful Python port of the historical RIS zero-forcing/water-filling
  objective and an optional loader for user-supplied MATLAB channel records.
- A time-bounded CEC2013 paper-performance parity benchmark with machine-readable
  summary results.

### Changed

- Established Python as the canonical implementation.
- Standardized the project license as Apache-2.0.
- Routed stochastic operations through one NumPy random generator per run.
- Restored stateful CMA parameter search for single-algorithm design and
  polynomial parameter mutation for population-level design.
- Preserved MATLAB-style `AlgP`, `AlgQ`, and `AlgN` settings during Python
  namespace normalization.
- Stabilized covariance handling in CMA-based components.
- Updated project metadata and documentation for version 1.2.0.
- Made public options case- and underscore-insensitive, rejected unknown
  options, and validated budgets before execution.
- Enforced `ProbFE` as a hard per-stage objective-evaluation budget and aligned
  `AlgFE` with MATLAB as the post-initialization proposal budget.
- Restored MATLAB mode-specific defaults and per-stage sequential budgets.
- Corrected rotations, transforms, buffer ordering, and composition indexing in
  all CEC 2013 functions to match the official reference implementation.
- Made intensification and surrogate-assisted modes perform exact held-out
  final evaluation and removed fixed hidden surrogate budgets.
- Added contextual errors for failing, vector-valued, and non-finite objective
  results.
- Restored paper-listed arithmetic crossover, simulated-binary crossover, and
  EDA operators to the continuous automated-design space.
- Corrected serial multi-step execution so every search call updates the
  population before the next local iteration or search step.
- Prevented intensification from shrinking candidate performance matrices
  after evaluating only the first of multiple training instances.
- Restored one-value-per-algorithm performance aggregation for design-level
  CMA updates across multiple problem instances and runs.
- Aligned disturbance probabilities and shared pathway operators, statistical
  racing/intensification, AUC checkpoints, historical-best tracking, embedded
  archives, shared component auxiliary state, embedding order, and operator
  details with the original MATLAB implementation.
- Added problem-specific repair hooks, uncertain-problem aggregation, and
  retained evaluator accessory data on solutions.
- Added an optional MATLAB-equivalent 1000-tree random-forest surrogate while
  retaining a NumPy-only deterministic fallback for lightweight installs.

### Removed

- Dependence on benchmark files in the historical MATLAB source tree.
- Pickle as the only algorithm interchange format (legacy pickle loading is
  retained with an explicit security warning).

## 1.1.0 - 2025-09-15

- Added the initial Python translation alongside the MATLAB implementation.

## 1.0.0 - 2023-10-18

- Initial public MATLAB release.

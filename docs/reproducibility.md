# Reproducibility

Every experiment should use an explicit `Seed` and a separate `OutputDir`.
AutoOptLib writes `experiment.json` with the package, Python, NumPy, platform,
mode, normalized options, and instance identifiers. Keep this manifest with the
portable `Algorithm_*.json` files and result tables.

For paper results, archive:

- the exact source revision or release tag;
- the train/test split and all instance-generation seeds;
- the complete AutoOptLib call and evaluation budgets;
- raw per-run results, not only means and standard deviations;
- external simulator and operating-system versions;
- a checksum or durable DOI for non-public datasets.

The application examples use synthetic data generators so that CI and new
users can run them. They demonstrate the same *problem classes* as the paper,
but do not reproduce the paper's proprietary stacking dataset or its original
beamforming channel files. Published numerical claims must be regenerated from
the archived research instances.

The configuration-driven harness in `examples/reproducibility/` executes the
disclosed discrete baselines and retains every seed and raw run. Its default
configuration is a smoke benchmark. Create a separate archival configuration
for the full budgets, run counts, real test instances, and designed algorithm
JSON files used by a publication.

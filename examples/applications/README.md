# Application examples

These scripts are executable integration examples for the two application
classes discussed in the AutoOptLib paper. They deliberately generate public,
synthetic instances. They do **not** reproduce the paper's industrial records,
wireless channel files, baseline implementations, or numerical result tables.

Run from an installed checkout:

```bash
python examples/applications/material_stacking.py
python examples/applications/ris_beamforming.py
```

For a research reproduction, replace the generators with an archived mapping
of instance IDs to domain data and retain fixed, disclosed training/test IDs.

The `algorithms/` directory contains validated JSON encodings of the two
Alg* structures printed as pseudocode in the paper. They preserve the reported
components and numeric parameters. They do not by themselves reproduce the
paper tables because the research instance data are separate inputs.

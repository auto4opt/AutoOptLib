# Application adapters

AutoOptLib includes problem-definition code for two engineering examples.
Both accept user-provided instances and provide seeded synthetic generators.
No industrial dataset or measured experiment results are included.

## Material stacking

`generate_stacking_instance` and `make_material_stacking_problem` define
rack-level, position and orientation decisions. The model accounts for rack
space, shelf capacity and configurable grouping preferences. Replace synthetic
instances and weights with the rules of the application being solved.

## RIS beamforming

`generate_ris_instance` and `make_ris_beamforming_problem` define quantized
reconfigurable intelligent surface phase decisions for a multi-user downlink.
The adapter evaluates the effective channel, beamforming and sum rate.
`load_ris_matlab` can read suitably licensed user-supplied MATLAB data after
installing `.[applications]`.

The generated examples demonstrate software integration; they do not reproduce
published engineering results. Both factories support explicit CPU objective
adapters through `cpu=True`. See their docstrings and the
[CPU execution guide](cpu-execution.md).

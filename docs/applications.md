# Application problems

The package includes executable, synthetic reference implementations for the
two problem classes reported in the software paper.

## Raw-material stacking

`autooptlib.applications.stacking` models discrete rack level, within-level
position, and orientation decisions. Its objective combines occupied rack
space with configurable ergonomic and grouping preferences, while explicit
constraints enforce shelf width and weight limits.

The industrial dataset and its complete company-specific rule weights are not
in this repository. The included generator is therefore a transparent
surrogate for testing integration and teaching the workflow, not a source for
reproducing the numerical table in the paper.

## RIS passive beamforming

`autooptlib.applications.beamforming` optimizes quantized RIS phase shifts for
a MU-MISO downlink. It constructs the effective channel, uses normalized
zero-forcing and water-filling at the base station, and minimizes the reciprocal
weighted sum rate. This matches the evaluation structure in the historical
MATLAB application. A seeded Rayleigh-channel generator makes the example
reproducible.

The published experiment used channel data stored in the historical v1.1 tag.
Those files carried GPL notices while the Python package is Apache-2.0, so they
are not silently redistributed in the 1.2 wheel pending an explicit licensing
decision. Consequently, the generated example validates the public software
workflow but does not claim numerical equivalence to the paper.

If a user has a suitably licensed copy of the historical `Beanforming.mat`, it
can be loaded without changing the problem code:

```python
from autooptlib import load_ris_matlab, make_ris_beamforming_problem

instances = load_ris_matlab("Beanforming.mat")
problem = make_ris_beamforming_problem(instances)
```

Install the optional reader with `pip install "autooptlib[applications]"`.

Runnable scripts are located in `examples/applications/`. Replace the generated
instances with domain data while retaining the same train/test separation.

# CEC 2013 benchmark data

`cec2013.npz` contains the shift vectors and rotation matrices used by the
AutoOptLib Python translation of the CEC 2013 real-parameter optimization
benchmark. The archive is a lossless compressed conversion of the text data
distributed with the historical AutoOptLib MATLAB release.

The conversion contains `shift_data` plus ten rotation matrices for each
supported dimension (2, 5, 10, and multiples of 10 through 100). The Python
implementation is regression-tested at fixed points against the organizers'
official C implementation, rather than only checking the known optima.

The arrays are loaded through `importlib.resources`, so the benchmark behaves
the same in source checkouts and installed wheels.

Before redistributing this dataset independently of AutoOptLib, verify the
original CEC benchmark's data-distribution terms. Apache-2.0 covers AutoOptLib's
source code but does not replace any third-party terms that may apply to the
benchmark data.

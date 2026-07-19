# Application repeatability harness

`application_benchmarks.py` runs four disclosed discrete baselines on seeded
synthetic instances for both application classes. It preserves every run's
fitness and seed, a grouped summary, each run's `experiment.json`, and the
exact configuration.

```bash
python examples/reproducibility/application_benchmarks.py \
  --config examples/reproducibility/application-smoke.json \
  --output results/repeatability
```

The default configuration is intentionally small enough for CI-style checks.
It is not sized for scientific comparison. A paper artifact should provide a
separate configuration with the archived real instance identifiers, 30 runs,
the reported 50,000-evaluation budget, and the designed algorithm JSON files.
Designed algorithms can be added to `algorithms` as
`{"label": "Alg*", "file": "path/to/Algorithm_1.json", "applications": ["ris_beamforming"]}`.

The `algorithms` directory also contains portable profiles reconstructed from
the manuscript's complete pseudocode for CEC2013 f6, f13, and f21. The other
25 CEC2013 Alg* structures are not fully disclosed in the manuscript, so they
cannot be faithfully reconstructed from the paper alone.

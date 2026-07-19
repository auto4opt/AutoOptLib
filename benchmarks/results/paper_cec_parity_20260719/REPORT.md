# CEC2013 paper-performance parity check

## Decision

The disclosed Python `Alg*` profiles execute reliably and often meet or beat
the mean objective value reported in the paper, but this experiment does **not**
establish numerical reproduction of the paper table.

- All 270 runs completed without exceptions or non-finite fitness values.
- Total wall time was 1,224.5 seconds (20 minutes 24.5 seconds), below the
  one-hour limit.
- For minimization, the Python mean met or beat the paper mean in 6 of the 9
  tested function/dimension cases.
- The closest mean match was CEC2013 f6 at D=10 (absolute difference 21.66),
  f13 at D=30 (31.50), and f21 at D=30 (85.53).
- Only the f13/D=30 mean is statistically compatible with the published mean
  under an approximate comparison made from the two sets of summary statistics.
  The other cases differ materially in mean, variance, or both.

The appropriate status is therefore **stable execution; paper-performance
threshold frequently attained; numerical reproduction not yet established**.

## Method

The paper's exact test-stage settings were used:

- problems: CEC2013 f6, f13, and f21, whose `Alg*` pseudocodes are disclosed;
- candidate population: 50;
- budget per run: `1000 * D` objective evaluations;
- independent runs: 30;
- dimensions tested: 10, 30, and 50;
- evaluator: AutoOptLib's current official-C-compatible CEC2013 implementation;
- algorithm profiles: the serialized profiles under
  `examples/reproducibility/algorithms/`.

All three dimensions were tested because the paper says that two dimensions
were randomly selected for algorithm design and the remaining dimension was
used for testing, but it does not disclose the selected split for each
function. Seeds were fixed by the benchmark script for repeatability; they
cannot match the paper because its seeds are not published.

This is an **execution-parity** experiment. It evaluates the algorithm
structures reported by the paper. It does not repeat automated algorithm
design: the paper's design stage would require approximately 500 million
objective evaluations per function (`5000 proposals * 2 instances * 10 runs *
5000 evaluations`), which cannot be completed locally within one hour.

## Results

Lower fitness is better. `Gap ratio` is the Python distance from the known
optimum divided by the paper's distance from the optimum; values below 1 mean
that Python performed better than the published mean.

| Function | D | Python mean | Python sample SD | Paper mean | Paper SD | Gap ratio | Meets paper mean? |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| f6 | 10 | -890.15 | 14.22 | -868.49 | 8.87 | 0.312 | yes |
| f6 | 30 | -836.98 | 22.68 | -868.49 | 8.87 | 2.000 | no |
| f6 | 50 | -796.05 | 39.21 | -868.49 | 8.87 | 3.299 | no |
| f13 | 10 | -170.98 | 7.07 | 5.90 | 98.80 | 0.141 | yes |
| f13 | 30 | -25.60 | 62.22 | 5.90 | 98.80 | 0.847 | yes |
| f13 | 50 | -111.62 | 74.44 | 5.90 | 98.80 | 0.429 | yes |
| f21 | 10 | 1023.10 | 66.46 | 1113.42 | 4.15 | 0.577 | yes |
| f21 | 30 | 1027.89 | 46.72 | 1113.42 | 4.15 | 0.599 | yes |
| f21 | 50 | 1319.03 | 324.98 | 1113.42 | 4.15 | 1.963 | no |

The results are not merely uniformly worse than the paper. Several are much
better, while some dimensions are worse, and the f21 variance is substantially
larger than reported. That pattern is more consistent with an incomplete
reproduction specification and known implementation differences than with a
single systematic failure in the Python runner.

## Interpretation and limitations

The experiment supports these claims:

1. The three published algorithm structures run stably under the paper's
   test-stage population, budget, and replication count.
2. Their performance is in the expected optimization range, and six of nine
   dimension cases attain or improve on the published mean.
3. The evidence is insufficient to claim that Python numerically reproduces
   the paper table.

The main unresolved provenance gaps are:

- the per-function train/test dimension split is not published;
- random seeds and raw 30-run observations are not published;
- the paper used the original MATLAB CEC evaluator, whereas the Python package
  intentionally retains the official-C-compatible evaluator;
- NumPy and MATLAB use different random-number streams;
- several Python components deliberately correct or harden MATLAB behavior
  (for example, per-pair one-point crossover and CMA covariance handling).

Consequently, differences from the paper table cannot, on their own, be
classified as Python defects. A defensible exact-reproduction claim requires
publishing the original split, seeds, evaluator/data version, complete
serialized algorithm profiles, and raw run-level results. A direct paired
MATLAB/Python trace test would then be the decisive next validation.

## Reproduction

From the repository root:

```bash
.venv/bin/python benchmarks/paper_cec_parity.py \
  --output benchmarks/results/paper_cec_parity_20260719 \
  --runs 30 \
  --seed 20260719 \
  --max-seconds 3300
```

Machine-readable results are in `results.json` and `summary.csv`.
`run_fitness.csv` retains all 270 final run-level fitness values. Large
generated solution matrices and pickle files are intentionally not versioned.

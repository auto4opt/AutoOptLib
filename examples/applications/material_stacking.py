"""Small, reproducible material-stacking workflow (not the proprietary data)."""

from autooptlib import (
    autoopt,
    generate_stacking_instance,
    make_material_stacking_problem,
)

instances = {
    "train-a": generate_stacking_instance(12, seed=11),
    "train-b": generate_stacking_instance(14, seed=12),
    "test-a": generate_stacking_instance(13, seed=21),
}
problem = make_material_stacking_problem(instances)

algorithms, _ = autoopt(
    Mode="design",
    Problem=problem,
    InstanceTrain=["train-a", "train-b"],
    InstanceTest=["test-a"],
    AlgN=1,
    AlgFE=10,
    AlgRuns=2,
    ProbN=10,
    ProbFE=200,
    Seed=2026,
    OutputDir="results/material-stacking",
)
print(algorithms[0].operator_pheno)

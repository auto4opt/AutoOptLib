"""Small, reproducible RIS workflow using generated Rayleigh channels."""

from autooptlib import autoopt, generate_ris_instance, make_ris_beamforming_problem

instances = {
    "train-12": generate_ris_instance(12, seed=11),
    "train-16": generate_ris_instance(16, seed=12),
    "test-20": generate_ris_instance(20, seed=21),
}
problem = make_ris_beamforming_problem(instances)

algorithms, _ = autoopt(
    Mode="design",
    Problem=problem,
    InstanceTrain=["train-12", "train-16"],
    InstanceTest=["test-20"],
    AlgN=1,
    AlgFE=10,
    AlgRuns=2,
    ProbN=10,
    ProbFE=200,
    Seed=2026,
    OutputDir="results/ris-beamforming",
)
print(algorithms[0].operator_pheno)

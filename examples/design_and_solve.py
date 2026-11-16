"""Small end-to-end example; output is written to the chosen directory."""
import argparse
from pathlib import Path

from autooptlib import DesignSpace, autoopt, load_algorithm, make_problem, save_algorithm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/example"))
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    backend = {"ComponentBackend": "native", "StreamBackend": "batch"} if args.native else {}
    problem = make_problem(lambda x, _: float((x * x).sum()), bounds=(-5, 5))
    space = DesignSpace(
        choose=("choose_tournament", "choose_random"),
        search=("cross_point_one", "search_mu_uniform", "search_de_random"),
        update=("update_greedy", "update_pairwise"),
        max_pathways=2, max_stages=2,
        population_sizes=(10, 20), offspring_sizes=(10, 20),
    )
    algorithms, _ = autoopt(
        Mode="design", Problem=problem, InstanceTrain=[8], InstanceTest=[12],
        DesignSpace=space, ProbN=20, ProbFE=100, AlgN=1, AlgFE=4, AlgRuns=1,
        Seed=2026, OutputDir=args.output / "design", **backend,
    )
    path = args.output / "algorithm.json"
    save_algorithm(algorithms[0], path)
    best, _ = autoopt(
        Mode="solve", Problem=problem, InstanceSolve=[12],
        Algorithm=load_algorithm(path), ProbFE=200, AlgRuns=1,
        Seed=2026, OutputDir=args.output / "solve", **backend,
    )
    print(f"Saved algorithm: {path}")
    print(f"Solved instance groups: {len(best)}")


if __name__ == "__main__":
    main()

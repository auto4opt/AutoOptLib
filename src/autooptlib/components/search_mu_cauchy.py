"""Self-adaptive Cauchy mutation component."""

from ._adaptive_mutation import execute_adaptive_mutation, specification


def search_mu_cauchy(*args):
    return execute_adaptive_mutation(args, "cauchy_eta", "standard_cauchy")


search_mu_cauchy.__execution_computation__ = specification(search_mu_cauchy, "cauchy_eta", "standard_cauchy")

"""Self-adaptive Gaussian mutation component."""

from ._adaptive_mutation import execute_adaptive_mutation, specification


def search_mu_gaussian(*args):
    return execute_adaptive_mutation(args, "gaussian_eta", "standard_normal")


search_mu_gaussian.__execution_computation__ = specification(search_mu_gaussian, "gaussian_eta", "standard_normal")

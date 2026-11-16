"""Regression coverage for native access to CPython built-in type dictionaries."""
from types import SimpleNamespace

import pytest


def test_native_configuration_reads_builtin_type_dictionaries():
    native = pytest.importorskip("autooptlib.runtime._native._native_compatibility")
    configuration = dict(population_size=10, offspring_size=10,
                         crossover_rate=1.0, mutation_rate=1.0,
                         boundary_handling="clip")
    setting = SimpleNamespace(ProbN=10, ProbFE=100)
    assert native._prepare_configuration(configuration, setting) == configuration


def test_native_configuration_falls_back_for_nonstandard_namespace():
    native = pytest.importorskip("autooptlib.runtime._native._native_compatibility")

    class Settings(SimpleNamespace):
        pass

    assert native._prepare_configuration({}, Settings(ProbN=10)) is None

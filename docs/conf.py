"""Sphinx configuration for the canonical AutoOptLib documentation."""

from __future__ import annotations

from autooptlib import __version__

project = "AutoOptLib"
author = "AutoOptLib contributors"
copyright = "2026, AutoOptLib contributors"
release = __version__

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
myst_enable_extensions = ["colon_fence", "deflist"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build"]
html_theme = "sphinx_rtd_theme"
html_title = f"AutoOptLib {release}"

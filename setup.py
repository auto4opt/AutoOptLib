"""Optional native stream execution; the Python backend needs no compiler."""

import os
from pathlib import Path

import numpy as np
import pybind11
from setuptools import Extension, setup

flags = (
    ["/O2", "/std:c++17", "/fp:strict"]
    if os.name == "nt"
    else ["-O3", "-std=c++17", "-fno-fast-math", "-ffp-contract=off"]
)
setup(
    ext_modules=[] if os.environ.get("AUTOOPTLIB_BUILD_NATIVE") == "0" else [
        Extension(
            "autooptlib.runtime._native._native_compatibility",
            sources=["src/autooptlib/runtime/_native/compatibility.cpp"],
            depends=[
                str(path)
                for path in Path("src/autooptlib/runtime/_native").glob("*.hpp")
            ],
            include_dirs=[pybind11.get_include(), np.get_include()],
            define_macros=[("AOL_NUMPY_VERSION", '"' + np.__version__ + '"')],
            library_dirs=[
                os.path.join(os.path.dirname(np.__file__), "random", "lib"),
                os.path.join(np.get_include(), "..", "lib"),
            ],
            libraries=["npyrandom", "npymath"],
            language="c++",
            extra_compile_args=flags,
            optional=True,
        ),
        Extension(
            "autooptlib.runtime._native._native_stream",
            sources=["src/autooptlib/runtime/_native/engine.cpp"],
            depends=[
                "src/autooptlib/runtime/_native/" + name
                for name in [
                    "data.hpp",
                    "generation.hpp",
                    "operators.hpp",
                    "updates.hpp",
                    "profile.hpp",
                    "simd.hpp",
                    "blocks.hpp",
                    "program.hpp",
                    "dependencies.hpp",
                ]
            ],
            include_dirs=[pybind11.get_include()],
            language="c++",
            extra_compile_args=flags,
            optional=True,
        ),
    ]
)

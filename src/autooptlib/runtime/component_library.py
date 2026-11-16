"""Load the precompiled component library once; no per-graph compilation."""

import os
import sys
from importlib import import_module, util
from pathlib import Path

_NAME = "autooptlib.runtime._native._native_compatibility"


def load_component_library():
    if _NAME in sys.modules:
        return sys.modules[_NAME]
    directory = os.environ.get("AOL_COMPONENT_BUILD")
    if directory:
        candidates = [
            *Path(directory).glob("_native_compatibility*.so"),
            *Path(directory).glob("_native_compatibility*.pyd"),
        ]
        if len(candidates) != 1:
            raise ImportError(
                "AOL_COMPONENT_BUILD must contain one compiled component library"
            )
        spec = util.spec_from_file_location(_NAME, candidates[0])
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load native component library: {candidates[0]}")
        module = util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sys.modules[_NAME] = module
        return module
    return import_module(_NAME)

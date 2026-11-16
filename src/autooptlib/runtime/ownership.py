"""Conservative ownership admission for isolated native evaluation requests."""
from pathlib import Path, PurePath, PosixPath, PurePosixPath, WindowsPath, PureWindowsPath
from types import SimpleNamespace, FunctionType, BuiltinFunctionType, CodeType, GetSetDescriptorType
import sys

import numpy as np


def _python_mutable_graph(root, record_types):
    """Return mutable owner identities, or None for an opaque copy protocol."""
    atomic = {id(t) for t in (type(None), bool, int, float, complex, str, bytes,
                              range, type, FunctionType, BuiltinFunctionType, CodeType)}
    records = {id(t): t for t in record_types}
    paths = {id(t) for t in (Path, PurePath, PosixPath, PurePosixPath, WindowsPath, PureWindowsPath)}
    bit_generators = {id(t) for t in (np.random.PCG64, np.random.MT19937, np.random.Philox, np.random.SFC64)}
    containers = {id(t) for t in (list, set, frozenset)}
    scalars = {id(t) for t in np.sctypeDict.values() if isinstance(t, type)}
    owners, seen, pending = set(), set(), [root]
    hooks = {'__deepcopy__', '__reduce__', '__reduce_ex__', '__getstate__', '__setstate__',
             '__new__', '__getnewargs__', '__getnewargs_ex__'}
    while pending:
        value = pending.pop()
        cls, identity = type(value), id(value)
        kind = id(cls)
        if kind in atomic or identity in seen:
            continue
        seen.add(identity)
        if cls is tuple:
            pending.extend(value)
            continue
        owners.add(identity)
        if cls is dict:
            for key, item in value.items():
                pending.extend((key, item))
        elif kind in containers:
            pending.extend(value)
        elif cls is np.ndarray:
            if value.dtype.hasobject or value.base is not None:
                return None
        elif cls is np.random.Generator:
            if id(type(value.bit_generator)) not in bit_generators:
                return None
            owners.add(id(value.bit_generator))
        elif kind in scalars:
            if value.dtype.hasobject:
                return None
        elif kind in paths:
            if any('__deepcopy__' in vars(base) for base in cls.__mro__):
                return None
            reduce = cls.__reduce__
            if (cls.__reduce_ex__ is not object.__reduce_ex__
                    or type(reduce) is not FunctionType
                    or reduce.__module__ != 'pathlib'
                    or reduce.__qualname__ != 'PurePath.__reduce__'):
                return None
        elif kind in records:
            if type(cls) is not type:
                return None
            if cls is not SimpleNamespace and cls.__getattribute__ is not object.__getattribute__:
                return None
            if cls is not SimpleNamespace and type(vars(cls).get('__dict__')) is not GetSetDescriptorType:
                return None
            if any(name in vars(base) for base in cls.__mro__
                   if base is not object and base is not SimpleNamespace for name in hooks):
                return None
            fields = object.__getattribute__(value, '__dict__')
            if type(fields) is not dict or not all(type(key) is str for key in fields):
                return None
            if any(name in fields for name in hooks):
                return None
            pending.append(fields)
        else:
            return None
    return owners


def _mutable_graph(root, record_types):
    library = sys.modules.get('autooptlib.runtime._native._native_compatibility')
    if library is not None and hasattr(library, '_mutable_owners'):
        return library._mutable_owners(root, record_types)
    return _python_mutable_graph(root, record_types)


def copy_evaluation_task(task):
    """Copy standard task graphs natively; retain opaque deepcopy protocols."""
    from copy import deepcopy
    from ..utils.design import Design
    from ..utils.design._helpers import SearchParam, SearchStep
    from ..utils.design._stream_graph import StreamPathway, StreamPathwayParam, StreamStage, StreamStageParam
    from .design import _DesignTask
    from .shared import SharedArraySpec

    library = sys.modules.get('autooptlib.runtime._native._native_compatibility')
    if library is None or type(task.setting) is not SimpleNamespace:
        return deepcopy(task)
    values = vars(task.setting)
    if (type(values.get('ComponentBackend')) is not str
            or values.get('ComponentBackend') != 'native'
            or type(values.get('StreamBackend', 'batch')) is not str
            or values.get('StreamBackend', 'batch') != 'batch'):
        return deepcopy(task)
    records = [Design, SimpleNamespace, SearchParam, SearchStep, StreamPathway,
               StreamPathwayParam, StreamStage, StreamStageParam, _DesignTask, SharedArraySpec]
    ioh = sys.modules.get('autooptlib.problems.ioh')
    if ioh is not None and 'IOHInstance' in vars(ioh):
        records.append(vars(ioh)['IOHInstance'])
    return library._copy_task(task, records)


def transfer_owned_setting(design, setting):
    """Admit transfer only after transport/copy already isolated the request.

    Mutable cross-links between candidate and setting require a second copy:
    transferring those would change which candidate fields a callback mutates.
    Unknown objects retain their original deepcopy protocol.
    """
    from ..utils.design import Design
    from ..utils.design._evaluate import _EVALUATE_IMPLEMENTATION
    from ..utils.design._helpers import SearchParam, SearchStep
    from ..utils.design._stream_graph import StreamPathway, StreamPathwayParam, StreamStage, StreamStageParam

    if type(design) is not Design or type(setting) is not SimpleNamespace:
        return None
    implementation, code = _EVALUATE_IMPLEMENTATION
    if Design.evaluate is not implementation or implementation.__code__ is not code:
        return None
    values = vars(setting)
    if (not all(type(key) is str for key in values)
            or type(values.get('ComponentBackend')) is not str
            or values.get('ComponentBackend') != 'native'
            or type(values.get('StreamBackend', 'batch')) is not str
            or values.get('StreamBackend', 'batch') != 'batch'
            or values.get('_NativeBatchReference', False) is not False):
        return None
    records = [Design, SimpleNamespace, SearchParam, SearchStep, StreamPathway,
               StreamPathwayParam, StreamStage, StreamStageParam]
    ioh = sys.modules.get('autooptlib.problems.ioh')
    if ioh is not None and 'IOHInstance' in vars(ioh):
        records.append(vars(ioh)['IOHInstance'])
    setting_owners = _mutable_graph(setting, records)
    if setting_owners is None:
        return None
    candidate_owners = _mutable_graph(design, records)
    if candidate_owners is None or not setting_owners.isdisjoint(candidate_owners):
        return None
    return setting

#pragma once
#include <Python.h>

// CPython 3.12 stores some static built-in type dictionaries per interpreter.
// Read through the public accessor instead of dereferencing a null tp_dict.
// The returned dictionary is borrowed and must remain read-only.
inline PyObject *aol_type_dict(PyTypeObject *type) {
#if PY_VERSION_HEX >= 0x030C0000
  return PyType_GetDict(type);
#else
  return type->tp_dict;
#endif
}

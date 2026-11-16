// Evaluate the existing frozen dependency proof without Python generator scans.
#pragma once
#include "python_compat.hpp"
#include <pybind11/pybind11.h>
#include <vector>

namespace cpu_program {
namespace py = pybind11;
struct Binding {
  py::dict names;
  py::object key, expected;
  explicit Binding(py::handle row)
      : names(row[py::int_(0)]), key(row[py::int_(1)]), expected(row[py::int_(2)]) {}
  bool same() const { return PyDict_GetItem(names.ptr(), key.ptr()) == expected.ptr(); }
};
struct DependencyProof {
  py::object numeric_reader;
  py::list functions, cells, classes;
  bool complete;
  py::dict numpy, operators, builtins;
  std::vector<Binding> bindings, builtin_lookups;
  explicit DependencyProof(py::object guard)
      : functions(guard.attr("functions")), cells(guard.attr("cells")),
        classes(guard.attr("classes")), complete(py::cast<bool>(guard.attr("complete"))) {
    auto module = py::module_::import("autooptlib.components._execution");
    numeric_reader = module.attr("_NUMERIC_OPERATOR_READER");
    operators = module.attr("_NUMERIC_OPERATORS");
    numpy = py::module_::import("numpy").attr("__dict__");
    builtins = py::module_::import("builtins").attr("__dict__");
    for (auto row : guard.attr("bindings")) bindings.emplace_back(row);
    for (auto row : guard.attr("builtin_lookups")) builtin_lookups.emplace_back(row);
  }
  bool trusted(bool numeric=true) const {
    if (!complete) return false;
    if(numeric) {
    auto reader = PyDict_GetItemString(numpy.ptr(), "set_numeric_ops");
    if ((reader ? reader : Py_None) != numeric_reader.ptr()) return false;
    if (!numeric_reader.is_none()) {
      py::dict current = numeric_reader();
      if (current.size() != operators.size()) return false;
      for (auto item : operators)
        if (PyDict_GetItem(current.ptr(), item.first.ptr()) != item.second.ptr()) return false;
    }
    }
    for (const auto &b : builtin_lookups) {
      auto value = PyDict_GetItem(b.names.ptr(), b.key.ptr());
      if (!value) value = PyDict_GetItem(builtins.ptr(), b.key.ptr());
      if (value != b.expected.ptr()) return false;
    }
    for (auto row : cells) {
      py::tuple pair = py::reinterpret_borrow<py::tuple>(row);
      if (PyCell_GET(pair[0].ptr()) != pair[1].ptr()) return false;
    }
    for (const auto &b : bindings) if (!b.same()) return false;
    for (auto row : functions) {
      py::tuple f = py::reinterpret_borrow<py::tuple>(row);
      if (PyFunction_GET_CODE(f[0].ptr()) != f[1].ptr()) return false;
      auto defaults = PyFunction_GET_DEFAULTS(f[0].ptr());
      if ((defaults ? defaults : Py_None) != f[2].ptr()) return false;
      auto keywords = PyFunction_GET_KW_DEFAULTS(f[0].ptr());
      py::tuple expected = f[3];
      if ((keywords ? PyDict_Size(keywords) : 0) != py::len(expected)) return false;
      for (auto item : expected) {
        py::tuple pair = py::reinterpret_borrow<py::tuple>(item);
        if (PyDict_GetItem(keywords, pair[0].ptr()) != pair[1].ptr()) return false;
      }
    }
    for (auto row : classes) {
      py::tuple pair = py::reinterpret_borrow<py::tuple>(row);
      auto dictionary = aol_type_dict(reinterpret_cast<PyTypeObject *>(pair[0].ptr()));
      for (auto item : pair[1]) {
        py::tuple attr = py::reinterpret_borrow<py::tuple>(item);
        if (PyDict_GetItem(dictionary, attr[0].ptr()) != attr[1].ptr()) return false;
      }
    }
    return true;
  }
};
} // namespace cpu_program

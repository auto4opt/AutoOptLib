#pragma once
#include "python_compat.hpp"
#include <cctype>
namespace native_components {
// Alias spellings are static metadata. Values are read at their original
// execution points, including after callbacks; no winning spelling is cached.
struct BatchAccess {
  std::unordered_map<std::string,std::vector<py::str>> spellings;
  py::object generator_type,integer_type,default_rng,namespace_type;
  explicit BatchAccess(py::object numpy) {
    auto variants=py::module_::import("autooptlib.utils._attributes").attr("name_variants");
    for(auto name:{"rng","random_state","_AlgorithmOffspringSize","_AlgorithmCrossoverRate","_AlgorithmMutationRate","setting","_AlgorithmBoundaryHandling"})
      for(auto value:variants(name))spellings[name].push_back(py::reinterpret_borrow<py::str>(value));
    generator_type=numpy.attr("random").attr("Generator");integer_type=numpy.attr("integer");default_rng=numpy.attr("random").attr("default_rng");
    namespace_type=py::module_::import("types").attr("SimpleNamespace");
  }
  static std::string normalized(const std::string &s) {
    std::string out;for(unsigned char c:s)if((c>='a'&&c<='z')||(c>='0'&&c<='9'))out+=c;else if(c>='A'&&c<='Z')out+=c-'A'+'a';return out;
  }
  py::object flex(py::handle object,const char *name,py::object fallback)const {
    if(object.is_none())return fallback;
    py::object fields;
    const bool exact_dict=PyDict_CheckExact(object.ptr());
    bool ordinary=exact_dict||Py_TYPE(object.ptr())==reinterpret_cast<PyTypeObject *>(namespace_type.ptr());
    if(ordinary)fields=exact_dict?py::reinterpret_borrow<py::object>(object):py::reinterpret_steal<py::object>(PyObject_GenericGetDict(object.ptr(),nullptr));
    for(auto &candidate:spellings.at(name)) {
      if(ordinary) {
        // Exact strings cannot invoke user hash/equality on immutable built-in
        // class dictionaries. Dict fields do not use their descriptors at all.
        bool class_field=(!exact_dict||!PyUnicode_CheckExact(candidate.ptr()))&&
          (PyDict_GetItem(aol_type_dict(Py_TYPE(object.ptr())),candidate.ptr())||PyDict_GetItem(aol_type_dict(&PyBaseObject_Type),candidate.ptr()));
        if(exact_dict||!class_field) {
          auto value=PyDict_GetItem(fields.ptr(),candidate.ptr());
          if(value) {
            if(exact_dict)return py::reinterpret_borrow<py::object>(value);
            // A namespace dictionary may contain a non-string key with an
            // observable equality method. Retain the second attribute lookup.
            auto second=PyObject_GetAttr(object.ptr(),candidate.ptr());if(!second)throw py::error_already_set();return py::reinterpret_steal<py::object>(second);
          }
          if(!class_field)continue;
        }
      }
      if(PyDict_Check(object.ptr())) {
        auto value=PyDict_GetItem(object.ptr(),candidate.ptr());if(value)return py::reinterpret_borrow<py::object>(value);
      }
      // Python get_flex uses hasattr followed by getattr, so property reads
      // occur twice. Keep that behavior for user-defined setting objects.
      auto first=PyObject_GetAttr(object.ptr(),candidate.ptr());
      if(first) {
        Py_DECREF(first);auto second=PyObject_GetAttr(object.ptr(),candidate.ptr());
        if(!second)throw py::error_already_set();return py::reinterpret_steal<py::object>(second);
      }
      if(!PyErr_ExceptionMatches(PyExc_AttributeError))throw py::error_already_set();PyErr_Clear();
    }
    py::object dictionary;
    if(ordinary)dictionary=fields;
    else if(PyDict_Check(object.ptr()))dictionary=py::reinterpret_borrow<py::object>(object);
    else {
      auto raw=PyObject_GetAttrString(object.ptr(),"__dict__");
      if(!raw){if(!PyErr_ExceptionMatches(PyExc_AttributeError)&&!PyErr_ExceptionMatches(PyExc_TypeError))throw py::error_already_set();PyErr_Clear();return fallback;}
      dictionary=py::reinterpret_steal<py::object>(raw);
    }
    auto target=normalized(name);
    for(auto key:dictionary)if(normalized(py::cast<std::string>(py::str(key)))==target) {
      if(PyDict_Check(object.ptr()))return py::reinterpret_borrow<py::object>(object[key]);
      auto raw=PyObject_GetAttr(object.ptr(),key.ptr());if(!raw)throw py::error_already_set();return py::reinterpret_steal<py::object>(raw);
    }
    return fallback;
  }
  py::object rng(py::handle setting)const {
    auto value=flex(setting,"rng",py::none());
    auto generator=[&](py::handle object){if(Py_TYPE(object.ptr())==reinterpret_cast<PyTypeObject *>(generator_type.ptr()))return true;int result=PyObject_IsInstance(object.ptr(),generator_type.ptr());if(result<0)throw py::error_already_set();return result!=0;};
    auto integer=[&](py::handle object){int result=PyObject_IsInstance(object.ptr(),integer_type.ptr());if(result<0)throw py::error_already_set();return PyLong_Check(object.ptr())||result!=0;};
    if(generator(value))return value;
    if(integer(value))return default_rng(py::int_(value));
    value=flex(setting,"random_state",py::none());if(generator(value))return value;
    if(integer(value))return default_rng(py::int_(value));return default_rng();
  }
  static bool same_parameter(py::handle left,py::handle right) {
    if(left.is_none()||right.is_none())return left.is_none()&&right.is_none();
    auto convert=[](py::handle value){auto raw=PyArray_FromAny(value.ptr(),nullptr,0,0,NPY_ARRAY_ENSUREARRAY,nullptr);if(!raw)throw py::error_already_set();return py::reinterpret_steal<py::array>(raw);};
    auto a=convert(left),b=convert(right);if(a.ndim()!=b.ndim())return false;
    for(py::ssize_t i=0;i<a.ndim();++i)if(a.shape(i)!=b.shape(i))return false;
    auto raw_equal=PyObject_RichCompare(a.ptr(),b.ptr(),Py_EQ);if(!raw_equal)throw py::error_already_set();auto equal=py::reinterpret_steal<py::object>(raw_equal);
    auto values=convert(equal);auto raw=PyArray_All(reinterpret_cast<PyArrayObject *>(values.ptr()),AOL_RAVEL_AXIS,nullptr);
    if(!raw)throw py::error_already_set();auto result=py::reinterpret_steal<py::object>(raw);return py::cast<bool>(result);
  }
};
}

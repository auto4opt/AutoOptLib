#pragma once
#include "python_compat.hpp"
namespace native_components {
struct NativeOwnership {
  py::object numpy,namespace_type,generator_type,path_reduce,path_reduce_code;
  std::unordered_set<PyTypeObject *> atomic,paths,scalars,bit_generators;
  explicit NativeOwnership(py::object np):numpy(std::move(np)),namespace_type(py::module_::import("types").attr("SimpleNamespace")),
      generator_type(numpy.attr("random").attr("Generator")) {
    atomic={Py_TYPE(Py_None),&PyBool_Type,&PyLong_Type,&PyFloat_Type,&PyComplex_Type,&PyUnicode_Type,&PyBytes_Type,
            &PyRange_Type,&PyType_Type,&PyFunction_Type,&PyCFunction_Type,&PyCode_Type};
    auto pathlib=py::module_::import("pathlib");
    for(auto name:{"Path","PurePath","PosixPath","PurePosixPath","WindowsPath","PureWindowsPath"})
      paths.insert(reinterpret_cast<PyTypeObject *>(pathlib.attr(name).ptr()));
    path_reduce=pathlib.attr("PurePath").attr("__reduce__");path_reduce_code=py::getattr(path_reduce,"__code__",py::none());
    auto random=numpy.attr("random");for(auto name:{"PCG64","MT19937","Philox","SFC64"})
      bit_generators.insert(reinterpret_cast<PyTypeObject *>(random.attr(name).ptr()));
    for(auto entry:py::cast<py::dict>(numpy.attr("sctypeDict")))if(PyType_Check(entry.second.ptr()))
      scalars.insert(reinterpret_cast<PyTypeObject *>(entry.second.ptr()));
  }
  py::object collect(py::object root,py::sequence record_types,bool copying=false)const {
    std::unordered_set<PyTypeObject *> records;for(auto type:record_types)if(PyType_Check(type.ptr()))records.insert(reinterpret_cast<PyTypeObject *>(type.ptr()));
    std::unordered_set<PyTypeObject *> checked_records;
    std::unordered_set<PyObject *> seen,owners;
    std::vector<PyObject *> pending{root.ptr()};std::vector<py::object> keepers;
    const char *hooks[]={"__deepcopy__","__reduce__","__reduce_ex__","__getstate__","__setstate__","__new__","__getnewargs__","__getnewargs_ex__"};
    while(!pending.empty()) {
      auto value=pending.back();pending.pop_back();auto type=Py_TYPE(value);
      if(atomic.count(type)||!seen.insert(value).second)continue;
      if(PyTuple_CheckExact(value)){for(Py_ssize_t i=0;i<PyTuple_GET_SIZE(value);++i)pending.push_back(PyTuple_GET_ITEM(value,i));continue;}
      if(!copying)owners.insert(value);
      if(PyDict_CheckExact(value)) {
        PyObject *key,*item;Py_ssize_t at=0;while(PyDict_Next(value,&at,&key,&item)){
          if(copying&&!PyUnicode_CheckExact(key)&&!PyLong_CheckExact(key))return py::none();
          pending.push_back(key);pending.push_back(item);
        }
      }else if(PyList_CheckExact(value)) {
        for(Py_ssize_t i=0;i<PyList_GET_SIZE(value);++i)pending.push_back(PyList_GET_ITEM(value,i));
      }else if(type==&PySet_Type||PyFrozenSet_CheckExact(value)) {
        auto iterator=py::iter(py::handle(value));while(auto item=PyIter_Next(iterator.ptr())){
          keepers.push_back(py::reinterpret_steal<py::object>(item));
          if(copying&&!atomic.count(Py_TYPE(item)))return py::none();
          pending.push_back(item);
        }
        if(PyErr_Occurred())throw py::error_already_set();
      }else if(PyArray_CheckExact(value)) {
        auto array=reinterpret_cast<PyArrayObject *>(value);
        if((!copying&&PyArray_BASE(array))||PyDataType_FLAGCHK(PyArray_DESCR(array),NPY_ITEM_REFCOUNT))return py::none();
      }else if(copying&&PyArray_DescrCheck(value)) {
        auto dtype=py::reinterpret_borrow<py::object>(value);
        if(!dtype.attr("metadata").is_none()||!dtype.attr("fields").is_none()||!dtype.attr("subdtype").is_none())return py::none();
      }else if(type==reinterpret_cast<PyTypeObject *>(generator_type.ptr())) {
        auto generator=py::reinterpret_borrow<py::object>(value).attr("bit_generator");
        if(!bit_generators.count(Py_TYPE(generator.ptr())))return py::none();if(!copying)owners.insert(generator.ptr());
      }else if(scalars.count(type)) {
        if(copying&&PyArray_IsScalar(value,Void))return py::none();
        auto raw=PyArray_DescrFromScalar(value);if(!raw)throw py::error_already_set();
        auto dtype=py::reinterpret_steal<py::object>(reinterpret_cast<PyObject *>(raw));
        if(PyDataType_FLAGCHK(raw,NPY_ITEM_REFCOUNT))return py::none();
      }else if(paths.count(type)) {
        auto mro=py::reinterpret_borrow<py::tuple>(type->tp_mro);
        for(auto base:mro)if(PyDict_GetItemString(aol_type_dict(reinterpret_cast<PyTypeObject *>(base.ptr())),"__deepcopy__"))return py::none();
        auto cls=py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(type));
        if(!PyFunction_Check(path_reduce.ptr())||cls.attr("__reduce__").ptr()!=path_reduce.ptr()||PyFunction_GetCode(path_reduce.ptr())!=path_reduce_code.ptr())return py::none();
        if(cls.attr("__reduce_ex__").ptr()!=py::reinterpret_borrow<py::object>(reinterpret_cast<PyObject *>(&PyBaseObject_Type)).attr("__reduce_ex__").ptr())return py::none();
      }else if(records.count(type)) {
        if(checked_records.insert(type).second) {
        if(Py_TYPE(reinterpret_cast<PyObject *>(type))!=&PyType_Type)return py::none();
        if(type!=reinterpret_cast<PyTypeObject *>(namespace_type.ptr())&&type->tp_getattro!=PyObject_GenericGetAttr)return py::none();
        if(type!=reinterpret_cast<PyTypeObject *>(namespace_type.ptr())) {
          auto descriptor=PyDict_GetItemString(aol_type_dict(type),"__dict__");
          if(!descriptor||Py_TYPE(descriptor)!=&PyGetSetDescr_Type)return py::none();
        }
        auto mro=py::reinterpret_borrow<py::tuple>(type->tp_mro);
        for(auto base:mro) {
          auto parent=reinterpret_cast<PyTypeObject *>(base.ptr());if(parent==&PyBaseObject_Type||base.ptr()==namespace_type.ptr())continue;
          for(auto hook:hooks)if(PyDict_GetItemString(aol_type_dict(parent),hook))return py::none();
          if(copying&&(PyDict_GetItemString(aol_type_dict(parent),"__slots__")||PyDict_GetItemString(aol_type_dict(parent),"__del__")))return py::none();
        }
        }
        auto raw=PyObject_GenericGetDict(value,nullptr);if(!raw)throw py::error_already_set();
        keepers.push_back(py::reinterpret_steal<py::object>(raw));
        if(!PyDict_CheckExact(raw))return py::none();
        PyObject *key,*item;Py_ssize_t at=0;while(PyDict_Next(raw,&at,&key,&item))if(!PyUnicode_CheckExact(key))return py::none();
        for(auto hook:hooks)if(PyDict_GetItemString(raw,hook))return py::none();pending.push_back(raw);
      }else return py::none();
    }
    if(copying)return py::bool_(true);
    py::set result;for(auto owner:owners)result.add(py::int_(reinterpret_cast<uintptr_t>(owner)));return result;
  }
  // Only a graph admitted above reaches this copier. Opaque copy protocols
  // fall back for the entire request, before invoking any user code.
  py::object clone(py::object root,py::sequence record_types)const {
    auto deepcopy=py::module_::import("copy").attr("deepcopy");
    if(collect(root,record_types,true).is_none())return deepcopy(root);
    std::unordered_set<PyTypeObject *> records;
    for(auto type:record_types)records.insert(reinterpret_cast<PyTypeObject *>(type.ptr()));
    py::dict memo;
    std::function<py::object(py::handle)> copy=[&](py::handle value)->py::object {
      auto type=Py_TYPE(value.ptr());
      if(atomic.count(type))return py::reinterpret_borrow<py::object>(value);
      auto key=py::int_(reinterpret_cast<uintptr_t>(value.ptr()));
      if(memo.contains(key))return py::reinterpret_borrow<py::object>(memo[key]);
      if(Py_EnterRecursiveCall(" while copying an object"))throw py::error_already_set();
      struct Depth {~Depth(){Py_LeaveRecursiveCall();}} depth;
      py::object result;
      if(PyList_CheckExact(value.ptr())) {
        py::list output;memo[key]=output;
        for(auto item:py::reinterpret_borrow<py::list>(value))output.append(copy(item));
        result=std::move(output);
      }else if(PyDict_CheckExact(value.ptr())) {
        py::dict output;memo[key]=output;
        for(auto item:py::reinterpret_borrow<py::dict>(value)) {
          // Match deepcopy's assignment evaluation order: value before key.
          auto copied=copy(item.second);output[copy(item.first)]=std::move(copied);
        }
        result=std::move(output);
      }else if(PyTuple_CheckExact(value.ptr())) {
        auto input=py::reinterpret_borrow<py::tuple>(value);py::tuple output(input.size());bool changed=false;
        for(py::ssize_t i=0;i<input.size();++i){auto item=copy(input[i]);changed|=!item.is(input[i]);output[i]=std::move(item);}
        // A mutable child may have recursively copied this same tuple.
        if(memo.contains(key))return py::reinterpret_borrow<py::object>(memo[key]);
        result=changed?py::object(std::move(output)):py::reinterpret_borrow<py::object>(value);
      }else if(records.count(type)) {
        auto raw=type->tp_alloc(type,0);if(!raw)throw py::error_already_set();
        result=py::reinterpret_steal<py::object>(raw);memo[key]=result;
        auto fields=py::reinterpret_steal<py::object>(PyObject_GenericGetDict(value.ptr(),nullptr));
        if(!fields)throw py::error_already_set();auto state=copy(fields);
        auto target=PyObject_GenericGetDict(result.ptr(),nullptr);if(!target)throw py::error_already_set();
        auto dictionary=py::reinterpret_steal<py::object>(target);
        if(PyDict_Update(dictionary.ptr(),state.ptr())<0)throw py::error_already_set();
      }else result=deepcopy(value,memo);
      if(!result.is(value))memo[key]=result;
      return result;
    };
    return copy(root);
  }
};
}

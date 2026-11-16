#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include "array_workspace.hpp"
#include "execution_services.hpp"
namespace native_components {
namespace py=pybind11;

inline py::object reduction(const py::array &input,int axis,bool mean=false) {
  auto raw=reinterpret_cast<PyArrayObject *>(input.ptr());
  auto result=mean?PyArray_Mean(raw,axis,NPY_DOUBLE,nullptr):PyArray_Sum(raw,axis,NPY_DOUBLE,nullptr);
  if(!result)throw py::error_already_set();return py::reinterpret_steal<py::object>(result);
}

// Sharing must not remove a Python reduction or floating-error callback.
inline bool pure_native_reductions() {
  if(!native_floating_depth)return false;
#if NPY_ABI_VERSION < 0x02000000
  // NumPy 1 deprecates its only arithmetic-table accessor. Introspection
  // must not emit a new warning or call a user's warning handler. Hold the
  // GIL and restore the same filter list; do not invalidate warning registries.
  auto module=PyDict_GetItemString(PyImport_GetModuleDict(),"warnings");
  if(!module||!PyModule_CheckExact(module))return false;
  auto raw_filters=PyDict_GetItemString(PyModule_GetDict(module),"filters");
  if(!raw_filters||!PyList_CheckExact(raw_filters))return false;
  auto filters=py::reinterpret_borrow<py::list>(raw_filters);
  auto ignore=py::make_tuple("ignore",py::none(),py::reinterpret_borrow<py::object>(PyExc_DeprecationWarning),py::none(),0);
  if(PyList_Insert(filters.ptr(),0,ignore.ptr())<0)throw py::error_already_set();
  auto raw=PyArray_GetNumericOps();
  PyObject *error_type,*error_value,*error_trace;PyErr_Fetch(&error_type,&error_value,&error_trace);
  for(py::ssize_t i=0;i<PyList_GET_SIZE(filters.ptr());++i)if(PyList_GET_ITEM(filters.ptr(),i)==ignore.ptr()) {
    if(PySequence_DelItem(filters.ptr(),i)<0)PyErr_Clear();break;
  }
  PyErr_Restore(error_type,error_value,error_trace);
  if(!raw)throw py::error_already_set();
  auto ops=py::reinterpret_steal<py::object>(raw);
  for(auto key:{"add","true_divide"}){auto value=PyDict_GetItemString(ops.ptr(),key);
    if(!value||Py_TYPE(value)!=&PyUFunc_Type)return false;}
#endif
  return true;
}
// Identical NumPy mean and squared-sum reductions can serve multiple ddof values.
struct NativeMoments {
  // A default array_t constructs an empty ndarray. These handles are assigned
  // real reduction results before use, so keep them null until that assignment.
  py::array_t<double> mean=py::reinterpret_borrow<py::array_t<double>>(py::handle());
  py::array_t<double> sum=py::reinterpret_borrow<py::array_t<double>>(py::handle());
  py::ssize_t count;
  NativeMoments(py::array_t<double,py::array::c_style> input,int axis,py::object known_mean=py::none()) {
    bool all=axis==AOL_RAVEL_AXIS;
    if(!all&&axis<0)axis+=input.ndim();
    if(!all&&(axis<0||axis>=input.ndim()))throw py::value_error("axis out of bounds");
    auto raw_mean=known_mean.is_none()?reduction(input,axis,true):known_mean;
    auto mean_raw=PyArray_FromAny(raw_mean.ptr(),PyArray_DescrFromType(NPY_DOUBLE),0,0,NPY_ARRAY_CARRAY_RO,nullptr);
    if(!mean_raw)throw py::error_already_set();mean=py::reinterpret_steal<py::array_t<double>>(mean_raw);
    auto squared=workspace_shaped<double>(input.request().shape);
    py::ssize_t trailing=1;count=all?input.size():input.shape(axis);
    if(!all)for(int i=axis+1;i<input.ndim();++i)trailing*=input.shape(i);
    auto source=input.data(),center=mean.data();auto output=squared.mutable_data();
    for(py::ssize_t k=0;k<input.size();++k){auto index=all?0:(k/(count*trailing))*trailing+k%trailing;
      double difference=source[k]-center[index];output[k]=difference*difference;}
    auto raw_sum=reduction(squared,axis);
    auto sum_raw=PyArray_FromAny(raw_sum.ptr(),PyArray_DescrFromType(NPY_DOUBLE),0,0,NPY_ARRAY_CARRAY,nullptr);
    if(!sum_raw)throw py::error_already_set();sum=py::reinterpret_steal<py::array_t<double>>(sum_raw);
  }
  py::array_t<double> deviation(int ddof,bool consume=true) {
    py::array_t<double> result=py::reinterpret_borrow<py::array_t<double>>(py::handle());
    if(consume)result=std::move(sum);
    else {auto copy=PyArray_NewCopy(reinterpret_cast<PyArrayObject *>(sum.ptr()),NPY_CORDER);
      if(!copy)throw py::error_already_set();result=py::reinterpret_steal<py::array_t<double>>(copy);}
    double denominator=std::max<py::ssize_t>(count-ddof,0);
    for(py::ssize_t k=0;k<result.size();++k)result.mutable_data()[k]=std::sqrt(result.data()[k]/denominator);
    return result;
  }
};
inline py::array_t<double> standard_deviation(py::array_t<double,py::array::c_style> input,int axis,int ddof) {
  NativeMoments moments(input,axis);return moments.deviation(ddof);
}
}

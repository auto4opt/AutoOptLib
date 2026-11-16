#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <memory>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace native_components {
namespace py=pybind11;
// A read-only, call-scoped descriptor. Its native owner retains all NumPy
// allocations; constructing/destroying this descriptor requires the GIL.
// Every row is contiguous. No Python access occurs in row()/numeric loops.
struct DecisionInput;
template<class T> struct IndexedInput {
  const DecisionInput *owner;
  int ndim() const { return 2; }
  py::ssize_t shape(int axis) const;
};

struct DecisionInput {
  std::shared_ptr<void> keepalive;
  py::array allocation=py::reinterpret_borrow<py::array>(py::none());
  std::vector<const void *> rows;
  py::ssize_t n=0,d=0;
  bool integer=false;
  const void *dense=nullptr;
  const void *row(py::ssize_t i)const {
    return dense?static_cast<const char *>(dense)+i*d*8:rows[i];
  }
  template<class T> IndexedInput<T> typed() const {
    if(integer!=std::is_same<T,int64_t>::value)
      throw py::type_error("Indexed input requires the consumer's native dtype");
    return {this};
  }
  py::array materialize() const {
    if(dense && allocation)return allocation;
    py::array result(integer?py::dtype::of<int64_t>():py::dtype::of<double>(),
                     std::vector<py::ssize_t>{n,d});
    for(py::ssize_t i=0;i<n;++i)
      std::memcpy(static_cast<char *>(result.mutable_data())+i*d*8,row(i),d*8);
    return result;
  }
};
template<class T> py::ssize_t IndexedInput<T>::shape(int axis) const {
  return axis==0?owner->n:owner->d;
}
template<class T> inline const T *input_row(const py::array &input,py::ssize_t i) {
  return reinterpret_cast<const T *>(static_cast<const char *>(input.data())+i*input.strides(0));
}
template<class T,class U> inline const T *input_row(const IndexedInput<U>&input,py::ssize_t i) {
  if constexpr(!std::is_same<T,U>::value)
    throw py::type_error("Indexed input dtype mismatch");
  return static_cast<const T *>(input.owner->row(i));
}
template<class T> inline void copy_input(const py::array &input,T *output) {
  std::copy_n(static_cast<const T *>(input.data()),input.shape(0)*input.shape(1),output);
}
template<class T,class U> inline void copy_input(const IndexedInput<U>&input,T *output) {
  for(py::ssize_t i=0;i<input.shape(0);++i)
    std::copy_n(input_row<T>(input,i),input.shape(1),output+i*input.shape(1));
}
inline py::array input_dense(const py::array &input) { return input; }
template<class T> inline py::array input_dense(const IndexedInput<T>&input) {
  return input.owner->materialize();
}
template<class Input> struct InputRowsView {
  const Input &input;
  int ndim()const{return 2;}
  py::ssize_t shape(int axis)const{return input.shape(axis);}
  const double *row(py::ssize_t i)const{return input_row<double>(input,i);}
  double at(py::ssize_t i,py::ssize_t j)const{return row(i)[j];}
};
}

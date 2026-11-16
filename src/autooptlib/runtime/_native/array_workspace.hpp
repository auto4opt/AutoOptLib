#pragma once
#include <pybind11/numpy.h>
#include <vector>

namespace native_components {
namespace py=pybind11;
class ArrayWorkspace {
  // Only backing arrays are cached. Each lease is a fresh view holding its
  // backing owner, so retained arrays, slices and memoryviews prevent reuse.
  std::vector<py::array> backing_;
public:
  bool recording=false;
  uint64_t allocations=0,reuses=0;
  size_t capacity_bytes=0;
  template<class T>py::array &acquire_owner(py::ssize_t count) {
    auto dtype=py::dtype::of<T>();py::array *selected=nullptr;
    for(auto &owner:backing_) {
      auto weak=Py_TYPE(owner.ptr())->tp_weaklistoffset&&*PyObject_GET_WEAKREFS_LISTPTR(owner.ptr());
      if(Py_REFCNT(owner.ptr())==1&&!weak&&owner.dtype().is(dtype)&&owner.size()>=count
          &&(!selected||owner.size()<selected->size()))selected=&owner;
    }
    if(!selected) {
      auto capacity=std::max<py::ssize_t>(count,16);
      backing_.emplace_back(py::array_t<T>(capacity));selected=&backing_.back();
      if(recording)++allocations;capacity_bytes+=capacity*sizeof(T);
    }else if(recording)++reuses;
    return *selected;
  }
  template<class T>py::array_t<T> acquire(py::ssize_t count) {
    auto &owner=acquire_owner<T>(count);
    return py::array_t<T>({count},{py::ssize_t(sizeof(T))},static_cast<T *>(owner.mutable_data()),owner);
  }
  template<class T>py::array_t<T> acquire_shaped(py::ssize_t count,const std::vector<py::ssize_t> &shape,const std::vector<py::ssize_t> &strides) {
    auto &owner=acquire_owner<T>(count);
    return py::array_t<T>(shape,strides,static_cast<T *>(owner.mutable_data()),owner);
  }
};
inline thread_local ArrayWorkspace *active_array_workspace=nullptr;
struct ArrayWorkspaceContext {
  ArrayWorkspace *previous;
  explicit ArrayWorkspaceContext(ArrayWorkspace *workspace):previous(active_array_workspace){active_array_workspace=workspace;}
  ~ArrayWorkspaceContext(){active_array_workspace=previous;}
};
template<class T>py::array_t<T> workspace_array(py::ssize_t count) {
  return active_array_workspace?active_array_workspace->acquire<T>(count):py::array_t<T>(count);
}
// Private intermediates only: the lease keeps the backing unavailable to reuse.
template<class T>py::array_t<T> workspace_shaped(const std::vector<py::ssize_t> &shape) {
  if(!active_array_workspace)return py::array_t<T>(shape);
  std::vector<py::ssize_t> strides(shape.size());py::ssize_t count=1;
  for(size_t k=shape.size();k>0;--k){strides[k-1]=count*sizeof(T);count*=shape[k-1];}
  return active_array_workspace->acquire_shaped<T>(count,shape,strides);
}
} // namespace native_components

#pragma once
#include <numpy/npy_math.h>
#include "execution_services.hpp"
namespace native_components {
// Invoke NumPy's compiled gufunc loop with explicit core dimensions/strides.
// This retains the installed NumPy/LAPACK arithmetic without its Python
// linalg wrappers or version-specific error callbacks (NumPy 1.x/2.x).
struct Lapack {
  py::object owner,error;
  PyUFuncGenericFunction loop=nullptr;void *data=nullptr;
  explicit Lapack(py::object function,py::object exception):owner(function),error(exception) {
    auto u=reinterpret_cast<PyUFuncObject *>(function.ptr());
    for(int i=0;i<u->ntypes;++i) {
      bool doubles=true;for(int j=0;j<u->nargs;++j)doubles&=u->types[i*u->nargs+j]==NPY_DOUBLE;
      if(doubles){loop=u->functions[i];data=u->data[i];break;}
    }
    if(!loop)throw py::value_error("Missing native float64 LAPACK loop");
  }
  void run(char **args,npy_intp *dimensions,npy_intp *steps,const char *message) const {
    RegionScope timing(Region::Numeric);
    int flags;
    {py::gil_scoped_release release;
      npy_clear_floatstatus_barrier(args[0]);
      loop(args,dimensions,steps,data);
      flags=npy_clear_floatstatus_barrier(args[0]);}
    if(PyErr_Occurred())throw py::error_already_set();
    if(flags&NPY_FPE_INVALID){PyErr_SetString(error.ptr(),message);throw py::error_already_set();}
  }
  py::array multiply(py::array_t<double> a,py::array_t<double> b) const {
    if(a.ndim()<1||a.ndim()>2||b.ndim()<1||b.ndim()>2)
      throw py::value_error("Native matrix product requires vectors or matrices");
    bool av=a.ndim()==1,bv=b.ndim()==1;
    auto rows=av?1:a.shape(0),inner=a.shape(a.ndim()-1),columns=bv?1:b.shape(1);
    if(inner!=b.shape(0))throw py::value_error("Matrix product dimensions do not match");
    std::vector<py::ssize_t> shape;
    if(!av)shape.push_back(rows);if(!bv)shape.push_back(columns);
    py::array_t<double> out(shape);
    char *args[]={reinterpret_cast<char *>(const_cast<double *>(a.data())),reinterpret_cast<char *>(const_cast<double *>(b.data())),reinterpret_cast<char *>(out.mutable_data())};
    npy_intp dimensions[]={1,rows,inner,columns};
    npy_intp steps[]={0,0,0,av?0:a.strides(0),a.strides(a.ndim()-1),b.strides(0),bv?0:b.strides(1),av?0:out.strides(0),bv?0:out.strides(out.ndim()-1)};
    // Component matrix products occur in the reference's errstate(ignore)
    // regions; direct loops avoid invoking a Python warning handler.
    {RegionScope timing(Region::Numeric);py::gil_scoped_release release;
      npy_clear_floatstatus_barrier(args[0]);loop(args,dimensions,steps,data);
      npy_clear_floatstatus_barrier(args[0]);}
    if(PyErr_Occurred())throw py::error_already_set();return out;
  }
  py::object execute(py::array_t<double> input,py::object right,const std::string &kind) const {
    if(input.ndim()!=2||input.shape(0)!=input.shape(1))throw py::value_error("Last 2 dimensions of the array must be square");
    auto n=input.shape(0);auto ptr=const_cast<double *>(input.data());
    npy_intp dimensions[]={1,n,n};
    if(kind=="eigh") {
      py::array_t<double> values(n),vectors({n,n});
      char *args[]={reinterpret_cast<char *>(ptr),reinterpret_cast<char *>(values.mutable_data()),reinterpret_cast<char *>(vectors.mutable_data())};
      npy_intp steps[]={0,0,0,input.strides(0),input.strides(1),8,vectors.strides(0),vectors.strides(1)};
      run(args,dimensions,steps,"Eigenvalues did not converge");return py::make_tuple(values,vectors);
    }
    if(kind=="cholesky") {
      py::array_t<double> out({n,n});
      char *args[]={reinterpret_cast<char *>(ptr),reinterpret_cast<char *>(out.mutable_data())};
      npy_intp steps[]={0,0,input.strides(0),input.strides(1),out.strides(0),out.strides(1)};
      run(args,dimensions,steps,"Matrix is not positive definite");return out;
    }
    auto b=py::cast<py::array_t<double>>(right);bool vector=b.ndim()==1;
    if(b.ndim()<1||b.ndim()>2||b.shape(0)!=n)throw py::value_error("Solve operands have incompatible dimensions");
    py::array_t<double> out(b.request().shape);dimensions[2]=vector?1:b.shape(1);
    char *args[]={reinterpret_cast<char *>(ptr),reinterpret_cast<char *>(const_cast<double *>(b.data())),reinterpret_cast<char *>(out.mutable_data())};
    npy_intp steps[]={0,0,0,input.strides(0),input.strides(1),b.strides(0),vector?out.strides(0):b.strides(1),vector?0:out.strides(0),vector?0:out.strides(1)};
    run(args,dimensions,steps,"Singular matrix");return out;
  }
};
}

#pragma once
#include "python_compat.hpp"
#include <cfenv>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
// Shared component services; bindings and the search core use one implementation.
namespace native_components {
// Fixed native-call policy: no Python floating-point diagnostics or user hooks.
// These are NumPy C builtins / ContextVar C methods, not numpy.seterr's Python
// wrappers. The caller's thread/context policy is restored on every exit.
struct FloatingPolicy {
  py::object get,set,make,context;
  explicit FloatingPolicy(py::handle numpy) {
    auto dictionary=py::reinterpret_borrow<py::object>(numpy).attr("__dict__");
    auto getter=PyDict_GetItemString(dictionary.ptr(),"geterrobj");
    if(getter) {get=py::reinterpret_borrow<py::object>(getter);set=dictionary[py::str("seterrobj")];}
    else {
      auto umath=py::module_::import("numpy._core.umath");
      make=umath.attr("_make_extobj");context=umath.attr("_extobj_contextvar");
    }
  }
  struct Scope {
    const FloatingPolicy &policy;py::object saved;
    explicit Scope(const FloatingPolicy &p,bool establish=true):policy(p) {
      if(!establish){++native_floating_depth;return;}
      if(p.get) {
        saved=p.get();auto values=py::reinterpret_steal<py::list>(PySequence_List(saved.ptr()));
        values[1]=py::int_(0);p.set(values);
      } else saved=p.context.attr("set")(p.make("all"_a="ignore"));
      ++native_floating_depth;
    }
    ~Scope() {
      --native_floating_depth;
      if(!saved)return;
      PyObject *type,*value,*trace;PyErr_Fetch(&type,&value,&trace);
      try {if(policy.get)policy.set(saved);else policy.context.attr("reset")(saved);}
      catch(py::error_already_set &error){error.discard_as_unraisable("restoring native floating-point policy");}
      PyErr_Restore(type,value,trace);
    }
  };
};
// One session-owned binding of NumPy's readonly generator resources. State and
// lock ownership remain NumPy's; every component acquires/releases as before.
struct RandomBinding {
  std::vector<py::object> types;py::object generator_type;
  py::object generator=py::none(),owner,capsule,lock;
  unsigned generator_version=0,owner_version=0;
  explicit RandomBinding(py::handle numpy) {
    generator_type=numpy.attr("random").attr("Generator");
    for(auto name:{"PCG64","MT19937","Philox","SFC64"})types.push_back(numpy.attr("random").attr(name));
  }
  bool valid(py::handle value)const {
    return generator.is(value)&&generator_version&&owner_version&&
      Py_TYPE(value.ptr())->tp_version_tag==generator_version&&Py_TYPE(owner.ptr())->tp_version_tag==owner_version;
  }
  static bool readonly(PyTypeObject *type,const char *key) {
    PyObject *descriptor=nullptr;auto mro=type->tp_mro;
    for(Py_ssize_t i=0;mro&&i<PyTuple_GET_SIZE(mro)&&!descriptor;++i)
      descriptor=PyDict_GetItemString(aol_type_dict(reinterpret_cast<PyTypeObject *>(PyTuple_GET_ITEM(mro,i))),key);
    return descriptor&&Py_TYPE(descriptor)==&PyGetSetDescr_Type&&!reinterpret_cast<PyGetSetDescrObject *>(descriptor)->d_getset->set;
  }
  void remember(py::object value,py::object o,py::object c,py::object l) {
    generator=py::none();
    if(Py_TYPE(value.ptr())!=reinterpret_cast<PyTypeObject *>(generator_type.ptr())||Py_TYPE(o.ptr())->tp_getattro!=PyObject_GenericGetAttr||
       !readonly(Py_TYPE(value.ptr()),"bit_generator"))return;
    bool standard=false;for(auto &t:types)standard|=Py_TYPE(o.ptr())==reinterpret_cast<PyTypeObject *>(t.ptr());
    if(!standard)return;
    for(auto key:{"capsule","lock"})if(!readonly(Py_TYPE(o.ptr()),key))return;
    generator=std::move(value);owner=std::move(o);capsule=std::move(c);lock=std::move(l);
    generator_version=Py_TYPE(generator.ptr())->tp_version_tag;owner_version=Py_TYPE(owner.ptr())->tp_version_tag;
  }
};
struct RandomBoundary {
  py::object owner, capsule, lock;
  Random random;
  explicit RandomBoundary(py::object generator,RandomBinding *binding=nullptr):random(nullptr) {
    if(binding&&binding->valid(generator)){owner=binding->owner;capsule=binding->capsule;lock=binding->lock;}
    else {
      owner=generator.attr("bit_generator");capsule=owner.attr("capsule");lock=owner.attr("lock");
      if(binding)binding->remember(generator,owner,capsule,lock);
    }
    random=Random(static_cast<bitgen_t *>(PyCapsule_GetPointer(capsule.ptr(),"BitGenerator")));
    if (PyErr_Occurred()) throw py::error_already_set();
    // The exact builtin BitGenerator lock is acquired at the boundary. Native
    // draws consume its shared state while holding the same lock as NumPy.
    lock.attr("acquire")();
  }
  ~RandomBoundary() {
    PyObject *type,*value,*trace;
    PyErr_Fetch(&type,&value,&trace);
    auto released=PyObject_CallMethod(lock.ptr(),"release",nullptr);
    if(released)Py_DECREF(released);
    else PyErr_WriteUnraisable(lock.ptr());
    PyErr_Restore(type,value,trace);
  }
};

py::array_t<double> draws(py::object generator,const std::string &kind,
                          size_t count,double first,double second) {
  RandomBoundary boundary(generator);
  py::array_t<double> result(count);
  auto out=result.mutable_data();
  if(kind=="random")boundary.random.uniform(out,count);
  else if(kind=="standard_normal")boundary.random.normal(out,count);
  else if(kind=="normal")for(size_t i=0;i<count;++i)out[i]=boundary.random.normal(first,second);
  else if(kind=="standard_cauchy")for(size_t i=0;i<count;++i)out[i]=boundary.random.cauchy();
  else if(kind=="binomial")for(size_t i=0;i<count;++i)out[i]=boundary.random.binomial(first,second);
  else throw py::value_error("Unknown native random operation");
  return result;
}

struct Unary {
  PyUFuncGenericFunction loop=nullptr;
  void *data=nullptr;
  py::object owner;
  bool exact_scalar_sqrt=false;
  explicit Unary(py::object function,bool scalar_sqrt=false):owner(function),exact_scalar_sqrt(scalar_sqrt) {
    auto u=reinterpret_cast<PyUFuncObject *>(function.ptr());
    if(u->nin!=1 || u->nout!=1)throw py::value_error("Expected unary ufunc");
    for(int i=0;i<u->ntypes;++i)
      if(u->types[2*i]==NPY_DOUBLE && u->types[2*i+1]==NPY_DOUBLE) {
        loop=u->functions[i];data=u->data[i];break;
      }
    if(!loop)throw py::value_error("Missing float64 ufunc loop");
  }
  py::array_t<double> execute(py::array_t<double,py::array::c_style> input) const {
    py::array_t<double> output(input.request().shape);
    execute_into(input,output);
    return output;
  }
  // Internal callers supply their existing private destination. The public
  // execute() still returns independent owning storage.
  void execute_into(const py::array_t<double,py::array::c_style> &input,py::array_t<double> &output) const {
    char *args[]={reinterpret_cast<char *>(const_cast<double *>(input.data())),
                  reinterpret_cast<char *>(output.mutable_data())};
    npy_intp size=input.size(),steps[]={sizeof(double),sizeof(double)};
    {RegionScope timing(Region::Numeric);py::gil_scoped_release release;loop(args,&size,steps,data);}
    if(PyErr_Occurred())throw py::error_already_set();
  }
  double scalar(double input) const {
#if defined(__x86_64__) || defined(_M_X64)
    if(exact_scalar_sqrt&&native_floating_depth&&std::isfinite(input)&&input>=0&&
       std::fegetround()==FE_TONEAREST&&(_mm_getcsr()&~0x3fu)==0x1f80u) {
      double result;char *args[]={reinterpret_cast<char *>(&input),reinterpret_cast<char *>(&result)};
      npy_intp count=1,steps[]={8,8};loop(args,&count,steps,data);
      if(PyErr_Occurred())throw py::error_already_set();return result;
    }
#endif
    auto value=py::float_(input);PyObject *arguments[]={value.ptr()};
    auto raw=PyObject_Vectorcall(owner.ptr(),arguments,1,nullptr);
    if(!raw)throw py::error_already_set();
    return py::cast<double>(py::reinterpret_steal<py::object>(raw));
  }
};

// Generalized ufuncs (matmul/LAPACK) are compiled NumPy C entry points, not
// numpy.linalg's Python wrappers. Exact ndarray inputs prevent user dispatch.
struct NativeMath {
  FloatingPolicy floating;
  Unary logarithm,exponential,square_root;
  py::object matmul, eigh, cholesky, solve_vector, solve_matrix, power;
  py::object linalg_error;
  std::optional<uint64_t> bit_environment() const {
#if defined(__x86_64__) || defined(_M_X64)
    auto control=_mm_getcsr() & ~0x3fu;
    if(std::fegetround()==FE_TONEAREST&&control==0x1f80u)return control;
#endif
    return std::nullopt;
  }
  std::array<double,2> normal_parameters(int64_t dimension) const {
    auto environment=bit_environment();auto key=reinterpret_cast<uintptr_t>(square_root.owner.ptr());
    if(active_workspace&&environment&&active_workspace->normal_dimension==dimension&&
       active_workspace->normal_math==key&&active_workspace->normal_environment==*environment)return active_workspace->normal_scales;
    std::array<double,2> value={1./square_root.scalar(2.*square_root.scalar(dimension)),1./square_root.scalar(2.*dimension)};
    if(active_workspace&&environment){auto &w=*active_workspace;w.normal_dimension=dimension;w.normal_math=key;w.normal_environment=*environment;w.normal_scales=value;}
    return value;
  }
  bool bit_cache_key(bool fastga,uint64_t &environment,uintptr_t &function) const {
    if(!active_workspace)return false;
    auto current=bit_environment();if(!current)return false;
    if(!fastga&&Py_TYPE(power.ptr())!=&PyUFunc_Type)return false;
    environment=*current;function=fastga?0:reinterpret_cast<uintptr_t>(power.ptr());return true;
  }
  const std::vector<double> *cached_bit_weights(int64_t d,bool fastga) const {
    uint64_t environment;uintptr_t function;
    if(!bit_cache_key(fastga,environment,function))return nullptr;
    auto &w=*active_workspace;
    if(w.bit_dimension!=d||w.bit_fastga!=fastga||w.bit_math!=function||w.bit_environment!=environment)return nullptr;
    ++w.bit_table_hits;return &w.bit_weights;
  }
  void remember_bit_weights(int64_t d,bool fastga,const std::vector<double> &values) const {
    uint64_t environment;uintptr_t function;
    if(!bit_cache_key(fastga,environment,function))return;
    auto &w=*active_workspace;
    w.bit_dimension=-1;w.bit_cdf.clear();w.bit_weights=values;w.bit_fastga=fastga;
    w.bit_math=function;w.bit_environment=environment;w.bit_dimension=d;++w.bit_table_builds;
  }
  const std::vector<double> *bit_cdf(const std::vector<double> &weights,bool fastga)const {
    if(!active_workspace||&weights!=&active_workspace->bit_weights)return nullptr;
    auto &cdf=active_workspace->bit_cdf;
    if(cdf.empty()&&!weights.empty()) {
      cdf.reserve(weights.size());std::partial_sum(weights.begin(),weights.end(),std::back_inserter(cdf));
      if(!fastga){auto total=cdf.back();for(auto &value:cdf)value/=total;}
    }
    return &cdf;
  }
  std::optional<Lapack> eigen_loop,cholesky_loop,solve_vector_loop,solve_matrix_loop,matmul_loop;
  NativeMath(py::object numpy,py::object linalg)
      :floating(numpy),logarithm(numpy.attr("log")),exponential(numpy.attr("exp")),square_root(numpy.attr("sqrt"),true),
       matmul(numpy.attr("matmul")),eigh(linalg.attr("eigh_lo")),
       cholesky(linalg.attr("cholesky_lo")),solve_vector(linalg.attr("solve1")),
       solve_matrix(linalg.attr("solve")),power(numpy.attr("power")) {
    auto error=py::module_::import("numpy.linalg").attr("LinAlgError");
    linalg_error=error;
    eigen_loop.emplace(eigh,error);cholesky_loop.emplace(cholesky,error);
    solve_vector_loop.emplace(solve_vector,error);solve_matrix_loop.emplace(solve_matrix,error);
    matmul_loop.emplace(matmul,error);

  }
  py::object invoke(py::handle function,py::handle a,py::handle b=py::handle()) const {
    if(function.is(matmul))return matmul_loop->multiply(py::cast<py::array_t<double>>(a),py::cast<py::array_t<double>>(b));
    if(function.is(eigh))return eigen_loop->execute(py::cast<py::array_t<double>>(a),py::none(),"eigh");
    if(function.is(cholesky))return cholesky_loop->execute(py::cast<py::array_t<double>>(a),py::none(),"cholesky");
    if(function.is(solve_vector)||function.is(solve_matrix)) {
      auto &loop=function.is(solve_vector)?solve_vector_loop:solve_matrix_loop;
      return loop->execute(py::cast<py::array_t<double>>(a),py::reinterpret_borrow<py::object>(b),"solve");
    }
    PyObject *arguments[]={a.ptr(),b.ptr(),nullptr};
    size_t positional=b?2:1;
    auto result=PyObject_Vectorcall(function.ptr(),arguments,positional,nullptr);
    if(!result)throw py::error_already_set();
    return py::reinterpret_steal<py::object>(result);
  }
};
}

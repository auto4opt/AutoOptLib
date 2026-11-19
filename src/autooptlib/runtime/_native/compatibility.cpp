// Boundary probes for the native compatibility layer. NumPy object discovery
// happens at module initialization; the numerical loops use C entry points.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/ufuncobject.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <optional>
#include <limits>
#include "execution_services.hpp"
#if NPY_ABI_VERSION >= 0x02000000
#define AOL_RAVEL_AXIS NPY_RAVEL_AXIS
#else
#define AOL_RAVEL_AXIS NPY_MAXDIMS
#endif
#include "random.hpp"
#include "decision_input.hpp"
#include "selection.hpp"
#include "replacement.hpp"
#include "discrete.hpp"
#include "crossover.hpp"
#include "continuous.hpp"
#include "differential.hpp"
#include "clustering.hpp"
#include "archive.hpp"
#include "stateful.hpp"
#include "cma.hpp"
#include "lapack.hpp"
namespace py=pybind11;
using namespace pybind11::literals;

#include "component_runtime.hpp"

#include "component.hpp"
#include "data.hpp"
#include "native_evaluation.hpp"
#include "batch_core.hpp"
#include "batch_run.hpp"
#include "batch_bindings.hpp"
#include "batch_calibration.hpp"

#include "cpu_objective_kernels.hpp"
PYBIND11_MODULE(_native_compatibility,m) {
  if(_import_array()<0 || _import_umath()<0)throw py::error_already_set();
  bind_cpu_objective_kernels(m);
  auto numpy=py::module_::import("numpy");
#ifdef AOL_NUMPY_VERSION
  if(py::cast<std::string>(numpy.attr("__version__"))!=AOL_NUMPY_VERSION)
    throw py::import_error("Native component RNG compatibility requires the NumPy version used at build time (" AOL_NUMPY_VERSION "). Rebuild with the installed NumPy using --no-build-isolation.");
  m.attr("numpy_build_version")=AOL_NUMPY_VERSION;
#endif
  m.def("draws",&native_components::draws,"generator"_a,"kind"_a,"count"_a,
        "first"_a=0.,"second"_a=1.);
  m.def("selection",[](const std::string &name,py::array_t<double,py::array::c_style> fits,
                        py::object generator,int64_t count,double parameter,py::dict state) {
    native_components::RandomBoundary b(generator);
    return native_components::selection(name,fits.size(),fits,b.random,count,parameter,native_components::NativeState(state,true)).python();
  },"name"_a,"fitness"_a,"generator"_a,"count"_a,"parameter"_a=.2,"state"_a=py::dict());
  m.def("integers",[](py::object generator,int64_t low,int64_t high,size_t n) {
    native_components::RandomBoundary b(generator);py::array_t<int64_t> out(n);
    b.random.integers(low,high,out.mutable_data(),n);return out;
  });
  m.def("permutation",[](py::object generator,py::array_t<int64_t,py::array::c_style> values) {
    native_components::RandomBoundary b(generator);py::array_t<int64_t> out(values.size());
    std::copy_n(values.data(),values.size(),out.mutable_data());
    b.random.permutation(out.mutable_data(),out.size());return out;
  });
  m.def("choice",[](py::object generator,size_t population,size_t n,bool replace,
                      py::object weights,bool shuffle) {
    std::optional<py::array_t<double,py::array::c_style>> p;
    if(!weights.is_none()) {
      p.emplace(py::cast<py::array_t<double,py::array::c_style>>(weights));
      if(p->ndim()!=1 || p->size()!=static_cast<py::ssize_t>(population))
        throw py::value_error("a and p must have same size");
      double sum=0,correction=0;
      for(py::ssize_t i=0;i<p->size();++i) {
        double weight=p->data()[i];
        if(std::isnan(weight))throw py::value_error("probabilities contain NaN");
        if(weight<0)throw py::value_error("probabilities are not non-negative");
        double adjusted=weight-correction,next=sum+adjusted;
        correction=(next-sum)-adjusted;sum=next;
      }
      if(!std::isfinite(sum) || std::abs(sum-1)>std::sqrt(std::numeric_limits<double>::epsilon()))
        throw py::value_error("probabilities do not sum to 1");
    }
    native_components::RandomBoundary b(generator);
    if(!p&&replace) {
      if(!population&&n)throw std::invalid_argument("a must be positive");
      py::array_t<int64_t> out(n);b.random.integers(0,population,out.mutable_data(),n);return out;
    }
    auto values=b.random.choice(population,n,replace,p?p->data():nullptr,shuffle);
    py::array_t<int64_t> out(n);std::copy(values.begin(),values.end(),out.mutable_data());return out;
  },"generator"_a,"population"_a,"n"_a,"replace"_a=true,"weights"_a=py::none(),"shuffle"_a=true);
  py::class_<native_components::Unary>(m,"Unary").def("execute",&native_components::Unary::execute);
  for(auto name:{"exp","log","sqrt"})
    m.attr(name)=py::cast(native_components::Unary(numpy.attr(name)));
  auto exponential=std::make_shared<native_components::Unary>(numpy.attr("exp"));
  m.def("replacement",[exponential](const std::string &name,
              py::array_t<double,py::array::c_style> fits,py::object generator,
              int64_t count,double parameter,int64_t generation,int64_t maximum,py::dict state) {
    native_components::RandomBoundary b(generator);
    return native_components::replacement(name,fits,count,b.random,parameter,generation,maximum,native_components::NativeState(state,true),*exponential,
        native_components::FitnessValues{fits,py::reinterpret_borrow<py::array_t<double>>(py::none())}).python();
  },"name"_a,"fitness"_a,"generator"_a,"count"_a,"parameter"_a=.5,
    "generation"_a=1,"maximum"_a=10,"state"_a=py::dict());
  auto math=std::make_shared<native_components::NativeMath>(numpy,
      py::module_::import("numpy.linalg._umath_linalg"));
  native_components::bind_batch_core(m,math);
  native_components::bind_data_calibration(m,math);
  m.def("fixed_bit_events",[](py::list items,py::array bound,py::object generator,
                               py::object solution_class,int64_t count,int64_t strength) -> py::object {
    // Admission has no conversions/callbacks and consumes no RNG on rejection.
    // Only final deferred events enter here, so the whole interval is callback-free.
    if(count<=0||items.empty()||bound.ndim()!=2||bound.shape(0)!=2||
       !bound.dtype().is(py::dtype::of<int64_t>())||
       !PyArray_ISCARRAY_RO(reinterpret_cast<PyArrayObject *>(bound.ptr())))return py::none();
    auto d=bound.shape(1);if(d<strength)return py::none();
    if(strength!=1&&strength!=3&&strength!=5)return py::none();
    auto limits=static_cast<const int64_t *>(bound.data());
    for(py::ssize_t j=0;j<d;++j)if(limits[j]!=0||limits[d+j]!=1)return py::none();
    std::vector<py::array> rows;rows.reserve(items.size());
    for(auto item:items) {
      if(Py_TYPE(item.ptr())!=reinterpret_cast<PyTypeObject *>(solution_class.ptr()))return py::none();
      auto value=native_components::field(item,"dec");
      if(!PyArray_CheckExact(value.ptr()))return py::none();
      auto row=py::reinterpret_borrow<py::array>(value);
      if(row.ndim()!=1||row.size()!=d||!row.dtype().is(py::dtype::of<int64_t>())||
         !PyArray_ISCARRAY_RO(reinterpret_cast<PyArrayObject *>(row.ptr())))return py::none();
      auto data=static_cast<const int64_t *>(row.data());
      for(py::ssize_t j=0;j<d;++j)if(data[j]!=0&&data[j]!=1)return py::none();
      rows.push_back(std::move(row));
    }
    py::array_t<int64_t> output({static_cast<py::ssize_t>(count),d}),parents(count);
    native_components::RandomBoundary boundary(generator);auto &rng=boundary.random;
    for(int64_t i=0;i<count;++i) {
      rng.integer(0,1); // Original one-path routing draw, including its no-draw case.
      int64_t parent;rng.integers(0,rows.size(),&parent,1);
      parents.mutable_data()[i]=parent;
      auto positions=rng.choice(d,strength,false);
      auto destination=output.mutable_data()+i*d;
      std::copy_n(static_cast<const int64_t *>(rows[parent].data()),d,destination);
      native_components::flip_bits(destination,positions);
    }
    return py::make_tuple(output,parents);
  });
  m.def("component",[math,numpy](const std::string &name,py::object solution_type) {
    native_components::Component<native_components::NativeMath> component{
      name,math,solution_type,numpy.attr("random").attr("Generator"),numpy.attr("random").attr("default_rng")};
    return py::cpp_function([component](py::args args){return component(args);},py::name(name.c_str()));
  });
  m.def("cma_sample",[math](py::array_t<double,py::array::c_style> input,
       py::array_t<double,py::array::c_style> population,py::array_t<double,py::array::c_style> bounds,
       py::dict state,bool context,bool algorithm,py::object generator) {
    native_components::RandomBoundary b(generator);
    return native_components::cma_sample(input,population,bounds,native_components::NativeState(state,true),context,algorithm,b.random,*math).python();
  });
  m.def("cma_update",[math](py::array_t<double,py::array::c_style> fitness,
       py::array_t<double,py::array::c_style> bounds,py::dict state,bool solution) {
    return native_components::cma_update(fitness,bounds,native_components::NativeState(state,true),solution,*math).python();
  });
  m.def("continuous",[math](const std::string &name,
             py::array_t<double,py::array::c_style> input,
             py::array_t<double,py::array::c_style> population,
             py::array_t<double,py::array::c_style> bounds,
             const std::vector<double> &parameters,py::dict state,int64_t inner,py::object generator) {
    if(input.ndim()!=2 || population.ndim()!=2 || population.shape(1)!=input.shape(1)
       || bounds.ndim()!=2 || bounds.shape(0)!=2 || bounds.shape(1)!=input.shape(1))
      throw py::value_error("Expected compatible input, population, and bounds matrices");
    native_components::RandomBoundary b(generator);
    return native_components::continuous_variation(name,input,population,bounds,parameters,native_components::NativeState(state,true),inner,b.random,*math).python();
  });
  m.def("continuous_crossover",[math](const std::string &name,
             py::array_t<double,py::array::c_style> input,
             const std::vector<double> &parameters,py::object generator) {
    native_components::RandomBoundary b(generator);
    return native_components::continuous_crossover(name,input,parameters,b.random,*math);
  });
  m.def("differential",[](const std::string &name,
              py::array_t<double,py::array::c_style> targets,
              py::array_t<double,py::array::c_style> population,
              py::array_t<double,py::array::c_style> fitness,
              py::array_t<double,py::array::c_style> target_fitness,
              const std::vector<double> &parameters,py::dict state,bool context,py::object generator) {
    if(targets.ndim()!=2||population.ndim()!=2||targets.shape(1)!=population.shape(1)
        ||fitness.size()!=population.shape(0)||target_fitness.size()!=targets.shape(0))
      throw py::value_error("Incompatible differential evolution inputs");
    native_components::RandomBoundary b(generator);
    return native_components::differential(name,targets,population,fitness,target_fitness,parameters,native_components::NativeState(state,true),context,b.random).python();
  });
  m.def("niche",[math](py::array_t<double,py::array::c_style> input,
              py::array_t<double,py::array::c_style> fits,int64_t generation,int64_t maximum,py::object generator) {
    if(input.ndim()!=2||fits.size()!=input.shape(0))throw py::value_error("Fitness count must match decision rows for niching.");
    native_components::RandomBoundary b(generator);
    return native_components::niche(input,fits,generation,maximum,b.random,*math);
  });
  m.def("brainstorm",[math](py::array_t<double,py::array::c_style> input,
              py::array_t<double,py::array::c_style> fits,int64_t clusters,double gamma,int64_t count,py::object generator) {
    if(input.ndim()!=2||fits.size()!=input.shape(0)||count<0)throw py::value_error("Incompatible brainstorm selection inputs");
    native_components::RandomBoundary b(generator);
    return native_components::brainstorm(input,fits,clusters,gamma,count,b.random,*math);
  });
  m.def("archive_statistics",[](py::array_t<double,py::array::c_style> fitness,
                                py::array_t<double,py::array::c_style> previous) {
    return native_components::archive_statistics(fitness,previous);
  });
  m.def("archive_diversity",[math](py::array_t<double,py::array::c_style> input,int64_t count,bool continuous,py::object generator) {
    if(input.ndim()!=2)throw py::value_error("Expected two-dimensional decisions");
    native_components::RandomBoundary b(generator);
    return native_components::archive_diverse(input,count,continuous,b.random,*math);
  });
  m.def("imperialist",[](py::array_t<double,py::array::c_style> input,
              py::array_t<double,py::array::c_style> fits,
              py::array_t<double,py::array::c_style> bounds,
              const std::vector<double> &parameters,py::dict state,py::object generator) {
    if(input.ndim()!=2||fits.size()!=input.shape(0)||bounds.ndim()!=2||bounds.shape(0)!=2||bounds.shape(1)!=input.shape(1))
      throw py::value_error("Incompatible imperialist inputs");
    native_components::RandomBoundary b(generator);
    return native_components::imperialist(input,fits,bounds,parameters,native_components::NativeState(state,true),b.random).python();
  });
  m.def("particle_fly",[](py::array_t<double,py::array::c_style> input,
              py::array_t<double,py::array::c_style> velocity,
              py::array_t<double,py::array::c_style> personal,
              py::array_t<double,py::array::c_style> guide,
              py::array_t<int64_t,py::array::c_style> targets,double inertia,py::dict state,bool context,py::object generator) {
    native_components::RandomBoundary b(generator);
    return native_components::particle_fly(input,velocity,personal,guide,targets,inertia,native_components::NativeState(state,true),context,b.random).python();
  });
  m.def("discrete",[](const std::string &name,
             py::array_t<int64_t,py::array::c_style> input,
             py::array_t<int64_t,py::array::c_style> bounds,
             const std::vector<double> &parameters,py::object generator) {
    if(input.ndim()!=2 || bounds.ndim()!=2 || bounds.shape(0)!=2 || bounds.shape(1)!=input.shape(1))
      throw py::value_error("Expected a matrix and (2, dimension) bounds");
    native_components::RandomBoundary b(generator);
    return native_components::discrete_variation(name,input,bounds,parameters,b.random);
  });
  m.def("bit",[math](const std::string &name,
             py::array_t<int64_t,py::array::c_style> input,
             py::array_t<int64_t,py::array::c_style> bounds,py::dict state,py::object generator) {
    if(input.ndim()!=2 || bounds.ndim()!=2 || bounds.shape(0)!=2 || bounds.shape(1)!=input.shape(1))
      throw py::value_error("Expected a matrix and (2, dimension) bounds");
    native_components::RandomBoundary b(generator);
    return native_components::bit_variation(name,input,bounds,native_components::NativeState(state,true),b.random,*math).python();
  });
  m.def("point_crossover",[](const std::string &name,py::array input,
              const std::vector<double> &parameters,py::dict state,py::object generator) {
    if(input.ndim()!=2)throw py::value_error("Expected a two-dimensional parent array");
    native_components::RandomBoundary b(generator);
    if(input.dtype().is(py::dtype::of<int64_t>()))
      return native_components::point_crossover<int64_t>(name,input,parameters,native_components::NativeState(state,true),b.random).python();
    if(input.dtype().is(py::dtype::of<double>()))
      return native_components::point_crossover<double>(name,input,parameters,native_components::NativeState(state,true),b.random).python();
    throw py::type_error("Native crossover supports float64 and int64 arrays");
  });
  m.def("order_crossover",[](const std::string &name,
              py::array_t<int64_t,py::array::c_style> input,int64_t points,py::object generator) {
    if(input.ndim()!=2)throw py::value_error("Expected a two-dimensional parent array");
    native_components::RandomBoundary b(generator);
    return native_components::order_crossover(name,input,points,b.random);
  });
  m.def("_x86_kernel_info",[]{auto c=aol_x86::capabilities();return py::dict("avx2"_a=c.avx2,"avx512dq_vl"_a=c.convert,"popcnt"_a=c.popcnt,"mode"_a=static_cast<int>(aol_x86::mode()));});
  m.def("_set_x86_kernel_mode",[](int mode){if(mode< -1||mode>4)throw py::value_error("Invalid private kernel mode");aol_x86::mode_override=mode;});
  m.def("matmul",[math](py::array_t<double> a,py::array_t<double> b) {
    return math->invoke(math->matmul,a,b);
  });
  m.def("power",[math](py::array_t<double> a,py::array_t<double> b) {
    return math->invoke(math->power,a,b);
  });
  m.def("eigh",[math](py::array_t<double> a) {return math->invoke(math->eigh,a);});
  m.def("cholesky",[math](py::array_t<double> a) {return math->invoke(math->cholesky,a);});
  m.def("solve",[math](py::array_t<double> a,py::array_t<double> b) {
    return math->invoke(b.ndim()==1?math->solve_vector:math->solve_matrix,a,b);
  });
  m.def("sum",[](py::array input,int axis) {
    auto result=PyArray_Sum(reinterpret_cast<PyArrayObject *>(input.ptr()),axis,NPY_DOUBLE,nullptr);
    if(!result)throw py::error_already_set();return py::reinterpret_steal<py::object>(result);
  },"input"_a,"axis"_a=AOL_RAVEL_AXIS);
  m.def("argsort",[](py::array input,int axis,bool stable) {
    auto result=PyArray_ArgSort(reinterpret_cast<PyArrayObject *>(input.ptr()),axis,
                             stable?NPY_STABLESORT:NPY_QUICKSORT);
    if(!result)throw py::error_already_set();return py::reinterpret_steal<py::object>(result);
  },"input"_a,"axis"_a=-1,"stable"_a=false);
  m.def("mean",[](py::array input,int axis) {
    auto result=PyArray_Mean(reinterpret_cast<PyArrayObject *>(input.ptr()),axis,NPY_DOUBLE,nullptr);
    if(!result)throw py::error_already_set();return py::reinterpret_steal<py::object>(result);
  },"input"_a,"axis"_a=AOL_RAVEL_AXIS);
  m.def("std",[](py::array input,int axis,int ddof) {
    return native_components::standard_deviation(input,axis,ddof);
  },"input"_a,"axis"_a=AOL_RAVEL_AXIS,"ddof"_a=0);
}

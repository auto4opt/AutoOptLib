#pragma once
#include "component_kind.hpp"
// Public component boundary. Standard Solution/SolutionSet records are read
// directly; no decs(), fits(), component helper, or Generator method is called.
namespace native_components {

// Set when the dense search storage is registered. Its fields are C++ owners,
// not a Python instance dictionary; all other compatibility reads stay exact.
inline PyTypeObject *population_storage_type = nullptr;

inline py::object field(py::handle object,const char *key,py::object fallback=py::none()) {
  if(object.is_none())return fallback;
  if(population_storage_type && PyObject_TypeCheck(object.ptr(), population_storage_type)) {
    auto value=PyObject_GetAttrString(object.ptr(),key);
    if(value)return py::reinterpret_steal<py::object>(value);
    if(!PyErr_ExceptionMatches(PyExc_AttributeError))throw py::error_already_set();
    PyErr_Clear();return fallback;
  }
  py::object dictionary;
  if(PyDict_CheckExact(object.ptr()))dictionary=py::reinterpret_borrow<py::object>(object);
  else {
    auto raw=PyObject_GenericGetDict(object.ptr(),nullptr);
    if(!raw){PyErr_Clear();return fallback;}
    dictionary=py::reinterpret_steal<py::object>(raw);
  }
  auto raw=PyDict_GetItemString(dictionary.ptr(),key);
  return raw?py::reinterpret_borrow<py::object>(raw):fallback;
}
// Only flex_get consumers use these aliases; exact getattr/state reads stay exact.
inline py::object flex_field(py::handle object,const char *key,py::object fallback=py::none()) {
  const std::string_view name(key);std::array<const char *,3> names{key,nullptr,nullptr};
  if(name=="N")names={"N","n",nullptr};
  else if(name=="Gmax")names={"Gmax","gmax","GMAX"};
  else if(name=="bound")names={"bound","BOUND","Bound"};
  else if(name=="type")names={"type","TYPE","Type"};
  else if(name=="_selection_count")names={"_selection_count","_SELECTION_COUNT","SelectionCount"};
  // field returns its fallback unchanged; a null handle distinguishes a
  // missing field from a present None without allocating a Python sentinel.
  for(auto candidate:names) {
    if(!candidate)break;
    auto value=field(object,candidate,py::object());if(value)return value;
  }
  return fallback;
}
inline double total_violation(py::handle item) {
  auto value = field(item, "con");
  return value.is_none() ? 0. : py::cast<double>(value);
}
inline auto record_fitness_key(py::handle item) {
  return feasibility_key(py::cast<double>(field(item, "fit")), total_violation(item));
}
inline bool less_fitness(py::handle a,py::handle b) {
  return record_fitness_key(a) < record_fitness_key(b);
}
inline py::list records(py::handle source) {
  auto items=field(source,"_items");
  if(!items.is_none())return py::cast<py::list>(items);
  if(PyList_CheckExact(source.ptr()))return py::reinterpret_borrow<py::list>(source);
  if(PyTuple_CheckExact(source.ptr()))return py::reinterpret_steal<py::list>(PySequence_List(source.ptr()));
  throw py::type_error("Native components require standard SolutionSet, list, tuple, or ndarray inputs");
}
inline std::vector<py::array> decision_rows(py::handle source,bool integer=false) {
  auto items=records(source);
  std::vector<py::array> rows;rows.reserve(items.size());py::ssize_t d=0;
  const int dtype=integer?NPY_INT64:NPY_DOUBLE;
  for(auto item:items) {
    auto value=field(item,"dec");
    if(value.is_none())throw py::type_error("Native component records require a dec array");
    auto row=py::reinterpret_borrow<py::array>(py::handle());
    if(PyArray_CheckExact(value.ptr()) && PyArray_TYPE(reinterpret_cast<PyArrayObject *>(value.ptr()))==dtype
        && PyArray_ISCARRAY_RO(reinterpret_cast<PyArrayObject *>(value.ptr()))
        && PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject *>(value.ptr()))
        && PyArray_NDIM(reinterpret_cast<PyArrayObject *>(value.ptr()))==1)
      row=py::reinterpret_borrow<py::array>(value);
    else {
      auto raw=PyArray_FromAny(value.ptr(),PyArray_DescrFromType(dtype),1,1,NPY_ARRAY_CARRAY_RO|NPY_ARRAY_FORCECAST,nullptr);
      if(!raw)throw py::error_already_set();row=py::reinterpret_steal<py::array>(raw);
    }
    if(rows.empty())d=row.size();
    if(row.size()!=d)throw py::value_error("Decision rows must have the same dimension");
    rows.push_back(std::move(row));
  }
  return rows;
}
inline py::array matrix(py::handle source,bool integer=false) {
  auto dense=field(source,"_batch_X");
  if(!dense.is_none())return matrix(dense,integer);
  if(PyArray_CheckExact(source.ptr())) {
    auto dtype=integer?NPY_INT64:NPY_DOUBLE;
    auto raw=PyArray_FromAny(source.ptr(),PyArray_DescrFromType(dtype),2,2,NPY_ARRAY_CARRAY_RO|NPY_ARRAY_FORCECAST,nullptr);
    if(!raw)throw py::error_already_set();return py::reinterpret_steal<py::array>(raw);
  }
  auto rows=decision_rows(source,integer);auto n=rows.size();
  py::ssize_t d=n?rows[0].size():0;
  // Read-only numerical inputs can borrow consecutive rows of one allocation.
  // The base reference owns the storage; no cache survives a callback.
  if(n && d) {
    auto base=rows[0].base();
    bool contiguous=n==1 || (base && !base.is_none());
    for(size_t i=1;i<n&&contiguous;++i)
      contiguous=rows[i].base().is(base) && static_cast<const char *>(rows[i].data())==static_cast<const char *>(rows[0].data())+i*d*8;
    if(contiguous)return py::array(rows[0].dtype(),
        {static_cast<py::ssize_t>(n),d},{d*8,py::ssize_t(8)},rows[0].data(),rows[0]);
  }
  py::array out(integer?py::dtype::of<int64_t>():py::dtype::of<double>(),{static_cast<py::ssize_t>(n),d});
  for(size_t i=0;i<n;++i)std::memcpy(static_cast<char *>(out.mutable_data())+i*d*8,rows[i].data(),d*8);
  return out;
}
// Sparse consumers read donor/personal-best rows without packing the population.
// Conversion and rectangular-shape checks still finish before any RNG draws.
struct DecisionRows {
  std::vector<py::array> rows;
  std::vector<const double *> pointers;
  py::object dense_owner=py::none();
  const double *dense_data=nullptr;
  py::ssize_t n=0,d=0;
  DecisionInput row_input;
  explicit DecisionRows(const NativePopulationOwner &source,bool indexed=false);
  explicit DecisionRows(py::handle source) {
    if(PyArray_CheckExact(source.ptr())||!field(source,"_batch_X").is_none()) {
      auto dense=matrix(source);n=dense.shape(0);d=dense.shape(1);
      dense_data=static_cast<const double *>(dense.data());dense_owner=dense;
    } else {
      rows=decision_rows(source);n=rows.size();d=n?rows[0].size():0;
      pointers.reserve(n);for(auto &row:rows)pointers.push_back(static_cast<const double *>(row.data()));
    }
  }
  int ndim() const {return 2;}
  py::ssize_t shape(int axis) const {return axis==0?n:d;}
  const double *row(py::ssize_t i) const {return dense_data?dense_data+i*d:pointers[i];}
  double at(py::ssize_t i,py::ssize_t j) const {
    return row(i)[j];
  }
};
inline FitnessValues fitness_values(py::handle source) {
  if((population_storage_type&&PyObject_TypeCheck(source.ptr(),population_storage_type))||!field(source,"_batch_X").is_none()) {
    auto value=field(source,"_batch_F");
    if(value.is_none())throw py::value_error("Batch component requires evaluated fitness");
    auto violations=field(source,"constraints");
    return {
        py::array_t<double,py::array::c_style>::ensure(value),
        violations.is_none() ? py::reinterpret_borrow<py::array_t<double>>(py::none()) :
            py::array_t<double>(py::array_t<double,py::array::c_style>::ensure(violations)),
    };
  }
  auto items=records(source);py::array_t<double> result(items.size()), violations(items.size());
  for(size_t i=0;i<items.size();++i) {
    auto value=field(items[i],"fit");
    if(value.is_none())throw py::type_error("Native component records require scalar fit");
    result.mutable_data()[i]=py::cast<double>(value);
    violations.mutable_data()[i]=total_violation(items[i]);
  }
  return {result, violations};
}
inline py::list select_records(py::handle source,py::array_t<int64_t> indices) {
  auto items=records(source);py::list selected;
  for(py::ssize_t i=0;i<indices.size();++i)selected.append(items[indices.data()[i]]);
  return selected;
}
inline py::object solution_set(py::handle type,py::list items) {
  auto result=PyType_GenericAlloc(reinterpret_cast<PyTypeObject *>(type.ptr()),0);
  if(!result)throw py::error_already_set();auto out=py::reinterpret_steal<py::object>(result);
  if(PyObject_GenericSetAttr(result,py::str("_items").ptr(),items.ptr())<0)throw py::error_already_set();
  return out;
}
inline std::vector<double> parameters(py::handle value) {
  if(value.is_none())return {};
  auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(value);
  if(!a)throw py::type_error("Parameters must be numeric");return {a.data(),a.data()+a.size()};
}
inline void load_parameters(py::handle value,std::vector<double>&output) {
  output.clear();if(value.is_none())return;
  auto array=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(value);
  if(!array)throw py::type_error("Parameters must be numeric");output.assign(array.data(),array.data()+array.size());
}
inline py::array_t<double> bounds(py::handle problem,py::ssize_t d,bool flexible=false) {
  auto bound=flexible?flex_field(problem,"bound"):field(problem,"bound");
  if(bound.is_none()) {py::array_t<double> result({py::ssize_t(2),d});std::fill_n(result.mutable_data(),2*d,0.);return result;}
  auto a=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(bound);
  if(!a||a.ndim()!=2||a.shape(0)!=2||a.shape(1)!=d)throw py::value_error("Expected (2, dimension) bounds");
  return a;
}
inline py::object component_generator(py::handle aux,py::handle problem,py::handle generator_type,py::handle default_rng,bool problem_rng) {
  auto expected=reinterpret_cast<PyTypeObject *>(generator_type.ptr());
  for(auto source:{aux,problem_rng?problem:py::handle(Py_None)}) {
    if(Py_TYPE(source.ptr())==expected)return py::reinterpret_borrow<py::object>(source);
    if(PyType_IsSubtype(Py_TYPE(source.ptr()),expected))throw py::type_error("Native components require a standard NumPy Generator; use the Python backend for Generator subclasses");
    auto rng=field(source,"rng");
    if(Py_TYPE(rng.ptr())==expected)return rng;
    if(PyType_IsSubtype(Py_TYPE(rng.ptr()),expected))throw py::type_error("Native components require a standard NumPy Generator; use the Python backend for Generator subclasses");
  }
  // Constructing a missing stream is a boundary operation; the native core
  // receives its capsule. Never call user RNG subclasses from inside a core.
  auto raw=PyObject_CallNoArgs(default_rng.ptr());
  if(!raw)throw py::error_already_set();return py::reinterpret_steal<py::object>(raw);
}
inline py::object component_generator(NativeState state,py::handle problem,py::handle generator_type,py::handle default_rng,bool problem_rng) {
  auto rng=state.object("rng");auto expected=reinterpret_cast<PyTypeObject *>(generator_type.ptr());
  if(Py_TYPE(rng.ptr())==expected)return rng;
  if(PyType_IsSubtype(Py_TYPE(rng.ptr()),expected))throw py::type_error("Native components require a standard NumPy Generator; use the Python backend for Generator subclasses");
  return component_generator(py::none(),problem,generator_type,default_rng,problem_rng);
}

} // namespace native_components
#include "batch_storage.hpp"
namespace native_components {
inline DecisionRows::DecisionRows(const NativePopulationOwner &source,bool indexed) {
  if(indexed&&!source->integer()&&source->rows_compatible()) {
    row_input=BatchStorage::read_rows(source);n=row_input.n;d=row_input.d;
    pointers.reserve(n);for(py::ssize_t i=0;i<n;++i)pointers.push_back(static_cast<const double *>(row_input.row(i)));return;
  }
  auto dense=matrix(source->dense());n=dense.shape(0);d=dense.shape(1);
  dense_data=static_cast<const double *>(dense.data());dense_owner=dense;
}

struct ComponentInvocation {
  RandomBinding *random_binding=nullptr;
  NativePopulationOwner population;
  py::object source=py::none(),problem=py::none(),parameter=py::none(),data=py::none();
  NativeState state;
  py::object raw_auxiliary=py::none(),raw_generation=py::none(),raw_inner=py::none();
  int64_t generation_value=0,inner_value=1;
  bool has_state=true,algorithm=false,native_scalars=true,indexed_inputs=false;
  int64_t generation()const{return native_scalars?generation_value:raw_generation.is_none()?0:py::cast<int64_t>(raw_generation);}
  int64_t inner()const{return native_scalars?inner_value:raw_inner.is_none()?1:py::cast<int64_t>(raw_inner);}
};

template<class Math>
struct Component {
  std::string name;
  std::shared_ptr<Math> math;
  py::object solution_type,generator_type,default_rng;
  ComponentMetadata metadata{name};
  py::object operator()(py::args args) const {
    if(args.size()<2)throw py::type_error("Expected component arguments and execution mode");
    bool has_mode=PyUnicode_Check(args[args.size()-1].ptr());
    auto count=args.size()-(has_mode?1:0);
    auto mode=has_mode?py::cast<std::string>(args[args.size()-1]):"solution";
    if(mode!="execute"&&mode!="algorithm"&&mode!="solution")throw py::value_error("Native component entry accepts execution modes only");
    auto arg=[&](size_t i)->py::object{return i<count?py::reinterpret_borrow<py::object>(args[i]):py::none();};
    ComponentInvocation call;call.source=arg(0);call.problem=arg(1);call.parameter=arg(2);
    call.algorithm=mode=="algorithm";call.native_scalars=false;
    call.raw_auxiliary=arg(3);
    if(metadata.para||(call.algorithm&&(metadata.op==Operation::search_cma||metadata.op==Operation::search_mu_polynomial)))call.raw_auxiliary=arg(2);
    call.has_state=PyDict_CheckExact(call.raw_auxiliary.ptr());
    if(call.has_state)call.state=NativeState(py::reinterpret_borrow<py::dict>(call.raw_auxiliary),true);
    call.raw_generation=arg(4);call.raw_inner=arg(5);call.data=arg(6);
    return execute(call).python();
  }
  NativeResult execute(const ComponentInvocation &call) const {
    const bool pure_integer=call.indexed_inputs&&call.population&&call.population->integer()&&
      (metadata.point_normal||metadata.point_fastga)&&call.random_binding&&call.random_binding->valid(call.state.object("rng"));
    FloatingPolicy::Scope floating_scope(math->floating,!pure_integer);
    auto source=call.source,problem=call.problem,raw_aux=call.raw_auxiliary;
    bool algorithm=call.algorithm;
    auto state=call.state;
    auto auxiliary_result=[&](py::object value){return call.has_state?state_result(value,state):state_result(value,raw_aux);};
    if(metadata.op==Operation::archive_tabu)return state_result(source,py::none());
    bool batch=bool(call.population)||(population_storage_type&&PyObject_TypeCheck(source.ptr(),population_storage_type))||!field(source,"_batch_X").is_none();
    auto source_matrix=[&](bool integer=false){return call.population?matrix(call.population->dense(),integer):matrix(source,integer);};
    auto source_values=[&](){
      if(call.population){
        if(!call.population->evaluated)throw py::value_error("Batch component requires evaluated fitness");
        return call.population->fitness_values();
      }
      return fitness_values(source);
    };
    auto source_fitness=[&](){return source_values().selection_scores();};
    if(metadata.op==Operation::para_pso) {
      if(!state.contains("Pbest"))return state;
      if(batch) {
        auto personal_owner=state.population("Pbest");
        auto [personal,current]=shared_selection_scores(
            personal_owner?personal_owner->fitness_values():fitness_values(state.object("Pbest")),
            source_values());
        auto n=personal.size();py::array_t<int64_t> indices(n);py::array_t<double> chosen(n);
        for(py::ssize_t i=0;i<n;++i){indices.mutable_data()[i]=i;chosen.mutable_data()[i]=personal.data()[i];}
        auto raw_targets=state.index_array("_batch_pso_targets");
        auto targets=py::reinterpret_borrow<py::array_t<int64_t,py::array::c_style|py::array::forcecast>>(py::handle());
        if(!raw_targets.is_none()) {
          targets=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(raw_targets);
          if(!targets||targets.size()!=current.size())throw py::value_error("PSO target count must match evaluated rows");
        }
        for(py::ssize_t j=0;j<current.size();++j) {
          auto i=raw_targets.is_none()?j:targets.data()[j];
          if(raw_targets.is_none()&&i>=n)break;
          if(i<0||i>=n)throw py::index_error("PSO target exceeds personal-best rows");
          if(current.data()[j]<chosen.data()[i]){indices.mutable_data()[i]=n+j;chosen.mutable_data()[i]=current.data()[j];}
        }
        auto best=PyArray_ArgMin(reinterpret_cast<PyArrayObject *>(chosen.ptr()),AOL_RAVEL_AXIS,nullptr);
        if(!best)throw py::error_already_set();
        return state_result(indices,py::reinterpret_steal<py::object>(best));
      }
      auto personal=records(state.object("Pbest")),current=records(source);py::list next;
      auto count=std::min(personal.size(),current.size());
      for(size_t i=0;i<personal.size();++i)next.append(i<count&&less_fitness(current[i],personal[i])?current[i]:personal[i]);
      state.set("Pbest",solution_set(solution_type,next));
      auto f=fitness_values(state.object("Pbest")).selection_scores();
      auto best=PyArray_ArgMin(reinterpret_cast<PyArrayObject *>(f.ptr()),AOL_RAVEL_AXIS,nullptr);
      if(!best)throw py::error_already_set();auto index=py::cast<int64_t>(py::reinterpret_steal<py::object>(best));
      state.set("Gbest",next[index]);return state;
    }
    if(metadata.op==Operation::para_cma||metadata.op==Operation::para_cmaes) {
      auto f=py::reinterpret_borrow<py::array_t<double,py::array::c_style>>(py::handle());
      if(algorithm&&!batch)f=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(field(source,"avePerformAll"));
      else f=source_fitness();
      auto target=(PyList_CheckExact(problem.ptr())||PyTuple_CheckExact(problem.ptr()))?py::reinterpret_borrow<py::object>(problem[py::int_(0)]):problem;
      auto b=field(target,"bound");py::array_t<double,py::array::c_style> bound= b.is_none()?py::array_t<double,py::array::c_style|py::array::forcecast>(0):py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(b);
      return cma_update(f,bound,state,!algorithm,*math);
    }
    if(metadata.op==Operation::archive_best||metadata.op==Operation::archive_statistic) {
      if(metadata.op==Operation::archive_statistic) {
        auto f=source_values().values;
        auto previous=call.problem;auto old=py::reinterpret_borrow<py::array_t<double,py::array::c_style>>(py::handle());
        if(previous.is_none()||(PyList_CheckExact(previous.ptr())&&PyList_GET_SIZE(previous.ptr())==0))old=py::array_t<double>(std::vector<py::ssize_t>{0,2});
        else old=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(previous);
        return state_result(archive_statistics(f,old,call.population&&!call.population->external),py::none());
      }
      if(batch) {
        auto f=source_fitness();
        auto raw=PyArray_ArgMin(reinterpret_cast<PyArrayObject *>(f.ptr()),AOL_RAVEL_AXIS,nullptr);
        if(!raw)throw py::error_already_set();
        py::array_t<int64_t> index(1);index.mutable_data()[0]=py::cast<int64_t>(py::reinterpret_steal<py::object>(raw));
        return state_result(index,py::none());
      }
      auto f=source_fitness();
      auto raw=PyArray_ArgMin(reinterpret_cast<PyArrayObject *>(f.ptr()),AOL_RAVEL_AXIS,nullptr);
      if(!raw)throw py::error_already_set();auto best=py::cast<int64_t>(py::reinterpret_steal<py::object>(raw));
      auto previous=call.problem;py::list result;
      if(!previous.is_none())for(auto item:records(previous))result.append(item);
      result.append(records(source)[best]);return state_result(result,py::none());
    }
    bool problem_rng=metadata.problem_rng,random_required=metadata.random_required;
    Random unused_rng(nullptr);std::optional<RandomBoundary> boundary;
    if(random_required) {
      auto generator=metadata.op==Operation::archive_diversity?component_generator(call.parameter,problem,generator_type,default_rng,problem_rng):
          call.has_state?component_generator(state,problem,generator_type,default_rng,problem_rng):
          component_generator(raw_aux,problem,generator_type,default_rng,problem_rng);
      boundary.emplace(generator,call.random_binding);
    }
    auto &rng=boundary?boundary->random:unused_rng;
    auto raw_parameter=call.parameter;
    ScratchVector<double> parameter_lease;auto &para=parameter_lease.values;
    if(metadata.op!=Operation::archive_diversity&&!algorithm)load_parameters(raw_parameter,para);
    bool default_parameter=raw_parameter.is_none()||algorithm||metadata.op==Operation::archive_diversity;
    if(metadata.op==Operation::update_simulated_annealing&&para.empty())
      throw py::value_error("Parameter for simulated annealing must be convertible to float");
    if(!default_parameter&&para.empty()) {
      bool accepts_empty=metadata.op==Operation::choose_brainstorm||metadata.op==Operation::search_mu_polynomial||metadata.op==Operation::search_ica||metadata.op==Operation::search_reset_creep;
      bool ignores_parameter=metadata.op==Operation::choose_random||metadata.op==Operation::choose_traverse||metadata.op==Operation::choose_nich||metadata.op==Operation::choose_tournament||metadata.op==Operation::choose_stochastic_tournament||metadata.op==Operation::choose_fastga_stochastic_tournament||metadata.op==Operation::choose_roulette_wheel||metadata.op==Operation::choose_fastga_proportional||metadata.op==Operation::search_eda||metadata.op==Operation::search_cma||metadata.op==Operation::search_mu_gaussian||metadata.op==Operation::search_mu_cauchy||metadata.reinit||(metadata.bit_normal||metadata.bit_fastga)||metadata.op==Operation::search_swap||metadata.op==Operation::search_swap_multi||metadata.op==Operation::search_insert||metadata.op==Operation::search_scramble||metadata.op==Operation::search_reset_one||metadata.op==Operation::cross_order_two||metadata.op==Operation::cross_point_one||metadata.op==Operation::cross_point_two||metadata.point_fastga||metadata.op==Operation::cross_point_three||metadata.op==Operation::cross_point_five||metadata.update;
      if(!accepts_empty&&!ignores_parameter) {
        if(metadata.op==Operation::cross_arithmetic)throw py::value_error("Arithmetic ratio cannot be empty");
        throw py::index_error("index 0 is out of bounds for axis 0 with size 0");
      }
    }
    auto parameter=[&](size_t i,double value){return para.size()>i?para[i]:value;};
    int64_t generation=call.generation();
    int64_t maximum=py::cast<int64_t>(flex_field(problem,"Gmax",py::int_(generation+1)));
    if(metadata.op==Operation::archive_diversity) {
      if(batch) {
        auto p=call.parameter,type=flex_field(p,"type",py::str("continuous"));
        if(PyList_CheckExact(type.ptr())||PyTuple_CheckExact(type.ptr()))type=type[py::int_(0)];
        auto values=source_matrix();
        auto count=py::cast<int64_t>(flex_field(p,"N",py::int_(values.shape(0))));
        return state_result(archive_diverse(values,count,py::cast<std::string>(type)=="continuous",rng,*math),py::none());
      }
      py::list combined;for(auto x:records(source))combined.append(x);
      auto archive=call.problem;if(!archive.is_none())for(auto x:records(archive))combined.append(x);
      if(!combined.size())return state_result(combined,py::none());
      auto p=call.parameter,type=flex_field(p,"type",py::str("continuous"));
      if(PyList_CheckExact(type.ptr())||PyTuple_CheckExact(type.ptr()))type=type[py::int_(0)];
      auto count=py::cast<int64_t>(flex_field(p,"N",py::int_(records(source).size())));
      auto indices=archive_diverse(matrix(combined),count,py::cast<std::string>(type)=="continuous",rng,*math);
      return state_result(select_records(combined,indices),py::none());
    }
    if(metadata.choose) {
      auto n=call.population?call.population->size():batch?(population_storage_type&&PyObject_TypeCheck(source.ptr(),population_storage_type)?py::cast<py::ssize_t>(field(source,"_batch_size")):py::cast<py::array>(field(source,"_batch_X")).shape(0)):PyArray_CheckExact(source.ptr())?py::cast<py::array>(source).shape(0):static_cast<py::ssize_t>(records(source).size());
      if(metadata.op==Operation::choose_traverse&&!n)n=py::cast<int64_t>(flex_field(problem,"N",py::int_(0)));
      auto requested=state.object("_selection_count",state.object("_SELECTION_COUNT",state.object("SelectionCount")));
      int64_t count=py::cast<int64_t>(requested.is_none()?flex_field(problem,"N",py::int_(n)):requested);
      if(metadata.op==Operation::choose_traverse&&requested.is_none()) {
        if(!n)n=py::cast<int64_t>(flex_field(problem,"N",py::int_(0)));
        py::array_t<int64_t> out(n);std::iota(out.mutable_data(),out.mutable_data()+n,0);return state_result(out,py::none());
      }
      if(metadata.op==Operation::choose_traverse||metadata.op==Operation::choose_random) {
        // These selectors consume only the population size, never fitness.
        // Keep the old placeholder array's shape error and error precedence.
        if(n<0)throw py::value_error("negative dimensions are not allowed");
        auto absent=py::reinterpret_borrow<py::array_t<double,py::array::c_style>>(py::handle());
        return selection(name,n,absent,rng,count,parameter(0,.2),state);
      }
      auto f=source_fitness();
      if(metadata.op==Operation::choose_nich)return state_result(niche(source_matrix(),f,generation,maximum,rng,*math),py::none());
      if(metadata.op==Operation::choose_brainstorm) {
        auto type=flex_field(problem,"type",py::str("continuous"));
        if(PyList_CheckExact(type.ptr())||PyTuple_CheckExact(type.ptr()))type=type[py::int_(0)];
        if(py::cast<std::string>(type).find("continuous")==std::string::npos)throw py::value_error("choose_brainstorm is only available for continuous problems");
        return state_result(brainstorm(source_matrix(),f,std::max<int64_t>(1,std::nearbyint(parameter(0,default_parameter?2:.5))),parameter(1,.5),count,rng,*math),py::none());
      }
      return selection(name,f.size(),f,rng,count,parameter(0,metadata.op==Operation::choose_ica?1:.2),state);
    }
    if(metadata.update) {
      if(batch) {
        auto values=source_values();auto f=values.selection_scores();auto size=f.size();
        bool full_default=metadata.op==Operation::update_always||metadata.op==Operation::update_greedy||metadata.op==Operation::update_round_robin||metadata.op==Operation::update_iterated_local_search;
        auto n=py::cast<int64_t>(flex_field(problem,"N",py::int_(full_default?size:size/2)));
        return replacement(name,f,n,rng,parameter(0,.5),generation,maximum,state,math->exponential,values);
      }
      auto items=records(source);auto size=items.size();
      bool full_default=metadata.op==Operation::update_always||metadata.op==Operation::update_greedy||metadata.op==Operation::update_round_robin||metadata.op==Operation::update_iterated_local_search;
      auto n=py::cast<int64_t>(flex_field(problem,"N",py::int_(full_default?size:size/2)));
      if(metadata.op==Operation::update_pairwise) {
        n=std::min<int64_t>(n,size);py::list selected;
        for(int64_t i=0;i<n;++i)selected.append(items[i]);
        for(int64_t i=n;i<static_cast<int64_t>(size);++i) {
          if(!n){PyErr_SetString(PyExc_ZeroDivisionError,"integer modulo by zero");throw py::error_already_set();}
          auto target=(i-n)%n;if(less_fitness(items[i],selected[target]))selected[target]=items[i];
        }
        return state_result(selected,py::none());
      }
      auto values=source_values();auto f=values.selection_scores();
      auto result=replacement(name,f,n,rng,parameter(0,.5),generation,maximum,state,math->exponential,values);
      result.value=select_records(source,py::cast<py::array_t<int64_t>>(result.value));return result;
    }
    auto decision_source=call.population&&call.indexed_inputs?py::object(py::none()):call.population?py::object(call.population->dense()):batch?field(source,"_batch_X"):source;
    bool point_integer=(metadata.point_normal||metadata.point_fastga) &&
        (call.population&&call.indexed_inputs?call.population->integer():PyArray_CheckExact(decision_source.ptr())?py::cast<py::array>(decision_source).dtype().is(py::dtype::of<int64_t>()):(records(source).size()!=0&&py::cast<py::array>(field(records(source)[0],"dec")).dtype().is(py::dtype::of<int64_t>())));
    bool discrete=point_integer||metadata.bit_normal||metadata.bit_fastga||metadata.reset||metadata.op==Operation::search_swap||metadata.op==Operation::search_swap_multi||metadata.op==Operation::search_scramble||metadata.op==Operation::search_insert||metadata.op==Operation::reinit_discrete||metadata.op==Operation::reinit_permutation||metadata.order;
    auto execute_numerical=[&](const auto &input)->NativeResult {
    auto d=input.shape(1);
    auto native_context=state.population("_population_context");
    auto context=native_context?py::object(py::none()):state.object("_population_context");bool has_context=bool(native_context)||!context.is_none();
    auto context_matrix=[&](){return native_context?matrix(native_context->dense()):matrix(context);};
    if(metadata.op==Operation::search_pso) {
      auto population=has_context?context:source;
      auto native_population=has_context?native_context:call.population;
      if(batch) {
        DecisionRows model=native_population?DecisionRows(native_population,call.indexed_inputs):DecisionRows(population);auto n=model.shape(0);
        if(!state.contains("Pbest")) {
          if(native_population)state.set_population("Pbest",native_population);else state.set("Pbest",population);
          auto best=rng.integer(0,n);py::array_t<double> guide(d);
          for(py::ssize_t j=0;j<d;++j)guide.mutable_data()[j]=model.at(best,j);state.set("Gbest",guide);
          py::array_t<double> velocity({n,d});std::fill_n(velocity.mutable_data(),n*d,0.);state.set("V",velocity);
        }
        if(!state.contains("V")) {py::array_t<double> velocity({n,d});std::fill_n(velocity.mutable_data(),n*d,0.);state.set("V",velocity);}
        auto velocity=py::array_t<double>::ensure(state.object("V"));
        if(!velocity)throw py::value_error("Particle velocity must be numeric");
        auto targets=py::reinterpret_borrow<py::array_t<int64_t,py::array::c_style>>(py::handle());
        if(has_context) {
          targets=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(state.object("_target_population_indices"));
          if(!targets||targets.size()!=input.shape(0))throw py::value_error("Batch PSO requires one population index per target");
        } else {targets=py::array_t<int64_t>(input.shape(0));std::iota(targets.mutable_data(),targets.mutable_data()+targets.size(),0);}
        auto guide_object=state.object("Gbest");
        auto guide=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(guide_object);
        if(!guide)throw py::value_error("Batch PSO guide must be a numeric row");
        auto personal=state.population("Pbest");
        return personal?particle_fly_rows(input,velocity,DecisionRows(personal,call.indexed_inputs),guide,targets,parameter(0,.5),state,has_context,rng):
            particle_fly_rows(input,velocity,DecisionRows(state.object("Pbest")),guide,targets,parameter(0,.5),state,has_context,rng);
      }
      auto population_items=records(population);auto n=population_items.size();
      if(!state.contains("Pbest")) {
        py::list personal;for(auto item:population_items)personal.append(item);
        state.set("Pbest",solution_set(solution_type,personal));state.set("Gbest",personal[rng.integer(0,n)]);
        py::array_t<double> velocity({static_cast<py::ssize_t>(n),d});std::fill_n(velocity.mutable_data(),n*d,0.);state.set("V",velocity);
      }
      auto velocity=state.contains("V")?py::array_t<double>::ensure(state.object("V")):py::array_t<double>({static_cast<py::ssize_t>(n),d});
      if(!velocity)throw py::value_error("Particle velocity must be numeric");
      if(!state.contains("V"))std::fill_n(velocity.mutable_data(),n*d,0.);
      auto targets=py::reinterpret_borrow<py::array_t<int64_t,py::array::c_style>>(py::handle());
      if(has_context) {
        targets=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(state.object("_target_population_indices"));
        if(!targets||targets.size()!=input.shape(0))throw py::value_error("Stream PSO requires one current-population index per target.");
      }else {targets=py::array_t<int64_t>(input.shape(0));std::iota(targets.mutable_data(),targets.mutable_data()+targets.size(),0);}
      for(py::ssize_t i=0;i<targets.size();++i)if(targets.data()[i]<0||targets.data()[i]>=static_cast<int64_t>(n))throw py::value_error("Stream PSO requires one current-population index per target.");
      auto guide=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(field(state.object("Gbest"),"dec"));
      if(!guide)throw py::value_error("Particle guide must be numeric");
      return particle_fly_rows(input,velocity,DecisionRows(state.object("Pbest")),guide,targets,parameter(0,.5),state,has_context,rng);
    }
    if(metadata.point_normal||metadata.point_fastga) {
      // Preserve the parent's integer dtype for discrete crossovers.
      bool integer=point_integer;
      auto result=integer?point_crossover<int64_t>(name,input,para,state,rng,call.population&&!call.population->external):point_crossover<double>(name,input,para,state,rng,call.population&&!call.population->external);
      if(!metadata.point_fastga)return auxiliary_result(result.value);
      return result;
    }
    if(metadata.order)return auxiliary_result(order_crossover(name,input,static_cast<int64_t>(std::nearbyint(parameter(0,1))),rng));
    if(metadata.op==Operation::cross_arithmetic||metadata.op==Operation::cross_sim_binary)return auxiliary_result(continuous_crossover(name,input,para,rng,*math));
    py::array_t<double,py::array::c_style> bound=py::reinterpret_borrow<py::array_t<double,py::array::c_style>>(py::none());
    py::array_t<int64_t,py::array::c_style> bit_bounds=py::reinterpret_borrow<py::array_t<int64_t,py::array::c_style>>(py::none());
    if(metadata.bit_normal||metadata.bit_fastga) {
      auto raw=algorithm?problem:flex_field(problem,"bound");
      if(PyArray_CheckExact(raw.ptr())) {
        auto array=py::reinterpret_borrow<py::array>(raw);
        if(array.ndim()==2&&array.shape(0)==2&&array.shape(1)==d&&array.dtype().is(py::dtype::of<int64_t>())&&(array.flags()&py::array::c_style))bit_bounds=py::reinterpret_borrow<py::array_t<int64_t>>(array);
      }
      if(bit_bounds.is_none()) {
        // The original conversion also defines malformed/fractional-bound errors.
        auto b=py::reinterpret_borrow<py::array_t<double,py::array::c_style|py::array::forcecast>>(py::handle());
        if(!algorithm&&raw.is_none())b=py::array_t<double,py::array::c_style|py::array::forcecast>({py::ssize_t(2),d});
        else b=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(raw);
        if(!algorithm&&raw.is_none())std::fill_n(b.mutable_data(),2*d,0.);
        if(!b||b.ndim()!=2||b.shape(0)!=2||b.shape(1)!=d)throw py::value_error("Expected (2, dimension) bounds");
        for(py::ssize_t j=0;j<d;++j)if(b.data()[j]!=0||b.data()[d+j]!=1)throw py::value_error(name+" requires binary [0, 1] problem bounds.");
        bit_bounds=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(b);
      }else for(py::ssize_t j=0;j<d;++j)if(bit_bounds.data()[j]!=0||bit_bounds.data()[d+j]!=1)throw py::value_error(name+" requires binary [0, 1] problem bounds.");
    }else bound=algorithm?py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(problem):py::array_t<double,py::array::c_style|py::array::forcecast>(bounds(problem,d,discrete||metadata.op==Operation::search_cma||metadata.op==Operation::search_ica));
    if(metadata.op==Operation::search_cma)return cma_sample(input_dense(input),has_context?context_matrix():input_dense(input),bound,state,has_context,algorithm,rng,*math);
    if(metadata.op==Operation::search_ica) {
      if(flex_field(problem,"bound").is_none()&&input.shape(0)) {
        bool same=true;
        auto dense_input=input_dense(input);auto values=static_cast<const double *>(dense_input.data());
        for(py::ssize_t j=0;j<d;++j) {
          double lo=values[j],hi=lo;
          for(py::ssize_t i=1;i<input.shape(0);++i) {lo=std::min(lo,values[i*d+j]);hi=std::max(hi,values[i*d+j]);}
          bound.mutable_data()[j]=lo;bound.mutable_data()[d+j]=hi;
          same&=std::abs(lo-hi)<=1e-8+1e-5*std::abs(hi);
        }
        if(same)for(py::ssize_t j=0;j<d;++j){bound.mutable_data()[j]-=1;bound.mutable_data()[d+j]+=1;}
      }
      auto f=PyArray_CheckExact(source.ptr())?zeros_vector(input.shape(0)):source_fitness();
      return imperialist(input_dense(input),f,bound,para,state,rng,call.population&&!call.population->external);
    }
    if(metadata.de) {
      auto model=has_context?context:source;
      bool needs_fitness=name.find("pbest")!=std::string::npos||metadata.op==Operation::search_de_current_best;
      auto unused=py::reinterpret_borrow<py::array_t<double>>(py::handle());
      std::optional<DecisionRows> population_rows;
      if(native_context)population_rows.emplace(native_context,call.indexed_inputs);else if(has_context)population_rows.emplace(model);
      auto model_fitness=unused,selected_fitness=unused;
      if(needs_fitness) {
        auto selected=source_values();
        if(has_context) {
          auto context_values=native_context?native_context->fitness_values():fitness_values(model);
          std::tie(model_fitness,selected_fitness)=shared_selection_scores(context_values,selected);
        } else model_fitness=selected_fitness=selected.selection_scores();
      }
      auto run=[&](const auto &rows) {
        if(!needs_fitness) {
          // Match the consumer's exact wrapper type: absent data stays absent,
          // without constructing an ndarray or invoking a typed conversion.
          auto absent=py::reinterpret_borrow<py::array_t<double,py::array::c_style>>(py::handle());
          return differential_rows(name,input,rows,absent,absent,para,state,has_context,rng,call.population&&!call.population->external);
        }
        // Public F can change dtype/strides. Retain the original argument
        // conversions and their order whenever the variant consumes fitness.
        return differential_rows(name,input,rows,model_fitness,selected_fitness,para,state,has_context,rng,call.population&&!call.population->external);
      };
      auto result=has_context?run(*population_rows):run(InputRowsView<std::decay_t<decltype(input)>>{input});
      if(!call.has_state&&metadata.op!=Operation::search_fastga_de_current_to_pbest)return auxiliary_result(result.value);
      return result;
    }
    if(metadata.bit_normal||metadata.bit_fastga) {
      auto result=bit_variation(name,input,bit_bounds,state,rng,*math,batch,true);
      if(!metadata.bit_fastga) {
        auto output=auxiliary_result(result.value);output.binary_output=result.binary_output;return output;
      }
      return result;
    }
    if(discrete) {
      auto ib=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(bound);
      return auxiliary_result(discrete_variation(name,input,ib,para,rng));
    }
    if(algorithm&&metadata.op==Operation::search_mu_polynomial)para={1.,30.};
    int64_t inner=call.inner();
    bool private_noise=(metadata.op==Operation::search_mu_gaussian||metadata.op==Operation::search_mu_cauchy)&&call.population&&call.population->private_rows();
    auto sample=[&](const auto &population){return continuous_variation(name,input,population,bound,para,state,inner,rng,*math,private_noise);};
    auto result=metadata.op==Operation::search_eda?sample(has_context?context_matrix():input_dense(input)):sample(input);
    if(algorithm)return state_result(result.value,py::none());
    // Stateless components preserve the exact auxiliary argument, including None.
    if(!call.has_state||(metadata.op!=Operation::search_mu_gaussian&&metadata.op!=Operation::search_mu_cauchy))return auxiliary_result(result.value);
    return result;
    };
    if(call.indexed_inputs&&call.population&&call.population->rows_compatible()&&call.population->integer()==discrete) {
      auto rows=BatchStorage::read_rows(call.population);
      return discrete?execute_numerical(rows.typed<int64_t>()):execute_numerical(rows.typed<double>());
    }
    return execute_numerical(source_matrix(discrete));
  }
};
}

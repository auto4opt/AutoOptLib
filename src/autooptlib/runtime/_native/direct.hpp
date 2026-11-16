// Fixed component paths. No computation plan, frame, register file or effect log.
#pragma once
#include "blocks.hpp"
#include "dependencies.hpp"
#include <memory>
#include <optional>
#include <unordered_map>

struct NativeBindingScope {
  py::object context, token=py::none();
  NativeBindingScope(py::object variable, bool active) : context(variable) {
    if(active) {
      auto parent=context.attr("get")();
      if(!parent.is_none()) {
        parent.attr("unknown")("direct-native-entry");
        token=context.attr("set")(py::none());
      }
    }
  }
  ~NativeBindingScope() { if(!token.is_none())context.attr("reset")(token); }
};

struct DirectBinding {
  py::dict names;
  py::object name, function, code;
  explicit DirectBinding(py::handle binding)
      : names(binding.attr("namespace")), name(binding.attr("name")),
        function(binding.attr("base")), code(binding.attr("code")) {}
  py::object read() const {
    auto value=PyDict_GetItem(names.ptr(),name.ptr());
    if(value)return py::reinterpret_borrow<py::object>(value);
    auto builtins=PyDict_GetItemString(names.ptr(),"__builtins__");
    if(builtins && PyModule_Check(builtins))builtins=PyModule_GetDict(builtins);
    value=builtins ? PyDict_GetItem(builtins,name.ptr()) : nullptr;
    if(value)return py::reinterpret_borrow<py::object>(value);
    PyErr_Format(PyExc_NameError,"name '%U' is not defined",name.ptr());
    throw py::error_already_set();
  }
  bool original(py::handle current) const {
    return current.ptr()==function.ptr() && PyFunction_Check(current.ptr())
        && PyFunction_GET_CODE(current.ptr())==code.ptr();
  }
};
struct DirectComponent {
  py::object reference, code;
  std::vector<DirectBinding> bindings;
  std::vector<std::unique_ptr<cpu_program::DependencyProof>> proofs;
  py::object clustering=py::none(), clustering_code=py::none();
  DirectComponent(py::object function, bool niche) : reference(function) {
    auto specification=function.attr("__execution_computation__");
    code=specification.attr("reference_code");
    for(auto binding:specification.attr("bindings"))bindings.emplace_back(binding);
    auto guards=specification.attr("native_dependencies");
    if(niche) {
      for(int i=0;i<3;++i)proofs.emplace_back(new cpu_program::DependencyProof(guards[py::int_(i)]));
      clustering=guards[py::int_(3)][py::int_(0)];
      clustering_code=guards[py::int_(3)][py::int_(1)];
    } else proofs.emplace_back(new cpu_program::DependencyProof(guards));
  }
};
struct DirectNative : std::enable_shared_from_this<DirectNative> {
  cpu_blocks::Kernels kernels;
  struct Elite { std::vector<double> fitness; std::vector<int> indices; };
  std::shared_ptr<const Elite> elite_record;
  uint64_t elite_requests=0, elite_hits=0;


  bool preparations=false, native=false, recording=true;
  int64_t max_bytes=32*1024*1024;
  uint64_t splices=0, prefixes=0, finishes=0, distance_requests=0, distance_hits=0;
  std::unordered_map<std::string,std::shared_ptr<DirectComponent>> components;
  struct Distance {
    py::array x, fitness, order, values;
    py::object environment;
    int64_t bytes;
  };
  std::shared_ptr<const Distance> distance;
  py::object numpy=py::module_::import("numpy");
  py::object float64=numpy.attr("float64");
  py::object geterr=numpy.attr("geterr");
  std::vector<py::object> blas_threads;
  void configure(bool reuse, bool blocks, int64_t budget) {
    preparations=reuse; native=blocks; max_bytes=budget; distance.reset();
    blas_threads.clear();
    if(reuse) {
      auto controller=py::module_::import("threadpoolctl").attr("ThreadpoolController")();
      for(auto library:controller.attr("lib_controllers"))
        if(py::cast<std::string>(library.attr("user_api"))=="blas")
          blas_threads.emplace_back(library.attr("get_num_threads"));
    }
  }
  py::object environment() {
    if(numpy.attr("geterr").ptr()!=geterr.ptr())return py::none();
    auto hardware=cpu_blocks::numeric_environment();
    if(hardware.is_none())return py::none();
    py::dict errors=geterr();
    if(py::cast<std::string>(errors["under"])!="ignore")return py::none();
    py::list threads;
    for(auto &read:blas_threads)threads.append(read());
    return py::make_tuple(hardware,errors,py::tuple(threads));
  }
  std::shared_ptr<DirectComponent> component(const std::string &name, py::object fn) {
    if(!(native||preparations))return nullptr;
    bool niche=name=="choose_nich";
    if(!niche && name!="cross_point_one" && name!="cross_point_two" && name!="cross_point_uniform")return nullptr;
    if(!niche && !native)return nullptr;
    auto &entry=components[name];
    if(!entry || entry->reference.ptr()!=fn.ptr())entry=std::make_shared<DirectComponent>(fn,niche);
    if(PyFunction_GET_CODE(fn.ptr())!=entry->code.ptr())return nullptr;
    return entry;
  }
  static bool identical(const py::array &a,const py::array &b) {
    if(a.ndim()!=b.ndim() || !a.dtype().is(b.dtype()) || a.nbytes()!=b.nbytes())return false;
    for(py::ssize_t i=0;i<a.ndim();++i)if(a.shape(i)!=b.shape(i))return false;
    return std::memcmp(a.data(),b.data(),a.nbytes())==0;
  }
  py::object distances(DirectComponent &entry, py::object current, py::object x, py::object fitness) {
    auto a=py::reinterpret_borrow<py::array>(py::handle()), f=py::reinterpret_borrow<py::array>(py::handle());
    bool admitted=preparations && entry.bindings[1].original(current)
        && kernels.doubles(x,a) && kernels.doubles(fitness,f,1);
    auto owner=distance; // callbacks/reentry may replace the latest record
    if(admitted && recording)++distance_requests;
    bool save=admitted && entry.proofs[0]->trusted();
    py::object policy=save ? environment() : py::object(py::none());
    save=save && !policy.is_none();
    if(save && owner && identical(a,owner->x) && identical(f,owner->fitness)
        && policy.equal(owner->environment)) {
      if(recording)++distance_hits;
      return py::make_tuple(owner->order.attr("copy")(),owner->values.attr("copy")());
    }
    py::object result=current(x,fitness);
    if(save) {
      py::tuple pair=py::cast<py::tuple>(result);
      auto order=py::reinterpret_borrow<py::array>(py::handle()), values=py::reinterpret_borrow<py::array>(py::handle());
      if(kernels.integers(pair[0],order) && kernels.doubles(pair[1],values)) {
        int64_t bytes=a.nbytes()+f.nbytes()+order.nbytes()+values.nbytes();
        if(bytes+(elite_record ? elite_record->fitness.size()*(sizeof(double)+sizeof(int)) : 0)<=static_cast<uint64_t>(max_bytes))
          distance=std::make_shared<Distance>(Distance{
            py::cast<py::array>(a.attr("copy")()),py::cast<py::array>(f.attr("copy")()),
            py::cast<py::array>(order.attr("copy")()),py::cast<py::array>(values.attr("copy")()),policy,bytes});
      }
    }
    return result;
  }
  static py::tuple distance_pair(py::object value) {
    // Match Python's two-value unpack, including iterator effects and errors.
    if(PyTuple_CheckExact(value.ptr()) && PyTuple_GET_SIZE(value.ptr())==2)
      return py::reinterpret_borrow<py::tuple>(value);
    auto raw=PyObject_GetIter(value.ptr());
    if(!raw) {
      if(PyErr_ExceptionMatches(PyExc_TypeError) && !Py_TYPE(value.ptr())->tp_iter && !PySequence_Check(value.ptr()))
        PyErr_Format(PyExc_TypeError,"cannot unpack non-iterable %.200s object",Py_TYPE(value.ptr())->tp_name);
      throw py::error_already_set();
    }
    auto iterator=py::reinterpret_steal<py::object>(raw);
    py::tuple result(2);
    for(int i=0;i<2;++i) {
      auto item=PyIter_Next(iterator.ptr());
      if(!item) {
        if(PyErr_Occurred())throw py::error_already_set();
        throw py::value_error("not enough values to unpack (expected 2, got "+std::to_string(i)+")");
      }
      result[i]=py::reinterpret_steal<py::object>(item);
    }
    auto extra=PyIter_Next(iterator.ptr());
    if(extra) {Py_DECREF(extra);throw py::value_error("too many values to unpack (expected 2)");}
    if(PyErr_Occurred())throw py::error_already_set();
    return result;
  }
  std::optional<py::object> execute(const std::string &name,py::object fn,const py::tuple &args) {
    auto entry=component(name,fn);
    if(!entry)return std::nullopt;
    auto &b=entry->bindings;
    // Unknown readers retain the complete reference call before any effects.
    auto reader=b[0].read();
    if(!b[0].original(reader))return std::nullopt;
    if(name=="choose_nich") {
      py::tuple inputs=py::cast<py::tuple>(reader(args));
      auto prepared=distance_pair(distances(*entry,b[1].read(),inputs[0],inputs[1]));
      auto species_fn=b[2].read();
      py::object species;
      if(native && b[2].original(species_fn))
        species=species_fn(prepared[1],inputs[2],inputs[3],py::cast(shared_from_this()));
      else species=species_fn(prepared[1],inputs[2],inputs[3]);
      auto indices=b[3].read()(species,prepared[0],inputs[4]);
      return b[4].read()(indices);
    }
    std::string kind=name.substr(std::string("cross_point_").size());
    py::tuple inputs=py::cast<py::tuple>(reader(args,kind));
    auto cuts=b[1].read()(inputs[0],inputs[4],inputs[3],inputs[6]);
    auto result=b[3].read(); // Python binds the outer callable before splice.
    auto splice=b[2].read();
    py::object offspring=py::none();
    if(b[2].original(splice) && entry->proofs[0]->trusted(false)) {
      auto parts=kernels.splice("splice_"+kind,py::make_tuple(inputs[4],inputs[5],cuts),true);
      if(!parts.is_none()) {
        py::list children;children.append(parts[py::int_(0)]);children.append(parts[py::int_(1)]);
        auto np=py::reinterpret_borrow<py::dict>(PyFunction_GetGlobals(splice.ptr()))["np"];
        offspring=np.attr("vstack")(children);
        if(recording)++splices;
      }
    }
    if(offspring.is_none())offspring=splice(inputs[4],inputs[5],cuts);
    return result(offspring,inputs[2],args);
  }
  bool clustering(py::object fn) {
    auto &entry=*components.at("choose_nich");
    return fn.ptr()==entry.clustering.ptr() && PyFunction_Check(fn.ptr())
        && PyFunction_GET_CODE(fn.ptr())==entry.clustering_code.ptr();
  }
  py::object prefix(py::object fn,py::object distances) {
    auto &entry=*components.at("choose_nich");
    if(fn.ptr()!=entry.proofs[1]->functions[0][py::int_(0)].ptr() || !entry.proofs[1]->trusted(false))return py::none();
    auto result=kernels.nich_nearest(py::make_tuple(distances));
    if(recording && !result.is_none())++prefixes;
    return result;
  }
  py::object finish(py::object fn,py::tuple inputs) {
    auto &entry=*components.at("choose_nich");
    if(fn.ptr()!=entry.proofs[2]->functions[0][py::int_(0)].ptr() || !entry.proofs[2]->trusted(false))return py::none();
    auto value=inputs[3];
    if(!PyFloat_CheckExact(value.ptr()) && reinterpret_cast<PyObject *>(Py_TYPE(value.ptr()))!=float64.ptr())return py::none();
    py::tuple normalized(inputs.size());
    for(size_t i=0;i<inputs.size();++i)normalized[i]=i==3 ? py::object(py::float_(PyFloat_AsDouble(value.ptr()))) : py::reinterpret_borrow<py::object>(inputs[i]);
    auto result=kernels.nich_finish(normalized,true);
    if(recording && !result.is_none())++finishes;
    return result;
  }
  py::dict stats() const {
    return py::dict("splices"_a=splices,"nich_prefixes"_a=prefixes,"nich_finishes"_a=finishes,
                    "distance_requests"_a=distance_requests,"distance_hits"_a=distance_hits);
  }
};

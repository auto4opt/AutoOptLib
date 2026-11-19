// Native decision storage, ordered clone matching, and built-in
// repair/evaluation. No RNG implementation, objective arithmetic, or evaluation
// policy is replaced.
#pragma once
#include "profile.hpp"
#include "constraint_order.hpp"
#include "execution_services.hpp"
#include "array_workspace.hpp"
#include "native_interval.hpp"
#include "simd.hpp"
#include "x86_kernels.hpp"
#include "decision_input.hpp"
#include "fragment.hpp"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <optional>
#include <functional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <vector>
namespace py = pybind11;
using namespace pybind11::literals;

inline bool ordinary_generator(const py::dict &state, py::handle generator_type) {
  if(!PyDict_CheckExact(state.ptr()))return false;
  auto rng=PyDict_GetItemString(state.ptr(),"rng");
  return rng && reinterpret_cast<PyObject *>(Py_TYPE(rng))==generator_type.ptr();
}

struct DecisionRow {
  py::object owner;
  bool integer;
  py::ssize_t dimension, stride;
  const char *data;
  static bool read(py::handle obj, DecisionRow &row) {
    if (!py::isinstance<py::array>(obj))
      return false;
    auto array = py::reinterpret_borrow<py::array>(obj);
    row.owner = array;
    if (array.ndim() != 1)
      return false;
    auto dtype = array.dtype();
    // Read the NumPy descriptor directly; constructing dtype objects per row
    // would reintroduce Python work in every population traversal.
    if (dtype.itemsize() != 8 || dtype.byteorder() != '=')
      return false;
    if (dtype.kind() == 'i')
      row.integer = true;
    else if (dtype.kind() == 'f')
      row.integer = false;
    else
      return false;
    row.dimension = array.shape(0);
    row.stride = array.strides(0);
    row.data = static_cast<const char *>(array.data());
    return true;
  }
  int64_t integer_at(py::ssize_t j) const {
    int64_t v;
    std::memcpy(&v, data + j * stride, sizeof(v));
    return v;
  }
  double number_at(py::ssize_t j) const {
    if (integer)
      return static_cast<double>(integer_at(j));
    double v;
    std::memcpy(&v, data + j * stride, sizeof(v));
    return v;
  }
  bool equal(const DecisionRow &other, bool simd = false) const {
    if (dimension != other.dimension)
      return false;
    if (simd && integer == other.integer && stride == 8 && other.stride == 8)
      return exact_simd::equal(data, other.data, dimension, integer);
    // Numeric equality, not memcmp: signed zeros match; NaNs do not.
    for (py::ssize_t j = 0; j < dimension; ++j) {
      if (integer && other.integer) {
        if (integer_at(j) != other.integer_at(j))
          return false;
      } else if (number_at(j) != other.number_at(j))
        return false;
    }
    return true;
  }
};

struct PopulationView {
  // Retain exact source objects and array owners; no copied/stale population
  // cache.
  py::object sources=py::none();
  bool load_rows(const native_components::DecisionInput &input) {
    rows.clear();rows.reserve(input.n);
    for(py::ssize_t i=0;i<input.n;++i) {
      DecisionRow row;row.integer=input.integer;row.dimension=input.d;row.stride=8;
      row.data=static_cast<const char *>(input.row(i));row.owner=py::none();rows.push_back(std::move(row));
    }
    sources=py::list();return true;
  }
  std::vector<DecisionRow> rows;
  bool load_matrix(py::array matrix,bool keep_sources=true) {
    if(keep_sources)sources=py::list();
    if(matrix.ndim()!=2 || !(matrix.flags() & py::array::c_style))return false;
    bool integer=matrix.dtype().is(py::dtype::of<int64_t>());
    if(!integer&&!matrix.dtype().is(py::dtype::of<double>()))return false;
    rows.reserve(matrix.shape(0));
    for(py::ssize_t i=0;i<matrix.shape(0);++i) {
      DecisionRow row;row.owner=matrix;row.integer=integer;
      row.dimension=matrix.shape(1);row.stride=8;
      row.data=static_cast<const char *>(matrix.data())+i*matrix.strides(0);
      rows.push_back(std::move(row));if(keep_sources)py::cast<py::list>(sources).append(py::int_(i));
    }
    return true;
  }
  bool load(py::handle source, py::object list_type) {
    AOL_REGION_TIME(7);
    // Dense search inputs carry position IDs rather than Python Solution records.
    if (PyDict_CheckExact(source.ptr())) {
      auto dictionary=py::reinterpret_borrow<py::dict>(source);
      if (dictionary.contains("_batch_X")) {
        sources=py::list();
        auto matrix=py::cast<py::array>(dictionary["_batch_X"]);
        if(matrix.ndim()!=2 || !(matrix.flags() & py::array::c_style))return false;
        bool integer=matrix.dtype().is(py::dtype::of<int64_t>());
        if(!integer&&!matrix.dtype().is(py::dtype::of<double>()))return false;
        for(py::ssize_t i=0;i<matrix.shape(0);++i) {
          DecisionRow row;row.owner=matrix;row.integer=integer;
          row.dimension=matrix.shape(1);row.stride=8;
          row.data=static_cast<const char *>(matrix.data())+i*matrix.strides(0);
          rows.push_back(std::move(row));py::cast<py::list>(sources).append(py::int_(i));
        }
        return true;
      }
    }
    if (py::isinstance<py::list>(source))
      sources = py::reinterpret_borrow<py::list>(source);
    else
      sources = list_type(source);
    rows.reserve(py::len(sources));
    for (auto item : py::cast<py::list>(sources)) {
      DecisionRow row;
      if (!DecisionRow::read(item.attr("dec"), row))
        return false;
      rows.push_back(std::move(row));
    }
    return true;
  }
  template<bool Measure = false>
  int64_t match(const DecisionRow &query, bool simd = false,
                   uint64_t *comparisons = nullptr) const {
    for (size_t i = 0; i < rows.size(); ++i) {
      if constexpr (Measure) ++*comparisons;
      if (rows[i].equal(query, simd && query.dimension >= 32))
        return i;
    }
    return -1;
  }
};

struct ManagedScratch {
  py::object lease = py::none();
  ~ManagedScratch() {
    if (!lease.is_none()) {
      // The lease owns no scientific state; cleanup must preserve a reference
      // exception already propagating through the evaluator boundary.
      PyObject *type, *value, *traceback;
      PyErr_Fetch(&type, &value, &traceback);
      PyObject *result = PyObject_CallMethod(lease.ptr(), "release", nullptr);
      if (result)
        Py_DECREF(result);
      else
        PyErr_WriteUnraisable(lease.ptr());
      PyErr_Restore(type, value, traceback);
    }
  }
};

struct NativeData {
  // One stage owns the mask and any observable ndarray until its commit ends.
  // Values stay native until a Python boundary needs the original array/view.
  struct Mask {
    const std::vector<bool> &values;
    py::ssize_t view_size=-1;
    py::object published=py::none();
    explicit Mask(const std::vector<bool> &v):values(v){}
    py::object materialize() {
      if(published.is_none()) {
        py::array_t<bool> owner(values.size());
        for(size_t i=0;i<values.size();++i)owner.mutable_data()[i]=values[i];
        published=view_size<0?py::object(owner):py::object(py::array_t<bool>({view_size},{py::ssize_t(1)},owner.data(),owner));
      }
      return published;
    }
  };
  struct EvaluatedRow {
    py::array decision;
    double objective,constraint;
    py::object auxiliary;
  };
  struct EvaluationResult {
    std::vector<int64_t> sources;
    std::vector<EvaluatedRow> evaluated;
    int64_t used=0;
    bool interrupted=false,available=false;
  };
  using Rows = std::vector<py::array>;
  using Results = std::vector<py::tuple>;
  using RowEvaluator = void (NativeData::*)(Rows &, EvaluationResult *, Results &, py::object, py::object);
  // Bound once per run. The ordinary path adds no policy checks or row gates.
  RowEvaluator evaluate_rows = &NativeData::evaluate_reference_rows;
  py::object cpu_rows = py::none();
  void evaluate_reference_rows(Rows &evaluated_decisions, EvaluationResult *native_result,
                               Results &results, py::object problem, py::object data) {
    auto evaluator = solve.attr("_call_evaluator");
    size_t completed_requests=0;
    for (auto &decision : evaluated_decisions) {
      AOL_REGION_TIME(4);
      auto native=native_result&&evaluator_fast_path?evaluator_fast_path(problem,data,decision):std::nullopt;
      if(native){native_result->evaluated.push_back(std::move(*native));if(recording)++native_evaluator_requests;}
      else {
        native_components::RegionScope callback_timing(native_components::Region::Callback);
        auto result=py::cast<py::tuple>(evaluator(problem, data, decision));
        if(native_result)native_result->evaluated.push_back({decision,py::cast<double>(result[0]),
            py::cast<double>(result[1]),py::reinterpret_borrow<py::object>(result[2])});
        else results.push_back(std::move(result));
        if(recording)++callback_evaluator_requests;
      }
      if (recording) ++objective_calls;
      if(native_result&&++completed_requests%32==0)native_components::cooperative_boundary();
    }
  }
  void evaluate_cpu_rows(Rows &decisions, EvaluationResult *native_result,
                         Results &results, py::object, py::object) {
    if(decisions.empty())return;
    AOL_REGION_TIME(4);
    py::list inputs;for(auto &row:decisions)inputs.append(row);
    auto outputs=py::cast<py::list>(cpu_rows(inputs));
    for(size_t i=0;i<decisions.size();++i) {
      auto result=py::cast<py::tuple>(outputs[i]);
      if(native_result)native_result->evaluated.push_back({decisions[i],py::cast<double>(result[0]),
          py::cast<double>(result[1]),py::reinterpret_borrow<py::object>(result[2])});
      else results.push_back(std::move(result));
    }
    if(recording){objective_calls+=decisions.size();callback_evaluator_requests+=decisions.size();}
    native_components::cooperative_boundary();
  }
  using ChangeEntry = py::object (NativeData::*)(py::object,py::object,py::object,py::object,int64_t,py::object,py::object,
      bool,bool,bool,bool,bool,EvaluationResult*,py::array*,
      const native_components::DecisionInput*,const native_components::DecisionInput*,
      const std::function<const native_components::DecisionInput&()>*,py::handle,
      const native_components::PreparedFragment*,Mask*);
  ChangeEntry change_plain=&NativeData::changed_impl<false,false>;
  ChangeEntry change_native=&NativeData::changed_impl<true,false>;
  int64_t cpu_constraints=0;
  using NativeCPU = void (*)(int64_t,int64_t,const void*,double*,double*,const void*);
  NativeCPU cpu_native=nullptr;
  const void *cpu_context=nullptr;
  void bind_cpu(py::object callback,bool arrays=false,int64_t constraints=0,uintptr_t address=0,uintptr_t context=0) {
    cpu_rows=std::move(callback);cpu_constraints=constraints;
    cpu_native=reinterpret_cast<NativeCPU>(address);cpu_context=reinterpret_cast<const void*>(context);
    change_plain=arrays?&NativeData::changed_impl<false,true>:&NativeData::changed_impl<false,false>;
    change_native=arrays?&NativeData::changed_impl<true,true>:&NativeData::changed_impl<true,false>;
    if(cpu_rows.is_none())evaluate_rows=&NativeData::evaluate_reference_rows;
    else if(!arrays)evaluate_rows=&NativeData::evaluate_cpu_rows;
    else if(address)evaluate_rows=constraints?&NativeData::evaluate_cpu_arrays<true,true>:&NativeData::evaluate_cpu_arrays<false,true>;
    else evaluate_rows=constraints?&NativeData::evaluate_cpu_arrays<true,false>:&NativeData::evaluate_cpu_arrays<false,false>;
  }
  template<bool Constrained,bool Native>void evaluate_cpu_arrays(Rows &decisions,EvaluationResult *native_result,
                         Results &results,py::object,py::object) {
    if(decisions.empty())return;
    AOL_REGION_TIME(4);
    const auto n=py::ssize_t(decisions.size()),d=decisions[0].shape(0);
    py::array x(decisions[0].dtype(),std::vector<py::ssize_t>{n,d},
      std::vector<py::ssize_t>{d*8,8},decisions[0].data(),decisions[0]);
    auto y=native_components::workspace_array<double>(n);
    auto c=native_components::workspace_shaped<double>({n,cpu_constraints});
    if constexpr(Native){cpu_native(n,d,x.data(),y.mutable_data(),c.mutable_data(),cpu_context);if(PyErr_Occurred())throw py::error_already_set();}
    else cpu_rows(x,y,c);
    const auto *yp=y.data(),*cp=c.data();
    // The scalar contract validates original raw constraints before max/sum.
    // These are necessary value checks, without dtype/policy dispatch per row.
    for(py::ssize_t i=0;i<n;++i) {
      bool finite=std::isfinite(yp[i]);
      if constexpr(Constrained)for(py::ssize_t j=0;j<cpu_constraints;++j)finite&=std::isfinite(cp[i*cpu_constraints+j]);
      if(!finite) {
        py::array row(py::dtype::of<double>(),std::vector<py::ssize_t>{cpu_constraints},
          std::vector<py::ssize_t>{8},cp+i*cpu_constraints,c);
        solve.attr("_normalize_evaluator_output")(py::make_tuple(yp[i],row,py::none()),py::list());
        throw std::runtime_error("CPU objective output validation did not raise");
      }
    }
    py::array_t<double> sums=py::reinterpret_borrow<py::array_t<double>>(py::handle());
    if constexpr(Constrained) {
      sums=native_components::workspace_array<double>(n);
      np.attr("maximum")(0.,c,"out"_a=c);
      np.attr("sum")(c,"axis"_a=1,"out"_a=sums);
    }
    for(py::ssize_t i=0;i<n;++i) {
      const double con=Constrained?sums.data()[i]:0.;
      if(native_result)native_result->evaluated.push_back({decisions[i],yp[i],con,py::none()});
      else results.push_back(py::make_tuple(yp[i],con,py::none()));
    }
    if(recording){objective_calls+=n;callback_evaluator_requests+=n;}
    native_components::cooperative_boundary();
  }
  std::function<std::optional<EvaluatedRow>(py::object,py::object,py::array)> evaluator_fast_path;
  uint64_t native_evaluator_requests=0,callback_evaluator_requests=0;
  py::object solve, np, list_type, solution_type, solution_set;
  bool native_integer_width;
  std::function<py::object(py::object,const char *,py::object)> native_flex;
  py::object read_flex(py::object object,const char *name,py::object fallback) {
    return native_flex?native_flex(object,name,fallback):solve.attr("get_flex")(object,name,fallback);
  }
  bool recording = true;
  py::object array_type;
  bool simd = false;
  bool cost_selection = false;
  uint64_t cost_scalar = 0, cost_region = 0;
  uint64_t calls = 0, rows_repaired = 0, clone_comparisons = 0, clone_hits = 0;
  uint64_t objective_calls = 0, fallbacks = 0, mutation_rows = 0;
  uint64_t region_calls = 0, reused_output_rows = 0;
  uint64_t workspace_growths = 0, materialized_bytes = 0;
  uint64_t converted_input_bytes = 0, integer_snapshot_bytes = 0;
  uint64_t identity_repair_rows = 0;
  uint64_t target_input_allocations = 0;
  uint64_t fragment_rows=0,fragment_grouped_rows=0,fragment_packed_rows=0,fragment_fallbacks=0,fragment_row_simd_rows=0;
  // Only internal repaired values live here. All callback inputs are owning
  // copies materialized before the first callback (including reentrant ones).
  native_components::ArrayWorkspace repair_workspace;
  py::object workspace_provider = py::none();
  py::object generator_type;
  py::object namespace_type,asarray_owner,asarray_key=py::str("asarray");
  explicit NativeData(py::object module) : solve(module) {
    np = py::module_::import("numpy");
    array_type = np.attr("ndarray");asarray_owner=np.attr("asarray");
    generator_type=np.attr("random").attr("Generator");
    namespace_type=py::module_::import("types").attr("SimpleNamespace");
    list_type = py::module_::import("builtins").attr("list");
    solution_type = solve.attr("Solution");
    solution_set = solve.attr("SolutionSet");
    native_integer_width =
        py::cast<int>(
            np.attr("dtype")(py::module_::import("builtins").attr("int"))
                .attr("itemsize")) == 8;
  }
  struct Repair {
    py::array_t<double, py::array::c_style | py::array::forcecast> bounds=py::reinterpret_borrow<py::array_t<double,py::array::c_style|py::array::forcecast>>(py::handle());
    bool discrete;
    py::ssize_t dimension;
  };
  bool repair_contract(py::object problem, Repair &r) {
    if (PyCallable_Check(py::getattr(problem, "repair", py::none()).ptr()))
      return false;
    if (PyCallable_Check(py::getattr(problem, "name", py::none()).ptr()) &&
        !PyCallable_Check(py::getattr(problem, "evaluate", py::none()).ptr()))
      return false;
    auto raw_type = py::getattr(problem, "type", py::none());
    if (!py::isinstance<py::list>(raw_type) &&
        !py::isinstance<py::tuple>(raw_type))
      return false;
    if (py::len(raw_type) == 0)
      return false;
    std::string type = py::str(raw_type[py::int_(0)]);
    r.discrete = type == "discrete";
    if (!r.discrete && type != "continuous")
      return false;
    if (r.discrete) {
      auto setting = py::hasattr(problem, "setting")
                         ? problem.attr("setting")
                         : read_flex(problem, "setting", py::str(""));
      if (py::isinstance<py::str>(setting) &&
          py::cast<std::string>(setting).find("dec_diff") != std::string::npos)
        return false;
    } else {
      auto strategy =
          read_flex(problem, "_AlgorithmBoundaryHandling", py::str("clip"));
      if (py::cast<std::string>(py::str(strategy).attr("lower")()) != "clip")
        return false;
    }
    auto bound = py::getattr(problem, "bound", py::none());
    // NumPy computes continuous midpoints in the original bounds dtype.
    if (!r.discrete && (!py::isinstance<py::array>(bound) ||
                        !py::reinterpret_borrow<py::array>(bound).dtype().is(
                            py::dtype::of<double>())))
      return false;
    if (!py::isinstance<py::array>(bound))
      return false;
    r.bounds =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            bound);
    if (!r.bounds || r.bounds.ndim() != 2 || r.bounds.shape(0) != 2)
      return false;
    r.dimension = r.bounds.shape(1);
    auto b = r.bounds.unchecked<2>();
    for (py::ssize_t j = 0; j < r.dimension; ++j) {
      if (!std::isfinite(b(0, j)) || !std::isfinite(b(1, j)) ||
          b(0, j) > b(1, j))
        return false;
      // Signed-zero ties depend on NumPy's SIMD min/max implementation.
      // Keep the reference path for zero bounds and overflowing midpoints.
      if (!r.discrete && (b(0, j) == 0. || b(1, j) == 0. ||
                          !std::isfinite(b(1, j) - b(0, j)) ||
                          std::fpclassify(b(1, j) - b(0, j)) == FP_SUBNORMAL))
        return false;
      if (r.discrete && (std::abs(b(0, j)) > 9007199254740991. ||
                         std::abs(b(1, j)) > 9007199254740991.))
        return false;
    }
    return true;
  }
  static double round_even(double value) {
    return native_components::repair_round_even(value);
  }
  template<bool NativeRows=false>
  py::object changed(py::object decisions, py::object sources,
                     py::object problem, py::object data, int64_t remaining,
                     py::object invalidated, py::object fallback,
                     bool planned = false, bool reuse_output = false,
                     bool binary_output = false,
                     bool workspace_allowed = true, bool aligned_sources = false,
                     EvaluationResult *native_result = nullptr,py::array *source_matrix=nullptr,
                     const native_components::DecisionInput *decision_input=nullptr,
                     const native_components::DecisionInput *source_input=nullptr,
                     const std::function<const native_components::DecisionInput&()> *source_loader=nullptr,
                     py::handle binary_output_owner=py::handle(),
                     const native_components::PreparedFragment *fragment=nullptr,
                     Mask *native_mask=nullptr) {
    if constexpr(NativeRows)return (this->*change_native)(std::move(decisions),std::move(sources),std::move(problem),std::move(data),remaining,std::move(invalidated),std::move(fallback),planned,reuse_output,binary_output,workspace_allowed,aligned_sources,native_result,source_matrix,decision_input,source_input,source_loader,binary_output_owner,fragment,native_mask);
    else return (this->*change_plain)(std::move(decisions),std::move(sources),std::move(problem),std::move(data),remaining,std::move(invalidated),std::move(fallback),planned,reuse_output,binary_output,workspace_allowed,aligned_sources,native_result,source_matrix,decision_input,source_input,source_loader,binary_output_owner,fragment,native_mask);
  }
  template<bool NativeRows=false,bool Bulk=false>
  py::object changed_impl(py::object decisions, py::object sources,
                     py::object problem, py::object data, int64_t remaining,
                     py::object invalidated, py::object fallback,
                     bool planned = false, bool reuse_output = false,
                     bool binary_output = false,
                     bool workspace_allowed = true, bool aligned_sources = false,
                     EvaluationResult *native_result = nullptr,py::array *source_matrix=nullptr,
                     const native_components::DecisionInput *decision_input=nullptr,
                     const native_components::DecisionInput *source_input=nullptr,
                     const std::function<const native_components::DecisionInput&()> *source_loader=nullptr,
                     py::handle binary_output_owner=py::handle(),
                     const native_components::PreparedFragment *fragment=nullptr,
                     Mask *native_mask=nullptr) {
    AOL_REGION_TIME(3);
    native_components::RegionScope repair_timing(native_components::Region::Repair);
    if (recording) ++calls;
    const bool matrix_mode=source_matrix||source_input||source_loader||(PyDict_CheckExact(sources.ptr()) && py::reinterpret_borrow<py::dict>(sources).contains("_batch_X"));
    // Native stage masks never escape. Materialize the original owning array
    // (and final-stage view) only when a Python consumer can observe it.
    auto materialize_mask=[&]() {
      if(!native_mask||!invalidated.is_none())return;
      invalidated=native_mask->materialize();
    };
    auto fail = [&]() -> py::object {
      materialize_mask();
      native_components::RegionScope callback_timing(native_components::Region::Callback);
      if (recording) ++fallbacks;
      if(fragment){decisions=fragment->materialize();if(recording)++fragment_fallbacks;}
      if(decision_input)decisions=decision_input->materialize();
      if(source_input){py::dict descriptor;descriptor["_batch_X"]=source_input->materialize();sources=descriptor;}
      if(source_loader){py::dict descriptor;descriptor["_batch_X"]=(*source_loader)().materialize();sources=descriptor;}
      if(source_matrix){py::dict descriptor;descriptor["_batch_X"]=*source_matrix;sources=descriptor;}
      if(aligned_sources)return fallback(decisions,sources,problem,data,remaining,invalidated,"aligned_sources"_a=true);
      return fallback(decisions, sources, problem, data, remaining,
                      invalidated);
    };
    Repair repair;
    if (!native_integer_width || !repair_contract(problem, repair))
      return fail();
    if (py::isinstance<py::dict>(
            py::getattr(problem, "_autoopt_eval_cache", py::none())))
      return fail();
    if(fragment&&!fragment->supports(repair.discrete,repair.bounds.data(),repair.bounds.data()+repair.dimension,repair.dimension))return fail();
    if (!fragment&&!decision_input&&!py::isinstance<py::array>(decisions))
      return fail();
    // Typed row/fragment inputs carry their own dtype and shape; only public
    // ndarray inputs need the array validation/conversion below.
    auto raw=(!fragment&&!decision_input)?py::reinterpret_borrow<py::array>(decisions):py::reinterpret_borrow<py::array>(py::none());
    if(!raw.is_none()) {
      if(raw.ndim()!=1&&raw.ndim()!=2)return fail();
      if(raw.dtype().kind()!='f'&&raw.dtype().kind()!='i'&&raw.dtype().kind()!='u')return fail();
    }
    bool snapshot_integer = !fragment && !decision_input && !planned && repair.discrete &&
                            reinterpret_cast<PyObject *>(Py_TYPE(raw.ptr())) == array_type.ptr() &&
                            (raw.flags() & py::array::c_style) &&
                            raw.dtype().is(py::dtype::of<int64_t>());
    bool direct_integer = (fragment&&fragment->integer) || (decision_input&&decision_input->integer) || snapshot_integer || (!fragment&&!decision_input&&planned &&
                          (raw.flags() & py::array::c_style) &&
                          raw.dtype().is(py::dtype::of<int64_t>()));
    py::object values;
    const double *float_values = nullptr;
    native_components::ScratchVector<int64_t> snapshot_lease;auto &integer_snapshot=snapshot_lease.values;
    py::ssize_t n = fragment?fragment->count:decision_input?decision_input->n:raw.ndim() == 1 ? 1 : raw.shape(0),
                d = fragment?fragment->d:decision_input?decision_input->d:raw.shape(raw.ndim() - 1);
    if (!fragment&&!decision_input&&!direct_integer) {
      auto converted = py::array_t<double, py::array::c_style |
                                            py::array::forcecast>::ensure(raw);
      if (recording && converted.data() != raw.data())
        converted_input_bytes += n * d * sizeof(double);
      float_values = converted.data();
      values = std::move(converted);
    }
    if (d != repair.dimension)
      return fail();
    if (snapshot_integer) {
      // B's float conversion used to snapshot integer input before mask/source
      // callbacks. Keep that timing without the float array or integer rint.
      integer_snapshot.reserve(n*d);
      for (py::ssize_t k = 0; k < n*d; ++k) {
        int64_t value;
        std::memcpy(&value, static_cast<const char *>(raw.data())+k*8, 8);
        if (value < -9007199254740991LL || value > 9007199254740991LL)
          return fail();
        integer_snapshot.push_back(value);
      }
      if (recording) integer_snapshot_bytes += n*d*sizeof(int64_t);
    }
    auto number = [&](py::ssize_t k) -> double {
      if constexpr(NativeRows) {
        if(decision_input){auto row=decision_input->row(k/d);return decision_input->integer?static_cast<double>(static_cast<const int64_t *>(row)[k%d]):static_cast<const double *>(row)[k%d];}
      }
      if (!direct_integer) return float_values[k];
      if (snapshot_integer) return static_cast<double>(integer_snapshot[k]);
      int64_t value;
      std::memcpy(&value, static_cast<const char *>(raw.data())+k*8, 8);
      return static_cast<double>(value);
    };
    // An actual native bit kernel has already proved its private output is
    // binary. Revalidate CURRENT bounds after RNG before eliminating repair.
    bool identity = planned && reuse_output && binary_output &&
                    direct_integer && repair.discrete;
    if (identity) {
      auto b = repair.bounds.unchecked<2>();
      for (py::ssize_t j = 0; j < d; ++j)
        identity &= b(0, j) == 0. && b(1, j) == 1.;
    }
    if (repair.discrete && !fragment && !identity && !snapshot_integer)
      for (py::ssize_t i = 0; i < n; ++i) {
        const void *row=nullptr;if constexpr(NativeRows)if(decision_input)row=decision_input->row(i);
        for(py::ssize_t j=0;j<d;++j) {
        double v=row?(decision_input->integer?static_cast<double>(static_cast<const int64_t *>(row)[j]):static_cast<const double *>(row)[j]):number(i*d+j);
        if (!std::isfinite(v) || std::abs(v) > 9007199254740991.)
          return fail();
        }
      }
    const bool direct_flags=native_mask&&PyCFunction_Check(asarray_owner.ptr())&&
      PyDict_GetItem(PyModule_GetDict(np.ptr()),asarray_key.ptr())==asarray_owner.ptr();
    native_components::ScratchVector<bool> force_lease;auto &flags_storage=force_lease.values;
    // Empty is the default false mask; one element is a frozen broadcast.
    // Non-native arrays must still be snapshotted before evaluator callbacks.
    auto force=[&](py::ssize_t i){return direct_flags?bool(native_mask->values[i]):flags_storage.empty()?false:bool(flags_storage[flags_storage.size()==1?0:i]);};
    bool need_sources=n>0;
    if(direct_flags){need_sources=false;for(py::ssize_t i=0;i<n;++i)need_sources|=!force(i);}
    else if(!invalidated.is_none()||native_mask) {
      materialize_mask();
      auto flags=py::array_t<bool,py::array::c_style|py::array::forcecast>::ensure(
        np.attr("asarray")(invalidated,"dtype"_a=py::module_::import("builtins").attr("bool")));
      if(flags.size()!=1&&flags.size()!=n)throw py::value_error("Stream invalidation mask must contain one value per offspring.");
      auto data=reinterpret_cast<const uint8_t *>(flags.data());need_sources=false;
      flags_storage.reserve(std::min<py::ssize_t>(n,flags.size()));
      if(n&&flags.size()==1){flags_storage.push_back(data[0]!=0);need_sources=!force(0);}
      else for(py::ssize_t i=0;i<n;++i){flags_storage.push_back(data[i]!=0);need_sources|=!force(i);}
    }
    PopulationView population;
    if (need_sources && !(source_loader?population.load_rows((*source_loader)()):source_input?population.load_rows(*source_input):source_matrix?population.load_matrix(*source_matrix,false):population.load(sources, list_type)))
      return fail();
    // The graph proves that a native operator's output dies at repair. Its
    // storage can become the repaired slot. Otherwise use internal typed
    // scratch; never borrow a caller-owned array for writing.
    bool in_place = !fragment && !decision_input && planned && reuse_output && raw.writeable() &&
                    (raw.flags() & py::array::c_style) &&
                    (repair.discrete ? raw.dtype().is(py::dtype::of<int64_t>())
                                     : raw.dtype().is(py::dtype::of<double>()));
    // The development calibration finds no stable workspace-only gain for
    // multi-row float input. Private-output reuse and integer conversion
    // elimination are distinct candidates; never weaken their safety guards.
    planned = planned && (workspace_allowed || in_place);
    if (planned && cost_selection) {
      planned = in_place || direct_integer || n == 1;
      if (recording) {
        if (planned) ++cost_region;
        else ++cost_scalar;
      }
    }
    auto bounds = repair.bounds.unchecked<2>();
    // A fact from the actual successful native bit kernel, tied to its owner.
    // Consume the original pre-callback integer snapshot, never a caller array.
    bool binary_identity=false;
    if(snapshot_integer&&repair.discrete&&binary_output_owner&&!binary_output_owner.is_none()&&
       reinterpret_cast<PyObject *>(Py_TYPE(binary_output_owner.ptr()))==array_type.ptr()) {
      auto owner=py::reinterpret_borrow<py::array>(binary_output_owner);
      binary_identity=owner.ndim()==2&&owner.dtype().is(py::dtype::of<int64_t>())&&
        owner.data()==raw.data()&&owner.shape(1)==d&&owner.shape(0)>=n&&
        (owner.flags()&py::array::c_style);
      for(py::ssize_t j=0;j<d&&binary_identity;++j)binary_identity=bounds(0,j)==0.&&bounds(1,j)==1.;
    }
    if(recording&&fragment){fragment_rows+=n;if(fragment->bits)fragment_packed_rows+=n;if(fragment->row_simd&&!fragment->integer)fragment_row_simd_rows+=n;}
    if(recording&&binary_identity)identity_repair_rows+=n;
    auto repair_row = [&](py::ssize_t i, char *destination) {
      if(!d)return;
      if(fragment){fragment->emit(i,destination,repair.bounds.data(),repair.bounds.data()+d);return;}
      if(binary_identity){if(d)std::memcpy(destination,integer_snapshot.data()+i*d,d*8);return;}
      const void *row=nullptr;if constexpr(NativeRows)if(decision_input)row=decision_input->row(i);
      if(!row)row=snapshot_integer?static_cast<const void *>(integer_snapshot.data()+i*d):
        direct_integer?static_cast<const void *>(static_cast<const char *>(raw.data())+i*d*8):static_cast<const void *>(float_values+i*d);
      auto run=[&](auto discrete_tag,auto integer_tag) {
        using Discrete=decltype(discrete_tag);using Integer=decltype(integer_tag);
        auto scalar=[&](size_t j) {
          double x;
          if constexpr(Integer::value){int64_t value;std::memcpy(&value,static_cast<const char *>(row)+j*8,8);x=static_cast<double>(value);}
          else std::memcpy(&x,static_cast<const char *>(row)+j*8,8);
          double lo=bounds(0,j),hi=bounds(1,j);
          x=native_components::repair_coordinate<Discrete::value,Integer::value>(x,bounds(0,j),bounds(1,j));
        if constexpr(Discrete::value)
          reinterpret_cast<int64_t *>(destination)[j] = static_cast<int64_t>(x);
        else
          reinterpret_cast<double *>(destination)[j] = x;
        };
        aol_x86::repair<Discrete::value,Integer::value>(row,destination,&bounds(0,0),&bounds(1,0),d,repair.bounds.strides(1)==8,scalar);
      };
      if(repair.discrete){if(direct_integer)run(std::true_type{},std::true_type{});else run(std::true_type{},std::false_type{});}
      else {if(direct_integer)run(std::false_type{},std::true_type{});else run(std::false_type{},std::false_type{});}
    };
    // A bulk kernel is pure and reads final admitted storage. Row headers are
    // retained by BatchRow; one shared payload owner survives with those rows.
    struct BulkTargets {
      py::array owner=py::reinterpret_borrow<py::array>(py::handle());
      py::ssize_t next=0,capacity,d;bool integer;
      using Entry=py::array (BulkTargets::*)();Entry entry=&BulkTargets::first;
      BulkTargets(py::ssize_t cap,py::ssize_t width,bool ints):capacity(cap),d(width),integer(ints){}
      py::array first() {
        owner=integer?py::array(native_components::workspace_shaped<int64_t>({capacity,d})):
                      py::array(native_components::workspace_shaped<double>({capacity,d}));
        entry=&BulkTargets::view;return view();
      }
      py::array view() {auto i=next++;return py::array(owner.dtype(),std::vector<py::ssize_t>{d},
        std::vector<py::ssize_t>{8},static_cast<const char*>(owner.data())+i*d*8,owner);}
    };
    BulkTargets bulk(std::min<int64_t>(n,std::max<int64_t>(0,remaining)),d,repair.discrete);
    auto allocate_target = [&]() -> py::array {
      if constexpr(Bulk)return (bulk.*bulk.entry)();
      else return repair.discrete ? py::array(py::array_t<int64_t>(d))
                                  : py::array(py::array_t<double>(d));
    };
    auto match = [&](const DecisionRow &query,py::ssize_t index) {
      if(aligned_sources) {
        if(recording && size_t(index)<population.rows.size())++clone_comparisons;
        bool equal=false;
        if(size_t(index)<population.rows.size()) {
          auto &source=population.rows[index];
          if(fragment&&fragment->bits&&source.integer&&source.stride==8&&source.dimension==d)
            equal=fragment->bits->equal(index,reinterpret_cast<const int64_t *>(source.data));
          else equal=source.equal(query,simd&&query.dimension>=32);
        }
        if(recording&&equal)++clone_hits;
        return equal?int64_t(index):int64_t(-1);
      }
      auto clone = recording ? population.match<true>(query, simd, &clone_comparisons)
                             : population.match(query, simd);
      if (recording && clone>=0) ++clone_hits;
      return clone;
    };
    py::object resolved=native_result?py::object(py::none()):py::object(py::list());
    auto resolve=[&](int64_t clone) {
      if(native_result)native_result->sources.push_back(clone);
      else py::cast<py::list>(resolved).append(clone<0?py::object(py::none()):matrix_mode?py::object(py::int_(clone)):py::reinterpret_borrow<py::object>(population.sources[py::int_(clone)]));
    };
    std::vector<py::array> evaluated_decisions;
    std::optional<ManagedScratch> managed;
    size_t changed_count = 0;
    const auto limit = static_cast<size_t>(std::max<int64_t>(0, remaining));
    // Only all-forced batches have a known evaluation count before matching.
    // Clone-heavy batches must not allocate for rows which need no evaluator.
    if(!need_sources)evaluated_decisions.reserve(std::min<size_t>(n,limit));
    if(native_result)native_result->sources.reserve(native_result->sources.size()+n);
    if(fragment&&!need_sources) {
      // Group only after all original RNG/state work and admission reads. Final
      // owning evaluator arrays are filled directly, before the first callback.
      // RNG and fragment admission have already finished. emit_many is pure;
      // rows past the FE budget have no consumer and need no dummy payload.
      const auto admitted=std::min<size_t>(n,limit);
      for(py::ssize_t first=0;size_t(first)<admitted;first+=4) {
        void *destinations[4];auto count=std::min<py::ssize_t>(4,admitted-first);
        for(py::ssize_t r=0;r<count;++r){
          auto row=allocate_target();destinations[r]=row.mutable_data();evaluated_decisions.push_back(std::move(row));
          resolve(-1);
        }
        fragment->emit_many(first,count,destinations,repair.bounds.data(),repair.bounds.data()+d);
        if(recording&&fragment->grouped&&!fragment->integer&&count==4)fragment_grouped_rows+=4;
      }
      for(size_t i=admitted;i<size_t(n);++i)resolve(-1);
      changed_count=n;
    } else if (planned) {
      char *storage;
      auto storage_owner=py::reinterpret_borrow<py::array>(py::handle());
      if (recording) ++region_calls;
      if (in_place) {
        storage = static_cast<char *>(raw.mutable_data());
        if (recording) reused_output_rows += n;
      } else if (!workspace_provider.is_none()) {
        managed.emplace();
        managed->lease = workspace_provider(repair.discrete, n, d);
        auto owned = py::cast<py::array>(managed->lease.attr("array"));
        auto dtype = owned.dtype();
        if (owned.ndim() != 2 || owned.shape(0) != n || owned.shape(1) != d ||
            !owned.writeable() || !(owned.flags() & py::array::c_style) ||
            dtype.itemsize() != 8 || !py::cast<bool>(dtype.attr("isnative")) ||
            dtype.kind() != (repair.discrete ? 'i' : 'f'))
          throw std::runtime_error("Invalid managed repair workspace binding");
        storage = static_cast<char *>(owned.mutable_data());
      } else {
        repair_workspace.recording=recording;auto before=repair_workspace.allocations;
        storage_owner=repair.discrete?py::array(repair_workspace.acquire<int64_t>(n*d)):py::array(repair_workspace.acquire<double>(n*d));
        if(recording)workspace_growths+=repair_workspace.allocations-before;
        storage=static_cast<char *>(storage_owner.mutable_data());
      }
      identity &= in_place;
      if (recording && identity) identity_repair_rows += n;
      if (!identity)
        for (py::ssize_t i=0; i<n; ++i) repair_row(i, storage+i*d*8);
      native_components::ScratchVector<int64_t> changed_lease;auto &changed_rows=changed_lease.values;
      for (py::ssize_t i=0; i<n; ++i) {
        int64_t clone = -1;
        if (!force(i)) {
          DecisionRow query;
          query.integer=repair.discrete; query.dimension=d; query.stride=8;
          query.data=storage+i*d*8;
          clone=match(query,i);
        }
        resolve(clone);
        if (clone<0) changed_rows.push_back(i);
      }
      changed_count=changed_rows.size();
      if constexpr(Bulk)bulk.capacity=std::min(changed_count,limit);
      for (size_t i=0; i<std::min(changed_count,limit); ++i) {
        auto row=allocate_target();
        if (d) std::memcpy(row.mutable_data(),storage+changed_rows[i]*d*8,d*8);
        evaluated_decisions.push_back(std::move(row));
        if (recording) materialized_bytes += d*8;
      }
    } else {
      // Only unresolved clone candidates need temporary repair storage. All
      // final owning callback inputs are ready before the first evaluator.
      auto scratch_owner=py::reinterpret_borrow<py::array>(py::handle());
      char *scratch = nullptr;
      for (py::ssize_t i=0; i<n; ++i) {
        const bool final_storage=force(i) && changed_count<limit;
        py::object target;
        char *destination;
        if (final_storage) {
          auto row=allocate_target();
          destination=static_cast<char *>(row.mutable_data());
          target=std::move(row);
        } else {
          if (!scratch) {
            scratch_owner=repair.discrete?py::array(native_components::workspace_array<int64_t>(d)):py::array(native_components::workspace_array<double>(d));
            scratch=static_cast<char *>(scratch_owner.mutable_data());
          }
          destination=scratch;
        }
        repair_row(i,destination);
        int64_t clone=-1;
        if (!force(i)) {
          DecisionRow query;
          query.integer=repair.discrete; query.dimension=d; query.stride=8;
          query.data=destination;
          clone=match(query,i);
        }
        resolve(clone);
        if (clone<0) {
          ++changed_count;
          if (changed_count<=limit) {
            if (!final_storage) {
              auto row=allocate_target();
              if (d) std::memcpy(row.mutable_data(),scratch,d*8);
              if (recording) materialized_bytes += d*8;
              target=std::move(row);
            }
            evaluated_decisions.push_back(py::reinterpret_borrow<py::array>(target));
          }
        }
      }
    }
    if (recording) {
      rows_repaired += n;
      target_input_allocations += evaluated_decisions.size();
    }
    const bool interrupted=changed_count>0 && static_cast<int64_t>(changed_count)>=remaining;
    const size_t count=evaluated_decisions.size();
    std::vector<py::tuple> results;
    if(native_result)native_result->evaluated.reserve(native_result->evaluated.size()+count);
    else results.reserve(count);
    (this->*evaluate_rows)(evaluated_decisions,native_result,results,problem,data);
    if(native_result) {
      native_result->used=interrupted?count:changed_count;
      native_result->interrupted=interrupted;native_result->available=true;
      return py::none();
    }
    if(matrix_mode) {
      py::list inputs, outputs;
      for(auto &decision:evaluated_decisions)inputs.append(decision);
      for(auto &result:results)outputs.append(result);
      return py::make_tuple(resolved, interrupted?count:changed_count, inputs, outputs, interrupted);
    }
    py::list evaluated;
    for (size_t i = 0; i < count; ++i) {
      auto result = results[i];
      double obj = py::cast<double>(result[0]),
             con = py::cast<double>(result[1]);
      double fit = native_components::fitness_value(obj, con);
      evaluated.append(
          solution_type(evaluated_decisions[i], obj, con, fit, result[2]));
    }
    if (interrupted)
      return py::make_tuple(py::none(), count, evaluated);
    size_t next = 0;
    auto resolved_list=py::reinterpret_borrow<py::list>(resolved);
    for (py::ssize_t i = 0; i < n; ++i)
      if (resolved_list[i].is_none())
        resolved_list[i] = evaluated[next++];
    return py::make_tuple(solution_set(resolved), changed_count,
                          evaluated);
  }
  py::object uniform_mutation(py::object selected, py::object problem,
                              py::dict state) {
    if(!ordinary_generator(state,generator_type))return py::none();
    PopulationView view;
    Repair repair;
    if (!native_integer_width || !repair_contract(problem, repair) ||
        !repair.discrete || !view.load(selected, list_type) ||
        view.rows.empty())
      return py::none();
    py::ssize_t d = repair.dimension, n = view.rows.size();
    if (d == 0)
      return py::none();
    auto bounds = repair.bounds.unchecked<2>();
    for (py::ssize_t j = 0; j < d; ++j)
      if (bounds(0, j) != 0. || bounds(1, j) != 1.)
        return py::none();
    for (auto &row : view.rows) {
      if (!row.integer || row.dimension != d)
        return py::none();
      for (py::ssize_t j = 0; j < d; ++j)
        if (row.integer_at(j) != 0 && row.integer_at(j) != 1)
          return py::none();
    }
    py::array_t<int64_t> output({n, d});
    auto dest = output.mutable_unchecked<2>();
    auto rng = state["rng"];
    for (py::ssize_t i = 0; i < n; ++i) {
      for (py::ssize_t j = 0; j < d; ++j)
        dest(i, j) = view.rows[i].integer_at(j);
      auto count = py::cast<int>(rng.attr("integers")(0, d));
      if (count) {
        auto indices = py::cast<py::array_t<int64_t>>(
            rng.attr("choice")(d, "size"_a = count, "replace"_a = false));
        for (int j = 0; j < count; ++j)
          dest(i, indices.data()[j]) ^= 1;
      }
    }
    if (recording) mutation_rows += n;
    return py::make_tuple(output, state);
  }
  py::dict stats() const {
    auto result = py::dict(
        "data_metrics_enabled"_a = recording,
        "calls"_a = calls, "rows_repaired"_a = rows_repaired,
        "clone_comparisons"_a = clone_comparisons, "clone_hits"_a = clone_hits,
        "evaluator_requests"_a = objective_calls, "fallbacks"_a = fallbacks,
        "native_evaluator_requests"_a=native_evaluator_requests,"callback_evaluator_requests"_a=callback_evaluator_requests,
        "mutation_rows"_a = mutation_rows, "region_calls"_a = region_calls,
        "reused_output_rows"_a = reused_output_rows,
        "repair_arrays"_a = 0,
        "workspace_growths"_a = workspace_growths,
        "workspace_capacity_bytes"_a =
            repair_workspace.capacity_bytes,
        "materialized_bytes"_a = materialized_bytes,
        "converted_input_bytes"_a = converted_input_bytes,
        "integer_snapshot_bytes"_a = integer_snapshot_bytes,
        "fragment_rows"_a=fragment_rows,"fragment_row_simd_rows"_a=fragment_row_simd_rows,"fragment_grouped_rows"_a=fragment_grouped_rows,"fragment_packed_rows"_a=fragment_packed_rows,"fragment_fallbacks"_a=fragment_fallbacks,
        "target_input_allocations"_a = target_input_allocations,
        "identity_repair_rows"_a = identity_repair_rows,
        "cost_scalar"_a = cost_scalar, "cost_region"_a = cost_region);
    if (!recording)
      for (auto item : result)
        if (!item.first.equal(py::str("data_metrics_enabled")) &&
            !item.first.equal(py::str("workspace_capacity_bytes"))) result[item.first]=py::none();
    return result;
  }
};

inline py::object gather_decisions(py::object items, py::object destination = py::none()) {
  PopulationView view;
  auto list_type = py::module_::import("builtins").attr("list");
  if (!view.load(items, list_type))
    return py::none();
  py::ssize_t n = view.rows.size(), d = n ? view.rows[0].dimension : 0;
  bool integer = n > 0;
  for (auto &row : view.rows) {
    if (row.dimension != d)
      return py::none();
    integer &= row.integer;
  }
  auto output=py::reinterpret_borrow<py::array>(py::handle());
  if (destination.is_none())
    output = integer ? py::array(py::array_t<int64_t>({n, d}))
                     : py::array(py::array_t<double>({n, d}));
  else {
    if (!py::isinstance<py::array>(destination))
      return py::none();
    output = py::reinterpret_borrow<py::array>(destination);
    auto dtype = output.dtype();
    if (output.ndim() != 2 || output.shape(0) != n || output.shape(1) != d ||
        !output.writeable() || !(output.flags() & py::array::c_style) ||
        dtype.itemsize() != 8 || !py::cast<bool>(dtype.attr("isnative")) ||
        dtype.kind() != (integer ? 'i' : 'f'))
      return py::none();
  }
  for (py::ssize_t i = 0; i < n; ++i)
    for (py::ssize_t j = 0; j < d; ++j) {
      if (integer)
        static_cast<int64_t *>(output.mutable_data())[i * d + j] =
            view.rows[i].integer_at(j);
      else if (!destination.is_none() && !view.rows[i].integer)
        std::memcpy(static_cast<double *>(output.mutable_data()) + i * d + j,
                    view.rows[i].data + j * view.rows[i].stride, sizeof(double));
      else
        static_cast<double *>(output.mutable_data())[i * d + j] =
            view.rows[i].number_at(j);
    }
  return output;
}
inline py::object gather_scalar(py::object items, const std::string &field) {
  py::ssize_t n = py::len(items);
  py::array_t<double> output({n, py::ssize_t(1)});
  for (py::ssize_t i = 0; i < n; ++i) {
    auto value = items[py::int_(i)].attr(field.c_str());
    if (!PyFloat_Check(value.ptr()))
      return py::none();
    output.mutable_data()[i] = py::cast<double>(value);
  }
  return output;
}

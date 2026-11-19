#pragma once
#include "batch_storage.hpp"
#include "batch_access.hpp"
#include "run_snapshot.hpp"
#include "batch_plan.hpp"
#include <chrono>
#include <unordered_set>

namespace native_components {
struct ComponentResult {
  BatchPtr population;
  NativeState state;
  py::object indices=py::none(),statistics=py::none(),invalidated=py::none();
  py::object binary_output_owner;
  PreparedFragment *fragment=nullptr;
  py::object direct_output=py::none();
};
struct GenerationResult {
  BatchPtr population;
  int64_t used=0,events=0;
  RowPtr best;
  bool cancelled=false;
};

struct BoundEntry {
  py::object spelling=py::none();
  std::string name;
  Component<NativeMath> *entry=nullptr;
  int arity=1;
  bool data_rows=false;
  FragmentBinding fragment;
};
struct BoundStage {py::object owner=py::none();BoundEntry variation,selection;std::shared_ptr<StageTemplate> data_plan;};
struct BoundPath {py::object owner=py::none();std::vector<BoundStage> stages;BoundEntry update;};

struct BatchCore {
  DataPlanner planner;
  py::object data_cost_loader=py::none();
  bool data_cost_loader_attempted=true,data_plan_candidate=false;
  std::shared_ptr<NativeMath> math;
  py::object solve,numpy,solution_type,set_type,generator_type,default_rng;
  NativeData evaluation;
  NativeEvaluation objective;
  RandomBinding random_binding;
  BatchAccess access;
  std::unordered_map<std::string,Component<NativeMath>> entries;
  std::unordered_map<std::string,int> arities;
  std::unordered_set<std::string> bound_names;
  py::object registry;
  py::object fragment_asarray,simple_namespace_type;
  mutable py::object fragment_problem_dictionary=py::none();
  mutable uint64_t fragment_problem_version=0;
  py::object constructor_paths=py::none();
  bool constructor_admission_unused=true;
  std::vector<py::object> description_types;
  std::unordered_map<std::string,py::str> description_keys;
  std::vector<BoundPath> bound_paths;
  uint64_t entry_bindings=0,direct_submissions=0;
  py::object audit_hook=py::none();
  py::object plain_description(py::handle object,const char *key)const {
    bool known=false;for(auto &type:description_types)known|=Py_TYPE(object.ptr())==reinterpret_cast<PyTypeObject *>(type.ptr());
    if(!known||Py_TYPE(object.ptr())->tp_getattro!=PyObject_GenericGetAttr)return py::object();
    const auto &spelling=description_keys.at(key);
    auto descriptor=PyDict_GetItem(Py_TYPE(object.ptr())->tp_dict,spelling.ptr());
    if(descriptor&&PyDescr_IsData(descriptor))return py::object();
    auto raw=PyObject_GenericGetDict(object.ptr(),nullptr);if(!raw){PyErr_Clear();return py::object();}
    auto dictionary=py::reinterpret_steal<py::dict>(raw);auto value=PyDict_GetItem(dictionary.ptr(),spelling.ptr());
    return value?py::reinterpret_borrow<py::object>(value):py::object();
  }
  static bool plain_parameter(py::handle value) {
    if(value.is_none())return true;if(!PyArray_CheckExact(value.ptr()))return false;
    auto array=reinterpret_cast<PyArrayObject *>(value.ptr());
    return PyArray_NDIM(array)==1&&PyArray_ISCARRAY_RO(array)&&PyArray_ISNOTSWAPPED(array)&&
      (PyArray_TYPE(array)==NPY_INT64||PyArray_TYPE(array)==NPY_DOUBLE);
  }
  bool rows_admitted(BatchPtr input,py::handle parameter,NativeState state,bool storage_checked=false)const {
    if(!audit_hook.is_none()||(!storage_checked&&!input->rows_compatible())||!plain_parameter(parameter))return false;
    for(auto key:{"V","Gbest","gaussian_eta","cauchy_eta"})if(state.contains(key)) {
      auto value=state.object(key);if(value.is_none())continue;
      if(!PyArray_CheckExact(value.ptr()))return false;
      auto array=reinterpret_cast<PyArrayObject *>(value.ptr());
      if(!PyArray_ISCARRAY_RO(array)||!PyArray_ISNOTSWAPPED(array)||PyArray_TYPE(array)!=NPY_DOUBLE)return false;
    }
    return true;
  }
  bool fragment_admitted(const BoundEntry &entry,BatchPtr selected,py::handle problem,py::handle rng,bool all_active)const {
    if((entry.fragment.kind==FragmentKind::UniformCross&&selected->integer()&&!packed_fragments())||!all_active||!fragments_enabled()||!audit_hook.is_none()||entry.fragment.kind==FragmentKind::None||selected->external||
       (entry.fragment.kind==FragmentKind::Bit&&!packed_fragments())||
       Py_TYPE(rng.ptr())!=reinterpret_cast<PyTypeObject *>(generator_type.ptr())||
       Py_TYPE(problem.ptr())!=reinterpret_cast<PyTypeObject *>(simple_namespace_type.ptr())||
       !PyCFunction_Check(fragment_asarray.ptr())||PyDict_GetItemString(PyModule_GetDict(numpy.ptr()),"asarray")!=fragment_asarray.ptr())return false;
    // Reject values whose conversion could call Python at the repair boundary.
    // Inspection is dictionary-only and consumes no RNG on rejection.
    auto type=field(problem,"type"),bound=field(problem,"bound");
    if(!(PyList_CheckExact(type.ptr())||PyTuple_CheckExact(type.ptr()))||!PyArray_CheckExact(bound.ptr()))return false;
    auto length=PyList_CheckExact(type.ptr())?PyList_GET_SIZE(type.ptr()):PyTuple_GET_SIZE(type.ptr());
    if(!length)return false;
    auto first=PyList_CheckExact(type.ptr())?PyList_GET_ITEM(type.ptr(),0):PyTuple_GET_ITEM(type.ptr(),0);
    if(!PyUnicode_CheckExact(first))return false;
    auto raw=reinterpret_cast<PyArrayObject *>(bound.ptr());
    if(PyArray_TYPE(raw)!=NPY_DOUBLE&&PyArray_TYPE(raw)!=NPY_INT64)return false;
    auto dictionary=py::reinterpret_steal<py::dict>(PyObject_GenericGetDict(problem.ptr(),nullptr));
    // One-entry converter-admission cache only. No numerical bounds, current
    // list contents, dtype/layout or output validity is cached. Python versions
    // without the dictionary version field use the same conservative scan.
    uint64_t version=0;
#if PY_VERSION_HEX < 0x030C0000
    version=reinterpret_cast<PyDictObject *>(dictionary.ptr())->ma_version_tag;
#endif
    if(version&&fragment_problem_dictionary.is(dictionary)&&fragment_problem_version==version)return true;
    Py_ssize_t position=0;PyObject *key,*value;
    while(PyDict_Next(dictionary.ptr(),&position,&key,&value)) {
      if(!PyUnicode_CheckExact(key)||!PyUnicode_IS_ASCII(key))return false;
      auto normalized=BatchAccess::normalized(PyUnicode_AsUTF8(key));
      if((normalized=="setting"||normalized=="algorithmboundaryhandling")&&!PyUnicode_CheckExact(value))return false;
    }
    auto strategy=field(problem,"_AlgorithmBoundaryHandling");
    if(!strategy.is_none()&&!PyUnicode_CheckExact(strategy.ptr()))return false;
    fragment_problem_dictionary=dictionary;fragment_problem_version=version;
    return true;
  }
  PlanFacts data_input_facts(const StageTemplate &plan,BatchPtr selected,int64_t active,int64_t count)const {
    const auto &name=plan.kernel;const auto &family=plan.capability.family;
    bool integer=family=="point"?selected->integer():family=="permutation"||family=="discrete"||family=="bit";
    bool same_type=integer==selected->integer();
    PlanFacts facts{selected->size(),selected->dimension(),active,selected->integer(),selected->dense_ready,same_type,
      active<selected->size()||name.find("_bit_")!=std::string::npos||name.rfind("cross_fastga_",0)==0,
      "unexamined"};
    facts.may_crop=selected->size()>count;
    facts.selected_evaluated=selected->evaluated;
    return facts;
  }
  std::pair<PlanFacts,bool> bind_data_input(const StageTemplate &plan,BatchPtr selected,int64_t active,
                                           int64_t count,py::handle param,NativeState state,py::handle rng) {
    bool safe=audit_hook.is_none()&&selected->rows_compatible()&&Py_TYPE(rng.ptr())==reinterpret_cast<PyTypeObject *>(generator_type.ptr());
    if(safe&&active) {
      auto search_parameter=plain_description(param,"search");
      auto value=search_parameter?plain_description(search_parameter,"primary"):py::object();
      safe=value&&rows_admitted(selected,value,state,true);
    }
    auto facts=data_input_facts(plan,selected,active,count);safe&=facts.overlay_compatible;
    if(planner.mode=="graph"&&planner.model&&planner.model->valid&&safe){++planner.layout_queries;facts.layout=selected->layout_kind();}
    return {std::move(facts),safe};
  }
  DataDecision plan_data_input(const StageTemplate &plan,BatchPtr selected,int64_t active,int64_t count,
                              py::handle parameter,NativeState state,py::handle rng) {
    if(planner.mode=="graph"&&plan.capability.indexed_supported&&!data_cost_loader_attempted) {
      data_cost_loader_attempted=true;auto loader=std::move(data_cost_loader);data_cost_loader=py::none();
      auto model=loader();if(!model.is_none())planner.model=py::cast<std::shared_ptr<DataCostModel>>(model);
      else planner.mode="dense";
    }
    auto facts=data_input_facts(plan,selected,active,count);
    if(auto skip=planner.precheck(plan,facts))return *skip;
    ++planner.input_admissions;
    auto binding=bind_data_input(plan,selected,active,count,parameter,state,rng);
    return planner.choose(plan,binding.first,binding.second);
  }
  py::object read_description(py::handle owner,const char *key) {
    const auto &spelling=description_keys.at(key);
    for(auto &type:description_types)if(Py_TYPE(owner.ptr())==reinterpret_cast<PyTypeObject *>(type.ptr())) {
      auto descriptor=PyDict_GetItem(Py_TYPE(owner.ptr())->tp_dict,spelling.ptr());
      if(Py_TYPE(owner.ptr())->tp_getattro!=PyObject_GenericGetAttr||(descriptor&&PyDescr_IsData(descriptor)))break;
      auto dictionary=PyObject_GenericGetDict(owner.ptr(),nullptr);
      if(dictionary) {
        auto values=py::reinterpret_steal<py::dict>(dictionary);
        auto raw=PyDict_GetItem(values.ptr(),spelling.ptr());if(raw)return py::reinterpret_borrow<py::object>(raw);
      }else PyErr_Clear();
      break;
    }
    auto raw=PyObject_GetAttr(owner.ptr(),spelling.ptr());if(!raw)throw py::error_already_set();return py::reinterpret_steal<py::object>(raw);
  }
  BoundEntry &bind_entry(BoundEntry &bound,py::object spelling) {
    RegionScope timing(Region::Plan);
    if(!bound.spelling.is(spelling)) {
      bound.spelling=std::move(spelling);bound.name=py::cast<std::string>(bound.spelling);
      bound.entry=&require(bound.name);bound.arity=arities.at(bound.name);++entry_bindings;
      bound.data_rows=input_capability(bound.name).indexed_supported;bound.fragment.bind(bound.name);
    }
    return bound;
  }
  BoundPath &bind_path(size_t index,py::object path) {
    RegionScope timing(Region::Plan);
    if(bound_paths.size()<=index)bound_paths.resize(index+1);
    auto &bound=bound_paths[index];if(!bound.owner.is(path)){bound=BoundPath{};bound.owner=std::move(path);}return bound;
  }
  BatchPtr current;
  std::shared_ptr<RunControl> control=std::make_shared<RunControl>();
  NativeProfile profile;
  RunWorkspace workspace;
  ArrayWorkspace array_workspace;
  std::shared_ptr<RunSnapshot> last_state;
  bool closed=false;
  void close() {
    if(control->progress().running)throw std::runtime_error("Cannot close a running session; cancel and wait for its safe boundary");
    if(closed)return;
    std::vector<double>().swap(workspace.bit_weights);std::vector<double>().swap(workspace.bit_cdf);workspace.bit_dimension=-1;
    closed=true;audit_hook=py::none();last_state.reset();current.reset();public_rows.clear();constructor_paths=py::none();
    random_binding.generator=py::none();random_binding.owner=py::object();random_binding.capsule=py::object();random_binding.lock=py::object();
    objective.target=py::none();objective.target_method=py::none();
    data_cost_loader=py::none();fragment_problem_dictionary=py::none();fragment_problem_version=0;
    bound_paths.clear();bound_names.clear();entries.clear();
    auto allocations=array_workspace.allocations,reuses=array_workspace.reuses;
    array_workspace=ArrayWorkspace();array_workspace.allocations=allocations;array_workspace.reuses=reuses;
    workspace.numbers.slots.clear();workspace.indices.slots.clear();workspace.masks.slots.clear();
    workspace.numbers.capacity_total.store(0);workspace.indices.capacity_total.store(0);workspace.masks.capacity_total.store(0);
    evaluation.repair_workspace=ArrayWorkspace();
    evaluation.bind_cpu(py::none());
  }
  std::shared_ptr<StorageMetrics> storage_metrics=std::make_shared<StorageMetrics>();
  std::vector<RowPtr> public_rows;
  bool diagnostics=false;
  uint64_t generations=0,component_calls=0,evaluation_batches=0;
  struct EntryCounts {uint64_t calls=0,max_rows=0;};
  std::unordered_map<std::string,EntryCounts> entry_counts;
  py::dict run(py::dict context);
  py::dict run_impl(py::dict context);
  py::list run_many(py::iterable contexts,py::object on_result=py::none(),bool collect_results=true);
  explicit BatchCore(std::shared_ptr<NativeMath> m,py::object module,py::sequence pathways)
      :math(std::move(m)),solve(std::move(module)),numpy(py::module_::import("numpy")),evaluation(solve),objective(solve),random_binding(numpy),access(numpy) {
    for(auto key:{"stages","search","primary","choose","update"})description_keys.emplace(key,py::str(key));
    solution_type=solve.attr("Solution");set_type=solve.attr("SolutionSet");
    generator_type=numpy.attr("random").attr("Generator");default_rng=numpy.attr("random").attr("default_rng");
    for(auto spec:{std::make_pair("autooptlib.utils.design._stream_graph",std::vector<const char *>{"StreamPathway","StreamPathwayParam","StreamStage","StreamStageParam"}),
                   std::make_pair("autooptlib.utils.design._helpers",std::vector<const char *>{"SearchStep","SearchParam"})}) {
      auto module=py::module_::import(spec.first);for(auto name:spec.second)description_types.push_back(module.attr(name));
    }
    evaluation.recording=false;fragment_asarray=numpy.attr("asarray");simple_namespace_type=py::module_::import("types").attr("SimpleNamespace");
    evaluation.evaluator_fast_path=[this](py::object problem,py::object data,py::array decisions){return objective.evaluate(problem,data,decisions);};
    evaluation.native_flex=[this](py::object object,const char *name,py::object fallback){return access.flex(object,name,fallback);};
    registry=py::module_::import("autooptlib.components");
    auto names=py::cast<py::dict>(registry.attr("_COMPONENT_MODULES"));
    // The Python registry remains the single name/category authority.
    for(auto item:names) {
      auto name=py::cast<std::string>(item.first);
      entries.emplace(name,Component<NativeMath>{name,math,solution_type,generator_type,default_rng});
      arities.emplace(name,py::cast<int>(registry.attr("component_parent_arity")(name)));
    }
    constructor_paths=py::reinterpret_borrow<py::object>(pathways);admit_paths(pathways);
  }
  void admit_paths(py::sequence pathways) {
    for(auto path:pathways) {
      for(auto stage:path.attr("stages")) {
        auto name=py::cast<std::string>(stage.attr("search").attr("primary"));require(name);
        data_plan_candidate|=input_capability(name).indexed_supported;
        if(!stage.attr("choose").is_none())require(py::cast<std::string>(stage.attr("choose")));
      }
      require(py::cast<std::string>(path.attr("update")));
    }
  }
  void admit_context(py::dict context) {
    auto cpu=context.contains("cpu_rows")?py::reinterpret_borrow<py::object>(context["cpu_rows"]):py::none();
    const bool arrays=context.contains("cpu_arrays")&&py::cast<bool>(context["cpu_arrays"]);
    evaluation.bind_cpu(cpu,arrays,arrays?py::cast<int64_t>(context["cpu_constraints"]):0,
      arrays?py::cast<uintptr_t>(context["cpu_address"]):0,arrays?py::cast<uintptr_t>(context["cpu_context"]):0);
    auto paths=py::reinterpret_borrow<py::object>(context["pathways"]);
    bool admitted=constructor_admission_unused&&paths.is(constructor_paths);constructor_admission_unused=false;
    if(!admitted){bound_names.clear();data_plan_candidate=false;admit_paths(py::cast<py::sequence>(paths));}
    for(auto name:py::cast<py::sequence>(context["archive_names"]))require(py::cast<std::string>(name));
  }
  Component<NativeMath>&require(const std::string&name) {
    auto i=entries.find(name);
    if(i==entries.end()||(!bound_names.count(name)&&py::cast<py::dict>(registry.attr("_custom_components")).contains(py::str(name))))
      throw py::value_error("Population batch library does not contain '"+name+"'");
    bound_names.insert(name);return i->second;
  }
  static py::object pop(NativeState state,const char *key) {return state.pop(key);}
  static py::array_t<int64_t> index_array(const std::vector<int64_t>&values) {
    py::array_t<int64_t> out(values.size());std::copy(values.begin(),values.end(),out.mutable_data());return out;
  }
  ComponentResult component_impl(const std::string&name,BatchPtr population,py::object problem,py::object parameter,
                            NativeState state,int64_t generation,py::object data,py::object count=py::none(),
                            BatchPtr context=nullptr,py::object targets=py::none(),py::object archive=py::none(),Component<NativeMath> *bound=nullptr,int input_choice=-1) {
    auto &entry=bound?*bound:require(name);if(diagnostics){++component_calls;auto &counts=entry_counts[name];++counts.calls;counts.max_rows=std::max<uint64_t>(counts.max_rows,population->size());}
    ComponentResult out{population,state};
    const bool para=entry.metadata.para,arch=entry.metadata.archive;
    const bool choose=entry.metadata.choose,update=entry.metadata.update;
    if((para||choose||update||name=="archive_best"||name=="archive_statistic")&&name!="choose_random"&&name!="choose_traverse") {
      if(!population->evaluated)throw py::value_error("This component requires evaluated population rows");
      for(py::ssize_t i=0;i<population->size();++i)if(!population->valid.data()[i])throw py::value_error("This component requires evaluated population rows");
    }
    ComponentInvocation call;call.random_binding=&random_binding;call.population=population;call.problem=problem;call.parameter=parameter;call.state=state;call.generation_value=generation;call.data=data;
    call.indexed_inputs=(input_choice>=0?input_choice==1:planner.mode=="rules"&&input_capability(name).indexed_supported)&&rows_admitted(population,parameter,state);
    if(para) {
      auto raw=entry.execute(call);
      if(name=="para_pso"&&raw.paired) {
        auto result=raw;auto personal=state.population("Pbest");if(!personal)personal=BatchStorage::from_population(state.object("Pbest"));
        auto combined=BatchStorage::concatenate({personal,population});auto next=combined->take(result.value,audit_hook.is_none());
        state.set_population("Pbest",next);auto best=py::cast<int64_t>(result.auxiliary);state.set("Gbest",next->copy_row(best));
      }else if(raw.state)out.state=*raw.state;
      return out;
    }
    if(arch) {
      NativeResult raw;
      if(name=="archive_diversity") {
        auto combined=archive.is_none()?population:BatchStorage::concatenate({population,BatchStorage::from_population(archive)});
        call.population=combined;call.problem=py::none();call.parameter=problem;raw=entry.execute(call);
        out.indices=raw.value;out.population=combined->take(out.indices,audit_hook.is_none());return out;
      }
      call.problem=archive;call.parameter=py::none();raw=entry.execute(call);
      out.indices=raw.value;
      if(name=="archive_statistic"){out.statistics=out.indices;out.indices=py::none();}
      else if(name=="archive_tabu")out.indices=py::none();
      else if(name!="archive_tabu") {
        auto best=population->take(out.indices,audit_hook.is_none());
        out.population=archive.is_none()?best:BatchStorage::concatenate({BatchStorage::from_population(archive),best});
      }
      return out;
    }
    if(!count.is_none())state.set("_selection_count",count);
    if(context)state.set_population("_population_context",context);
    if(!targets.is_none())state.set("_target_population_indices",targets);
    NativeResult raw;
    try {raw=entry.execute(call);}
    catch(...) {for(auto key:{"_selection_count","_population_context","_target_population_indices"})pop(state,key);throw;}
    for(auto key:{"_selection_count","_population_context","_target_population_indices"})pop(state,key);
    auto result=raw;
    if(result.state) {
      out.state=*result.state;
      if(state.contains("rng")&&!out.state.contains("rng"))out.state.set("rng",state.object("rng"));
      for(auto key:{"_selection_count","_population_context","_target_population_indices"})pop(out.state,key);
    }
    if(choose)out.indices=result.value;
    else if(update){out.indices=result.value;out.population=population->take(out.indices,audit_hook.is_none());}
    else {out.invalidated=pop(out.state,"_stream_invalidated_mask");
      if(fragment_sink&&fragment_sink->kind!=FragmentKind::None){out.fragment=fragment_sink;out.population.reset();return out;}
      auto values=py::cast<py::array>(result.value);
      if(result.binary_output)out.binary_output_owner=result.value;
      if(direct_result_capture) {
        if(values.ndim()!=2||!(values.flags()&py::array::c_style)||
           (values.dtype().num()!=NPY_INT64&&values.dtype().num()!=NPY_DOUBLE)||!PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject *>(values.ptr())))
          throw py::value_error("Population decisions must be a contiguous 2-D float64 or int64 matrix");
        out.direct_output=values;out.population.reset();
      }else out.population=std::make_shared<BatchStorage>(values);
    }
    return out;
  }
  ComponentResult component(const std::string&name,BatchPtr population,py::object problem,py::object parameter,
                            NativeState state,int64_t generation,py::object data,py::object count=py::none(),
                            BatchPtr context=nullptr,py::object targets=py::none(),py::object archive=py::none(),Component<NativeMath> *bound=nullptr,int input_choice=-1) {
    RegionScope timing(Region::Component);
    auto result=component_impl(name,population,problem,parameter,state,generation,data,count,context,targets,archive,bound,input_choice);
    if(!audit_hook.is_none()) {
      result.binary_output_owner=py::object();
      audit_hook(name,py::cast(result.population),result.state.python(),result.indices,result.statistics,result.invalidated);
      result.state.refresh();
    }
    return result;
  }
  static std::vector<RowPtr> boundary_rows(BatchPtr population) {
    std::vector<RowPtr> rows;rows.reserve(population->size());
    for(py::ssize_t i=0;i<population->size();++i)rows.push_back(population->snapshot_row(i));return rows;
  }
  py::object public_solutions(bool exported=false) {
    if(current){current->dense();current->ensure_values();}
    py::list records;for(auto &row:public_rows){records.append(row->materialize(solution_type));row->escaped|=exported;}return set_type(records);
  }
  // Uses the exact existing native repair and FE kernel. Only the result
  // representation differs; unsupported repairs use the shared policy adapter.
  std::pair<BatchPtr,std::vector<RowPtr>> changed(py::array decisions,BatchPtr sources,py::object problem,
                                               py::object data,int64_t remaining,py::object invalidated,
                                               int64_t&used,bool&interrupted,BatchPtr raw_rows=nullptr,bool use_source_rows=false,
                                               py::handle binary_output_owner=py::handle(),const PreparedFragment *fragment=nullptr,
                                               NativeData::Mask *native_mask=nullptr) {
    if(diagnostics)++evaluation_batches;
    // Snapshot source metadata before the first user callback, as in the old
    // to_solutions boundary, without making Python individual objects.
    // Unexposed native rows already are the immutable metadata snapshot.
    // Borrow that table; forced reevaluation then allocates no source snapshot.
    // Exposed/matrix-backed rows retain the original pre-callback freeze.
    bool borrow_sources=!sources->external&&sources->evaluated&&sources->objectives.is_none();
    for(py::ssize_t i=0;i<sources->size()&&borrow_sources;++i) {
      const auto &row=sources->rows[i];
      borrow_sources=sources->provenance[i]&&row&&row->record.is_none()&&sources->private_row(row);
    }
    std::vector<RowPtr> frozen_sources;
    const auto *source_rows=&sources->rows;
    if(!borrow_sources) {
      frozen_sources.reserve(sources->size());
      for(py::ssize_t i=0;i<sources->size();++i)frozen_sources.push_back(sources->snapshot_row(i));
      source_rows=&frozen_sources;
    }
    std::optional<DecisionInput> input,source;
    std::optional<py::array> source_matrix;
    if(raw_rows)input.emplace(BatchStorage::read_rows(raw_rows));
    use_source_rows=use_source_rows&&sources->rows_compatible();
    std::function<const DecisionInput&()> source_loader=[&]()->const DecisionInput&{
      if(input&&raw_rows==sources) {
        if(sources->metrics&&sources->metrics->enabled)++sources->metrics->row_table_reuses;
        return *input;
      }
      if(!source)source.emplace(BatchStorage::read_rows(sources));return *source;
    };
    if(!use_source_rows)source_matrix.emplace(sources->dense());
    // Consume the actual unescaped bit result as the repair slot. This is an
    // owner proof from this invocation, never a cached assertion on input rows.
    bool private_binary=false;
    if(fragments_enabled()&&!fragment&&!raw_rows&&binary_output_owner&&!binary_output_owner.is_none()&&
       PyArray_CheckExact(binary_output_owner.ptr())&&!decisions.is_none()) {
      auto owner=py::reinterpret_borrow<py::array>(binary_output_owner);
      private_binary=owner.ndim()==2&&decisions.ndim()==2&&owner.data()==decisions.data()&&
        owner.shape(0)>=decisions.shape(0)&&owner.shape(1)==decisions.shape(1)&&owner.dtype().is(py::dtype::of<int64_t>())&&
        decisions.dtype().is(py::dtype::of<int64_t>())&&decisions.writeable()&&(decisions.flags()&py::array::c_style);
    }
    NativeData::EvaluationResult evaluated;
    auto fallback=(input||use_source_rows)?evaluation.changed<true>(decisions,py::none(),problem,data,remaining,invalidated,
                solve.attr("_batch_evaluate_changed"),private_binary,private_binary,private_binary,true,true,&evaluated,
                use_source_rows?nullptr:&*source_matrix,input?&*input:nullptr,nullptr,use_source_rows?&source_loader:nullptr,py::handle(),fragment,native_mask):
                evaluation.changed(decisions,py::none(),problem,data,remaining,invalidated,
                solve.attr("_batch_evaluate_changed"),private_binary,private_binary,private_binary,true,true,&evaluated,&*source_matrix,nullptr,nullptr,nullptr,binary_output_owner,fragment,native_mask);
    // Only the compatibility fallback crosses the legacy tuple protocol.
    if(!evaluated.available) {
      auto raw=py::cast<py::tuple>(fallback);auto resolved=py::cast<py::list>(raw[0]);
      for(auto index:resolved)evaluated.sources.push_back(index.is_none()?-1:py::cast<int64_t>(index));
      evaluated.used=py::cast<int64_t>(raw[1]);evaluated.interrupted=py::cast<bool>(raw[4]);
      auto inputs=py::cast<py::list>(raw[2]),results=py::cast<py::list>(raw[3]);
      for(size_t i=0;i<inputs.size();++i){auto result=py::cast<py::tuple>(results[i]);
        evaluated.evaluated.push_back({py::cast<py::array>(inputs[i]),py::cast<double>(result[0]),
            py::cast<double>(result[1]),py::reinterpret_borrow<py::object>(result[2])});}
    }
    used=evaluated.used;interrupted=evaluated.interrupted;
    std::vector<RowPtr> observed;observed.reserve(evaluated.evaluated.size());
    for(auto &row:evaluated.evaluated)observed.push_back(std::make_shared<BatchRow>(
        std::move(row.decision),row.objective,row.constraint,fitness_value(row.objective,row.constraint),std::move(row.auxiliary)));
    if(interrupted)return {nullptr,observed};
    std::vector<RowPtr> rows;rows.reserve(evaluated.sources.size());size_t next=0;
    for(auto index:evaluated.sources)rows.push_back(index<0?observed.at(next++):source_rows->at(index));
    auto output=BatchStorage::from_rows(rows,sources->dimension(),sources->integer(),storage_metrics,audit_hook.is_none());
    return {output,observed};
  }
  static BatchPtr from_rows(const std::vector<RowPtr>&rows,py::ssize_t dimension,bool empty_integer=false,std::shared_ptr<StorageMetrics> metrics=nullptr) {
    return BatchStorage::from_rows(rows,dimension,empty_integer,std::move(metrics));
  }
  static RowPtr minimum(const std::vector<RowPtr>&rows,RowPtr best=nullptr) {
    for(auto &row:rows)if(row){row->refresh_record();if(!best||row->key()<best->key())best=row;}return best;
  }
  static RowPtr minimum(BatchPtr batch,RowPtr best=nullptr) {
    auto values=batch->fitness_values();for(py::ssize_t i=0;i<batch->size();++i)if(!best||values.key(i)<best->key())best=batch->snapshot_row(i);return best;
  }
  std::vector<int64_t> resize(py::handle raw,py::ssize_t n,int64_t required,py::object rng) {
    std::vector<int64_t> values;
    if(raw) {
      auto a=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(raw);if(!a)throw py::type_error("Invalid parent indices");
      values.reserve(a.size());for(py::ssize_t i=0;i<a.size();++i)if(a.data()[i]>=0&&a.data()[i]<n)values.push_back(a.data()[i]);
    }
    if(values.empty()){values.reserve(n);for(int64_t i=0;i<n;++i)values.push_back(i);}
    if(values.size()>=static_cast<size_t>(required)){values.resize(required);return values;}
    const auto original_size=values.size(),extra=static_cast<size_t>(required)-original_size;
    if(!original_size&&extra)throw py::value_error("a cannot be empty unless no samples are taken");
    values.reserve(required);
    if(Py_TYPE(rng.ptr())==reinterpret_cast<PyTypeObject *>(generator_type.ptr())) {
      RandomBoundary boundary(rng);auto positions=workspace_array<int64_t>(extra);boundary.random.integers(0,original_size,positions.mutable_data(),extra);
      for(size_t i=0;i<extra;++i)values.push_back(values[positions.data()[i]]);
    }else {
      auto draw=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(rng.attr("choice")(index_array(values),"size"_a=extra,"replace"_a=true));
      if(!draw||draw.size()!=static_cast<py::ssize_t>(extra))throw py::value_error("Invalid parent random batch");values.insert(values.end(),draw.data(),draw.data()+draw.size());
    }
    return values;
  }
  GenerationResult generation(py::sequence pathways,py::sequence parameters,py::object solutions,std::vector<NativeState> &auxiliary,
                              py::object problem,py::object data,py::object setting,int64_t generation,int64_t remaining,
                              py::sequence archive_names,std::vector<ArchiveValue> &archives,py::object event_budget) {
    RegionScope timing(Region::Generation);
    if(diagnostics)++generations;
    auto progress_base=control->progress();
    auto rng=access.rng(setting);
    auto parent=current?current:BatchStorage::from_solutions(solutions);parent->metrics=storage_metrics;
    auto flex=[&](py::handle object,const char *name,py::object fallback){return access.flex(object,name,fallback);};
    int64_t target=py::cast<int64_t>(flex(setting,"_AlgorithmOffspringSize",py::int_(parent->size())));
    if(!event_budget.is_none())target=std::min(target,std::max<int64_t>(0,py::cast<int64_t>(event_budget)));
    if(target<=0||remaining<=0){public_rows=boundary_rows(parent);return {parent,0,0,nullptr};}
    std::vector<int> crosses,plain;crosses.reserve(pathways.size());plain.reserve(pathways.size());
    for(py::ssize_t i=0;i<py::len(pathways);++i) {
      bool cross=false;for(auto stage:pathways[i].attr("stages")) {
        auto primary=stage.attr("search").attr("primary");
        bool found;
        if(PyUnicode_CheckExact(primary.ptr())&&PyUnicode_IS_ASCII(primary.ptr()))
          found=PyUnicode_GET_LENGTH(primary.ptr())>=6&&std::memcmp(PyUnicode_1BYTE_DATA(primary.ptr()),"cross_",6)==0;
        else if(PyUnicode_CheckExact(primary.ptr()))found=py::cast<std::string>(primary).rfind("cross_",0)==0;
        else found=py::cast<bool>(primary.attr("startswith")("cross_"));
        if(found){cross=true;break;}
      }
      (cross?crosses:plain).push_back(i);
    }
    auto route_owner=workspace_array<int64_t>(target);auto route=route_owner.mutable_data();
    ScratchVector<int64_t> owners_lease;auto &owners=owners_lease.values;
    if(crosses.size()==1&&plain.size()==1) {
      ScratchVector<double> values_lease;auto &values=values_lease.values;
      if(Py_TYPE(rng.ptr())==reinterpret_cast<PyTypeObject *>(generator_type.ptr())) {
        RandomBoundary boundary(rng);boundary.random.uniform(values,target);
      }else {
        auto draw=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(rng.attr("random")(target));
        if(!draw||draw.size()!=target)throw py::value_error("Invalid routing random batch");values.assign(draw.data(),draw.data()+target);
      }
      auto rate=py::cast<double>(flex(setting,"_AlgorithmCrossoverRate",py::float_(1.)));
      for(int64_t i=0;i<target;++i)route[i]=values[i]<rate?crosses[0]:plain[0];
    }else if(Py_TYPE(rng.ptr())==reinterpret_cast<PyTypeObject *>(generator_type.ptr())) {
      RandomBoundary boundary(rng);boundary.random.integers(0,py::len(pathways),route,target);
    }else {
      auto draw=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(rng.attr("integers")(0,py::len(pathways),"size"_a=target));
      if(!draw||draw.size()!=target)throw py::value_error("Invalid routing random batch");std::copy_n(draw.data(),target,route);
    }
    // These positions also supply stage counts and update groups. Owner order
    // remains first-arrival order, independently of the hash table's order.
    struct Routed {std::vector<int64_t> positions;int64_t offset=0;};
    std::unordered_map<int64_t,Routed> routed;
    for(int64_t i=0;i<target;++i) {
      auto item=routed.try_emplace(route[i]);
      if(item.second)owners.push_back(route[i]);
      item.first->second.positions.push_back(i);
    }
    std::vector<BatchPtr> outputs;outputs.reserve(pathways.size());int64_t output_rows=0;
    std::vector<RowPtr> observed;int64_t used=0;bool interrupted=false;
    for(auto owner:owners) {
      auto path=pathways[owner],parameter=parameters[owner];auto &description=bind_path(owner,py::reinterpret_borrow<py::object>(path));
      int64_t count=routed.at(owner).positions.size();
      auto path_state=auxiliary[owner].copy();
      auto &states=path_state.stages();auto stages=py::cast<py::sequence>(read_description(path,"stages"));
      auto params=py::cast<py::sequence>(read_description(parameter,"stages"));
      description.stages.resize(std::min(py::len(stages),py::len(params)));
      path_state.grow_stages(py::len(stages));
      auto pool=parent;
      for(py::ssize_t j=0;j<std::min(py::len(stages),py::len(params));++j) {
        if(control->cancelled()) {
          public_rows=boundary_rows(parent);
          return {parent,used,target,minimum(observed),true};
        }
        if(PyErr_CheckSignals()<0)throw py::error_already_set();
        auto stage=stages[j],param=params[j];
        if(description.stages.size()<=static_cast<size_t>(j))description.stages.resize(j+1);
        auto &slot=description.stages[j];
        if(!slot.owner.is(stage)){slot=BoundStage{};slot.owner=py::reinterpret_borrow<py::object>(stage);}
        auto &variation=bind_entry(slot.variation,read_description(read_description(stage,"search"),"primary"));
        const auto &name=variation.name;
        auto state=states[j];state.set("rng",rng);
        state.set("_stream_generation_target",flex(setting,"_AlgorithmOffspringSize",py::int_(parent->size())));
        int64_t required=variation.arity==2?2*((count+1)/2):count;
        py::object raw_indices;
        auto choose=read_description(stage,"choose");
        if(!choose.is_none()) {
          auto &selection=bind_entry(slot.selection,read_description(stage,"choose"));
          auto chosen=component(selection.name,pool,problem,read_description(param,"choose"),state,generation+j,data,py::int_(required),nullptr,py::none(),py::none(),selection.entry);
          state=chosen.state;raw_indices=chosen.indices;
        }
        auto indices=resize(raw_indices,pool->size(),required,rng);
        bool identity=indices.size()==static_cast<size_t>(pool->size());
        for(size_t i=0;i<indices.size()&&identity;++i)identity=indices[i]==static_cast<int64_t>(i);
        auto selected=identity?pool:pool->take(indices,audit_hook.is_none());
        ScratchVector<int64_t> active_lease;auto &active=active_lease.values;
        if(j) {
          ScratchVector<double> mask_lease;auto &mask=mask_lease.values;
          if(Py_TYPE(rng.ptr())==reinterpret_cast<PyTypeObject *>(generator_type.ptr())) {
            RandomBoundary boundary(rng);boundary.random.uniform(mask,selected->size());
          }else {
            auto draw=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(rng.attr("random")(selected->size()));
            if(!draw||draw.size()!=selected->size())throw py::value_error("Invalid mutation random batch");mask.assign(draw.data(),draw.data()+draw.size());
          }
          auto rate=py::cast<double>(flex(setting,"_AlgorithmMutationRate",py::float_(1.)));
          for(size_t i=0;i<mask.size();++i)if(mask[i]<rate)active.push_back(i);
        }else{active.reserve(selected->size());for(py::ssize_t i=0;i<selected->size();++i)active.push_back(i);}
        DataDecision data_plan;data_plan.materialize_selected=true;
        if(planner.dump||(variation.data_rows&&(planner.mode=="rules"||(planner.mode=="graph"&&((planner.model&&planner.model->valid)||!data_cost_loader_attempted))))) {
          if(!slot.data_plan||slot.data_plan->kernel!=name) {
            planner.replans+=bool(slot.data_plan);++planner.template_builds;
            slot.data_plan=std::make_shared<StageTemplate>(name);
          }else ++planner.template_cache_hits;
          data_plan=plan_data_input(*slot.data_plan,selected,active.size(),count,param,state,rng);
          if(planner.dump&&!planner.records.empty()) {
            auto &record=planner.records.back();record["pathway"]=owner;record["stage"]=j;record["generation"]=generation+j;
          }
        }
        if(planner.mode=="dense"&&variation.data_rows&&audit_hook.is_none()) {
          auto search=plain_description(param,"search");auto parameter=search?plain_description(search,"primary"):py::object();
          if(parameter&&rows_admitted(selected,parameter,state)) {
            data_plan.rows=true;data_plan.source_rows=true;data_plan.overlay=true;data_plan.materialize_selected=false;
          }
        }
        py::array decisions=py::reinterpret_borrow<py::array>(py::none());BatchPtr raw_decisions;
        py::object binary_output_owner;
        PreparedFragment *fragment=nullptr;std::optional<PreparedFragment> frame;
        if(data_plan.materialize_selected)decisions=selected->dense();else raw_decisions=selected;
        ScratchVector<bool> force_lease(selected->size());auto &force=force_lease.values;
        NativeData::Mask mask(force);
        if(!active.empty()) {
          const auto &metadata=variation.entry->metadata;
          const bool targets_required=metadata.target_indices&&pool==parent;
          ScratchVector<int64_t> targets_lease;auto &targets=targets_lease.values;
          if(targets_required)for(auto i:active)targets.push_back(indices[i]);
          ComponentResult varied;
          {
          if(fragment_admitted(variation,selected,problem,rng,active.size()==static_cast<size_t>(selected->size())))frame.emplace();
          FragmentCapture capture(frame?&*frame:nullptr,fragments_enabled()&&audit_hook.is_none()&&active.size()==static_cast<size_t>(selected->size()));
          varied=component(name,active.size()==static_cast<size_t>(selected->size())?selected:selected->take(active,audit_hook.is_none()),problem,
                     read_description(read_description(param,"search"),"primary"),state,generation+j,data,py::none(),metadata.population_context?parent:nullptr,targets_required?py::object(index_array(targets)):py::none(),py::none(),variation.entry,data_plan.rows?1:0);
          }
          state=varied.state;
          if(varied.fragment){fragment=varied.fragment;raw_decisions.reset();decisions=py::reinterpret_borrow<py::array>(py::none());}
          else if(!varied.direct_output.is_none()) {
            auto output=py::cast<py::array>(varied.direct_output);
            if(output.shape(0)!=static_cast<py::ssize_t>(active.size())||output.shape(1)!=selected->dimension())throw py::value_error(name+" returned incompatible batch dimensions");
            decisions=output;raw_decisions.reset();binary_output_owner=varied.binary_output_owner;
            if(diagnostics)++direct_submissions;
          }else {
          if(varied.population->size()!=static_cast<py::ssize_t>(active.size())||varied.population->dimension()!=selected->dimension())
            throw py::value_error(name+" returned incompatible batch dimensions");
          if(active.size()==static_cast<size_t>(selected->size())) {
            binary_output_owner=varied.binary_output_owner;
            if(data_plan.source_rows||data_plan.rows||data_plan.overlay){raw_decisions=varied.population;decisions=py::reinterpret_borrow<py::array>(py::none());}
            else decisions=varied.population->dense();
          } else if((data_plan.overlay||!data_plan.materialize_selected)&&selected->integer()==varied.population->integer()&&varied.population->rows_compatible()) {
            raw_decisions=data_plan.overlay?BatchStorage::overlay(selected,varied.population,active):BatchStorage::merge(selected,varied.population,active);
            decisions=py::reinterpret_borrow<py::array>(py::none());
          } else {
            raw_decisions.reset();if(decisions.is_none())decisions=selected->dense();
            auto npdtype=numpy.attr("result_type")(decisions.dtype(),varied.population->dense().dtype());
            auto converted=py::array::ensure(decisions.attr("astype")(npdtype,"copy"_a=true));
            auto values=py::array::ensure(varied.population->dense().attr("astype")(npdtype,"copy"_a=false));
            for(size_t k=0;k<active.size();++k)std::memcpy(static_cast<char *>(converted.mutable_data())+active[k]*selected->dimension()*8,static_cast<const char *>(values.data())+k*selected->dimension()*8,selected->dimension()*8);
            decisions=converted;
            if(storage_metrics->enabled)storage_metrics->merge_bytes+=converted.nbytes()+active.size()*selected->dimension()*8;
          }
          }
          auto flags=py::reinterpret_borrow<py::array_t<bool,py::array::c_style|py::array::forcecast>>(py::handle());
          if(!varied.invalidated.is_none())flags=py::array_t<bool,py::array::c_style|py::array::forcecast>::ensure(varied.invalidated);
          if(!varied.invalidated.is_none()&&flags.size()!=1&&flags.size()!=static_cast<py::ssize_t>(active.size()))throw py::value_error("Invalid variation mask dimensions");
          for(size_t k=0;k<active.size();++k)force[active[k]]=varied.invalidated.is_none()?true:flags.data()[flags.size()==1?0:k];
        }
        if(j==py::len(path.attr("stages"))-1) {
          auto n=std::min<int64_t>(count,fragment?fragment->count:raw_decisions?raw_decisions->size():decisions.shape(0));
          if(fragment)fragment->count=n;
          else if(raw_decisions) {
            if(n<raw_decisions->size()){std::vector<int64_t> prefix;prefix.reserve(n);for(int64_t i=0;i<n;++i)prefix.push_back(i);raw_decisions=raw_decisions->take(prefix,audit_hook.is_none());}
          }else decisions=py::array(decisions.dtype(),std::vector<py::ssize_t>{n,decisions.shape(1)},std::vector<py::ssize_t>{decisions.strides(0),decisions.strides(1)},decisions.data(),decisions);
          mask.view_size=n;
        }
        int64_t consumed=0;auto evaluated=changed(decisions,selected,problem,data,remaining-used,py::none(),consumed,interrupted,raw_decisions,data_plan.source_rows,binary_output_owner,fragment,&mask);
        used+=consumed;observed.insert(observed.end(),evaluated.second.begin(),evaluated.second.end());path_state.set_stage(j,state);
        if(interrupted)break;
        pool=evaluated.first;
        if(name=="search_cma")component("para_cma",pool,problem,py::none(),state,0,py::none());
        else if(name=="search_pso") {
          auto targets=state.take_indices("_stream_pso_target_indices");state.set_indices("_batch_pso_targets",targets?std::move(*targets):indices);
          component("para_pso",pool,problem,py::none(),state,0,py::none());pop(state,"_batch_pso_targets");
        }
        if(progress_base.running) {
          control->publish(progress_base.evaluations+used,progress_base.events+target,generation);
          cooperative_boundary();
        }
      }
      path_state.commit_stages();auxiliary[owner]=path_state;
      if(interrupted)break;
      routed.at(owner).offset=output_rows;output_rows+=pool->size();outputs.push_back(pool);
    }
    if(interrupted) {
      auto previous=solutions.is_none()?public_rows:BatchStorage::from_solutions(solutions)->rows;
      auto best=minimum(previous,minimum(observed));public_rows=boundary_rows(parent);
      return {parent,used,target,best};
    }
    auto combined=BatchStorage::concatenate(outputs);ScratchVector<int64_t> order_lease;auto &order=order_lease.values;
    for(int64_t i=0;i<target;++i)order.push_back(routed.at(route[i]).offset++);
    auto children=combined->take(order,audit_hook.is_none());
    struct Group{std::string name,search;py::object parameter;int64_t owner;std::vector<int64_t> rows;};
    std::vector<Group> groups;
    for(auto owner:owners) {
      auto path=pathways[owner],param=parameters[owner];auto raw_update=read_description(path,"update");auto name=py::cast<std::string>(raw_update);
      auto search=py::cast<std::string>(path.attr("stages")[py::int_(-1)].attr("search").attr("primary"));
      auto &description=bind_path(owner,py::reinterpret_borrow<py::object>(path));
      bind_entry(description.update,raw_update);
      auto raw_parameter=read_description(param,"update");Group *found=nullptr;
      for(auto &g:groups)if(g.name==name&&BatchAccess::same_parameter(g.parameter,raw_parameter)&&(name!="update_iterated_local_search"||g.search==search)){found=&g;break;}
      if(!found){groups.push_back({name,search,py::reinterpret_borrow<py::object>(raw_parameter),owner,{}});found=&groups.back();}
      auto &positions=routed.at(owner).positions;
      if(found->rows.empty())found->rows=std::move(positions);
      else found->rows.insert(found->rows.end(),positions.begin(),positions.end());
    }
    auto next=parent;
    for(auto &g:groups) {
      auto state=auxiliary[g.owner];state.set("rng",rng);state.set("_active_search_operator",g.search);
      std::sort(g.rows.begin(),g.rows.end());auto merged=BatchStorage::concatenate({next,children->take(g.rows,audit_hook.is_none())});
      next=component(g.name,merged,problem,g.parameter,state,generation,data,py::none(),nullptr,py::none(),py::none(),bound_paths[g.owner].update.entry).population;pop(state,"_active_search_operator");
      if(next->size()!=parent->size()) {
        auto scores=next->fitness_values().selection_scores();
        auto raw=PyArray_ArgSort(reinterpret_cast<PyArrayObject *>(scores.ptr()),-1,NPY_QUICKSORT);if(!raw)throw py::error_already_set();
        auto all=py::reinterpret_steal<py::array_t<int64_t>>(raw);std::vector<int64_t> keep(all.data(),all.data()+std::min(parent->size(),all.size()));next=next->take(keep,audit_hook.is_none());
      }
    }
    current=next;
    for(py::ssize_t i=0;i<py::len(archive_names);++i) {
      auto name=py::cast<std::string>(archive_names[i]);py::object archive=archives.at(i).object();
      if(name!="archive_statistic"&&!archive.is_none()&&!(population_storage_type&&PyObject_TypeCheck(archive.ptr(),population_storage_type))) {
        if(py::hasattr(archive,"_batch_X"))archive=py::cast(BatchStorage::from_population(archive));
        else archive=py::len(archive)?py::cast(BatchStorage::from_solutions(archive,current->dimension(),current->integer())):py::none();
      }
      NativeState state;state.set("rng",rng);auto result=component(name,current,problem,py::none(),state,generation,data,py::none(),nullptr,py::none(),archive);
      if(name=="archive_statistic")archives[i]=ArchiveValue(result.statistics);
      else {archives[i]=ArchiveValue();archives[i].population=result.population;}
    }
    public_rows=boundary_rows(current);
    return {current,used,target,minimum(public_rows,minimum(observed))};
  }
};
}

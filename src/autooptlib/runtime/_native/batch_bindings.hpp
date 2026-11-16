#pragma once
#include "configuration.hpp"
#include "ownership.hpp"
namespace native_components {
inline void bind_batch_core(py::module_ &m,std::shared_ptr<NativeMath> math) {
  m.def("_set_fragment_mode",[](int mode){if(mode< -1||mode>3)throw py::value_error("fragment mode out of range");fragment_mode_override=mode;});
  m.def("component_capabilities",[]() {
    py::dict result;auto registry=py::module_::import("autooptlib.components");
    for(auto item:py::cast<py::dict>(registry.attr("_COMPONENT_MODULES"))) {
      auto name=py::cast<std::string>(item.first);auto cap=input_capability(name);
      result[item.first]=py::dict("need"_a=need_name(cap.need),"indexed_supported"_a=cap.indexed_supported,
        "ordered_read_only"_a=cap.ordered_read_only,"population_context"_a=cap.population_context,
        "stateful"_a=cap.stateful,"family"_a=cap.family,"version"_a=1);
    }return result;
  });
  m.def("_bounded_data_plan",[](unsigned decisions,py::list alternatives,uint64_t limit,unsigned maximum,uint32_t linear) {
    if(decisions>16||alternatives.size()!=(1u<<decisions))throw py::value_error("Provide all alternatives for at most 16 decisions");
    auto solution=bounded_plan(decisions,limit,maximum,[&](uint32_t mask) {
      auto source=py::cast<py::dict>(alternatives[mask]);PlanAlternative a{py::cast<double>(source["cost"]),{},true};
      if(source.contains("legal"))a.legal=py::cast<bool>(source["legal"]);
      for(auto raw:py::cast<py::list>(source["buffers"])) {
        auto b=py::cast<py::tuple>(raw);a.buffers.push_back({py::cast<uint32_t>(b[0]),py::cast<uint32_t>(b[1]),py::cast<uint64_t>(b[2])});
      }return a;
    },linear);
    return py::dict("found"_a=solution.found,"mask"_a=solution.mask,"cost"_a=solution.cost,"peak"_a=solution.peak,"bounded_fallback"_a=solution.bounded_fallback);
  },"decisions"_a,"alternatives"_a,"memory_limit"_a,"maximum"_a=8,"linear"_a=0);
  m.def("_private_batch",[](py::array input) {
    if(input.ndim()!=2||!(input.flags()&py::array::c_style))throw py::value_error("Expected contiguous 2-D input");
    py::array copy(input.dtype(),std::vector<py::ssize_t>{input.shape(0),input.shape(1)});
    std::memcpy(copy.mutable_data(),input.data(),input.nbytes());return std::make_shared<BatchStorage>(copy);
  });
  m.def("_row_data_facts",[](BatchPtr input) {
    auto compatible=input->rows_compatible();auto layout=input->layout_kind();
    return py::dict("compatible"_a=compatible,"owners"_a=input->owner_count,"layout"_a=layout);
  });
  py::class_<DataCostModel,std::shared_ptr<DataCostModel>>(m,"DataCostModel")
    .def(py::init([](py::dict model){auto result=std::make_shared<DataCostModel>();result->load(model);return result;}))
    .def("predict",[](DataCostModel &model,const std::string &family,const std::string &layout,bool integer,int64_t n,int64_t d)->py::object {
      auto value=model.predict(family,layout,integer,n,d);if(!value)return py::none();auto &s=*value;
      return py::dict("rows"_a=s.rows,"dense"_a=s.dense,"gather"_a=s.gather,"row_table"_a=s.row_table,
        "repair_rows"_a=s.repair_rows,"repair_dense"_a=s.repair_dense,"merge"_a=s.merge,"overlay"_a=s.overlay,"crop"_a=s.crop,"residual"_a=s.residual,"planning_ns"_a=s.planning_ns);
    });
  auto preparation=std::make_shared<NativeConfiguration>(py::module_::import("numpy"));
  m.def("_prepare_configuration",[preparation](py::dict source,py::object setting){return preparation->prepare(std::move(source),std::move(setting));});
  auto ownership=std::make_shared<NativeOwnership>(py::module_::import("numpy"));
  m.def("_mutable_owners",[ownership](py::object root,py::sequence types){return ownership->collect(std::move(root),std::move(types));});
  m.def("_copy_task",[ownership](py::object root,py::sequence types){return ownership->clone(std::move(root),std::move(types));});
  py::class_<RunSnapshot,std::shared_ptr<RunSnapshot>>(m,"RunSnapshot")
    .def_property_readonly("schema",[](RunSnapshot&){return 3;})
    .def_property_readonly("evaluations",[](RunSnapshot&s){return s.evaluations;})
    .def_property_readonly("complete",[](RunSnapshot&s){return s.complete;})
    .def("_legacy_live",[](RunSnapshot &s,py::object solve){return s.legacy_live(solve.attr("Solution"),solve.attr("SolutionSet"));})
    .def("to_legacy",[](RunSnapshot &s,py::object solve){return s.legacy(solve.attr("Solution"),solve.attr("SolutionSet"));});
  py::class_<RunControl,std::shared_ptr<RunControl>>(m,"RunControl")
    .def(py::init<>()).def("cancel",&RunControl::cancel).def("reset",&RunControl::reset)
    .def_property_readonly("progress",[](RunControl &control){auto p=control.progress();return py::dict(
        "evaluations"_a=p.evaluations,"events"_a=p.events,"generation"_a=p.generation,
        "completed_runs"_a=p.completed_runs,"running"_a=p.running,"cancelled"_a=p.cancelled);});
  m.def("initialize_population",&initialize_population);
  auto storage=py::class_<BatchStorage,BatchPtr>(m,"PopulationStorage")
    .def(py::init([](py::array x){auto p=std::make_shared<BatchStorage>(x);p->external=true;return p;}))
    .def_static("from_population",&BatchStorage::from_population)
    .def_static("concatenate",&BatchStorage::concatenate)
    .def_static("from_solutions",&BatchStorage::from_solutions,"solutions"_a,"dimension"_a=0,"integer"_a=false)
    .def("__deepcopy__",[](py::object source,py::dict memo){
      SnapshotClone clone(memo);auto result=py::cast(clone.population(py::cast<BatchPtr>(source)));
      memo[py::int_(reinterpret_cast<uintptr_t>(source.ptr()))]=result;return result;})
    .def_property_readonly("X",[](BatchStorage &p){return p.expose_dense();})
    .def_property_readonly("_batch_size",&BatchStorage::size)
    .def_property_readonly("is_view",[](BatchStorage &p){return !p.dense_ready;})
    .def_property_readonly("_batch_X",[](BatchStorage &p){return p.dense();})
    .def_property_readonly("F",[](BatchStorage &p)->py::object{p.ensure_values();p.external=true;return p.evaluated?py::object(p.f):py::none();})
    .def_property_readonly("_batch_F",[](BatchStorage &p)->py::object{p.ensure_values();p.external=true;return p.evaluated?py::object(p.f):py::none();})
    .def_property_readonly("objectives",[](BatchStorage &p)->py::object{p.ensure_values();p.external=true;return p.evaluated?py::object(p.objectives):py::none();})
    .def_property_readonly("constraints",[](BatchStorage &p)->py::object{p.ensure_values();p.external=true;return p.evaluated?py::object(p.constraints):py::none();})
    .def_property_readonly("valid",[](BatchStorage &p){p.ensure_values();p.external=true;return p.valid;})
    .def_property_readonly("row_ids",[](BatchStorage &p){p.ensure_values();p.external=true;return p.ids;})
    .def_property_readonly("auxiliary",[](BatchStorage &p){py::list a;for(auto &r:p.rows)a.append(r?r->auxiliary:py::none());return py::tuple(a);})
    .def("__len__",&BatchStorage::size)
    .def("take",[](BatchStorage &p,py::object indices){return p.take(indices);})
    .def("to_solutions",&BatchStorage::solutions)
    .def("to_population",&BatchStorage::to_population);
  population_storage_type=reinterpret_cast<PyTypeObject *>(storage.ptr());
  py::class_<BatchCore>(m,"RunSession")
    .def(py::init([math](py::object solve,py::sequence paths){return std::make_unique<BatchCore>(math,solve,paths);}))
    .def("close",&BatchCore::close)
    .def("bind_cpu_objective",[](BatchCore &c,py::object rows) {
      if(c.closed||c.control->progress().running)throw std::runtime_error("Bind objective only on an idle open session");
      c.evaluation.bind_cpu(std::move(rows));
    })
    .def("bind_cpu_arrays",[](BatchCore &c,py::object callback,int64_t constraints,uintptr_t address,uintptr_t context) {
      if(c.closed||c.control->progress().running)throw std::runtime_error("Bind objective only on an idle open session");
      c.evaluation.bind_cpu(callback,true,constraints,address,context);
    },"callback"_a,"constraints"_a=0,"address"_a=0,"context"_a=0)
    .def("configure_data_plan",[](BatchCore &c,const std::string &mode,py::object model,uint64_t limit,unsigned maximum,bool dump) {
      if(c.closed||c.control->progress().running)throw std::runtime_error("Configure data planning only on an idle open session");
      if(mode!="dense"&&mode!="rules"&&mode!="graph")throw py::value_error("Data plan mode must be dense, rules or graph");
      if(maximum>8)throw py::value_error("At most eight free data-plan decisions are supported");
      DataPlanner planner;planner.mode=mode;planner.memory_limit=limit;planner.max_decisions=maximum;planner.dump=dump;
      c.data_cost_loader=py::none();c.data_cost_loader_attempted=true;
      if(!model.is_none()) {if(py::isinstance<DataCostModel>(model))planner.model=py::cast<std::shared_ptr<DataCostModel>>(model);else{auto parsed=std::make_shared<DataCostModel>();parsed->load(py::cast<py::dict>(model));planner.model=parsed;}}
      // The existing dense fallback already admits private native rows. A
      // missing optional cost table must not force extra gathers and merges.
      else if(mode=="graph")planner.mode="dense";
      c.planner=std::move(planner);
    },"mode"_a="graph","cost_model"_a=py::none(),"memory_limit"_a=32u*1024u*1024u,"max_decisions"_a=8,"dump"_a=false)
    .def_property_readonly("data_plan_eligible",[](BatchCore &c){return c.data_plan_candidate;})
    .def("_defer_data_plan_costs",[](BatchCore &c,py::object loader,bool dump) {
      if(c.closed||c.control->progress().running)throw std::runtime_error("Configure data planning only on an idle open session");
      c.planner=DataPlanner{};c.planner.dump=dump;c.data_cost_loader=std::move(loader);c.data_cost_loader_attempted=false;
    },"loader"_a,"dump"_a=false)
    .def("data_plan_dump",[](BatchCore &c){if(c.control->progress().running)throw std::runtime_error("Plan dump requires a safe boundary");py::list result;for(auto &r:c.planner.records)result.append(r);return result;})
    .def_property_readonly("closed",[](BatchCore &c){return c.closed;})
    .def("__enter__",[](py::object self){auto &c=py::cast<BatchCore&>(self);if(c.closed)throw std::runtime_error("RunSession is closed");return self;})
    .def("__exit__",[](BatchCore &c,py::object,py::object,py::object){c.close();return false;})
    .def("component",[](BatchCore &c,const std::string &name,BatchPtr p,py::object problem,py::object parameter,py::dict state,int64_t generation,py::object data,py::object count,py::object archive) {
      if(c.closed)throw std::runtime_error("RunSession is closed");
      RunningScope running(*c.control);c.constructor_admission_unused=false;ProfileContext profile(&c.profile);WorkspaceContext workspace(&c.workspace);ArrayWorkspaceContext arrays(&c.array_workspace);
      auto out=c.component(name,p,problem,parameter,NativeState(state,true),generation,data,count,nullptr,py::none(),archive);
      return py::make_tuple(out.population,out.state.python(),out.indices,out.statistics,out.invalidated);
    },"name"_a,"population"_a,"problem"_a,"parameter"_a=py::none(),"state"_a=py::dict(),"generation"_a=0,"data"_a=py::none(),"count"_a=py::none(),"archive"_a=py::none())
    .def("generation",[](BatchCore &c,py::sequence paths,py::sequence params,py::object solutions,py::list aux,py::object problem,py::object data,py::object setting,int64_t generation,int64_t remaining,py::sequence names,py::list archives,py::object budget) {
      if(c.closed)throw std::runtime_error("RunSession is closed");
      RunningScope running(*c.control);c.constructor_admission_unused=false;ProfileContext profile(&c.profile);WorkspaceContext workspace(&c.workspace);ArrayWorkspaceContext arrays(&c.array_workspace);
      auto states=import_auxiliary(aux);
      auto archive_states=import_archives(archives);
      auto out=c.generation(paths,params,solutions,states,problem,data,setting,generation,remaining,names,archive_states,budget);
      auto exported=export_auxiliary(states);for(size_t i=0;i<exported.size();++i)aux[i]=exported[i];
      auto exported_archives=export_archives(archive_states);for(size_t i=0;i<exported_archives.size();++i)archives[i]=exported_archives[i];
      py::list best;if(out.best)best.append(out.best->materialize(c.solution_type));
      return py::make_tuple(c.public_solutions(true),aux,out.used,archives,best,out.events);
    })
    .def("evaluate_batch",[](BatchCore &c,py::array decisions,BatchPtr sources,py::object problem,py::object data,int64_t remaining,py::object invalidated) {
      if(c.closed)throw std::runtime_error("RunSession is closed");
      RunningScope running(*c.control);c.constructor_admission_unused=false;ProfileContext profile(&c.profile);WorkspaceContext workspace(&c.workspace);ArrayWorkspaceContext arrays(&c.array_workspace);
      int64_t used=0;bool interrupted=false;
      auto out=c.changed(decisions,sources,problem,data,remaining,invalidated,used,interrupted);
      // The Python reference only consumes the minimum observed row. Keeping
      // that row preserves its identity without materializing all individuals.
      py::list observed;auto best=BatchCore::minimum(out.second);
      if(best)observed.append(best->materialize(c.solution_type));
      return py::make_tuple(out.first?py::cast(out.first):py::none(),used,observed);
    })
    .def("run",&BatchCore::run)
    .def("snapshot",[](BatchCore &c){
      if(c.control->progress().running)throw std::runtime_error("Snapshot requires a completed safe run boundary");
      if(!c.last_state)throw std::runtime_error("No run state is available");
      RunningScope running(*c.control);auto source=c.last_state;
      WorkspaceContext workspace(nullptr);ArrayWorkspaceContext arrays(nullptr);
      ProfileContext context(&c.profile);RegionScope timing(Region::Snapshot);return source->clone();})
    .def("restore",[](BatchCore &c,py::dict context,std::shared_ptr<RunSnapshot> snapshot){
      auto request=py::reinterpret_steal<py::dict>(PyDict_Copy(context.ptr()));request["native_snapshot"]=py::cast(snapshot);
      return c.run(request);})
    .def("run_many",&BatchCore::run_many,"contexts"_a,"on_result"_a=py::none(),"collect_results"_a=true)
    .def_property_readonly("control",[](BatchCore &c){return c.control;})
    .def("set_audit_hook",[](BatchCore &c,py::object hook){if(c.closed)throw std::runtime_error("RunSession is closed");if(c.control->progress().running)throw std::runtime_error("Cannot change audit hook during a run");c.audit_hook=std::move(hook);})
    .def("configure_diagnostics",[](BatchCore &c,bool enabled){
      if(c.control->progress().running)throw std::runtime_error("Cannot reconfigure diagnostics during a run");
      c.diagnostics=enabled;c.profile.enabled=enabled;c.storage_metrics->enabled=enabled;c.evaluation.recording=enabled;
      c.workspace.numbers.recording=enabled;c.workspace.indices.recording=enabled;c.workspace.masks.recording=enabled;
      c.array_workspace.recording=enabled;})
    .def("statistics",[](BatchCore &c){
      py::dict entries;for(auto &item:c.entry_counts)entries[py::str(item.first)]=py::dict("calls"_a=item.second.calls,"max_rows"_a=item.second.max_rows);
      py::dict regions;for(size_t i=0;i<size_t(Region::Count);++i){auto value=c.profile.read(Region(i));regions[py::str(region_names[i])]=py::dict("calls"_a=value.calls,"inclusive_ns"_a=value.inclusive_ns,"exclusive_ns"_a=value.exclusive_ns);}
      auto &w=c.workspace;
      py::dict workspace("growths"_a=w.numbers.growths+w.indices.growths+w.masks.growths,
          "reuses"_a=w.numbers.reuses+w.indices.reuses+w.masks.reuses,
          "capacity_bytes"_a=w.numbers.capacity_bytes()+w.indices.capacity_bytes()+w.masks.capacity_bytes(),
          "array_allocations"_a=c.array_workspace.allocations,"array_reuses"_a=c.array_workspace.reuses,
          "array_capacity_bytes"_a=c.array_workspace.capacity_bytes,
          "bit_table_builds"_a=w.bit_table_builds,"bit_table_hits"_a=w.bit_table_hits,
          "bit_table_bytes"_a=w.bit_weights.capacity()*sizeof(double));
      if(!c.diagnostics)for(auto key:{"growths","reuses","array_allocations","array_reuses"})workspace[py::str(key)]=py::none();
      return py::dict("data_plan"_a=c.planner.statistics(),"snapshot_copy_bytes"_a=c.profile.snapshot_copy_bytes.load(),"regions"_a=regions,"workspace"_a=workspace,"repair_data"_a=c.evaluation.stats(),"entry_bindings"_a=c.entry_bindings,"direct_submissions"_a=c.direct_submissions,"storage"_a=py::dict("evaluation_pack_bytes"_a=c.storage_metrics->evaluation_pack_bytes,"metadata_copy_bytes"_a=c.storage_metrics->metadata_copy_bytes,"gather_bytes"_a=c.storage_metrics->gather_bytes,"snapshot_bytes"_a=c.storage_metrics->snapshot_bytes,
          "row_table_bytes"_a=c.storage_metrics->row_table_bytes,"row_table_reuses"_a=c.storage_metrics->row_table_reuses,"overlay_rows"_a=c.storage_metrics->overlay_rows,"merge_bytes"_a=c.storage_metrics->merge_bytes,"identity_scan_bytes"_a=c.storage_metrics->identity_scan_bytes,"take_views"_a=c.storage_metrics->take_views,"concat_views"_a=c.storage_metrics->concat_views,"gathers"_a=c.storage_metrics->gathers),"enabled"_a=c.diagnostics,"generations"_a=c.generations,"component_calls"_a=c.component_calls,"evaluation_batches"_a=c.evaluation_batches,"components"_a=entries);
    });
}
}

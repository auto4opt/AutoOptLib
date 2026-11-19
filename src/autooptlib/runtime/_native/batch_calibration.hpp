#pragma once
// Explicit research calibration. None of these clocks are on production paths.
namespace native_components {
inline void bind_data_calibration(py::module_ &m,std::shared_ptr<NativeMath> math) {
  m.def("_measure_data_costs",[math](BatchPtr source,const std::string &name,py::object problem,py::object parameter,int64_t repeats,double minimum_ms,bool reverse) {
    if(repeats<1||!source->rows_compatible())throw py::value_error("Calibration requires private rows and positive repetitions");
    RunWorkspace native_workspace;ArrayWorkspace native_arrays;
    WorkspaceContext workspace(&native_workspace);ArrayWorkspaceContext arrays(&native_arrays);
    auto np=py::module_::import("numpy");auto solve=py::module_::import("autooptlib.utils.solve");
    auto measure=[&](auto operation) {
      auto start=std::chrono::steady_clock::now();int64_t count=0,batch=repeats;double elapsed=0;
      do {for(int64_t i=0;i<batch;++i)operation();count+=batch;
        elapsed=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
        batch=std::min<int64_t>(batch*2,1048576);
      }while(elapsed<minimum_ms*1e6);
      return elapsed/count;
    };
    auto gather=measure([&](){auto array=py::array(source->dtype(),std::vector<py::ssize_t>{source->size(),source->dimension()});
      for(py::ssize_t i=0;i<source->size();++i)source->write_row(i,static_cast<char *>(array.mutable_data())+i*source->dimension()*8);});
    auto table=measure([&](){auto rows=BatchStorage::read_rows(source);});
    // Retain original indexed owners and create an independent dense copy.
    auto descriptor=BatchStorage::read_rows(source);auto dense=std::make_shared<BatchStorage>(descriptor.materialize());
    source->ensure_values();
    dense->evaluated=source->evaluated;dense->f=source->f;dense->valid=source->valid;
    dense->objectives=source->objectives;dense->constraints=source->constraints;
    Component<NativeMath> entry{name,math,solve.attr("SolutionSet"),np.attr("random").attr("Generator"),np.attr("random").attr("default_rng")};
    auto kernel=[&](bool indexed) {
      ComponentInvocation call;call.population=indexed?source:dense;call.problem=problem;call.parameter=parameter;
      call.indexed_inputs=indexed;call.state.set("rng",np.attr("random").attr("default_rng")(104729));
      for(int i=0;i<3;++i){entry.execute(call);call.state.erase("_stream_invalidated_mask");}
      return measure([&](){entry.execute(call);call.state.erase("_stream_invalidated_mask");});
    };
    double rows,dense_kernel;
    if(reverse){dense_kernel=kernel(false);rows=kernel(true);}else{rows=kernel(true);dense_kernel=kernel(false);}
    std::vector<int64_t> active(std::max<int64_t>(1,source->size()/2));std::iota(active.begin(),active.end(),0);
    auto varied=source->take(active);
    // An overlay is consumed through a newly built repair-input row map.
    // Include that map in the measured primitive and in its modeled payload.
    auto overlay=measure([&](){auto out=BatchStorage::overlay(source,varied,active);auto rows=BatchStorage::read_rows(out);});
    auto merge=measure([&](){auto out=BatchStorage::merge(source,varied,active);});
    std::vector<int64_t> prefix(std::max<int64_t>(0,source->size()-1));std::iota(prefix.begin(),prefix.end(),0);
    auto crop=measure([&](){auto out=source->take(prefix);});
    auto repair=[&](bool indexed) {
      NativeData data(solve);data.recording=false;BatchAccess access(np);
      data.native_flex=[&](py::object object,const char *key,py::object fallback){return access.flex(object,key,fallback);};
      auto fallback=solve.attr("_batch_evaluate_changed");
      return measure([&](){NativeData::EvaluationResult result;
        auto raw=py::none();auto source_matrix=dense->dense();
        auto compatible=data.changed<true>(indexed?py::object(py::none()):py::object(source_matrix),py::none(),problem,py::none(),source->size()+1,py::bool_(false),fallback,
          false,false,false,true,true,&result,indexed?nullptr:&source_matrix,indexed?&descriptor:nullptr,indexed?&descriptor:nullptr);
        if(!result.available) {
          auto response=py::cast<py::tuple>(compatible);
          if(py::cast<int64_t>(response[1])!=0||py::len(response[2])!=0)throw std::runtime_error("Calibration fallback unexpectedly evaluated an objective");
        }else if(!result.evaluated.empty())throw std::runtime_error("Calibration unexpectedly evaluated an objective");
      });
    };
    double repair_rows,repair_dense;
    if(reverse){repair_dense=repair(false);repair_rows=repair(true);}else{repair_rows=repair(true);repair_dense=repair(false);}
    return py::dict("rows"_a=std::max(0.,rows-table),"dense"_a=dense_kernel,"gather"_a=gather,"row_table"_a=table,
      "repair_rows"_a=repair_rows,"repair_dense"_a=repair_dense,"merge"_a=merge,"overlay"_a=overlay,"crop"_a=crop,
      "measured_rows_total"_a=rows,"repeats"_a=repeats);
  },"source"_a,"name"_a,"problem"_a,"parameters"_a=py::none(),"repeats"_a=1,"minimum_ms"_a=0.,"reverse"_a=false);
  m.def("_measure_data_planner",[math](py::dict model,BatchPtr source,const std::string &name,int64_t repeats,double minimum_ms) {
    if(repeats<1)throw py::value_error("Repetitions must be positive");
    auto costs=std::make_shared<DataCostModel>();costs->load(model);
    auto solve=py::module_::import("autooptlib.utils.solve");BatchCore core(math,solve,py::list());core.planner.model=costs;core.planner.cache_plans=false;
    WorkspaceContext workspace(&core.workspace);ArrayWorkspaceContext arrays(&core.array_workspace);
    auto helpers=py::module_::import("autooptlib.utils.design._helpers");
    auto graph=py::module_::import("autooptlib.utils.design._stream_graph");
    auto parameter=graph.attr("StreamStageParam")(py::none(),helpers.attr("SearchParam")(py::none()));
    auto rng=core.default_rng(104729);NativeState state;state.set("rng",rng);
    StageTemplate plan(name);
    volatile uint64_t observed=0;auto start=std::chrono::steady_clock::now();int64_t total=0,batch=repeats;double elapsed=0;
    do {
      for(int64_t i=0;i<batch;++i) {
        source->layout_class=0;
        if(source->owner_count!=1)source->owner_count=-1;
        observed+=core.plan_data_input(plan,source,std::max<int64_t>(1,source->size()/2),source->size(),parameter,state,rng).rows;
      }
      total+=batch;batch=std::min<int64_t>(batch*2,1048576);
      elapsed=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
    }while(elapsed<minimum_ms*1e6);
    return elapsed/total;
  },"model"_a,"source"_a,"name"_a,"repeats"_a,"minimum_ms"_a=0.);
}
}

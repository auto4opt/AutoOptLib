#pragma once
namespace native_components {
inline BatchPtr initialize_storage(py::object solve,py::object rng,py::object problem,py::object data,py::object setting,std::shared_ptr<StorageMetrics> metrics=nullptr,NativeData *native_data=nullptr) {
  auto np=py::module_::import("numpy");auto flex=solve.attr("get_flex");
  auto type=solve.attr("get_problem_type")(problem);auto kind=type.is_none()?std::string("continuous"):py::cast<std::string>(type);if(kind.empty())kind="continuous";
  auto n=py::cast<int64_t>(flex(setting,"_AlgorithmPopulationSize",flex(problem,"N",flex(setting,"ProbN",10))));
  auto supplied=flex(setting,"_InitialPopulation",py::none());auto decisions=py::reinterpret_borrow<py::array>(py::handle());
  if(!supplied.is_none()) {
    auto x=py::array::ensure(supplied);
    if(!x||x.ndim()!=2)throw py::value_error("Each supplied initial population must be a 2-D array.");
    auto bound=py::array::ensure(flex(problem,"bound"));if(!bound||bound.ndim()==0)throw py::index_error("tuple index out of range");auto d=bound.shape(bound.ndim()-1);
    if(x.shape(1)!=d)throw py::value_error("Supplied initial-population dimension does not match the problem.");
    if(x.shape(0)<n)throw py::value_error("A supplied initial population must contain at least ProbN rows.");
    decisions=py::cast<py::array>(np.attr("array")(x[py::slice(0,n,1)],"copy"_a=true));
  }else if(kind=="continuous") {
    auto b=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(flex(problem,"bound"));if(b.ndim()!=2||b.shape(0)!=2)throw py::value_error("Problem bounds must have shape (2, dimension)");auto d=b.shape(1);
    py::array_t<double> random(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(d)});{RandomBoundary boundary(rng);boundary.random.uniform(random.mutable_data(),n*d);}
    // Ufuncs preserve the reference's per-step rounding and error policy.
    auto lower=py::array(b.dtype(),{d},{py::ssize_t(8)},b.data(),b);
    auto upper=py::array(b.dtype(),{d},{py::ssize_t(8)},b.data()+d,b);
    decisions=py::cast<py::array>(np.attr("add")(lower,np.attr("multiply")(np.attr("subtract")(upper,lower),random)));
  }else if(kind=="discrete") {
    auto b=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(flex(problem,"bound"));if(b.ndim()!=2||b.shape(0)!=2)throw py::value_error("Problem bounds must have shape (2, dimension)");auto d=b.shape(1);
    py::array_t<int64_t> x(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(d)});auto column=workspace_array<int64_t>(n);
    {RandomBoundary boundary(rng);for(py::ssize_t j=0;j<d;++j) {
      if(b.data()[d+j]==std::numeric_limits<int64_t>::max())throw py::value_error("high is out of bounds for int64");
      boundary.random.integers(b.data()[j],b.data()[d+j]+1,column.mutable_data(),n);
      for(int64_t i=0;i<n;++i)x.mutable_data()[i*d+j]=column.data()[i];
    }}decisions=x;
  }else if(kind=="permutation") {
    auto b=py::array::ensure(flex(problem,"bound"));int64_t d=(b.ndim()==2&&b.shape(1)>0)?b.shape(1):py::cast<int64_t>(flex(problem,"dimension",flex(problem,"D",0)));
    if(!d)throw py::value_error("Permutation problem requires explicit dimension or bound");
    py::array_t<int64_t> x(std::vector<py::ssize_t>{static_cast<py::ssize_t>(n),static_cast<py::ssize_t>(d)});{RandomBoundary boundary(rng);for(int64_t i=0;i<n;++i){auto row=x.mutable_data()+i*d;std::iota(row,row+d,1);boundary.random.permutation(row,d);}}decisions=x;
  }else{PyErr_SetString(PyExc_NotImplementedError,("Unsupported problem type: "+kind).c_str());throw py::error_already_set();}
  bool native_initial=native_data&&Py_TYPE(problem.ptr())==reinterpret_cast<PyTypeObject *>(native_data->namespace_type.ptr());
  if(native_initial) {
    for(auto item:py::cast<py::tuple>(solve.attr("_NATIVE_INITIAL_POLICY"))) {
      auto pair=py::cast<py::tuple>(item);auto function=py::getattr(solve,py::cast<std::string>(pair[0]).c_str());
      native_initial&=function.ptr()==pair[1].ptr()&&PyFunction_Check(function.ptr())&&PyFunction_GetCode(function.ptr())==pair[2].ptr();
    }
    auto dictionary=py::getattr(problem,"__dict__");for(auto key:dictionary)native_initial&=PyUnicode_CheckExact(key.ptr());
    if(native_initial) {
      auto behavior=field(problem,"type");native_initial&=PyList_CheckExact(behavior.ptr());
      if(native_initial)for(auto item:py::cast<py::list>(behavior))native_initial&=PyUnicode_CheckExact(item.ptr());
      auto setting=field(problem,"setting");native_initial&=setting.is_none()||PyUnicode_CheckExact(setting.ptr());
      native_initial&=PyArray_CheckExact(field(problem,"bound").ptr());
    }
    if(native_initial) {
      auto strategy=native_data->read_flex(problem,"_AlgorithmBoundaryHandling",py::str("clip"));
      native_initial&=PyUnicode_CheckExact(strategy.ptr());
    }
  }
  if(native_initial) {
    NativeData::EvaluationResult result;
    py::array empty(decisions.dtype(),std::vector<py::ssize_t>{0,decisions.shape(1)});
    auto fallback=py::cpp_function([solve](py::args args,py::kwargs) {
      RegionScope timing(Region::Callback);
      auto raw=py::cast<py::tuple>(solve.attr("_evaluate_decisions")(args[0],args[2],args[3]));
      auto inputs=py::cast<py::list>(raw[0]);py::list resolved;for(size_t i=0;i<inputs.size();++i)resolved.append(py::none());
      return py::make_tuple(resolved,inputs.size(),inputs,raw[1],false);
    });
    auto compatibility=native_data->changed(decisions,py::none(),problem,data,decisions.shape(0)+1,py::bool_(true),fallback,
        false,false,false,true,true,&result,&empty);
    if(!result.available) {
      auto raw=py::cast<py::tuple>(compatibility);auto inputs=py::cast<py::list>(raw[2]),outputs=py::cast<py::list>(raw[3]);
      for(size_t i=0;i<inputs.size();++i){auto value=py::cast<py::tuple>(outputs[i]);result.evaluated.push_back({py::cast<py::array>(inputs[i]),py::cast<double>(value[0]),py::cast<double>(value[1]),py::reinterpret_borrow<py::object>(value[2])});}
    }
    std::vector<RowPtr> rows;rows.reserve(result.evaluated.size());
    for(auto &row:result.evaluated)rows.push_back(std::make_shared<BatchRow>(std::move(row.decision),row.objective,row.constraint,
        row.constraint<=0?row.objective:row.constraint+1e8,std::move(row.auxiliary)));
    return BatchStorage::from_rows(rows,decisions.shape(1),decisions.dtype().num()==NPY_INT64,metrics);
  }
  py::tuple evaluated;
  {RegionScope callback(Region::Callback);evaluated=py::cast<py::tuple>(solve.attr("_evaluate_decisions")(decisions,problem,data));}
  auto inputs=py::cast<py::list>(evaluated[0]),results=py::cast<py::list>(evaluated[1]);std::vector<RowPtr> rows;
  for(size_t i=0;i<inputs.size();++i){auto r=py::cast<py::tuple>(results[i]);auto o=py::cast<double>(r[0]),c=py::cast<double>(r[1]);rows.push_back(std::make_shared<BatchRow>(py::cast<py::array>(inputs[i]),o,c,c<=0?o:c+1e8,py::reinterpret_borrow<py::object>(r[2])));}
  return BatchCore::from_rows(rows,decisions.shape(1),decisions.dtype().num()==NPY_INT64,metrics);
}
inline py::object initialize_population(py::object solve,py::object rng,py::object problem,py::object data,py::object setting) {
  return initialize_storage(solve,rng,problem,data,setting)->solutions(solve.attr("Solution"),solve.attr("SolutionSet"));
}

inline py::dict BatchCore::run(py::dict c) {
  if(closed)throw std::runtime_error("RunSession is closed");
  RunningScope running(*control);
  if(!control->cancelled())admit_context(c);
  return run_impl(std::move(c));
}
inline py::dict BatchCore::run_impl(py::dict c) {
  ProfileContext profiling(&profile);
  WorkspaceContext workspace_scope(&workspace);
  ArrayWorkspaceContext array_workspace_scope(&array_workspace);
  RegionScope run_timing(Region::Run);
  current.reset();public_rows.clear();last_state.reset();
  control->publish(0,0,0);
  if(control->cancelled()) {
    return py::dict("solutions"_a=set_type(py::list()),"history"_a=py::list(),"fit_history"_a=py::list(),
        "evaluations"_a=0,"elapsed"_a=0.,"archives"_a=py::list(),"best_solution"_a=py::none(),
        "stream_events"_a=0,"stream_event_budget"_a=c["stream_event_budget"],
        "termination_reason"_a="cancelled","execution_semantics"_a="population_batch_v1","execution_backend"_a="cpp");
  }
  const bool managed=c.contains("resume_state");
  std::shared_ptr<RunSnapshot> restored;
  if(c.contains("native_snapshot")) {
    auto snapshot=py::cast<std::shared_ptr<RunSnapshot>>(c["native_snapshot"]);
    if(!snapshot->signature.is_none()&&c.contains("checkpoint_signature")) {
      int same=PyObject_RichCompareBool(snapshot->signature.ptr(),c["checkpoint_signature"].ptr(),Py_EQ);
      if(same<0)throw py::error_already_set();if(!same)throw py::value_error("Native snapshot does not match this search context");
    }
    restored=snapshot->clone(c["rng"]);
    py::reinterpret_borrow<py::object>(c["rng"]).attr("bit_generator").attr("state")=restored->rng_state;
  }
  bool loaded_complete=false;BatchPtr initial;
  if(managed) {
    auto state=py::reinterpret_borrow<py::object>(c["resume_state"]);
    auto setting=py::reinterpret_borrow<py::object>(c["setting"]);
    auto flex=solve.attr("get_flex");
    auto names=py::cast<py::sequence>(c["archive_names"]);
    if(restored) {
      loaded_complete=restored->complete;initial=restored->population;
      c["solutions"]=py::none();c["auxiliary"]=py::list();
      c["archives"]=py::list();
      c["history"]=py::list();c["fit_history"]=py::list();c["evaluations"]=restored->evaluations;
      c["elapsed_seconds"]=restored->elapsed;c["generation"]=restored->generation;c["stream_events"]=restored->events;
      c["termination_reason"]=loaded_complete?restored->reason:"completed";
    }else if(state.is_none()) {
      {RegionScope initialization(Region::Initialize);
      initial=initialize_storage(solve,c["rng"],c["problem"],c["data"],setting,storage_metrics,&evaluation);}
      c["solutions"]=py::none();py::list aux,archives;
      for(py::ssize_t i=0;i<py::len(c["pathways"]);++i)aux.append(py::dict());
      for(auto name:names)archives.append(py::cast<std::string>(name)=="archive_statistic"?py::object(py::array_t<double>(std::vector<py::ssize_t>{0,2})):py::object(py::list()));
      c["auxiliary"]=aux;c["archives"]=archives;c["history"]=py::list();c["fit_history"]=py::list();
      c["evaluations"]=initial->size();c["elapsed_seconds"]=0.;c["generation"]=1;c["stream_events"]=0;c["termination_reason"]="completed";
    }else {
      auto resume=py::cast<py::dict>(state);loaded_complete=py::cast<bool>(resume["complete"]);
      c["solutions"]=solve.attr("SolutionSet")(resume["solutions"]);
      c["auxiliary"]=resume["aux_cache"];c["archives"]=resume["archives"];
      for(auto key:{"history","fit_history","evaluations","elapsed_seconds","generation"})c[py::str(key)]=resume[py::str(key)];
      c["stream_events"]=resume.contains("stream_events")?resume["stream_events"]:py::object(py::int_(0));
      c["termination_reason"]=loaded_complete&&resume.contains("termination_reason")?resume["termination_reason"]:py::object(py::str("completed"));
    }
    auto metric=flex(setting,"metric",py::str("quality"));auto kind=py::cast<std::string>(metric);
    auto raw_time=flex(setting,"tmax",py::float_(INFINITY));double limit=raw_time.is_none()?INFINITY:py::cast<double>(raw_time);
    auto raw_threshold=flex(setting,"thres",py::float_(-INFINITY));double threshold=raw_threshold.is_none()?-INFINITY:py::cast<double>(raw_threshold);
    c["metric"]=metric;c["Tmax"]=kind=="runtimeSec"?limit:kind=="runtimeFE"?std::min<double>(py::cast<int64_t>(c["evaluation_budget"]),std::isfinite(limit)?std::trunc(limit):py::cast<int64_t>(c["evaluation_budget"])):INFINITY;
    c["threshold"]=kind=="auc"?-INFINITY:threshold;
    auto aux=py::cast<py::list>(c["auxiliary"]);
    c["zero_generations"]=restored?py::object(py::int_(restored->zero)):aux.size()&&PyDict_Check(aux[0].ptr())&&py::cast<py::dict>(aux[0]).contains("_stream_zero_evaluation_generations")?aux[0]["_stream_zero_evaluation_generations"]:py::object(py::int_(0));
  }
  auto paths=py::cast<py::sequence>(c["pathways"]),params=py::cast<py::sequence>(c["parameters"]);
  auto solutions=py::reinterpret_borrow<py::object>(c["solutions"]);
  auto aux=py::cast<py::list>(c["auxiliary"]),archives=py::cast<py::list>(c["archives"]);
  auto native_aux=restored?restored->auxiliary:import_auxiliary(aux);
  auto native_archives=restored?restored->archives:import_archives(archives);
  auto names=py::cast<py::sequence>(c["archive_names"]);
  auto problem=py::reinterpret_borrow<py::object>(c["problem"]),data=py::reinterpret_borrow<py::object>(c["data"]),setting=py::reinterpret_borrow<py::object>(c["setting"]);
  auto metric=py::cast<std::string>(c["metric"]);double tmax=py::cast<double>(c["Tmax"]),threshold=py::cast<double>(c["threshold"]);
  auto budget=py::cast<int64_t>(c["evaluation_budget"]),evaluations=py::cast<int64_t>(c["evaluations"]),gen=py::cast<int64_t>(c["generation"]),events=py::cast<int64_t>(c["stream_events"]);
  auto raw_event_budget=py::reinterpret_borrow<py::object>(c["stream_event_budget"]);
  int64_t event_limit=raw_event_budget.is_none()?std::numeric_limits<int64_t>::max():py::cast<int64_t>(raw_event_budget);
  int64_t zero=py::cast<int64_t>(c["zero_generations"]),zero_limit=py::cast<int64_t>(solve.attr("STREAM_MAX_ZERO_EVALUATION_GENERATIONS"));
  double elapsed=py::cast<double>(c["elapsed_seconds"]);auto reason=py::cast<std::string>(c["termination_reason"]);
  auto checkpoint=py::reinterpret_borrow<py::object>(c["checkpoint"]);auto every=py::cast<int64_t>(c["checkpoint_every"]);
  std::vector<RowPtr> history;std::vector<double> fits;
  for(auto value:py::cast<py::list>(c["history"])) {
    if(value.is_none())history.push_back(nullptr);
    else history.push_back(std::make_shared<BatchRow>(py::cast<py::array>(field(value,"dec")),py::cast<double>(field(value,"obj")),py::cast<double>(field(value,"con")),py::cast<double>(field(value,"fit")),field(value,"acc"),py::reinterpret_borrow<py::object>(value)));
  }
  for(auto f:py::cast<py::list>(c["fit_history"]))fits.push_back(py::cast<double>(f));
  if(restored){history=restored->history;fits=restored->fits;}
  current=initial?initial:BatchStorage::from_solutions(solutions);current->metrics=storage_metrics;public_rows=current->rows;
  if(history.empty()) {auto best=minimum(current->rows);history.push_back(best);fits.push_back(best?best->fit:INFINITY);}
  auto clock=c.contains("clock")?py::reinterpret_borrow<py::object>(c["clock"]):py::object(py::none());
  auto now=[&](){return clock.is_none()?std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count():py::cast<double>(clock());};
  auto public_history=[&](){py::list h;for(auto &row:history)h.append(row?row->materialize(solution_type):py::none());return h;};
  auto public_fits=[&](){py::list f;for(auto value:fits)f.append(value);return f;};
  auto capture=[&](bool complete) {
    auto state=std::make_shared<RunSnapshot>();state->population=current;state->public_rows=public_rows;
    state->auxiliary=native_aux;state->archives=native_archives;state->history=history;state->fits=fits;
    state->rng=c["rng"];state->rng_state=state->rng.attr("bit_generator").attr("state");
    if(c.contains("checkpoint_signature"))state->signature=c["checkpoint_signature"];
    state->evaluations=evaluations;state->elapsed=elapsed;state->generation=gen;state->events=events;
    state->zero=zero;state->complete=complete;state->reason=complete?reason:"running";return state;
  };
  std::shared_ptr<RunSnapshot> pending_checkpoint;
  auto snapshot=[&](bool complete)->py::object {
    RegionScope timing(Region::Snapshot);
    if(c.contains("native_checkpoint")&&py::cast<bool>(c["native_checkpoint"])) {
      pending_checkpoint=capture(complete);return py::cast(pending_checkpoint);
    }
    py::dict state;
    state["solutions"]=public_solutions();aux=export_auxiliary(native_aux);state["aux_cache"]=aux;archives=export_archives(native_archives);state["archives"]=archives;
    state["history"]=public_history();state["fit_history"]=public_fits();state["evaluations"]=evaluations;
    state["elapsed_seconds"]=elapsed;state["generation"]=gen;state["rng"]=c["rng"];
    state["complete"]=complete;state["stream_events"]=events;state["termination_reason"]=complete?reason:"running";
    return state;
  };
  auto refresh_checkpoint=[&]() {
    if(pending_checkpoint) {
      if(pending_checkpoint->boundary_auxiliary)aux=*pending_checkpoint->boundary_auxiliary;
      else aux=export_auxiliary(native_aux);
      if(pending_checkpoint->boundary_archives)archives=*pending_checkpoint->boundary_archives;
      else archives=export_archives(native_archives);
    }
    native_aux=import_auxiliary(aux);native_archives=import_archives(archives);pending_checkpoint.reset();
  };
  while(true) {
    control->publish(evaluations,events,gen);
    cooperative_boundary();
    if(control->cancelled()){reason="cancelled";break;}
    if(PyErr_CheckSignals()<0)throw py::error_already_set();
    if(loaded_complete||evaluations>=budget)break;
    if(events>=event_limit){reason="stream_event_budget_exhausted";break;}
    if(metric=="quality"&&fits.back()<=threshold)break;
    if(metric=="runtimeFE"&&(evaluations>=tmax||fits.back()<=threshold))break;
    if(metric=="runtimeSec"&&elapsed>=tmax)break;
    auto start=metric=="runtimeSec"||!clock.is_none()?now():0.;
    int64_t remaining=budget-evaluations;if(metric=="runtimeFE")remaining=std::min(remaining,static_cast<int64_t>(tmax)-evaluations);
    auto result=generation(paths,params,py::none(),native_aux,problem,data,setting,gen,remaining,names,native_archives,
                            raw_event_budget.is_none()?py::object(py::none()):py::object(py::int_(event_limit-events)));
    current=result.population;evaluations+=result.used;events+=result.events;
    if(events>=event_limit&&evaluations<budget)reason="stream_event_budget_exhausted";
    if(result.best) {
      auto previous=history.empty()?nullptr:history.back();if(previous)previous->refresh_record();auto best=(!previous||result.best->fit<previous->fit)?result.best:previous;
      history.push_back(best);fits.push_back(best->fit);
    }
    if(result.cancelled) {
      reason="cancelled";
      if(metric=="runtimeSec")elapsed+=now()-start;
      break;
    }
    if(result.used==0) {
      if(!result.best)break;
      ++zero;if(native_aux.size())native_aux[0].set("_stream_zero_evaluation_generations",zero);
      if(zero>=zero_limit)break;
    }else{
      zero=0;if(native_aux.size())native_aux[0].set("_stream_zero_evaluation_generations",int64_t(0));
    }
    if(events>=event_limit&&evaluations<budget){reason="stream_event_budget_exhausted";break;}
    if(metric=="runtimeSec")elapsed+=now()-start;
    ++gen;
    if(!checkpoint.is_none()) {
      if(every==0){PyErr_SetString(PyExc_ZeroDivisionError,"integer modulo by zero");throw py::error_already_set();}
      if(gen%every==0) {
        RegionScope timing(Region::Checkpoint);
        auto observed=checkpoint(snapshot(false));
        refresh_checkpoint();
        bool observable=!PyBool_Check(observed.ptr())||py::cast<bool>(observed);
        for(auto &row:public_rows)if(row){row->escaped|=observable;current->private_row(row);}
        for(auto &row:history)if(row){row->escaped|=observable;current->private_row(row);}
      }
    }
  }
  control->publish(evaluations,events,gen);
  if(!loaded_complete&&!checkpoint.is_none()){
    RegionScope timing(Region::Checkpoint);checkpoint(snapshot(true));
    refresh_checkpoint();
  }
  auto best=minimum(history);
  last_state=capture(reason!="cancelled");last_state->reason=reason;archives=export_archives(native_archives);
  return py::dict("solutions"_a=public_solutions(),"history"_a=public_history(),"fit_history"_a=public_fits(),
      "evaluations"_a=evaluations,"elapsed"_a=elapsed,"archives"_a=archives,
      "best_solution"_a=best?best->materialize(solution_type):py::none(),"stream_events"_a=events,
      "stream_event_budget"_a=raw_event_budget,"termination_reason"_a=reason,
      "execution_semantics"_a="population_batch_v1","execution_backend"_a="cpp");
}

inline py::list BatchCore::run_many(py::iterable contexts,py::object on_result,bool collect_results) {
  if(closed)throw std::runtime_error("RunSession is closed");
  RunningScope running(*control);
  py::list output;
  if(control->cancelled())return output;
  auto iterator=py::iter(contexts);int64_t index=0;
  while(!control->cancelled()) {
    if(PyErr_CheckSignals()<0)throw py::error_already_set();
    auto raw=PyIter_Next(iterator.ptr());
    if(!raw){if(PyErr_Occurred())throw py::error_already_set();break;}
    auto context=py::reinterpret_steal<py::object>(raw);
    auto c=py::cast<py::dict>(context);
    admit_context(c);
    auto result=run_impl(c);control->completed_run();
    if(collect_results)output.append(result);
    if(!on_result.is_none()){RegionScope timing(Region::Callback);on_result(result,index);}
    ++index;
  }
  return output;
}

}

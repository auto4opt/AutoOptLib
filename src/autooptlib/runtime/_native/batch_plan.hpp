#pragma once
#include <array>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace native_components {
enum class LayoutKind { Dense, Indexed, Segmented, Overlay };
enum class InputNeed { MetadataOnly, Rows, Dense };
enum class PlanOp { Select, Take, ActiveTake, Variation, Overlay, RepairEvaluate,
                    RouteRestore, Join, Update, Archive, EnsureDense, ReleaseDerived };
struct InputCapability {
  InputNeed need=InputNeed::Dense;
  bool indexed_supported=false,ordered_read_only=true,population_context=false,stateful=false;
  std::string family="dense";
};
// Names are enumerated ONLY by the existing component registry. Unknown kernels
// conservatively require Dense; this is a capability policy, not a second registry.
inline InputCapability input_capability(const std::string &name) {
  InputCapability c;
  if(name.rfind("choose_",0)==0||name.rfind("update_",0)==0||name=="archive_best"||name=="archive_statistic"||name=="archive_tabu"||name.rfind("para_",0)==0) {
    c.need=(name=="choose_nich"||name=="choose_brainstorm")?InputNeed::Dense:InputNeed::MetadataOnly;
    c.family=c.need==InputNeed::MetadataOnly?"metadata":"dense";
    c.stateful=name.rfind("para_",0)==0;return c;
  }
  if(name.rfind("cross_point_",0)==0||name.rfind("cross_fastga_",0)==0)c.family="point";
  else if(name.rfind("cross_order_",0)==0)c.family="permutation";
  else if(name=="cross_arithmetic"||name=="cross_sim_binary")c.family="continuous_cross";
  else if(name.find("_de_")!=std::string::npos){c.family="de";c.population_context=true;}
  else if(name=="search_pso"){c.family="pso";c.population_context=true;c.stateful=true;}
  else if(name.rfind("search_mu_",0)==0){c.family="continuous";c.stateful=name=="search_mu_gaussian"||name=="search_mu_cauchy";}
  else if(name.find("_bit_")!=std::string::npos)c.family="bit";
  else if(name.rfind("search_reset_",0)==0)c.family="discrete";
  else if(name=="search_swap"||name=="search_swap_multi"||name=="search_insert"||name=="search_scramble"||name=="reinit_permutation")c.family="permutation";
  else if(name=="reinit_continuous")c.family="continuous";
  else if(name=="reinit_discrete")c.family="discrete";
  if(c.family!="dense"){c.need=InputNeed::Rows;c.indexed_supported=true;}
  return c;
}
inline const char *need_name(InputNeed n) {
  return n==InputNeed::Rows?"rows":n==InputNeed::Dense?"dense":"metadata";
}
struct ValueSpec {
  uint32_t id,producer,last_use;
  std::vector<uint32_t> consumers;
  bool observable=false;
};
struct PlannedInput {uint32_t value;InputNeed need;};
struct PlannedStep {PlanOp op;std::vector<PlannedInput> inputs;uint32_t output;};
struct StagePlanCache;
struct StageTemplate {
  std::string kernel;
  InputCapability capability;
  std::vector<ValueSpec> values;
  std::vector<PlannedStep> steps;
  mutable std::shared_ptr<StagePlanCache> cache;
  explicit StageTemplate(const std::string &name):kernel(name),capability(input_capability(name)) {
    // Def-use covers the actual stage scope. Evaluation is an effect boundary;
    // future algorithm getters/parameters are never read to build this template.
    steps={{PlanOp::Take,{},0},{PlanOp::ActiveTake,{{0,InputNeed::Rows}},1},
           {PlanOp::Variation,{{1,capability.need}},2},
           {PlanOp::Overlay,{{0,InputNeed::Rows},{2,InputNeed::Rows}},3},
           {PlanOp::RepairEvaluate,{{3,InputNeed::Rows},{0,InputNeed::Rows}},4}};
    for(uint32_t i=0;i<5;++i)values.push_back({i,i,i,{},i==4});
    for(uint32_t i=0;i<steps.size();++i)for(auto input:steps[i].inputs) {
      values[input.value].consumers.push_back(i);values[input.value].last_use=i;
    }
  }
};

// Generic bounded planner. Lifetimes, not summed allocations, determine peak
// derived memory. The same implementation is independently exhaustively tested.
struct BufferLife {uint32_t first,last;uint64_t bytes;};
struct PlanAlternative {double cost;std::vector<BufferLife> buffers;bool legal=true;};
struct PlanSolution {uint32_t mask=0;double cost=INFINITY;uint64_t peak=0;bool bounded_fallback=false,found=false;};
inline uint64_t peak_bytes(const std::vector<BufferLife> &buffers) {
  uint32_t last=0;for(auto &b:buffers)last=std::max(last,b.last);
  uint64_t peak=0;
  for(uint32_t step=0;step<=last;++step){uint64_t live=0;for(auto &b:buffers)if(b.first<=step&&step<=b.last)live+=b.bytes;peak=std::max(peak,live);}
  return peak;
}
template<class Cost> PlanSolution bounded_plan(unsigned decisions,uint64_t limit,unsigned max_decisions,Cost cost,uint32_t linear_choice=0) {
  PlanSolution best;best.bounded_fallback=decisions>max_decisions||decisions>8;
  uint32_t count=best.bounded_fallback?1:1u<<decisions;
  for(uint32_t candidate=0;candidate<count;++candidate) {
    uint32_t mask=best.bounded_fallback?linear_choice:candidate;
    auto alternative=cost(mask);if(!alternative.legal||!std::isfinite(alternative.cost))continue;
    auto peak=peak_bytes(alternative.buffers);if(peak>limit)continue;
    if(!best.found||alternative.cost<best.cost){best={mask,alternative.cost,peak,best.bounded_fallback,true};}
  }
  return best;
}

struct CostSample {
  std::string family,layout;
  bool integer=false;
  int64_t n=0,d=0;
  double rows=0,dense=0,gather=0,row_table=0,repair_rows=0,repair_dense=0,merge=0,overlay=0,crop=0,residual=0,planning_ns=0;
};
struct DataCostModel {
  bool valid=false;
  struct Grid {std::vector<int64_t> ns,ds;std::map<std::pair<int64_t,int64_t>,CostSample> cells;};
  std::unordered_map<std::string,Grid> grids;
  struct Gain {double gather=0,merge=0,repair=0,kernel=0,planning=INFINITY;};
  struct GainGrid {std::vector<int64_t> ns,ds;std::map<std::pair<int64_t,int64_t>,Gain> cells;};
  std::unordered_map<std::string,GainGrid> gain_grids;
  static std::string key(const std::string &family,const std::string &layout,bool integer){return family+":"+layout+(integer?":i":":f");}
  std::vector<CostSample> samples;
  double decision_ns=0,margin=.05;
  void load(py::dict model) {
    if(py::cast<std::string>(model["schema"])!="autooptlib.data-plan-costs"||py::cast<int>(model["version"])!=3)
      throw py::value_error("Unsupported data-plan cost model");
    decision_ns=py::cast<double>(model["decision_ns"]);
    if(!std::isfinite(decision_ns)||decision_ns<0)throw py::value_error("Invalid planner cost");
    samples.clear();grids.clear();gain_grids.clear();
    for(auto item:py::cast<py::list>(model["samples"])) {
      auto row=py::cast<py::dict>(item);CostSample s;
      s.family=py::cast<std::string>(row["family"]);s.layout=py::cast<std::string>(row["layout"]);
      s.integer=py::cast<bool>(row["integer"]);s.n=py::cast<int64_t>(row["n"]);s.d=py::cast<int64_t>(row["d"]);
      for(auto pair:{std::make_pair("rows",&s.rows),{"dense",&s.dense},{"gather",&s.gather},{"row_table",&s.row_table},
                     {"repair_rows",&s.repair_rows},{"repair_dense",&s.repair_dense},{"merge",&s.merge},{"overlay",&s.overlay},{"crop",&s.crop},{"residual",&s.residual}}) {
        *pair.second=py::cast<double>(row[pair.first]);
        if(!std::isfinite(*pair.second)||*pair.second<0)throw py::value_error("Costs must be finite nonnegative nanoseconds");
      }
      if(s.n<1||s.d<1)throw py::value_error("Invalid cost grid shape");samples.push_back(s);
      s.planning_ns=row.contains("planning_ns")?py::cast<double>(row["planning_ns"]):decision_ns;
      if(!std::isfinite(s.planning_ns)||s.planning_ns<0)throw py::value_error("Invalid planner cost");
      samples.back().planning_ns=s.planning_ns;
    }
    for(auto &s:samples){auto &g=grids[key(s.family,s.layout,s.integer)];g.ns.push_back(s.n);g.ds.push_back(s.d);g.cells[{s.n,s.d}]=s;}
    for(auto &item:grids)for(auto axis:{&item.second.ns,&item.second.ds}){std::sort(axis->begin(),axis->end());axis->erase(std::unique(axis->begin(),axis->end()),axis->end());}
    for(const auto &s:samples) {
      auto &g=gain_grids[key(s.family,"",s.integer)];g.ns.push_back(s.n);g.ds.push_back(s.d);
      auto &cell=g.cells[{s.n,s.d}];
      cell.gather=std::max(cell.gather,s.gather);cell.merge=std::max(cell.merge,s.merge);
      cell.repair=std::max(cell.repair,std::max(0.,s.repair_dense-s.repair_rows));
      cell.kernel=std::max(cell.kernel,std::max(0.,s.dense-s.rows));
      cell.planning=std::min(cell.planning,s.planning_ns);
    }
    for(auto &item:gain_grids)for(auto axis:{&item.second.ns,&item.second.ds}){std::sort(axis->begin(),axis->end());axis->erase(std::unique(axis->begin(),axis->end()),axis->end());}
    valid=!samples.empty();
  }
  // Bilinear interpolation is confined to a calibrated family/layout/dtype
  // rectangle. Outside the grid, return no prediction and use the old path.
  std::optional<CostSample> predict(const std::string &family,const std::string &layout,bool integer,int64_t n,int64_t d) const {
    auto found=grids.find(key(family,layout,integer));if(found==grids.end())return std::nullopt;
    const auto &g=found->second;
    auto bracket=[](const auto &axis,int64_t value)->std::optional<std::pair<int64_t,int64_t>> {
      if(axis.empty()||value<axis.front()||value>axis.back())return std::nullopt;
      auto hi=std::lower_bound(axis.begin(),axis.end(),value);
      return std::make_pair(*hi==value?value:*std::prev(hi),*hi);
    };
    auto nr=bracket(g.ns,n),dr=bracket(g.ds,d);if(!nr||!dr)return std::nullopt;
    auto n0=nr->first,n1=nr->second,d0=dr->first,d1=dr->second;
    auto find=[&](int64_t a,int64_t b)->const CostSample*{auto cell=g.cells.find({a,b});return cell==g.cells.end()?nullptr:&cell->second;};
    auto a=find(n0,d0),b=find(n1,d0),c=find(n0,d1),e=find(n1,d1);if(!a||!b||!c||!e)return std::nullopt;
    double x=n0==n1?0.:double(n-n0)/(n1-n0),y=d0==d1?0.:double(d-d0)/(d1-d0);
    CostSample result=*a;result.n=n;result.d=d;
    for(auto member:{&CostSample::rows,&CostSample::dense,&CostSample::gather,&CostSample::row_table,&CostSample::repair_rows,
                     &CostSample::repair_dense,&CostSample::merge,&CostSample::overlay,&CostSample::crop,&CostSample::residual})
      result.*member=(1-y)*((1-x)*(a->*member)+x*(b->*member))+y*((1-x)*(c->*member)+x*(e->*member));
    result.planning_ns=(1-y)*((1-x)*a->planning_ns+x*b->planning_ns)+y*((1-x)*c->planning_ns+x*e->planning_ns);
    return result;
  }
  // An envelope of the model's possible savings over all layouts. It ignores
  // row-map/overlay costs, so rejecting here is conservative relative to the
  // calibrated model. No owner, parameter or layout scan is needed.
  std::optional<bool> could_benefit(const std::string &family,bool integer,int64_t n,int64_t d,int64_t active,bool dense_available)const {
    auto found=gain_grids.find(key(family,"",integer));if(found==gain_grids.end())return std::nullopt;
    const auto &g=found->second;
    auto bounds=[](const auto &axis,int64_t value)->std::optional<std::pair<int64_t,int64_t>> {
      if(axis.empty()||value<axis.front()||value>axis.back())return std::nullopt;
      auto hi=std::lower_bound(axis.begin(),axis.end(),value);return std::make_pair(*hi==value?value:*std::prev(hi),*hi);
    };
    auto nr=bounds(g.ns,std::max<int64_t>(1,n)),ar=bounds(g.ns,std::max<int64_t>(1,active)),dr=bounds(g.ds,d);
    if(!nr||!ar||!dr)return std::nullopt;
    double whole=0,kernel=0,planning=INFINITY;
    const bool partial=active>0&&active<n;
    for(auto i:{nr->first,nr->second})for(auto j:{dr->first,dr->second}) {
      auto cell=g.cells.find({i,j});if(cell==g.cells.end())return true;
      const auto &value=cell->second;
      whole=std::max(whole,(!dense_available?value.gather:0.)+(partial?value.merge:0.)+value.repair);
      planning=std::min(planning,value.planning);
    }
    if(active)for(auto i:{ar->first,ar->second})for(auto j:{dr->first,dr->second}) {
      auto cell=g.cells.find({i,j});if(cell==g.cells.end())return true;
      kernel=std::max(kernel,cell->second.kernel+(partial?cell->second.gather:0.));
    }
    return whole+kernel>planning;
  }
};
struct PlanFacts {
  int64_t n,d,active;
  bool integer,dense_available,overlay_compatible=true,needs_sources=true;
  std::string layout;
  // Count is already known at this read point; a later last-stage check may
  // crop a padded batch. Reserve its row map without reading that future getter.
  bool may_crop=false;
  bool selected_evaluated=true;
};
struct DataDecision {
  bool rows=false,overlay=false,materialize_selected=false,source_rows=false;
  double cost=0,reference_cost=0;
  uint64_t peak=0;
  unsigned alternatives=0;
  std::string reason="disabled";
};
struct StagePlanCache {
  std::shared_ptr<const DataCostModel> model;
  uint64_t limit=0;unsigned maximum=0,shape_next=0,plan_next=0;
  struct Shape {int64_t n=-1,d=-1,active=-1;bool integer=false,dense=false;int worthwhile=0;};
  struct Physical {bool present=false;PlanFacts facts{};DataDecision decision;};
  std::array<Shape,8> shapes;
  std::array<Physical,8> plans;
};
inline uint64_t row_metadata_bound(int64_t n,bool evaluated=false) {
  // Layout/row vectors reserve their exact logical extent. vector<bool>'s
  // provenance storage is rounded up to a native 64-bit word on both supported
  // 64-bit standard libraries. Array/object/allocator headers are not payload.
  auto count=uint64_t(n);
  return count*(sizeof(DecisionRef)+sizeof(RowPtr)+9+(evaluated?3*sizeof(double):0))+((count+63)/64)*8;
}
struct DataPlanner {
  std::string mode="graph";
  uint64_t memory_limit=32u*1024u*1024u;
  unsigned max_decisions=8;
  bool dump=false;
  bool cache_plans=true; // Explicit calibration measures cold decisions too.
  std::shared_ptr<const DataCostModel> model;
  uint64_t decisions=0,rows_plans=0,dense_plans=0,overlays=0,fallbacks=0,enumerations=0,peak=0;
  uint64_t template_builds=0,template_cache_hits=0,replans=0,unknown_boundaries=0,linear_fallbacks=0;
  uint64_t early_skips=0,physical_cache_hits=0;
  uint64_t input_admissions=0,layout_queries=0;
  std::vector<py::dict> records;
  StagePlanCache *cache(const StageTemplate &plan)const {
    if(!cache_plans)return nullptr;
    if(!plan.cache||plan.cache->model!=model||plan.cache->limit!=memory_limit||plan.cache->maximum!=max_decisions) {
      auto value=std::make_shared<StagePlanCache>();value->model=model;value->limit=memory_limit;value->maximum=max_decisions;plan.cache=std::move(value);
    }
    return plan.cache.get();
  }
  std::optional<DataDecision> precheck(const StageTemplate &plan,const PlanFacts &f) {
    const char *reason=nullptr;
    if(mode=="dense")reason="forced_dense";
    else if(!plan.capability.indexed_supported)reason="dense_only";
    else if(mode=="graph") {
      if(!model||!model->valid)reason="uncalibrated";
      else if(!memory_limit)reason="memory_limit";
      else {
        auto cached=cache(plan);std::optional<int> worthwhile;
        if(cached)for(const auto &s:cached->shapes)if(s.n==f.n&&s.d==f.d&&s.active==f.active&&s.integer==f.integer&&s.dense==f.dense_available){worthwhile=s.worthwhile;break;}
        if(!worthwhile){auto estimate=model->could_benefit(plan.capability.family,f.integer,f.n,f.d,f.active,f.dense_available);worthwhile=estimate?(*estimate?1:0):-1;
          if(cached)cached->shapes[cached->shape_next++%cached->shapes.size()]={f.n,f.d,f.active,f.integer,f.dense_available,*worthwhile};}
        if(*worthwhile<=0)reason=*worthwhile<0?"uncalibrated":"early_no_benefit";
      }
    }
    if(!reason)return std::nullopt;
    ++early_skips;DataDecision r;r.materialize_selected=true;r.reason=reason;return remember(r,plan,f);
  }
  static bool same_facts(const PlanFacts &a,const PlanFacts &b) {
    return a.n==b.n&&a.d==b.d&&a.active==b.active&&a.integer==b.integer&&a.dense_available==b.dense_available&&
      a.overlay_compatible==b.overlay_compatible&&a.needs_sources==b.needs_sources&&a.layout==b.layout&&
      a.may_crop==b.may_crop&&a.selected_evaluated==b.selected_evaluated;
  }
  DataDecision choose(const StageTemplate &plan,const PlanFacts &f,bool safe) {
    if(mode!="graph"||!safe||!model||!plan.capability.indexed_supported||!cache_plans)return choose_uncached(plan,f,safe);
    auto cached=cache(plan);
    for(const auto &entry:cached->plans)if(entry.present&&same_facts(entry.facts,f)) {
      ++physical_cache_hits;auto result=entry.decision;result.alternatives=0;return remember(result,plan,f);
    }
    auto result=choose_uncached(plan,f,safe);
    auto &entry=cached->plans[cached->plan_next++%cached->plans.size()];entry.present=true;entry.facts=f;entry.decision=result;
    return result;
  }
  DataDecision choose_uncached(const StageTemplate &plan,const PlanFacts &f,bool safe) {
    DataDecision r;
    if(mode=="dense"||!safe||!plan.capability.indexed_supported){r.materialize_selected=true;r.reason=!safe?"observable_boundary":mode=="dense"?"forced_dense":"dense_only";return remember(r,plan,f);}
    if(mode=="rules") {
      r.rows=!f.dense_available;r.overlay=f.overlay_compatible;r.source_rows=true;r.reason="simple_rules";
      auto metadata=row_metadata_bound(f.n,f.active==0&&f.selected_evaluated);
      bool partial=f.active>0&&f.active<f.n;
      auto input_map=(partial||f.may_crop||(f.active==0&&!f.dense_available))?uint64_t(f.n*2*sizeof(void*)):0;
      auto source_map=(f.needs_sources&&!f.dense_available&&!(f.active==0&&!f.may_crop))?uint64_t(f.n*2*sizeof(void*)):0;
      r.peak=std::max(uint64_t(f.active*2*sizeof(void*))+metadata,metadata*(f.may_crop?2:1)+input_map+source_map);
      if(r.peak>memory_limit){r=DataDecision{};r.materialize_selected=true;r.reason="memory_limit_rules";}
      return remember(r,plan,f);
    }
    auto whole=(model&&model->valid)?model->predict(plan.capability.family,f.layout,f.integer,std::max<int64_t>(1,f.n),f.d):std::nullopt;
    auto active=(model&&model->valid)?model->predict(plan.capability.family,f.layout,f.integer,std::max<int64_t>(1,f.active),f.d):std::nullopt;
    if(!whole||!active){r.materialize_selected=true;r.reason="uncalibrated";return remember(r,plan,f);}
    auto cost=[&](uint32_t mask)->PlanAlternative {
      bool selected=mask&1,dense=mask&2,merged=mask&4;
      bool all=f.active>0&&f.active==f.n,partial=f.active>0&&f.active<f.n;
      bool source_dense=selected||f.dense_available||(dense&&all);
      bool overlay=partial&&!merged;
      bool rows_repair=f.active==0?!selected:all?(!selected||!dense||!merged):(!selected||!merged);
      auto bytes=uint64_t(f.n*f.d*8),active_bytes=uint64_t(f.active*f.d*8);
      if(!f.overlay_compatible&&!merged)return {0,{},false};
      double total=0;std::vector<BufferLife> buffers;
      if(selected&&!f.dense_available){total+=whole->gather;buffers.push_back({0,4,bytes});}
      if(f.active) {
        if(dense) {
          total+=active->dense;
          if(!(all&&(selected||f.dense_available))){total+=active->gather;buffers.push_back({1,all?4u:2u,active_bytes});}
        }else{
          total+=active->rows;
          if(!(all&&(selected||f.dense_available))){total+=active->row_table;buffers.push_back({1,2,uint64_t(f.active*2*sizeof(void*))});}
        }
      }
      if(f.active>0&&f.active<f.n) {
        total+=merged?whole->merge:whole->overlay;
        if(merged)buffers.push_back({3,4,bytes});
      }
      if(rows_repair) {
        auto metadata=row_metadata_bound(f.n,f.active==0&&f.selected_evaluated);
        // Keeping a storage owner through repair retains its row metadata.
        // During a possible prefix take, old and new layouts can coexist.
        buffers.push_back({f.active==0?4u:all?2u:3u,4,metadata});
        if(f.may_crop){total+=whole->crop;buffers.push_back({4,4,metadata});}
      }
      // Overlay calibration includes its repair-input row map. Other row
      // inputs need a map when indexed, or may need one after final cropping.
      bool raw_map=rows_repair&&!overlay&&((f.active==0&&!selected&&!f.dense_available)||f.may_crop);
      if(raw_map)total+=whole->row_table;
      if(raw_map||overlay)buffers.push_back({4,4,uint64_t(f.n*2*sizeof(void*))});
      total+=rows_repair?whole->repair_rows:whole->repair_dense;
      // With no variation and no possible crop, repair input and source are
      // the same logical value. The executor shares their one descriptor.
      bool shared=f.active==0&&!selected&&!f.may_crop;
      if(f.needs_sources&&!selected&&!source_dense&&!shared){total+=whole->row_table;buffers.push_back({4,4,uint64_t(f.n*2*sizeof(void*))});}
      return {total,std::move(buffers),true};
    };
    auto reference=cost(7);r.reference_cost=reference.cost;
    const auto planning=whole->planning_ns;
    // Small fragments have already been rejected before binding. All remaining
    // alternatives use the same overhead/uncertainty rule; no cheaper-policy
    // branch may silently omit the cost of making its decision.
    auto best=bounded_plan(3,memory_limit,max_decisions,cost,0);
    r.alternatives=max_decisions<3?1:8;
    double margin=std::max(model->margin*reference.cost,2*whole->residual);
    if(!best.found||reference.cost-best.cost<=planning+margin) {
      r.materialize_selected=true;r.cost=reference.cost;r.reason=best.found?"insufficient_margin":"memory_limit";
    }else {
      r.materialize_selected=best.mask&1;r.rows=!(best.mask&2);r.overlay=!(best.mask&4);
      r.source_rows=!r.materialize_selected;r.cost=best.cost;r.peak=best.peak;
      r.reason=best.bounded_fallback?"bounded_linear":"cost_selected";
    }
    return remember(r,plan,f);
  }
  DataDecision remember(DataDecision r,const StageTemplate &plan,const PlanFacts &f) {
    ++decisions;rows_plans+=r.rows&&f.active>0;dense_plans+=!r.rows&&f.active>0;overlays+=r.overlay&&f.active>0&&f.active<f.n;fallbacks+=r.materialize_selected;
    enumerations+=r.alternatives;peak=std::max(peak,r.peak);
    unknown_boundaries+=r.reason=="observable_boundary";linear_fallbacks+=r.reason=="bounded_linear";
    if(dump){py::list values;for(auto &v:plan.values)values.append(py::dict("id"_a=v.id,"producer"_a=v.producer,"consumers"_a=v.consumers,"last_use"_a=v.last_use,"observable"_a=v.observable));
      records.push_back(py::dict("kernel"_a=plan.kernel,"n"_a=f.n,"d"_a=f.d,"active"_a=f.active,
      "layout"_a=f.layout,"integer"_a=f.integer,"rows"_a=r.rows,"overlay"_a=r.overlay,
      "may_crop"_a=f.may_crop,"input_source_shared"_a=(f.active==0&&!f.may_crop&&!r.materialize_selected&&r.source_rows),
      "materialize_selected"_a=r.materialize_selected,"source_rows"_a=r.source_rows,"reason"_a=r.reason,
      "predicted_ns"_a=r.cost,"reference_ns"_a=r.reference_cost,"peak_bytes"_a=r.peak,
      "alternatives"_a=r.alternatives,"effect_boundary"_a="ordered_repair_evaluate",
      "values"_a=values,"last_use"_a=py::make_tuple(4,2,3,4,4),"version"_a=1));}
    return r;
  }
  py::dict statistics()const {
    return py::dict("mode"_a=mode,"calibrated"_a=(model&&model->valid),"decisions"_a=decisions,"rows"_a=rows_plans,
      "dense"_a=dense_plans,"overlays"_a=overlays,"fallbacks"_a=fallbacks,"alternatives"_a=enumerations,
      "derived_peak_bytes"_a=peak,"memory_limit_bytes"_a=memory_limit,
      "plan_templates_built"_a=template_builds,"plan_cache_hits"_a=template_cache_hits,"replans"_a=replans,
      "plan_bindings"_a=input_admissions,"unknown_boundaries"_a=unknown_boundaries,"linear_fallbacks"_a=linear_fallbacks,
      "early_skips"_a=early_skips,"physical_cache_hits"_a=physical_cache_hits,"layout_queries"_a=layout_queries);
  }
};
}

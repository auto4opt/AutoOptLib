#pragma once
#include "selection.hpp"
namespace native_components {

inline int64_t excluding(Random &rng,int64_t n,int64_t first,int64_t second=-1) {
  int64_t count=n-(first>=0)-(second>=0&&second!=first);
  if(!count)return rng.integer(0,n);
  auto rank=rng.integer(0,count);
  // The same rank in the ordered complement, without scanning all n entries.
  auto low=first,high=second;
  if(low<0|| (high>=0&&high<low))std::swap(low,high);
  if(low>=0&&rank>=low)++rank;
  if(high>=0&&high!=low&&rank>=high)++rank;
  return rank;
}

template<class Population,class Targets>
NativeResult differential_rows(const std::string &name,
    const Targets &targets,
    const Population &population,
    py::array_t<double,py::array::c_style> fitness,
    py::array_t<double,py::array::c_style> target_fitness,
    const std::vector<double> &parameters,NativeState state,bool context,Random &rng,bool private_input=false) {
  RegionScope numeric_timing(Region::Numeric);
  auto t=targets.shape(0),d=targets.shape(1),n=population.shape(0);
  bool pbest=name=="search_de_current_to_pbest"||name=="search_fastga_de_current_to_pbest";
  PreparedFragment *prepared=nullptr;
  if(fragment_sink&&private_input&&t&&d) {
    prepared=fragment_sink;prepared->configure(pbest?FragmentKind::Select:FragmentKind::Differential,t,d);
    prepared->set_parents(targets);
  }
  auto output=py::reinterpret_borrow<py::array_t<double>>(py::handle());double *output_data=nullptr;
  if(!prepared){output=py::array_t<double>({t,d});output_data=output.mutable_data();if(!private_input)copy_input<double>(targets,output_data);}
  bool fastga=name=="search_fastga_de_current_to_pbest";
  double scale=parameters.empty()?.5:parameters[0],cr=parameters.size()<2?.5:parameters[1];
  const bool vector=aol_x86::dense_enabled(d,t*d);
  if(prepared){prepared->scale=scale;prepared->probability=cr;prepared->masks.values.reserve(t*d);}
  if(pbest) {
    if(prepared){prepared->donor_values=workspace_array<double>(t*d);prepared->forced.reserve(t);}
    if(!n||!d||!t) {
      if(private_input)copy_input<double>(targets,output.mutable_data());
      if(fastga) {
        state.erase("_fastga_match_first_target");
        py::array_t<bool> flags(t);std::fill_n(flags.mutable_data(),t,true);state.set("_stream_invalidated_mask",flags);
      }
      return state_result(output,state);
    }
    if(!parameters.empty()&&parameters.size()!=3)
      throw py::value_error("current-to-pbest requires parameters F, CR, and p_de.");
    double fraction=parameters.empty()?.2:parameters[2];
    for(auto x:{scale,cr,fraction})if(!std::isfinite(x)||x<0||x>1)
      throw py::value_error("F, CR, and p_de must be finite and in [0, 1].");
    auto elite=order(fitness,true);auto count=std::max<int64_t>(1,std::ceil(fraction*n));
    auto indices=py::reinterpret_borrow<py::array_t<int64_t>>(py::handle());
    bool valid=false;
    if(state.contains("_target_population_indices") && !state.object("_target_population_indices").is_none()) {
      indices=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(state.object("_target_population_indices"));
      valid=indices.size()==t;
    }
    bool match=fastga?valid:(state.contains("_fastga_match_first_target")&&state.number<bool>("_fastga_match_first_target"));
    auto donor=prepared?py::reinterpret_borrow<py::array_t<double>>(py::handle()):workspace_array<double>(d);
    std::optional<ScratchVector<double>> coins_lease;if(!prepared)coins_lease.emplace();
    for(py::ssize_t i=0;i<t;++i) {
      auto target_row=input_row<double>(targets,i);
      int64_t target=-1;
      if(match) {
        for(int64_t k=0;k<n;++k)if(fitness.data()[k]==target_fitness.data()[i]) {
          bool equal=true;for(py::ssize_t j=0;j<d;++j)equal&=population.at(k,j)==target_row[j];
          if(equal){target=k;break;}
        }
        if(target<0)throw py::value_error("FastGA current-to-pbest target is not a clone of the current population.");
      } else if(valid&&indices.data()[i]>=0&&indices.data()[i]<n)target=indices.data()[i];
      auto best=elite.data()[rng.integer(0,count)];
      auto first=excluding(rng,n,target),second=excluding(rng,n,target,first);
      auto best_row=population.row(best),first_row=population.row(first),second_row=population.row(second);
      auto donor_data=prepared?prepared->donor_values.mutable_data()+i*d:donor.mutable_data();
      aol_x86::donor(target_row,first_row,second_row,best_row,scale,donor_data,d,vector);
      int64_t forced;
      if(match)forced=rng.integer(0,d);
      else {
        if(prepared)for(py::ssize_t j=0;j<d;++j)prepared->masks.values.push_back(rng.uniform());
        else rng.uniform(coins_lease->values,d);
        forced=rng.integer(0,d);
      }
      if(prepared) {
        prepared->forced.push_back(forced);
        if(match)for(py::ssize_t j=0;j<d;++j)prepared->masks.values.push_back(j==forced?1.:rng.uniform());
      }
      else if(!match)aol_x86::select(donor.data(),private_input?target_row:output_data+i*d,coins_lease->values.data(),cr,forced,output_data+i*d,d,vector);
      else for(py::ssize_t j=0;j<d;++j) {
        bool take=j==forced||rng.uniform()<cr;
        if(take)output_data[i*d+j]=donor.data()[j];
        else if(private_input)output_data[i*d+j]=target_row[j];
      }
    }
    if(fastga) {
      state.erase("_fastga_match_first_target");
      py::array_t<bool> flags(t);std::fill_n(flags.mutable_data(),t,true);state.set("_stream_invalidated_mask",flags);
    }
  } else {
    int donor_count=name=="search_de_random"?3:name=="search_de_current_best"?1:2;
    std::vector<py::array_t<int64_t>> donors;donors.reserve(donor_count);
    for(int k=0;k<donor_count;++k)donors.push_back(workspace_array<int64_t>(t));
    for(auto &indices:donors) {
      auto values=indices.mutable_data();
      if(context)rng.integers(0,n,values,t);
      else {std::iota(values,values+t,0);rng.permutation(values,t);}
    }
    int64_t best=0;
    if(name=="search_de_current_best") {
      for(int64_t i=0;i<n;++i)if(std::isnan(fitness.data()[i])||fitness.data()[i]<fitness.data()[best]) {
        best=i;if(std::isnan(fitness.data()[i]))break;
      }
    }
    ScratchVector<double> coins_lease;auto &coins=prepared?prepared->masks.values:coins_lease.values;rng.uniform(coins,t*d);
    if(prepared){
      prepared->model_owner=std::make_shared<Population>(population);
      prepared->bases.reserve(t);prepared->seconds.reserve(t);prepared->thirds.reserve(t);

    }
    for(py::ssize_t i=0;i<t;++i) {auto target_row=input_row<double>(targets,i);
     auto base_row=name=="search_de_random"?population.row(donors[0].data()[i]):target_row;
     auto second_row=name=="search_de_current_best"?population.row(best):population.row(donors[donor_count-2].data()[i]);
     auto third_row=population.row(donors[donor_count-1].data()[i]);
     if(prepared){prepared->bases.push_back(base_row);prepared->seconds.push_back(second_row);prepared->thirds.push_back(third_row);continue;}
     aol_x86::scalar_each(d,[&](size_t j) {
      double base=base_row[j],second=second_row[j],third=third_row[j];
      double donor=base+scale*(second-third);
      if(coins[i*d+j]<cr)output_data[i*d+j]=donor;
      else if(private_input)output_data[i*d+j]=target_row[j];
     });
    }
  }
  if(prepared){return state_result(py::none(),state);}
  return state_result(output,state);
}
struct DenseDecisionRows {
  py::array_t<double,py::array::c_style> array;
  int ndim() const {return array.ndim();}
  py::ssize_t shape(int axis) const {return array.shape(axis);}
  const double *row(py::ssize_t i) const {return array.data()+i*array.shape(1);}
  double at(py::ssize_t i,py::ssize_t j) const {return row(i)[j];}
};
inline NativeResult differential(const std::string &name,
    py::array_t<double,py::array::c_style> targets,
    py::array_t<double,py::array::c_style> population,
    py::array_t<double,py::array::c_style> fitness,
    py::array_t<double,py::array::c_style> target_fitness,
    const std::vector<double> &parameters,NativeState state,bool context,Random &rng) {
  return differential_rows(name,targets,DenseDecisionRows{population},fitness,target_fitness,parameters,state,context,rng);
}

}

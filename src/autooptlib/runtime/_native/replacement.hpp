#pragma once
#include "selection.hpp"
#include "array_workspace.hpp"
#include "constraint_order.hpp"
namespace native_components {
inline py::array_t<int64_t> indices_array(const int64_t *values,size_t count) {
  py::array_t<int64_t> result(count);
  std::copy_n(values,count,result.mutable_data());return result;
}

template<class Exp>
NativeResult replacement(const std::string &name,py::array_t<double,py::array::c_style> fitness,
                      int64_t n,Random &rng,double initial,int64_t generation,
                      int64_t maximum,NativeState state,Exp &exponential,const FitnessValues &values) {
  RegionScope numeric_timing(Region::Numeric);
  int64_t size=fitness.size();auto fits=fitness.data();
  ScratchVector<int64_t> result_lease;auto &result=result_lease.values;
  if(name=="update_always") {
    int64_t first=n==0?0:std::max<int64_t>(0,size-n);
    for(int64_t i=first;i<size;++i)result.push_back(i);
  } else if(name=="update_iterated_local_search") {
    std::string active=state.contains("_active_search_operator")?py::cast<std::string>(state.object("_active_search_operator")):"";
    n=std::min(n,size);
    if(active.rfind("reinit_",0)==0) {
      for(int64_t i=n==0?0:size-n;i<size;++i)result.push_back(i);
    } else {
      auto ranked=order(fitness,true);
      if(n>=0)return state_result(indices_array(ranked.data(),n),state);
      result.assign(ranked.data(),ranked.data()+n);
    }
    return state_result(indices_array(result.data(),result.size()),state);
  } else if(name=="update_greedy") {
    auto ranked=order(fitness);n=std::min(n,size);
    if(n>=0)return state_result(indices_array(ranked.data(),n),py::none());
    result.assign(ranked.data(),ranked.data()+n);
  } else if(name=="update_fastga_plus") {
    if(n<=0||size<n)throw py::value_error("FastGA plus replacement requires a parent population.");
    for(int64_t i=n;i<size;++i)result.push_back(i);
    for(int64_t i=0;i<n;++i)result.push_back(i);
    std::stable_sort(result.begin(),result.end(),[&](auto a,auto b){return fits[a]<fits[b];});
    result.resize(n);
  } else if(name=="update_pairwise"||name=="update_simulated_annealing") {
    const bool anneal=name=="update_simulated_annealing";
    if(anneal && size<n)throw py::value_error("update_simulated_annealing requires an old population");
    if(!anneal)n=std::min(n,size);
    if(anneal&&(!std::isfinite(initial)||initial<=0))
      throw py::value_error("The simulated-annealing initial temperature must be finite and positive.");
    const double rate=anneal?std::pow(.01/initial,1./std::max<int64_t>(maximum,1)):0;
    const double temperature=anneal?initial*std::pow(rate,generation):0;
    result.reserve(n);for(int64_t i=0;i<n;++i)result.push_back(i);
    for(int64_t i=n;i<size;++i) {
      if(!n) {PyErr_SetString(PyExc_ZeroDivisionError,"integer modulo by zero");throw py::error_already_set();}
      auto k=(i-n)%n;
      auto old=values.key(result[k]),next=values.key(i);
      bool accept=next<old;
      if(anneal && next.first==old.first && !accept) {
        double denominator=(std::abs(old.second)+1e-6)*std::max(temperature,1e-8);
        double exponent=std::min(std::max((old.second-next.second)/denominator,-700.),700.);
        accept=rng.uniform()<exponential.scalar(exponent);
      }
      if(accept)result[k]=i;
    }
  } else if(name=="update_round_robin") {
    n=std::min(n,size);int64_t k=std::min<int64_t>(10,n-1);
    if(k<=0) {auto ranked=order(fitness);if(n>=0)return state_result(indices_array(ranked.data(),n),py::none());result.assign(ranked.data(),ranked.data()+n);}
    else {
      auto keys=workspace_shaped<double>({size,size});rng.uniform(keys.mutable_data(),size*size);
      py::array_t<int64_t> kth(1);kth.mutable_data()[0]=k-1;
      auto raw=PyArray_ArgPartition(reinterpret_cast<PyArrayObject *>(keys.ptr()),
                  reinterpret_cast<PyArrayObject *>(kth.ptr()),1,NPY_INTROSELECT);
      if(!raw)throw py::error_already_set();
      auto opponents=py::reinterpret_steal<py::array_t<int64_t>>(raw);
      py::array_t<int64_t> wins(size);
      for(int64_t i=0;i<size;++i) {
        int64_t count=0;for(int64_t j=0;j<k;++j)count+=fits[i]<=fits[opponents.data()[i*size+j]];
        wins.mutable_data()[i]=count;
      }
      auto ranked=order(wins);
      for(int64_t i=0;i<n;++i)result.push_back(ranked.data()[size-1-i]);
    }
  } else if(name.rfind("update_ssga_",0)==0) {
    if(n<=0||size<n)throw py::value_error("SSGA replacement requires a non-empty parent population.");
    int64_t children=size-n;
    if(children>n)throw py::value_error("SSGA replacement requires offspring_size <= population_size.");
    bool fastga=name.rfind("update_ssga_fastga_",0)==0;
    if(!(fastga&&children==n)) {
      result.reserve(n);for(int64_t i=0;i<n;++i)result.push_back(i);
      for(int64_t child=0;child<children;++child) {
        int64_t victim=0,length=result.size();
        if(name.find("worst")!=std::string::npos) {
          for(int64_t i=1;i<length;++i)
            if(std::isnan(fits[result[i]])||fits[result[i]]>fits[result[victim]]) {
              victim=i;if(std::isnan(fits[result[i]]))break;
            }
        } else {
          int64_t first,second;
          bool stochastic=name.find("stochastic")!=std::string::npos;
          if(fastga) {
            first=rng.integer(0,length);second=first;
            if(stochastic)second=rng.integer(0,length);
            else if(length>1)while(second==first)second=rng.integer(0,length);
          } else {
            int64_t contestants[2];rng.integers(0,length,contestants,2);
            first=contestants[0];second=contestants[1];
          }
          bool worse=stochastic?rng.uniform()<.75:true;
          bool first_worse=fastga&&stochastic?fits[result[first]]>fits[result[second]]
                                                       :fits[result[first]]>=fits[result[second]];
          victim=first_worse==worse?first:second;
        }
        result.erase(result.begin()+victim);
      }
    }
    for(int64_t i=n;i<size;++i)result.push_back(i);
  } else throw py::value_error("Native update core not implemented: "+name);
  return state_result(indices_array(result.data(),result.size()),py::none());
}
}

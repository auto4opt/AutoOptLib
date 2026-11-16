// Complete native selection cores. Inputs have been normalized at the public
// boundary; no component/helper or Generator method is called from here.
#pragma once
#include "random.hpp"
#include "x86_kernels.hpp"
#include "execution_services.hpp"
#include "array_workspace.hpp"
#include "native_state.hpp"
#include "native_interval.hpp"
#include "decision_input.hpp"
#include "fragment.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <numeric>
namespace native_components {
namespace py=pybind11;

inline py::array_t<int64_t> order(const py::array &fitness,bool stable=false) {
  auto out=PyArray_ArgSort(reinterpret_cast<PyArrayObject *>(fitness.ptr()),-1,
                          stable?NPY_STABLESORT:NPY_QUICKSORT);
  if(!out)throw py::error_already_set();
  return py::reinterpret_steal<py::array_t<int64_t>>(out);
}

inline NativeResult selection(const std::string &name,int64_t n,py::array_t<double,py::array::c_style> fits,
                           Random &rng,int64_t count,double parameter,NativeState state) {
  RegionScope numeric_timing(Region::Numeric);
  if(count<0)throw py::value_error("Selection count cannot be negative.");
  py::array_t<int64_t> selected(n?count:0);
  auto out=selected.mutable_data();auto f=fits?fits.data():nullptr;
  if(!n)return state_result(selected,py::none());
  if(name=="choose_random")rng.integers(0,n,out,count);
  else if(name=="choose_traverse") {
    int64_t cursor=state.contains("choose_traverse_cursor")?state.number<int64_t>("choose_traverse_cursor"):0;
    if(state.contains("choose_traverse_population_size") && state.number<int64_t>("choose_traverse_population_size")!=n)cursor=0;
    for(int64_t i=0;i<count;++i)out[i]=((cursor+i)%n+n)%n;
    NativeState next=state.copy();
    next.set("choose_traverse_cursor",(cursor+count)%n);
    next.set("choose_traverse_population_size",n);
    next.erase("_selection_count");
    return state_result(selected,next);
  } else if(name=="choose_tournament" || name=="choose_stochastic_tournament") {
    auto contestants_owner=workspace_array<int64_t>(count*2);auto contestants=contestants_owner.mutable_data();rng.integers(0,n,contestants,count*2);
    ScratchVector<double> coins_lease;auto &coins=coins_lease.values;
    if(name=="choose_stochastic_tournament") {rng.uniform(coins,count);}
    for(int64_t i=0;i<count;++i) {
      auto a=contestants[2*i],b=contestants[2*i+1];
      if(name=="choose_tournament") {
        if(std::isnan(f[a])||std::isnan(f[b]))throw py::index_error("index 0 is out of bounds for axis 0 with size 0");
        out[i]=f[a]<=f[b]?a:b;
      } else {
        bool better=f[a]<=f[b];out[i]=(coins[i]<.75)==better?a:b;
      }
    }
  } else if(name=="choose_fastga_stochastic_tournament") {
    for(int64_t i=0;i<count;++i) {
      auto a=rng.integer(0,n),b=rng.integer(0,n);bool better=rng.uniform()<.75;
      out[i]=(f[a]<=f[b])==better?a:b;
    }
  } else if(name=="choose_elite_fraction") {
    if(!std::isfinite(parameter)||parameter<0||parameter>1)
      throw py::value_error("Elite fraction p must be finite and in [0, 1].");
    auto ranked=order(fits,true);int64_t elite=std::max<int64_t>(1,std::ceil(parameter*n));
    rng.integers(0,elite,out,count);
    for(int64_t i=0;i<count;++i)out[i]=ranked.data()[out[i]];
  } else if(name=="choose_ica") {
    auto ranked=order(fits);int64_t k=std::isfinite(parameter)?std::max<int64_t>(1,std::nearbyint(parameter)):1;
    NativeState next;next.set("imperialist_count",std::min(k,n));
    return state_result(ranked,next);
  } else if(name=="choose_roulette_wheel"||name=="choose_fastga_proportional") {
    const bool fastga=name=="choose_fastga_proportional";
    ScratchVector<double> weights_lease(n);auto &weights=weights_lease.values;bool any=false,all_nonnegative=true;
    double scale=1,minimum=INFINITY,maximum=-INFINITY;
    for(int64_t i=0;i<n;++i) {
      double x=fastga?-f[i]:f[i];
      if(std::isfinite(x)) {any=true;scale=std::max(scale,std::abs(x));minimum=std::min(minimum,x);maximum=std::max(maximum,x);}
      else all_nonnegative=false;
      if(x<0)all_nonnegative=false;
    }
    if(!any) {rng.integers(0,n,out,count);return state_result(selected,py::none());}
    if(fastga&&all_nonnegative)for(int64_t i=0;i<n;++i)weights[i]=-f[i];
    else {
      minimum/=scale;maximum/=scale;
      bool positive=false;
      for(int64_t i=0;i<n;++i) {
        double x=fastga?-f[i]:f[i];
        if(std::isfinite(x)) {
          double shifted=fastga?x/scale-minimum:maximum-x/scale;
          positive|=shifted>0;weights[i]=shifted;
        }
      }
      for(int64_t i=0;i<n;++i)if(std::isfinite(f[i]))
        weights[i]=(fastga&&!positive)?1:weights[i]+std::numeric_limits<double>::epsilon();
    }
    std::partial_sum(weights.begin(),weights.end(),weights.begin());
    const double total=weights.back();
    if(!fastga)for(auto &x:weights)x/=total;
    ScratchVector<double> coins_lease;auto &coins=coins_lease.values;rng.uniform(coins,count);
    for(int64_t i=0;i<count;++i) {
      auto found=fastga?std::upper_bound(weights.begin(),weights.end(),coins[i]*total)
                       :std::lower_bound(weights.begin(),weights.end(),coins[i]);
      out[i]=std::min<int64_t>(found-weights.begin(),n-1);
    }
  } else throw py::value_error("Native selection core not implemented: "+name);
  return state_result(selected,py::none());
}
}

// NumPy-compatible native draws. The Python boundary supplies the existing
// BitGenerator capsule; component loops never invoke Generator methods.
// choice/permutation reproduce NumPy 1.26.4 _generator.pyx algorithms.
// Copyright (c) NumPy Developers. BSD-3-Clause; see NUMPY_LICENSE.txt.
#pragma once
#include <numpy/random/bitgen.h>
#include <numpy/random/distributions.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <iterator>
#include <stdexcept>
#include <vector>
#include "execution_services.hpp"

namespace native_components {
class Random {
  bitgen_t *state_;
  binomial_t binomial_cache_{};
public:
  explicit Random(bitgen_t *state) : state_(state) {}
  double uniform() { return state_->next_double(state_->state); }
  void uniform(double *out, size_t n) {
    random_standard_uniform_fill(state_, n, out);
  }
  void normal(double *out, size_t n) {
    random_standard_normal_fill(state_, n, out);
  }
  // NumPy's fill functions perform these exact scalar calls in this order.
  // Append constructs each element with its final value, without a zero pass.
  void uniform(std::vector<double> &out,size_t n) {
    out.clear();out.reserve(n);for(size_t i=0;i<n;++i)out.push_back(uniform());
  }
  void normal(std::vector<double> &out,size_t n) {
    out.clear();out.reserve(n);for(size_t i=0;i<n;++i)out.push_back(random_standard_normal(state_));
  }
  double normal(double mean, double scale) {
    return random_normal(state_, mean, scale);
  }
  double cauchy() { return random_standard_cauchy(state_); }
  int64_t binomial(int64_t trials, double p) {
    return random_binomial(state_, p, trials, &binomial_cache_);
  }
  int64_t integer(int64_t low, int64_t high) {
    if (low >= high) throw std::invalid_argument("low >= high");
    uint64_t out;
    random_bounded_uint64_fill(state_, static_cast<uint64_t>(low),
        static_cast<uint64_t>(high)-static_cast<uint64_t>(low)-1, 1, false, &out);
    return static_cast<int64_t>(out);
  }
  void integers(int64_t low, int64_t high, int64_t *out, size_t n) {
    if (!n) return;
    if (low >= high) throw std::invalid_argument("low >= high");
    random_bounded_uint64_fill(state_, static_cast<uint64_t>(low),
        static_cast<uint64_t>(high)-static_cast<uint64_t>(low)-1, n, false,
        reinterpret_cast<uint64_t *>(out));
  }
  // Generator.shuffle/permutation uses masked random_interval. choice's
  // final shuffle instead uses bounded Lemire draws; these are not interchangeable.
  void permutation(int64_t *values, size_t n) {
    for (size_t i=n; i>1; --i)
      std::swap(values[i-1], values[random_interval(state_, i-1)]);
  }
  void choice_shuffle(int64_t *values, size_t n, size_t first=1) {
    for (size_t i=n; i>first; --i)
      std::swap(values[i-1], values[random_bounded_uint64(state_,0,i-1,0,false)]);
  }
  std::vector<int64_t> choice(size_t population,size_t n,bool replace=true,const double *weights=nullptr,bool shuffle=true) {
    std::vector<int64_t> result;choice_into(result,population,n,replace,weights,shuffle);return result;
  }
  void choice_into(std::vector<int64_t> &result,size_t population,size_t n,bool replace=true,const double *weights=nullptr,bool shuffle=true) {
    if (!population && n) throw std::invalid_argument("a must be positive");
    if (!replace && n>population)
      throw std::invalid_argument("Cannot take a larger sample than population when replace is False");
    result.clear();result.reserve(n);
    if (weights) {
      if(replace) {
        ScratchVector<double> cdf_lease,coins_lease;auto &cdf=cdf_lease.values;auto &coins=coins_lease.values;
        cdf.reserve(population);std::partial_sum(weights,weights+population,std::back_inserter(cdf));
        const double total=cdf.empty()?0:cdf.back();for(auto &x:cdf)x/=total;
        uniform(coins,n);
        for(size_t i=0;i<n;++i)result.push_back(std::upper_bound(cdf.begin(),cdf.end(),coins[i])-cdf.begin());
        return;
      }
      ScratchVector<double> p_lease,cdf_lease;auto &p=p_lease.values;auto &cdf=cdf_lease.values;
      p.assign(weights,weights+population);cdf.reserve(population);
      if (!replace && static_cast<size_t>(std::count_if(p.begin(),p.end(),
              [](double x){return x>0;}))<n)
        throw std::invalid_argument("Fewer non-zero entries in p than size");
      size_t found=0;
      do {
        if (!replace)
          for(size_t i=0;i<found;++i)p[result[i]]=0;
        cdf.clear();std::partial_sum(p.begin(),p.end(),std::back_inserter(cdf));
        const double total=cdf.empty()?0:cdf.back();
        for(auto &x:cdf)x/=total;
        const size_t requested=n-found;
        ScratchVector<double> coins_lease;auto &coins=coins_lease.values;
        uniform(coins,requested);
        ScratchVector<int64_t> added_lease;auto &added=added_lease.values;
        uint64_t mask=static_cast<uint64_t>(1.2*requested);mask|=mask>>1;mask|=mask>>2;mask|=mask>>4;mask|=mask>>8;mask|=mask>>16;mask|=mask>>32;
        ScratchVector<int64_t> seen_lease(mask+1,-1);auto &seen=seen_lease.values;
        for(double x:coins) {
          int64_t index=std::upper_bound(cdf.begin(),cdf.end(),x)-cdf.begin();
          auto location=static_cast<uint64_t>(index)&mask;
          while(seen[location]!=-1&&seen[location]!=index)location=(location+1)&mask;
          if(seen[location]==-1){seen[location]=index;added.push_back(index);}
        }
        for(auto index:added){result.push_back(index);++found;}
      } while(found<n);
      return;
    }
    if(replace) {
      result.resize(n); // The unchanged bulk integer API needs live elements.
      integers(0,population,result.data(),n);
    } else if(population>10000 && n>population/(shuffle?50:20)) {
      ScratchVector<int64_t> all_lease;auto &all=all_lease.values;all.reserve(population);
      for(size_t i=0;i<population;++i)all.push_back(i);
      choice_shuffle(all.data(),population,std::max<size_t>(population-n,1));
      result.assign(all.end()-n,all.end());
    } else {
      // NumPy 1.26's Floyd algorithm, including its shuffle and draw order.
      uint64_t mask=static_cast<uint64_t>(1.2*n);
      mask|=mask>>1;mask|=mask>>2;mask|=mask>>4;
      mask|=mask>>8;mask|=mask>>16;mask|=mask>>32;
      ScratchVector<int64_t> seen_lease(mask+1,-1);
      auto seen=reinterpret_cast<uint64_t *>(seen_lease.values.data());
      for(size_t j=population-n;j<population;++j) {
        uint64_t value=random_bounded_uint64(state_,0,j,0,false),location=value&mask;
        while(seen[location]!=UINT64_MAX && seen[location]!=value)
          location=(location+1)&mask;
        if(seen[location]!=UINT64_MAX) {
          value=j;location=value&mask;
          while(seen[location]!=UINT64_MAX)location=(location+1)&mask;
        }
        seen[location]=value;result.push_back(value);
      }
      if(shuffle)choice_shuffle(result.data(),n);
    }
    return;
  }
};
} // namespace native_components

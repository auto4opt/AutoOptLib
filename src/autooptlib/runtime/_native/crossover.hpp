#pragma once
#include "selection.hpp"
#include <array>
#include <optional>

namespace native_components {

template<class T,class Input>
NativeResult point_crossover(const std::string &name,
    const Input &input,const std::vector<double> &parameters,
    NativeState state,Random &rng,bool private_input=false) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0),d=input.shape(1),half=(n+1)/2;
  bool fastga=name.rfind("cross_fastga_",0)==0;
  std::string kind=name.substr(fastga?13:12);
  int64_t points=kind=="three"?3:kind=="five"?5:kind=="n"?
      (parameters.empty()?1:static_cast<int64_t>(std::nearbyint(parameters[0]))):0;
  if(fastga && (n<2||d==0))throw py::value_error("FastGA crossover requires at least two parent solutions and non-empty bit strings.");
  if(fastga)for(py::ssize_t i=0;i<n;++i)if(!aol_x86::binary(input_row<T>(input,i),d))
    throw py::value_error("FastGA bit crossover requires binary decisions in {0, 1}.");
  if(!fastga && kind=="two" && d<2)throw py::value_error("Two-point crossover requires D >= 2");
  if(!fastga && points && d<=points)throw py::value_error(std::to_string(points)+"-point crossover requires dimension > "+std::to_string(points)+".");
  if(kind=="n"&&points<1)throw py::value_error("The number of crossover points must be positive.");
  if(fastga&&points&&d<2)throw py::value_error("FastGA n-point crossover requires at least two bits.");
  if(fragment_sink&&!fastga&&kind=="uniform"&&private_input) {
    auto prepared=fragment_sink;prepared->configure(FragmentKind::UniformCross,n,d);
    prepared->set_parents(input);prepared->integer=std::is_same_v<T,int64_t>;
    prepared->probability=parameters.empty()?.5:parameters[0];
    rng.uniform(prepared->coins.values,half*d);
    // Packing is a forced diagnostic until encode/decode amortization is proven.
    if(prepared->integer&&packed_fragments()) {
      bool binary=true;for(py::ssize_t i=0;i<n&&binary;++i)binary=aol_x86::binary(input_row<T>(input,i),d);
      if(binary) {
        // Odd crossover has overlapping parent roles; retain the original
        // mapping by packing a private expanded pair owner, never in-place.
        prepared->bits=std::make_unique<PrivateBits>(2*half,d);prepared->bits->pack_pairs(input);
        std::vector<uint64_t> mask((d+63)/64);
        for(py::ssize_t i=0;i<half;++i){std::fill(mask.begin(),mask.end(),0);for(py::ssize_t j=0;j<d;++j)if(prepared->coins.values[i*d+j]<prepared->probability)mask[j/64]|=uint64_t(1)<<(j%64);prepared->bits->exchange(i,half+i,mask);}
      }
    }
    return state_result(py::none(),state);
  }
  py::array_t<T> backing({2*half,d});auto out=backing.mutable_data();
  // Caller arrays retain the old snapshot-before-RNG boundary. The native
  // core can generate directly from its unexposed population storage.
  if(!private_input)for(py::ssize_t i=0;i<half;++i) {
    std::copy_n(input_row<T>(input,i),d,out+i*d);
    std::copy_n(input_row<T>(input,n/2+i),d,out+(half+i)*d);
  }
  py::array_t<bool> invalidated(n);std::fill_n(invalidated.mutable_data(),n,false);
  auto one_cuts=!fastga&&kind=="one"?workspace_array<int64_t>(half):py::reinterpret_borrow<py::array_t<int64_t>>(py::handle());
  {
  NumericInterval interval(n*d);
  ScratchVector<int64_t> cuts_lease;auto &cuts=cuts_lease.values;
  ScratchVector<double> coins_lease;auto &coins=coins_lease.values;
  if(!fastga&&kind=="one")rng.integers(0,d,one_cuts.mutable_data(),half);
  if(!fastga&&kind=="two")for(py::ssize_t i=0;i<half;++i) {
    auto p=rng.choice(d,2,false);std::sort(p.begin(),p.end());cuts.insert(cuts.end(),p.begin(),p.end());
  }
  if(!fastga&&kind=="uniform") {rng.uniform(coins,half*d);}
  for(py::ssize_t i=0;i<half;++i) {
    auto a=out+i*d,b=out+(half+i)*d;
    const T *p=private_input?input_row<T>(input,i):a;
    const T *q=private_input?input_row<T>(input,n/2+i):b;
    bool changed=false;
    auto emit=[&](py::ssize_t j,bool exchange) {
      // Load both before writing: p/q may be the compatibility snapshot a/b.
      T first=p[j],second=q[j];a[j]=exchange?second:first;b[j]=exchange?first:second;
    };
    if(fastga&&kind=="uniform") {
      ScratchVector<int64_t> different_lease;auto &different=different_lease.values;
      aol_x86::differences(p,q,d,different);
      rng.uniform(coins,different.size());
      changed=aol_x86::uniform_different(p,q,a,b,d,different,coins.data());
    } else if(fastga&&kind=="one") {
      auto cut=rng.integer(0,d);
      for(int64_t j=0;j<cut;++j)changed|=p[j]!=q[j];
      for(py::ssize_t j=0;j<d;++j)emit(j,changed&&j<cut);
    } else if(fastga&&points) {
      ScratchVector<bool> toggles_lease(d);auto &toggles=toggles_lease.values;int64_t remaining=std::min<int64_t>(d-1,points);
      while(remaining) {auto bit=rng.integer(0,d);if(!toggles[bit]) {toggles[bit]=true;--remaining;}}
      bool exchange=false;emit(0,false);
      for(py::ssize_t j=1;j<d;++j) {if(toggles[j])exchange=!exchange;emit(j,exchange);}
      changed=true;
    } else if(kind=="one"||kind=="two") {
      auto lo=kind=="one"?one_cuts.data()[i]:cuts[2*i],hi=kind=="one"?d:cuts[2*i+1]+1;
      if(private_input) {
        std::copy_n(p,lo,a);std::copy_n(q,lo,b);
        std::copy_n(q+lo,hi-lo,a+lo);std::copy_n(p+lo,hi-lo,b+lo);
        std::copy_n(p+hi,d-hi,a+hi);std::copy_n(q+hi,d-hi,b+hi);
      } else for(py::ssize_t j=0;j<d;++j)emit(j,j>=lo&&j<hi);
    } else if(kind=="uniform") {
      double probability=parameters.empty()?.5:parameters[0];
      aol_x86::uniform(p,q,a,b,coins.data()+i*d,probability,d);
    } else if(points) {
      auto positions=rng.choice(d-1,points,false);ScratchVector<bool> toggles_lease(d);auto &toggles=toggles_lease.values;
      for(auto cut:positions)toggles[cut+1]=true;
      bool exchange=false;
      for(py::ssize_t j=0;j<d;++j) {if(toggles[j])exchange=!exchange;emit(j,exchange);}
    } else throw py::value_error("Native crossover core not implemented: "+name);
    if(fastga) {invalidated.mutable_data()[i]=changed;if(half+i<n)invalidated.mutable_data()[half+i]=changed;}
  }
  }
  py::array result(py::dtype::of<T>(),{n,d},{d*py::ssize_t(sizeof(T)),py::ssize_t(sizeof(T))},backing.data(),backing);
  if(fastga)state.set("_stream_invalidated_mask",invalidated);
  auto output=state_result(result,state);output.binary_output=fastga&&std::is_same_v<T,int64_t>;return output;
}

// Call-local membership for arbitrary int64 labels. Inline tiny sets avoid
// hashing work; larger sets use flat scratch, without per-element node malloc.
class ValueMembership {
  std::optional<ScratchVector<int64_t>> table_;
  std::optional<ScratchVector<bool>> occupied_;
  std::array<int64_t,8> small_;size_t small_count_=0,mask_=0;
  size_t slot(int64_t value) const {
    uint64_t h=static_cast<uint64_t>(value);h^=h>>33;h*=0xff51afd7ed558ccdULL;h^=h>>33;
    return static_cast<size_t>(h)&mask_;
  }
public:
  template<class Value>ValueMembership(size_t count,Value value) {
    // Up to eight exact comparisons cost less work than hash preparation.
    if(count<=small_.size()){small_count_=count;for(size_t i=0;i<count;++i)small_[i]=value(i);return;}
    size_t capacity=1;while(capacity<count*2)capacity*=2;mask_=capacity-1;
    table_.emplace(capacity);occupied_.emplace(capacity,false);
    for(size_t i=0;i<count;++i){auto v=value(i);auto index=slot(v);
      while(occupied_->values[index]&&table_->values[index]!=v)index=(index+1)&mask_;
      occupied_->values[index]=true;table_->values[index]=v;}
  }
  bool contains(int64_t value) const {
    if(small_count_)return std::find(small_.begin(),small_.begin()+small_count_,value)!=small_.begin()+small_count_;
    if(!mask_)return false;auto index=slot(value);
    while(occupied_->values[index]){if(table_->values[index]==value)return true;index=(index+1)&mask_;}return false;
  }
};

template<class Input>
py::array order_crossover(const std::string &name,
    const Input &input,int64_t requested,Random &rng) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0),d=input.shape(1),half=(n+1)/2;
  py::array_t<int64_t> backing({2*half,d});auto out=backing.mutable_data();
  for(py::ssize_t i=0;i<half;++i) {
    auto first=input_row<int64_t>(input,i),second=input_row<int64_t>(input,n/2+i);
    for(int child=0;child<2;++child) {
      if(name=="cross_order_n" && child)continue;
      auto a=child?second:first,b=child?first:second;
      auto destination=out+(child*half+i)*d;
      std::copy_n(a,d,destination);
      if(name=="cross_order_two") {
        auto pair=rng.choice(d,2,false);std::sort(pair.begin(),pair.end());
        ScratchVector<int64_t> remainder_lease;auto &remainder=remainder_lease.values;
        ValueMembership selected(pair[1]-pair[0]+1,[&](size_t j){return a[pair[0]+j];});
        for(py::ssize_t j=0;j<d;++j)if(!selected.contains(b[j]))remainder.push_back(b[j]);
        size_t k=0;
        for(py::ssize_t j=0;j<d;++j)if(j<pair[0]||j>pair[1]) {
          if(k>=remainder.size())throw py::value_error("Order crossover requires matching permutations");
          destination[j]=remainder[k++];
        }
      } else {
        // N-order uses one common chosen set for both children, unlike the
        // two independent segment draws in order-two.
        auto chosen=rng.choice(d,std::min<int64_t>(std::max<int64_t>(1,requested),d),false);
        for(int which=0;which<2;++which) {
          auto p=which?second:first,q=which?first:second;
          auto target=out+(which*half+i)*d;
          std::copy_n(p,d,target);ScratchVector<int64_t> remainder_lease;auto &remainder=remainder_lease.values;
          ValueMembership removed_values(chosen.size(),[&](size_t j){return p[chosen[j]];});
          ScratchVector<bool> selected_lease(d,false);auto &selected=selected_lease.values;
          for(auto j:chosen)selected[j]=true;
          for(py::ssize_t j=0;j<d;++j)if(!removed_values.contains(q[j]))remainder.push_back(q[j]);
          size_t k=0;
          for(py::ssize_t j=0;j<d;++j)if(!selected[j]) {
            if(k>=remainder.size())throw py::value_error("Order crossover requires matching permutations");
            target[j]=remainder[k++];
          }
        }
      }
    }
  }
  return py::array(py::dtype::of<int64_t>(),{n,d},{d*py::ssize_t(8),py::ssize_t(8)},backing.data(),backing);
}
}

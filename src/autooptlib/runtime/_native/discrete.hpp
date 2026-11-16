#pragma once
#include "selection.hpp"
#include <string_view>
namespace native_components {

inline void flip_bits(int64_t *row, const std::vector<int64_t> &positions) {
  for (auto column : positions) row[column] ^= 1;
}

template<class Input>
py::array_t<int64_t> discrete_variation(const std::string &name,
    const Input &input,
    py::array_t<int64_t,py::array::c_style> bounds,
    const std::vector<double> &parameters,Random &rng) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0),d=input.shape(1);
  if(name=="reinit_permutation") {
    auto keys=workspace_shaped<double>({n,d});rng.uniform(keys.mutable_data(),n*d);
    auto ranked=order(keys);
    for(py::ssize_t k=0;k<n*d;++k)++ranked.mutable_data()[k];
    return ranked;
  }
  py::array_t<int64_t> output({n,d});
  auto out=output.mutable_data();
  if(name!="reinit_discrete"&&name!="reinit_permutation")copy_input<int64_t>(input,out);
  auto lower=bounds.data(),upper=lower+d;
  if(name=="reinit_discrete") {
    auto column_owner=workspace_array<int64_t>(n);auto column=column_owner.mutable_data();
    for(py::ssize_t j=0;j<d;++j) {
      rng.integers(lower[j],upper[j]+1,column,n);
      for(py::ssize_t i=0;i<n;++i)out[i*d+j]=column[i];
    }
  } else if(name=="search_swap"||name=="search_swap_multi"||name=="search_scramble"||name=="search_insert") {
    for(py::ssize_t i=0;i<n;++i) {
      auto pair=rng.choice(d,2,false);int64_t a=pair[0],b=pair[1];
      if(name=="search_swap")std::swap(out[i*d+a],out[i*d+b]);
      else {
        if(a>b)std::swap(a,b);
        if(name=="search_swap_multi")std::reverse(out+i*d+a,out+i*d+b+1);
        else if(name=="search_scramble")rng.permutation(out+i*d+a,b-a+1);
        else {
          auto value=out[i*d+b];
          for(auto j=b;j>a+1;--j)out[i*d+j]=out[i*d+j-1];
          out[i*d+a+1]=value;
        }
      }
    }
  } else if(name=="search_reset_one"||name=="search_reset_n") {
    ScratchVector<int64_t> columns_lease;auto &mutable_columns=columns_lease.values;
    for(py::ssize_t j=0;j<d;++j)if(upper[j]>lower[j])mutable_columns.push_back(j);
    if(mutable_columns.empty())return output;
    auto reset=[&](py::ssize_t row,int64_t column) {
      auto value=rng.integer(lower[column],upper[column]+1);
      while(value==out[row*d+column])value=rng.integer(lower[column],upper[column]+1);
      out[row*d+column]=value;
    };
    if(name=="search_reset_one") {
      auto columns=workspace_array<int64_t>(n);rng.integers(0,mutable_columns.size(),columns.mutable_data(),n);
      for(py::ssize_t i=0;i<n;++i)reset(i,mutable_columns[columns.data()[i]]);
    } else {
      auto requested=parameters.empty()?1:static_cast<int64_t>(std::nearbyint(parameters[0]));
      auto count=std::min<int64_t>(std::max<int64_t>(requested,1),mutable_columns.size());
      for(py::ssize_t i=0;i<n;++i)
        for(auto j:rng.choice(mutable_columns.size(),count,false))reset(i,mutable_columns[j]);
    }
  } else if(name=="search_reset_rand") {
    double probability=parameters.empty()?.1:parameters[0];
    if(!std::isfinite(probability)||probability<0||probability>1)
      throw py::value_error("Reset probability must be finite and in [0, 1].");
    for(py::ssize_t j=0;j<d;++j)if(upper[j]<lower[j])
      throw py::value_error("Every upper bound must be at least its lower bound.");
    ScratchVector<double> coins_lease;auto &coins=coins_lease.values;rng.uniform(coins,n*d);
    for(py::ssize_t i=0;i<n;++i)for(py::ssize_t j=0;j<d;++j) {
      auto k=i*d+j;if(coins[k]<probability)out[k]=rng.integer(lower[j],upper[j]+1);
    }
  } else if(name=="search_reset_creep") {
    double probability=parameters.empty()?.1:parameters[0],ratio=parameters.size()<2?.1:parameters[1];
    auto step_owner=workspace_array<int64_t>(n*d),column_owner=workspace_array<int64_t>(n);auto step=step_owner.mutable_data(),column=column_owner.mutable_data();
    for(py::ssize_t j=0;j<d;++j) {
      int64_t amplitude=std::max<int64_t>(1,std::nearbyint((upper[j]-lower[j]+1)*ratio));
      rng.integers(1,amplitude+1,column,n);
      for(py::ssize_t i=0;i<n;++i)step[i*d+j]=column[i];
    }
    ScratchVector<double> sign_lease,mask_lease;auto &sign=sign_lease.values;auto &mask=mask_lease.values;rng.uniform(sign,n*d);rng.uniform(mask,n*d);
    for(py::ssize_t i=0;i<n;++i)for(py::ssize_t j=0;j<d;++j) {
      auto k=i*d+j;if(!(mask[k]<probability))continue;
      uint64_t value=static_cast<uint64_t>(out[k])+
          static_cast<uint64_t>(sign[k]<.5?-step[k]:step[k]);
      out[k]=std::min(std::max(static_cast<int64_t>(value),lower[j]),upper[j]);
    }
  } else throw py::value_error("Native discrete core not implemented: "+name);
  return output;
}

template<class Math,class Input>
NativeResult bit_variation(const std::string &name,
    const Input &input,
    py::array_t<int64_t,py::array::c_style> bounds,NativeState state,Random &rng,Math &math,bool publish_validity=false,bool bounds_validated=false) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0),d=input.shape(1);bool fastga=name.rfind("search_fastga_bit_",0)==0;
  auto law=std::string_view(name).substr(fastga?18:11);
  if(!d)throw py::value_error("Solution must be a non-empty 2-D matrix for "+name+".");
  for(py::ssize_t i=0;i<n;++i)if(!aol_x86::binary(input_row<int64_t>(input,i),d))
    throw py::value_error(name+" requires binary decisions in {0, 1}.");
  if(!bounds_validated)for(py::ssize_t j=0;j<d;++j)if(bounds.data()[j]!=0||bounds.data()[d+j]!=1)
    throw py::value_error(name+" requires binary [0, 1] problem bounds.");
  int64_t fixed=law=="one"?1:law=="three"?3:law=="five"?5:0;
  if(fixed>d)throw py::value_error("Cannot flip "+std::to_string(fixed)+" distinct bits in dimension "+std::to_string(d)+".");
  PreparedFragment *prepared=nullptr;
  auto output=py::reinterpret_borrow<py::array_t<int64_t>>(py::handle());
  if(fragment_sink&&packed_fragments()) {
    prepared=fragment_sink;prepared->configure(FragmentKind::Bit,n,d);prepared->integer=true;
    prepared->bits=std::make_unique<PrivateBits>(n,d);prepared->bits->pack(input);
  }else {output=py::array_t<int64_t>({n,d});copy_input<int64_t>(input,output.mutable_data());}
  py::array_t<bool> invalidated(n);std::fill_n(invalidated.mutable_data(),n,false);
  auto binomial=[&](double p) {
    if(!fastga)return rng.binomial(d,p);
    ScratchVector<double> coins_lease;auto &coins=coins_lease.values;rng.uniform(coins,d);
    return aol_x86::count_less(coins.data(),d,p);
  };
  ScratchVector<double> weights_lease;auto &weights=weights_lease.values;double total=0;
  const std::vector<double> *distribution=&weights;
  if(law=="fast" && d/2) {
    if(auto cached=math.cached_bit_weights(d,fastga))distribution=cached;
    else {
    weights.reserve(d/2);
    if(fastga) {
      for(py::ssize_t j=0;j<d/2;++j) {weights.push_back(std::pow(double(j+1),-1.5));total+=weights.back();}
    } else {
      py::array_t<double> values(d/2),exponent(1);
      std::iota(values.mutable_data(),values.mutable_data()+d/2,1.);exponent.mutable_data()[0]=-1.5;
      auto powered=py::cast<py::array_t<double>>(math.invoke(math.power,values,exponent));
      weights.assign(powered.data(),powered.data()+d/2);
      auto sum=PyArray_Sum(reinterpret_cast<PyArrayObject *>(powered.ptr()),AOL_RAVEL_AXIS,NPY_DOUBLE,nullptr);
      if(!sum)throw py::error_already_set();total=py::cast<double>(py::reinterpret_steal<py::object>(sum));
    }
    for(auto &x:weights)x/=total;
    math.remember_bit_weights(d,fastga,weights);
    }
  }
  const std::vector<double> *cdf=nullptr;bool checked_cdf=false;
  ScratchVector<int64_t> positions_lease;
  for(py::ssize_t i=0;i<n;++i) {
    int64_t count=fixed;
    if(law=="uniform")count=rng.integer(0,d);
    else if(law=="standard")count=binomial(1./d);
    else if(law=="conditional")do {count=binomial(1./d);} while(!count);
    else if(law=="shifted")count=std::max<int64_t>(1,binomial(.5));
    else if(law=="normal") {
      double dimension=d;count=std::trunc(rng.normal(1./d,math.logarithm.scalar(dimension)));
      if(count<0||count>=d)count=rng.integer(0,d);
    } else if(law=="fast") {
      double rate=1;
      if(d/2) {
        if(fastga) {
          double coin=rng.uniform();
          if(!checked_cdf){cdf=math.bit_cdf(*distribution,fastga);checked_cdf=true;}
          if(cdf){auto position=std::lower_bound(cdf->begin(),cdf->end(),coin);if(position!=cdf->end())rate=double(position-cdf->begin()+1)/d;}
          else {double cumulative=0;for(py::ssize_t j=0;j<d/2;++j){cumulative+=(*distribution)[j];if(cumulative>=coin){rate=double(j+1)/d;break;}}}
        } else {
          if(!checked_cdf){cdf=math.bit_cdf(*distribution,fastga);checked_cdf=true;}
          if(cdf){double coin;rng.uniform(&coin,1);rate=double(std::upper_bound(cdf->begin(),cdf->end(),coin)-cdf->begin()+1)/d;}
          else rate=double(rng.choice(d/2,1,true,distribution->data())[0]+1)/d;
        }
      } else if(fastga)rng.uniform();
      count=binomial(rate);
    }
    if(count) {
      auto &positions=positions_lease.values;rng.choice_into(positions,d,count,false);
      if(prepared)for(auto j:positions)prepared->bits->flip(i,j);
      else flip_bits(output.mutable_data()+i*d,positions);
      invalidated.mutable_data()[i]=true;
    }
  }
  if(fastga||publish_validity)state.set("_stream_invalidated_mask",invalidated);
  if(prepared){return state_result(py::none(),state);}
  auto result=state_result(output,state);result.binary_output=true;return result;
}
}

#pragma once
#include "selection.hpp"
#include "statistics.hpp"
namespace native_components {

inline py::array_t<double> native_statistic(const py::array &a,int axis,int operation,int ddof=0) {
  if(operation==1)return standard_deviation(a,axis,ddof);
  auto value=reduction(a,axis,operation==0);
  auto raw=PyArray_FromAny(value.ptr(),PyArray_DescrFromType(NPY_DOUBLE),0,0,NPY_ARRAY_CARRAY_RO,nullptr);
  if(!raw)throw py::error_already_set();return py::reinterpret_steal<py::array_t<double>>(raw);
}

template<class Math,class Input,class Population>
NativeResult continuous_variation(const std::string &name,
    const Input &input,
    const Population &population,
    py::array_t<double,py::array::c_style> bounds,
    const std::vector<double> &parameters,NativeState state,int64_t inner,Random &rng,Math &math,bool private_input=false) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0),d=input.shape(1);auto lower=bounds.data(),upper=lower+d;
  if(fragment_sink&&(name=="reinit_continuous"||name=="search_mu_uniform")) {
    auto kind=name=="reinit_continuous"?FragmentKind::Reinit:FragmentKind::Uniform;
    auto prepared=fragment_sink;prepared->configure(kind,n,d);
    if(kind==FragmentKind::Uniform)prepared->set_parents(input);
    prepared->bound_owner=bounds;prepared->lower=lower;prepared->upper=upper;
    prepared->probability=parameters.empty()?.1:parameters[0];
    if(kind==FragmentKind::Uniform){rng.uniform(prepared->masks.values,n*d);}
    rng.uniform(prepared->coins.values,n*d);
    return state_result(py::none(),state);
  }
  py::array_t<double> output({n,d});auto out=output.mutable_data();
  bool direct_input=private_input&&n*d<NumericInterval::release_threshold&&(name=="search_mu_uniform"||name=="search_mu_gaussian"||name=="search_mu_cauchy");
  if(direct_input&&name!="search_mu_uniform") {
    auto scale=state.object(name=="search_mu_gaussian"?"gaussian_eta":"cauchy_eta");
    // A conversion callback or GIL release can expose/change the source after
    // the original snapshot point. Such calls retain the copy-first path.
    direct_input=scale.is_none()||(PyArray_CheckExact(scale.ptr())&&PyArray_TYPE(reinterpret_cast<PyArrayObject *>(scale.ptr()))==NPY_DOUBLE&&
      !(Py_TYPE(scale.ptr())->tp_weaklistoffset&&*PyObject_GET_WEAKREFS_LISTPTR(scale.ptr())));
  }
  if(name!="reinit_continuous"&&name!="search_eda"&&!direct_input)copy_input<double>(input,out);
  if(name=="reinit_continuous"||name=="search_mu_uniform") {
    NumericInterval interval(n*d);
    ScratchVector<double> mask_lease;auto &mask=mask_lease.values;
    if(name=="search_mu_uniform") {rng.uniform(mask,n*d);}
    ScratchVector<double> coins_lease;auto &coins=coins_lease.values;rng.uniform(coins,n*d);
    double probability=parameters.empty()?.1:parameters[0];
    for(py::ssize_t i=0;i<n;++i)aol_x86::scalar_each(d,[&](size_t j) {
      auto k=i*d+j;if(mask.empty()||mask[k]<probability)out[k]=uniform_coordinate(lower[j],upper[j],coins[k]);
      else if(direct_input)out[k]=input_row<double>(input,i)[j];
    });
  } else if(name=="search_mu_gaussian"||name=="search_mu_cauchy") {
    const char *key=name=="search_mu_gaussian"?"gaussian_eta":"cauchy_eta";
    auto eta=py::reinterpret_borrow<py::array_t<double>>(py::handle());
    bool found=false;
    if(inner!=1 && state.contains(key) && !state.object(key).is_none()) {
      eta=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(state.object(key));
      found=eta.ndim()==2&&eta.shape(0)==n&&eta.shape(1)==d;
    }
    if(!found) {eta=py::array_t<double>({n,d});rng.uniform(eta.mutable_data(),n*d);state.set(key,eta);}
    ScratchVector<double> disturbance_lease;auto &disturbance=disturbance_lease.values;
    {NumericInterval interval(n*d);auto scale=eta.data();
      if(name=="search_mu_gaussian")rng.normal(disturbance,n*d);
      else {disturbance.reserve(n*d);for(py::ssize_t k=0;k<n*d;++k)disturbance.push_back(rng.cauchy());}
      if(direct_input) {
        for(py::ssize_t i=0;i<n;++i) {
          auto source=input_row<double>(input,i);
          for(py::ssize_t j=0;j<d;++j){auto k=i*d+j;out[k]=source[j]+scale[k]*disturbance[k];}
        }
      }else aol_x86::add_scaled(out,scale,disturbance.data(),n*d);}
    auto tau=math.normal_parameters(d);double tau1=tau[0],tau2=tau[1];
    ScratchVector<double> row_lease,coordinate_lease;auto &row=row_lease.values;auto &coordinate=coordinate_lease.values;
    auto exponent=workspace_shaped<double>({n,d});
    auto e=exponent.mutable_data();
    {NumericInterval interval(n*d);rng.normal(row,n);rng.normal(coordinate,n*d);
      for(py::ssize_t i=0;i<n;++i)aol_x86::scalar_each(d,[&](size_t j) {auto k=i*d+j;e[k]=tau2*row[i]+tau1*coordinate[k];});}
    math.exponential.execute_into(exponent,exponent);py::array_t<double> next({n,d});
    {auto output=next.mutable_data();auto scale=eta.data(),factor=exponent.data();NumericInterval interval(n*d);
      aol_x86::scalar_each(n*d,[&](size_t k){output[k]=scale[k]*factor[k];});}
    state.set(key,next);
  } else if(name=="search_eda") {
    auto model_n=population.shape(0);auto model=input_dense(population);
    auto mean=py::reinterpret_borrow<py::array_t<double>>(py::handle()), deviation=py::reinterpret_borrow<py::array_t<double>>(py::handle()), fallback=py::reinterpret_borrow<py::array_t<double>>(py::handle());
    if(pure_native_reductions()) {
      NativeMoments moments(model,0);mean=moments.mean;deviation=std::move(moments.sum);
      // Only the final effective scale is consumed by EDA. Reuse squared sums
      // and evaluate population ddof only when the sample scale needs fallback.
      double denominator=std::max<py::ssize_t>(model_n-(model_n>1?1:0),0);
      for(py::ssize_t j=0;j<d;++j) {
        double sum=deviation.data()[j],scale=std::sqrt(sum/denominator);
        if(!std::isfinite(scale)||scale<=0)scale=std::sqrt(sum/std::max<py::ssize_t>(model_n,0));
        deviation.mutable_data()[j]=scale>0?scale:1e-12;
      }
    } else {
      mean=native_statistic(model,0,0);deviation=native_statistic(model,0,1,model_n>1?1:0);
      fallback=native_statistic(model,0,1,0);
    }
    NumericInterval interval(n*d);
    for(py::ssize_t i=0;i<n;++i)for(py::ssize_t j=0;j<d;++j) {
      auto k=i*d+j;double scale=deviation.data()[j];
      if(fallback&&(!std::isfinite(scale)||scale<=0))scale=fallback.data()[j];
      if(!(scale>0))scale=1e-12;
      out[k]=rng.normal(mean.data()[j],scale);
    }
  } else if(name=="search_mu_polynomial") {
    double probability=parameters.empty()?.1:parameters[0],distribution=parameters.size()<2?30:parameters[1];
    ScratchVector<double> sites_lease,coins_lease;auto &sites=sites_lease.values;auto &coins=coins_lease.values;rng.uniform(sites,n*d);rng.uniform(coins,n*d);
    bool pure_power=native_floating_depth&&Py_TYPE(math.power.ptr())==&PyUFunc_Type;
    ScratchVector<int64_t> left_lease,right_lease;
    if(pure_power)for(py::ssize_t i=0;i<n;++i)for(py::ssize_t j=0;j<d;++j) {
      auto k=i*d+j;if(sites[k]<probability&&upper[j]>lower[j])
        (coins[k]<=.5?left_lease.values:right_lease.values).push_back(k);
    }
    for(bool left:{true,false}) {
      ScratchVector<int64_t> positions_lease;
      auto &positions=pure_power?(left?left_lease.values:right_lease.values):positions_lease.values;
      if(!pure_power)for(py::ssize_t k=0;k<n*d;++k)
        if(sites[k]<probability && upper[k%d]>lower[k%d] && (coins[k]<=.5)==left)positions.push_back(k);
      auto base=pure_power?workspace_shaped<double>({static_cast<py::ssize_t>(positions.size())}):py::array_t<double>(positions.size());
      py::array_t<double> power(1);power.mutable_data()[0]=distribution+1;
      for(size_t i=0;i<positions.size();++i) {
        auto k=positions[i],j=k%d;double span=upper[j]-lower[j];
        base.mutable_data()[i]=1.-(left?out[k]-lower[j]:upper[j]-out[k])/span;
      }
      auto powered=py::cast<py::array_t<double>>(math.invoke(math.power,base,power));
      for(size_t i=0;i<positions.size();++i) {
        auto k=positions[i];double u=coins[k],p=powered.data()[i];
        base.mutable_data()[i]=left?2.*u+(1.-2.*u)*p:2.*(1.-u)+2.*(u-.5)*p;
      }
      power.mutable_data()[0]=1./(distribution+1.);
      powered=py::cast<py::array_t<double>>(math.invoke(math.power,base,power));
      for(size_t i=0;i<positions.size();++i) {
        auto k=positions[i],j=k%d;
        out[k]=out[k]+(upper[j]-lower[j])*(left?powered.data()[i]-1.:1.-powered.data()[i]);
      }
    }
  } else throw py::value_error("Native continuous core not implemented: "+name);
  return state_result(output,state);
}

template<class Math,class Input>
py::array continuous_crossover(const std::string &name,
    const Input &input,const std::vector<double> &parameters,Random &rng,Math &math) {
  RegionScope numeric_timing(Region::Numeric);
  if(input.ndim()!=2)throw py::value_error("Expected two-dimensional parents");
  if(name=="cross_arithmetic"&&!parameters.empty()&&parameters.size()!=1&&parameters.size()!=static_cast<size_t>(input.shape(1)))
    throw py::value_error("Arithmetic ratio must be scalar or one value per coordinate");
  auto n=input.shape(0),d=input.shape(1),half=(n+1)/2;
  const bool prepare=fragment_sink&&(name=="cross_arithmetic"||(native_floating_depth&&Py_TYPE(math.power.ptr())==&PyUFunc_Type));
  auto backing=py::reinterpret_borrow<py::array_t<double>>(py::handle());double *out=nullptr;
  if(!prepare){backing=py::array_t<double>({2*half,d});out=backing.mutable_data();}
  bool vector=aol_x86::dense_enabled(d,half*d);
  if(name=="cross_arithmetic") {
    // Reject the unprofitable kernel once, rather than redispatching every pair.
    if(vector&&aol_x86::mode()!=aol_x86::Mode::Simd&&aol_x86::mode()!=aol_x86::Mode::Packed)vector=false;
    const bool scalar=parameters.size()<=1;double ratio=parameters.empty()?.5:parameters[0];
    if(prepare) {
      auto prepared=fragment_sink;prepared->configure(FragmentKind::Arithmetic,n,d);prepared->set_parents(input);prepared->probability=ratio;
      if(!scalar){py::array_t<double> coefficients(d);std::copy(parameters.begin(),parameters.end(),coefficients.mutable_data());prepared->coefficients=coefficients;}
      return py::reinterpret_borrow<py::array>(py::none());
    }
    for(py::ssize_t i=0;i<half;++i) {
      auto first=input_row<double>(input,i),second=input_row<double>(input,n/2+i);
      aol_x86::pair(first,second,out+i*d,out+(half+i)*d,scalar?nullptr:parameters.data(),ratio,d,false,vector);
    }
  } else {
    py::array_t<double> beta({half,d});
    double eta=parameters.empty()?20:parameters[0];ScratchVector<double> mu_lease;auto &mu=mu_lease.values;rng.uniform(mu,half*d);
    for(bool left:{true,false}) {
      ScratchVector<int64_t> positions_lease;auto &positions=positions_lease.values;
      for(py::ssize_t k=0;k<half*d;++k)if((mu[k]<=.5)==left)positions.push_back(k);
      py::array_t<double> base(positions.size()),power(1);power.mutable_data()[0]=(left?1.:-1.)/(1.+eta);
      for(size_t i=0;i<positions.size();++i)base.mutable_data()[i]=left?2.*mu[positions[i]]:2.-2.*mu[positions[i]];
      auto result=py::cast<py::array_t<double>>(math.invoke(math.power,base,power));
      for(size_t i=0;i<positions.size();++i)beta.mutable_data()[positions[i]]=result.data()[i];
    }
    if(prepare) {
      auto prepared=fragment_sink;prepared->configure(FragmentKind::SBX,n,d);prepared->set_parents(input);prepared->coefficients=beta;
      return py::reinterpret_borrow<py::array>(py::none());
    }
    for(py::ssize_t i=0;i<half;++i) {
      auto first=input_row<double>(input,i),second=input_row<double>(input,n/2+i);
      aol_x86::pair(first,second,out+i*d,out+(half+i)*d,beta.data()+i*d,0.,d,true,vector);
    }
  }
  return py::array(py::dtype::of<double>(),{n,d},{d*py::ssize_t(8),py::ssize_t(8)},backing.data(),backing);
}
}

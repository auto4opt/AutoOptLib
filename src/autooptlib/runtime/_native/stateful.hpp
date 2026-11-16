#pragma once
#include "continuous.hpp"
#include "differential.hpp"
namespace native_components {

inline NativeResult imperialist(py::array_t<double,py::array::c_style> input,
    py::array_t<double,py::array::c_style> fitness,
    py::array_t<double,py::array::c_style> bounds,
    const std::vector<double> &parameters,NativeState state,Random &rng,bool private_input=false) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0),d=input.shape(1);if(!n)return state_result(input,state);
  double assimilation=parameters.empty()?.6:parameters[0],revolution=parameters.size()<2?.5:parameters[1];
  int64_t count=state.contains("imperialist_count")?state.number<int64_t>("imperialist_count"):static_cast<int64_t>(std::sqrt(n));
  count=std::max<int64_t>(1,std::min<int64_t>(count,n));auto ranked=order(fitness);
  auto lower=bounds.data(),upper=lower+d;
  py::array_t<double> output({n,d});auto output_data=output.mutable_data();
  if(!private_input)std::copy_n(input.data(),n*d,output.mutable_data());
  else for(int64_t i=0;i<count;++i)std::copy_n(input.data()+ranked.data()[i]*d,d,output.mutable_data()+ranked.data()[i]*d);
  auto colonies=n-count;
  if(colonies*d) {
    auto assigned_owner=workspace_array<int64_t>(colonies);auto assigned=assigned_owner.mutable_data();rng.integers(0,count,assigned,colonies);
    ScratchVector<double> scale_lease;auto &scale=scale_lease.values;rng.uniform(scale,colonies*d);
    for(py::ssize_t i=0;i<colonies;++i) {
     auto source=ranked.data()[count+i],target=ranked.data()[assigned[i]];
     aol_x86::scalar_each(d,[&](size_t j) {
      double x=input.data()[source*d+j],weight=assimilation*scale[i*d+j];
      double moved=x+weight*(input.data()[target*d+j]-x);
      output_data[source*d+j]=std::min(std::max(moved,lower[j]),upper[j]);
     });
    }
  }
  ScratchVector<double> mask_lease;auto &mask=mask_lease.values;rng.uniform(mask,n*d);
  if(std::any_of(mask.begin(),mask.end(),[&](double x){return x<revolution;})) {
    ScratchVector<double> coins_lease;auto &coins=coins_lease.values;rng.uniform(coins,n*d);
    for(py::ssize_t i=0;i<n;++i)aol_x86::scalar_each(d,[&](size_t j) {
      auto k=i*d+j;if(mask[k]<revolution)output_data[k]=lower[j]+(upper[j]-lower[j])*coins[k];
    });
  }
  return state_result(output,state);
}

template<class Personal,class Input>
NativeResult particle_fly_rows(const Input &input,
    py::array_t<double> velocity,
    const Personal &personal,
    py::array_t<double,py::array::c_style> guide,
    py::array_t<int64_t,py::array::c_style> targets,double inertia,
    NativeState state,bool context,Random &rng) {
  RegionScope numeric_timing(Region::Numeric);
  if(input.ndim()!=2||targets.size()!=input.shape(0))
    throw py::value_error("Expected one particle target index per input row");
  auto t=input.shape(0),d=input.shape(1);
  ScratchVector<double> r1_lease,r2_lease;auto &r1=r1_lease.values;auto &r2=r2_lease.values;rng.uniform(r1,t);rng.uniform(r2,t);
  // The reference draws r1/r2 before NumPy indexes velocity/personal state.
  if(velocity.ndim()!=2||personal.ndim()!=2||velocity.shape(1)!=d
      ||(personal.shape(1)!=d&&personal.shape(1)!=1)||(guide.size()!=d&&guide.size()!=1))
    throw py::value_error("Incompatible particle state dimensions");
  for(py::ssize_t i=0;i<t;++i)
    if(targets.data()[i]<0||targets.data()[i]>=velocity.shape(0)||targets.data()[i]>=personal.shape(0))
      throw py::index_error("Particle target index exceeds the velocity or personal-best rows");
  auto old_velocity=velocity.unchecked<2>();
  ScratchVector<double> next_lease;auto &next=next_lease.values;next.reserve(t*d);py::array_t<double> output({t,d});auto output_data=output.mutable_data();
  for(py::ssize_t i=0;i<t;++i) {
   auto row=input_row<double>(input,i);double cognitive=2.*r1[i],social=2.*r2[i];
   auto personal_row=personal.row(targets.data()[i]);
   auto coordinate=[&](size_t j) {
    auto k=i*d+j;
    double first=inertia*old_velocity(targets.data()[i],j);
    double second=cognitive*(personal_row[personal.shape(1)==1?0:j]-row[j]);
    double third=social*(guide.data()[guide.size()==1?0:j]-row[j]);
    double value=(first+second)+third;
    next.push_back(value);output_data[k]=row[j]+value;
   };
   aol_x86::scalar_each(d,coordinate);
  }
  // Commit after every result has been calculated: repeated indices must all
  // read the old velocity, with the final assignment winning as in NumPy.
  auto writable_velocity=velocity.mutable_unchecked<2>();
  for(py::ssize_t i=0;i<t;++i)for(py::ssize_t j=0;j<d;++j)
    writable_velocity(targets.data()[i],j)=next[i*d+j];
  state.set("V",velocity);
  if(context) {
    state.record_targets(targets.data(),t);
  }
  return state_result(output,state);
}
inline NativeResult particle_fly(py::array_t<double,py::array::c_style> input,
    py::array_t<double> velocity,py::array_t<double,py::array::c_style> personal,
    py::array_t<double,py::array::c_style> guide,
    py::array_t<int64_t,py::array::c_style> targets,double inertia,
    NativeState state,bool context,Random &rng) {
  return particle_fly_rows(input,velocity,DenseDecisionRows{personal},guide,targets,inertia,state,context,rng);
}

}

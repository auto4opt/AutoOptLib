#pragma once
#include "clustering.hpp"
namespace native_components {

inline py::array_t<double> archive_statistics(
    py::array_t<double,py::array::c_style> fitness,
    py::array_t<double,py::array::c_style> previous,bool private_input=false) {
  RegionScope numeric_timing(Region::Numeric);
  // Tiny statistical records do not repay pooled ndarray view preparation.
  ArrayWorkspaceContext owned_statistics(nullptr);
  auto rows=previous.size()/2;py::array_t<double> result({rows+1,py::ssize_t(2)});
  std::copy_n(previous.data(),rows*2,result.mutable_data());
  auto mean=reduction(fitness,AOL_RAVEL_AXIS,true);
  result.mutable_data()[rows*2]=py::cast<double>(mean);
  if(fitness.size()>1){NativeMoments moments(fitness,AOL_RAVEL_AXIS,private_input&&pure_native_reductions()?mean:py::none());result.mutable_data()[rows*2+1]=moments.deviation(1).data()[0];}
  else result.mutable_data()[rows*2+1]=0.;
  return result;
}

template<class Math>
py::array_t<int64_t> archive_diverse(py::array_t<double,py::array::c_style> input,
                                    int64_t requested,bool continuous,Random &rng,Math &math) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0),d=input.shape(1);if(!n)return py::array_t<int64_t>(0);
  auto distance=py::reinterpret_borrow<py::array_t<double>>(py::handle());
  if(continuous) {
    auto sums=squared_distances(input,input);distance=math.square_root.execute(sums);
  } else {
    distance=py::array_t<double>({n,n});
    const auto words=(d+63)/64;std::optional<ScratchVector<int64_t>> packed;bool binary=false;
    if(aol_x86::use_packed(n,d)){
      packed.emplace(n*words,0);binary=true;
      for(py::ssize_t i=0;i<n&&binary;++i)binary=aol_x86::pack(input.data()+i*d,reinterpret_cast<uint64_t *>(packed->values.data()+i*words),d);
    }
    for(py::ssize_t i=0;i<n;++i)for(py::ssize_t j=0;j<n;++j) {
      auto different=binary?aol_x86::packed_hamming(reinterpret_cast<const uint64_t *>(packed->values.data()+i*words),reinterpret_cast<const uint64_t *>(packed->values.data()+j*words),words):aol_x86::hamming(input.data()+i*d,input.data()+j*d,d);
      distance.mutable_data()[i*n+j]=double(different)/d;
    }
  }
  std::vector<int64_t> chosen{rng.integer(0,n)};
  ScratchVector<bool> selected_lease(n,false);auto &selected=selected_lease.values;selected[chosen[0]]=true;
  while(static_cast<int64_t>(chosen.size())<requested && chosen.size()<static_cast<size_t>(n)) {
    auto rows=rows_copy(distance,chosen.data(),chosen.size());auto sums=py::cast<py::array_t<double>>(reduction(rows,0));
    auto ranked=order(sums);
    for(py::ssize_t j=n;j>0;--j) {
      auto i=ranked.data()[j-1];
      if(!selected[i]) {chosen.push_back(i);selected[i]=true;break;}
    }
  }
  if(requested>=0 && static_cast<size_t>(requested)<chosen.size())chosen.resize(requested);
  return indices_array(chosen.data(),chosen.size());
}
}

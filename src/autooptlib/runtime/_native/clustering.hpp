#pragma once
#include "statistics.hpp"
#include "selection.hpp"
#include "replacement.hpp"
#include <unordered_map>
namespace native_components {

inline py::array_t<double> rows_copy(const py::array_t<double> &input,const int64_t *rows,size_t count) {
  auto d=input.shape(1);py::array_t<double> out({static_cast<py::ssize_t>(count),d});
  for(size_t i=0;i<count;++i)std::copy_n(input.data()+rows[i]*d,d,out.mutable_data()+i*d);
  return out;
}
// Only call where the next consumer cannot observe ndarray ownership. Public
// numeric-op callbacks retain rows_copy's independent owning input.
inline py::array_t<double> private_rows_copy(const py::array_t<double> &input,const int64_t *rows,size_t count) {
  auto d=input.shape(1);auto out=workspace_shaped<double>({static_cast<py::ssize_t>(count),d});
  for(size_t i=0;i<count;++i)std::copy_n(input.data()+rows[i]*d,d,out.mutable_data()+i*d);
  return out;
}

// Keep the original per-pair subtraction, squaring, and last-axis NumPy
// reduction. Bound only the temporary square tensor, not the final distances.
inline py::array_t<double> squared_distances(const py::array_t<double> &a,const py::array_t<double> &b) {
  auto n=a.shape(0),m=b.shape(0),d=a.shape(1);
  auto square_kernel=aol_x86::square_kernel(d,n*m*d);
  constexpr py::ssize_t scratch_bytes=1024*1024;
  if(!pure_native_reductions()) {
    py::array_t<double> square({n,m,d});auto square_data=square.mutable_data();
    for(py::ssize_t i=0;i<n;++i)for(py::ssize_t j=0;j<m;++j)
      square_kernel(a.data()+i*d,b.data()+j*d,square_data+(i*m+j)*d,d);
    return py::cast<py::array_t<double>>(reduction(square,2));
  }
  // A portable memory cap, independent of CPU model or a calibrated cost table.
  py::array_t<double> sums({n,m});
  auto capacity=std::max<py::ssize_t>(1,std::min(n*m,scratch_bytes/(8*std::max<py::ssize_t>(d,1))));
  auto square=workspace_shaped<double>({capacity,d});auto square_data=square.mutable_data();
  auto view=py::array_t<double>({capacity,d},{d*py::ssize_t(8),py::ssize_t(8)},square.data(),square);
  for(py::ssize_t begin=0;begin<n*m;begin+=capacity) {
    auto count=std::min(capacity,n*m-begin);
    for(py::ssize_t row=0;row<count;++row) {
      auto i=(begin+row)/m,j=(begin+row)%m;auto first=a.data()+i*d,second=b.data()+j*d;
      square_kernel(first,second,square_data+row*d,d);
    }
    PyArray_DIMS(reinterpret_cast<PyArrayObject *>(view.ptr()))[0]=count;
    auto destination=py::array_t<double>({count},{py::ssize_t(8)},sums.mutable_data()+begin,sums);
    auto reduced=PyArray_Sum(reinterpret_cast<PyArrayObject *>(view.ptr()),1,NPY_DOUBLE,
                             reinterpret_cast<PyArrayObject *>(destination.ptr()));
    if(!reduced)throw py::error_already_set();Py_DECREF(reduced);
  }
  return sums;
}

template<class Math>
py::array_t<double> euclidean_pairs(py::array_t<double,py::array::c_style> input,Math &math) {
  auto n=input.shape(0),d=input.shape(1);double scale=1;
  for(py::ssize_t k=0;k<n*d;++k) {
    if(!std::isfinite(input.data()[k]))throw py::value_error("Pairwise distances require finite coordinates.");
    scale=std::max(scale,std::abs(input.data()[k]));
  }
  auto normal=workspace_shaped<double>({n,d});
  for(py::ssize_t k=0;k<n*d;++k)normal.mutable_data()[k]=input.data()[k]/scale;
  PyArrayObject *arguments[]={reinterpret_cast<PyArrayObject *>(normal.ptr()),reinterpret_cast<PyArrayObject *>(normal.ptr())};
  auto norm_raw=PyArray_EinsteinSum(const_cast<char *>("ij,ij->i"),2,arguments,nullptr,NPY_KEEPORDER,NPY_SAFE_CASTING,nullptr);
  if(!norm_raw)throw py::error_already_set();
  auto norms=py::reinterpret_steal<py::array_t<double>>(reinterpret_cast<PyObject *>(norm_raw));
  auto trans_raw=PyArray_Transpose(reinterpret_cast<PyArrayObject *>(normal.ptr()),nullptr);
  if(!trans_raw)throw py::error_already_set();
  auto trans=py::reinterpret_steal<py::array>(trans_raw);
  auto gram=py::cast<py::array_t<double>>(math.invoke(math.matmul,normal,trans));
  auto squared=workspace_shaped<double>({n,n});
  for(py::ssize_t i=0;i<n;++i)for(py::ssize_t j=0;j<n;++j) {
    double value=(norms.data()[i]+norms.data()[j])-2.*gram.data()[i*n+j];
    squared.mutable_data()[i*n+j]=std::max(value,0.);
  }
  auto result=math.square_root.execute(squared);
  for(py::ssize_t k=0;k<n*n;++k) {
    double value=result.data()[k]*scale;
    if(std::isnan(value))value=0;
    else if(value==INFINITY)value=std::numeric_limits<double>::max();
    else if(value==-INFINITY)value=0;
    result.mutable_data()[k]=value;
  }
  for(py::ssize_t i=0;i<n;++i)result.mutable_data()[i*n+i]=0;
  return result;
}

template<class Math>
py::array_t<int64_t> niche(py::array_t<double,py::array::c_style> input,
    py::array_t<double,py::array::c_style> fits,int64_t generation,int64_t maximum,Random &rng,Math &math) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0);auto ranked=order(fits,true);
  auto values=private_rows_copy(input,ranked.data(),n);auto distance=euclidean_pairs(values,math);
  std::vector<int64_t> nearest,follow(n,1),seeds{0};nearest.reserve(n);if(n)nearest.push_back(-1);
  py::array_t<double> lengths(n);if(n)lengths.mutable_data()[0]=0.;
  for(py::ssize_t i=1;i<n;++i) {
    int64_t best=0;
    for(py::ssize_t j=1;j<i;++j)if(distance.data()[i*n+j]<distance.data()[i*n+best])best=j;
    nearest.push_back(best);lengths.mutable_data()[i]=distance.data()[i*n+best];
  }
  for(py::ssize_t i=n;i>1;--i)follow[nearest[i-1]]+=follow[i-1];
  double mean=0;
  if(n>1) {
    auto tail=pure_native_reductions()?py::array_t<double>({n-1},{py::ssize_t(8)},lengths.data()+1,lengths):py::array_t<double>(n-1);
    if(tail.owndata())std::copy_n(lengths.data()+1,n-1,tail.mutable_data());
    mean=py::cast<double>(reduction(tail,AOL_RAVEL_AXIS,true));
  }
  auto length_order=order(lengths);
  int64_t minimum=std::floor(3.+double(generation)/std::max<int64_t>(maximum,1)*7.+.5);
  for(py::ssize_t k=n;k>0;--k) {
    auto node=length_order.data()[k-1];
    if(lengths.data()[node]<=mean||nearest[node]<0)continue;
    std::vector<int64_t> chain{nearest[node]};auto top=nearest[node];
    while(nearest[top]!=-1){top=nearest[top];chain.push_back(top);}
    if(follow[node]>=minimum && follow[top]-follow[node]>=minimum) {
      nearest[node]=-1;lengths.mutable_data()[node]=0;seeds.push_back(node);
      for(auto parent:chain)follow[parent]-=follow[node];
    }
  }
  std::vector<int64_t> roots,indices;roots.reserve(n);
  for(py::ssize_t i=0;i<n;++i) {auto top=i;while(nearest[top]!=-1)top=nearest[top];roots.push_back(top);}
  std::vector<std::vector<int64_t>> groups(n);
  for(py::ssize_t i=0;i<n;++i)groups[roots[i]].push_back(i);
  for(auto seed:seeds) {
    auto &members=groups[seed];
    if(members.empty())continue;
    rng.permutation(members.data(),members.size());
    for(auto member:members)indices.push_back(ranked.data()[member]);
  }
  bool odd=indices.size()%2;if(odd)indices.push_back(indices.back());
  py::array_t<int64_t> result(indices.size()-(odd?1:0));size_t count=0;
  for(size_t i=0;i<indices.size();i+=2)result.mutable_data()[count++]=indices[i];
  for(size_t i=1;i<indices.size()&&count<static_cast<size_t>(result.size());i+=2)result.mutable_data()[count++]=indices[i];
  return result;
}

template<class Math>
py::array_t<int64_t> brainstorm(py::array_t<double,py::array::c_style> input,
    py::array_t<double,py::array::c_style> fits,int64_t clusters,double gamma,int64_t count,Random &rng,Math &math) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0),d=input.shape(1);if(!n)return py::array_t<int64_t>(0);
  clusters=std::min<int64_t>(std::max<int64_t>(1,clusters),n);
  double scale=1;
  for(py::ssize_t k=0;k<n*d;++k) {
    if(!std::isfinite(input.data()[k]))throw py::value_error("Brainstorm clustering requires finite decision values.");
    scale=std::max(scale,std::abs(input.data()[k]));
  }
  auto working=workspace_shaped<double>({n,d});
  for(py::ssize_t k=0;k<n*d;++k)working.mutable_data()[k]=input.data()[k]/scale;
  auto centers=private_rows_copy(working,rng.choice(n,clusters,false).data(),clusters);std::vector<int64_t> labels(n,-1);
  std::vector<std::vector<int64_t>> groups(clusters);
  for(int iteration=0;iteration<50;++iteration) {
    auto sums=squared_distances(working,centers);auto distances=math.square_root.execute(sums);
    std::vector<int64_t> next;next.reserve(n);
    for(py::ssize_t i=0;i<n;++i) {
      int64_t best=0;for(int64_t j=1;j<clusters;++j)
        if(distances.data()[i*clusters+j]<distances.data()[i*clusters+best])best=j;
      next.push_back(best);
    }
    if(next==labels)break;labels=std::move(next);
    for(auto &group:groups)group.clear();
    for(py::ssize_t i=0;i<n;++i)groups[labels[i]].push_back(i);
    for(int64_t j=0;j<clusters;++j) {
      const auto &members=groups[j];
      if(members.empty())std::copy_n(working.data()+rng.integer(0,n)*d,d,centers.mutable_data()+j*d);
      else {
        if(pure_native_reductions()) {
          auto rows=private_rows_copy(working,members.data(),members.size());
          auto destination=py::array_t<double>({d},{py::ssize_t(8)},centers.mutable_data()+j*d,centers);
          auto mean=PyArray_Mean(reinterpret_cast<PyArrayObject *>(rows.ptr()),0,NPY_DOUBLE,reinterpret_cast<PyArrayObject *>(destination.ptr()));
          if(!mean)throw py::error_already_set();Py_DECREF(mean);
        }else {
          auto rows=rows_copy(working,members.data(),members.size());
          auto mean=py::cast<py::array_t<double>>(reduction(rows,0,true));
          std::copy_n(mean.data(),d,centers.mutable_data()+j*d);
        }
      }
    }
  }
  std::vector<int64_t> center_ids;center_ids.reserve(clusters);auto members=std::move(groups);
  py::array_t<double> weights(clusters);
  for(int64_t j=0;j<clusters;++j) {
    if(members[j].empty())members[j].push_back(rng.integer(0,n));
    auto best=members[j][0];for(auto i:members[j])if(std::isnan(fits.data()[i])||fits.data()[i]<fits.data()[best]) {
      best=i;if(std::isnan(fits.data()[i]))break;
    }
    center_ids.push_back(best);weights.mutable_data()[j]=double(members[j].size())/n;
    members[j].erase(std::remove(members[j].begin(),members[j].end(),best),members[j].end());
  }
  double total=py::cast<double>(reduction(weights,AOL_RAVEL_AXIS));
  for(int64_t j=0;j<clusters;++j)weights.mutable_data()[j]=total<=0?1./clusters:weights.data()[j]/total;
  std::unordered_map<int64_t,std::vector<int64_t>> remaining_by_center;
  for(int64_t j=0;j<clusters;++j)remaining_by_center[center_ids[j]]=members[j];
  auto chosen=rng.choice(clusters,count,true,weights.data());
  for(auto &cluster:chosen) {
    auto center=center_ids[cluster];const auto &rest=remaining_by_center.at(center);
    if(!rest.empty()&&rng.uniform()<=gamma)center=rest[rng.integer(0,rest.size())];
    cluster=center;
  }
  return indices_array(chosen.data(),chosen.size());
}
}

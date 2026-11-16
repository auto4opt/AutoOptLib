// Precompiled pure numerical intervals. No RNG, Python callback or state
// commit occurs in these loops. All refusal is before output publication.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <cfenv>
#include <limits>
#if defined(__x86_64__) || defined(_M_X64)
#include <xmmintrin.h>
#endif

namespace cpu_blocks {
namespace py = pybind11;

inline py::object numeric_environment() {
  if(std::fegetround()!=FE_TONEAREST)return py::none();
#if defined(__x86_64__) || defined(_M_X64)
  auto control=_mm_getcsr() & ~0x3fu; // ignore sticky exception flags only
  if(control!=0x1f80u)return py::none(); // traps/rounding/FTZ/DAZ unsupported
  return py::make_tuple("x86-sse",control);
#elif defined(__aarch64__)
  uint64_t control;
  asm volatile("mrs %0, fpcr" : "=r"(control));
  if(control!=0)return py::none();
  return py::make_tuple("arm64-fpcr",control);
#else
  return py::none();
#endif
}

struct Kernels {
  py::object array_type;
  Kernels() : array_type(py::module_::import("numpy").attr("ndarray")) {}

inline bool ordinary(py::handle value, py::array &a, int ndim) {
  if (reinterpret_cast<PyObject *>(Py_TYPE(value.ptr()))!=array_type.ptr()) return false;
  a = py::reinterpret_borrow<py::array>(value);
  return a.ndim() == ndim && (a.flags() & py::array::c_style)
         && a.itemsize()>0
         && (reinterpret_cast<uintptr_t>(a.data()) % a.itemsize() == 0)
         && (a.dtype().byteorder() == '=' || a.dtype().byteorder() == '|');
}
inline bool doubles(py::handle value, py::array &a, int ndim = 2) {
  return ordinary(value, a, ndim) && a.dtype().is(py::dtype::of<double>());
}
inline bool integers(py::handle value, py::array &a, int ndim = 1) {
  return ordinary(value, a, ndim) && a.dtype().is(py::dtype::of<int64_t>());
}
inline bool bounded(const py::array &a) {
  auto p = static_cast<const double *>(a.data());
  auto size=a.size();
  for (py::ssize_t i = 0; i < size; ++i)
    if (!std::isfinite(p[i]) || std::abs(p[i]) > 1e100) return false;
  return true;
}
inline bool bounded_rows(const py::array &a, const int64_t *rows, py::ssize_t n) {
  auto p=static_cast<const double *>(a.data());
  auto d=a.shape(1);
  for(py::ssize_t i=0;i<n;++i)for(py::ssize_t j=0;j<d;++j) {
    double value=p[rows[i]*d+j];
    if(!std::isfinite(value)||std::abs(value)>1e100)return false;
  }
  return true;
}
inline bool scalar(py::handle value, double &out) {
  if (!PyFloat_CheckExact(value.ptr())) return false;
  out = PyFloat_AS_DOUBLE(value.ptr());
  return std::isfinite(out) && std::abs(out) <= 1e100;
}
inline bool same(const py::array &a, const py::array &b) {
  return a.ndim() == b.ndim() && a.size() == b.size()
         && a.shape(0) == b.shape(0) && a.shape(1) == b.shape(1);
}
inline bool can_double(const py::array &a) {
  auto limit=std::numeric_limits<py::ssize_t>::max();
  return a.shape(0)<=limit/2 && a.size()<=limit/(2*a.itemsize());
}

inline py::object splice(const std::string &kind, const py::tuple &v, bool separate = false) {
  if (v.size() != 3) return py::none();
  auto a=py::reinterpret_borrow<py::array>(py::handle()), b=py::reinterpret_borrow<py::array>(py::handle()), cuts=py::reinterpret_borrow<py::array>(py::handle());
  if (!ordinary(v[0], a, 2) || !ordinary(v[1], b, 2) || !same(a,b)
      || !a.dtype().is(b.dtype())
      || (!a.dtype().is(py::dtype::of<double>()) && !a.dtype().is(py::dtype::of<int64_t>())))
    return py::none();
  auto n = a.shape(0), d = a.shape(1);
  if(!can_double(a))return py::none();
  if (kind == "splice_uniform") {
    if (!ordinary(v[2], cuts, 2) || !same(a,cuts)
        || !cuts.dtype().is(py::dtype::of<bool>())) return py::none();
  } else {
    int rank = kind == "splice_two" ? 2 : 1;
    if (!integers(v[2], cuts, rank) || cuts.shape(0) != n
        || (rank == 2 && cuts.shape(1) != 2)) return py::none();
    auto p = static_cast<const int64_t *>(cuts.data());
    for (py::ssize_t i = 0; i < n; ++i) {
      if (rank == 1) { if (p[i] < 0 || p[i] > d) return py::none(); }
      else if (p[2*i] < 0 || p[2*i+1] < p[2*i] || p[2*i+1] >= d) return py::none();
    }
  }
  py::array output(a.dtype(), std::vector<py::ssize_t>{separate ? n : 2*n,d});
  auto other=py::reinterpret_borrow<py::array>(py::handle());
  if (separate) other = py::array(a.dtype(), std::vector<py::ssize_t>{n,d});
  auto first = static_cast<const char *>(a.data());
  auto second = static_cast<const char *>(b.data());
  auto out = static_cast<char *>(output.mutable_data());
  auto points = static_cast<const int64_t *>(cuts.data());
  auto mask = static_cast<const uint8_t *>(cuts.data());
  for (py::ssize_t i = 0; i < n; ++i) {
    if (kind != "splice_uniform") {
      py::ssize_t lo = kind == "splice_two" ? points[2*i] : points[i];
      py::ssize_t hi = kind == "splice_two" ? points[2*i+1]+1 : d;
      for (int child = 0; child < 2; ++child) {
        auto x = child ? second : first;
        auto y = child ? first : second;
        auto row = (child && separate ? static_cast<char *>(other.mutable_data()) : out + child*n*d*8) + i*d*8;
        if (lo) std::memcpy(row, x+i*d*8, lo*8);
        if (hi>lo) std::memcpy(row+lo*8, y+(i*d+lo)*8, (hi-lo)*8);
        if (d>hi) std::memcpy(row+hi*8, x+(i*d+hi)*8, (d-hi)*8);
      }
    } else {
      for (py::ssize_t j = 0; j < d; ++j) {
        auto k = i*d+j;
        std::memcpy(out+k*8, (mask[k] ? second : first)+k*8, 8);
        auto second_out = separate ? static_cast<char *>(other.mutable_data()) : out+n*d*8;
        std::memcpy(second_out+k*8, (mask[k] ? first : second)+k*8, 8);
      }
    }
  }
  return separate ? py::object(py::make_tuple(output, other)) : py::object(py::make_tuple(output));
}

inline py::object de(const py::tuple &v) {
  if (v.size() != 6) return py::none();
  auto x=py::reinterpret_borrow<py::array>(py::handle()), base=py::reinterpret_borrow<py::array>(py::handle()), p2=py::reinterpret_borrow<py::array>(py::handle()), p3=py::reinterpret_borrow<py::array>(py::handle()), mask=py::reinterpret_borrow<py::array>(py::handle());
  double f;
  if (!doubles(v[0],x) || !doubles(v[1],base) || !doubles(v[2],p2)
      || !doubles(v[3],p3) || !ordinary(v[4],mask,2)
      || !same(x,base) || !same(x,p2) || !same(x,p3) || !same(x,mask)
      || !mask.dtype().is(py::dtype::of<bool>()) || !scalar(v[5],f)
      || !bounded(base) || (!p2.is(base) && !bounded(p2))
      || (!p3.is(base) && !p3.is(p2) && !bounded(p3))) return py::none();
  py::array_t<double> output({x.shape(0),x.shape(1)});
  auto a=static_cast<const double *>(x.data()), b=static_cast<const double *>(base.data());
  auto c=static_cast<const double *>(p2.data()), d=static_cast<const double *>(p3.data());
  auto m=static_cast<const uint8_t *>(mask.data()); auto out=output.mutable_data();
  auto size=x.size();
  for (py::ssize_t k=0;k<size;++k) {
    double difference=c[k]-d[k];
    double product=f*difference;
    double donor=b[k]+product;
    out[k]=m[k]?donor:a[k];
  }
  return py::make_tuple(output);
}

inline py::object arithmetic(const py::tuple &v) {
  if (v.size()!=3) return py::none();
  auto a=py::reinterpret_borrow<py::array>(py::handle()), b=py::reinterpret_borrow<py::array>(py::handle()), r=py::reinterpret_borrow<py::array>(py::handle());
  if (!doubles(v[0],a)||!doubles(v[1],b)||!same(a,b)||!bounded(a)||!bounded(b))
    return py::none();
  // Ratio is the original scalar or (1,D) NumPy conversion, not a new cast.
  auto rank=py::isinstance<py::array>(v[2])?py::cast<py::array>(v[2]).ndim():-1;
  if ((rank!=0 && rank!=2)||!doubles(v[2],r,rank)||!bounded(r)
      ||(rank==2 && (r.shape(0)!=1 || (r.shape(1)!=1 && r.shape(1)!=a.shape(1)))))
    return py::none();
  auto n=a.shape(0),d=a.shape(1);
  if(!can_double(a))return py::none();
  py::array_t<double> output({2*n,d});
  auto x=static_cast<const double *>(a.data()),y=static_cast<const double *>(b.data());
  auto ratio=static_cast<const double *>(r.data()); auto out=output.mutable_data();
  bool scalar_ratio=r.size()==1;
  for(py::ssize_t i=0;i<n;++i) for(py::ssize_t j=0;j<d;++j) {
    auto k=i*d+j; double q=ratio[scalar_ratio?0:j]; double complement=1.0-q;
    double a1=q*x[k],b1=complement*y[k],a2=q*y[k],b2=complement*x[k];
    out[k]=a1+b1;out[n*d+k]=a2+b2;
  }
  return py::make_tuple(output);
}

inline py::object pso(const py::tuple &v) {
  if(v.size()!=8 && v.size()!=9) return py::none();
  if(v.size()==9 && !PyBool_Check(v[8].ptr()))return py::none();
  bool selected=v.size()==9 && v[8].ptr()==Py_True;
  double inertia;auto velocity=py::reinterpret_borrow<py::array>(py::handle()), indices=py::reinterpret_borrow<py::array>(py::handle()), r1=py::reinterpret_borrow<py::array>(py::handle()), pbest=py::reinterpret_borrow<py::array>(py::handle()), x=py::reinterpret_borrow<py::array>(py::handle()), r2=py::reinterpret_borrow<py::array>(py::handle()), gbest=py::reinterpret_borrow<py::array>(py::handle());
  if(!scalar(v[0],inertia)||!doubles(v[1],velocity)||!integers(v[2],indices)
     ||!doubles(v[3],r1)||!doubles(v[4],pbest)||!doubles(v[5],x)
     ||!doubles(v[6],r2)||!doubles(v[7],gbest)
     ||(selected && !same(x,pbest))||pbest.shape(1)!=x.shape(1)||velocity.shape(1)!=x.shape(1)
     ||indices.size()!=x.shape(0)||r1.shape(0)!=x.shape(0)||r1.shape(1)!=1
     ||!same(r1,r2)||gbest.shape(0)!=1||gbest.shape(1)!=x.shape(1))return py::none();
  auto idx=static_cast<const int64_t *>(indices.data());
  for(py::ssize_t i=0;i<indices.size();++i)
    if(idx[i]<0||idx[i]>=velocity.shape(0)||(!selected && idx[i]>=pbest.shape(0)))return py::none();
  if(!bounded_rows(velocity,idx,indices.size())||!bounded(r1)||!bounded(r2)
     ||!(selected?bounded(pbest):bounded_rows(pbest,idx,indices.size()))
     ||!bounded(x)||!bounded(gbest))return py::none();
  auto n=x.shape(0),d=x.shape(1);
  py::array_t<double> next({n,d}),offspring({n,d});
  auto vel=static_cast<const double *>(velocity.data()),best=static_cast<const double *>(pbest.data());
  auto position=static_cast<const double *>(x.data()),guide=static_cast<const double *>(gbest.data());
  auto a=static_cast<const double *>(r1.data()),b=static_cast<const double *>(r2.data());
  auto next_data=next.mutable_data(),child_data=offspring.mutable_data();
  for(py::ssize_t i=0;i<n;++i){
    double cognitive=2.0*a[i],social=2.0*b[i];
    for(py::ssize_t j=0;j<d;++j){
      auto k=i*d+j; auto source=idx[i]*d+j;
      double first=inertia*vel[source];
      double second=cognitive*(best[selected?k:source]-position[k]);
      double third=social*(guide[j]-position[k]);
      double value=(first+second)+third;
      next_data[k]=value;child_data[k]=position[k]+value;
    }
  }
  return py::make_tuple(next,offspring);
}

inline py::object nich_nearest(const py::tuple &v) {
  auto distances=py::reinterpret_borrow<py::array>(py::handle());
  if(v.size()!=1||!doubles(v[0],distances)||distances.shape(0)!=distances.shape(1)
      ||!distances.shape(0))return py::none();
  auto n=distances.shape(0);auto d=static_cast<const double *>(distances.data());
  py::array_t<int64_t> nearest(n),follow(n);py::array_t<double> values(n);
  auto p=nearest.mutable_data(),f=follow.mutable_data();auto w=values.mutable_data();
  for(py::ssize_t i=0;i<n;++i){p[i]=-1;f[i]=1;w[i]=0.;}
  for(py::ssize_t i=1;i<n;++i){
    int64_t best=0;
    double best_value=d[i*n];
    // NumPy argmin chooses the first NaN and otherwise the first minimum.
    if(!std::isnan(best_value))for(py::ssize_t j=1;j<i;++j){
      double candidate=d[i*n+j];
      if(std::isnan(candidate)){best=j;best_value=candidate;break;}
      if(candidate<best_value){best=j;best_value=candidate;}
    }
    p[i]=best;w[i]=best_value;
  }
  for(py::ssize_t i=n-1;i>0;--i)if(p[i]>=0)f[p[i]]+=f[i];
  return py::make_tuple(nearest,values,follow);
}

inline py::object nich_finish(const py::tuple &v, bool in_place = false) {
  auto nearest=py::reinterpret_borrow<py::array>(py::handle()), values=py::reinterpret_borrow<py::array>(py::handle()), follow=py::reinterpret_borrow<py::array>(py::handle()), order=py::reinterpret_borrow<py::array>(py::handle());double mean;
  if(v.size()!=6||!integers(v[0],nearest)||!doubles(v[1],values,1)
     ||!integers(v[2],follow)||!PyFloat_CheckExact(v[3].ptr())
     ||!PyLong_CheckExact(v[5].ptr()))return py::none();
  mean=PyFloat_AS_DOUBLE(v[3].ptr());
  if(reinterpret_cast<PyObject *>(Py_TYPE(v[4].ptr()))!=array_type.ptr())return py::none();
  order=py::reinterpret_borrow<py::array>(v[4]);
  if(order.ndim()!=1||!order.dtype().is(py::dtype::of<int64_t>()))return py::none();
  auto n=nearest.size();
  if(!n||values.size()!=n||follow.size()!=n||order.size()!=n)return py::none();
  if (in_place) {
    if (!nearest.writeable() || !values.writeable() || !follow.writeable()) return py::none();
    std::vector<py::array> arrays{nearest,values,follow,order};
    for (size_t i=0;i<arrays.size();++i) for(size_t j=0;j<i;++j) {
      auto low=[](const py::array &a) { auto p=reinterpret_cast<intptr_t>(a.data()); return p+std::min<py::ssize_t>(0,(a.size()-1)*a.strides(0)); };
      auto high=[](const py::array &a) { auto p=reinterpret_cast<intptr_t>(a.data()); return p+std::max<py::ssize_t>(0,(a.size()-1)*a.strides(0))+a.itemsize(); };
      if(low(arrays[i])<high(arrays[j]) && low(arrays[j])<high(arrays[i])) return py::none();
    }
  }
  auto parent=static_cast<const int64_t *>(nearest.data()),counts=static_cast<const int64_t *>(follow.data());
  auto rank_data=static_cast<const char *>(order.data());auto rank_stride=order.strides(0);
  // order stays owned and disjoint from mutable outputs in the in-place path.
  auto rank=[&](py::ssize_t i){int64_t value;std::memcpy(&value,rank_data+i*rank_stride,sizeof(value));return value;};
  auto weights=static_cast<const double *>(values.data());
  std::vector<int64_t> parent_copy, count_copy, seeds{0};
  std::vector<double> weight_copy;
  if (!in_place) {
    parent_copy.assign(parent,parent+n); count_copy.assign(counts,counts+n);
    weight_copy.assign(weights,weights+n);
  }
  auto p = in_place ? static_cast<int64_t *>(nearest.mutable_data()) : parent_copy.data();
  auto f = in_place ? static_cast<int64_t *>(follow.mutable_data()) : count_copy.data();
  auto w = in_place ? static_cast<double *>(values.mutable_data()) : weight_copy.data();
  int overflow=0;
  auto min_size=PyLong_AsLongLongAndOverflow(v[5].ptr(),&overflow);
  if(overflow || PyErr_Occurred()){PyErr_Clear();return py::none();}
  for(py::ssize_t i=0;i<n;++i)
    if(p[i]<-1||p[i]>=i||rank(i)<0||rank(i)>=n||f[i]<1||f[i]>n)return py::none();
  for(py::ssize_t k=0;k<n;++k){
    auto child=rank(k);if(w[child]<=mean)continue;
    auto above=p[child];if(above<0)continue;
    std::vector<int64_t> chain{above};auto top=above;
    while(p[top]!=-1){top=p[top];chain.push_back(top);}
    if(f[child]>=min_size && f[top]-f[child]>=min_size){
      p[child]=-1;w[child]=0.;seeds.push_back(child);
      for(auto node:chain)f[node]-=f[child];
    }
  }
  py::array_t<int64_t> seed_array(seeds.size()),roots(n);
  std::memcpy(seed_array.mutable_data(),seeds.data(),seeds.size()*sizeof(int64_t));
  auto root_data=roots.mutable_data();
  for(py::ssize_t i=0;i<n;++i){auto node=i;while(p[node]!=-1)node=p[node];root_data[i]=node;}
  return py::make_tuple(seed_array,roots);
}

inline py::object execute(const std::string &kernel, const py::tuple &inputs) {
  if(kernel=="splice_one"||kernel=="splice_two"||kernel=="splice_uniform")return splice(kernel,inputs);
  if(kernel=="nich_nearest")return nich_nearest(inputs);
  if(kernel=="nich_finish")return nich_finish(inputs);
  if(kernel=="de_vary")return de(inputs);
  if(kernel=="arithmetic")return arithmetic(inputs);
  if(kernel=="pso")return pso(inputs);
  return py::none();
}
};

} // namespace cpu_blocks

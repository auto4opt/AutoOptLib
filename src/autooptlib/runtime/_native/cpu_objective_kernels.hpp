#pragma once
#include "x86_kernels.hpp"
#include "stacking_cpu_kernel.hpp"
// Public flat C ABI examples. SIMD lanes are independent candidates; every
// candidate keeps its original left-to-right multiply/add reduction order.
extern "C" {
void autoopt_cpu_sum_squares_scalar(int64_t n,int64_t d,const void *raw,double *y,double*,const void*) {
  auto x=static_cast<const double*>(raw);
  for(int64_t i=0;i<n;++i){double s=0.;for(int64_t j=0;j<d;++j)s+=x[i*d+j]*x[i*d+j];y[i]=s;}
}
void autoopt_cpu_residual_scalar(int64_t n,int64_t d,const void *raw,double *y,double*c,const void*ctx) {
  auto x=static_cast<const double*>(raw),a=static_cast<const double*>(ctx);const auto *target=a+d*d;
  for(int64_t i=0;i<n;++i){double sum=0.;for(int64_t r=0;r<d;++r){double v=0.;for(int64_t j=0;j<d;++j)v+=a[r*d+j]*x[i*d+j];v-=target[r];sum+=v*v;}y[i]=sum;
    double total=0.;for(int64_t j=0;j<d;++j)total+=x[i*d+j];c[2*i]=total-1.;c[2*i+1]=-total-1.;}
}
#if AOL_X86
AOL_AVX2 void autoopt_cpu_sum_squares_avx2(int64_t n,int64_t d,const void *raw,double *y,double*c,const void*ctx) {
  auto x=static_cast<const double*>(raw);int64_t i=0;
  for(;i+4<=n;i+=4){auto sum=_mm256_setzero_pd();for(int64_t j=0;j<d;++j){auto v=_mm256_set_pd(x[(i+3)*d+j],x[(i+2)*d+j],x[(i+1)*d+j],x[i*d+j]);sum=_mm256_add_pd(sum,_mm256_mul_pd(v,v));}_mm256_storeu_pd(y+i,sum);}
  autoopt_cpu_sum_squares_scalar(n-i,d,x+i*d,y+i,c,ctx);
}
AOL_AVX2 void autoopt_cpu_residual_avx2(int64_t n,int64_t d,const void *raw,double*y,double*c,const void*ctx) {
  auto x=static_cast<const double*>(raw),a=static_cast<const double*>(ctx);const auto *target=a+d*d;int64_t i=0;
  for(;i+4<=n;i+=4){auto sum=_mm256_setzero_pd();for(int64_t r=0;r<d;++r){auto v=_mm256_setzero_pd();for(int64_t j=0;j<d;++j){auto values=_mm256_set_pd(x[(i+3)*d+j],x[(i+2)*d+j],x[(i+1)*d+j],x[i*d+j]);v=_mm256_add_pd(v,_mm256_mul_pd(_mm256_set1_pd(a[r*d+j]),values));}v=_mm256_sub_pd(v,_mm256_set1_pd(target[r]));sum=_mm256_add_pd(sum,_mm256_mul_pd(v,v));}_mm256_storeu_pd(y+i,sum);
    auto total=_mm256_setzero_pd();for(int64_t j=0;j<d;++j)total=_mm256_add_pd(total,_mm256_set_pd(x[(i+3)*d+j],x[(i+2)*d+j],x[(i+1)*d+j],x[i*d+j]));
    alignas(32) double values[4];_mm256_store_pd(values,total);for(int k=0;k<4;++k){c[2*(i+k)]=values[k]-1.;c[2*(i+k)+1]=-values[k]-1.;}}
  autoopt_cpu_residual_scalar(n-i,d,x+i*d,y+i,c+2*i,ctx);
}
#endif
}
inline void bind_cpu_objective_kernels(py::module_ &m) {
  m.def("cpu_objective_kernel",[](const std::string &name){
    using Fn=void(*)(int64_t,int64_t,const void*,double*,double*,const void*);
    Fn fn=nullptr;bool vector=false;int constraints=0;
#if AOL_X86
    vector=aol_x86::mode()!=aol_x86::Mode::Baseline&&aol_x86::capabilities().avx2&&aol_x86::ordinary_fp();
#endif
    if(name=="ordered_sum_squares") {
      fn=&autoopt_cpu_sum_squares_scalar;
#if AOL_X86
      if(vector)fn=&autoopt_cpu_sum_squares_avx2;
#endif
    } else if(name=="ordered_residual") {
      constraints=2;fn=&autoopt_cpu_residual_scalar;
#if AOL_X86
      if(vector)fn=&autoopt_cpu_residual_avx2;
#endif
    } else if(name=="material_stacking") {
      fn=&autoopt_cpu_stacking;vector=false;constraints=-1;
    } else throw py::value_error("Unknown ordered CPU objective kernel");
    return py::dict("address"_a=reinterpret_cast<uintptr_t>(fn),"isa"_a=vector?"avx2":"scalar",
                    "constraints"_a=constraints,"lanes"_a=vector?4:1);
  });
}

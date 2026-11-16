// Local, exact x86 kernels. No RNG, reductions, allocations or Python calls.
// The portable build remains the baseline; ISA functions have local targets.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define AOL_X86 1
#include <immintrin.h>
#define AOL_AVX2 __attribute__((target("avx2")))
#define AOL_CONVERT __attribute__((target("avx512f,avx512dq,avx512vl")))
#else
#define AOL_X86 0
#define AOL_AVX2
#define AOL_CONVERT
#endif
namespace aol_x86 {
enum class Mode { Baseline, Auto, Simd, Adaptive, Packed };
inline Mode parse(const char *s) {
  if(s&&!std::strcmp(s,"baseline"))return Mode::Baseline;
  if(s&&!std::strcmp(s,"auto"))return Mode::Auto;
  if(s&&!std::strcmp(s,"simd"))return Mode::Simd;
  if(s&&!std::strcmp(s,"packed"))return Mode::Packed;
  return Mode::Adaptive;
}
inline Mode configured_mode(){static const auto m=parse(std::getenv("AOL_X86_KERNELS"));return m;}
inline thread_local int mode_override=-1; // Private diagnostic only; no timing/calibration.
inline Mode mode(){return mode_override<0?configured_mode():static_cast<Mode>(mode_override);}
struct Capabilities {bool avx2=false,convert=false,popcnt=false;};
inline const Capabilities &capabilities() {
  static const Capabilities c=[] {Capabilities v;
#if AOL_X86
    __builtin_cpu_init();v.avx2=__builtin_cpu_supports("avx2");v.popcnt=__builtin_cpu_supports("popcnt");
    v.convert=__builtin_cpu_supports("avx512f")&&__builtin_cpu_supports("avx512dq")&&__builtin_cpu_supports("avx512vl");
#endif
    return v;}();return c;
}
inline bool ordinary_fp(){
#if AOL_X86
  // Ignore sticky exception bits, retain rounding, exception masks, FTZ/DAZ.
  return (_mm_getcsr()&~unsigned(0x3f))==0x1f80;
#else
  return false;
#endif
}
inline bool enabled(size_t n,bool floating=false) {
  auto m=mode();return m!=Mode::Baseline&&n>=(m==Mode::Simd||m==Mode::Packed?4u:64u)&&
    capabilities().avx2&&(!floating||ordinary_fp());
}
template<class F> inline void scalar_each(size_t n,F f){for(size_t j=0;j<n;++j)f(j);}
template<class F> AOL_AVX2 inline void isa_each(size_t n,F f){for(size_t j=0;j<n;++j)f(j);}
// Targeted compiler vectorization for exact, independent coordinate formulas.
// The caller retains the original expression tree and all side-effect barriers.
template<class F> inline void each(size_t n,F f) {
  if(enabled(n,true))isa_each(n,f);else scalar_each(n,f);
}
constexpr unsigned bits4[16]={0,1,1,2,1,2,2,3,1,2,2,3,2,3,3,4};
#if AOL_X86
template<class T> AOL_AVX2 inline bool binary_vector(const T *p,size_t n) {
  size_t j=0;
  for(;j+4<=n;j+=4){int mask;
    if constexpr(std::is_same_v<T,int64_t>){auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p+j));
      auto yes=_mm256_or_si256(_mm256_cmpeq_epi64(x,_mm256_setzero_si256()),_mm256_cmpeq_epi64(x,_mm256_set1_epi64x(1)));
      mask=_mm256_movemask_pd(_mm256_castsi256_pd(yes));
    }else{auto x=_mm256_loadu_pd(p+j);mask=_mm256_movemask_pd(_mm256_or_pd(_mm256_cmp_pd(x,_mm256_setzero_pd(),_CMP_EQ_OQ),_mm256_cmp_pd(x,_mm256_set1_pd(1),_CMP_EQ_OQ)));}
    if(mask!=15)return false;
  }
  for(;j<n;++j)if(p[j]!=0&&p[j]!=1)return false;return true;
}
AOL_AVX2 inline int64_t count_vector(const double *p,size_t n,double probability) {
  size_t j=0;int64_t count=0;auto q=_mm256_set1_pd(probability);
  for(;j+4<=n;j+=4)count+=bits4[_mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(p+j),q,_CMP_LT_OQ))];
  for(;j<n;++j)count+=p[j]<probability;return count;
}
template<class T> AOL_AVX2 inline unsigned difference_mask(const T *p,const T *q) {
  if constexpr(std::is_same_v<T,int64_t>)return unsigned(~_mm256_movemask_pd(_mm256_castsi256_pd(_mm256_cmpeq_epi64(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)),_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q))))))&15;
  else return _mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(p),_mm256_loadu_pd(q),_CMP_NEQ_UQ));
}
template<class T> AOL_AVX2 inline void exchange_block(const T *p,const T *q,T *a,T *b,unsigned mask) {
  // Bitwise selection preserves all 64 bits, including int64 and NaN payloads.
  auto x=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)),y=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q));
  auto m=_mm256_set_epi64x(-int64_t((mask>>3)&1),-int64_t((mask>>2)&1),-int64_t((mask>>1)&1),-int64_t(mask&1));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(a),_mm256_blendv_epi8(x,y,m));
  _mm256_storeu_si256(reinterpret_cast<__m256i *>(b),_mm256_blendv_epi8(y,x,m));
}
template<class T> AOL_AVX2 inline void uniform_vector(const T *p,const T *q,T *a,T *b,const double *coins,double probability,size_t n) {
  size_t j=0;auto threshold=_mm256_set1_pd(probability);
  for(;j+4<=n;j+=4)exchange_block(p+j,q+j,a+j,b+j,_mm256_movemask_pd(_mm256_cmp_pd(_mm256_loadu_pd(coins+j),threshold,_CMP_LT_OQ)));
  for(;j<n;++j){T x=p[j],y=q[j];bool take=coins[j]<probability;a[j]=take?y:x;b[j]=take?x:y;}
}
AOL_AVX2 inline int64_t hamming_vector(const double *p,const double *q,size_t n) {
  int64_t count=0;size_t j=0;for(;j+4<=n;j+=4)count+=bits4[difference_mask(p+j,q+j)];
  for(;j<n;++j)count+=p[j]!=q[j];return count;
}
AOL_AVX2 inline bool pack_vector(const double *p,uint64_t *words,size_t n) {
  size_t j=0;
  for(;j+4<=n;j+=4){auto x=_mm256_loadu_pd(p+j),zero=_mm256_cmp_pd(x,_mm256_setzero_pd(),_CMP_EQ_OQ),one=_mm256_cmp_pd(x,_mm256_set1_pd(1),_CMP_EQ_OQ);
    if(_mm256_movemask_pd(_mm256_or_pd(zero,one))!=15)return false;
    words[j/64]|=uint64_t(_mm256_movemask_pd(one))<<(j%64);
  }
  for(;j<n;++j){if(p[j]!=0&&p[j]!=1)return false;words[j/64]|=uint64_t(p[j]==1)<<(j%64);}return true;
}
__attribute__((target("popcnt"))) inline int64_t packed_count(const uint64_t *p,const uint64_t *q,size_t words) {
  int64_t count=0;for(size_t j=0;j<words;++j)count+=__builtin_popcountll(p[j]^q[j]);return count;
}
// Finite-only vectors; special coordinates retain the caller's authoritative
// scalar repair (including NaNs, signed zeros and nonstandard bounds).
AOL_AVX2 inline bool finite4(__m256d x) {
  auto bits=_mm256_and_si256(_mm256_castpd_si256(x),_mm256_set1_epi64x(0x7ff0000000000000LL));
  return _mm256_movemask_epi8(_mm256_cmpeq_epi64(bits,_mm256_set1_epi64x(0x7ff0000000000000LL)))==0;
}
AOL_CONVERT inline __m256d load_integer4(const void *p){return _mm256_cvtepi64_pd(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(p)));}
AOL_CONVERT inline void store_integer4(void *p,__m256d x){_mm256_storeu_si256(reinterpret_cast<__m256i *>(p),_mm256_cvttpd_epi64(x));}
template<bool Discrete,bool Integer,class Scalar>
AOL_AVX2 inline void repair_vector(const void *source,void *destination,const double *lo,const double *hi,size_t n,Scalar scalar) {
  size_t j=0;
  for(;j+4<=n;j+=4){auto x=Integer?load_integer4(static_cast<const char *>(source)+j*8):_mm256_loadu_pd(reinterpret_cast<const double *>(source)+j);
    auto lower=_mm256_loadu_pd(lo+j),upper=_mm256_loadu_pd(hi+j);
    if(!finite4(x)||!finite4(lower)||!finite4(upper)){for(size_t k=0;k<4;++k)scalar(j+k);continue;}
    if constexpr(Discrete&&!Integer)x=_mm256_round_pd(x,_MM_FROUND_TO_NEAREST_INT|_MM_FROUND_NO_EXC);
    // Blend deliberately chooses the bound on equality, unlike std::max/min.
    x=_mm256_blendv_pd(lower,x,_mm256_cmp_pd(x,lower,_CMP_GT_OQ));
    x=_mm256_blendv_pd(upper,x,_mm256_cmp_pd(x,upper,_CMP_LT_OQ));
    if constexpr(Discrete)store_integer4(static_cast<char *>(destination)+j*8,x);
    else _mm256_storeu_pd(reinterpret_cast<double *>(destination)+j,x);
  }
  for(;j<n;++j)scalar(j);
}
#endif
template<class T> inline bool binary(const T *p,size_t n) {
#if AOL_X86
  if(enabled(n)&&mode()!=Mode::Auto)return binary_vector(p,n);
#endif
  for(size_t j=0;j<n;++j)if(p[j]!=0&&p[j]!=1)return false;return true;
}
inline int64_t count_less(const double *p,size_t n,double probability) {
#if AOL_X86
  if(enabled(n)&&mode()!=Mode::Auto)return count_vector(p,n,probability);
#endif
  int64_t count=0;for(size_t j=0;j<n;++j)count+=p[j]<probability;return count;
}
template<class T,class Indices> inline void differences(const T *p,const T *q,size_t n,Indices &out) {
  size_t j=0;
#if AOL_X86
  if(enabled(n)&&mode()!=Mode::Auto)for(;j+4<=n;j+=4){unsigned mask=difference_mask(p+j,q+j);while(mask){unsigned bit=__builtin_ctz(mask);out.push_back(j+bit);mask&=mask-1;}}
#endif
  for(;j<n;++j)if(p[j]!=q[j])out.push_back(j);
}
template<class T> inline void uniform(const T *p,const T *q,T *a,T *b,const double *coins,double probability,size_t n) {
#if AOL_X86
  if(enabled(n)&&mode()!=Mode::Auto){uniform_vector(p,q,a,b,coins,probability,n);return;}
#endif
  auto body=[&](size_t j){T x=p[j],y=q[j];bool take=coins[j]<probability;a[j]=take?y:x;b[j]=take?x:y;};
  if(mode()==Mode::Auto)each(n,body);else scalar_each(n,body);
}
template<class T,class Indices> inline bool uniform_different(const T *p,const T *q,T *a,T *b,size_t n,const Indices &positions,const double *coins) {
  size_t j=0,k=0;bool changed=false;
#if AOL_X86
  if(enabled(n)&&mode()!=Mode::Auto)for(;j+4<=n;j+=4){unsigned mask=0;while(k<positions.size()&&size_t(positions[k])<j+4){bool take=coins[k]<.5;if(take)mask|=1u<<(positions[k]-j);changed|=take;++k;}exchange_block(p+j,q+j,a+j,b+j,mask);}
#endif
  for(;j<n;++j){bool take=false;if(k<positions.size()&&size_t(positions[k])==j){take=coins[k++]<.5;changed|=take;}T x=p[j],y=q[j];a[j]=take?y:x;b[j]=take?x:y;}return changed;
}
inline int64_t hamming(const double *p,const double *q,size_t n) {
#if AOL_X86
  if(enabled(n)&&mode()!=Mode::Auto)return hamming_vector(p,q,n);
#endif
  int64_t count=0;for(size_t j=0;j<n;++j)count+=p[j]!=q[j];return count;
}
inline bool use_packed(size_t n,size_t d) {
  if(!d||!capabilities().popcnt||!capabilities().avx2)return false;
  if(mode()==Mode::Packed)return true;
  if(mode()!=Mode::Adaptive)return false;
  // Include validation/packing and pair scan, without reading the data twice.
  return double(n)*d+double(n)*n*((d+63)/64)<double(n)*n*((d+3)/4);
}
inline bool pack(const double *p,uint64_t *words,size_t n) {
#if AOL_X86
  return pack_vector(p,words,n);
#else
  return false;
#endif
}
inline int64_t packed_hamming(const uint64_t *p,const uint64_t *q,size_t words) {
#if AOL_X86
  return packed_count(p,q,words);
#else
  int64_t count=0;for(size_t j=0;j<words;++j){auto x=p[j]^q[j];while(x){++count;x&=x-1;}}return count;
#endif
}
template<bool Discrete,bool Integer,class Scalar> inline void repair(const void *src,void *dst,const double *lo,const double *hi,size_t n,bool contiguous,Scalar scalar) {
#if AOL_X86
  if(contiguous&&enabled(n,true)&&mode()!=Mode::Auto&&(!(Discrete||Integer)||capabilities().convert)){
    repair_vector<Discrete,Integer>(src,dst,lo,hi,n,scalar);return;
  }
#endif
  scalar_each(n,scalar);
}
// Choose once at the component boundary; tiny rows keep the original loop.
inline bool dense_enabled(size_t d,size_t total) {
  if(mode()==Mode::Simd||mode()==Mode::Packed)return enabled(d,true);
  return d>=64&&total>=512&&enabled(d,true);
}
#if AOL_X86
AOL_AVX2 inline void add_scaled_vector(double *out,const double *scale,const double *noise,size_t n) {
  size_t j=0;for(;j+4<=n;j+=4){auto a=_mm256_loadu_pd(out+j),b=_mm256_loadu_pd(scale+j),c=_mm256_loadu_pd(noise+j);
    if(!finite4(a)||!finite4(b)||!finite4(c)){for(size_t k=j;k<j+4;++k){double v=scale[k]*noise[k];out[k]=out[k]+v;}continue;}
    _mm256_storeu_pd(out+j,_mm256_add_pd(a,_mm256_mul_pd(b,c)));}
  for(;j<n;++j){double v=scale[j]*noise[j];out[j]=out[j]+v;}
}
AOL_AVX2 inline void pair_vector(const double *first,const double *second,double *a,double *b,const double *ratios,double ratio,size_t n,bool sbx) {
  size_t j=0;auto one=_mm256_set1_pd(1),half=_mm256_set1_pd(.5);
  auto scalar=[&](size_t k){double x=first[k],y=second[k],q=ratios?ratios[k]:ratio;if(sbx){a[k]=.5*((1.+q)*x+(1.-q)*y);b[k]=.5*((1.-q)*x+(1.+q)*y);}else{double c=1.-q;a[k]=q*x+c*y;b[k]=q*y+c*x;}};
  for(;j+4<=n;j+=4){auto x=_mm256_loadu_pd(first+j),y=_mm256_loadu_pd(second+j),q=ratios?_mm256_loadu_pd(ratios+j):_mm256_set1_pd(ratio);
    if(!finite4(x)||!finite4(y)||!finite4(q)){for(size_t k=j;k<j+4;++k)scalar(k);continue;}
    auto complement=_mm256_sub_pd(one,q);
    if(sbx){auto plus=_mm256_add_pd(one,q);
      _mm256_storeu_pd(a+j,_mm256_mul_pd(half,_mm256_add_pd(_mm256_mul_pd(plus,x),_mm256_mul_pd(complement,y))));
      _mm256_storeu_pd(b+j,_mm256_mul_pd(half,_mm256_add_pd(_mm256_mul_pd(complement,x),_mm256_mul_pd(plus,y))));
    }else{
      _mm256_storeu_pd(a+j,_mm256_add_pd(_mm256_mul_pd(q,x),_mm256_mul_pd(complement,y)));
      _mm256_storeu_pd(b+j,_mm256_add_pd(_mm256_mul_pd(q,y),_mm256_mul_pd(complement,x)));}
  }for(;j<n;++j)scalar(j);
}
AOL_AVX2 inline void square_vector(const double *first,const double *second,double *out,size_t n) {
  size_t j=0;for(;j+4<=n;j+=4){auto a=_mm256_loadu_pd(first+j),b=_mm256_loadu_pd(second+j);
    if(!finite4(a)||!finite4(b)){for(size_t k=j;k<j+4;++k){double value=first[k]-second[k];out[k]=value*value;}continue;}
    auto v=_mm256_sub_pd(a,b);_mm256_storeu_pd(out+j,_mm256_mul_pd(v,v));}
  for(;j<n;++j){double value=first[j]-second[j];out[j]=value*value;}
}
AOL_AVX2 inline void donor_vector(const double *base,const double *second,const double *third,const double *best,double scale,double *out,size_t n) {
  size_t j=0;auto f=_mm256_set1_pd(scale);
  auto scalar=[&](size_t k){double x=base[k];out[k]=best?(x+scale*(best[k]-x))+scale*(second[k]-third[k]):x+scale*(second[k]-third[k]);};
  if(!std::isfinite(scale)){for(;j<n;++j)scalar(j);return;}
  for(;j+4<=n;j+=4){auto x=_mm256_loadu_pd(base+j),a=_mm256_loadu_pd(second+j),b=_mm256_loadu_pd(third+j),p=best?_mm256_loadu_pd(best+j):x;
    if(!finite4(x)||!finite4(a)||!finite4(b)||!finite4(p)){for(size_t k=j;k<j+4;++k)scalar(k);continue;}
    auto left=best?_mm256_add_pd(x,_mm256_mul_pd(f,_mm256_sub_pd(p,x))):x;
    _mm256_storeu_pd(out+j,_mm256_add_pd(left,_mm256_mul_pd(f,_mm256_sub_pd(a,b))));
  }for(;j<n;++j)scalar(j);
}
AOL_AVX2 inline void select_vector(const double *donor,const double *previous,const double *coins,double cr,int64_t forced,double *out,size_t n) {
  size_t j=0;auto threshold=_mm256_set1_pd(cr);
  for(;j+4<=n;j+=4){auto x=_mm256_loadu_pd(donor+j),y=_mm256_loadu_pd(previous+j),mask=_mm256_cmp_pd(_mm256_loadu_pd(coins+j),threshold,_CMP_LT_OQ);
    if(forced>=int64_t(j)&&forced<int64_t(j+4))mask=_mm256_or_pd(mask,_mm256_castsi256_pd(_mm256_set_epi64x(forced==int64_t(j+3)?-1:0,forced==int64_t(j+2)?-1:0,forced==int64_t(j+1)?-1:0,forced==int64_t(j)?-1:0)));
    _mm256_storeu_pd(out+j,_mm256_blendv_pd(y,x,mask));}
  for(;j<n;++j)out[j]=(int64_t(j)==forced||coins[j]<cr)?donor[j]:previous[j];
}
#endif
inline void add_scaled(double *out,const double *scale,const double *noise,size_t n) {
#if AOL_X86
  if((mode()==Mode::Simd||mode()==Mode::Packed)&&dense_enabled(n,n)){add_scaled_vector(out,scale,noise,n);return;}
#endif
  for(size_t j=0;j<n;++j){double v=scale[j]*noise[j];out[j]=out[j]+v;}
}
inline double pair_coordinate(double x,double y,double q,bool sbx,bool second) {
  if(sbx)return second?.5*((1.-q)*x+(1.+q)*y):.5*((1.+q)*x+(1.-q)*y);
  double c=1.-q;return second?q*y+c*x:q*x+c*y;
}
inline void pair(const double *first,const double *second,double *a,double *b,const double *ratios,double ratio,size_t n,bool sbx,bool vector) {
#if AOL_X86
  if(vector&&mode()!=Mode::Auto&&(sbx||mode()==Mode::Simd||mode()==Mode::Packed)){pair_vector(first,second,a,b,ratios,ratio,n,sbx);return;}
#endif
  for(size_t j=0;j<n;++j){double x=first[j],y=second[j],q=ratios?ratios[j]:ratio;a[j]=pair_coordinate(x,y,q,sbx,false);b[j]=pair_coordinate(x,y,q,sbx,true);}
}
inline void square_scalar(const double *first,const double *second,double *out,size_t n){for(size_t j=0;j<n;++j){double value=first[j]-second[j];out[j]=value*value;}}
using Square=void(*)(const double *,const double *,double *,size_t);
inline Square square_kernel(size_t n,size_t total){
#if AOL_X86
  if((mode()==Mode::Simd||mode()==Mode::Packed)&&dense_enabled(n,total))return square_vector;
#endif
  return square_scalar;
}
inline void donor(const double *base,const double *second,const double *third,const double *best,double scale,double *out,size_t n,bool vector) {
#if AOL_X86
  if(vector&&mode()!=Mode::Auto){donor_vector(base,second,third,best,scale,out,n);return;}
#endif
  for(size_t j=0;j<n;++j){double x=base[j];out[j]=best?(x+scale*(best[j]-x))+scale*(second[j]-third[j]):x+scale*(second[j]-third[j]);}
}
inline void select(const double *donor,const double *previous,const double *coins,double cr,int64_t forced,double *out,size_t n,bool vector) {
#if AOL_X86
  if(vector&&mode()!=Mode::Auto){select_vector(donor,previous,coins,cr,forced,out,n);return;}
#endif
  for(size_t j=0;j<n;++j)out[j]=(int64_t(j)==forced||coins[j]<cr)?donor[j]:previous[j];
}

} // namespace aol_x86

#pragma once
// Physical fragments own only private, callback-free work between RNG and repair.
// No graph cost search, calibration, persistent writable mirrors or JIT.
#include "x86_kernels.hpp"
#include "decision_input.hpp"
#include "execution_services.hpp"
#include <memory>
#include <vector>
#include <pybind11/numpy.h>
namespace native_components {
namespace py=pybind11;
inline double uniform_coordinate(double lo,double hi,double coin){return lo+(hi-lo)*coin;}
inline double repair_round_even(double value) {
  double lower=std::floor(value),fraction=value-lower;
  if(fraction<.5)return lower;if(fraction>.5)return lower+1.;
  return std::fmod(lower,2.)==0.?lower:lower+1.;
}
template<bool Discrete,bool Integer> inline double repair_coordinate(double x,double lo,double hi) {
  if constexpr(Discrete){if constexpr(!Integer)x=repair_round_even(x);}
  else {if(std::isnan(x))x=lo+(hi-lo)*.5;else if(x==INFINITY)x=hi;else if(x==-INFINITY)x=lo;}
  x=std::isnan(x)?x:(x>lo?x:lo);return std::isnan(x)?x:(x<hi?x:hi);
}
// A packed owner never outlives this private interval. Its constructor snapshots
// input, mutation changes only its own words, and no Python object exports it.
// Thus destruction, rather than a mutable external version counter, invalidates it.
class PrivateBits {
  size_t n_,d_,stride_;std::vector<uint64_t> words_;
public:
  PrivateBits(size_t n,size_t d):n_(n),d_(d),stride_((d+63)/64),words_(n*stride_,0){}
  template<class Input> void pack(const Input &input) {
    for(size_t i=0;i<n_;++i){auto row=input_row<int64_t>(input,i);for(size_t j=0;j<d_;++j)if(row[j])words_[i*stride_+j/64]|=uint64_t(1)<<(j%64);}
  }
  template<class Input> void pack_pairs(const Input &input) {
    auto half=n_/2;auto original=input.shape(0);
    for(size_t i=0;i<n_;++i){auto row=input_row<int64_t>(input,i<half?i:original/2+i-half);
      for(size_t j=0;j<d_;++j)if(row[j])words_[i*stride_+j/64]|=uint64_t(1)<<(j%64);}
  }
  int64_t get(size_t i,size_t j)const{return (words_[i*stride_+j/64]>>(j%64))&1;}
  void flip(size_t i,size_t j){words_[i*stride_+j/64]^=uint64_t(1)<<(j%64);}
  void exchange(size_t i,size_t k,const std::vector<uint64_t>&mask) {
    for(size_t w=0;w<stride_;++w){auto &a=words_[i*stride_+w],&b=words_[k*stride_+w];auto delta=(a^b)&mask[w];a^=delta;b^=delta;}
  }
  bool equal(size_t i,const int64_t *row)const {
    for(size_t w=0;w<stride_;++w){uint64_t value=0;for(size_t j=w*64;j<std::min(d_,(w+1)*64);++j){if(row[j]!=0&&row[j]!=1)return false;value|=uint64_t(row[j])<<(j%64);}if(value!=words_[i*stride_+w])return false;}return true;
  }
};
enum class FragmentKind {None,Uniform,Reinit,UniformCross,Bit,Differential,Select,Arithmetic,SBX};
struct FragmentBinding {
  FragmentKind kind=FragmentKind::None;
  void bind(const std::string &name) {
    kind=name=="search_mu_uniform"?FragmentKind::Uniform:name=="reinit_continuous"?FragmentKind::Reinit:
      name=="cross_point_uniform"?FragmentKind::UniformCross:
      name=="cross_arithmetic"?FragmentKind::Arithmetic:name=="cross_sim_binary"?FragmentKind::SBX:
      (name.rfind("search_de_",0)==0||name=="search_fastga_de_current_to_pbest")?FragmentKind::Differential:
      (name.rfind("search_bit_",0)==0||name.rfind("search_fastga_bit_",0)==0)?FragmentKind::Bit:FragmentKind::None;
  }
};
inline thread_local int fragment_mode_override=-1;
inline const char *fragment_mode(){
  static const std::string configured=[](){auto p=std::getenv("AOL_FRAGMENT_MODE");return std::string(p?p:"on");}();
  auto value=configured.c_str();
  static const char *names[]={"off","scalar","on","packed"};
  return fragment_mode_override>=0?names[fragment_mode_override]:value?value:"on";
}
inline bool fragments_enabled(){return std::strcmp(fragment_mode(),"off")!=0;}
inline bool packed_fragments(){return std::strcmp(fragment_mode(),"packed")==0;}
struct PreparedFragment {
  FragmentKind kind;py::ssize_t n,d,count;bool integer=false,grouped=false,row_simd=false;
  DecisionInput parents;py::array coefficients=py::reinterpret_borrow<py::array>(py::none());const double *lower=nullptr,*upper=nullptr;py::object bound_owner;
  ScratchVector<double> masks,coins;double probability=.1;
  std::unique_ptr<PrivateBits> bits;
  std::shared_ptr<void> model_owner;
  std::vector<const double *> bases,seconds,thirds;
  py::array_t<double> donor_values=py::reinterpret_borrow<py::array_t<double>>(py::handle());
  std::vector<int64_t> forced;
  double scale=.5;
  using RowKernel=void(*)(const PreparedFragment &,py::ssize_t,double *,const double *,const double *);
  using FourKernel=void(*)(const PreparedFragment &,py::ssize_t,void **,const double *,const double *);
  RowKernel row_kernel=nullptr;FourKernel four_kernel=nullptr;
  PreparedFragment() :kind(FragmentKind::None),n(0),d(0),count(0){}
  void configure(FragmentKind k,py::ssize_t rows,py::ssize_t dim) {
    kind=k;n=count=rows;d=dim;
    // One capability/FP-environment decision per fragment, never per row.
    bool vector=std::strcmp(fragment_mode(),"scalar")!=0&&aol_x86::mode()!=aol_x86::Mode::Baseline&&
      aol_x86::capabilities().avx2&&aol_x86::ordinary_fp();
    grouped=vector&&fragment_mode_override==2&&n>=4&&d>0&&d<=40;
    row_simd=vector&&d>=4;
    static const std::pair<RowKernel,FourKernel> kernels[]={
      {nullptr,nullptr},
      {row_entry<FragmentKind::Uniform>,four_entry<FragmentKind::Uniform>},
      {row_entry<FragmentKind::Reinit>,four_entry<FragmentKind::Reinit>},
      {row_entry<FragmentKind::UniformCross>,four_entry<FragmentKind::UniformCross>},
      {nullptr,nullptr},
      {row_entry<FragmentKind::Differential>,four_entry<FragmentKind::Differential>},
      {row_entry<FragmentKind::Select>,four_entry<FragmentKind::Select>},
      {row_entry<FragmentKind::Arithmetic>,four_entry<FragmentKind::Arithmetic>},
      {row_entry<FragmentKind::SBX>,four_entry<FragmentKind::SBX>}};
    auto kernel=kernels[static_cast<size_t>(kind)];row_kernel=kernel.first;four_kernel=kernel.second;
  }
  void set_parents(const py::array &input) {
    parents.n=input.shape(0);parents.d=input.shape(1);parents.integer=input.dtype().is(py::dtype::of<int64_t>());
    parents.allocation=input;parents.dense=input.data();
  }
  template<class T> void set_parents(const IndexedInput<T>&input){parents=*input.owner;}
  template<class T> const T *parent_row(py::ssize_t i)const{return static_cast<const T *>(parents.row(i));}
  int64_t integer_value(py::ssize_t i,py::ssize_t j)const {
    if(bits)return bits->get(i,j);
    auto half=(n+1)/2,pair=i<half?i:i-half;
    auto row=i<half?pair:n/2+pair,other=i<half?n/2+pair:pair;
    return parent_row<int64_t>(coins.values[pair*d+j]<probability?other:row)[j];
  }
  template<FragmentKind K> double coordinate(py::ssize_t i,py::ssize_t j)const {
    if constexpr(K==FragmentKind::Differential||K==FragmentKind::Select) {
      auto k=i*d+j;bool take=masks.values[k]<probability||(K==FragmentKind::Select&&j==forced[i]);
      if(!take)return parent_row<double>(i)[j];
      if constexpr(K==FragmentKind::Select)return donor_values.data()[k];
      return bases[i][j]+scale*(seconds[i][j]-thirds[i][j]);
    }
    if constexpr(K==FragmentKind::Uniform||K==FragmentKind::Reinit) {
      auto k=i*d+j;
      if(K==FragmentKind::Reinit||masks.values[k]<probability)return uniform_coordinate(lower[j],upper[j],coins.values[k]);
      return parent_row<double>(i)[j];
    }
    auto half=(n+1)/2,pair=i<half?i:i-half;
    if constexpr(K==FragmentKind::Arithmetic||K==FragmentKind::SBX) {
      auto p=parent_row<double>(pair),qrow=parent_row<double>(n/2+pair);
      auto q=!coefficients.is_none()?static_cast<const double *>(coefficients.data())[K==FragmentKind::SBX?pair*d+j:j]:probability;
      return aol_x86::pair_coordinate(p[j],qrow[j],q,K==FragmentKind::SBX,i>=half);
    }
    auto row=i<half?pair:n/2+pair,other=i<half?n/2+pair:pair;
    auto source=coins.values[pair*d+j]<probability?other:row;
    return integer?double(parent_row<int64_t>(source)[j]):parent_row<double>(source)[j];
  }
  bool supports(bool discrete,const double *lo,const double *hi,py::ssize_t dim)const {
    if(dim!=d)return false;
    if(!discrete)return !integer;
    if(!bits)return false;
    for(py::ssize_t j=0;j<d;++j)if(lo[j]!=0.||hi[j]!=1.)return false;return true;
  }
  py::array materialize()const {
    py::array result(integer?py::dtype::of<int64_t>():py::dtype::of<double>(),std::vector<py::ssize_t>{count,d});
    for(py::ssize_t i=0;i<count;++i) {
      if(integer)for(py::ssize_t j=0;j<d;++j)static_cast<int64_t *>(result.mutable_data())[i*d+j]=integer_value(i,j);
      else row_kernel(*this,i,static_cast<double *>(result.mutable_data())+i*d,nullptr,nullptr);
    }
    return result;
  }
  template<FragmentKind K> static void row_entry(const PreparedFragment &p,py::ssize_t i,double *destination,const double *lo,const double *hi) {
#if AOL_X86
    if(lo&&p.row_simd){p.emit_vector<K>(i,destination,lo,hi);return;}
#endif
    for(py::ssize_t j=0;j<p.d;++j){double x=p.coordinate<K>(i,j);destination[j]=lo?repair_coordinate<false,false>(x,lo[j],hi[j]):x;}
  }
  template<FragmentKind K> static void four_entry(const PreparedFragment &p,py::ssize_t i,void **destinations,const double *lo,const double *hi) {
#if AOL_X86
    p.emit_four<K>(i,destinations,lo,hi);
#else
    for(int r=0;r<4;++r)row_entry<K>(p,i+r,static_cast<double *>(destinations[r]),lo,hi);
#endif
  }
  void emit(py::ssize_t i,void *destination,const double *lo,const double *hi)const {
    if(integer){for(py::ssize_t j=0;j<d;++j)static_cast<int64_t *>(destination)[j]=integer_value(i,j);return;}
    row_kernel(*this,i,static_cast<double *>(destination),lo,hi);
  }
#if AOL_X86
  template<FragmentKind K> AOL_AVX2 void emit_vector(py::ssize_t i,double *destination,const double *lo,const double *hi)const {
    py::ssize_t j=0;auto one=_mm256_set1_pd(1.);
    for(;j+4<=d;j+=4) {
      __m256d x;
      if constexpr(K==FragmentKind::Uniform||K==FragmentKind::Reinit) {
        auto l=_mm256_loadu_pd(lower+j),h=_mm256_loadu_pd(upper+j);
        x=_mm256_add_pd(l,_mm256_mul_pd(_mm256_sub_pd(h,l),_mm256_loadu_pd(coins.values.data()+i*d+j)));
        if constexpr(K==FragmentKind::Uniform)x=_mm256_blendv_pd(_mm256_loadu_pd(parent_row<double>(i)+j),x,
          _mm256_cmp_pd(_mm256_loadu_pd(masks.values.data()+i*d+j),_mm256_set1_pd(probability),_CMP_LT_OQ));
      }else if constexpr(K==FragmentKind::Differential||K==FragmentKind::Select) {
        auto donor=K==FragmentKind::Select?_mm256_loadu_pd(donor_values.data()+i*d+j):
          _mm256_add_pd(_mm256_loadu_pd(bases[i]+j),_mm256_mul_pd(_mm256_set1_pd(scale),_mm256_sub_pd(_mm256_loadu_pd(seconds[i]+j),_mm256_loadu_pd(thirds[i]+j))));
        auto take=_mm256_cmp_pd(_mm256_loadu_pd(masks.values.data()+i*d+j),_mm256_set1_pd(probability),_CMP_LT_OQ);
        if constexpr(K==FragmentKind::Select)take=_mm256_or_pd(take,_mm256_cmp_pd(_mm256_set_pd(j+3,j+2,j+1,j),_mm256_set1_pd(forced[i]),_CMP_EQ_OQ));
        x=_mm256_blendv_pd(_mm256_loadu_pd(parent_row<double>(i)+j),donor,take);
      }else {
        auto half=(n+1)/2,pair=i<half?i:i-half;
        auto first=_mm256_loadu_pd(parent_row<double>(pair)+j),second=_mm256_loadu_pd(parent_row<double>(n/2+pair)+j);
        if constexpr(K==FragmentKind::UniformCross) {
          auto take=_mm256_cmp_pd(_mm256_loadu_pd(coins.values.data()+pair*d+j),_mm256_set1_pd(probability),_CMP_LT_OQ);
          x=i<half?_mm256_blendv_pd(first,second,take):_mm256_blendv_pd(second,first,take);
        }else {
          auto q=!coefficients.is_none()?_mm256_loadu_pd(static_cast<const double *>(coefficients.data())+(K==FragmentKind::SBX?pair*d:0)+j):_mm256_set1_pd(probability);
          if constexpr(K==FragmentKind::SBX){auto plus=_mm256_add_pd(one,q),minus=_mm256_sub_pd(one,q);
            x=_mm256_mul_pd(_mm256_set1_pd(.5),_mm256_add_pd(_mm256_mul_pd(i<half?plus:minus,first),_mm256_mul_pd(i<half?minus:plus,second)));
          }else x=_mm256_add_pd(_mm256_mul_pd(q,i<half?first:second),_mm256_mul_pd(_mm256_sub_pd(one,q),i<half?second:first));
        }
      }
      auto l=_mm256_loadu_pd(lo+j),h=_mm256_loadu_pd(hi+j);
      auto midpoint=_mm256_add_pd(l,_mm256_mul_pd(_mm256_sub_pd(h,l),_mm256_set1_pd(.5)));
      x=_mm256_blendv_pd(x,midpoint,_mm256_cmp_pd(x,x,_CMP_UNORD_Q));
      _mm256_storeu_pd(destination+j,_mm256_min_pd(_mm256_max_pd(x,l),h));
    }
    for(;j<d;++j)destination[j]=repair_coordinate<false,false>(coordinate<K>(i,j),lo[j],hi[j]);
  }
  template<FragmentKind K> AOL_AVX2 void emit_four(py::ssize_t start,void **destinations,const double *lo,const double *hi)const {
    // Four independent individuals per instruction, including D=10/40. No
    // population transpose, no reduction reorder and no FMA contraction.
    for(py::ssize_t j=0;j<d;++j){
      __m256d x;
      if constexpr(K==FragmentKind::Uniform||K==FragmentKind::Reinit) {
        auto c=_mm256_set_pd(coins.values[(start+3)*d+j],coins.values[(start+2)*d+j],coins.values[(start+1)*d+j],coins.values[start*d+j]);
        x=_mm256_add_pd(_mm256_set1_pd(lower[j]),_mm256_mul_pd(_mm256_set1_pd(upper[j]-lower[j]),c));
        if constexpr(K==FragmentKind::Uniform) {
          auto m=_mm256_set_pd(masks.values[(start+3)*d+j],masks.values[(start+2)*d+j],masks.values[(start+1)*d+j],masks.values[start*d+j]);
          auto parent=_mm256_set_pd(parent_row<double>(start+3)[j],parent_row<double>(start+2)[j],parent_row<double>(start+1)[j],parent_row<double>(start)[j]);
          x=_mm256_blendv_pd(parent,x,_mm256_cmp_pd(m,_mm256_set1_pd(probability),_CMP_LT_OQ));
        }
      }else x=_mm256_set_pd(coordinate<K>(start+3,j),coordinate<K>(start+2,j),coordinate<K>(start+1,j),coordinate<K>(start,j));
      auto l=_mm256_set1_pd(lo[j]),h=_mm256_set1_pd(hi[j]);
      auto midpoint=_mm256_add_pd(l,_mm256_mul_pd(_mm256_sub_pd(h,l),_mm256_set1_pd(.5)));
      x=_mm256_blendv_pd(x,midpoint,_mm256_cmp_pd(x,x,_CMP_UNORD_Q));
      x=_mm256_min_pd(_mm256_max_pd(x,l),h);
      alignas(32) double values[4];_mm256_store_pd(values,x);
      for(int r=0;r<4;++r)static_cast<double *>(destinations[r])[j]=values[r];
    }
  }
#endif
  void emit_many(py::ssize_t start,size_t count,void **destinations,const double *lo,const double *hi)const {
#if AOL_X86
    if(grouped&&!integer&&count==4){four_kernel(*this,start,destinations,lo,hi);return;}
#endif
    for(size_t r=0;r<count;++r)emit(start+r,destinations[r],lo,hi);
  }
};
// The generation owns this frame until repair/commit finishes. It is never
// exported, cached across callbacks, or allocated as a shared heap object.
inline thread_local PreparedFragment *fragment_sink=nullptr;
inline thread_local bool direct_result_capture=false;
struct FragmentCapture {
  PreparedFragment *previous;bool previous_direct;
  explicit FragmentCapture(PreparedFragment *frame,bool direct=false):previous(fragment_sink),previous_direct(direct_result_capture){fragment_sink=frame;direct_result_capture=direct;}
  ~FragmentCapture(){fragment_sink=previous;direct_result_capture=previous_direct;}
};
}

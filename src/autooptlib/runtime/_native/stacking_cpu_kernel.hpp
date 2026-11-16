#pragma once
#include "array_workspace.hpp"
// Retain NumPy's reduction tree. Variance follows public np.var's ordered
// sum/divide, subtract/square, sum/divide steps; PyArray_Std's reciprocal
// multiplication is not bitwise equivalent to that final division.
extern "C" void autoopt_cpu_stacking(int64_t n,int64_t d,const void*raw,double*y,double*c,const void*opaque) {
 try {
  const auto *x=static_cast<const int64_t*>(raw),*unused=x;
  (void)unused;const auto *p=static_cast<const double*>(opaque);const int64_t N=d/3,L=int64_t(p[0]),P=int64_t(p[1]);
  const auto *width=p+20,*height=width+N,*weight=height+N,*frequency=weight+N,*shared=frequency+N;
  const auto *mat=shared+N,*prod=mat+int64_t(p[9])+1+int64_t(mat[int64_t(p[9])]);
  auto scratch=native_components::workspace_array<double>(3*N+L+1);
  auto single=native_components::workspace_array<float>(N+1);
  double *a=scratch.mutable_data(),*b=a+N,*loc=b+N,*level=loc+N;
  // These private headers never escape a C reduction. Only their lengths
  // change, as for the existing float32 sum header below; payloads stay put.
  py::array_t<double> av({py::ssize_t(0)},{py::ssize_t(8)},a,scratch);
  py::array_t<double> bv({py::ssize_t(0)},{py::ssize_t(8)},b,scratch);
  py::array_t<double> lv({py::ssize_t(0)},{py::ssize_t(8)},level,scratch);
  auto output_storage=native_components::workspace_array<double>(1);
  py::array_t<double> output(std::vector<py::ssize_t>{},std::vector<py::ssize_t>{},output_storage.mutable_data(),output_storage);
  py::array_t<float> foutput(std::vector<py::ssize_t>{},std::vector<py::ssize_t>{},single.mutable_data()+N,single);
  auto fscalar=py::array_t<float>({py::ssize_t(0)}, {py::ssize_t(4)}, single.mutable_data(), single);
  auto sum=[&](py::array_t<double> &view,int64_t size,bool f32=false)->double {
    if(f32) {
      for(int64_t j=0;j<size;++j)single.mutable_data()[j]=float(view.data()[j]);
      PyArray_DIMS(reinterpret_cast<PyArrayObject*>(fscalar.ptr()))[0]=size;
      auto result=py::reinterpret_steal<py::object>(PyArray_Sum(reinterpret_cast<PyArrayObject*>(fscalar.ptr()),AOL_RAVEL_AXIS,NPY_FLOAT,reinterpret_cast<PyArrayObject*>(foutput.ptr())));
      if(!result)throw py::error_already_set();return double(*foutput.data());
    }
    PyArray_DIMS(reinterpret_cast<PyArrayObject*>(view.ptr()))[0]=size;
    auto result=py::reinterpret_steal<py::object>(PyArray_Sum(reinterpret_cast<PyArrayObject*>(view.ptr()),AOL_RAVEL_AXIS,NPY_DOUBLE,reinterpret_cast<PyArrayObject*>(output.ptr())));
    if(!result)throw py::error_already_set();return *output.data();
  };
  auto spread=[&](const double *groups,int64_t count)->double {
    double total=0.;const auto *indices=groups+count+1;
    for(int64_t g=0;g<count;++g){auto begin=int64_t(groups[g]),end=int64_t(groups[g+1]);auto size=end-begin;if(size<=1)continue;
      for(int64_t j=0;j<size;++j)a[j]=loc[int64_t(indices[begin+j])];
      const double mean=sum(av,size)/double(size);
      for(int64_t j=0;j<size;++j){const double difference=a[j]-mean;a[j]=difference*difference;}
      total+=sum(av,size)/double(size);}
    return total/p[18];
  };
  for(int64_t i=0;i<n;++i){const auto *dec=x+i*d,*positions=dec+N,*orient=positions+N;
    for(int64_t j=0;j<N;++j){
      uint64_t bits=uint64_t(dec[j])*uint64_t(P)+uint64_t(positions[j]);int64_t value;std::memcpy(&value,&bits,8);loc[j]=double(value);}
    // Decode orientation at the existing membership test. The h/w arrays were
    // only read here; selected items retain their original collection order.
    for(int64_t l=0;l<L;++l){int64_t k=0;double maximum=0.;for(int64_t j=0;j<N;++j)if(dec[j]==l){a[k]=orient[j]==0?width[j]:height[j];b[k]=weight[j];maximum=std::max(maximum,orient[j]==0?height[j]:width[j]);++k;}
      level[l]=maximum;auto widths=sum(av,k,p[12]!=0),weights=sum(bv,k,p[13]!=0);
      c[i*2*L+l]=p[14]?double(float(widths)-float(p[2])):widths-p[2];
      c[i*2*L+L+l]=p[15]?double(float(weights)-float(p[3])):weights-p[3];}
    const double occupied=p[2]*sum(lv,L);
    for(int64_t j=0;j<N;++j)a[j]=weight[j]*double(dec[j]);const double heavy=sum(av,N)/p[16];
    const double material=spread(mat,int64_t(p[9])),product=spread(prod,int64_t(p[10]));
    int64_t k=0;for(int64_t j=0;j<N;++j)if(shared[j]!=0)a[k++]=std::abs(double(positions[j])-p[19]);const double central=sum(av,k)/p[18];
    for(int64_t j=0;j<N;++j)a[j]=frequency[j]*double(positions[j]);const double shipping=sum(av,N)/p[17];
    y[i]=occupied+(p[4]*heavy+p[5]*material+p[6]*product+p[7]*central+p[8]*shipping);
  }
 } catch(py::error_already_set &error){error.restore();} catch(const std::exception &error){PyErr_SetString(PyExc_RuntimeError,error.what());}
}

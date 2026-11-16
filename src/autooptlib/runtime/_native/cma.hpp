#pragma once
#include "continuous.hpp"
namespace native_components {

inline py::array_t<double> identity_matrix(py::ssize_t d) {
  py::array_t<double> out({d,d});std::fill_n(out.mutable_data(),d*d,0.);
  for(py::ssize_t i=0;i<d;++i)out.mutable_data()[i*d+i]=1.;return out;
}
inline py::array_t<double> zeros_vector(py::ssize_t d) {
  py::array_t<double> out(d);std::fill_n(out.mutable_data(),d,0.);return out;
}
inline py::array transpose(const py::array &input) {
  auto result=PyArray_Transpose(reinterpret_cast<PyArrayObject *>(input.ptr()),nullptr);
  if(!result)throw py::error_already_set();return py::reinterpret_steal<py::array>(result);
}
inline double sanitize(double x,double nan,double positive,double negative) {
  return std::isnan(x)?nan:x==INFINITY?positive:x==-INFINITY?negative:x;
}

template<class Math>
py::array_t<double> regularize_covariance(py::array_t<double> input,Math &math) {
  if(input.ndim()!=2||input.shape(0)!=input.shape(1)||!input.shape(0))
    throw py::value_error("CMA covariance must be a non-empty square matrix.");
  auto d=input.shape(0);auto symmetric=workspace_shaped<double>({d,d});
  // Each input coordinate is clipped once. Load both members of a pair before
  // writing the private symmetric output; the original expression order stays.
  auto source=input.unchecked<2>();
  for(py::ssize_t i=0;i<d;++i)for(py::ssize_t j=i;j<d;++j) {
    double a=std::clamp(sanitize(source(i,j),0,1e6,-1e6),-1e6,1e6);
    double b=i==j?a:std::clamp(sanitize(source(j,i),0,1e6,-1e6),-1e6,1e6);
    symmetric.mutable_data()[i*d+j]=.5*(a+b);
    if(i!=j)symmetric.mutable_data()[j*d+i]=.5*(b+a);
  }
  py::tuple decomposition;
  try {decomposition=py::cast<py::tuple>(math.invoke(math.eigh,symmetric));}
  catch(py::error_already_set &error) {
    if(!error.matches(math.linalg_error))throw;
    return identity_matrix(d);
  }
  auto eigenvalues=py::cast<py::array_t<double>>(decomposition[0]);
  auto eigenvectors=py::cast<py::array_t<double>>(decomposition[1]);
  double largest=-INFINITY;
  for(py::ssize_t i=0;i<d;++i) {
    eigenvalues.mutable_data()[i]=sanitize(eigenvalues.data()[i],1,1e6,1e-12);
    largest=std::max(largest,eigenvalues.data()[i]);
  }
  largest=std::clamp(largest,1e-12,1e6);double floor=std::max(1e-12,largest*1e-12);
  auto scaled=workspace_shaped<double>({d,d});
  for(py::ssize_t i=0;i<d;++i)for(py::ssize_t j=0;j<d;++j)
    scaled.mutable_data()[i*d+j]=eigenvectors.data()[i*d+j]*std::clamp(eigenvalues.data()[j],floor,1e6);
  auto result=py::cast<py::array_t<double>>(math.invoke(math.matmul,scaled,transpose(eigenvectors)));
  for(py::ssize_t k=0;k<d*d;++k)result.mutable_data()[k]=sanitize(result.data()[k],0,1e6,-1e6);
  // Lapack::multiply returns a fresh owning result. No callback or GIL release
  // occurs here, so symmetrize paired coordinates directly in that owner.
  for(py::ssize_t i=0;i<d;++i)for(py::ssize_t j=i;j<d;++j) {
    double a=result.data()[i*d+j],b=result.data()[j*d+i];
    double x=.5*(a+b),y=i==j?x:.5*(b+a);
    if(!std::isfinite(x)||!std::isfinite(y))return identity_matrix(d);
    result.mutable_data()[i*d+j]=x;if(i!=j)result.mutable_data()[j*d+i]=y;
  }
  return result;
}

template<class T>inline void set_default(NativeState state,const char *key,const T &value) {
  if(!state.contains(key))state.set(key,value);
}

template<class Math>
NativeResult cma_sample(py::array_t<double,py::array::c_style> input,
    py::array_t<double,py::array::c_style> population,
    py::array_t<double,py::array::c_style> bounds,
    NativeState state,bool context,bool algorithm,Random &rng,Math &math) {
  RegionScope numeric_timing(Region::Numeric);
  auto n=input.shape(0),d=input.shape(1);
  int64_t model_n=state.contains("_stream_generation_target")?state.number<int64_t>("_stream_generation_target"):n;
  if(algorithm)model_n=n;
  bool incompatible=false;
  for(auto key:{"cma_mean","cma_sigma","cma_ps","cma_pc","cma_lower","cma_upper","cma_C"}) {
    if(!state.contains(key))continue;
    auto a=py::array::ensure(state.object(key));
    bool matrix=std::string(key)=="cma_C";
    if(!a||a.ndim()!=(matrix?2:1)||a.shape(0)!=d||(matrix&&a.shape(1)!=d))incompatible=true;
  }
  if(state.contains("cma_Disturb")) {
    auto a=py::array::ensure(state.object("cma_Disturb"));
    if(!a||a.ndim()!=2||a.shape(1)!=d)incompatible=true;
  }
  if(state.contains("cma_lambda")&&state.number<int64_t>("cma_lambda")!=model_n)incompatible=true;
  if(incompatible) {
    state.erase_prefix("cma_");
  }
  int64_t half=std::max<int64_t>(1,std::floor(model_n/2.+.5));
  const char *defaults[]={"cma_w","cma_betterN","cma_csigma","cma_dsigma","cma_chiN","cma_cc","cma_ccov","cma_cmu","cma_hth"};
  const bool pure_preparation=pure_native_reductions();
  bool prepare=!pure_preparation;for(auto key:defaults)prepare|=!state.contains(key);
  if(prepare) {
    auto sequence=workspace_shaped<double>({half});std::iota(sequence.mutable_data(),sequence.mutable_data()+half,1.);
    auto logs=math.logarithm.execute(sequence);double lead=math.logarithm.scalar(half+.5);
    py::array_t<double> weights(half);
    for(int64_t i=0;i<half;++i)weights.mutable_data()[i]=lead-logs.data()[i];
    double total=py::cast<double>(reduction(weights,AOL_RAVEL_AXIS));
    auto squares=workspace_shaped<double>({half});
    for(int64_t i=0;i<half;++i){weights.mutable_data()[i]/=total;squares.mutable_data()[i]=weights.data()[i]*weights.data()[i];}
    double better=1./py::cast<double>(reduction(squares,AOL_RAVEL_AXIS));
    set_default(state,"cma_lambda",model_n);set_default(state,"cma_halfN",half);set_default(state,"cma_w",weights);
    auto put=[&](const char *key,auto operation){if(!pure_preparation||!state.contains(key)){auto value=operation();set_default(state,key,value);}};
    put("cma_betterN",[&]{return better;});
    put("cma_csigma",[&]{return (better+2)/(d+better+5);});
    double csigma=state.number<double>("cma_csigma");
    put("cma_dsigma",[&]{return csigma+2*std::max(math.square_root.scalar((better-1)/(d+1.))-1,0.)+1;});
    put("cma_chiN",[&]{return math.square_root.scalar(d)*(1-1/(4.*d)+1/(21.*d*d));});
    put("cma_cc",[&]{return (4+better/d)/(4+d+2*better/d);});
    put("cma_ccov",[&]{return 2/(std::pow(d+1.3,2)+better);});
    double ccov=state.number<double>("cma_ccov");
    put("cma_cmu",[&]{return std::min(1-ccov,2*(better-2+1/better)/(std::pow(d+2.,2)+2*better/2));});
    put("cma_hth",[&]{return (1.4+2/(d+1.))*state.number<double>("cma_chiN");});
  } else {set_default(state,"cma_lambda",model_n);set_default(state,"cma_halfN",half);}
  auto initial=py::reinterpret_borrow<py::array_t<double>>(py::handle()),sigma=py::reinterpret_borrow<py::array_t<double>>(py::handle());
  py::array_t<double> lower(d),upper(d);
  bool need_mean=!state.contains("cma_mean"),need_sigma=!state.contains("cma_sigma");
  if((need_mean||!pure_preparation)&&(!(context||algorithm)||!pure_preparation))initial=py::array_t<double>(d);
  if(need_sigma||!pure_preparation)sigma=py::array_t<double>(d);
  if(context||algorithm) {
    if(need_mean||!pure_preparation) {
      const bool private_mean=pure_preparation&&pure_native_reductions();
      auto mean=py::cast<py::array_t<double>>(reduction(algorithm?input:population,0,true));
      if(private_mean)initial=std::move(mean); // Fresh private PyArray_Mean output.
      else {if(!initial)initial=py::array_t<double>(d);std::copy_n(mean.data(),d,initial.mutable_data());}
    }
  } else if(initial)rng.uniform(initial.mutable_data(),d);
  else for(py::ssize_t j=0;j<d;++j)rng.uniform(); // Discard values, retain every original draw.
  for(py::ssize_t j=0;j<d;++j) {
    double lo=bounds.data()[j],hi=bounds.data()[d+j];
    lower.mutable_data()[j]=lo;upper.mutable_data()[j]=hi;
    if(need_mean||!pure_preparation){double value=context||algorithm?initial.data()[j]:lo+(hi-lo)*initial.data()[j];
      if(!std::isfinite(value))value=lo+.5*(hi-lo);initial.mutable_data()[j]=std::clamp(value,lo,hi);}
    if(need_sigma||!pure_preparation)sigma.mutable_data()[j]=.1*(hi-lo);
  }
  // These newly owning initial values can be published directly; they never
  // borrow the array workspace and are not overwritten after publication.
  if(need_mean)state.set("cma_mean",initial);if(need_sigma)state.set("cma_sigma",sigma);
  if(!state.contains("cma_ps"))state.set("cma_ps",zeros_vector(d));
  if(!state.contains("cma_pc"))state.set("cma_pc",zeros_vector(d));
  if(!state.contains("cma_C"))state.set("cma_C",identity_matrix(d));set_default(state,"cma_generation",py::int_(0));
  state.set("cma_lower",lower);state.set("cma_upper",upper);
  auto old_mean=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(state.object("cma_mean")),old_sigma=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(state.object("cma_sigma"));
  py::array_t<double> mean(d),current_sigma(d);
  for(py::ssize_t j=0;j<d;++j) {
    double x=old_mean.data()[j];if(!std::isfinite(x))x=lower.data()[j]+.5*(upper.data()[j]-lower.data()[j]);
    mean.mutable_data()[j]=std::clamp(x,lower.data()[j],upper.data()[j]);
    current_sigma.mutable_data()[j]=std::clamp(sanitize(old_sigma.data()[j],1e-3,1e6,1e-12),1e-12,1e6);
  }
  auto covariance=regularize_covariance(py::cast<py::array_t<double>>(state.object("cma_C")),math);
  state.set("cma_C",covariance);state.set("cma_mean",mean);state.set("cma_sigma",current_sigma);
  auto eigenvalues=py::reinterpret_borrow<py::array_t<double>>(py::handle()), eigenvectors=py::reinterpret_borrow<py::array_t<double>>(py::handle());
  try {
    auto pair=py::cast<py::tuple>(math.invoke(math.eigh,covariance));
    eigenvalues=py::cast<py::array_t<double>>(pair[0]);eigenvectors=py::cast<py::array_t<double>>(pair[1]);
  } catch(py::error_already_set &error) {
    if(!error.matches(math.linalg_error))throw;
    covariance=identity_matrix(d);state.set("cma_C",covariance);eigenvectors=identity_matrix(d);
    eigenvalues=py::array_t<double>(d);std::fill_n(eigenvalues.mutable_data(),d,1.);
  }
  auto factor=workspace_shaped<double>({d,d});
  for(py::ssize_t j=0;j<d;++j)eigenvalues.mutable_data()[j]=std::clamp(sanitize(eigenvalues.data()[j],1,std::numeric_limits<double>::max(),-std::numeric_limits<double>::max()),1e-12,1e6);
  auto roots=math.square_root.execute(eigenvalues);
  for(py::ssize_t i=0;i<d;++i)for(py::ssize_t j=0;j<d;++j)factor.mutable_data()[i*d+j]=eigenvectors.data()[i*d+j]*roots.data()[j];
  auto normal=workspace_shaped<double>({n,d});rng.normal(normal.mutable_data(),n*d);
  auto disturbance=py::cast<py::array_t<double>>(math.invoke(math.matmul,normal,transpose(factor)));
  bool finite=true;for(py::ssize_t k=0;k<n*d;++k)finite&=std::isfinite(disturbance.data()[k]);
  if(!finite) {
    state.set("cma_C",identity_matrix(d));state.set("cma_ps",zeros_vector(d));state.set("cma_pc",zeros_vector(d));
    disturbance=py::array_t<double>({n,d});rng.normal(disturbance.mutable_data(),n*d);
  }
  py::array_t<double> output({n,d});
  for(py::ssize_t k=0;k<n*d;++k)output.mutable_data()[k]=mean.data()[k%d]+current_sigma.data()[k%d]*disturbance.data()[k];
  state.set("cma_Disturb",disturbance);return state_result(output,state);
}

template<class Math>
double vector_norm(const py::array_t<double> &input,Math &math) {
  auto raw=PyArray_MatrixProduct2(input.ptr(),input.ptr(),nullptr);
  if(!raw)throw py::error_already_set();
  double squared=py::cast<double>(py::reinterpret_steal<py::object>(raw));
  return math.square_root.scalar(squared);
}

template<class Math>
NativeState cma_update(py::array_t<double,py::array::c_style> fitness,
    py::array_t<double,py::array::c_style> bounds,NativeState state,bool solution,Math &math) {
  RegionScope numeric_timing(Region::Numeric);
  auto disturbance=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(state.object("cma_Disturb"));
  if(!disturbance.size())return state;
  auto d=disturbance.shape(1),rows=disturbance.shape(0);
  auto scalar=[&](const char *key){return state.number<double>(key);};
  auto weights=py::cast<py::array_t<double>>(state.object("cma_w"));
  auto mean=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(state.object("cma_mean")),sigma=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(state.object("cma_sigma"));
  auto old_ps=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(state.object("cma_ps")),old_pc=py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(state.object("cma_pc"));
  auto covariance=regularize_covariance(py::cast<py::array_t<double>>(state.object("cma_C")),math);
  auto ranked=order(fitness);
  for(py::ssize_t i=0;i<ranked.size();++i)if(ranked.data()[i]<0||ranked.data()[i]>=rows)
    throw py::index_error("CMA fitness rank exceeds the sampled disturbance rows");
  auto selected=std::min<int64_t>({state.number<int64_t>("cma_halfN"),weights.size(),fitness.size()});
  if(!selected)return state;
  py::array_t<double> selected_weights(std::vector<py::ssize_t>{static_cast<py::ssize_t>(selected)},std::vector<py::ssize_t>{weights.strides(0)},weights.data(),weights);
  auto squares=workspace_shaped<double>({selected}),selected_disturbance=workspace_shaped<double>({selected,d});
  auto read_weights=weights.unchecked<1>();
  double total=py::cast<double>(reduction(selected_weights,AOL_RAVEL_AXIS));
  auto write_weights=weights.mutable_unchecked<1>();
  for(int64_t i=0;i<selected;++i) {
    write_weights(i)/=total; // reference w[:selected] remains a view
    squares.mutable_data()[i]=read_weights(i)*read_weights(i);
    std::copy_n(disturbance.data()+ranked.data()[i]*d,d,selected_disturbance.mutable_data()+i*d);
  }
  double better=1./py::cast<double>(reduction(squares,AOL_RAVEL_AXIS));
  auto movement=py::cast<py::array_t<double>>(math.invoke(math.matmul,selected_weights,selected_disturbance));
  py::array_t<double> next_mean(d),ps(d),pc(d);
  for(py::ssize_t j=0;j<d;++j) {
    movement.mutable_data()[j]=sanitize(movement.data()[j],0,1e6,-1e6);
    next_mean.mutable_data()[j]=mean.data()[j]+sigma.data()[j]*movement.data()[j];
  }
  auto solved=py::reinterpret_borrow<py::array_t<double>>(py::handle());
  try {
    auto chol=py::cast<py::array_t<double>>(math.invoke(math.cholesky,covariance));
    solved=py::cast<py::array_t<double>>(math.invoke(math.solve_vector,transpose(chol),movement));
  } catch(py::error_already_set &error) {
    if(!error.matches(math.linalg_error))throw;
    covariance=identity_matrix(d);solved=py::array_t<double>(d);std::copy_n(movement.data(),d,solved.mutable_data());
  }
  double csigma=scalar("cma_csigma"),dsigma=scalar("cma_dsigma"),cc=scalar("cma_cc"),
         ccov=scalar("cma_ccov"),cmu=scalar("cma_cmu"),chi=scalar("cma_chiN");
  double coefficient=math.square_root.scalar(csigma*(2-csigma)*better);
  for(py::ssize_t j=0;j<d;++j)
    ps.mutable_data()[j]=(1-csigma)*sanitize(old_ps.data()[j],0,1e6,-1e6)+coefficient*solved.data()[j];
  int64_t generation=state.contains("cma_generation")?state.number<int64_t>("cma_generation")+1:1;
  double denominator=math.square_root.scalar(1-std::pow(1-csigma,2*generation));
  double norm=vector_norm(ps,math),hs=norm/denominator<scalar("cma_hth")?1.:0.;
  coefficient=hs*math.square_root.scalar(cc*(2-cc)*better);
  for(py::ssize_t j=0;j<d;++j)
    pc.mutable_data()[j]=(1-cc)*sanitize(old_pc.data()[j],0,1e6,-1e6)+coefficient*movement.data()[j];
  double delta=(1-hs)*cc*(2-cc);
  auto updated=workspace_shaped<double>({d,d});
  for(py::ssize_t i=0;i<d;++i)for(py::ssize_t j=0;j<d;++j) {
    auto k=i*d+j;
    updated.mutable_data()[k]=(1-ccov-cmu)*covariance.data()[k]+ccov*(pc.data()[i]*pc.data()[j]+delta*covariance.data()[k]);
  }
  for(int64_t row=0;row<selected;++row) {
    double factor=cmu*read_weights(row);
    for(py::ssize_t i=0;i<d;++i)for(py::ssize_t j=0;j<d;++j)
      updated.mutable_data()[i*d+j]+=factor*(selected_disturbance.data()[row*d+i]*selected_disturbance.data()[row*d+j]);
  }
  covariance=regularize_covariance(updated,math);
  double exponent=.3*csigma/dsigma*(norm/chi-1);
  double multiplier=math.exponential.scalar(std::clamp(exponent,-20.,20.));
  py::array_t<double> next_sigma(d);
  for(py::ssize_t j=0;j<d;++j) {
    next_sigma.mutable_data()[j]=std::clamp(sanitize(sigma.data()[j]*multiplier,1e-3,1e6,1e-12),1e-12,1e6);
    if(solution&&bounds.size())next_mean.mutable_data()[j]=std::clamp(next_mean.data()[j],bounds.data()[j],bounds.data()[d+j]);
  }
  state.set("cma_mean",next_mean);state.set("cma_ps",ps);state.set("cma_pc",pc);state.set("cma_C",covariance);
  state.set("cma_sigma",next_sigma);state.set("cma_generation",generation);return state;
}
}

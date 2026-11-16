#pragma once
#include <array>
#include <memory>
#include <unordered_map>
#include <variant>
#include "array_workspace.hpp"

namespace native_components {

// Individual metadata has native identity. Python records are created lazily at
// public/history/checkpoint boundaries; callback inputs retain their own owner.
struct BatchRow {
  py::array decision;
  // A private population snapshot can initially share this owning row. Detach
  // before publishing a mutable Solution, including history/archive records.
  struct PopulationValue {py::array decision;double objective,constraint,fit;};
  std::optional<PopulationValue> population;
  bool escaped=false;
  double objective=0., constraint=0., fit=0.;
  py::object auxiliary=py::none(), record=py::none();
  BatchRow(py::array x,double o,double c,double f,py::object a,py::object r=py::none())
      :decision(std::move(x)),objective(o),constraint(c),fit(f),auxiliary(std::move(a)),record(std::move(r)){}
  void refresh_record() {
    if(record.is_none())return;
    decision=py::cast<py::array>(field(record,"dec"));objective=py::cast<double>(field(record,"obj"));
    constraint=py::cast<double>(field(record,"con"));fit=py::cast<double>(field(record,"fit"));auxiliary=field(record,"acc");
  }
  py::object materialize(py::handle solution_type) {
    if(record.is_none()) {
      if(population&&population->decision.is(decision)) {
        auto copy=PyArray_NewCopy(reinterpret_cast<PyArrayObject *>(decision.ptr()),NPY_KEEPORDER);
        if(!copy)throw py::error_already_set();population->decision=py::reinterpret_steal<py::array>(copy);
      }
      record=solution_type(decision,objective,constraint,fit,auxiliary);
    }
    return record;
  }
};
using RowPtr=std::shared_ptr<BatchRow>;

struct StorageMetrics {
  bool enabled=false;
  uint64_t evaluation_pack_bytes=0,metadata_copy_bytes=0,gather_bytes=0,snapshot_bytes=0,identity_scan_bytes=0,take_views=0,concat_views=0,gathers=0;
  uint64_t row_table_bytes=0,row_table_reuses=0,overlay_rows=0,merge_bytes=0;
};
struct DecisionRef {
  std::variant<py::array,RowPtr> owner;py::ssize_t index;
  const py::array &array()const {
    if(auto row=std::get_if<RowPtr>(&owner))return (*row)->population->decision;
    return std::get<py::array>(owner);
  }
};
struct BatchStorage;
using BatchPtr=std::shared_ptr<BatchStorage>;
struct BatchStorage {
  enum class MetadataInitialization {Defaults, Overwrite, Imported};
  py::array x;
  py::array_t<double> f=py::reinterpret_borrow<py::array_t<double>>(py::none());
  py::array_t<double> objectives=f,constraints=f;
  py::array_t<bool> valid=py::reinterpret_borrow<py::array_t<bool>>(py::handle());
  py::array_t<int64_t> ids=py::reinterpret_borrow<py::array_t<int64_t>>(py::handle());
  std::vector<RowPtr> rows;
  bool evaluated=false,records_available=false,dense_ready=true,external=false;
  py::ssize_t count=0,columns=0;
  bool integer_kind=false;
  // Layouts are immutable between native construction/materialization points.
  // Pointers are retained by layout's strong owners. Validate current NumPy
  // metadata on admission; never cache mutable dtype/contiguity assertions.
  mutable std::array<PyArrayObject *,4> row_owners; // Only [0,owner_count) is read.
  mutable int owner_count=0; // -1 unknown, 5 overflow (use the conservative scan)
  mutable unsigned char layout_class=1; // unknown/continuous/random/repeated/multiple
  mutable bool has_views=false;
  std::vector<DecisionRef> layout;
  std::vector<bool> provenance;
  std::shared_ptr<StorageMetrics> metrics;
  static bool weakly_exposed(PyObject *object) {
    return Py_TYPE(object)->tp_weaklistoffset && *PyObject_GET_WEAKREFS_LISTPTR(object);
  }
  static bool private_row(const RowPtr &row) {
    auto expected=(row->record.is_none()?1:2)+(row->population&&row->population->decision.is(row->decision)?1:0);
    if(Py_REFCNT(row->decision.ptr())>expected||weakly_exposed(row->decision.ptr())
      ||(!row->record.is_none()&&(Py_REFCNT(row->record.ptr())>1||weakly_exposed(row->record.ptr()))))row->escaped=true;
    return !row->escaped;
  }
  BatchStorage(py::ssize_t n,py::ssize_t d,bool integer):
      x(py::reinterpret_borrow<py::array>(py::none())),count(n),columns(d),integer_kind(integer) {
    dense_ready=false;layout.reserve(n);rows.resize(n);provenance.resize(n);
    valid=workspace_array<bool>(n);ids=workspace_array<int64_t>(n);
  }
  py::array_t<double> fitness() {
    if(evaluated&&f.is_none()) {
      f=workspace_array<double>(size());
      for(py::ssize_t i=0;i<size();++i)f.mutable_data()[i]=rows[i]->population?rows[i]->population->fit:rows[i]->fit;
    }
    return f;
  }
  void ensure_values() {
    fitness();
    if(evaluated&&objectives.is_none()) {
      objectives=workspace_array<double>(size());constraints=workspace_array<double>(size());
      for(py::ssize_t i=0;i<size();++i){objectives.mutable_data()[i]=rows[i]->population?rows[i]->population->objective:rows[i]->objective;constraints.mutable_data()[i]=rows[i]->population?rows[i]->population->constraint:rows[i]->constraint;}
    }
  }
  bool private_rows()const {
    if(external||!evaluated)return false;
    for(auto &row:rows)if(!row||!row->record.is_none()||!private_row(row))return false;
    return true;
  }
  explicit BatchStorage(py::array decisions,MetadataInitialization metadata=MetadataInitialization::Defaults):x(std::move(decisions)) {
    if(x.ndim()!=2 || !(x.flags()&py::array::c_style) ||
        ((x.dtype().num()!=NPY_INT64&&x.dtype().num()!=NPY_DOUBLE)||!PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject *>(x.ptr()))))
      throw py::value_error("Population decisions must be a contiguous 2-D float64 or int64 matrix");
    count=x.shape(0);columns=x.shape(1);integer_kind=x.dtype().num()==NPY_INT64;
    layout.reserve(count);for(py::ssize_t i=0;i<count;++i)layout.push_back({x,i});
    provenance.assign(count,false);
    if(metadata!=MetadataInitialization::Imported) {
      valid=workspace_array<bool>(size());ids=workspace_array<int64_t>(size());
      if(metadata==MetadataInitialization::Defaults) {
        std::fill_n(valid.mutable_data(),size(),false);
        std::iota(ids.mutable_data(),ids.mutable_data()+size(),0);
      }
    }
    rows.resize(size());
    reset_data_facts();
  }
  void reset_data_facts()const {
    owner_count=size()?1:0;row_owners[0]=reinterpret_cast<PyArrayObject *>(x.ptr());layout_class=1;
  }
  void invalidate_data_facts()const {owner_count=-1;layout_class=0;}
  void discover_row_owners()const {
    if(owner_count>=0)return;
    owner_count=0;
    for(const auto &ref:layout) {
      if(std::holds_alternative<RowPtr>(ref.owner)){owner_count=5;return;}
      auto owner=reinterpret_cast<PyArrayObject *>(ref.array().ptr());
      bool present=false;for(int i=0;i<owner_count;++i)present|=row_owners[i]==owner;
      if(!present){if(owner_count==4){owner_count=5;return;}row_owners[owner_count++]=owner;}
    }
  }
  py::ssize_t size()const{return count;}
  py::ssize_t dimension()const{return columns;}
  bool integer()const{return integer_kind;}
  bool rows_compatible()const {
    if(external)return false;
    discover_row_owners();
    auto compatible=[&](PyArrayObject *owner) {
      const auto type=PyArray_TYPE(owner);
      return PyArray_CheckExact(reinterpret_cast<PyObject *>(owner))&&PyArray_ISCARRAY_RO(owner)&&
        PyArray_ISNOTSWAPPED(owner)&&(type==NPY_INT64||type==NPY_DOUBLE)&&((type==NPY_INT64)==integer());
    };
    if(owner_count<=4){for(int i=0;i<owner_count;++i)if(!compatible(row_owners[i]))return false;}
    else {PyArrayObject *previous=nullptr;for(const auto &ref:layout){auto owner=reinterpret_cast<PyArrayObject *>(ref.array().ptr());if(owner!=previous&&!compatible(owner))return false;previous=owner;}}
    return true;
  }
  std::string layout_kind()const {
    if(dense_ready)return "continuous";
    if(layout.empty())return "continuous";
    if(!layout_class) {
      discover_row_owners();
      if(owner_count>1)layout_class=4;
      else {
        bool continuous=true;for(size_t i=1;i<layout.size();++i)continuous&=layout[i].index==layout[0].index+py::ssize_t(i);
        if(continuous)layout_class=1;
        else {
          ScratchVector<int64_t> indices_lease;auto &indices=indices_lease.values;
          indices.reserve(layout.size());for(const auto &ref:layout)indices.push_back(ref.index);
          std::sort(indices.begin(),indices.end());
          layout_class=std::adjacent_find(indices.begin(),indices.end())==indices.end()?2:3;
        }
      }
    }
    return layout_class==1?"continuous":layout_class==2?"random":layout_class==3?"repeated":"multiple";
  }
  static DecisionInput read_rows(const BatchPtr &source) {
    DecisionInput out;out.n=source->size();out.d=source->dimension();out.integer=source->integer();out.keepalive=source;
    if(!source->rows_compatible())throw py::value_error("Indexed input requires private, homogeneous native rows");
    if(source->dense_ready){out.allocation=source->x;out.dense=source->x.data();}
    if(!source->dense_ready) {
      out.rows.reserve(out.n);
      for(auto &ref:source->layout){out.rows.push_back(static_cast<const char *>(ref.array().data())+ref.index*out.d*8);}
    }
    source->has_views=true;
    if(source->metrics&&source->metrics->enabled&&!source->dense_ready)source->metrics->row_table_bytes+=out.n*sizeof(void*);
    return out;
  }
  static BatchPtr overlay(BatchPtr base,BatchPtr changed,const std::vector<int64_t> &active) {
    if(base->integer()!=changed->integer()||base->dimension()!=changed->dimension()||changed->size()!=static_cast<py::ssize_t>(active.size()))
      throw py::value_error("Overlay requires matching native row types and dimensions");
    auto out=std::make_shared<BatchStorage>(base->size(),base->dimension(),base->integer());
    out->count=base->size();out->dense_ready=false;out->metrics=base->metrics;out->layout=base->layout;
    std::fill_n(out->valid.mutable_data(),out->count,false);
    std::copy_n(base->ids.data(),out->count,out->ids.mutable_data());
    for(size_t i=0;i<active.size();++i){if(active[i]<0||active[i]>=base->size())throw py::index_error("Overlay row index out of range");out->layout[active[i]]=changed->layout[i];}
    out->invalidate_data_facts();
    base->has_views=true;changed->has_views=true;
    if(out->metrics&&out->metrics->enabled)out->metrics->overlay_rows+=out->count;
    return out;
  }
  static BatchPtr merge(BatchPtr base,BatchPtr changed,const std::vector<int64_t> &active) {
    auto result=py::array(base->dtype(),std::vector<py::ssize_t>{base->size(),base->dimension()});
    for(py::ssize_t i=0;i<base->size();++i)base->write_row(i,static_cast<char *>(result.mutable_data())+i*base->dimension()*8);
    for(size_t i=0;i<active.size();++i)changed->write_row(i,static_cast<char *>(result.mutable_data())+active[i]*base->dimension()*8);
    if(base->metrics&&base->metrics->enabled)base->metrics->merge_bytes+=(base->size()+active.size())*base->dimension()*8;
    auto out=std::make_shared<BatchStorage>(result);out->metrics=base->metrics;return out;
  }
  py::dtype dtype()const{return integer()?py::dtype::of<int64_t>():py::dtype::of<double>();}
  void write_row(py::ssize_t i,char *destination)const {
    const auto &ref=layout.at(i);
    auto source=static_cast<const char *>(ref.array().data())+ref.index*dimension()*8;
    if((ref.array().dtype().num()==NPY_INT64)==integer())std::memcpy(destination,source,dimension()*8);
    else for(py::ssize_t j=0;j<dimension();++j)reinterpret_cast<double *>(destination)[j]=static_cast<double>(reinterpret_cast<const int64_t *>(source)[j]);
  }
  py::array dense() {
    if(!dense_ready) {
      x=py::array(dtype(),std::vector<py::ssize_t>{size(),dimension()});
      for(py::ssize_t i=0;i<size();++i)write_row(i,static_cast<char *>(x.mutable_data())+i*dimension()*8);
      // Releasing an old owner can run a NumPy weakref callback. A callback
      // must never see owner facts retained from an earlier partial layout.
      for(py::ssize_t i=0;i<size();++i){invalidate_data_facts();layout[i]={x,i};}
      dense_ready=true;
      reset_data_facts();
      if(metrics&&metrics->enabled){++metrics->gathers;metrics->gather_bytes+=size()*dimension()*8;}
    }
    return x;
  }
  py::array expose_dense() {
    dense();
    if(has_views&&!external) {
      auto copy=py::array(dtype(),std::vector<py::ssize_t>{size(),dimension()});
      std::memcpy(copy.mutable_data(),x.data(),x.nbytes());x=std::move(copy);
      for(py::ssize_t i=0;i<size();++i){invalidate_data_facts();layout[i]={x,i};}
      reset_data_facts();
      if(metrics&&metrics->enabled)metrics->snapshot_bytes+=x.nbytes();
    }
    external=true;return x;
  }
  void allocate_values(bool initialize_valid=true) {
    f=workspace_array<double>(size());objectives=workspace_array<double>(size());constraints=workspace_array<double>(size());
    evaluated=true;records_available=true;if(initialize_valid)std::fill_n(valid.mutable_data(),size(),true);
  }
  py::array copy_row(py::ssize_t i)const {
    py::array out(dtype(),std::vector<py::ssize_t>{dimension()});
    write_row(i,static_cast<char *>(out.mutable_data()));
    if(metrics&&metrics->enabled)metrics->snapshot_bytes+=dimension()*8;
    return out;
  }
  static bool same_double(double a,double b){return std::memcmp(&a,&b,8)==0;}
  bool same_row(const py::array &a,py::ssize_t i)const {
    if(a.ndim()!=1||a.size()!=dimension()||!PyArray_EquivTypes(reinterpret_cast<PyArray_Descr *>(a.dtype().ptr()),reinterpret_cast<PyArray_Descr *>(dtype().ptr())))return false;
    auto source=static_cast<const char *>(a.data());const auto &ref=layout.at(i);
    auto converted=py::reinterpret_borrow<py::array>(py::handle());
    const char *target;
    if((ref.array().dtype().num()==NPY_INT64)==integer())target=static_cast<const char *>(ref.array().data())+ref.index*dimension()*8;
    else{converted=copy_row(i);target=static_cast<const char *>(converted.data());}
    if(metrics&&metrics->enabled)metrics->identity_scan_bytes+=dimension()*8;
    if(a.strides(0)==8)return std::memcmp(source,target,dimension()*8)==0;
    for(py::ssize_t j=0;j<dimension();++j)if(std::memcmp(source+j*a.strides(0),target+j*8,8))return false;
    return true;
  }
  RowPtr snapshot_row(py::ssize_t i) {
    if(!evaluated)throw py::value_error("This component requires evaluated population rows");
    auto row=rows.at(i);
    if(!external&&provenance.at(i)&&row&&private_row(row)&&objectives.is_none())return row;
    ensure_values();
    bool known=provenance.at(i)&&!external&&row&&private_row(row);
    bool match=records_available && row && (known||same_row(row->decision,i))
        && same_double(row->fit,f.data()[i]) && same_double(row->objective,objectives.data()[i])
        && same_double(row->constraint,constraints.data()[i]);
    if(match&&!known&&!row->record.is_none()) {
      auto dec=py::array::ensure(field(row->record,"dec"));match=dec&&same_row(dec,i);
      for(auto pair:{std::make_pair("fit",f.data()[i]),std::make_pair("obj",objectives.data()[i]),std::make_pair("con",constraints.data()[i])}) {
        auto value=field(row->record,pair.first);match=match&&!value.is_none()&&same_double(py::cast<double>(value),pair.second);
      }
      match=match&&field(row->record,"acc").is(row->auxiliary);
    }
    if(!match)row=std::make_shared<BatchRow>(copy_row(i),objectives.data()[i],constraints.data()[i],f.data()[i],row?row->auxiliary:py::none());
    return row;
  }
  py::object public_row(py::ssize_t i,py::handle solution_type) {
    dense();ensure_values();
    // As in PopulationBatch.to_solutions, a replacement boundary record must
    // not overwrite the original record association held by the matrix.
    return snapshot_row(i)->materialize(solution_type);
  }
  py::object solutions(py::handle solution_type,py::handle set_type) {
    dense();ensure_values();
    py::list records;for(py::ssize_t i=0;i<size();++i){auto row=snapshot_row(i);records.append(row->materialize(solution_type));row->escaped=true;}
    return set_type(records);
  }
  static bool decision_integer(std::vector<PyArrayObject *>&arrays,bool empty_integer=false) {
    if(arrays.empty())return empty_integer;
    const auto integer_dtype=py::dtype::of<int64_t>(),float_dtype=py::dtype::of<double>();
    bool integer=true,native=true;
    for(auto array:arrays) {
      auto dtype=reinterpret_cast<PyObject *>(PyArray_DESCR(array));
      if(dtype==float_dtype.ptr())integer=false;
      else if(dtype!=integer_dtype.ptr()){native=false;break;}
    }
    if(native)return integer;
    // Match np.stack: promote every row before choosing the storage dtype.
    auto raw_dtype=PyArray_ResultType(arrays.size(),arrays.data(),0,nullptr);
    if(!raw_dtype)throw py::error_already_set();
    auto dtype=py::reinterpret_steal<py::dtype>(reinterpret_cast<PyObject *>(raw_dtype));
    if(dtype.num()!=NPY_INT64&&dtype.num()!=NPY_DOUBLE)
      throw py::value_error("Population decisions must be a 2-D float64 or int64 matrix");
    return dtype.num()==NPY_INT64;
  }
  static BatchPtr from_rows(const std::vector<RowPtr>&rows,py::ssize_t dimension,bool empty_integer=false,std::shared_ptr<StorageMetrics> metrics=nullptr,bool borrow=false) {
    RegionScope timing(Region::Pack);
    for(auto &row:rows)row->refresh_record();
    std::vector<PyArrayObject *> decisions;decisions.reserve(rows.size());
    for(auto &row:rows)decisions.push_back(reinterpret_cast<PyArrayObject *>(row->decision.ptr()));
    bool integer=BatchStorage::decision_integer(decisions,empty_integer);
    if(borrow) {
      for(auto &row:rows)borrow&=row->record.is_none()&&private_row(row)&&PyArray_CheckExact(row->decision.ptr())&&
        PyArray_ISNOTSWAPPED(reinterpret_cast<PyArrayObject *>(row->decision.ptr()))&&row->decision.ndim()==1&&row->decision.size()==dimension&&(row->decision.flags()&py::array::c_style)&&
        ((row->decision.dtype().num()==NPY_INT64)==integer);
      if(borrow) {
        auto out=std::make_shared<BatchStorage>(rows.size(),dimension,integer);out->rows=rows;out->metrics=metrics;
        out->evaluated=true;out->records_available=true;
        std::fill_n(out->valid.mutable_data(),out->size(),true);std::iota(out->ids.mutable_data(),out->ids.mutable_data()+out->size(),0);
        for(auto &row:rows){if(!row->population)row->population=BatchRow::PopulationValue{row->decision,row->objective,row->constraint,row->fit};out->layout.push_back({row,0});}
        std::fill(out->provenance.begin(),out->provenance.end(),true);out->invalidate_data_facts();return out;
      }
    }
    py::array x(integer?py::dtype::of<int64_t>():py::dtype::of<double>(),std::vector<py::ssize_t>{static_cast<py::ssize_t>(rows.size()),dimension});
    auto out=std::make_shared<BatchStorage>(x,MetadataInitialization::Overwrite);out->allocate_values();
    std::iota(out->ids.mutable_data(),out->ids.mutable_data()+out->size(),0);out->rows=rows;out->metrics=metrics;
    if(metrics&&metrics->enabled)metrics->evaluation_pack_bytes+=rows.size()*dimension*8;
    for(size_t i=0;i<rows.size();++i) {
      auto &row=rows[i];auto a=integer?py::array(py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(row->decision)):py::array(py::array_t<double,py::array::c_style|py::array::forcecast>::ensure(row->decision));
      if(!a||a.ndim()!=1||a.size()!=dimension)throw py::value_error("Evaluation decision dimension changed");
      std::memcpy(static_cast<char *>(x.mutable_data())+i*dimension*8,a.data(),dimension*8);
      out->f.mutable_data()[i]=row->fit;out->objectives.mutable_data()[i]=row->objective;out->constraints.mutable_data()[i]=row->constraint;
    }
    for(size_t i=0;i<rows.size();++i)out->provenance[i]=rows[i]->decision.dtype().is(out->dtype());
    return out;
  }
  static BatchPtr from_solutions(py::handle source,py::ssize_t dimension=0,bool integer=false) {
    auto values=records(source);auto n=static_cast<py::ssize_t>(values.size());
    std::vector<PyArrayObject *> decisions;decisions.reserve(n);
    for(auto value:values) {
      auto decision=py::cast<py::array>(field(value,"dec"));
      decisions.push_back(reinterpret_cast<PyArrayObject *>(decision.ptr()));
    }
    integer=decision_integer(decisions,integer);
    auto x=py::reinterpret_borrow<py::array>(py::handle());
    if(n)x=matrix(source,integer);
    else x=py::array(integer?py::dtype::of<int64_t>():py::dtype::of<double>(),std::vector<py::ssize_t>{0,dimension});
    // An unobserved, owning packed matrix already is the required snapshot.
    // Borrowed rows, public matrices and weakly observed owners still copy.
    if(n&&!(PyArray_CheckExact(x.ptr())&&x.owndata()&&Py_REFCNT(x.ptr())==1&&!weakly_exposed(x.ptr()))) {
      py::array copy(x.dtype(),std::vector<py::ssize_t>{n,x.shape(1)});std::memcpy(copy.mutable_data(),x.data(),x.nbytes());x=std::move(copy);
    }
    auto out=std::make_shared<BatchStorage>(x,MetadataInitialization::Overwrite);out->allocate_values();
    std::iota(out->ids.mutable_data(),out->ids.mutable_data()+out->size(),0);
    for(py::ssize_t i=0;i<n;++i) {
      auto value=values[i];double o=py::cast<double>(field(value,"obj")),c=py::cast<double>(field(value,"con")),f=py::cast<double>(field(value,"fit"));
      out->f.mutable_data()[i]=f;out->objectives.mutable_data()[i]=o;out->constraints.mutable_data()[i]=c;
      out->rows[i]=std::make_shared<BatchRow>(py::cast<py::array>(field(value,"dec")),o,c,f,field(value,"acc"),py::reinterpret_borrow<py::object>(value));
    }
    for(py::ssize_t i=0;i<n;++i)out->provenance[i]=out->rows[i]->decision.dtype().is(out->dtype());
    return out;
  }
  static BatchPtr from_population(py::handle source) {
    if(population_storage_type&&PyObject_TypeCheck(source.ptr(),population_storage_type))return py::cast<BatchPtr>(source);
    auto x=py::cast<py::array>(field(source,"_batch_X"));auto out=std::make_shared<BatchStorage>(x,MetadataInitialization::Imported);out->external=true;
    auto rawf=field(source,"_batch_F");
    if(!rawf.is_none()) {
      out->f=py::cast<py::array_t<double>>(rawf);out->evaluated=true;
      auto o=field(source,"objectives"),c=field(source,"constraints");
      if(!o.is_none())out->objectives=py::cast<py::array_t<double>>(o);
      else {out->objectives=workspace_array<double>(out->size());std::copy_n(out->f.data(),out->size(),out->objectives.mutable_data());}
      if(!c.is_none())out->constraints=py::cast<py::array_t<double>>(c);
      else {out->constraints=workspace_array<double>(out->size());std::fill_n(out->constraints.mutable_data(),out->size(),0.);}
      auto aux=field(source,"auxiliary"),recs=field(source,"_records");out->records_available=!recs.is_none();
      for(py::ssize_t i=0;i<out->size();++i)out->rows[i]=std::make_shared<BatchRow>(out->copy_row(i),out->objectives.data()[i],out->constraints.data()[i],out->f.data()[i],aux.is_none()?py::none():py::reinterpret_borrow<py::object>(aux[py::int_(i)]),recs.is_none()?py::none():py::reinterpret_borrow<py::object>(recs[py::int_(i)]));
    }
    auto v=field(source,"valid"),ids=field(source,"row_ids");
    if(!v.is_none())out->valid=py::cast<py::array_t<bool>>(v);
    else {out->valid=workspace_array<bool>(out->size());std::fill_n(out->valid.mutable_data(),out->size(),out->evaluated);}
    if(!ids.is_none())out->ids=py::cast<py::array_t<int64_t>>(ids);
    else {out->ids=workspace_array<int64_t>(out->size());std::iota(out->ids.mutable_data(),out->ids.mutable_data()+out->size(),0);}
    return out;
  }
  BatchPtr take(const int64_t *indices,size_t count,bool native_rows=false) {
    const bool borrow=native_rows&&private_rows();
    if(evaluated&&!borrow)ensure_values();
    auto out=std::make_shared<BatchStorage>(count,dimension(),integer());
    out->count=count;out->dense_ready=false;out->external=external;out->metrics=metrics;
    if(evaluated&&!borrow)out->allocate_values(false);else out->evaluated=evaluated;out->records_available=records_available;
    for(size_t j=0;j<count;++j) {
      auto i=indices[j];if(i<0)i+=size();if(i<0||i>=size())throw py::index_error("Population row index out of range");
      out->layout.push_back(layout[i]);out->rows[j]=rows[i];out->provenance[j]=provenance[i];
      out->valid.mutable_data()[j]=valid.data()[i];out->ids.mutable_data()[j]=ids.data()[i];
      if(evaluated&&!borrow){out->f.mutable_data()[j]=f.data()[i];out->objectives.mutable_data()[j]=objectives.data()[i];out->constraints.mutable_data()[j]=constraints.data()[i];}
    }
    has_views=true;if(external){out->dense();out->external=false;}
    if(!out->dense_ready) {
      out->invalidate_data_facts();
      if(!out->size())out->owner_count=0;
      else if(owner_count==1){out->owner_count=1;out->row_owners[0]=row_owners[0];}
    }
    if(metrics&&metrics->enabled){++metrics->take_views;metrics->metadata_copy_bytes+=count*(evaluated&&!borrow?33:9);}
    return out;
  }
  BatchPtr take(const std::vector<int64_t>&indices,bool native_rows=false) {
    return take(indices.data(),indices.size(),native_rows);
  }
  BatchPtr take(py::handle indices,bool native_rows=false) {
    auto a=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(indices);
    if(!a)throw py::type_error("Population indices must be integers");
    return take(a.data(),a.size(),native_rows);
  }
  static BatchPtr concatenate(const std::vector<BatchPtr>&parts) {
    if(parts.empty())throw py::value_error("At least one population batch is required");
    if(parts.size()==1)return parts[0];
    py::ssize_t n=0,d=parts[0]->dimension();bool integer=true,evaluated=true,available=true,borrow=true;
    for(auto &p:parts){if(p->dimension()!=d)throw py::value_error("Population dimensions must match");n+=p->size();integer&=p->integer();evaluated&=p->evaluated;available&=p->records_available;borrow&=p->private_rows();}
    auto out=std::make_shared<BatchStorage>(n,d,integer);
    out->count=n;out->dense_ready=false;out->metrics=parts[0]->metrics;
    if(evaluated&&!borrow)out->allocate_values(false);else out->evaluated=evaluated;out->records_available=available;py::ssize_t offset=0;
    for(auto &p:parts) {
      p->has_views=true;
      if(p->external) {
        // Freeze only an externally mutable segment, never the whole join.
        py::array snapshot(p->dtype(),std::vector<py::ssize_t>{p->size(),d});
        for(py::ssize_t i=0;i<p->size();++i)p->write_row(i,static_cast<char *>(snapshot.mutable_data())+i*d*8);
        for(py::ssize_t i=0;i<p->size();++i)out->layout.push_back({snapshot,i});
        if(out->metrics&&out->metrics->enabled)out->metrics->snapshot_bytes+=p->size()*d*8;
      }else out->layout.insert(out->layout.end(),p->layout.begin(),p->layout.end());
      for(py::ssize_t i=0;i<p->size();++i){out->rows[offset+i]=p->rows[i];out->provenance[offset+i]=p->provenance[i]&&p->integer()==integer;out->valid.mutable_data()[offset+i]=p->valid.data()[i];out->ids.mutable_data()[offset+i]=p->ids.data()[i];}
      if(evaluated&&!borrow){p->ensure_values();std::copy_n(p->f.data(),p->size(),out->f.mutable_data()+offset);std::copy_n(p->objectives.data(),p->size(),out->objectives.mutable_data()+offset);std::copy_n(p->constraints.data(),p->size(),out->constraints.mutable_data()+offset);}
      offset+=p->size();
    }
    if(out->metrics&&out->metrics->enabled){++out->metrics->concat_views;out->metrics->metadata_copy_bytes+=n*(evaluated&&!borrow?33:9);}
    out->invalidate_data_facts();
    return out;
  }
  py::object to_population(py::handle type,py::handle solution_type) {
    dense();ensure_values();
    py::list acc,recs;for(py::ssize_t i=0;i<size();++i){acc.append(rows[i]?rows[i]->auxiliary:py::none());if(records_available)recs.append(evaluated?public_row(i,solution_type):py::none());}
    return type(expose_dense(),evaluated?py::object(f):py::none(),evaluated?py::object(objectives):py::none(),evaluated?py::object(constraints):py::none(),
        "valid"_a=valid,"row_ids"_a=ids,"records"_a=records_available?py::object(recs):py::none(),"auxiliary"_a=acc);
  }
};
inline std::optional<NativePopulationOwner> read_native_population(py::handle input) {
  if(population_storage_type&&PyObject_TypeCheck(input.ptr(),population_storage_type))return py::cast<BatchPtr>(input);
  return std::nullopt;
}
inline py::object native_population_object(const NativePopulationOwner &input){return py::cast(input);}
}

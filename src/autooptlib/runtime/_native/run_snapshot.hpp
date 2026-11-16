#pragma once
#include "batch_storage.hpp"
#include "native_state.hpp"

namespace native_components {
struct ArchiveValue {
  BatchPtr population;
  std::optional<py::array> statistics;
  py::object opaque=py::none();
  ArchiveValue()=default;
  explicit ArchiveValue(py::object value) {
    if(population_storage_type&&PyObject_TypeCheck(value.ptr(),population_storage_type))population=py::cast<BatchPtr>(value);
    else if(PyArray_CheckExact(value.ptr()))statistics=py::cast<py::array>(value);
    else opaque=std::move(value);
  }
  py::object object()const{return population?py::cast(population):statistics?py::object(*statistics):opaque;}
};
inline std::vector<ArchiveValue> import_archives(py::list values) {
  std::vector<ArchiveValue> out;out.reserve(values.size());for(auto value:values)out.emplace_back(py::reinterpret_borrow<py::object>(value));return out;
}
inline py::list export_archives(const std::vector<ArchiveValue>&values) {
  py::list out;for(auto &value:values)out.append(value.object());return out;
}
// A snapshot is an owned native graph. Numeric buffers and row/state identities
// are copied once with memoization; opaque owners use the same deepcopy memo.
class SnapshotClone {
  std::unordered_map<const StateFields *,NativeState> states_;
  std::unordered_map<const BatchStorage *,BatchPtr> populations_;
  std::unordered_map<const BatchRow *,RowPtr> rows_;
  py::dict memo_;
  std::vector<py::object> source_owners_;
  py::object deepcopy_=py::module_::import("copy").attr("deepcopy");
  static py::tuple population_key(const BatchPtr &input){return py::make_tuple("autooptlib.native_population",reinterpret_cast<uintptr_t>(input.get()));}
public:
  explicit SnapshotClone(py::dict memo=py::dict()):memo_(std::move(memo)){}
  void replace(py::handle old_owner,py::object new_owner){memo_[py::int_(reinterpret_cast<uintptr_t>(old_owner.ptr()))]=std::move(new_owner);}
  py::object owner(py::handle input) {
    if(input.is_none()||PyBool_Check(input.ptr())||PyLong_CheckExact(input.ptr())||PyFloat_CheckExact(input.ptr())||PyUnicode_CheckExact(input.ptr())||PyBytes_CheckExact(input.ptr()))return py::reinterpret_borrow<py::object>(input);
    auto key=py::int_(reinterpret_cast<uintptr_t>(input.ptr()));
    if(memo_.contains(key))return py::reinterpret_borrow<py::object>(memo_[key]);
    source_owners_.push_back(py::reinterpret_borrow<py::object>(input));
    py::object output;
    if(population_storage_type&&PyObject_TypeCheck(input.ptr(),population_storage_type))output=py::cast(population(py::cast<BatchPtr>(input)));
    else if(PyArray_CheckExact(input.ptr())&&!PyDataType_FLAGCHK(PyArray_DESCR(reinterpret_cast<PyArrayObject *>(input.ptr())),NPY_ITEM_REFCOUNT)) {
      RegionScope timing(Region::Pack);auto array=reinterpret_cast<PyArrayObject *>(input.ptr());
      auto copy=PyArray_NewCopy(array,NPY_KEEPORDER);if(!copy)throw py::error_already_set();
      output=py::reinterpret_steal<py::object>(copy);record_snapshot_copy(PyArray_NBYTES(array));
    }else {RegionScope timing(Region::Callback);output=deepcopy_(input,memo_);}
    memo_[key]=output;return output;
  }
  RowPtr row(const RowPtr &input) {
    if(!input)return nullptr;
    auto found=rows_.find(input.get());if(found!=rows_.end())return found->second;
    input->refresh_record();
    auto output=std::make_shared<BatchRow>(py::cast<py::array>(owner(input->decision)),input->objective,input->constraint,input->fit,owner(input->auxiliary));
    rows_.emplace(input.get(),output);
    if(!input->record.is_none())output->record=owner(input->record);
    return output;
  }
  BatchPtr population(const BatchPtr &input) {
    input->ensure_values();
    auto found=populations_.find(input.get());if(found!=populations_.end())return found->second;
    auto key=population_key(input);if(memo_.contains(key))return py::cast<BatchPtr>(memo_[key]);
    auto output=std::make_shared<BatchStorage>(py::array(input->dtype(),std::vector<py::ssize_t>{input->size(),input->dimension()}),BatchStorage::MetadataInitialization::Overwrite);
    populations_.emplace(input.get(),output);
    memo_[key]=py::cast(output);
    for(py::ssize_t i=0;i<input->size();++i)input->write_row(i,static_cast<char *>(output->x.mutable_data())+i*input->dimension()*8);
    if(input->evaluated)output->allocate_values(false);output->records_available=input->records_available;
    for(py::ssize_t i=0;i<input->size();++i) {
      output->rows[i]=row(input->rows[i]);output->provenance[i]=input->provenance[i];
      output->valid.mutable_data()[i]=input->valid.data()[i];output->ids.mutable_data()[i]=input->ids.data()[i];
      if(input->evaluated){output->f.mutable_data()[i]=input->f.data()[i];output->objectives.mutable_data()[i]=input->objectives.data()[i];output->constraints.mutable_data()[i]=input->constraints.data()[i];}
    }
    record_snapshot_copy(input->size()*(input->dimension()*8+33));
    return output;
  }
  BatchPtr root_population(const BatchPtr &input,const std::vector<RowPtr>&visible) {
    input->ensure_values();
    // A public Solution can be edited after the safe run boundary. The root
    // resumes those visible records, exactly as the legacy checkpoint does.
    // Auxiliary populations retain their own matrix authority when it differs.
    std::vector<RowPtr> cloned;cloned.reserve(visible.size());bool same=visible.size()==size_t(input->size());
    for(size_t i=0;i<visible.size();++i) {
      auto value=row(visible[i]);cloned.push_back(value);
      if(same) same=input->same_row(visible[i]->decision,i)
          &&BatchStorage::same_double(visible[i]->fit,input->f.data()[i])
          &&BatchStorage::same_double(visible[i]->objective,input->objectives.data()[i])
          &&BatchStorage::same_double(visible[i]->constraint,input->constraints.data()[i]);
    }
    auto output=BatchStorage::from_rows(cloned,input->dimension(),input->integer());
    record_snapshot_copy(cloned.size()*(input->dimension()*8+33));
    if(same){populations_.insert_or_assign(input.get(),output);memo_[population_key(input)]=py::cast(output);}
    return output;
  }
  NativeState state(const NativeState &input) {
    auto found=states_.find(input.identity());if(found!=states_.end())return found->second;
    if(input.fields_->dynamic){auto output=NativeState::opaque(py::cast<py::dict>(owner(input.python())));states_.emplace(input.identity(),output);return output;}
    NativeState output;states_.emplace(input.identity(),output);
    if(!input.fields_->origin.is_none()) {
      output.fields_->origin=py::dict();replace(input.fields_->origin,output.fields_->origin);
    }
    output.fields_->order=input.fields_->order;output.fields_->has_stages=input.fields_->has_stages;
    if(input.fields_->opaque_keys)output.fields_->opaque_keys=py::cast<py::dict>(owner(*input.fields_->opaque_keys));
    for(auto &entry:input.fields_->values) {
      auto value=std::visit([&](const auto &item)->StateValue {
        using T=std::decay_t<decltype(item)>;
        if constexpr(std::is_same_v<T,StateArray>)return StateArray{py::cast<py::array>(owner(item.owner))};
        else if constexpr(std::is_same_v<T,NativePopulationOwner>)return population(item);
        else if constexpr(std::is_same_v<T,py::object>)return owner(item);
        else if constexpr(std::is_same_v<T,StateIndices>)return StateIndices{item.values,py::none()};
        else return item;
      },entry.second);
      output.fields_->values.emplace(entry.first,std::move(value));
    }
    for(auto &child:input.fields_->stages)output.fields_->stages.push_back(state(child));
    if(!output.fields_->origin.is_none())output.python();
    return output;
  }
};
struct RunSnapshot {
  BatchPtr population;
  std::vector<NativeState> auxiliary;
  std::vector<ArchiveValue> archives;
  std::vector<RowPtr> history,public_rows;
  std::optional<py::list> boundary_auxiliary,boundary_archives;
  std::vector<double> fits;
  py::object rng=py::none(),rng_state=py::none(),signature=py::none();
  int64_t evaluations=0,generation=1,events=0,zero=0;
  double elapsed=0.;bool complete=false;
  std::string reason="running";
  std::shared_ptr<RunSnapshot> clone(py::object target_rng=py::none())const {
    SnapshotClone copy;
    if(!target_rng.is_none())copy.replace(rng,target_rng);
    auto output=std::make_shared<RunSnapshot>();
    output->rng=target_rng.is_none()?copy.owner(rng):target_rng;
    output->rng_state=copy.owner(rng_state);output->signature=signature;
    output->population=copy.root_population(population,public_rows);
    for(auto &s:auxiliary)output->auxiliary.push_back(copy.state(s));
    for(auto &a:archives)output->archives.emplace_back(copy.owner(a.object()));
    for(auto &r:history)output->history.push_back(copy.row(r));
    for(auto &r:public_rows)output->public_rows.push_back(copy.row(r));
    output->fits=fits;output->evaluations=evaluations;output->generation=generation;output->events=events;
    output->zero=zero;output->elapsed=elapsed;output->complete=complete;output->reason=reason;return output;
  }
  py::dict legacy_live(py::handle solution_type,py::handle set_type) {
    py::list saved_history,fit_history,solutions;
    for(auto &row:history)saved_history.append(row?row->materialize(solution_type):py::none());
    for(auto &row:public_rows)solutions.append(row->materialize(solution_type));
    for(auto fit:fits)fit_history.append(py::float_(fit));
    boundary_auxiliary=export_auxiliary(auxiliary);boundary_archives=export_archives(archives);
    return py::dict("solutions"_a=set_type(solutions),"aux_cache"_a=*boundary_auxiliary,
        "archives"_a=*boundary_archives,"history"_a=saved_history,"fit_history"_a=fit_history,"evaluations"_a=evaluations,
        "elapsed_seconds"_a=elapsed,"generation"_a=generation,"rng"_a=rng,"rng_state"_a=rng_state,
        "complete"_a=complete,"stream_events"_a=events,"termination_reason"_a=reason);
  }
  py::dict legacy(py::handle solution_type,py::handle set_type)const {return clone()->legacy_live(solution_type,set_type);}
};
} // namespace native_components

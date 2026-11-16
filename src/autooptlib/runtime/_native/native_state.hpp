#pragma once
#include <memory>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <optional>
#include <vector>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace native_components {
namespace py=pybind11;

// Numeric fields never require a Python scalar. Arrays retain the allocation
// owner; arbitrary user values retain their exact identity in the owner arm.
struct StateArray {py::array owner;};
struct StateIndices {std::vector<int64_t> values;py::object origin=py::none();};
struct BatchStorage;
using NativePopulationOwner=std::shared_ptr<BatchStorage>;
std::optional<NativePopulationOwner> read_native_population(py::handle input);
py::object native_population_object(const NativePopulationOwner &input);
using StateValue=std::variant<int64_t,double,bool,std::string,StateArray,StateIndices,NativePopulationOwner,py::object>;
struct StateFields;
struct StateImportMemo:std::unordered_map<PyObject *,std::shared_ptr<StateFields>> {std::unordered_set<PyObject *> active;};
using StateExportMemo=std::unordered_map<const StateFields *,py::dict>;
inline thread_local StateImportMemo *state_import_memo=nullptr;
inline thread_local StateExportMemo *state_export_memo=nullptr;
template<class Memo>struct StateMemoScope {
  Memo own;Memo *&active;Memo *previous;
  explicit StateMemoScope(Memo *&value):active(value),previous(value){if(!active)active=&own;}
  ~StateMemoScope(){active=previous;}
};
class NativeState {
  std::shared_ptr<StateFields> fields_;
public:
  NativeState();
  explicit NativeState(py::dict dictionary,bool live=false);
  static NativeState opaque(py::dict dictionary);
  bool contains(const char *key)const;
  py::object object(const char *key,py::object fallback=py::none())const;
  template<class T>T number(const char *key)const;
  void set(const char *key,py::handle value);
  void set(const char *key,int64_t value);
  void set(const char *key,double value);
  void set(const char *key,bool value);
  void set(const char *key,const std::string &value);
  void erase(const char *key);
  void erase_prefix(const char *prefix);
  py::object pop(const char *key);
  NativeState copy()const;
  std::vector<NativeState>&stages();
  void set_stage(size_t index,NativeState state);
  void grow_stages(size_t count);
  void commit_stages();
  py::dict python()const;
  void refresh();
  void record_targets(const int64_t *values,size_t count);
  void set_indices(const char *key,std::vector<int64_t> values);
  std::optional<std::vector<int64_t>> take_indices(const char *key);
  py::object index_array(const char *key)const;
  void set_population(const char *key,NativePopulationOwner value){fields_set_population(key,std::move(value));}
  NativePopulationOwner population(const char *key)const;
private:
  void remember(const char *key);
  void fields_set_population(const char *key,NativePopulationOwner value);
public:
  friend class SnapshotClone;
  const StateFields *identity()const{return fields_.get();}
};
struct StateFields {
  std::unordered_map<std::string,StateValue> values;
  std::vector<std::string> order;
  std::optional<py::dict> opaque_keys;
  std::vector<NativeState> stages;
  bool has_stages=false;
  bool live=false;
  bool dynamic=false,stages_loaded=false;
  std::optional<py::list> stage_list;
  py::object origin=py::none();
};
inline NativeState::NativeState():fields_(std::make_shared<StateFields>()){}
inline NativeState::NativeState(py::dict dictionary,bool live):NativeState(){
  StateMemoScope<StateImportMemo> scope(state_import_memo);
  auto existing=state_import_memo->find(dictionary.ptr());
  if(existing!=state_import_memo->end()){
    if(state_import_memo->active.count(dictionary.ptr())) {
      // A nonstandard cyclic stage graph remains an opaque Python owner.
      // Do not form a shared_ptr cycle in the standard native state graph.
      for(auto owner:state_import_memo->active)state_import_memo->at(owner)->dynamic=true;
      fields_->origin=dictionary;fields_->dynamic=true;return;
    }
    fields_=existing->second;return;
  }
  fields_->origin=dictionary;fields_->live=live;
  state_import_memo->emplace(dictionary.ptr(),fields_);refresh();
}
inline NativeState NativeState::opaque(py::dict dictionary){NativeState out;out.fields_->origin=dictionary;out.fields_->dynamic=true;return out;}
inline bool NativeState::contains(const char *key)const {
  if(fields_->dynamic)return PyDict_GetItemString(fields_->origin.ptr(),key)!=nullptr;
  return std::string(key)=="_stream_stage_aux"?fields_->has_stages:fields_->values.count(key)!=0;
}
inline py::object NativeState::object(const char *key,py::object fallback)const {
  if(fields_->dynamic){auto item=PyDict_GetItemString(fields_->origin.ptr(),key);return item?py::reinterpret_borrow<py::object>(item):fallback;}
  auto found=fields_->values.find(key);if(found==fields_->values.end())return fallback;
  return std::visit([](const auto &value)->py::object {
    using T=std::decay_t<decltype(value)>;
    if constexpr(std::is_same_v<T,py::object>)return value;
    else if constexpr(std::is_same_v<T,StateArray>)return value.owner;
    else if constexpr(std::is_same_v<T,NativePopulationOwner>)return native_population_object(value);
    else if constexpr(std::is_same_v<T,StateIndices>) {
      py::list out=value.origin.is_none()?py::list():py::cast<py::list>(value.origin);
      if(value.origin.is_none())for(auto index:value.values)out.append(py::int_(index));return out;
    }
    else return py::cast(value);
  },found->second);
}
template<class T>inline T NativeState::number(const char *key)const {
  if(fields_->dynamic){if(!contains(key))throw py::key_error(key);return py::cast<T>(object(key));}
  auto found=fields_->values.find(key);if(found==fields_->values.end())throw py::key_error(key);
  return std::visit([&](const auto &value)->T {
    using V=std::decay_t<decltype(value)>;
    if constexpr(std::is_arithmetic_v<V>) {
      // Only the standard stored type uses a direct native conversion.
      // Python rejects float-to-index casts and checks integer overflow.
      if constexpr(std::is_integral_v<T>&&!std::is_same_v<T,V>)
        return py::cast<T>(py::cast(value));
      else return static_cast<T>(value);
    }
    else if constexpr(std::is_same_v<V,py::object>)return py::cast<T>(value);
    else return py::cast<T>(object(key));
  },found->second);
}
inline void NativeState::remember(const char *key){if(!fields_->values.count(key))fields_->order.emplace_back(key);}
inline void NativeState::set(const char *key,py::handle value) {
  if(fields_->dynamic){py::cast<py::dict>(fields_->origin)[py::str(key)]=value;return;}
  if(std::string(key)=="_stream_pso_target_indices"&&PyList_CheckExact(value.ptr())) {
    remember(key);StateIndices targets;for(auto item:py::reinterpret_borrow<py::list>(value))targets.values.push_back(py::cast<int64_t>(item));
    if(fields_->live)targets.origin=py::reinterpret_borrow<py::object>(value);
    fields_->values.insert_or_assign(key,std::move(targets));
    if(fields_->live)py::cast<py::dict>(fields_->origin)[py::str(key)]=value;
    return;
  }
  if(PyLong_CheckExact(value.ptr())) {
    int overflow=0;auto number=PyLong_AsLongLongAndOverflow(value.ptr(),&overflow);
    if(!overflow&&!PyErr_Occurred()){set(key,int64_t(number));return;}
    if(PyErr_Occurred())throw py::error_already_set();
  }
  if(PyFloat_CheckExact(value.ptr())){set(key,PyFloat_AS_DOUBLE(value.ptr()));return;}
  if(PyBool_Check(value.ptr())){set(key,value.ptr()==Py_True);return;}
  if(PyUnicode_CheckExact(value.ptr())){set(key,py::cast<std::string>(value));return;}
  remember(key);
  if(PyArray_CheckExact(value.ptr()))fields_->values.insert_or_assign(key,StateArray{py::reinterpret_borrow<py::array>(value)});
  else if(auto native=read_native_population(value))fields_->values.insert_or_assign(key,*native);
  else fields_->values.insert_or_assign(key,py::reinterpret_borrow<py::object>(value));
  if(fields_->live)py::cast<py::dict>(fields_->origin)[py::str(key)]=value;
}
inline void NativeState::set(const char *key,int64_t value){if(!fields_->dynamic){remember(key);fields_->values.insert_or_assign(key,value);}if(fields_->live||fields_->dynamic)py::cast<py::dict>(fields_->origin)[py::str(key)]=py::int_(value);}
inline void NativeState::set(const char *key,double value){if(!fields_->dynamic){remember(key);fields_->values.insert_or_assign(key,value);}if(fields_->live||fields_->dynamic)py::cast<py::dict>(fields_->origin)[py::str(key)]=py::float_(value);}
inline void NativeState::set(const char *key,bool value){if(!fields_->dynamic){remember(key);fields_->values.insert_or_assign(key,value);}if(fields_->live||fields_->dynamic)py::cast<py::dict>(fields_->origin)[py::str(key)]=py::bool_(value);}
inline void NativeState::set(const char *key,const std::string &value){if(!fields_->dynamic){remember(key);fields_->values.insert_or_assign(key,value);}if(fields_->live||fields_->dynamic)py::cast<py::dict>(fields_->origin)[py::str(key)]=py::str(value);}
inline void NativeState::fields_set_population(const char *key,NativePopulationOwner value){
  if(!fields_->dynamic){remember(key);fields_->values.insert_or_assign(key,value);}if(fields_->live||fields_->dynamic)py::cast<py::dict>(fields_->origin)[py::str(key)]=native_population_object(value);
}
inline NativePopulationOwner NativeState::population(const char *key)const {
  if(fields_->dynamic){auto value=read_native_population(object(key));return value?*value:nullptr;}
  auto found=fields_->values.find(key);if(found==fields_->values.end())return nullptr;
  auto value=std::get_if<NativePopulationOwner>(&found->second);return value?*value:nullptr;
}
inline void NativeState::erase(const char *key){
  const bool stage=std::string(key)=="_stream_stage_aux";
  if(fields_->values.erase(key)||(stage&&fields_->has_stages))
    fields_->order.erase(std::remove(fields_->order.begin(),fields_->order.end(),key),fields_->order.end());
  if(stage){fields_->stages.clear();fields_->has_stages=false;}
  if(fields_->opaque_keys&&PyDict_DelItemString(fields_->opaque_keys->ptr(),key)<0)PyErr_Clear();
  if((fields_->live||fields_->dynamic)&&PyDict_DelItemString(fields_->origin.ptr(),key)<0)PyErr_Clear();
}
inline void NativeState::erase_prefix(const char *prefix){
  if(fields_->dynamic){py::list keys;for(auto item:py::cast<py::dict>(fields_->origin))if(py::cast<std::string>(py::str(item.first)).rfind(prefix,0)==0)keys.append(item.first);
    for(auto key:keys)if(PyDict_DelItem(fields_->origin.ptr(),key.ptr())<0)throw py::error_already_set();return;}
  fields_->order.erase(std::remove_if(fields_->order.begin(),fields_->order.end(),[&](const std::string &key){return key!="_stream_stage_aux"&&key.rfind(prefix,0)==0;}),fields_->order.end());
  for(auto item=fields_->values.begin();item!=fields_->values.end();)
    if(item->first.rfind(prefix,0)==0){if(fields_->live&&PyDict_DelItemString(fields_->origin.ptr(),item->first.c_str())<0)PyErr_Clear();item=fields_->values.erase(item);}else ++item;
}
inline py::object NativeState::pop(const char *key){
  if(fields_->dynamic){auto value=object(key);erase(key);return value;}
  auto found=fields_->values.find(key);
  if(found!=fields_->values.end())if(auto indices=std::get_if<StateIndices>(&found->second)) {
    py::array_t<int64_t> out(indices->values.size());std::copy(indices->values.begin(),indices->values.end(),out.mutable_data());erase(key);return out;
  }
  auto value=object(key);erase(key);return value;
}
inline void NativeState::record_targets(const int64_t *values,size_t count) {
  if(fields_->dynamic){if(!contains("_stream_pso_target_indices"))set("_stream_pso_target_indices",py::list());
    auto list=py::cast<py::list>(object("_stream_pso_target_indices"));for(size_t i=0;i<count;++i)list.append(py::int_(values[i]));return;}
  auto found=fields_->values.find("_stream_pso_target_indices");
  if(found==fields_->values.end()){remember("_stream_pso_target_indices");fields_->values.emplace("_stream_pso_target_indices",StateIndices{});found=fields_->values.find("_stream_pso_target_indices");}
  auto indices=std::get_if<StateIndices>(&found->second);
  if(!indices)throw py::type_error("PSO target state must be a list of indices");
  indices->values.insert(indices->values.end(),values,values+count);
  if(fields_->live) {
    if(indices->origin.is_none()){indices->origin=py::list();py::cast<py::dict>(fields_->origin)["_stream_pso_target_indices"]=indices->origin;}
    for(size_t i=0;i<count;++i)py::cast<py::list>(indices->origin).append(py::int_(values[i]));
  }
}
inline void NativeState::set_indices(const char *key,std::vector<int64_t> values){
  if(fields_->dynamic){py::list out;for(auto value:values)out.append(py::int_(value));set(key,out);return;}
  remember(key);fields_->values.insert_or_assign(key,StateIndices{std::move(values),py::none()});}
inline std::optional<std::vector<int64_t>> NativeState::take_indices(const char *key) {
  if(fields_->dynamic){if(!contains(key))return std::nullopt;
    auto array=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(object(key));
    if(!array)throw py::type_error("PSO target state must contain integer indices");
    std::vector<int64_t> out(array.data(),array.data()+array.size());erase(key);return out;}
  auto found=fields_->values.find(key);if(found==fields_->values.end())return std::nullopt;
  if(auto value=std::get_if<StateIndices>(&found->second)){auto out=std::move(value->values);erase(key);return out;}
  auto array=py::array_t<int64_t,py::array::c_style|py::array::forcecast>::ensure(object(key));
  if(!array)throw py::type_error("PSO target state must contain integer indices");
  std::vector<int64_t> out(array.data(),array.data()+array.size());erase(key);return out;
}
inline py::object NativeState::index_array(const char *key)const {
  if(fields_->dynamic)return object(key);
  auto found=fields_->values.find(key);
  if(found!=fields_->values.end())if(auto indices=std::get_if<StateIndices>(&found->second)) {
    py::array_t<int64_t> out(indices->values.size());std::copy(indices->values.begin(),indices->values.end(),out.mutable_data());return out;
  }
  return object(key);
}
inline NativeState NativeState::copy()const {
  if(fields_->dynamic)return opaque(py::reinterpret_steal<py::dict>(PyDict_Copy(fields_->origin.ptr())));
  NativeState out;out.fields_->order=fields_->order;out.fields_->values=fields_->values;out.fields_->stages=fields_->stages;
  if(fields_->opaque_keys)out.fields_->opaque_keys=py::reinterpret_steal<py::dict>(PyDict_Copy(fields_->opaque_keys->ptr()));
  out.fields_->has_stages=fields_->has_stages;return out;
}
inline std::vector<NativeState>&NativeState::stages(){
  if(fields_->dynamic&&!fields_->stages_loaded) {
    if(!contains("_stream_stage_aux"))set("_stream_stage_aux",py::list());
    fields_->stage_list=py::cast<py::list>(object("_stream_stage_aux"));fields_->stages.clear();
    for(auto item:*fields_->stage_list)fields_->stages.push_back(opaque(py::cast<py::dict>(item)));
    fields_->stages_loaded=true;
  }
  if(!fields_->has_stages)fields_->order.emplace_back("_stream_stage_aux");
  fields_->has_stages=true;return fields_->stages;
}
inline void NativeState::set_stage(size_t index,NativeState state) {
  fields_->stages.at(index)=state;
  if(fields_->dynamic&&fields_->stage_list)(*fields_->stage_list)[index]=state.python();
}
inline void NativeState::grow_stages(size_t count) {
  auto &values=stages();while(values.size()<count) {
    if(fields_->dynamic){auto state=opaque(py::dict());fields_->stage_list->append(state.python());values.push_back(std::move(state));}
    else values.emplace_back();
  }
}
inline void NativeState::commit_stages(){if(fields_->dynamic&&fields_->stage_list)py::cast<py::dict>(fields_->origin)["_stream_stage_aux"]=*fields_->stage_list;}
inline py::dict NativeState::python()const {
  if(fields_->dynamic){auto out=py::cast<py::dict>(fields_->origin);if(fields_->stage_list)out["_stream_stage_aux"]=*fields_->stage_list;return out;}
  StateMemoScope<StateExportMemo> scope(state_export_memo);
  auto existing=state_export_memo->find(fields_.get());if(existing!=state_export_memo->end())return existing->second;
  py::dict out=fields_->origin.is_none()?py::dict():py::cast<py::dict>(fields_->origin);
  state_export_memo->emplace(fields_.get(),out);
  PyDict_Clear(out.ptr());
  if(fields_->opaque_keys)for(auto item:*fields_->opaque_keys)out[item.first]=item.second;
  for(auto &key:fields_->order) {
    if(key=="_stream_stage_aux"&&fields_->has_stages){py::list stages;for(auto &state:fields_->stages)stages.append(state.python());out["_stream_stage_aux"]=stages;}
    else if(fields_->values.count(key))out[py::str(key)]=object(key.c_str());
  }
  fields_->origin=out;return out;
}
inline void NativeState::refresh() {
  if(fields_->origin.is_none())return;
  if(fields_->dynamic){fields_->stages.clear();fields_->stage_list.reset();fields_->stages_loaded=false;return;}
  StateMemoScope<StateImportMemo> scope(state_import_memo);
  state_import_memo->insert_or_assign(fields_->origin.ptr(),fields_);
  bool inserted=state_import_memo->active.insert(fields_->origin.ptr()).second;
  struct Finish {StateImportMemo *memo;PyObject *key;bool inserted;~Finish(){if(inserted)memo->active.erase(key);}} finish{state_import_memo,fields_->origin.ptr(),inserted};
  auto dictionary=py::cast<py::dict>(fields_->origin);
  // Nonstandard keys retain Python dictionary identity, ordering and lookup.
  for(auto item:dictionary)if(!PyUnicode_CheckExact(item.first.ptr())){fields_->dynamic=true;return;}
  fields_->order.clear();fields_->values.clear();fields_->opaque_keys.reset();fields_->stages.clear();fields_->has_stages=false;
  for(auto item:dictionary) {
    if(!PyUnicode_CheckExact(item.first.ptr())) {
      if(!fields_->opaque_keys)fields_->opaque_keys.emplace();
      (*fields_->opaque_keys)[item.first]=item.second;
      if(!PyUnicode_Check(item.first.ptr()))continue;
    }
    auto key=py::cast<std::string>(item.first);
    if(key=="_stream_stage_aux") {
      fields_->order.push_back(key);fields_->has_stages=true;
      for(auto stage:py::cast<py::list>(item.second))fields_->stages.emplace_back(py::cast<py::dict>(stage));
      if(fields_->dynamic){fields_->stage_list=py::cast<py::list>(item.second);fields_->stages_loaded=true;}
    }else {
      bool live=fields_->live;fields_->live=false;
      try{set(key.c_str(),item.second);}catch(...){fields_->live=live;throw;}
      fields_->live=live;
      // A compatibility call appends into the caller's existing target list.
      // Import must retain that owner as well as its numeric contents.
      if(live) {
        auto found=fields_->values.find(key);
        if(found!=fields_->values.end())if(auto indices=std::get_if<StateIndices>(&found->second))
          indices->origin=py::reinterpret_borrow<py::object>(item.second);
      }
    }
  }
}

// Shared mathematical routines return this native record. Only the exported
// compatibility entry creates a tuple/dict; the search core reads fields.
struct NativeResult {
  py::object value=py::none(),auxiliary=py::none();
  std::optional<NativeState> state;
  bool paired=false,state_only=false;
  bool binary_output=false; // Private result fact, never exported as user state.
  NativeResult()=default;
  NativeResult(py::object object):value(std::move(object)){}
  NativeResult(NativeState fields):state(std::move(fields)),state_only(true){}
  py::object python()const {
    auto second=state?py::object(state->python()):auxiliary;
    if(state_only)return second;
    return paired?py::object(py::make_tuple(value,second)):value;
  }
};
inline NativeResult state_result(py::object value,NativeState state) {
  NativeResult out(std::move(value));out.state=std::move(state);out.paired=true;return out;
}
inline NativeResult state_result(py::object value,py::object auxiliary) {
  NativeResult out(std::move(value));out.auxiliary=std::move(auxiliary);out.paired=true;return out;
}
inline std::vector<NativeState> import_auxiliary(py::list source) {
  StateMemoScope<StateImportMemo> scope(state_import_memo);
  std::vector<NativeState> out;out.reserve(source.size());
  for(auto item:source)out.emplace_back(PyDict_Check(item.ptr())?py::cast<py::dict>(item):py::dict());return out;
}
inline py::list export_auxiliary(const std::vector<NativeState> &source) {
  StateMemoScope<StateExportMemo> scope(state_export_memo);
  py::list out;for(auto &state:source)out.append(state.python());return out;
}
} // namespace native_components

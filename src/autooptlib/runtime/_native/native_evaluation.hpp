#pragma once
#include "python_compat.hpp"
#include <string_view>
namespace native_components {
// IOH supplies the actual C++ objective. Use its public bound C entry without
// duplicating benchmark formulas, objective counters, transformations or logs.
// Non-default policy, uncertain problems and replaced Python hooks retain the
// authoritative policy callback before any objective has been invoked.
struct NativeEvaluation {
  py::object solve,numpy,builtins,namespace_type,ioh_module=py::none(),ioh_data_type=py::none(),ioh_code=py::none();
  struct Check {py::object key,expected,code;};
  std::vector<Check> functions,normalization;
  py::object policy_source,normalization_source;
  std::unordered_map<std::string_view,py::str> keys;
  py::object attribute(py::handle owner,const char *name)const {
    // __dict__ is a data descriptor of the exact built-in module type.
    // Other names keep their one GenericGetAttr lookup, including missing-name
    // hooks and equality effects of non-string module dictionary keys.
    auto raw=PyObject_GetAttr(owner.ptr(),keys.at(name).ptr());
    if(!raw)throw py::error_already_set();return py::reinterpret_steal<py::object>(raw);
  }
  py::object module_dictionary(py::handle owner)const {
    if(PyModule_CheckExact(owner.ptr()))return py::reinterpret_borrow<py::object>(PyModule_GetDict(owner.ptr()));
    return attribute(owner,"__dict__");
  }
  py::object read(py::handle dictionary,const char *name,py::object fallback=py::none())const {
    auto value=PyDict_GetItem(dictionary.ptr(),keys.at(name).ptr());
    return value?py::reinterpret_borrow<py::object>(value):fallback;
  }
  mutable py::object target=py::none(),target_method=py::none();
  mutable unsigned target_version=0;
  explicit NativeEvaluation(py::object module):solve(std::move(module)),numpy(py::module_::import("numpy")),
      builtins(py::module_::import("builtins")),namespace_type(py::module_::import("types").attr("SimpleNamespace")) {
    for(auto key:{"_NATIVE_EVALUATOR_POLICY","np","_NORMALIZE_MODULE","_NORMALIZE_FLOAT",
                  "__dict__","float","_NORMALIZE_NUMPY","_native_execution","evaluate","type",
                  "_autoopt_eval_timeout","_autoopt_eval_log","_autoopt_eval_cache","_autoopt_eval_retries",
                  "_autoopt_eval_failure","dtype","objective","raise"})keys.emplace(key,py::str(key));
    policy_source=attribute(solve,"_NATIVE_EVALUATOR_POLICY");
    for(auto item:policy_source){auto t=py::cast<py::tuple>(item);functions.push_back({py::reinterpret_borrow<py::object>(t[0]),py::reinterpret_borrow<py::object>(t[1]),py::reinterpret_borrow<py::object>(t[2])});}
    normalization_source=attribute(solve,"_NORMALIZE_NUMPY");
    for(auto item:normalization_source){auto t=py::cast<py::tuple>(item);normalization.push_back({py::reinterpret_borrow<py::object>(t[0]),py::reinterpret_borrow<py::object>(t[1]),py::none()});}
    auto imported=PyDict_GetItemString(PyImport_GetModuleDict(),"autooptlib.problems.ioh");
    if(!imported)return;
    ioh_module=py::reinterpret_borrow<py::object>(imported);ioh_data_type=ioh_module.attr("_ResidentIOHData");
    auto factory=ioh_module.attr("make_ioh_problem");
    if(!PyFunction_Check(factory.ptr()))return;
    std::function<void(py::object)> find=[&](py::object code){
      for(auto value:py::cast<py::tuple>(code.attr("co_consts")))if(PyCode_Check(value.ptr())) {
        auto child=py::reinterpret_borrow<py::object>(value);
        if(py::cast<std::string>(child.attr("co_name"))=="evaluate")ioh_code=child;
        else find(child);
      }
    };
    find(py::reinterpret_borrow<py::object>(PyFunction_GetCode(factory.ptr())));
  }
  bool policy_unchanged()const {
    if(!PyModule_CheckExact(solve.ptr())||!PyModule_CheckExact(numpy.ptr())||(!ioh_module.is_none()&&!PyModule_CheckExact(ioh_module.ptr())))return false;
    if(attribute(solve,"_NATIVE_EVALUATOR_POLICY").ptr()!=policy_source.ptr())return false;
    for(const auto &check:functions) {
      py::object missing;auto value=PyDict_GetItem(PyModule_GetDict(solve.ptr()),check.key.ptr());
      if(!value){auto raw=PyObject_GetAttr(solve.ptr(),check.key.ptr());if(!raw)throw py::error_already_set();missing=py::reinterpret_steal<py::object>(raw);value=missing.ptr();}
      if(value!=check.expected.ptr()||!PyFunction_Check(value)||PyFunction_GetCode(value)!=check.code.ptr())return false;
    }
    if(attribute(solve,"np").ptr()!=numpy.ptr()||attribute(solve,"_NORMALIZE_MODULE").ptr()!=numpy.ptr())return false;
    if(!ioh_module.is_none()&&attribute(ioh_module,"np").ptr()!=numpy.ptr())return false;
    if(attribute(solve,"_NORMALIZE_FLOAT").ptr()!=reinterpret_cast<PyObject *>(&PyFloat_Type))return false;
    auto float_type=reinterpret_cast<PyObject *>(&PyFloat_Type);
    auto local_float=PyDict_GetItemString(module_dictionary(solve).ptr(),"float");
    if(local_float&&local_float!=float_type)return false;
    if(attribute(builtins,"float").ptr()!=float_type)return false;
    if(!ioh_module.is_none()){auto local=PyDict_GetItemString(module_dictionary(ioh_module).ptr(),"float");if(local&&local!=float_type)return false;}
    auto dictionary=PyModule_GetDict(numpy.ptr());auto current=attribute(solve,"_NORMALIZE_NUMPY");
    if(current.ptr()==normalization_source.ptr()) {
      for(const auto &check:normalization)if(PyDict_GetItem(dictionary,check.key.ptr())!=check.expected.ptr())return false;
    }else for(auto item:current) {
      auto pair=py::cast<py::tuple>(item);auto value=PyDict_GetItem(dictionary,pair[0].ptr());
      if(!value||value!=pair[1].ptr())return false;
    }
    return true;
  }
  std::optional<NativeData::EvaluatedRow> evaluate(py::object problem,py::object data,py::array decision)const {
    if(ioh_code.is_none()||Py_TYPE(problem.ptr())!=reinterpret_cast<PyTypeObject *>(namespace_type.ptr())
        ||Py_TYPE(data.ptr())!=reinterpret_cast<PyTypeObject *>(ioh_data_type.ptr())||!policy_unchanged())return std::nullopt;
    if(Py_TYPE(data.ptr())->tp_getattro!=PyObject_GenericGetAttr)return std::nullopt;
    for(auto key:{"objective","dtype","minimize"}) {
      auto descriptor=PyDict_GetItemString(aol_type_dict(Py_TYPE(data.ptr())),key);
      if(descriptor&&PyDescr_IsData(descriptor))return std::nullopt;
    }
    // These exact standard objects have already passed the descriptor checks.
    // Keep only dictionary handles within this call, never cached field values.
    auto problem_fields=py::reinterpret_steal<py::object>(PyObject_GenericGetDict(problem.ptr(),nullptr));
    if(!problem_fields){PyErr_Clear();return std::nullopt;}
    auto evaluate=read(problem_fields,"evaluate");
    if(!PyFunction_Check(evaluate.ptr())||PyFunction_GetCode(evaluate.ptr())!=ioh_code.ptr()
        ||PyFunction_GetGlobals(evaluate.ptr())!=module_dictionary(ioh_module).ptr()
        ||PyFunction_GetClosure(evaluate.ptr()))return std::nullopt;
    auto behavior=read(problem_fields,"type");
    if(!PyList_CheckExact(behavior.ptr())||py::len(behavior)<3)return std::nullopt;
    auto certain=py::cast<py::list>(behavior)[2];
    if(!PyUnicode_CheckExact(certain.ptr())||py::cast<std::string>(certain)!="certain")return std::nullopt;
    for(auto key:{"_autoopt_eval_timeout","_autoopt_eval_log","_autoopt_eval_cache"})if(!read(problem_fields,key).is_none())return std::nullopt;
    auto retries=read(problem_fields,"_autoopt_eval_retries");
    if(!retries.is_none()&&(!PyLong_CheckExact(retries.ptr())||py::cast<int64_t>(retries)!=0))return std::nullopt;
    auto failure=read(problem_fields,"_autoopt_eval_failure",keys.at("raise"));
    if(!PyUnicode_CheckExact(failure.ptr())||py::cast<std::string>(failure)!="raise")return std::nullopt;
    auto data_fields=py::reinterpret_steal<py::object>(PyObject_GenericGetDict(data.ptr(),nullptr));
    if(!data_fields){PyErr_Clear();return std::nullopt;}
    auto dtype=read(data_fields,"dtype");
    bool continuous=dtype.ptr()==reinterpret_cast<PyObject *>(&PyFloat_Type);
    bool discrete=dtype.ptr()==reinterpret_cast<PyObject *>(&PyLong_Type);
    if((!continuous&&!discrete)||decision.dtype().num()!=(continuous?NPY_DOUBLE:NPY_INT64))return std::nullopt;
    auto objective=read(data_fields,"objective");
    // Release dictionary owners before any target/descriptor callback. Replacing
    // __dict__ inside that callback must retain the original destruction point.
    problem_fields=py::object();data_fields=py::object();
    auto type=Py_TYPE(objective.ptr());
    // Cache only a bound method of an unchanged type with no instance dict.
    // Every objective call still checks policy, data and the current owner.
    if(!target.is(objective)||!target_version||target_version!=type->tp_version_tag||type->tp_dictoffset||type->tp_getattro!=PyObject_GenericGetAttr) {
      auto method=py::getattr(objective,"__call__",py::none());
      if(!PyMethod_Check(method.ptr())||!PyCFunction_Check(PyMethod_GET_FUNCTION(method.ptr())))return std::nullopt;
      auto module=py::cast<std::string>(py::getattr(py::type::of(objective),"__module__",py::str("")));
      if(module.rfind("ioh.iohcpp.problem",0)!=0)return std::nullopt;
      target=objective;target_method=std::move(method);target_version=type->tp_version_tag;
    }
    auto &method=target_method;
    auto execution=attribute(solve,"_native_execution");py::object token=py::none();
    if(PyContextVar_CheckExact(execution.ptr())) {
      PyObject *raw=nullptr;
      if(PyContextVar_Get(execution.ptr(),nullptr,&raw)<0)throw py::error_already_set();
      if(!raw){PyErr_SetObject(PyExc_LookupError,execution.ptr());throw py::error_already_set();}
      bool enabled;
      {auto current=py::reinterpret_steal<py::object>(raw);enabled=py::cast<bool>(current);}
      if(enabled) {
        auto changed=PyContextVar_Set(execution.ptr(),Py_False);if(!changed)throw py::error_already_set();
        token=py::reinterpret_steal<py::object>(changed);
      }
    }else if(py::cast<bool>(execution.attr("get")()))token=execution.attr("set")(false);
    struct Reset {
      py::object context,token;
      ~Reset(){if(token.is_none())return;PyObject *type,*value,*trace;PyErr_Fetch(&type,&value,&trace);
        if(PyContextVar_CheckExact(context.ptr())) {
          if(PyContextVar_Reset(context.ptr(),token.ptr())<0)PyErr_WriteUnraisable(context.ptr());
        }else {
          auto result=PyObject_CallMethod(context.ptr(),"reset","O",token.ptr());
          if(result)Py_DECREF(result);else PyErr_WriteUnraisable(context.ptr());
        }
        PyErr_Restore(type,value,trace);}
    };
    double value;
    try {
      Reset reset{execution,token};
      RegionScope timing(Region::Objective);
      // IOH's public binding consumes a vector. Iterating an ndarray first
      // boxes every coordinate as a NumPy scalar; its native-format buffer
      // view supplies the same values as builtin scalars without that boxing
      // or an intermediate payload copy. Keep the original array owner alive.
      auto view=py::reinterpret_steal<py::object>(PyMemoryView_FromObject(decision.ptr()));
      if(!view)throw py::error_already_set();
      PyObject *arguments[]={view.ptr()};auto raw=PyObject_Vectorcall(method.ptr(),arguments,1,nullptr);
      if(!raw)throw py::error_already_set();auto result=py::reinterpret_steal<py::object>(raw);
      value=py::cast<double>(result);
      int minimize=PyObject_IsTrue(field(data,"minimize").ptr());if(minimize<0)throw py::error_already_set();
      if(!minimize)value=-value;
    }catch(py::error_already_set &error){solve.attr("_native_evaluator_error")(error.value(),decision);throw;}
    if(!std::isfinite(value)) {
      auto preview=numpy.attr("asarray")(decision).attr("reshape")(-1)[py::slice(0,8,1)].attr("tolist")();
      solve.attr("_normalize_evaluator_output")(py::make_tuple(value,0.,py::none()),preview);
    }
    return NativeData::EvaluatedRow{decision,value,0.,py::none()};
  }
};
} // namespace native_components

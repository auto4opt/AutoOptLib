#pragma once
#include "batch_access.hpp"
namespace native_components {
// Fast admission for the fully specified standard configuration schema.
// Unsupported or invalid values return None so the original validator retains
// its conversion protocols, dynamic reads and exact errors.
struct NativeConfiguration {
  py::object population,namespace_type,absent;
  py::tuple bindings;
  std::unique_ptr<BatchAccess> access;
  std::vector<int64_t> default_population,default_offspring;
  explicit NativeConfiguration(py::object numpy)
      :population(py::module_::import("autooptlib.utils.design._population")),
       namespace_type(py::module_::import("types").attr("SimpleNamespace")),
       absent(population.attr("_NATIVE_CONFIG_ABSENT")),bindings(population.attr("_NATIVE_CONFIGURATION_BINDINGS")) {
    if(!unchanged())return;
    access=std::make_unique<BatchAccess>(numpy);
    auto variants=py::module_::import("autooptlib.utils._attributes").attr("name_variants");
    for(auto key:{"ProbN","prob_n","ProbFE","prob_fe","PopulationSizeSpace","population_size_space",
                  "OffspringSizeSpace","offspring_size_space","CrossoverRate","crossover_rate",
                  "MutationRate","mutation_rate","BoundaryHandling","boundary_handling","_autoopt_problem_type"})
      for(auto value:variants(key))access->spellings[key].push_back(py::reinterpret_borrow<py::str>(value));
    default_population=py::cast<std::vector<int64_t>>(population.attr("DEFAULT_POPULATION_SIZE_SPACE"));
    default_offspring=py::cast<std::vector<int64_t>>(population.attr("DEFAULT_OFFSPRING_SIZE_SPACE"));
  }
  bool unchanged()const {
    for(auto record:bindings) {
      auto entry=py::cast<py::tuple>(record);auto value=PyDict_GetItem(entry[0].ptr(),entry[1].ptr());
      if(entry[2].ptr()==absent.ptr()){if(value)return false;continue;}
      if(value!=entry[2].ptr())return false;
      if(PyFunction_Check(value)) {
        if(PyFunction_GetCode(value)!=entry[3].ptr())return false;
        auto defaults=PyFunction_GetDefaults(value);
        if(defaults!=(entry[4].is_none()?nullptr:entry[4].ptr()))return false;
        auto keywords=PyFunction_GetKwDefaults(value);
        if(entry[5].is_none()){if(keywords)return false;}
        else {
          auto pairs=py::cast<py::tuple>(entry[5]);if(!keywords||PyDict_Size(keywords)!=py::ssize_t(pairs.size()))return false;
          PyObject *key,*item;Py_ssize_t at=0;
          while(PyDict_Next(keywords,&at,&key,&item))if(!PyUnicode_CheckExact(key))return false;
          for(auto pair:pairs){auto p=py::cast<py::tuple>(pair);if(PyDict_GetItem(keywords,p[0].ptr())!=p[1].ptr())return false;}
        }
      }
    }
    return true;
  }
  static bool plain_keys(py::dict source) {
    for(auto entry:source)if(!PyUnicode_CheckExact(entry.first.ptr())||!PyUnicode_IS_ASCII(entry.first.ptr()))return false;
    return true;
  }
  static bool integer(py::handle input,int64_t &value) {
    if(PyLong_CheckExact(input.ptr())){int overflow=0;value=PyLong_AsLongLongAndOverflow(input.ptr(),&overflow);if(PyErr_Occurred())throw py::error_already_set();return overflow==0;}
    if(PyFloat_CheckExact(input.ptr())){auto number=PyFloat_AS_DOUBLE(input.ptr());
      if(!std::isfinite(number)||number<=0||number>=9223372036854775808.||std::floor(number)!=number)return false;
      value=static_cast<int64_t>(number);return true;}
    return false;
  }
  static bool real(py::handle input,double &value) {
    if(PyFloat_CheckExact(input.ptr())){value=PyFloat_AS_DOUBLE(input.ptr());return std::isfinite(value);}
    int64_t number;if(PyLong_CheckExact(input.ptr())&&integer(input,number)){value=number;return true;}return false;
  }
  static bool domain(py::handle input,const std::vector<int64_t>&defaults,int64_t requested,std::optional<int64_t> budget) {
    if(!PyList_CheckExact(input.ptr())&&!PyTuple_CheckExact(input.ptr()))return false;
    auto items=py::reinterpret_borrow<py::sequence>(input);if(!items.size())return false;
    std::vector<int64_t> values;values.reserve(items.size());
    for(auto item:items){int64_t value;if(!integer(item,value)||value<=0)return false;values.push_back(value);}
    std::sort(values.begin(),values.end());values.erase(std::unique(values.begin(),values.end()),values.end());
    if(!budget)return true;
    if(values.front()<=*budget)return true;
    return values==defaults&&requested>0&&requested<=*budget;
  }
  py::object prepare(py::dict source,py::object setting)const {
    if(!access||!unchanged()||Py_TYPE(setting.ptr())!=reinterpret_cast<PyTypeObject *>(namespace_type.ptr())||!PyDict_CheckExact(source.ptr())||!plain_keys(source))return py::none();
    auto dictionary=py::cast<py::dict>(py::getattr(setting,"__dict__"));if(!plain_keys(dictionary))return py::none();
    for(auto key:{"population_size","offspring_size","crossover_rate","mutation_rate","boundary_handling"})if(!source.contains(key))return py::none();
    int64_t n,offspring;if(!integer(source["population_size"],n)||!integer(source["offspring_size"],offspring)||n<=0||offspring<=0)return py::none();
    double crossover,mutation;if(!real(source["crossover_rate"],crossover)||!real(source["mutation_rate"],mutation)||crossover<0||crossover>1||mutation<0||mutation>1)return py::none();
    auto read=[&](const char *key,py::object fallback){return access->flex(setting,key,std::move(fallback));};
    auto paired=[&](const char *primary,const char *alias,py::object fallback){return read(primary,read(alias,std::move(fallback)));};
    int64_t requested;if(!integer(paired("ProbN","prob_n",py::int_(20)),requested)||requested<=0)return py::none();
    std::optional<int64_t> budget;auto raw_budget=paired("ProbFE","prob_fe",py::none());
    if(!raw_budget.is_none()){int64_t value;if(!integer(raw_budget,value)||value<=0)return py::none();budget=value;}
    if(!domain(paired("PopulationSizeSpace","population_size_space",population.attr("DEFAULT_POPULATION_SIZE_SPACE")),default_population,requested,budget)
        ||!domain(paired("OffspringSizeSpace","offspring_size_space",population.attr("DEFAULT_OFFSPRING_SIZE_SPACE")),default_offspring,requested,budget))return py::none();
    double ignored;if(!real(paired("CrossoverRate","crossover_rate",py::float_(1.)),ignored)||!real(paired("MutationRate","mutation_rate",py::float_(1.)),ignored))return py::none();
    auto kind=read("_autoopt_problem_type",py::str(""));if(!PyUnicode_CheckExact(kind.ptr())||!PyUnicode_IS_ASCII(kind.ptr()))return py::none();
    auto type=py::cast<std::string>(kind);for(auto &c:type)c=std::tolower(static_cast<unsigned char>(c));
    auto raw_boundary=source["boundary_handling"];if(!PyUnicode_CheckExact(raw_boundary.ptr())||!PyUnicode_IS_ASCII(raw_boundary.ptr()))return py::none();
    auto boundary=py::cast<std::string>(raw_boundary);for(auto &c:boundary)c=std::tolower(static_cast<unsigned char>(c));
    if(type=="discrete"||type=="permutation")boundary="clip";
    else {auto fallback=paired("BoundaryHandling","boundary_handling",py::str("clip"));if(!PyUnicode_CheckExact(fallback.ptr()))return py::none();}
    if(boundary!="clip"&&boundary!="reflect"&&boundary!="resample")return py::none();
    return py::dict("population_size"_a=n,"offspring_size"_a=offspring,"crossover_rate"_a=crossover,"mutation_rate"_a=mutation,"boundary_handling"_a=boundary);
  }
};
}

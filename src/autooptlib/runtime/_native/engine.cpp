// Native stream interpreter with reference-compatible population kernels.
// RNG and objective policy callbacks retain their Python implementations.
#include "data.hpp"
#include "operators.hpp"
#include "updates.hpp"
#include "blocks.hpp"
#include "program.hpp"
#include <algorithm>
#include <chrono>
#include <memory>
#include <pybind11/pybind11.h>
#include <string>
#include <vector>
namespace py = pybind11;
using namespace pybind11::literals;

static bool prefix(const std::string &s, const char *p) {
  return s.rfind(p, 0) == 0;
}
static py::dict copy_dict(py::handle value) {
  if (!py::isinstance<py::dict>(value))
    return py::dict();
  return py::reinterpret_steal<py::dict>(PyDict_Copy(value.ptr()));
}
static py::object pop(py::dict &d, const char *name) {
  return d.attr("pop")(name, py::none());
}
static py::object get(py::dict &d, const char *name) {
  return d.attr("get")(name, py::none());
}

struct Stage {
  int pathway = 0, index = 0;
  py::object source, choose, binding_region;
  py::object description = py::none();
  mutable py::object search_program = py::none(), search_owner = py::none();
  mutable py::object selector_program = py::none(), selector_owner = py::none();
  mutable py::object bound_search = py::none(), bound_choose = py::none();
  mutable int64_t search_revision = -1, choose_revision = -1;
  std::string name, kernel, selector;
  bool repair_region, reuse_output, selection_view, binary_output,
      native_context, input_bindings;
  int arity;
  bool mutation, crossover, ordinary_bit, deferred, cma, random_selection,
      uniform_bit;
};
struct Plan {
  int pathway = 0;
  std::vector<Stage> stages;
  py::object description;
  py::object path_owner, parameter_owner;
};
struct Engine {
  py::object solve, np, list, get_flex, ensure_rng, resolve, arity, mutation;
  py::object choose, evaluate_changed, resize, solution_set, pending;
  NativeData data_path;
  NativeOperators operators;
  NativeUpdates updates;
  bool native_operators = true, native_updates = true;
  bool native_data = true, native_mutation = true;
  bool region_workspace = true, region_fusion = true;
  bool dataflow = true;
  bool planned_computation = false;
  bool recording = true;
  py::object frame = py::none();
  py::object input_context;
  py::dict computation_descriptions, computation_calls, computation_rejections;
  uint64_t selection_materializations = 0, selection_views = 0;
  uint64_t context_regions = 0, context_fallbacks = 0;
  uint64_t plan_build_ns = 0;
  uint64_t event_calls = 0, stage_attempts = 0, interpreted_stages = 0;
  uint64_t numerical_attempts = 0, selector_attempts = 0;
  int next_pathway = 0;
  py::object analyze, contract, builtin, make_computation;
  py::object component_registry = py::none();
  py::object bound_entry(const std::string &name,py::object &entry,int64_t &revision) {
    auto current=py::cast<int64_t>(component_registry.attr("_component_revision"));
    if(entry.is_none() || current!=revision) {
      entry=resolve(name);revision=current;
    }
    return entry;
  }
  explicit Engine(py::object module)
      : solve(module), data_path(module), operators(module), updates(module) {
    input_context=py::module_::import("autooptlib.components._utils").attr("_input_bindings");
    np = py::module_::import("numpy");
    list = py::module_::import("builtins").attr("list");
    get_flex = solve.attr("get_flex");
    ensure_rng = solve.attr("ensure_rng");
    resolve = solve.attr("get_component");
    arity = solve.attr("component_parent_arity");
    mutation = solve.attr("is_mutation_component");
    choose = solve.attr("_stream_choose");
    evaluate_changed = solve.attr("_stream_evaluate_changed");
    resize = solve.attr("_resize_parent_indices");
    solution_set = solve.attr("SolutionSet");
    pending = solve.attr("_PendingStreamOffspring");
    analyze =
        py::module_::import("autooptlib.runtime.regions").attr("analyze_path");
    contract = py::module_::import("autooptlib.components._execution")
                   .attr("execution_contract");
    builtin = py::module_::import("autooptlib.components._execution")
                  .attr("is_builtin_component");
    make_computation = py::module_::import("autooptlib.components._execution")
                           .attr("computation");
  }
  void ensure_frame() {
    if (!frame.is_none()) return;
    frame = py::module_::import("autooptlib.runtime.frame").attr("RunFrame")();
    frame.attr("metrics").attr("configure")(recording);
    auto native = py::module_::import("autooptlib.runtime._native._native_stream");
    frame.attr("set_native")(native.attr("gather_decisions"), native.attr("execute_interval"),
                             native.attr("execute_numeric_kernel"), native.attr("numeric_environment"),
                             native.attr("compile_program"));
  }
  std::shared_ptr<Plan> compile(py::object path, py::object params,
                                py::object setting, int population_size,
                                py::object pathway = py::none()) {
    if (dataflow) ensure_frame();
    auto started = std::chrono::steady_clock::now();
    auto plan = std::make_shared<Plan>();
    plan->path_owner = path;
    plan->parameter_owner = params;
    plan->pathway = pathway.is_none() ? next_pathway++ : py::cast<int>(pathway);
    // N keeps the same numerical kernels but pays no new lifetime-analysis
    // cost. G must amortize its own graph planning in end-to-end timings.
    plan->description =
        (region_workspace || region_fusion || dataflow || planned_computation)
            ? analyze(path)
            : py::object(py::dict("version"_a = 1, "stages"_a = py::list(),
                                  "baseline"_a = true));
    py::sequence stages = path.attr("stages"),
                 parameters = params.attr("stages");
    for (py::ssize_t i = 0; i < py::len(stages); ++i) {
      Stage s;
      s.pathway = plan->pathway;
      s.index = int(i);
      s.source = stages[i];
      s.name = py::str(s.source.attr("search").attr("primary"));
      if (region_workspace || region_fusion || dataflow || planned_computation) {
        auto region = plan->description["stages"][py::int_(i)];
        s.description = region;
        s.kernel = py::cast<std::string>(region["kernel"]);
        s.native_context = py::cast<bool>(region["native_context"]);
        s.repair_region = py::cast<bool>(region["repair_region"]);
        s.reuse_output = py::cast<bool>(region["reuse_output"]);
        s.selection_view = py::cast<bool>(region["selection_view"]);
        s.binary_output =
            py::cast<std::string>(region["output_domain"]) == "binary";
        s.input_bindings = dataflow && (py::len(region["input_bindings"]) > 0 ||
                                      py::cast<bool>(region["row_broadcast"]));
        s.binding_region = plan->description["regions"][py::int_(i)];
      } else {
        s.kernel = py::cast<std::string>(
            py::getattr(contract(s.name), "kernel", py::str("")));
        s.repair_region = s.reuse_output = s.selection_view = s.binary_output =
            s.native_context = s.input_bindings = false;
      }
      auto selector = s.source.attr("choose");
      s.selector = py::str(selector);
      s.choose = selector;
      s.arity = py::cast<int>(arity(s.name));
      s.mutation = py::cast<bool>(mutation(s.name));
      s.crossover = prefix(s.name, "cross_");
      s.ordinary_bit = prefix(s.name, "search_bit_");
      s.deferred =
          i + 1 == py::len(stages) && (prefix(s.name, "search_fastga_bit_") ||
                                       prefix(s.name, "search_fastga_de_"));
      s.cma = s.name == "search_cma";
      s.random_selection =
          !selector.is_none() &&
          py::str(selector).cast<std::string>() == "choose_random";
      s.uniform_bit = s.name == "search_bit_uniform";
      if(!component_registry.is_none()) {
        bound_entry(s.name,s.bound_search,s.search_revision);
        if(!s.choose.is_none())bound_entry(s.selector,s.bound_choose,s.choose_revision);
        // Complete native entries accept record lists directly. Do not build
        // a SolutionSet just to unpack it again at the numerical boundary.
        s.selection_view=true;
      }
      plan->stages.push_back(std::move(s));
    }
    plan_build_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    return plan;
  }
  py::object bind_computation(const std::string &name, py::object implementation,
                              py::object &cached, py::object &owner) {
    if (!planned_computation)
      return py::none();
    bool reuse = py::cast<bool>(frame.attr("preparations").attr("enabled"));
    bool native = py::cast<bool>(frame.attr("native_blocks"));
    if (!reuse && !native && !recording) return py::none();
    auto active = [&](py::object plan) {
      return recording || (reuse && py::cast<bool>(plan.attr("has_preparations")))
          || (native && py::cast<bool>(plan.attr("has_native")));
    };
    auto current = contract(name, implementation);
    if (current.is_none() ||
        !py::cast<bool>(py::getattr(current, "computation", py::bool_(false)))) {
      auto reason = py::str(py::cast<bool>(builtin(name, implementation))
                               ? "unsupported-template" : "implementation");
      if (recording) {
        computation_rejections[reason] = py::int_(
            py::cast<uint64_t>(computation_rejections.attr("get")(reason, 0)) + 1);
        frame.attr("note_binding")(false, reason);
      }
      return py::none();
    }
    if (!cached.is_none() && owner.is(implementation) &&
        cached.attr("reference_code").ptr() == py::getattr(implementation, "__code__", py::none()).ptr() &&
        py::cast<bool>(cached.attr("specification").attr("entry_valid")())) {
      if (recording) frame.attr("note_binding")(true);
      return active(cached) ? cached : py::object(py::none());
    }
    auto started = std::chrono::steady_clock::now();
    cached = make_computation(name, implementation);
    owner = implementation;
    plan_build_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - started).count();
    if (!cached.is_none() && recording)
      computation_descriptions[py::str(name)] = cached.attr("description");
    else if (cached.is_none() && recording) {
      const char *reason = py::cast<bool>(builtin(name, implementation))
                               ? "unsupported-template" : "implementation";
      auto key = py::str(reason);
      computation_rejections[key] = py::int_(
          py::cast<uint64_t>(computation_rejections.attr("get")(key, 0)) + 1);
    }
    if (recording)
      frame.attr("note_binding")(!cached.is_none(), cached.is_none() ? py::object(py::str("implementation-template")) : py::object(py::none()));
    return !cached.is_none() && active(cached) ? cached : py::object(py::none());
  }
  py::tuple execute_computation(const std::string &name, py::object program,
                                py::tuple arguments, py::dict origin) {
    auto key = py::str(name);
    if (recording)
      computation_calls[key] = py::int_(
          py::cast<uint64_t>(computation_calls.attr("get")(key, 0)) + 1);
    // Once execution begins, None and exceptions are results/errors, never a
    // request to replay the component. Refusal happens only in binding above.
    return py::cast<py::tuple>(frame.attr("execute")(program, arguments, origin));
  }
  py::dict source(const Stage &stage, const char *phase, int64_t generation) {
    return py::dict("pathway"_a=stage.pathway, "stage"_a=stage.index,
                    "phase"_a=phase, "event"_a=generation);
  }
  py::object materialize_selection(py::object selected) {
    ++selection_materializations;
    if (planned_computation)
      return frame.attr("selection")(selected, solution_set);
    return solution_set(selected);
  }
  py::tuple select(const Stage &s, py::object parameter, py::object pool,
                   py::object population, py::dict state, py::object problem,
                   py::object data, py::object rng, int64_t generation,
                   py::object *targets = nullptr) {
    AOL_REGION_TIME(1);
    if (s.choose.is_none()) {
      if (pool.is(population)) {
        // A numeric region carries targeting as native execution data. Bit
        // and polynomial kernels do not read it; defer its materialization
        // until an unsupported kernel crosses back into Python.
        if (!targets || s.kernel == "de" || s.kernel == "fastga_de") {
          auto indices = np.attr("arange")(
              py::len(pool),
              "dtype"_a = py::module_::import("builtins").attr("int"));
          if (targets)
            *targets = indices;
          else
            state["_target_population_indices"] = indices;
        }
      } else if (!targets)
        pop(state, "_target_population_indices");
      return py::make_tuple(pool, state);
    }
    if (planned_computation)
      ++selector_attempts;
    auto choose_fn = component_registry.is_none()?resolve(s.choose):
        bound_entry(s.selector,s.bound_choose,s.choose_revision);
    bool random_selection = component_registry.is_none() && s.random_selection &&
                            py::cast<bool>(builtin(s.choose, choose_fn));
    if (!targets || !random_selection)
      state["_selection_count"] = s.arity;
    py::tuple result;
    if (random_selection && py::len(pool) > 0) {
      py::object selection_rng = state.attr("get")("rng", rng);
      result = py::make_tuple(
          selection_rng.attr("integers")(
              0, py::len(pool), "size"_a = s.arity,
              "dtype"_a = py::module_::import("builtins").attr("int")),
          py::none());
    } else {
      auto program = bind_computation(s.selector, choose_fn,
                                      s.selector_program, s.selector_owner);
      if (recording && !program.is_none() && !s.description.is_none())
        s.description["selector_program"] = program.attr("description");
      auto selection_parameter = py::getattr(parameter, "choose", py::none());
      if (!program.is_none())
        result = execute_computation(s.selector, program,
            py::make_tuple(pool, problem, selection_parameter, state,
                           generation, 1, data, "execute"), source(s, "selection", generation-s.index));
      else {
        std::optional<py::object> direct;
        if (component_registry.is_none() && !planned_computation && (operators.direct->native || operators.direct->preparations) && py::cast<bool>(builtin(s.choose,choose_fn)))
          direct=operators.direct->execute(s.selector,choose_fn,
              py::make_tuple(pool,problem,selection_parameter,state,generation,1,data,"execute"));
        if(direct) result=*direct;
        else result=choose_fn(pool,problem,selection_parameter,state,generation,1,data,"execute");
      }
    }
    pop(state, "_selection_count");
    if (py::isinstance<py::dict>(result[1])) {
      state = py::reinterpret_borrow<py::dict>(result[1]);
      state.attr("setdefault")("rng", rng);
    }
    pop(state, "_selection_count");
    py::object indices =
        random_selection && py::len(pool) > 0
            ? py::reinterpret_borrow<py::object>(result[0])
            : resize(result[0], "population_size"_a = py::len(pool),
                     "offspring_size"_a = s.arity, "rng"_a = rng);
    if (pool.is(population)) {
      auto value = np.attr("asarray")(
          indices, "dtype"_a = py::module_::import("builtins").attr("int"));
      if (targets)
        *targets = value;
      else
        state["_target_population_indices"] = value;
    } else if (!targets)
      pop(state, "_target_population_indices");
    py::list selected;
    for (auto index : indices)
      selected.append(pool[py::int_(py::cast<int>(index))]);
    if (region_fusion && s.selection_view) {
      ++selection_views;
      return py::make_tuple(selected, state);
    }
    return py::make_tuple(materialize_selection(selected), state);
  }
  py::tuple event(py::object path, py::object params, py::object population,
                  py::object aux, py::object problem, py::object data,
                  py::object setting, int64_t generation, int64_t remaining,
                  double mutation_rate, int maximum_outputs,
                  std::shared_ptr<Plan> plan) {
    AOL_REGION_TIME(0);
    NativeBindingScope context_scope(input_context, operators.direct->native || operators.direct->preparations);
    if (planned_computation)
      ++event_calls;
    py::object rng = ensure_rng(setting);
    py::dict path_state = copy_dict(aux);
    path_state.attr("setdefault")("rng", plan ? rng : ensure_rng(setting));
    py::list stage_states;
    for (auto state : path_state.attr("get")("_stream_stage_aux", py::list()))
      stage_states.append(copy_dict(state));
    py::sequence stages = path.attr("stages"),
                 parameters = params.attr("stages");
    while (py::len(stage_states) < py::len(stages))
      stage_states.append(py::dict());
    for (auto state : stage_states)
      state.attr("setdefault")("rng", plan ? rng : ensure_rng(setting));
    py::object pool = population;
    int64_t evaluations = 0;
    py::list observed;
    bool crossed = false;
    // Direct mode resolves stage metadata on every event, like the reference.
    // Planned mode loads immutable per-run Stage records.
    for (py::ssize_t i = 0; i < std::min(py::len(stages), py::len(parameters));
         ++i) {
      const Stage *compiled =
          plan && size_t(i) < plan->stages.size() ? &plan->stages[i] : nullptr;
      py::object stage = stages[i], parameter = parameters[i];
      // A prior callback may edit the graph. Guard before this stage's effects,
      // then interpret this stage without restarting the event or its RNG.
      if (compiled &&
          (!compiled->source.is(stage) ||
           compiled->name != py::str(stage.attr("search").attr("primary"))
                                 .cast<std::string>() ||
           compiled->selector !=
               py::str(stage.attr("choose")).cast<std::string>()))
        compiled = nullptr;
      if (planned_computation) {
        ++stage_attempts;
        interpreted_stages += compiled == nullptr;
      }
      py::dict state = py::reinterpret_borrow<py::dict>(stage_states[i]);
      state["_stream_generation_target"] = py::cast<int>(
          get_flex(setting, "_AlgorithmOffspringSize", py::len(population)));
      int parent_count =
          compiled ? compiled->arity
                   : py::cast<int>(arity(stage.attr("search").attr("primary")));
      bool local_frame = compiled && region_fusion && compiled->native_context;
      py::object target_indices = py::none();
      py::tuple selected_result =
          compiled
              ? select(*compiled, parameter, pool, population, state, problem,
                       data, rng, generation + i,
                       local_frame ? &target_indices : nullptr)
              : py::cast<py::tuple>(choose(pool, population, stage, parameter,
                                           state, problem, data, setting,
                                           generation + i, parent_count));
      py::object selected = selected_result[0];
      bool selected_view =
          compiled && region_fusion && compiled->selection_view;
      state = py::reinterpret_borrow<py::dict>(selected_result[1]);
      // Selection can invoke an RNG subclass. Revalidate at the reference
      // operator read point, after its effects and before invoking the kernel.
      if (compiled &&
          compiled->name !=
              py::str(stage.attr("search").attr("primary")).cast<std::string>())
        compiled = nullptr;
      std::string primary =
          compiled ? compiled->name
                   : py::cast<std::string>(
                         py::str(stage.attr("search").attr("primary")));
      if (crossed &&
          (compiled ? compiled->mutation : py::cast<bool>(mutation(primary)))) {
        // Draw after selection, including rates zero and one.
        if (!(py::cast<double>(rng.attr("random")()) < mutation_rate)) {
          if (compiled && planned_computation)
            frame.attr("boundary")(source(*compiled, "search", generation), "mutation-skipped");
          if (local_frame && pool.is(population)) {
            if (target_indices.is_none())
              target_indices =
                  np.attr("arange")(py::len(pool), "dtype"_a = np.attr("int_"));
            state["_target_population_indices"] = target_indices;
          } else if (local_frame)
            pop(state, "_target_population_indices");
          pool = selected_view ? materialize_selection(selected) : selected;
          stage_states[i] = state;
          continue;
        }
      }
      if (!local_frame)
        state["_population_context"] = population;
      if (planned_computation)
        ++numerical_attempts;
      py::object search = compiled && !component_registry.is_none()
          ? bound_entry(primary,compiled->bound_search,compiled->search_revision)
          : resolve(stage.attr("search").attr("primary"));
      py::object current_contract = compiled && component_registry.is_none() ? contract(primary, search) : py::none();
      py::object search_parameter =
          py::getattr(parameter.attr("search"), "primary", py::none());
      py::object native_result =
          compiled && !current_contract.is_none() && native_operators
              ? operators.execute(primary, selected, problem, search_parameter,
                                  state, data_path, compiled->kernel,
                                  local_frame, population, target_indices)
              : py::none();
      if (native_result.is_none() && compiled && !current_contract.is_none() && native_mutation &&
          compiled->uniform_bit)
        native_result = data_path.uniform_mutation(selected, problem, state);
      if (local_frame) {
        if (native_result.is_none())
          ++context_fallbacks;
        else
          ++context_regions;
      }
      // An unsupported parameter/layout cuts the region before the Python
      // operator. Materialize its normal container, without replaying
      // selection.
      if (native_result.is_none() && local_frame) {
        state["_population_context"] = population;
        if (pool.is(population)) {
          if (target_indices.is_none())
            target_indices =
                np.attr("arange")(py::len(pool), "dtype"_a = np.attr("int_"));
          state["_target_population_indices"] = target_indices;
        } else
          pop(state, "_target_population_indices");
      }
      if (native_result.is_none() && selected_view && component_registry.is_none())
        selected = materialize_selection(selected);
      if (native_result.is_none() && !planned_computation &&
          (operators.direct->native || operators.direct->preparations) && compiled &&
          !current_contract.is_none() && native_operators) {
        auto direct=operators.direct->execute(primary,search,
            py::make_tuple(selected,problem,search_parameter,state,generation+i,1,data,"execute"));
        if(direct)native_result=py::cast<py::tuple>(*direct);
      }
      bool bind_inputs = compiled && dataflow && compiled->input_bindings &&
                         !current_contract.is_none() && native_result.is_none();
      py::object program =
          compiled && native_result.is_none()
              ? bind_computation(primary, search, compiled->search_program,
                                 compiled->search_owner)
              : py::none();
      if (recording && !program.is_none() && !compiled->description.is_none())
        compiled->description["search_program"] = program.attr("description");
      py::tuple result;
      if (!native_result.is_none())
        result = py::cast<py::tuple>(native_result);
      else if (!program.is_none())
        result = execute_computation(primary, program,
            py::make_tuple(selected, problem, search_parameter, state,
                           generation + i, 1, data, "execute"), source(*compiled, "search", generation));
      else if (bind_inputs)
        result = py::cast<py::tuple>(frame.attr("invoke")(
            search, selected, problem, search_parameter, state,
            generation + i, 1, data, "execute", "region"_a = compiled->binding_region));
      else
        result = py::cast<py::tuple>(search(
            selected, problem, search_parameter, state,
            generation + i, 1, data, "execute"));
      py::object decisions = result[0];
      if (py::isinstance<py::dict>(result[1])) {
        state = py::reinterpret_borrow<py::dict>(result[1]);
        state.attr("setdefault")("rng", plan ? rng : ensure_rng(setting));
      }
      py::object invalidated = pop(state, "_stream_invalidated_mask");
      if (invalidated.is_none() &&
          !(compiled ? compiled->ordinary_bit : prefix(primary, "search_bit_")))
        invalidated = py::bool_(true);
      pop(state, "_population_context");
      pop(state, "_target_population_indices");
      bool deferred = compiled ? compiled->deferred
                               : (i + 1 == py::len(stages) &&
                                  (prefix(primary, "search_fastga_bit_") ||
                                   prefix(primary, "search_fastga_de_")));
      py::object array = np.attr("asarray")(decisions);
      if (py::cast<int>(array.attr("ndim")) == 1)
        array = array.attr("reshape")(1, -1);
      if (deferred) {
        if (compiled && planned_computation)
          frame.attr("boundary")(source(*compiled, "search", generation), "pending-transfer");
        array = array[py::slice(0, std::max(0, maximum_outputs), 1)];
        py::object flags =
            np.attr("asarray")(invalidated,
                               "dtype"_a =
                                   py::module_::import("builtins").attr("bool"))
                .attr("reshape")(-1);
        if (py::cast<int>(flags.attr("size")) == 1)
          flags = np.attr("repeat")(flags, py::len(array));
        else
          flags = flags[py::slice(0, py::len(array), 1)];
        if (py::cast<py::ssize_t>(flags.attr("size")) != py::len(array))
          throw py::value_error("A final FastGA mutation must return one "
                                "invalidation flag per offspring.");
        py::list results;
        for (py::ssize_t j = 0; j < py::len(array); ++j) {
          py::object decision = array[py::int_(j)];
          if (!py::cast<bool>(flags[py::int_(j)])) {
            py::object clone = py::none();
            PopulationView source_view;
            DecisionRow query;
            if (plan && native_data && DecisionRow::read(decision, query) &&
                source_view.load(selected, list)) {
              auto index = source_view.match(query);
              if(index >= 0) clone = source_view.sources[py::int_(index)];
            }
            else
              for (auto source : selected) {
                if (py::cast<bool>(np.attr("array_equal")(
                        np.attr("asarray")(source.attr("dec")), decision))) {
                  clone = py::reinterpret_borrow<py::object>(source);
                  break;
                }
              }
            if (clone.is_none())
              throw std::runtime_error(
                  "A valid FastGA mutation clone changed its decision values.");
            results.append(clone);
          } else {
            py::list yes;
            yes.append(true);
            results.append(
                pending(np.attr("asarray")(decision).attr("reshape")(1, -1),
                        selected, np.attr("asarray")(yes)));
          }
        }
        stage_states[i] = state;
        path_state["_stream_stage_aux"] = stage_states;
        return py::make_tuple(results, path_state, evaluations, observed);
      }
      if (i + 1 == py::len(stages)) {
        array = array[py::slice(0, std::max(0, maximum_outputs), 1)];
        if (!invalidated.is_none()) {
          py::object flags =
              np.attr("asarray")(
                    invalidated,
                    "dtype"_a = py::module_::import("builtins").attr("bool"))
                  .attr("reshape")(-1);
          if (py::cast<int>(flags.attr("size")) > 1)
            invalidated = flags[py::slice(0, py::len(array), 1)];
        }
      }
      bool reuse_storage =
          region_fusion && compiled && compiled->reuse_output &&
          !native_result.is_none() && py::isinstance<py::array>(decisions) &&
          py::reinterpret_borrow<py::array>(decisions).owndata();
      bool program_private = !program.is_none() && py::cast<bool>(frame.attr("last_private_output"));
      if (compiled && planned_computation)
        frame.attr("boundary")(source(*compiled, "search", generation), "repair-evaluate");
      py::tuple evaluated =
          plan && native_data
              ? py::cast<py::tuple>(data_path.changed(
                    array, selected, problem, data, remaining - evaluations,
                    invalidated, evaluate_changed,
                    (region_workspace || reuse_storage) && compiled &&
                        (compiled->repair_region || program_private) && !current_contract.is_none(),
                    reuse_storage, compiled && !current_contract.is_none() && compiled->binary_output,
                    region_workspace))
              : py::cast<py::tuple>(
                    evaluate_changed(array, selected, problem, data,
                                     remaining - evaluations, invalidated));
      py::object produced = evaluated[0];
      int64_t used = py::cast<int64_t>(evaluated[1]);
      if (produced.is_none()) {
        if (compiled && planned_computation)
          frame.attr("boundary")(source(*compiled, "search", generation), "budget-exit");
        evaluations += used;
        observed.attr("extend")(evaluated[2]);
        stage_states[i] = state;
        path_state["_stream_stage_aux"] = stage_states;
        return py::make_tuple(py::none(), path_state, evaluations, observed);
      }
      pool = produced;
      if (compiled ? compiled->cma : primary == "search_cma") {
        py::object disturbances = np.attr("asarray")(
            get(state, "cma_Disturb"),
            "dtype"_a = py::module_::import("builtins").attr("float"));
        if (py::cast<int>(disturbances.attr("ndim")) != 2 ||
            py::len(disturbances) < py::len(produced))
          throw std::runtime_error("Stream CMA did not retain one disturbance "
                                   "per evaluated sample.");
        state.attr("setdefault")("_stream_cma_disturbances", py::list())
            .attr("extend")(
                disturbances[py::slice(0, py::len(produced), 1)].attr(
                    "copy")());
        state.attr("setdefault")("_stream_cma_solutions", py::list())
            .attr("extend")(list(produced));
      }
      evaluations += used;
      observed.attr("extend")(evaluated[2]);
      stage_states[i] = state;
      if (compiled && planned_computation)
        frame.attr("boundary")(source(*compiled, "search", generation), "state-commit");
      crossed |= compiled ? compiled->crossover : prefix(primary, "cross_");
    }
    path_state["_stream_stage_aux"] = stage_states;
    return py::make_tuple(list(pool), path_state, evaluations, observed);
  }
#include "generation.hpp"
};
PYBIND11_MODULE(_native_stream, m) {
  py::class_<DirectNative,std::shared_ptr<DirectNative>>(m,"DirectNative")
      .def("prefix",&DirectNative::prefix).def("finish",&DirectNative::finish)
      .def("clustering",&DirectNative::clustering);
  py::class_<cpu_program::DependencyProof, std::shared_ptr<cpu_program::DependencyProof>>(m, "DependencyProof")
      .def("trusted", &cpu_program::DependencyProof::trusted, "numeric"_a=true);
  py::class_<cpu_program::Program>(m, "Program")
      .def("execute", &cpu_program::Program::execute);
  m.def("compile_program", [](py::object plan) { return cpu_program::Program(plan); });
  py::class_<UpdatePlan, std::shared_ptr<UpdatePlan>>(m, "UpdatePlan")
      .def_property_readonly("description",
                             [](const UpdatePlan &p) { return p.description; });
  py::class_<Plan, std::shared_ptr<Plan>>(m, "Plan")
      .def_property_readonly("description", [](const Plan &p) { return p.description; })
      .def("matches", [](const Plan &p, py::object path, py::object parameters) {
          return p.path_owner.is(path) && p.parameter_owner.is(parameters);
      });
  py::class_<Engine>(m, "Engine")
      .def(py::init<py::object>())
      .def("bind_components", [](Engine &e,py::object registry) {
        e.component_registry=registry;e.updates.component_registry=registry;
      })
      .def("compile", &Engine::compile, "path"_a, "params"_a, "setting"_a,
           "population_size"_a, "pathway"_a=py::none())
      .def("event", &Engine::event)
      .def("collect", &Engine::collect)
      .def("changed",
           [](Engine &e, py::object decisions, py::object sources,
              py::object problem, py::object data, int64_t remaining,
              py::object invalidated,bool aligned) {
             if (!e.native_data) {
               if(aligned)return e.evaluate_changed(decisions,sources,problem,data,remaining,invalidated,"aligned_sources"_a=true);
               return e.evaluate_changed(decisions,sources,problem,data,remaining,invalidated);
             }
             return e.data_path.changed(decisions,sources,problem,data,remaining,
                                        invalidated,e.evaluate_changed,e.region_workspace,
                                        false,false,true,aligned);
           },"decisions"_a,"sources"_a,"problem"_a,"data"_a,"remaining"_a,"invalidated"_a,
             "aligned_sources"_a=false)
      .def(
          "configure_cost_selection",
          [](Engine &e, bool enabled) { e.data_path.cost_selection = enabled; })
      .def("configure_isa",
           [](Engine &e, bool enabled) {
             e.data_path.simd = enabled && exact_simd::available();
           })
      .def("configure_dataflow", [](Engine &e, bool enabled) { e.dataflow = enabled; if(enabled)e.ensure_frame(); })
      .def("configure_metrics", [](Engine &e, bool enabled) {
          e.recording = enabled;
          e.data_path.recording = enabled;
          e.operators.direct->recording=enabled;
          if (!e.frame.is_none()) e.frame.attr("metrics").attr("configure")(enabled);
      }, "enabled"_a=true)
      .def("configure_acceleration", [](Engine &e, bool preparations, bool native_blocks,
                                         int64_t max_bytes) {
          if(max_bytes<0)throw py::value_error("Preparation byte budget must be nonnegative");
          e.operators.direct->elite_record.reset();
          if(e.planned_computation) {
            e.frame.attr("configure_acceleration")(preparations,native_blocks,max_bytes);
            e.operators.preparation_store=preparations ? e.frame.attr("preparations") : py::object(py::none());
            e.operators.direct->configure(false,false,max_bytes);
          } else {
            e.operators.preparation_store=py::none();
            e.operators.direct->configure(preparations,native_blocks,max_bytes);
          }
      }, "preparations"_a=false, "native_blocks"_a=false,
         "max_preparation_bytes"_a=32*1024*1024)
      .def("configure_computations", [](Engine &e, bool enabled, bool measure) {
          e.planned_computation = enabled;
          if(enabled) e.ensure_frame();
          if(!e.frame.is_none()) e.frame.attr("measure") = measure;
          if (enabled)
            e.data_path.workspace_provider = e.frame.attr("repair_workspace");
          else
            e.data_path.workspace_provider = py::none();
          if (enabled)
            e.updates.boundary = e.frame.attr("boundary");
          else
            e.updates.boundary = py::none();
      }, "enabled"_a, "measure"_a = false)
      .def("boundary", [](Engine &e, const std::string &phase, int64_t generation) {
          if (e.planned_computation)
            e.frame.attr("boundary")(py::dict("pathway"_a=-1, "stage"_a=-1,
                                             "event"_a=generation), phase);
      })
      .def("set_population_type",
           [](Engine &e, py::object type) {
             e.solution_set = type;
             e.data_path.solution_set = type;
           })
      .def(
          "configure_regions",
          [](Engine &e, bool workspace, bool fusion) {
            e.region_workspace = workspace;
            e.region_fusion = fusion;
          },
          "workspace"_a = true, "fusion"_a = true)
      .def(
          "configure_kernels",
          [](Engine &e, bool data, bool mutation) {
            e.native_data = data;
            e.native_mutation = mutation;
          },
          "data"_a = true, "mutation"_a = true)
      .def(
          "configure_operators",
          [](Engine &e, bool numerical, bool replacement) {
            e.native_operators = numerical;
            e.native_updates = replacement;
          },
          "numerical"_a = true, "replacement"_a = true)
      .def("compile_updates",
           [](Engine &e, py::sequence paths, py::sequence parameters) {
             return e.updates.compile(paths, parameters);
           })
      .def("finish_updates",
           [](Engine &e, std::shared_ptr<UpdatePlan> plan, py::list offspring,
              py::object solutions, py::list auxiliary, py::object problem,
              py::object data, py::object rng, int64_t generation) {
             return e.updates.finish(plan, offspring, solutions, auxiliary,
                                     problem, data, rng, generation,
                                     e.native_updates);
           })
      .def("operator",
           [](Engine &e, std::string name, py::object selected,
              py::object problem, py::object parameter, py::dict state) {
             return e.operators.execute(
                 name, selected, problem, parameter, state, e.data_path,
                 py::cast<std::string>(
                     py::getattr(e.contract(name), "kernel", py::str(""))));
           })
      .def("replacement",
           [](Engine &e, std::string name, py::list values, py::object problem,
              py::object parameter, py::dict state, int64_t generation) {
             return e.updates.replace(name, values, problem, parameter, state,
                                      generation);
           })
      .def("statistics", [](Engine &e) {
        auto result = e.data_path.stats();
        result["de_rows"] = e.operators.de_rows;
        result["polynomial_rows"] = e.operators.polynomial_rows;
        result["bit_rows"] = e.operators.bit_rows;
        result["native_replacements"] = e.updates.replacements;
        result["update_generations"] = e.updates.generations;
        result["update_fallbacks"] = e.updates.fallbacks;
        result["selection_materializations"] = e.selection_materializations;
        result["selection_views"] = e.selection_views;
        result["context_regions"] = e.context_regions;
        result["context_fallbacks"] = e.context_fallbacks;
        result["input_bindings"] = e.frame.is_none() ? py::object(py::dict()) : e.frame.attr("description")();
        result["direct_native"] = e.operators.direct->stats();
        result["computation_calls"] = e.computation_calls;
        result["computation_plans"] = e.computation_descriptions;
        result["computation_rejections"] = e.computation_rejections;
        result["plan_build_ns"] = e.plan_build_ns;
        py::dict managed = e.frame.is_none() ? py::dict() : py::cast<py::dict>(e.frame.attr("metrics").attr("description")());
        if(e.frame.is_none()) {
          managed["enabled"]=e.recording;
          managed["counters"]=e.recording ? py::dict(
            "preparation_requests"_a=e.operators.direct->elite_requests,
            "preparation_hits_de-elite"_a=e.operators.direct->elite_hits,
            "preparation_recomputations"_a=e.operators.direct->elite_requests-e.operators.direct->elite_hits) : py::dict();
        }
        managed["native_target_copy_bytes"] = e.recording ? py::object(py::int_(e.data_path.materialized_bytes)) : py::object(py::none());
        managed["native_target_allocations"] = e.recording ? py::object(py::int_(e.data_path.target_input_allocations)) : py::object(py::none());
        managed["native_repair_allocations"] = e.recording ? py::object(py::int_(0)) : py::object(py::none());
        managed["native_conversion_bytes"] = e.recording ? py::object(py::int_(e.data_path.converted_input_bytes)) : py::object(py::none());
        managed["plan_build_ns"] = e.plan_build_ns;
        managed["outer_coverage"] = py::dict(
            "enabled"_a=e.planned_computation, "event_calls"_a=e.event_calls,
            "stage_attempts"_a=e.stage_attempts, "interpreted_stages"_a=e.interpreted_stages,
            "numerical_attempts"_a=e.numerical_attempts,
            "compiled_selector_attempts"_a=e.selector_attempts);
        result["execution_metrics"] = managed;
        return result;
      });
  m.def("isa_available", &exact_simd::available);
  m.def(
      "row_equal",
      [](py::object a, py::object b, bool simd, int repeats) {
        DecisionRow x, y;
        if (!DecisionRow::read(a, x) || !DecisionRow::read(b, y))
          throw py::value_error("Expected native float64 or int64 rows");
        bool enabled = simd && exact_simd::available();
        uint64_t count = 0;
        for (int i = 0; i < repeats; ++i) {
#if defined(__GNUC__) || defined(__clang__)
          asm volatile("" ::: "memory"); // prevent benchmark-only loop hoisting
#endif
          count += x.equal(y, enabled);
        }
        return count;
      },
      "a"_a, "b"_a, "simd"_a = false, "repeats"_a = 1);
  m.def("region_profile", &region_profile);
  m.def("gather_decisions", &gather_decisions, "items"_a, "destination"_a=py::none());
  auto numeric_kernels = std::make_shared<cpu_blocks::Kernels>();
  m.def("execute_numeric_kernel", [numeric_kernels](const std::string &kernel, const py::tuple &inputs) {
        if (kernel != "splice_one" && kernel != "splice_two" && kernel != "splice_uniform"
            && cpu_blocks::numeric_environment().is_none()) return py::object(py::none());
        return numeric_kernels->execute(kernel, inputs);
      },
        "kernel"_a, "inputs"_a);
  m.def("numeric_environment", &cpu_blocks::numeric_environment);
  m.def("execute_interval", [](py::object region, py::object invocation, py::object workspace) {
    py::object reason = invocation.attr("admit_region")(region, workspace);
    if (!reason.is_none())
      return invocation.attr("refuse")(region, reason);
    auto native_outcome = invocation.attr("native_region")(region);
    if (!native_outcome.is_none()) return native_outcome;
    // No arithmetic is replaced here. This launcher consumes region bounds,
    // typed live inputs, successor PCs, actual writes and exact exits.
    auto entry = py::cast<int>(region.attr("entry"));
    auto exit = py::cast<int>(region.attr("exit"));
    py::object pc = py::int_(entry);
    while (!pc.is_none() && py::cast<int>(pc) >= entry && py::cast<int>(pc) < exit)
      pc = invocation.attr("step")(pc);
    return invocation.attr("leave_region")(region, pc);
  });
  m.def("gather_scalar", &gather_scalar);
}

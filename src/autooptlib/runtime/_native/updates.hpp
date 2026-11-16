// Shared Search sinks: preserve first-arrival grouping, replacement ties,
// stochastic acceptance order and the original Solution object identities.
#pragma once
#include "operators.hpp"
#include <memory>

struct UpdateEndpoint {
  std::string name, active_search;
  py::object parameter;
  py::object bound = py::none();
  int64_t revision = -1;
};
struct UpdatePlan {
  py::sequence paths, parameters;
  py::object description;
  std::vector<UpdateEndpoint> endpoints;
};
struct NativeUpdates {
  py::object boundary = py::none(), component_registry = py::none();
  py::object solve, np, flex, solution_type, contract, generator_type;
  bool native_integer_width;
  uint64_t generations = 0, replacements = 0, children = 0, fallbacks = 0;
  explicit NativeUpdates(py::object module)
      : solve(module), np(py::module_::import("numpy")) {
    flex = py::module_::import("autooptlib.components._utils").attr("flex_get");
    generator_type=np.attr("random").attr("Generator");
    solution_type = module.attr("SolutionSet");
    contract = py::module_::import("autooptlib.components._execution").attr("update_contract");
    native_integer_width =
        py::cast<int>(
            np.attr("dtype")(py::module_::import("builtins").attr("int"))
                .attr("itemsize")) == 8;
  }
  std::shared_ptr<UpdatePlan> compile(py::sequence paths,
                                      py::sequence parameters) {
    auto plan = std::make_shared<UpdatePlan>();
    plan->paths = paths;
    plan->parameters = parameters;
    plan->description = py::module_::import("autooptlib.runtime.regions")
                            .attr("analyze_updates")(paths);
    for (size_t i = 0; i < py::len(paths); ++i) {
      UpdateEndpoint e;
      e.name = py::str(paths[i].attr("update"));
      e.active_search = py::str(
          paths[i].attr("stages")[py::int_(-1)].attr("search").attr("primary"));
      if(!component_registry.is_none()) {
        e.bound=solve.attr("get_component")(e.name);
        e.revision=py::cast<int64_t>(component_registry.attr("_component_revision"));
      }
      plan->endpoints.push_back(std::move(e));
    }
    return plan;
  }
  py::object replace(const std::string &name, py::list values,
                     py::object problem, py::object parameter, py::dict state,
                     int64_t generation, py::object function = py::none()) {
    AOL_REGION_TIME(5);
    if ((name=="update_round_robin" || name=="update_simulated_annealing")
        && !ordinary_generator(state,generator_type))return py::none();
    if (!native_integer_width)
      return py::none();
    if (function.is_none())
      function = solve.attr("get_component")(name);
    if (contract(name, function).is_none())
      return py::none();
    if (name != "update_greedy" && name != "update_fastga_plus" &&
        name != "update_pairwise" && name != "update_simulated_annealing" &&
        name != "update_round_robin")
      return py::none();
    py::ssize_t size = py::len(values);
    py::ssize_t n = py::cast<py::ssize_t>(flex(
        problem, "N",
        name == "update_greedy" || name == "update_round_robin" ? size
                                                                : size / 2));
    if (n <= 0 || n > size)
      return py::none();
    py::array_t<double> fits(size);auto fitness=fits.mutable_data();py::ssize_t row=0;
    for (auto v : values) {
      auto raw = v.attr("fit");
      if (!PyFloat_Check(raw.ptr()))
        return py::none();
      double f = py::cast<double>(raw);
      if (!std::isfinite(f))
        return py::none();
      fitness[row++]=f;
    }
    py::list output;
    auto rng = state.attr("get")("rng", py::none());
    if (name == "update_greedy" || name == "update_round_robin") {
      py::object rank;
      int k = std::min<int>(10, n - 1);
      if (name == "update_greedy" || k <= 0)
        rank = np.attr("argsort")(fits);
      else {
        if (rng.is_none())
          return py::none();
        auto keys = rng.attr("random")(py::make_tuple(size, size));
        auto opponents = py::cast<py::array_t<int64_t>>(
            np.attr("argpartition")(keys, k - 1, "axis"_a = 1));
        py::array_t<int64_t> wins(size);
        auto op = opponents.unchecked<2>();
        for (py::ssize_t i = 0; i < size; ++i) {
          int64_t w = 0;
          for (int j = 0; j < k; ++j)
            w += fitness[i] <= fitness[op(i, j)];
          wins.mutable_data()[i] = w;
        }
        rank = np.attr("argsort")(
            wins)[py::slice(py::none(), py::none(), py::int_(-1))];
      }
      auto indices = py::cast<py::array_t<int64_t>>(rank);
      auto ix = indices.unchecked<1>();
      for (py::ssize_t i = 0; i < n; ++i)
        output.append(values[ix(i)]);
    } else if (name == "update_fastga_plus") {
      std::vector<py::ssize_t> order;
      for (py::ssize_t i = n; i < size; ++i)
        order.push_back(i);
      for (py::ssize_t i = 0; i < n; ++i)
        order.push_back(i);
      std::stable_sort(order.begin(), order.end(),
                       [&](auto a, auto b) { return fitness[a] < fitness[b]; });
      for (py::ssize_t i = 0; i < n; ++i)
        output.append(values[order[i]]);
    } else {
      double temperature = 0.;
      if (name == "update_simulated_annealing") {
        auto para =
            py::array_t<double, py::array::c_style |
                                    py::array::forcecast>::ensure(parameter);
        if (!para || !para.size() || !std::isfinite(para.data()[0]) ||
            para.data()[0] <= 0 || rng.is_none())
          return py::none();
        double initial = para.data()[0];
        auto maximum = std::max<int64_t>(
            1, py::cast<int64_t>(flex(problem, "Gmax", generation + 1)));
        // CPython float power and NumPy exp are retained for exact acceptance.
        auto rate = py::float_(.01 / initial).attr("__pow__")(1. / maximum);
        temperature =
            initial * py::cast<double>(rate.attr("__pow__")(generation));
      }
      for (py::ssize_t i = 0; i < n; ++i)
        output.append(values[i]);
      for (py::ssize_t i = n; i < size; ++i) {
        py::ssize_t target = (i - n) % n;
        double old = fitness[target], next = fitness[i];
        bool accept = next < old;
        if (name == "update_simulated_annealing") {
          double exponent = (old - next) / ((std::abs(old) + 1e-6) *
                                            std::max(temperature, 1e-8));
          if (!std::isnan(exponent))
            exponent = std::min(700., std::max(-700., exponent));
          if (!accept) {
            double coin = py::cast<double>(rng.attr("random")());
            accept = coin < py::cast<double>(np.attr("exp")(exponent));
          }
        }
        if (accept) {
          output[target] = values[i];
          fitness[target] = next;
        }
      }
    }
    ++replacements;
    children += size - n;
    return py::make_tuple(output, py::none());
  }
  py::object finish(std::shared_ptr<UpdatePlan> plan, py::list offspring,
                    py::object solutions, py::list auxiliary,
                    py::object problem, py::object data, py::object rng,
                    int64_t generation, bool kernels) {
    AOL_REGION_TIME(6);
    struct Group {
      size_t owner;
      py::list children;
    };
    std::vector<Group> groups;
    // Compare the current parameter values at the generation boundary.
    // Equality is not assumed reflexive (NaNs) or transitive (mixed dtypes).
    // Group representatives remain the first offspring's endpoint, as in
    // Python.
    // Capture CURRENT endpoint values at the reference generation boundary.
    // Mutable graph fields are facts owned by paths/parameters. Functions are
    // resolved at the reference call point, after grouping and prior effects.
    auto endpoints = plan->endpoints;
    if (py::len(plan->paths) != endpoints.size())
      endpoints = compile(plan->paths, plan->parameters)->endpoints;
    for (size_t i = 0; i < endpoints.size(); ++i) {
      auto &e = endpoints[i];
      auto path = plan->paths[i];
      std::string name = py::str(path.attr("update"));
      if(e.name!=name)e.bound=py::none();
      e.name = name;
      e.parameter = py::getattr(plan->parameters[i], "update", py::none());
      e.active_search = py::str(
          path.attr("stages")[py::int_(-1)].attr("search").attr("primary"));
    }
    size_t count = endpoints.size();
    std::vector<std::vector<bool>> compatible(count, std::vector<bool>(count));
    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < count; ++j) {
        const auto &a = endpoints[i], &b = endpoints[j];
        if (a.name != b.name || (a.name == "update_iterated_local_search" &&
                                 a.active_search != b.active_search))
          continue;
        compatible[i][j] =
            a.parameter.is_none() || b.parameter.is_none()
                ? a.parameter.is_none() && b.parameter.is_none()
                : py::cast<bool>(solve.attr("_same_optional_array")(
                      a.parameter, b.parameter));
      }
    for (auto entry : offspring) {
      auto pair = py::reinterpret_borrow<py::tuple>(entry);
      size_t owner = py::cast<size_t>(pair[0]);
      if (owner >= count)
        throw py::index_error("Unknown offspring pathway");
      auto found =
          std::find_if(groups.begin(), groups.end(), [&](const Group &g) {
            return compatible[g.owner][owner];
          });
      if (found == groups.end()) {
        groups.push_back({owner, py::list()});
        found = groups.end() - 1;
      }
      found->children.append(pair[1]);
    }
    py::object current = solutions;
    for (auto &group : groups) {
      auto &endpoint = endpoints[group.owner];
      py::dict origin("pathway"_a=group.owner, "stage"_a=-1,
                      "phase"_a="update", "event"_a=generation);
      if (!boundary.is_none())
        boundary(origin, "generation-update-entry");
      auto function = endpoint.bound;
      if(component_registry.is_none() || function.is_none() ||
          endpoint.revision!=py::cast<int64_t>(component_registry.attr("_component_revision")))
        function=solve.attr("get_component")(endpoint.name);
      py::object old_state = auxiliary[group.owner];
      py::dict state = py::isinstance<py::dict>(old_state)
                           ? py::reinterpret_borrow<py::dict>(old_state)
                           : py::dict("rng"_a = rng);
      state["_active_search_operator"] = endpoint.active_search;
      py::list combined =
          py::module_::import("builtins").attr("list")(current.attr("_items"));
      combined.attr("extend")(group.children);
      py::object result = kernels
                              ? replace(endpoint.name, combined, problem,
                                        endpoint.parameter, state, generation, function)
                              : py::none();
      if (result.is_none()) {
        ++fallbacks;
        result = function(combined, problem, endpoint.parameter, state,
                                   generation, 1, data, "execute");
      }
      py::tuple pair = py::cast<py::tuple>(result);
      state.attr("pop")("_active_search_operator", py::none());
      if (py::isinstance<py::dict>(pair[1])) {
        auto update_state = py::reinterpret_borrow<py::dict>(pair[1]);
        update_state.attr("pop")("_active_search_operator", py::none());
        update_state.attr("setdefault")("rng", rng);
        auxiliary[group.owner] = update_state;
      }
      py::object updated = pair[0];
      if (py::len(updated) != py::len(current))
        updated =
            py::module_::import("builtins")
                .attr("sorted")(
                    updated,
                    "key"_a =
                        py::module_::import("operator")
                            .attr("attrgetter")(
                                "fit"))[py::slice(0, py::len(current), 1)];
      current = solution_type(updated);
      if (!boundary.is_none())
        boundary(origin, "generation-update-commit");
    }
    ++generations;
    return current;
  }
};

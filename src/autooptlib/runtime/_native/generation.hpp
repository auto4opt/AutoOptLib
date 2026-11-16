// Engine member: ordered collection only. Evaluation/state/update semantics
// remain in the existing event core and generation finalizer.
py::tuple collect(py::sequence paths, py::sequence parameters,
                  py::object population, py::list auxiliary, py::object problem,
                  py::object data, py::object setting, int64_t generation,
                  int64_t remaining, py::object event_budget, py::object plan_for) {
  auto rng = ensure_rng(setting);
  int64_t target = py::cast<int64_t>(get_flex(setting, "_AlgorithmOffspringSize", py::len(population)));
  double crossover_rate = py::cast<double>(get_flex(setting, "_AlgorithmCrossoverRate", 1.0));
  double mutation_rate = py::cast<double>(get_flex(setting, "_AlgorithmMutationRate", 1.0));
  std::vector<py::ssize_t> crossed, plain;
  for (py::ssize_t i = 0; i < py::len(paths); ++i) {
    bool cross = false;
    for (auto stage : paths[i].attr("stages"))
      cross |= prefix(py::str(stage.attr("search").attr("primary")), "cross_");
    (cross ? crossed : plain).push_back(i);
  }
  int64_t limit = std::max<int64_t>(10, target * 10);
  if (!event_budget.is_none())
    limit = std::min(limit, std::max<int64_t>(0, py::cast<int64_t>(event_budget)));
  py::list offspring, observed;
  int64_t used = 0, attempts = 0;
  bool interrupted = false;
  while (py::len(offspring) < target && used < remaining && attempts < limit) {
    ++attempts;
    py::ssize_t owner;
    if (crossed.size() == 1 && plain.size() == 1)
      owner = py::cast<double>(rng.attr("random")()) < crossover_rate ? crossed[0] : plain[0];
    else
      owner = py::cast<py::ssize_t>(rng.attr("integers")(0, py::len(paths)));
    auto path = py::reinterpret_borrow<py::object>(paths[owner]);
    auto parameter = py::reinterpret_borrow<py::object>(parameters[owner]);
    // This is the original event preparation point, after routing and prior
    // callbacks. The provider preserves mutable graph/registration visibility.
    auto prepared = plan_for(path, parameter, population, setting);
    auto plan = prepared.is_none() ? std::shared_ptr<Plan>() : prepared.cast<std::shared_ptr<Plan>>();
    auto result = event(path, parameter, population, auxiliary[owner], problem,
                        data, setting, generation, remaining - used,
                        mutation_rate, target - py::len(offspring), plan);
    auxiliary[owner] = result[1];
    used += py::cast<int64_t>(result[2]);
    observed.attr("extend")(result[3]);
    auto children = py::reinterpret_borrow<py::object>(result[0]);
    if (children.is_none() || py::len(children) == 0) {
      interrupted = true;
      break;
    }
    auto count = std::min<py::ssize_t>(py::len(children), target - py::len(offspring));
    for (py::ssize_t i = 0; i < count; ++i)
      offspring.append(py::make_tuple(owner, children[py::int_(i)]));
  }
  return py::make_tuple(offspring, observed, used, attempts, interrupted);
}

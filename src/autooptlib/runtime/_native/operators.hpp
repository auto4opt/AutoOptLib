// Numerical kernels for the measured Search operators. NumPy owns the RNG
// and transcendental/sort operations whose exact implementation is observable.
#pragma once
#include "data.hpp"
#include "direct.hpp"
#include <algorithm>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_map>

struct NativeOperators {
  py::object np, list_type, population_type;
  py::object preparation_store = py::none();
  std::shared_ptr<DirectNative> direct = std::make_shared<DirectNative>();
  uint64_t de_rows = 0, bit_rows = 0, polynomial_rows = 0;
  std::unordered_map<py::ssize_t, py::object> probabilities;
  std::unordered_map<py::ssize_t, std::vector<double>> cumulative;
  explicit NativeOperators(py::object solve)
      : np(py::module_::import("numpy")),
        list_type(py::module_::import("builtins").attr("list")),
        population_type(solve.attr("SolutionSet")) {}

  bool load(py::object source, PopulationView &view) {
    if (py::isinstance<py::list>(source))
      return view.load(source, list_type);
    if (!py::isinstance(source, population_type))
      return false;
    return view.load(source.attr("_items"), list_type);
  }

  static bool finite_rows(const PopulationView &view, py::ssize_t d) {
    for (auto &r : view.rows) {
      if (r.dimension != d)
        return false;
      for (py::ssize_t j = 0; j < d; ++j)
        if (!std::isfinite(r.number_at(j)) || std::abs(r.number_at(j)) > 1e150)
          return false;
    }
    return true;
  }
  static bool fitness(const PopulationView &view, std::vector<double> &values) {
    for (auto item : view.sources) {
      auto f = item.attr("fit");
      if (!PyFloat_Check(f.ptr()))
        return false;
      double value = py::cast<double>(f);
      if (!std::isfinite(value))
        return false;
      values.push_back(value);
    }
    return true;
  }
  static int sample_excluding(py::object rng, int n, int first,
                              int second = -1) {
    int available =
        n - (first >= 0 ? 1 : 0) - (second >= 0 && second != first ? 1 : 0);
    if (available == 0)
      return py::cast<int>(rng.attr("integers")(0, n));
    int rank = py::cast<int>(rng.attr("integers")(0, available));
    for (int i = 0; i < n; ++i)
      if (i != first && i != second) {
        if (rank-- == 0)
          return i;
      }
    throw std::logic_error("Invalid native DE exclusion set");
  }
  py::object elite_preparation(py::object source, const PopulationView &population,
                               const std::vector<double> &fits) {
    auto metrics = preparation_store.attr("metrics");
    bool record_metrics = py::cast<bool>(metrics.attr("enabled"));
    auto count = [&](const char *name) {
      if (record_metrics) {
        auto counts = metrics.attr("counts");
        auto key = py::str(name);
        counts[key] = py::int_(py::cast<uint64_t>(counts[key]) + 1);
      }
    };
    count("preparation_requests");
    auto key = py::make_tuple("de-elite", reinterpret_cast<uintptr_t>(source.ptr()));
    auto records = preparation_store.attr("records");
    py::object record = records.attr("get")(key, py::none());
    bool match = false;
    if (!record.is_none() && record.attr("source").ptr() == source.ptr()) {
      py::tuple members = record.attr("members");
      py::tuple signature = record.attr("signature");
      py::bytes snapshot = signature[1];
      match = py::len(members) == fits.size()
          && PyBytes_GET_SIZE(snapshot.ptr()) == static_cast<py::ssize_t>(fits.size() * sizeof(double))
          && std::memcmp(PyBytes_AS_STRING(snapshot.ptr()), fits.data(), fits.size() * sizeof(double)) == 0;
      for (size_t i = 0; match && i < fits.size(); ++i)
        match = members[i].ptr() == population.sources[py::int_(i)].ptr();
    }
    if (match) {
      records.attr("move_to_end")(key);
      count("preparation_validated_hits");
      count("preparation_hits_de-elite");
      return record.attr("result");
    }
    if (!record.is_none()) count("preparation_validation_failures");
    preparation_store.attr("_drop")(key);
    count("preparation_recomputations");
    return py::none();
  }
  py::object de(py::object selected, py::object parameter, py::dict state,
                bool wrapper, bool local_frame = false,
                py::object frame_population = py::none(),
                py::object frame_targets = py::none()) {
    PopulationView targets, population;
    auto context = local_frame
                       ? frame_population
                       : state.attr("get")("_population_context", py::none());
    if (context.is_none())
      context = selected;
    if (!load(selected, targets) || !load(context, population) ||
        targets.rows.empty() || population.rows.empty())
      return py::none();
    py::ssize_t n = population.rows.size(), d = population.rows[0].dimension,
                t = targets.rows.size();
    if (!d || n > std::numeric_limits<int>::max() || !finite_rows(targets, d) ||
        !finite_rows(population, d))
      return py::none();
    std::vector<double> fits, target_fits;
    if (!fitness(population, fits) || !fitness(targets, target_fits))
      return py::none();
    double scale = .5, cr = .5, fraction = .2;
    if (!parameter.is_none()) {
      auto p = py::array_t<double, py::array::c_style |
                                       py::array::forcecast>::ensure(parameter);
      if (!p || p.size() != 3)
        return py::none();
      scale = p.data()[0];
      cr = p.data()[1];
      fraction = p.data()[2];
    }
    for (double v : {scale, cr, fraction})
      if (!std::isfinite(v) || v < 0 || v > 1)
        return py::none();
    auto raw = local_frame ? frame_targets
                           : state.attr("get")("_target_population_indices",
                                               py::none());
    bool match_first =
        wrapper
            ? (!raw.is_none() &&
               py::cast<py::ssize_t>(np.attr("asarray")(raw).attr("size")) == t)
            : py::cast<bool>(
                  state.attr("get")("_fastga_match_first_target", false));
    std::optional<
        py::array_t<int64_t, py::array::c_style | py::array::forcecast>>
        indices;
    if (!match_first && !raw.is_none())
      indices.emplace(
          py::array_t<int64_t,
                      py::array::c_style | py::array::forcecast>::ensure(raw));
    bool valid_indices = indices && *indices && indices->size() == t;
    std::vector<int> elite;const std::vector<int> *selected_elite=&elite;
    py::object prepared = py::none();
    auto elite_owner=direct->elite_record;
    bool elite_hit=false;
    if(direct->preparations) {
      if(direct->recording)++direct->elite_requests;
      elite_hit=elite_owner && elite_owner->fitness.size()==fits.size()
          && std::memcmp(elite_owner->fitness.data(),fits.data(),fits.size()*sizeof(double))==0;
      if(elite_hit) {selected_elite=&elite_owner->indices; if(direct->recording)++direct->elite_hits;}
    } else if (!preparation_store.is_none())
      prepared = elite_preparation(context, population, fits);
    if (!prepared.is_none()) {
      auto indices = py::cast<py::array_t<int64_t>>(prepared);
      elite.reserve(n);for (py::ssize_t i=0;i<n;++i)elite.push_back(static_cast<int>(indices.data()[i]));
    } else if (!elite_hit) {
      elite.reserve(n);for(py::ssize_t i=0;i<n;++i)elite.push_back(i);
      std::stable_sort(elite.begin(), elite.end(),
                       [&](int a, int b) { return fits[a] < fits[b]; });
      if(direct->preparations) {
        int64_t bytes=fits.size()*(sizeof(double)+sizeof(int));
        auto distance_bytes=direct->distance ? direct->distance->bytes : 0;
        if(bytes+distance_bytes<=direct->max_bytes)
          direct->elite_record=std::make_shared<DirectNative::Elite>(DirectNative::Elite{fits,elite});
      }
      if (!preparation_store.is_none()) {
        py::array_t<double> fit_snapshot(n);
        std::copy(fits.begin(), fits.end(), fit_snapshot.mutable_data());
        py::array_t<int64_t> indices(n);
        std::copy(elite.begin(), elite.end(), indices.mutable_data());
        preparation_store.attr("remember_elite")(context, fit_snapshot, indices,
                                                py::tuple(population.sources));
      }
    }
    int count = std::max(1, static_cast<int>(std::ceil(fraction * n)));
    py::array_t<double> output({t, d});
    auto out = output.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < t; ++i)
      for (py::ssize_t j = 0; j < d; ++j)
        out(i, j) = targets.rows[i].number_at(j);
    auto rng = state["rng"];
    for (py::ssize_t i = 0; i < t; ++i) {
      int target = -1;
      if (match_first) {
        for (int k = 0; k < n; ++k) {
          if (fits[k] != target_fits[i])
            continue;
          bool equal = true;
          for (py::ssize_t j = 0; j < d; ++j)
            if (population.rows[k].number_at(j) !=
                targets.rows[i].number_at(j)) {
              equal = false;
              break;
            }
          if (equal) {
            target = k;
            break;
          }
        }
        if (target < 0)
          throw py::value_error("FastGA current-to-pbest target is not a clone "
                                "of the current population.");
      } else if (valid_indices && indices->data()[i] >= 0 &&
                 indices->data()[i] < n)
        target = indices->data()[i];
      int best = (*selected_elite)[py::cast<int>(rng.attr("integers")(0, count))];
      int first = sample_excluding(rng, n, target),
          second = sample_excluding(rng, n, target, first);
      std::vector<double> donor;donor.reserve(d);
      for (py::ssize_t j = 0; j < d; ++j) {
        double x = targets.rows[i].number_at(j);
        donor.push_back((x + scale * (population.rows[best].number_at(j) - x)) +
                   scale * (population.rows[first].number_at(j) -
                            population.rows[second].number_at(j)));
      }
      if (match_first) {
        int forced = py::cast<int>(rng.attr("integers")(0, d));
        for (py::ssize_t j = 0; j < d; ++j)
          if (j == forced || py::cast<double>(rng.attr("random")()) < cr)
            out(i, j) = donor[j];
      } else {
        auto coins = py::cast<py::array_t<double>>(rng.attr("random")(d));
        int forced = py::cast<int>(rng.attr("integers")(0, d));
        for (py::ssize_t j = 0; j < d; ++j)
          if (j == forced || coins.data()[j] < cr)
            out(i, j) = donor[j];
      }
    }
    if (wrapper) {
      state.attr("pop")("_fastga_match_first_target", py::none());
      state["_stream_invalidated_mask"] =
          np.attr("ones")(t, "dtype"_a = np.attr("bool_"));
    }
    de_rows += t;
    return py::make_tuple(output, state);
  }

  py::object bit(const std::string &name, py::object selected,
                 py::object problem, py::dict state, NativeData &data) {
    bool fastga = name.rfind("search_fastga_bit_", 0) == 0;
    std::string law = name.substr(fastga ? 18 : 11);
    if (law != "uniform" && law != "standard" && law != "conditional" &&
        law != "shifted" && law != "normal" && law != "fast" && law != "one" &&
        law != "three" && law != "five")
      return py::none();
    PopulationView view;
    NativeData::Repair repair;
    if (!data.native_integer_width || !data.repair_contract(problem, repair) ||
        !repair.discrete || !load(selected, view) || view.rows.empty())
      return py::none();
    auto d = repair.dimension;
    py::ssize_t n = view.rows.size();
    int fixed = law == "one" ? 1 : law == "three" ? 3 : law == "five" ? 5 : 0;
    if (d <= 0 || fixed > d)
      return py::none();
    auto bounds = repair.bounds.unchecked<2>();
    for (py::ssize_t j = 0; j < d; ++j)
      if (bounds(0, j) != 0 || bounds(1, j) != 1)
        return py::none();
    for (auto &row : view.rows) {
      if (!row.integer || row.dimension != d)
        return py::none();
      for (py::ssize_t j = 0; j < d; ++j)
        if (row.integer_at(j) != 0 && row.integer_at(j) != 1)
          return py::none();
    }
    int half = d / 2;
    if (law == "fast" && half && fastga && !cumulative.count(d)) {
      py::list weights;
      for (int j = 1; j <= half; ++j)
        weights.append(py::float_(double(j)).attr("__pow__")(-1.5));
      double sum = py::cast<double>(
                 py::module_::import("builtins").attr("sum")(weights)),
             value = 0.;
      auto &cdf = cumulative[d];
      for (auto w : weights) {
        value += py::cast<double>(w) / sum;
        cdf.push_back(value);
      }
    }
    if (law == "fast" && half && !fastga && !probabilities.count(d)) {
      auto values =
          np.attr("arange")(1, half + 1, "dtype"_a = np.attr("float64"));
      auto weights = np.attr("power")(values, -1.5);
      probabilities[d] = np.attr("divide")(weights, np.attr("sum")(weights));
    }
    double deviation =
        law == "normal" ? py::cast<double>(np.attr("log")(d)) : 0.;
    auto rng = state["rng"];
    py::array_t<int64_t> output({n, d});
    auto out = output.mutable_unchecked<2>();
    py::object flags = py::none();
    bool *flag_data = nullptr;
    if (fastga) {
      py::array_t<bool> mask(n);
      flag_data = mask.mutable_data();
      std::fill_n(flag_data, n, false);
      flags = std::move(mask);
    }
    for (py::ssize_t i = 0; i < n; ++i)
      for (py::ssize_t j = 0; j < d; ++j)
        out(i, j) = view.rows[i].integer_at(j);
    auto binomial = [&](double rate) -> int {
      if (!fastga)
        return py::cast<int>(rng.attr("binomial")(d, rate));
      auto coins = py::cast<py::array_t<double>>(rng.attr("random")(d));
      int count = 0;
      for (py::ssize_t j = 0; j < d; ++j)
        count += coins.data()[j] < rate;
      return count;
    };
    for (py::ssize_t i = 0; i < n; ++i) {
      int count = fixed;
      if (law == "uniform")
        count = py::cast<int>(rng.attr("integers")(0, d));
      else if (law == "standard")
        count = binomial(1. / d);
      else if (law == "conditional") {
        do {
          count = binomial(1. / d);
        } while (!count);
      } else if (law == "shifted")
        count = std::max(1, binomial(.5));
      else if (law == "normal") {
        count = static_cast<int>(std::trunc(
            py::cast<double>(rng.attr("normal")(1. / d, deviation))));
        if (count < 0 || count >= d)
          count = py::cast<int>(rng.attr("integers")(0, d));
      } else if (law == "fast") {
        double rate = 1.;
        if (fastga) {
          double trigger = py::cast<double>(rng.attr("random")());
          for (int j = 0; j < half; ++j)
            if (cumulative[d][j] >= trigger) {
              rate = double(j + 1) / d;
              break;
            }
        } else if (half)
          rate = (py::cast<int>(
                      rng.attr("choice")(half, "p"_a = probabilities[d])) +
                  1.) /
                 d;
        count = binomial(rate);
      }
      if (count) {
        auto indices = py::cast<py::array_t<int64_t>>(
            rng.attr("choice")(d, "size"_a = count, "replace"_a = false));
        for (int j = 0; j < count; ++j)
          out(i, indices.data()[j]) ^= 1;
        if (fastga)
          flag_data[i] = true;
      }
    }
    if (fastga)
      state["_stream_invalidated_mask"] = flags;
    bit_rows += n;
    return py::make_tuple(output, state);
  }
  py::object polynomial(py::object selected, py::object problem,
                        py::object parameter, py::dict state) {
    PopulationView population;
    if (!load(selected, population) || population.rows.empty())
      return py::none();
    py::ssize_t n = population.rows.size(), d = population.rows[0].dimension;
    if (!finite_rows(population, d))
      return py::none();
    auto bound =
        py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(
            py::getattr(problem, "bound", py::none()));
    if (!bound || bound.ndim() != 2 || bound.shape(0) != 2 ||
        bound.shape(1) != d)
      return py::none();
    auto b = bound.unchecked<2>();
    for (py::ssize_t j = 0; j < d; ++j) {
      double span = b(1, j) - b(0, j);
      if (!std::isfinite(b(0, j)) || !std::isfinite(b(1, j)) ||
          !std::isfinite(span) || (span != 0 && std::abs(span) < 1e-150))
        return py::none();
    }
    double probability = .1, distribution = 30.;
    if (!parameter.is_none()) {
      auto p = py::array_t<double, py::array::c_style |
                                       py::array::forcecast>::ensure(parameter);
      if (!p)
        return py::none();
      if (p.size())
        probability = p.data()[0];
      if (p.size() > 1)
        distribution = p.data()[1];
    }
    if (!std::isfinite(probability) || !std::isfinite(distribution) ||
        probability < 0 || probability > 1 || distribution < 0 ||
        distribution > 1e6)
      return py::none();
    py::array_t<double> output({n, d});
    auto dest = output.mutable_data();
    for (py::ssize_t i = 0; i < n; ++i)
      for (py::ssize_t j = 0; j < d; ++j)
        dest[i * d + j] = population.rows[i].number_at(j);
    auto rng = state["rng"];
    auto sites =
        py::cast<py::array_t<double>>(rng.attr("random")(py::make_tuple(n, d)));
    auto coins =
        py::cast<py::array_t<double>>(rng.attr("random")(py::make_tuple(n, d)));
    for (bool lower : {true, false}) {
      std::vector<py::ssize_t> positions;
      for (py::ssize_t k = 0; k < n * d; ++k)
        if (sites.data()[k] < probability && b(1, k % d) > b(0, k % d) &&
            (lower ? coins.data()[k] <= .5 : coins.data()[k] > .5))
          positions.push_back(k);
      if (positions.empty())
        continue;
      py::array_t<double> base(positions.size());
      for (size_t i = 0; i < positions.size(); ++i) {
        auto k = positions[i], j = k % d;
        base.mutable_data()[i] =
            1. - (lower ? dest[k] - b(0, j) : b(1, j) - dest[k]) /
                     (b(1, j) - b(0, j));
      }
      // Keep NumPy's power implementation: scalar libm pow need not round
      // identically to the installed NumPy vector kernel.
      auto powered = py::cast<py::array_t<double>>(
          np.attr("power")(base, distribution + 1.));
      for (size_t i = 0; i < positions.size(); ++i) {
        double u = coins.data()[positions[i]], p = powered.data()[i];
        base.mutable_data()[i] = lower ? 2. * u + (1. - 2. * u) * p
                                       : 2. * (1. - u) + (2. * (u - .5)) * p;
      }
      powered = py::cast<py::array_t<double>>(
          np.attr("power")(base, 1. / (distribution + 1.)));
      for (size_t i = 0; i < positions.size(); ++i) {
        auto k = positions[i], j = k % d;
        dest[k] =
            dest[k] + (b(1, j) - b(0, j)) * (lower ? powered.data()[i] - 1.
                                                   : 1. - powered.data()[i]);
      }
    }
    polynomial_rows += n;
    return py::make_tuple(output, state);
  }
  py::object execute(const std::string &name, py::object selected,
                     py::object problem, py::object parameter, py::dict state,
                     NativeData &data, const std::string &kernel,
                     bool local_frame = false,
                     py::object frame_population = py::none(),
                     py::object frame_targets = py::none()) {
    AOL_REGION_TIME(2);
    if (kernel.empty() || !data.native_integer_width || !ordinary_generator(state,data.generator_type))
      return py::none();
    if (kernel == "polynomial")
      return polynomial(selected, problem, parameter, state);
    if (kernel == "de" || kernel == "fastga_de")
      return de(selected, parameter, state, kernel == "fastga_de", local_frame,
                frame_population, frame_targets);
    if (kernel == "bit")
      return bit(name, selected, problem, state, data);
    return py::none();
  }
};

#pragma once

#include <algorithm>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <pybind11/numpy.h>

namespace native_components {

inline double fitness_value(double objective, double violation) {
  return violation > 0. ? violation : objective;
}

inline double objective_score(double fitness, double violation) {
  return violation > 0. ? std::numeric_limits<double>::infinity() : fitness;
}

inline std::pair<bool, double> feasibility_key(double fitness, double violation) {
  return violation > 0. ? std::make_pair(true, violation)
                        : std::make_pair(false, fitness);
}

struct FitnessValues {
  pybind11::array_t<double> values;
  pybind11::array_t<double> violations;

  double violation(pybind11::ssize_t index) const {
    return violations && !violations.is_none() ? violations.data()[index] : 0.;
  }

  bool has_violations() const {
    return violations && !violations.is_none() &&
        std::any_of(violations.data(), violations.data() + violations.size(),
                    [](double value) { return value > 0.; });
  }

  auto key(pybind11::ssize_t index) const {
    return feasibility_key(values.data()[index], violation(index));
  }

  pybind11::array_t<double> selection_scores() const {
    if (!has_violations()) return values;
    if (std::all_of(violations.data(), violations.data() + violations.size(),
                    [](double value) { return value > 0.; })) return violations;

    std::vector<pybind11::ssize_t> order(values.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](auto left, auto right) {
      return key(left) < key(right);
    });
    pybind11::array_t<double> ranked(values.size());
    double rank = 0.;
    ranked.mutable_data()[order.front()] = rank;
    for (size_t i = 1; i < order.size(); ++i) {
      if (key(order[i - 1]) != key(order[i])) ++rank;
      ranked.mutable_data()[order[i]] = rank;
    }
    return ranked;
  }
};

// DE clone matching and PSO compare two populations on one common scale.
inline auto shared_selection_scores(const FitnessValues &left, const FitnessValues &right) {
  if (!left.has_violations() && !right.has_violations())
    return std::make_pair(left.values, right.values);
  const auto n = left.values.size(), m = right.values.size();
  pybind11::array_t<double> values(n + m), violations(n + m);
  std::copy_n(left.values.data(), n, values.mutable_data());
  std::copy_n(right.values.data(), m, values.mutable_data() + n);
  for (pybind11::ssize_t i = 0; i < n; ++i) violations.mutable_data()[i] = left.violation(i);
  for (pybind11::ssize_t i = 0; i < m; ++i) violations.mutable_data()[n + i] = right.violation(i);
  auto ranked = FitnessValues{values, violations}.selection_scores();
  pybind11::array_t<double> a(n), b(m);
  std::copy_n(ranked.data(), n, a.mutable_data());
  std::copy_n(ranked.data() + n, m, b.mutable_data());
  return std::make_pair(a, b);
}

}  // namespace native_components

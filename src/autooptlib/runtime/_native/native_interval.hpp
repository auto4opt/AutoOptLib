#pragma once
#include "execution_services.hpp"
#include <optional>
#include <thread>
#include <pybind11/pybind11.h>
namespace native_components {
// Owners, shapes and output arrays must be prepared before entering. This
// guard deliberately exposes no Python operation. Small loops stay on-thread
// to avoid a GIL handoff dominating their arithmetic.
struct NumericInterval {
  static constexpr size_t release_threshold=4096;
  RegionScope timing{Region::Numeric};
  std::optional<pybind11::gil_scoped_release> release;
  explicit NumericInterval(size_t elements){if(elements>=release_threshold)release.emplace();}
};
inline void cooperative_boundary(){pybind11::gil_scoped_release release;std::this_thread::yield();}
}

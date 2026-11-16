// Optional exclusive native-region timing. Compiled out of production builds.
#pragma once
#include <pybind11/pybind11.h>
#ifdef AOL_PROFILE_REGIONS
#include <array>
#include <chrono>
struct RegionTimer {
  using Clock = std::chrono::steady_clock;
  static inline thread_local RegionTimer *active = nullptr;
  static inline thread_local std::array<double, 8> seconds{};
  static inline thread_local std::array<uint64_t, 8> calls{};
  RegionTimer *parent;
  Clock::time_point start;
  double nested = 0;
  int category;
  explicit RegionTimer(int id)
      : parent(active), start(Clock::now()), category(id) {
    active = this;
  }
  ~RegionTimer() {
    double elapsed =
        std::chrono::duration<double>(Clock::now() - start).count();
    seconds[category] += elapsed - nested;
    ++calls[category];
    active = parent;
    if (parent)
      parent->nested += elapsed;
  }
};
#define AOL_REGION_TIME(id) RegionTimer region_timer(id)
#else
#define AOL_REGION_TIME(id) ((void)0)
#endif
inline pybind11::dict region_profile() {
  pybind11::dict result;
#ifdef AOL_PROFILE_REGIONS
  const char *names[] = {"event_control",    "selection",
                         "numeric_operator", "repair_clone_materialize",
                         "objective_policy", "replacement",
                         "sink_grouping",    "population_view"};
  for (int i = 0; i < 8; ++i)
    result[names[i]] =
        pybind11::make_tuple(RegionTimer::calls[i], RegionTimer::seconds[i]);
#endif
  return result;
}

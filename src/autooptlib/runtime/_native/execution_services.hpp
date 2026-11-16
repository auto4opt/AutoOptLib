#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <memory>
#include <algorithm>
#include <type_traits>

namespace native_components {
inline thread_local unsigned native_floating_depth=0;

// Scratch is never exported as an ndarray or callback input. Every live lease
// has a distinct slot, including nested calls and independent reentrant runs.
template<class T> struct ScratchPool {
  struct Slot {std::vector<T> values;bool busy=false;size_t previous_capacity=0;};
  std::vector<std::unique_ptr<Slot>> slots;
  std::atomic<uint64_t> growths{0},reuses{0};
  bool recording=false;
  std::atomic<size_t> capacity_total{0};
  static size_t bytes(size_t capacity){if constexpr(std::is_same_v<T,bool>)return (capacity+7)/8;else return capacity*sizeof(T);}
  Slot *acquire(size_t count) {
    Slot *selected=nullptr;
    for(auto &slot:slots)if(!slot->busy){selected=slot.get();break;}
    if(!selected){slots.push_back(std::make_unique<Slot>());selected=slots.back().get();}
    selected->previous_capacity=selected->values.capacity();
    selected->values.clear();selected->values.resize(count);selected->busy=true;return selected;
  }
  void release(Slot *slot)noexcept {
    auto capacity=slot->values.capacity();
    if(capacity>slot->previous_capacity){if(recording)++growths;capacity_total.fetch_add(bytes(capacity)-bytes(slot->previous_capacity));}else if(recording)++reuses;
    slot->busy=false;
  }
  size_t capacity_bytes()const {return capacity_total.load();}
};
struct RunWorkspace {
  ScratchPool<double> numbers;
  ScratchPool<int64_t> indices;
  ScratchPool<bool> masks;
  // One immutable probability table per run workspace; never population/RNG state.
  std::vector<double> bit_weights,bit_cdf;
  int64_t bit_dimension=-1;
  bool bit_fastga=false;
  uintptr_t bit_math=0;
  uint64_t bit_environment=0,bit_table_builds=0,bit_table_hits=0;
  int64_t normal_dimension=-1;uintptr_t normal_math=0;uint64_t normal_environment=0;
  std::array<double,2> normal_scales{};
  template<class T> ScratchPool<T>&pool() {
    if constexpr(std::is_same_v<T,double>)return numbers;
    else if constexpr(std::is_same_v<T,int64_t>)return indices;
    else return masks;
  }
};
inline thread_local RunWorkspace *active_workspace=nullptr;
struct WorkspaceContext {
  RunWorkspace *previous;
  explicit WorkspaceContext(RunWorkspace *workspace):previous(active_workspace){active_workspace=workspace;}
  ~WorkspaceContext(){active_workspace=previous;}
};
template<class T> class ScratchVector {
  using Slot=typename ScratchPool<T>::Slot;
  Slot *slot_=nullptr;
  ScratchPool<T> *pool_=active_workspace?&active_workspace->pool<T>():nullptr;
  std::vector<T> local_;
public:
  std::vector<T> &values;
  explicit ScratchVector(size_t count=0):slot_(active_workspace?active_workspace->pool<T>().acquire(count):nullptr),
      local_(slot_?0:count),values(slot_?slot_->values:local_){}
  ScratchVector(size_t count,T value):ScratchVector(){values.resize(count,value);}
  ~ScratchVector(){if(slot_)pool_->release(slot_);}
  ScratchVector(const ScratchVector&)=delete;
  ScratchVector&operator=(const ScratchVector&)=delete;
};

// This header has no Python dependency. A controller can be inspected or
// cancelled from a different thread without touching run-owned state.
struct ProgressValue {
  int64_t evaluations=0,events=0,generation=0,completed_runs=0;
  bool running=false,cancelled=false;
};
class RunControl {
  std::atomic<uint64_t> revision_{0};
  std::atomic<int64_t> evaluations_{0},events_{0},generation_{0},completed_{0};
  std::atomic<bool> running_{false},cancelled_{false};
public:
  void cancel() noexcept {cancelled_.store(true,std::memory_order_release);}
  bool cancelled()const noexcept{return cancelled_.load(std::memory_order_acquire);}
  void reset() {
    if(running_.load(std::memory_order_acquire))throw std::runtime_error("Cannot reset a running session");
    cancelled_.store(false,std::memory_order_release);publish(0,0,0);completed_.store(0);
  }
  void enter() {
    bool expected=false;
    if(!running_.compare_exchange_strong(expected,true,std::memory_order_acq_rel))
      throw std::runtime_error("RunSession is already executing; use an independent session for reentry");
  }
  void leave() noexcept {running_.store(false,std::memory_order_release);}
  void publish(int64_t evaluations,int64_t events,int64_t generation) noexcept {
    revision_.fetch_add(1,std::memory_order_acq_rel);
    evaluations_.store(evaluations,std::memory_order_relaxed);
    events_.store(events,std::memory_order_relaxed);
    generation_.store(generation,std::memory_order_relaxed);
    revision_.fetch_add(1,std::memory_order_release);
  }
  void completed_run() noexcept {completed_.fetch_add(1,std::memory_order_release);}
  ProgressValue progress()const noexcept {
    ProgressValue out;
    for(;;) {
      auto before=revision_.load(std::memory_order_acquire);if(before&1)continue;
      out.evaluations=evaluations_.load(std::memory_order_relaxed);
      out.events=events_.load(std::memory_order_relaxed);
      out.generation=generation_.load(std::memory_order_relaxed);
      if(before==revision_.load(std::memory_order_acquire))break;
    }
    out.completed_runs=completed_.load(std::memory_order_acquire);
    out.running=running_.load(std::memory_order_acquire);out.cancelled=cancelled();return out;
  }
};
struct RunningScope {
  RunControl &control;
  explicit RunningScope(RunControl &value):control(value){control.enter();}
  ~RunningScope(){control.leave();}
};

enum class Region:size_t {Run,Initialize,Plan,Generation,Component,Numeric,Repair,Callback,Pack,Snapshot,Checkpoint,Objective,Count};
inline constexpr std::array<const char *,size_t(Region::Count)> region_names={
  "run","initialize","plan","generation","component","numeric","repair","callback","pack","snapshot","checkpoint","objective"};
struct RegionValue {uint64_t calls=0,inclusive_ns=0,exclusive_ns=0;};
class NativeProfile {
  struct Counter {std::atomic<uint64_t> calls{0},inclusive{0},exclusive{0};};
  std::array<Counter,size_t(Region::Count)> values_{};
public:
  bool enabled=false;
  std::atomic<uint64_t> snapshot_copy_bytes{0};
  struct Frame {uint64_t children=0;};
  // Only the executing thread writes the nesting stack; snapshots read atomic
  // aggregates and never inspect a stack which may be inside a GIL-free kernel.
  std::vector<Frame *> stack;
  static uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void add(Region region,uint64_t inclusive,uint64_t exclusive)noexcept {
    auto &value=values_[size_t(region)];value.calls.fetch_add(1,std::memory_order_relaxed);
    value.inclusive.fetch_add(inclusive,std::memory_order_relaxed);
    value.exclusive.fetch_add(exclusive,std::memory_order_relaxed);
  }
  RegionValue read(Region region)const noexcept {
    auto &value=values_[size_t(region)];return {value.calls.load(),value.inclusive.load(),value.exclusive.load()};
  }
};
inline thread_local NativeProfile *active_native_profile=nullptr;
inline void record_snapshot_copy(size_t bytes){if(active_native_profile&&active_native_profile->enabled)active_native_profile->snapshot_copy_bytes.fetch_add(bytes,std::memory_order_relaxed);}
struct ProfileContext {
  NativeProfile *previous;
  explicit ProfileContext(NativeProfile *profile):previous(active_native_profile){active_native_profile=profile;}
  ~ProfileContext(){active_native_profile=previous;}
};
class RegionScope {
  NativeProfile *profile_;Region region_;uint64_t start_=0;NativeProfile::Frame frame_;
public:
  explicit RegionScope(Region region,NativeProfile *profile=active_native_profile)
      :profile_(profile&&profile->enabled?profile:nullptr),region_(region) {
    if(profile_){start_=NativeProfile::now();profile_->stack.push_back(&frame_);}
  }
  ~RegionScope() {
    if(!profile_)return;
    auto elapsed=NativeProfile::now()-start_;profile_->stack.pop_back();
    profile_->add(region_,elapsed,elapsed>=frame_.children?elapsed-frame_.children:0);
    if(!profile_->stack.empty())profile_->stack.back()->children+=elapsed;
  }
};
} // namespace native_components

// Prepared schedules for the existing invocation. No per-primitive Python
// interpreter, diagnostic receipt, or metric work in ordinary execution.
#pragma once
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>
#include "dependencies.hpp"

namespace cpu_program {
namespace py = pybind11;

struct Instruction {
  py::object source, binding, successor;
  std::vector<int> inputs, outputs, dead, prefetch;
  std::string preparation;
  bool effect, before;
  explicit Instruction(py::handle op)
      : source(py::reinterpret_borrow<py::object>(op)),
        binding(op.attr("binding")), successor(op.attr("successor")),
        inputs(py::cast<std::vector<int>>(op.attr("inputs"))),
        outputs(py::cast<std::vector<int>>(op.attr("outputs"))),
        dead(py::cast<std::vector<int>>(op.attr("dead"))),
        prefetch(py::cast<std::vector<int>>(op.attr("prefetch"))),
        preparation(py::cast<std::string>(op.attr("preparation"))),
        effect(py::len(op.attr("effects")) != 0),
        before(py::len(op.attr("reads")) != 0 || effect) {}
};
struct Region {
  py::object source;
  int entry, exit;
  bool kernel;
  explicit Region(py::handle r)
      : source(py::reinterpret_borrow<py::object>(r)),
        entry(py::cast<int>(r.attr("entry"))), exit(py::cast<int>(r.attr("exit"))),
        kernel(py::len(r.attr("kernel")) != 0) {}
};

struct Program {
  std::vector<Instruction> instructions;
  std::vector<Region> regions;
  std::unordered_map<int, size_t> entries;
  py::object unpack, no_proof, selected_type, array_type;
  explicit Program(py::object plan) {
    auto guard = plan.attr("specification").attr("dependencies");
    if (guard.attr("native_check").is_none())
      guard.attr("native_check") = py::cast(std::make_shared<DependencyProof>(guard));
    for (auto op : plan.attr("instructions")) instructions.emplace_back(op);
    for (auto r : plan.attr("regions")) {
      entries.emplace(py::cast<int>(r.attr("entry")), regions.size());
      regions.emplace_back(r);
    }
    auto frame = py::module_::import("autooptlib.runtime.frame");
    unpack = frame.attr("unpack_outputs");
    no_proof = frame.attr("_NO_PROOF");
    selected_type = py::module_::import("autooptlib.runtime.bindings").attr("SelectedRows");
    array_type = py::module_::import("numpy").attr("ndarray");
  }

  bool private_scalar_or_array(py::handle value) const {
    if (value.is_none() || PyBool_Check(value.ptr()) || PyLong_CheckExact(value.ptr())
        || PyFloat_CheckExact(value.ptr()) || PyUnicode_CheckExact(value.ptr())) return true;
    if (reinterpret_cast<PyObject *>(Py_TYPE(value.ptr())) != array_type.ptr()) return false;
    auto a = py::reinterpret_borrow<py::array>(value);
    return a.owndata()
           && std::string("biuf").find(a.dtype().kind()) != std::string::npos;
  }

  void execute(py::object invocation) const {
    auto frame = invocation.attr("frame");
    auto bindings = invocation.attr("bindings");
    auto preparations = frame.attr("preparations");
    bool prepare = py::cast<bool>(preparations.attr("enabled"));
    auto workspace = frame.attr("workspace");
    py::list slots = invocation.attr("slots"), retained = invocation.attr("retained");
    py::dict callables = invocation.attr("callables");
    py::dict tokens = invocation.attr("preparation_tokens");
    py::object pc = py::int_(0);
    while (!pc.is_none()) {
      const auto &region = regions[entries.at(py::cast<int>(pc))];
      auto reason = invocation.attr("admit_region")(region.source, workspace, false);
      if (!reason.is_none()) bindings.attr("unknown")("region-refusal");
      frame.attr("position") = invocation.attr("cursor");
      if (reason.is_none() && region.kernel
          && !invocation.attr("native_region")(region.source, false).is_none()) {
        pc = region.source.attr("continuation");
        continue;
      }
      while (!pc.is_none() && py::cast<int>(pc) >= region.entry
             && py::cast<int>(pc) < region.exit) {
        int index = py::cast<int>(pc);
        const auto &op = instructions[index];
        auto cursor = invocation.attr("cursor");
        cursor.attr("operation") = pc;
        frame.attr("position") = cursor;
        bindings.attr("current_writes") = op.source.attr("writes");
        bindings.attr("current_requirements") = op.source.attr("requirements");
        for (int target : op.prefetch)
          callables[py::int_(target)] = instructions[target].binding.attr("resolve")()[py::int_(0)];
        py::object function = callables.contains(pc)
            ? callables.attr("pop")(pc)
            : op.binding.attr("resolve")()[py::int_(0)];
        if (!py::cast<bool>(function.attr("trusted")())) {
          invocation.attr("unknown_helper") = true;
          bindings.attr("unknown")("implementation-dependency");
        }
        py::tuple operands(op.inputs.size());
        for (size_t j = 0; j < op.inputs.size(); ++j) {
          int slot = op.inputs[j];
          py::object value = slots[slot];
          if (reinterpret_cast<PyObject *>(Py_TYPE(value.ptr())) == selected_type.ptr()) {
            value = value.attr("reference_matrix")();
            slots[slot] = value;
          }
          operands[j] = value;
        }
        if (op.before) bindings.attr("before")(op.source, operands);
        py::object result = prepare && !op.preparation.empty()
            ? preparations.attr("invoke")(invocation, op.source, function, operands)
            : function(*operands);
        if (op.outputs.size() == 1) slots[op.outputs[0]] = result;
        else {
          py::tuple values = unpack(result, op.outputs.size());
          for (size_t j = 0; j < op.outputs.size(); ++j) slots[op.outputs[j]] = values[j];
        }
        if (op.effect || !tokens.contains(py::str(op.preparation))) {
          invocation.attr("_dependencies_trusted") = py::none();
          invocation.attr("_preparation_environment") = no_proof;
          bindings.attr("source_views").attr("clear")();
        }
        cursor.attr("effect") = py::len(bindings.attr("writes"));
        for (int slot : op.dead) {
          py::object value = slots[slot];
          if (!private_scalar_or_array(value)) retained.append(value);
          slots[slot] = py::none();
        }
        if (!op.successor.is_none() && py::cast<int>(op.successor) != index + 1)
          bindings.attr("unknown")("control-join-or-backedge");
        pc = op.successor;
        cursor.attr("operation") = pc;
      }
      if (!pc.is(region.source.attr("continuation")) && !pc.equal(region.source.attr("continuation")))
        bindings.attr("unknown")("control-join-or-backedge");
    }
  }
};
} // namespace cpu_program

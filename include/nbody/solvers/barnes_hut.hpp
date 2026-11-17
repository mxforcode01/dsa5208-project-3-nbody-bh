#pragma once

#include "nbody/core/device.hpp"
#include "nbody/core/force_loop.hpp"
#include "nbody/core/physics.hpp"
#include "nbody/solvers/force_solver.hpp"
#include "nbody/tree/octree.hpp"

namespace nbody {

// V1: Barnes-Hut. Rebuilds the octree from the current positions on every call,
// then walks it once per target. Targets are processed in leaf order for cache
// locality; each target is independent, so results do not depend on the thread
// count or the OpenMP schedule. threads == 1 is the serial V1 baseline used for
// all speedup figures.
class BarnesHutSolver final : public ForceSolver {
 public:
  BarnesHutSolver(PhysicsParams phys, TreeParams tree, int threads = 1, LoopSchedule sched = {},
                  Device device = Device::CPU, bool allow_host_fallback = false)
      : phys_(phys), tp_(tree), threads_(threads), sched_(sched), device_(device),
        allow_host_fallback_(allow_host_fallback) {}
  void compute_accelerations(ParticleSet& p, ForceStats& stats) override;
  std::string name() const override { return "bh"; }
  const FlatTree& last_tree() const { return tree_; }

 private:
  PhysicsParams phys_;
  TreeParams tp_;
  int threads_;
  LoopSchedule sched_;
  Device device_;
  bool allow_host_fallback_;
  FlatTree tree_;
};

}  // namespace nbody

#pragma once

#include "nbody/core/physics.hpp"
#include "nbody/solvers/force_solver.hpp"

namespace nbody {

// V0: O(N^2) pairwise summation, the accuracy reference for Barnes-Hut.
// Each target sums over all sources in index order (no Newton's-third-law
// pair sharing), so results are bitwise identical for any thread count.
// threads == 1 is the serial V0; threads > 1 is only a faster way to
// produce large reference solutions and is never used as a speedup baseline.
class DirectSumSolver final : public ForceSolver {
 public:
  explicit DirectSumSolver(PhysicsParams phys, int threads = 1) : phys_(phys), threads_(threads) {}
  void compute_accelerations(ParticleSet& p, ForceStats& stats) override;
  std::string name() const override { return "direct"; }

 private:
  PhysicsParams phys_;
  int threads_;
};

}  // namespace nbody

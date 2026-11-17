#pragma once

#include <functional>
#include <vector>

#include "nbody/core/particles.hpp"
#include "nbody/sim/diagnostics.hpp"
#include "nbody/solvers/force_solver.hpp"

namespace nbody {

struct SimParams {
  int steps = 10;
  double dt = 1e-3;
  int diag_every = 0;  // 0: first and last step only
  int threads = 1;     // OpenMP threads for kick/drift
  // Optional: turns per-rank diagnostics into global ones (MPI runs). Keeps the loop MPI-free.
  std::function<void(Diagnostics&)> reduce_diagnostics;
};

// Accumulated over the whole run (seconds).
struct StepTimings {
  double force = 0.0;
  double integrate = 0.0;
  double diagnostics = 0.0;
};

struct DiagRecord {
  int step = 0;
  double time = 0.0;
  Diagnostics d;
};

struct RunResult {
  StepTimings timings;
  double total_time = 0.0;  // initial force evaluation + all steps (excludes ICs and I/O)
  ForceStats stats;
  std::vector<DiagRecord> diag;  // always contains step 0 and the last step
};

// The kick-drift-kick time-stepping loop used by every implementation version.
class Simulation {
 public:
  Simulation(SimParams params, ForceSolver& solver) : params_(params), solver_(solver) {}
  RunResult run(ParticleSet& p) const;

 private:
  SimParams params_;
  ForceSolver& solver_;
};

}  // namespace nbody

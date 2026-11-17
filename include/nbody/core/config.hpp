#pragma once

#include <cstdint>
#include <string>

#include "nbody/core/comm_mode.hpp"
#include "nbody/core/device.hpp"
#include "nbody/core/force_loop.hpp"
#include "nbody/core/initial_conditions.hpp"
#include "nbody/core/physics.hpp"
#include "nbody/tree/octree.hpp"

namespace nbody {

// Run configuration shared by all executables.
struct RunConfig {
  std::uint64_t n = 1000;
  std::uint64_t seed = 42;
  Distribution dist = Distribution::Uniform;
  double scale = 1.0;
  int steps = 10;
  double dt = 1e-3;
  PhysicsParams phys;
  std::string solver = "direct";  // direct (V0) | bh (V1) | dist-bh (V2, nbody_mpi only)
  TreeParams tree;                // theta, leaf capacity, max depth, build strategy
  int rebalance_every = 1;        // MPI: recompute the decomposition every k force evaluations
  CommMode comm = CommMode::Blocking;  // MPI: LET exchange strategy
  int overlap_chunks = 32;        // MPI overlap: local work is split into this many chunks
  int let_boxes = 64;             // MPI: receiver region sent to senders as up to this many tight boxes
  bool shm = false;               // MPI: share trees inside a node through shared-memory windows
  int shm_group_size = 0;         // MPI shm: 0 = whole node, K = groups of K ranks (emulated nodes)
  Device device = Device::CPU;    // Barnes-Hut tree walks: cpu | gpu (OpenMP target offload)
  bool allow_host_fallback = false;  // gpu: accept target regions that ran on the host (testing only)
  int threads = 1;                // OpenMP threads for the force loop (1 = serial)
  LoopSchedule sched;             // OpenMP schedule of the Barnes-Hut force loops
  int diag_every = 0;             // 0: diagnostics at first and last step only
  std::string csv_path;           // append one summary row per run
  std::string diag_csv_path;      // per-diagnostic-step rows
  std::string snapshot_path;      // binary snapshot of the final state
  bool show_help = false;
};

// Parses "--key value" pairs on top of `defaults`. Throws std::invalid_argument on
// unknown keys, missing values or out-of-range values.
RunConfig parse_args(int argc, const char* const* argv, const RunConfig& defaults = RunConfig{});
std::string usage(const std::string& program);

}  // namespace nbody

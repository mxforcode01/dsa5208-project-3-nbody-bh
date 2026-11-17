#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "nbody/core/particles.hpp"

namespace nbody {

// Machine-independent work counters and solver-internal timings, accumulated over a
// run. In MPI runs every field is the value of the calling rank.
struct ForceStats {
  std::uint64_t pp = 0;  // particle-particle interactions
  std::uint64_t pc = 0;  // particle-cell (multipole) interactions; 0 for direct sum
  std::uint64_t pruned_violations = 0;  // opened a PRUNED remote node (must be 0)
  double mass_coverage_err = 0.0;       // max_i |mass seen by i - (M - m_i)| / M (0 = no source lost or doubled)
  double tree_build_time = 0.0;         // seconds (all timings); includes the two below
  double tree_keys_time = 0.0;          // Morton key computation (morton build only)
  double tree_sort_time = 0.0;          // key sort (morton build only)
  double traversal_time = 0.0;          // local + remote tree walks
  std::size_t tree_nodes = 0;           // nodes in the most recent (local) tree
  int tree_depth = 0;                   // deepest level seen in any tree

  // OpenMP balance of the force loops, summed over all loops of the run:
  // thread imbalance = loop_busy_max / loop_busy_mean, work imbalance = loop_work_max / loop_work_mean.
  double loop_busy_max = 0.0, loop_busy_mean = 0.0;
  double loop_work_max = 0.0, loop_work_mean = 0.0;
  double thread_imbalance() const { return loop_busy_mean > 0.0 ? loop_busy_max / loop_busy_mean : 1.0; }
  double thread_work_imbalance() const { return loop_work_mean > 0.0 ? loop_work_max / loop_work_mean : 1.0; }

  // Distributed solver only.
  double decomposition_time = 0.0;
  double migration_time = 0.0;
  double let_build_time = 0.0;
  double let_exchange_time = 0.0;  // bounds/size collectives + posting (blocking: the whole exchange)
  double let_wait_time = 0.0;      // inside MPI_Waitall / MPI_Testsome (blocking: inside MPI_Alltoallv)
  double force_local_time = 0.0;
  double force_remote_time = 0.0;
  std::uint64_t let_bytes_sent = 0;
  std::uint64_t let_bytes_received = 0;  // remote-tree copies this rank had to hold
  // GPU offload: what executed the walks, and where the time went.
  int executed = 0;              // nbody::Executed of the last evaluation: 0 cpu, 1 gpu, 2 host-fallback
  int num_devices = 0;           // omp_get_num_devices() seen by the last evaluation
  int device_id = -1;            // GPU used by this rank (-1 = none)
  double gpu_h2d_time = 0.0;
  double gpu_kernel_time = 0.0;
  double gpu_d2h_time = 0.0;
  double gpu_h2d_bytes = 0.0;
  // Shared-memory windows.
  double shm_publish_time = 0.0;         // copy into the window + synchronisation
  double shm_sync_time = 0.0;            // Allreduce + Win_sync + Barrier part of it
  std::uint64_t shm_segment_bytes = 0;   // current capacity of this rank's segment
  int shm_reallocations = 0;
  std::uint64_t let_nodes_sent = 0;
  std::uint64_t particles_migrated = 0;  // particles sent to another rank
};

// Every force method (direct sum, serial/distributed Barnes-Hut, GPU) fills
// p.ax, p.ay, p.az, p.pot and p.work for all particles in p. A distributed solver
// may also change which particles p holds (migration).
class ForceSolver {
 public:
  virtual ~ForceSolver() = default;
  virtual void compute_accelerations(ParticleSet& p, ForceStats& stats) = 0;
  virtual std::string name() const = 0;
};

}  // namespace nbody

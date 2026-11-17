#pragma once

#include <mpi.h>

#include <memory>

#include "nbody/core/comm_mode.hpp"
#include "nbody/core/device.hpp"
#include "nbody/core/force_loop.hpp"
#include "nbody/core/physics.hpp"
#include "nbody/mpi/decomposition.hpp"
#include "nbody/solvers/force_solver.hpp"
#include "nbody/tree/octree.hpp"

namespace nbody {

struct DistParams {
  int rebalance_every = 1;              // recompute the decomposition every k force evaluations
  CommMode comm = CommMode::Blocking;   // LET exchange strategy
  int overlap_chunks = 32;              // Overlap: local force loop split into this many chunks
  int let_boxes = 64;                   // receiver region = up to this many tight boxes (1 = single bounding box)
  bool shm = false;                     // V6: share local trees inside a node via MPI windows
  int shm_group_size = 0;               // 0 = whole node; K = groups of K ranks (emulates nodes)
  Device device = Device::CPU;          // V7: where the tree walks run
  bool allow_host_fallback = false;     // V7: accept target regions that ran on the host (testing)
};

class ShmTreeWindows;

// V2: distributed Barnes-Hut (MPI + OpenMP). Each call:
//   1. global cube (Allreduce); every `rebalance_every` calls a new work-weighted
//      Morton decomposition (exact splitters by parallel bisection)
//   2. migrate particles to their owners (Alltoall + Alltoallv)
//   3. build the rank-local octree in the global cube
//   4. exchange Locally Essential Trees (blocking | p2p | overlap)
//   5. OpenMP force loops (schedule configurable): walk the local tree, then, in a second
//      loop, every received LET. With Overlap, the local loop runs in chunks while the
//      LET messages are in flight and the master thread calls MPI_Testsome between chunks.
// Every mode visits the same trees in the same order per target -> identical forces.
// No rank ever holds the global tree. All MPI calls happen outside OpenMP regions.
class DistributedBHSolver final : public ForceSolver {
 public:
  DistributedBHSolver(MPI_Comm comm, PhysicsParams phys, TreeParams tree, DistParams dist, int threads = 1,
                      LoopSchedule sched = {});
  ~DistributedBHSolver() override;  // collective when shared-memory windows are in use
  void compute_accelerations(ParticleSet& p, ForceStats& stats) override;
  std::string name() const override { return "dist-bh"; }

  const Decomposition& decomposition() const { return decomp_; }
  const FlatTree& local_tree() const { return tree_; }

 private:
  MPI_Comm comm_;
  PhysicsParams phys_;
  TreeParams tp_;
  DistParams dp_;
  int threads_;
  LoopSchedule sched_;
  long long calls_ = 0;
  int gpu_node_rank_ = -1;  // this rank's index on its node, found on the first GPU call
  Decomposition decomp_;
  FlatTree tree_;
  std::unique_ptr<ShmTreeWindows> shm_;  // created on the first call when dp_.shm
};

}  // namespace nbody

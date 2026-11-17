#include "nbody/mpi/dist_bh.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nbody/core/timer.hpp"
#include "nbody/gpu/offload_walk.hpp"
#include "nbody/mpi/let.hpp"
#include "nbody/mpi/mpi_env.hpp"
#include "nbody/mpi/particle_comm.hpp"
#include "nbody/mpi/shm_trees.hpp"
#include "nbody/tree/traversal.hpp"

namespace nbody {

DistributedBHSolver::DistributedBHSolver(MPI_Comm comm, PhysicsParams phys, TreeParams tree, DistParams dist,
                                         int threads, LoopSchedule sched)
    : comm_(comm), phys_(phys), tp_(tree), dp_(dist), threads_(threads), sched_(sched) {}

DistributedBHSolver::~DistributedBHSolver() = default;

void DistributedBHSolver::compute_accelerations(ParticleSet& p, ForceStats& stats) {
  // 1. Decomposition.
  double t0 = wall_time();
  const Cube cube = global_cube(comm_, p);
  if (!decomp_.valid() || calls_ % dp_.rebalance_every == 0) decomp_.compute(comm_, p, cube);
  stats.decomposition_time += wall_time() - t0;

  // 2. Migration (positions do not change, so `cube` still contains every particle).
  t0 = wall_time();
  std::vector<int> dest(p.size());
  for (std::size_t i = 0; i < p.size(); ++i) dest[i] = decomp_.owner_of(p, i);
  stats.particles_migrated += migrate_particles(comm_, p, dest);
  stats.migration_time += wall_time() - t0;

  // 3. Local tree in the common global cube.
  t0 = wall_time();
  BuildTimings bt;
  tree_ = build_tree(p, cube, tp_, threads_, &bt);
  stats.tree_build_time += wall_time() - t0;
  stats.tree_keys_time += bt.keys;
  stats.tree_sort_time += bt.sort;
  stats.tree_nodes = tree_.nodes.size();
  stats.tree_depth = std::max(stats.tree_depth, tree_.depth);

  // 4. Shared-memory publication of the local tree (V6), then the LET exchange with every
  //    rank outside the shared-memory group (P2P waits immediately; Overlap waits after the
  //    local force loop).
  std::vector<char> skip;
  if (dp_.shm) {
    if (!shm_) shm_ = std::make_unique<ShmTreeWindows>(comm_, dp_.shm_group_size);
    const double sync_before = shm_->sync_time;
    t0 = wall_time();
    shm_->publish(tree_);
    stats.shm_publish_time += wall_time() - t0;
    stats.shm_sync_time += shm_->sync_time - sync_before;
    stats.shm_segment_bytes = shm_->capacity_bytes();
    stats.shm_reallocations = shm_->reallocations();
    skip.resize(comm_size(comm_));
    for (int r = 0; r < static_cast<int>(skip.size()); ++r) skip[r] = shm_->same_group(r);
  }
  LetExchanger ex;
  ex.max_boxes = dp_.let_boxes;
  ex.start(comm_, tree_, p, threads_, dp_.comm, dp_.shm ? &skip : nullptr);
  // (On the GPU there is no host-side local loop to overlap with, so Overlap waits here too.)
  if (dp_.comm != CommMode::Overlap || dp_.device == Device::GPU) ex.finish();

  // 5. Forces: local tree first, then each remote LET, accumulating into the same
  //    per-target sums (so with one rank the arithmetic is identical to V1).
  const long long n = static_cast<long long>(p.size());
  const double eps2 = phys_.eps * phys_.eps;
  const double G = phys_.G;
  const FlatTree& t = tree_;
  std::vector<WalkResult> acc(static_cast<std::size_t>(n));
  auto record = [&stats](const LoopBalance& lb) {
    stats.loop_busy_max += lb.busy_max;
    stats.loop_busy_mean += lb.busy_mean;
    stats.loop_work_max += lb.work_max;
    stats.loop_work_mean += lb.work_mean;
  };

  auto local_body = [&](long long k) {
    const std::uint32_t i = t.pidx[k];
    WalkResult& r = acc[i];
    walk_tree(t, p.x[i], p.y[i], p.z[i], i, eps2, r);
    return static_cast<double>(r.pp + r.pc);
  };
  // Remote sources in rank order: received LETs, and (V6) the full trees of shared-memory
  // peers. Walking a peer's full tree makes exactly the same accept/open decisions as
  // walking the LET it would have sent (every node the LET prunes is accepted by the point
  // MAC), so the arithmetic matches the conventional path bit for bit.
  auto collect_sources = [&]() {
    std::vector<TreeView> sources;
    const int P = comm_size(comm_), me = comm_rank(comm_);
    std::size_t next_let = 0;
    for (int r = 0; r < P; ++r) {
      if (r == me) continue;
      if (dp_.shm && shm_->same_group(r)) {
        const TreeView v = shm_->peer(r);
        if (v.n_nodes > 0) sources.push_back(v);
      } else if (next_let < ex.remote.size() && ex.remote_source[next_let] == r) {
        sources.push_back(view_of(ex.remote[next_let++]));
      }
    }
    return sources;
  };
  auto record_let = [&]() {
    stats.let_build_time += ex.build_time;
    stats.let_exchange_time += ex.exchange_time;
    stats.let_wait_time += ex.wait_time;
    stats.let_bytes_sent += ex.bytes_sent;
    stats.let_bytes_received += ex.bytes_received;
    stats.let_nodes_sent += ex.nodes_sent;
  };

  if (dp_.device == Device::GPU) {
    // V7: local tree and all remote sources walked on the offload device, same order as on
    // the CPU; the kernel time is reported as force time (local and remote together).
    record_let();
    const std::vector<TreeView> sources = collect_sources();
    std::vector<WalkResult> leaf;
    OffloadReport rep;
#ifdef _OPENMP
    // Ranks on one node take different GPUs (node rank modulo the number of devices).
    if (gpu_node_rank_ < 0) {
      MPI_Comm node;
      MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, comm_rank(comm_), MPI_INFO_NULL, &node);
      gpu_node_rank_ = comm_rank(node);
      MPI_Comm_free(&node);
    }
    if (omp_get_num_devices() > 0) {
      stats.device_id = gpu_node_rank_ % omp_get_num_devices();
      omp_set_default_device(stats.device_id);
    }
#endif
#ifdef _OPENMP
    // Ranks on one node take different GPUs (node rank modulo the number of devices).
    if (gpu_node_rank_ < 0) {
      MPI_Comm node;
      MPI_Comm_split_type(comm_, MPI_COMM_TYPE_SHARED, comm_rank(comm_), MPI_INFO_NULL, &node);
      gpu_node_rank_ = comm_rank(node);
      MPI_Comm_free(&node);
    }
    if (omp_get_num_devices() > 0) {
      stats.device_id = gpu_node_rank_ % omp_get_num_devices();
      omp_set_default_device(stats.device_id);
    }
#endif
    t0 = wall_time();
    offload_walks(t, sources, eps2, leaf, dp_.allow_host_fallback, rep);
    for (long long k = 0; k < n; ++k) acc[t.pidx[k]] = leaf[k];
    stats.force_local_time += wall_time() - t0;
    stats.traversal_time += wall_time() - t0;
    stats.executed = static_cast<int>(rep.executed);
    stats.num_devices = rep.num_devices;
    stats.gpu_h2d_time += rep.h2d;
    stats.gpu_kernel_time += rep.kernel;
    stats.gpu_d2h_time += rep.d2h;
    stats.gpu_h2d_bytes += rep.h2d_bytes;
  } else {
  stats.executed = static_cast<int>(Executed::CPU);
  t0 = wall_time();
  const double local_before = stats.force_local_time;
  if (dp_.comm == CommMode::Overlap) {
    // Chunks of the local loop; between chunks (outside any parallel region) the master
    // thread drives MPI progress. Iterations are the same as in the single loop.
    const long long chunks = std::max<long long>(1, std::min<long long>(dp_.overlap_chunks, n));
    const double wait_before = ex.wait_time;
    for (long long c = 0; c < chunks; ++c) {
      const long long b = n * c / chunks, e = n * (c + 1) / chunks;
      record(parallel_force_loop(e - b, threads_, sched_, [&](long long k) { return local_body(b + k); }));
      ex.progress();
    }
    stats.force_local_time += wall_time() - t0 - (ex.wait_time - wait_before);  // compute only
    ex.finish();
  } else {
    record(parallel_force_loop(n, threads_, sched_, local_body));
    stats.force_local_time += wall_time() - t0;
  }
  const double t1 = wall_time();
  record_let();
  const std::vector<TreeView> sources = collect_sources();

  // One loop over targets for all remote sources (one fork/join per step, not P-1).
  if (!sources.empty())
    record(parallel_force_loop(n, threads_, sched_, [&](long long k) {
      const std::uint32_t i = t.pidx[k];
      WalkResult& r = acc[i];
      const std::uint64_t before = r.pp + r.pc;
      for (const TreeView& src : sources) walk_tree(src, p.x[i], p.y[i], p.z[i], kNoSelf, eps2, r);
      return static_cast<double>(r.pp + r.pc - before);
    }));
  const double t2 = wall_time();
  stats.force_remote_time += t2 - t1;
  stats.traversal_time += (stats.force_local_time - local_before) + (t2 - t1);
  }  // CPU path

  double local_mass = 0.0;
  for (double m : p.m) local_mass += m;
  const double total_mass = allreduce(comm_, local_mass, MPI_SUM);

  std::uint64_t pp = 0, pc = 0, violations = 0;
  double coverage = 0.0;
  for (long long i = 0; i < n; ++i) {
    const WalkResult& r = acc[i];
    p.ax[i] = G * r.ax;
    p.ay[i] = G * r.ay;
    p.az[i] = G * r.az;
    p.pot[i] = G * r.phi;
    p.work[i] = static_cast<double>(r.pp + r.pc);
    pp += r.pp;
    pc += r.pc;
    violations += r.pruned_violations;
    coverage = std::max(coverage, std::fabs(r.mass_seen - (total_mass - p.m[i])) / total_mass);
  }
  stats.pp += pp;
  stats.pc += pc;
  stats.pruned_violations += violations;
  stats.mass_coverage_err = std::max(stats.mass_coverage_err, coverage);
  ++calls_;
}

}  // namespace nbody

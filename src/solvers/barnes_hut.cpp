#include "nbody/solvers/barnes_hut.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "nbody/core/timer.hpp"
#include "nbody/gpu/offload_walk.hpp"
#include "nbody/tree/traversal.hpp"

namespace nbody {

void BarnesHutSolver::compute_accelerations(ParticleSet& p, ForceStats& stats) {
  const double t0 = wall_time();
  BuildTimings bt;
  tree_ = build_tree(p, bounding_cube(p), tp_, threads_, &bt);
  const double t1 = wall_time();
  stats.tree_build_time += t1 - t0;
  stats.tree_keys_time += bt.keys;
  stats.tree_sort_time += bt.sort;
  stats.tree_nodes = tree_.nodes.size();
  stats.tree_depth = std::max(stats.tree_depth, tree_.depth);

  const long long n = static_cast<long long>(p.size());
  const double eps2 = phys_.eps * phys_.eps;
  const FlatTree& t = tree_;
  std::vector<WalkResult> acc(static_cast<std::size_t>(n));

  if (device_ == Device::GPU) {
    // Same walks on the offload device; results come back in leaf order.
    std::vector<WalkResult> leaf;
    OffloadReport rep;
    offload_walks(t, {}, eps2, leaf, allow_host_fallback_, rep);
    for (long long k = 0; k < n; ++k) acc[t.pidx[k]] = leaf[k];
    stats.executed = static_cast<int>(rep.executed);
    stats.num_devices = rep.num_devices;
    stats.gpu_h2d_time += rep.h2d;
    stats.gpu_kernel_time += rep.kernel;
    stats.gpu_d2h_time += rep.d2h;
    stats.gpu_h2d_bytes += rep.h2d_bytes;
  } else {
    // Targets in leaf order; iteration k is independent of every other iteration.
    const LoopBalance lb = parallel_force_loop(n, threads_, sched_, [&](long long k) {
      const std::uint32_t i = t.pidx[k];
      WalkResult& r = acc[i];
      walk_tree(t, p.x[i], p.y[i], p.z[i], i, eps2, r);
      return static_cast<double>(r.pp + r.pc);
    });
    stats.loop_busy_max += lb.busy_max;
    stats.loop_busy_mean += lb.busy_mean;
    stats.loop_work_max += lb.work_max;
    stats.loop_work_mean += lb.work_mean;
    stats.executed = static_cast<int>(Executed::CPU);
  }

  const double G = phys_.G;
  const double total_mass = t.nodes.empty() ? 0.0 : t.nodes[0].mass;
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
  stats.traversal_time += wall_time() - t1;
}

}  // namespace nbody

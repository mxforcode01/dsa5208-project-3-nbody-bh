// GPU offload (V7): the solver must report truthfully where the walks ran (checked against
// an independent target-region probe), refuse silent host fallback, and agree with the CPU
// walks. With a host fallback the code is the same, so results must be bitwise identical;
// on a real device the compiler may contract a*b+c into fused multiply-adds, so a
// rounding-level tolerance applies (derived below).
#include <cmath>
#include <cstdio>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nbody/core/initial_conditions.hpp"
#include "nbody/sim/accuracy.hpp"
#include "nbody/sim/simulation.hpp"
#include "nbody/solvers/barnes_hut.hpp"
#include "nbody/solvers/direct_sum.hpp"
#include "test_harness.hpp"

using namespace nbody;

namespace {

// Independent of the code under test: does a target region run on a device here?
bool target_runs_on_device() {
  int on = 0;
#pragma omp target map(from : on)
  {
#ifdef _OPENMP
    on = omp_is_initial_device() ? 0 : 1;
#endif
  }
  return on == 1;
}

ParticleSet make(Distribution d, std::uint64_t n) {
  ICParams ic;
  ic.n_total = n;
  ic.dist = d;
  ParticleSet p;
  generate_particles(ic, 0, n, p);
  remove_com_velocity(p);
  return p;
}

TreeParams tree(double theta) {
  TreeParams tp;
  tp.theta = theta;
  return tp;
}

}  // namespace

NB_TEST(device_report_matches_an_independent_probe) {
  const bool device = target_runs_on_device();
  std::printf("    probe: target regions run on %s\n", device ? "a DEVICE" : "the host");
  ParticleSet p = make(Distribution::Plummer, 3000);
  ForceStats st;
  BarnesHutSolver(PhysicsParams{}, tree(0.5), 1, {}, Device::GPU, true).compute_accelerations(p, st);
  std::printf("    solver: executed = %s, omp_get_num_devices = %d\n",
              to_string(static_cast<Executed>(st.executed)).c_str(), st.num_devices);
  NB_CHECK(st.executed == static_cast<int>(device ? Executed::GPU : Executed::HostFallback));
  ParticleSet q = make(Distribution::Plummer, 300);
  BarnesHutSolver(PhysicsParams{}, tree(0.5), 4).compute_accelerations(q, st);
  NB_CHECK(st.executed == static_cast<int>(Executed::CPU));
}

NB_TEST(host_fallback_is_refused_unless_allowed) {
  ParticleSet p = make(Distribution::Uniform, 1000);
  ForceStats st;
  BarnesHutSolver strict(PhysicsParams{}, tree(0.5), 1, {}, Device::GPU, false);
  if (target_runs_on_device()) {
    strict.compute_accelerations(p, st);  // must not throw
    NB_CHECK(st.executed == static_cast<int>(Executed::GPU));
  } else {
    NB_CHECK_THROWS(strict.compute_accelerations(p, st));
  }
}

NB_TEST(device_walks_agree_with_cpu_walks) {
  // Host fallback: identical code -> bitwise identical. Device: per interaction the result can
  // differ by a few ulp (fused multiply-add, device sqrt/division), and a MAC comparison can in
  // principle flip for a node at exactly the opening radius. A particle's acceleration is a sum
  // of K ~ 10^3 terms, so |da| <= ~K u sum|terms|; relative to |a| this stays far below the
  // Barnes-Hut error itself (~1e-3). We require rms_rel <= 1e-12, max_rel <= 1e-9 (floor-
  // normalised), and the interaction count equal to within 1e-6, and print the measured values.
  const bool device = target_runs_on_device();
  for (Distribution d : {Distribution::Plummer, Distribution::Uniform}) {
    ParticleSet cpu = make(d, 20000), gpu = cpu;
    ForceStats sc, sg;
    BarnesHutSolver(PhysicsParams{}, tree(0.5), 4).compute_accelerations(cpu, sc);
    BarnesHutSolver(PhysicsParams{}, tree(0.5), 1, {}, Device::GPU, true).compute_accelerations(gpu, sg);
    const ComparisonReport r = compare_by_id(gpu, cpu);
    const double work_diff = std::fabs(static_cast<double>(sg.pp + sg.pc) - static_cast<double>(sc.pp + sc.pc)) /
                             static_cast<double>(sc.pp + sc.pc);
    std::printf("    %-8s on %s: accel rms_rel %.3e max_rel %.3e | interactions rel diff %.3e | kernel %.4f s, "
                "h2d %.4f s, d2h %.4f s\n",
                to_string(d).c_str(), device ? "device" : "host", r.accel.rms_rel, r.accel.max_rel, work_diff,
                sg.gpu_kernel_time, sg.gpu_h2d_time, sg.gpu_d2h_time);
    if (!device) {
      NB_CHECK(r.accel.max_abs == 0.0 && r.potential.max_abs == 0.0);
      NB_CHECK(sg.pp == sc.pp && sg.pc == sc.pc);
    } else {
      NB_CHECK_LE(r.accel.rms_rel, 1e-12);
      NB_CHECK_LE(r.accel.max_rel, 1e-9);
      NB_CHECK_LE(work_diff, 1e-6);
    }
    NB_CHECK(sg.pruned_violations == 0u);
    NB_CHECK_LE(sg.mass_coverage_err, 1e-9);
  }
}

NB_TEST(device_trajectories_track_cpu) {
  // 5 KDK steps: trajectories differ only through the (rounding-level) force differences.
  PhysicsParams phys;
  phys.eps = 0.01;
  ParticleSet cpu = make(Distribution::Plummer, 5000), gpu = cpu;
  BarnesHutSolver sc(phys, tree(0.5), 4), sg(phys, tree(0.5), 1, {}, Device::GPU, true);
  SimParams sp;
  sp.steps = 5;
  sp.dt = 1e-3;
  Simulation(sp, sc).run(cpu);
  Simulation(sp, sg).run(gpu);
  const ComparisonReport r = compare_by_id(gpu, cpu);
  std::printf("    max |dx| %.3e, max |dv| %.3e\n", r.position.max_abs, r.velocity.max_abs);
  if (!target_runs_on_device()) NB_CHECK(r.position.max_abs == 0.0 && r.velocity.max_abs == 0.0);
  else NB_CHECK_LE(r.position.max_abs, 1e-12);
}

NB_TEST(tiny_and_empty_sets) {
  for (std::uint64_t n : {0u, 1u, 2u, 9u}) {
    ParticleSet cpu = make(Distribution::Uniform, n == 0 ? 1 : n), gpu = cpu;
    if (n == 0) {
      cpu.clear();
      gpu.clear();
    }
    ForceStats st;
    BarnesHutSolver(PhysicsParams{}, tree(0.5), 1).compute_accelerations(cpu, st);
    BarnesHutSolver(PhysicsParams{}, tree(0.5), 1, {}, Device::GPU, true).compute_accelerations(gpu, st);
    bool ok = true;
    for (std::size_t i = 0; i < cpu.size(); ++i)
      ok = ok && std::fabs(cpu.ax[i] - gpu.ax[i]) <= 1e-12 * (std::fabs(cpu.ax[i]) + 1e-300);
    NB_CHECK(ok);
  }
}

NB_TEST_MAIN()

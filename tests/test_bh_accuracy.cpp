// Barnes-Hut (V1) against direct sum (V0): exactness at theta = 0, error growth
// with theta, thread independence, coincident particles and short trajectories.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "nbody/core/initial_conditions.hpp"
#include "nbody/sim/accuracy.hpp"
#include "nbody/sim/simulation.hpp"
#include "nbody/solvers/barnes_hut.hpp"
#include "nbody/solvers/direct_sum.hpp"
#include "test_harness.hpp"

using namespace nbody;

namespace {

ParticleSet make(Distribution d, std::uint64_t n, std::uint64_t seed = 42) {
  ICParams ic;
  ic.n_total = n;
  ic.seed = seed;
  ic.dist = d;
  ParticleSet p;
  generate_particles(ic, 0, n, p);
  remove_com_velocity(p);
  return p;
}

TreeParams tree(double theta, int cap = 8, int depth = 21) {
  TreeParams tp;
  tp.theta = theta;
  tp.leaf_capacity = cap;
  tp.max_depth = depth;
  return tp;
}

ForceStats forces(ForceSolver&& s, ParticleSet& p) {
  ForceStats st;
  s.compute_accelerations(p, st);
  return st;
}

// At theta = 0 Barnes-Hut sums exactly the same pair terms as direct sum, only in a
// different order. Recursive summation of N terms has error <= (N-1) u sum|terms|
// (u = 2^-53), so we require |a_bh - a_ds| <= 2 N u S_i with S_i = sum_j |a_ij|.
void check_exact_at_theta_zero(ParticleSet p, const PhysicsParams& phys) {
  ParticleSet ref = p;
  const ForceStats sb = forces(BarnesHutSolver(phys, tree(0.0)), p);
  forces(DirectSumSolver(phys), ref);
  const std::size_t n = p.size();
  const double u = std::numeric_limits<double>::epsilon() / 2.0;
  const double eps2 = phys.eps * phys.eps;
  bool ok = true;
  double worst = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      if (j == i) continue;
      const double dx = p.x[j] - p.x[i], dy = p.y[j] - p.y[i], dz = p.z[j] - p.z[i];
      const double r2 = dx * dx + dy * dy + dz * dz + eps2;
      s += phys.G * p.m[j] * std::sqrt(dx * dx + dy * dy + dz * dz) / (r2 * std::sqrt(r2));
    }
    const double bound = 2.0 * n * u * s;
    const double err = std::hypot(p.ax[i] - ref.ax[i], p.ay[i] - ref.ay[i], p.az[i] - ref.az[i]);
    ok = ok && err <= bound;
    worst = std::max(worst, bound > 0 ? err / bound : 0.0);
    // Potential terms are all negative, so sum|terms| = |pot|.
    ok = ok && std::fabs(p.pot[i] - ref.pot[i]) <= 2.0 * n * u * std::fabs(ref.pot[i]);
  }
  std::printf("    theta=0, N=%zu: worst error / rounding bound = %.3f\n", n, worst);
  NB_CHECK(ok);
  NB_CHECK(sb.pc == 0u);
  NB_CHECK(sb.pp == static_cast<std::uint64_t>(n) * (n - 1));
}

}  // namespace

NB_TEST(theta_zero_equals_direct_sum_to_rounding) {
  PhysicsParams phys;
  phys.eps = 1e-3;
  check_exact_at_theta_zero(make(Distribution::Plummer, 2000), phys);
  check_exact_at_theta_zero(make(Distribution::Uniform, 2000), phys);
}

NB_TEST(root_leaf_is_bitwise_direct_sum) {
  // With max_depth = 0 everything is in one leaf whose particles are in index order,
  // so Barnes-Hut performs exactly the direct-sum loop and must match bit for bit.
  ParticleSet a = make(Distribution::Plummer, 700), b = a;
  forces(BarnesHutSolver(PhysicsParams{}, tree(0.5, 8, 0)), a);
  forces(DirectSumSolver(PhysicsParams{}), b);
  NB_CHECK(a.ax == b.ax && a.ay == b.ay && a.az == b.az && a.pot == b.pot);
}

NB_TEST(error_grows_and_work_shrinks_with_theta) {
  // Monopole Barnes-Hut: the force error must increase monotonically with theta and the
  // interaction count must decrease. Thresholds at theta = 0.5 are the measured values
  // (N = 5000, seed 42, GCC 15.2: rms_rel 1.23e-3 / 1.85e-3 and p99_rel 7.2e-3 / 1.19e-2 for
  // plummer / uniform) with ~2x headroom; they guard against regressions in the MAC or moments.
  for (Distribution d : {Distribution::Plummer, Distribution::Uniform}) {
    const ParticleSet base = make(d, 5000);
    ParticleSet ref = base;
    forces(DirectSumSolver(PhysicsParams{}, 4), ref);
    double prev_err = 0.0;
    std::uint64_t prev_work = std::numeric_limits<std::uint64_t>::max();
    std::printf("    %-8s %6s %12s %12s %12s %14s\n", to_string(d).c_str(), "theta", "rms_rel", "median_rel",
                "p99_rel", "interactions");
    for (double theta : {0.1, 0.2, 0.3, 0.5, 0.7, 1.0}) {
      ParticleSet p = base;
      const ForceStats st = forces(BarnesHutSolver(PhysicsParams{}, tree(theta)), p);
      const ComparisonReport r = compare_by_id(p, ref);
      const std::uint64_t work = st.pp + st.pc;
      std::printf("    %-8s %6.2f %12.4e %12.4e %12.4e %14llu\n", "", theta, r.accel.rms_rel, r.accel.median_rel,
                  r.accel.p99_rel, static_cast<unsigned long long>(work));
      NB_CHECK(r.accel.rms_rel > prev_err);
      NB_CHECK(work < prev_work);
      prev_err = r.accel.rms_rel;
      prev_work = work;
      if (theta == 0.5) {
        NB_CHECK_LE(r.accel.rms_rel, 4e-3);
        NB_CHECK_LE(r.accel.p99_rel, 2.5e-2);
      }
    }
  }
}

NB_TEST(thread_count_does_not_change_results) {
  ParticleSet a = make(Distribution::Plummer, 5000), b = a;
  forces(BarnesHutSolver(PhysicsParams{}, tree(0.5), 1), a);
  forces(BarnesHutSolver(PhysicsParams{}, tree(0.5), 4), b);
  NB_CHECK(a.ax == b.ax && a.ay == b.ay && a.az == b.az && a.pot == b.pot && a.work == b.work);
}

NB_TEST(morton_build_gives_bitwise_identical_forces) {
  ParticleSet a = make(Distribution::Plummer, 20000), b = a;
  TreeParams tm = tree(0.5);
  tm.build = TreeBuild::MortonSorted;
  forces(BarnesHutSolver(PhysicsParams{}, tree(0.5), 1), a);
  const ForceStats st = forces(BarnesHutSolver(PhysicsParams{}, tm, 4), b);
  NB_CHECK(a.ax == b.ax && a.ay == b.ay && a.az == b.az && a.pot == b.pot && a.work == b.work);
  NB_CHECK(st.tree_keys_time > 0.0 && st.tree_sort_time > 0.0);
}

NB_TEST(force_loop_runs_every_iteration_once_and_counts_work) {
  // Each iteration writes its own slot, so a missing or repeated iteration is visible.
  for (ScheduleKind kind : {ScheduleKind::Static, ScheduleKind::Dynamic, ScheduleKind::Guided})
    for (int chunk : {0, 1, 37})
      for (int threads : {1, 4}) {
        const long long n = 10007;
        std::vector<int> hits(n, 0);
        const LoopBalance lb = parallel_force_loop(n, threads, LoopSchedule{kind, chunk}, [&](long long k) {
          ++hits[k];
          return static_cast<double>(k % 7);
        });
        double expected = 0.0;
        for (long long k = 0; k < n; ++k) expected += static_cast<double>(k % 7);
        NB_CHECK(std::all_of(hits.begin(), hits.end(), [](int h) { return h == 1; }));
        NB_CHECK(lb.work_total == expected);
        NB_CHECK(lb.threads == threads);
        NB_CHECK(lb.work_max >= lb.work_mean && lb.busy_max >= lb.busy_mean && lb.busy_mean >= 0.0);
        NB_CHECK_NEAR(lb.work_mean * lb.threads, lb.work_total, 1e-9 * expected);
      }
}

NB_TEST(schedules_do_not_change_results) {
  // Targets are independent, so any schedule / chunk must give bitwise identical forces.
  ParticleSet ref = make(Distribution::Plummer, 20000);
  forces(BarnesHutSolver(PhysicsParams{}, tree(0.5), 1), ref);
  for (ScheduleKind kind : {ScheduleKind::Static, ScheduleKind::Dynamic, ScheduleKind::Guided})
    for (int chunk : {0, 1, 64}) {
      ParticleSet p = make(Distribution::Plummer, 20000);
      const ForceStats st = forces(BarnesHutSolver(PhysicsParams{}, tree(0.5), 4, LoopSchedule{kind, chunk}), p);
      NB_CHECK(p.ax == ref.ax && p.ay == ref.ay && p.az == ref.az && p.pot == ref.pot && p.work == ref.work);
      NB_CHECK(st.thread_imbalance() >= 1.0 && st.thread_work_imbalance() >= 1.0);
    }
}

NB_TEST(work_and_violation_counters) {
  ParticleSet p = make(Distribution::Plummer, 3000);
  const ForceStats st = forces(BarnesHutSolver(PhysicsParams{}, tree(0.5)), p);
  double total = 0.0;
  for (double w : p.work) total += w;
  NB_CHECK(total == static_cast<double>(st.pp + st.pc));
  NB_CHECK(st.pruned_violations == 0u);  // local trees never contain pruned nodes
  NB_CHECK(st.tree_nodes > 0u && st.tree_build_time >= 0.0 && st.traversal_time >= 0.0);
}

NB_TEST(coincident_particles_give_finite_correct_forces) {
  // 20 particles at one point plus 1000 others, softened: the over-full max-depth leaf must
  // be handled and theta = 0 must still reproduce direct sum.
  ParticleSet p = make(Distribution::Uniform, 1020);
  for (std::size_t i = 0; i < 20; ++i) {
    p.x[i] = 0.25;
    p.y[i] = -0.1;
    p.z[i] = 0.3;
  }
  PhysicsParams phys;
  phys.eps = 0.01;
  ParticleSet q = p;
  forces(BarnesHutSolver(phys, tree(0.5)), q);
  bool finite = true;
  for (std::size_t i = 0; i < q.size(); ++i)
    finite = finite && std::isfinite(q.ax[i]) && std::isfinite(q.ay[i]) && std::isfinite(q.az[i]);
  NB_CHECK(finite);
  check_exact_at_theta_zero(p, phys);
}

NB_TEST(short_trajectories_track_direct_sum) {
  // Over a short time t the trajectory difference is driven by the force error:
  // |dx| ~ |da| t^2 / 2 and |dv| ~ |da| t. With A = max |da| at t = 0 we require
  // |dx| <= A t^2 and |dv| <= 2 A t (factor-2 allowance for da changing along the orbit).
  ParticleSet a = make(Distribution::Plummer, 1000), b = a;
  PhysicsParams phys;
  phys.eps = 0.01;
  ParticleSet a0 = a, b0 = b;
  forces(BarnesHutSolver(phys, tree(0.3)), a0);
  forces(DirectSumSolver(phys), b0);
  const double A = compare_by_id(a0, b0).accel.max_abs;

  SimParams sp;
  sp.steps = 10;
  sp.dt = 1e-3;
  BarnesHutSolver bh(phys, tree(0.3));
  DirectSumSolver ds(phys);
  Simulation(sp, bh).run(a);
  Simulation(sp, ds).run(b);
  const ComparisonReport r = compare_by_id(a, b);
  const double t = sp.steps * sp.dt;
  std::printf("    A = %.3e | max |dx| = %.3e (bound %.3e) | max |dv| = %.3e (bound %.3e)\n", A,
              r.position.max_abs, A * t * t, r.velocity.max_abs, 2 * A * t);
  NB_CHECK(r.position.max_abs > 0.0);  // genuinely different methods
  NB_CHECK_LE(r.position.max_abs, A * t * t);
  NB_CHECK_LE(r.velocity.max_abs, 2.0 * A * t);
}

NB_TEST_MAIN()

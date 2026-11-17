// Distributed Barnes-Hut (V2). Run under mpiexec with any number of ranks (ctest runs
// P = 1, 2, 3, 4). Every test calls the same collectives on every rank; comparisons
// against serial references (direct sum, V1) are made on rank 0 after a gather.
#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "nbody/core/initial_conditions.hpp"
#include "nbody/mpi/dist_bh.hpp"
#include "nbody/mpi/mpi_env.hpp"
#include "nbody/mpi/particle_comm.hpp"
#include "nbody/sim/accuracy.hpp"
#include "nbody/sim/simulation.hpp"
#include "nbody/solvers/barnes_hut.hpp"
#include "nbody/solvers/direct_sum.hpp"
#include "test_harness.hpp"

using namespace nbody;

namespace {

const MPI_Comm W = MPI_COMM_WORLD;
constexpr double kCoverageTol = 1e-9;  // see let_accounts_for_every_source_exactly_once
int rank() { return comm_rank(W); }
int size() { return comm_size(W); }
bool root() { return rank() == 0; }

ICParams ic_params(Distribution d, std::uint64_t n, std::uint64_t seed = 42) {
  ICParams ic;
  ic.n_total = n;
  ic.seed = seed;
  ic.dist = d;
  return ic;
}

ParticleSet dist_ics(Distribution d, std::uint64_t n, std::uint64_t seed = 42) {
  ParticleSet p;
  generate_particles_distributed(W, ic_params(d, n, seed), p);
  remove_com_velocity_global(W, p);
  return p;
}

ParticleSet serial_ics(Distribution d, std::uint64_t n, std::uint64_t seed = 42) {
  ParticleSet p;
  generate_particles(ic_params(d, n, seed), 0, n, p);
  remove_com_velocity(p);
  return p;
}

TreeParams tree(double theta) {
  TreeParams tp;
  tp.theta = theta;
  return tp;
}

struct GlobalChecks {
  std::uint64_t violations = 0;
  double coverage = 0.0;
};

GlobalChecks global_checks(const ForceStats& st) {
  return {allreduce_u64(W, st.pruned_violations, MPI_SUM), allreduce(W, st.mass_coverage_err, MPI_MAX)};
}

// Same rounding bound as the serial theta = 0 test: |a - a_ref| <= 2 N u sum_j |a_ij|.
bool within_rounding_of_direct_sum(const ParticleSet& test, const PhysicsParams& phys) {
  ParticleSet ref = test;
  ForceStats st;
  DirectSumSolver(phys, 4).compute_accelerations(ref, st);
  const std::size_t n = ref.size();
  const double u = std::numeric_limits<double>::epsilon() / 2.0, eps2 = phys.eps * phys.eps;
  bool ok = true;
  for (std::size_t i = 0; i < n; ++i) {
    double s = 0.0;
    for (std::size_t j = 0; j < n; ++j) {
      if (j == i) continue;
      const double dx = ref.x[j] - ref.x[i], dy = ref.y[j] - ref.y[i], dz = ref.z[j] - ref.z[i];
      const double r2 = dx * dx + dy * dy + dz * dz + eps2;
      s += phys.G * ref.m[j] * std::sqrt(dx * dx + dy * dy + dz * dz) / (r2 * std::sqrt(r2));
    }
    ok = ok && std::hypot(test.ax[i] - ref.ax[i], test.ay[i] - ref.ay[i], test.az[i] - ref.az[i]) <= 2.0 * n * u * s;
  }
  return ok;
}

}  // namespace

NB_TEST(migration_conserves_particles_and_assigns_owners) {
  const std::uint64_t n = 20000;
  ParticleSet p = dist_ics(Distribution::Plummer, n);
  DistributedBHSolver solver(W, PhysicsParams{}, tree(0.5), DistParams{}, 2);
  ForceStats st;
  solver.compute_accelerations(p, st);

  NB_CHECK(allreduce_u64(W, p.size(), MPI_SUM) == n);
  std::uint64_t wrong_owner = 0;
  for (std::size_t i = 0; i < p.size(); ++i) wrong_owner += solver.decomposition().owner_of(p, i) != rank();
  NB_CHECK(allreduce_u64(W, wrong_owner, MPI_SUM) == 0u);
  double mass = 0.0;
  for (double m : p.m) mass += m;
  // Summing N masses of 1/N in any order rounds by at most N u (u = 2^-53).
  NB_CHECK_NEAR(allreduce(W, mass, MPI_SUM), 1.0, n * std::numeric_limits<double>::epsilon() / 2.0);

  const ParticleSet all = gather_to_root(W, p);
  if (root()) {
    std::vector<std::uint64_t> ids = all.id;
    std::sort(ids.begin(), ids.end());
    bool ok = ids.size() == n;
    for (std::size_t i = 0; ok && i < n; ++i) ok = ids[i] == i;
    NB_CHECK(ok);  // every id exactly once
  }
}

NB_TEST(decomposition_balances_load) {
  // Exact splitters split the weight to within one particle: with unit weights (first call)
  // the particle count of a rank differs from N/P by at most 1, i.e. imbalance <= 1 + P/N.
  // With interaction weights (second call) the predicted imbalance is bounded by the
  // heaviest single particle. The actual work of the second call is not the predicted one:
  // re-decomposing moves particles to new domains, which changes their LETs and therefore
  // their interaction counts (measured 1.0003-1.0008). We require it to stay within 1%,
  // against 2.5 for the earlier fixed-histogram version on Plummer at P = 4.
  const std::uint64_t n = 40000;
  for (Distribution d : {Distribution::Uniform, Distribution::Plummer}) {
    ParticleSet p = dist_ics(d, n);
    DistributedBHSolver solver(W, PhysicsParams{}, tree(0.5), DistParams{}, 2);
    ForceStats st;
    solver.compute_accelerations(p, st);
    const RankSpread counts = rank_spread(W, static_cast<double>(p.size()));
    solver.compute_accelerations(p, st);
    double work = 0.0, max_w = 0.0;
    for (double w : p.work) {
      work += w;
      max_w = std::max(max_w, w);
    }
    const RankSpread works = rank_spread(W, work);
    const double heaviest = allreduce(W, max_w, MPI_MAX);
    if (root())
      std::printf("    %-8s P=%d: count imbalance %.5f, predicted work imbalance %.5f, actual work imbalance %.5f\n",
                  to_string(d).c_str(), size(), counts.imbalance(), solver.decomposition().predicted_imbalance(),
                  works.imbalance());
    NB_CHECK_LE(counts.imbalance(), 1.0 + static_cast<double>(size()) / n);
    NB_CHECK_LE(solver.decomposition().predicted_imbalance(), 1.0 + 2.0 * heaviest / works.mean);
    NB_CHECK_LE(works.imbalance(), 1.01);
  }
}

NB_TEST(let_accounts_for_every_source_exactly_once) {
  // mass_coverage_err = max_i |mass that acted on i - (M - m_i)| / M: a missing or doubled
  // remote contribution shows up as an error of at least m_i / M = 1/N = 1.25e-4, while
  // rounding of the two N-term sums is <= 2 N u ~ 2e-12. kCoverageTol = 1e-9 separates them.
  for (Distribution d : {Distribution::Uniform, Distribution::Plummer})
    for (double theta : {0.3, 0.7, 1.0}) {
      ParticleSet p = dist_ics(d, 8000);
      ForceStats st;
      DistributedBHSolver(W, PhysicsParams{}, tree(theta), DistParams{}, 2).compute_accelerations(p, st);
      const GlobalChecks g = global_checks(st);
      const std::uint64_t bytes = allreduce_u64(W, st.let_bytes_sent, MPI_SUM);
      if (root())
        std::printf("    %-8s theta=%.1f: violations %llu, coverage error %.2e, LET %.1f kB\n", to_string(d).c_str(),
                    theta, static_cast<unsigned long long>(g.violations), g.coverage, bytes / 1024.0);
      NB_CHECK(g.violations == 0u);
      NB_CHECK_LE(g.coverage, kCoverageTol);
      if (size() == 1) NB_CHECK(bytes == 0u);
    }
}

NB_TEST(single_rank_is_bitwise_serial_bh) {
  // With one rank there is no LET and the local tree is the serial tree, so V2 must equal V1 exactly.
  if (size() != 1) {
    if (root()) std::printf("    (only checked when run with 1 rank)\n");
    return;
  }
  ParticleSet a = dist_ics(Distribution::Plummer, 5000), b = serial_ics(Distribution::Plummer, 5000);
  ForceStats st;
  DistributedBHSolver(W, PhysicsParams{}, tree(0.5), DistParams{}, 2).compute_accelerations(a, st);
  BarnesHutSolver(PhysicsParams{}, tree(0.5), 1).compute_accelerations(b, st);
  const ComparisonReport r = compare_by_id(a, b);
  NB_CHECK(r.accel.max_abs == 0.0 && r.potential.max_abs == 0.0);
}

NB_TEST(theta_zero_matches_direct_sum_to_rounding) {
  // theta = 0: no cell is ever accepted, so every rank receives every other rank's particles
  // and the result must equal direct sum up to summation-order rounding.
  PhysicsParams phys;
  ParticleSet p = dist_ics(Distribution::Plummer, 1500);
  ForceStats st;
  DistributedBHSolver(W, phys, tree(0.0), DistParams{}, 2).compute_accelerations(p, st);
  const GlobalChecks g = global_checks(st);
  NB_CHECK(g.violations == 0u);
  const ParticleSet all = gather_to_root(W, p);
  if (root()) NB_CHECK(within_rounding_of_direct_sum(all, phys));
}

NB_TEST(accuracy_matches_serial_bh) {
  // The distributed tree groups sources differently from the serial tree (cells are split
  // at domain boundaries), so results are not identical; its error against direct sum must
  // be no worse than 1.1x the serial V1 error.
  for (Distribution d : {Distribution::Plummer, Distribution::Uniform}) {
    ParticleSet p = dist_ics(d, 4000);
    ForceStats st;
    DistributedBHSolver(W, PhysicsParams{}, tree(0.5), DistParams{}, 2).compute_accelerations(p, st);
    const ParticleSet all = gather_to_root(W, p);
    if (!root()) continue;
    ParticleSet ref = serial_ics(d, 4000), v1 = ref;
    DirectSumSolver(PhysicsParams{}, 4).compute_accelerations(ref, st);
    BarnesHutSolver(PhysicsParams{}, tree(0.5), 1).compute_accelerations(v1, st);
    const ComparisonReport rd = compare_by_id(all, ref), rs = compare_by_id(v1, ref);
    std::printf("    %-8s P=%d: rms_rel V2 %.4e vs V1 %.4e | p99_rel V2 %.4e vs V1 %.4e\n", to_string(d).c_str(), size(),
                rd.accel.rms_rel, rs.accel.rms_rel, rd.accel.p99_rel, rs.accel.p99_rel);
    NB_CHECK_LE(rd.accel.rms_rel, 1.1 * rs.accel.rms_rel);
    NB_CHECK_LE(rd.accel.p99_rel, 1.1 * rs.accel.p99_rel);
  }
}

NB_TEST(multi_step_run_tracks_serial_bh) {
  // 10 KDK steps with migration every step. Trajectory differences are driven by the force
  // difference A = max |a_V2 - a_V1| at t = 0: require |dx| <= A t^2, |dv| <= 2 A t
  // (exactly 0 with one rank). Also checks particle count, ids and global diagnostics.
  PhysicsParams phys;
  phys.eps = 0.01;
  const std::uint64_t n = 2000;
  ParticleSet p0 = dist_ics(Distribution::Plummer, n);
  ParticleSet p = p0;
  ForceStats st;
  DistributedBHSolver(W, phys, tree(0.5), DistParams{}, 2).compute_accelerations(p0, st);
  const ParticleSet a0 = gather_to_root(W, p0);

  DistributedBHSolver solver(W, phys, tree(0.5), DistParams{}, 2);
  SimParams sp;
  sp.steps = 10;
  sp.dt = 1e-3;
  sp.threads = 2;
  sp.reduce_diagnostics = [](Diagnostics& d) {
    double v[7] = {d.mass, d.px, d.py, d.pz, d.momentum_scale, d.kinetic, d.potential};
    MPI_Allreduce(MPI_IN_PLACE, v, 7, MPI_DOUBLE, MPI_SUM, W);
    d = Diagnostics{v[0], v[1], v[2], v[3], v[4], v[5], v[6]};
  };
  const RunResult r = Simulation(sp, solver).run(p);
  NB_CHECK(allreduce_u64(W, p.size(), MPI_SUM) == n);
  NB_CHECK(global_checks(r.stats).violations == 0u);
  const ParticleSet all = gather_to_root(W, p);
  if (!root()) return;

  ParticleSet s0 = serial_ics(Distribution::Plummer, n), s = s0;
  BarnesHutSolver(phys, tree(0.5), 1).compute_accelerations(s0, st);
  const double A = compare_by_id(a0, s0).accel.max_abs;
  BarnesHutSolver v1(phys, tree(0.5), 1);
  SimParams ss = sp;
  ss.reduce_diagnostics = nullptr;
  ss.threads = 1;
  const RunResult rs = Simulation(ss, v1).run(s);
  const ComparisonReport c = compare_by_id(all, s);  // throws if ids were lost or duplicated
  const double t = sp.steps * sp.dt;
  std::printf("    P=%d: A = %.3e | max |dx| %.3e (bound %.3e) | max |dv| %.3e (bound %.3e)\n", size(), A,
              c.position.max_abs, A * t * t, c.velocity.max_abs, 2 * A * t);
  NB_CHECK_LE(c.position.max_abs, A * t * t);
  NB_CHECK_LE(c.velocity.max_abs, 2.0 * A * t);
  // Two sums of the same N masses in different orders: each rounds by <= N u.
  NB_CHECK_NEAR(r.diag.back().d.mass, rs.diag.back().d.mass, 2.0 * n * std::numeric_limits<double>::epsilon() / 2.0);
}

NB_TEST(morton_local_trees_give_identical_forces) {
  // The Morton builder reproduces the insertion tree exactly, so the distributed forces
  // (local trees and the LETs cut from them) must be bitwise identical.
  ParticleSet a = dist_ics(Distribution::Plummer, 20000), b = a;
  TreeParams tm = tree(0.5);
  tm.build = TreeBuild::MortonSorted;
  ForceStats st;
  DistributedBHSolver(W, PhysicsParams{}, tree(0.5), DistParams{}, 2).compute_accelerations(a, st);
  DistributedBHSolver(W, PhysicsParams{}, tm, DistParams{}, 2).compute_accelerations(b, st);
  const ParticleSet ga = gather_to_root(W, a), gb = gather_to_root(W, b);
  if (root()) {
    const ComparisonReport r = compare_by_id(gb, ga);
    NB_CHECK(r.accel.max_abs == 0.0 && r.potential.max_abs == 0.0);
  }
}

NB_TEST(schedules_give_identical_distributed_forces) {
  ParticleSet a = dist_ics(Distribution::Plummer, 20000), b = a, c = a;
  ForceStats st;
  DistributedBHSolver(W, PhysicsParams{}, tree(0.5), DistParams{}, 2).compute_accelerations(a, st);
  DistributedBHSolver(W, PhysicsParams{}, tree(0.5), DistParams{}, 2, LoopSchedule{ScheduleKind::Dynamic, 32})
      .compute_accelerations(b, st);
  DistributedBHSolver(W, PhysicsParams{}, tree(0.5), DistParams{}, 2, LoopSchedule{ScheduleKind::Guided, 0})
      .compute_accelerations(c, st);
  const ParticleSet ga = gather_to_root(W, a), gb = gather_to_root(W, b), gc = gather_to_root(W, c);
  if (root()) {
    NB_CHECK(compare_by_id(gb, ga).accel.max_abs == 0.0);
    NB_CHECK(compare_by_id(gc, ga).accel.max_abs == 0.0);
  }
}

NB_TEST(comm_modes_give_identical_forces_and_trajectories) {
  // Every mode delivers the same LETs and every target visits them in rank order, so
  // blocking, p2p and overlap (any chunk count) must agree bit for bit - over one force
  // evaluation and over a 5-step run with migration.
  struct Mode {
    CommMode comm;
    int chunks;
  };
  const Mode modes[] = {{CommMode::Blocking, 32}, {CommMode::P2P, 32}, {CommMode::Overlap, 1},
                        {CommMode::Overlap, 7}, {CommMode::Overlap, 32}};
  PhysicsParams phys;
  phys.eps = 0.01;
  ParticleSet ref_f, ref_s;
  std::uint64_t ref_bytes = 0;
  bool first = true;
  for (const Mode& m : modes) {
    DistParams dp;
    dp.comm = m.comm;
    dp.overlap_chunks = m.chunks;
    ParticleSet p = dist_ics(Distribution::Plummer, 20000);
    ForceStats st;
    DistributedBHSolver(W, phys, tree(0.5), dp, 2).compute_accelerations(p, st);
    const GlobalChecks g = global_checks(st);
    NB_CHECK(g.violations == 0u);
    NB_CHECK_LE(g.coverage, kCoverageTol);
    const std::uint64_t bytes = allreduce_u64(W, st.let_bytes_sent, MPI_SUM);
    const ParticleSet f = gather_to_root(W, p);

    ParticleSet q = dist_ics(Distribution::Plummer, 4000);
    DistributedBHSolver solver(W, phys, tree(0.5), dp, 2);
    SimParams sp;
    sp.steps = 5;
    sp.dt = 1e-3;
    Simulation(sp, solver).run(q);
    const ParticleSet s = gather_to_root(W, q);
    if (first) {
      ref_f = f;
      ref_s = s;
      ref_bytes = bytes;
      first = false;
      continue;
    }
    NB_CHECK(bytes == ref_bytes);
    if (root()) {
      const ComparisonReport rf = compare_by_id(f, ref_f), rs = compare_by_id(s, ref_s);
      NB_CHECK(rf.accel.max_abs == 0.0 && rf.potential.max_abs == 0.0);
      NB_CHECK(rs.position.max_abs == 0.0 && rs.velocity.max_abs == 0.0);
    }
  }
}

NB_TEST(let_boxes_change_volume_but_not_forces) {
  // A cell is pruned only if it is accepted for every receiver box, and a cell accepted for the
  // single bounding box is accepted for all tight sub-boxes inside it. So more boxes can only
  // shrink the LET, and every target still sees the same interactions in the same order: the
  // forces are bit-identical for every box count.
  PhysicsParams phys;
  phys.eps = 0.01;
  ParticleSet ref;
  std::uint64_t bytes1 = 0, prev = 0;
  bool first = true;
  for (int boxes : {1, 4, 64}) {
    DistParams dp;
    dp.let_boxes = boxes;
    ParticleSet p = dist_ics(Distribution::Plummer, 20000);
    ForceStats st;
    DistributedBHSolver(W, phys, tree(0.5), dp, 2).compute_accelerations(p, st);
    const GlobalChecks g = global_checks(st);
    NB_CHECK(g.violations == 0u);
    NB_CHECK_LE(g.coverage, kCoverageTol);
    const std::uint64_t bytes = allreduce_u64(W, st.let_bytes_sent, MPI_SUM);
    const ParticleSet f = gather_to_root(W, p);
    if (first) {
      ref = f;
      bytes1 = prev = bytes;
      first = false;
      continue;
    }
    NB_CHECK_LE(static_cast<double>(bytes), static_cast<double>(prev));
    prev = bytes;
    if (root()) {
      const ComparisonReport r = compare_by_id(f, ref);
      NB_CHECK(r.accel.max_abs == 0.0 && r.potential.max_abs == 0.0);
    }
  }
  if (size() >= 3) NB_CHECK_LE(static_cast<double>(prev), 0.9 * static_cast<double>(bytes1));
}

NB_TEST(shm_windows_give_identical_results) {
  // Walking a shared-memory peer's full tree makes the same decisions as walking the LET it
  // would have sent, and sources are visited in rank order, so the result must not change.
  // group 0 = whole machine (every peer through shm), 1 = no peers (all LETs), 2 = mixed
  // (emulated 2-rank nodes: shm inside, LETs between). Also with the overlap comm mode.
  struct Cfg {
    bool shm;
    int group;
    CommMode comm;
  };
  const Cfg cfgs[] = {{false, 0, CommMode::Blocking}, {true, 0, CommMode::Blocking}, {true, 1, CommMode::Blocking},
                      {true, 2, CommMode::Blocking},  {true, 2, CommMode::Overlap},  {true, 0, CommMode::P2P}};
  PhysicsParams phys;
  phys.eps = 0.01;
  ParticleSet ref_f, ref_s;
  double ref_recv = 0.0;
  bool first = true;
  for (const Cfg& cfg : cfgs) {
    DistParams dp;
    dp.shm = cfg.shm;
    dp.shm_group_size = cfg.group;
    dp.comm = cfg.comm;
    ParticleSet p = dist_ics(Distribution::Plummer, 20000);
    ForceStats st;
    DistributedBHSolver(W, phys, tree(0.5), dp, 2).compute_accelerations(p, st);
    const GlobalChecks g = global_checks(st);
    NB_CHECK(g.violations == 0u);
    NB_CHECK_LE(g.coverage, kCoverageTol);
    const double recv = allreduce(W, static_cast<double>(st.let_bytes_received), MPI_SUM);
    const ParticleSet f = gather_to_root(W, p);

    ParticleSet q = dist_ics(Distribution::Plummer, 4000);
    DistributedBHSolver solver(W, phys, tree(0.5), dp, 2);
    SimParams sp;
    sp.steps = 5;
    sp.dt = 1e-3;
    Simulation(sp, solver).run(q);
    const ParticleSet s = gather_to_root(W, q);
    if (root())
      std::printf("    shm=%d group=%d comm=%-8s: LET bytes received (all ranks) %.0f\n", cfg.shm, cfg.group,
                  to_string(cfg.comm).c_str(), recv);
    if (first) {
      ref_f = f;
      ref_s = s;
      ref_recv = recv;
      first = false;
      continue;
    }
    // All ranks of this run share one machine: group 0 -> no LET at all, group 1 -> all LETs.
    if (cfg.group == 0) NB_CHECK(recv == 0.0);
    if (cfg.group == 1) NB_CHECK(recv == ref_recv);
    if (cfg.group == 2 && size() >= 3) NB_CHECK(recv > 0.0 && recv < ref_recv);
    if (root()) {
      const ComparisonReport rf = compare_by_id(f, ref_f), rs = compare_by_id(s, ref_s);
      NB_CHECK(rf.accel.max_abs == 0.0 && rf.potential.max_abs == 0.0);
      NB_CHECK(rs.position.max_abs == 0.0 && rs.velocity.max_abs == 0.0);
    }
  }
}

NB_TEST(shm_windows_grow_when_trees_grow) {
  // One solver object: a small system, then a 20x larger one. The segments must be
  // reallocated collectively and the result must equal a solver without windows.
  DistParams dp;
  dp.shm = true;
  DistributedBHSolver solver(W, PhysicsParams{}, tree(0.5), dp, 2);
  ForceStats st;
  ParticleSet small = dist_ics(Distribution::Uniform, 2000);
  solver.compute_accelerations(small, st);
  const std::uint64_t cap_small = st.shm_segment_bytes;
  ParticleSet big = dist_ics(Distribution::Plummer, 40000), ref = big;
  solver.compute_accelerations(big, st);
  const double reallocs = allreduce(W, static_cast<double>(st.shm_reallocations), MPI_MIN);
  NB_CHECK(reallocs >= 2.0);  // first allocation + growth, on every rank
  NB_CHECK(st.shm_segment_bytes > cap_small);
  DistributedBHSolver(W, PhysicsParams{}, tree(0.5), DistParams{}, 2).compute_accelerations(ref, st);
  const ParticleSet a = gather_to_root(W, big), b = gather_to_root(W, ref);
  if (root()) NB_CHECK(compare_by_id(a, b).accel.max_abs == 0.0);
}

NB_TEST(gpu_walks_match_cpu_walks_distributed) {
  // Local tree + received LETs (+ shm peers) walked on the device in the CPU order. Host
  // fallback must be bitwise identical; a real device within the same tolerance as the
  // serial GPU test. Every rank uses the default device (ranks may share one GPU).
  for (bool shm : {false, true}) {
    ParticleSet cpu = dist_ics(Distribution::Plummer, 20000), gpu = cpu;
    DistParams dc, dg;
    dc.shm = dg.shm = shm;
    dg.device = Device::GPU;
    dg.allow_host_fallback = true;
    ForceStats sc, sg;
    DistributedBHSolver(W, PhysicsParams{}, tree(0.5), dc, 2).compute_accelerations(cpu, sc);
    DistributedBHSolver(W, PhysicsParams{}, tree(0.5), dg, 1).compute_accelerations(gpu, sg);
    const int executed = static_cast<int>(allreduce(W, static_cast<double>(sg.executed), MPI_MAX));
    NB_CHECK(global_checks(sg).violations == 0u);
    const ParticleSet a = gather_to_root(W, gpu), b = gather_to_root(W, cpu);
    if (root()) {
      const ComparisonReport r = compare_by_id(a, b);
      std::printf("    shm=%d executed=%s: accel rms_rel %.3e max_rel %.3e\n", shm,
                  to_string(static_cast<Executed>(executed)).c_str(), r.accel.rms_rel, r.accel.max_rel);
      if (executed == static_cast<int>(Executed::HostFallback)) {
        NB_CHECK(r.accel.max_abs == 0.0);
      } else {
        NB_CHECK_LE(r.accel.rms_rel, 1e-12);
        NB_CHECK_LE(r.accel.max_rel, 1e-9);
      }
    }
  }
}

NB_TEST(empty_ranks_are_handled) {
  // 49 coincident particles and one far away, all created on rank 0: the heavy bin forces
  // every rank but one or two to own nothing. Forces must still be exact at theta = 0.
  PhysicsParams phys;
  phys.eps = 0.01;
  for (CommMode comm : {CommMode::Blocking, CommMode::P2P, CommMode::Overlap})
  for (bool shm : {false, true})
  for (double theta : {0.0, 0.5}) {
    ParticleSet p;
    if (root()) {
      for (std::uint64_t i = 0; i < 49; ++i) p.push_back(i, 0.02, 0.2, 0.2, 0.2, 0, 0, 0);
      p.push_back(49, 0.02, -0.4, -0.4, -0.4, 0, 0, 0);
    }
    ForceStats st;
    DistParams dp;
    dp.comm = comm;
    dp.shm = shm;
    DistributedBHSolver(W, phys, tree(theta), dp, 2).compute_accelerations(p, st);
    const RankSpread counts = rank_spread(W, static_cast<double>(p.size()));
    NB_CHECK(allreduce_u64(W, p.size(), MPI_SUM) == 50u);
    NB_CHECK(global_checks(st).violations == 0u);
    NB_CHECK_LE(global_checks(st).coverage, kCoverageTol);
    if (size() >= 2) NB_CHECK(counts.min == 0.0);  // the scenario really has empty ranks
    const ParticleSet all = gather_to_root(W, p);
    if (root() && theta == 0.0) NB_CHECK(within_rounding_of_direct_sum(all, phys));
  }
}

int main(int argc, char** argv) {
  MpiEnv env(argc, argv);
  const int local = nbtest::run_all(env.rank() != 0);
  int global = 0;
  MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
  if (env.rank() == 0) std::printf("MPI ranks: %d -> %s\n", env.size(), global == 0 ? "ALL PASSED" : "FAILURES");
  return global;
}

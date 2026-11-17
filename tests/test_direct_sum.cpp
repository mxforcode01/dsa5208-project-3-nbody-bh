// Direct-sum solver: analytic two-body values, symmetric configurations,
// momentum conservation, interaction counts and thread-count independence.
#include <cmath>

#include "nbody/core/initial_conditions.hpp"
#include "nbody/sim/diagnostics.hpp"
#include "nbody/solvers/direct_sum.hpp"
#include "test_harness.hpp"

using namespace nbody;

namespace {

double norm(double a, double b, double c) { return std::sqrt(a * a + b * b + c * c); }

}  // namespace

NB_TEST(two_particles_match_analytic_values) {
  // m1 = 1 at origin, m2 = 3 at (2,0,0), G = 0.5, eps = 0:
  // a1 = G m2 / r^2 = 0.375 (+x), a2 = -G m1 / r^2 = -0.125, pot1 = -G m2 / r = -0.75, pot2 = -0.25.
  ParticleSet p;
  p.push_back(0, 1.0, 0, 0, 0, 0, 0, 0);
  p.push_back(1, 3.0, 2, 0, 0, 0, 0, 0);
  PhysicsParams phys;
  phys.G = 0.5;
  phys.eps = 0.0;
  ForceStats st;
  DirectSumSolver(phys).compute_accelerations(p, st);
  NB_CHECK_NEAR(p.ax[0], 0.375, 1e-16);
  NB_CHECK_NEAR(p.ax[1], -0.125, 1e-16);
  NB_CHECK(p.ay[0] == 0.0 && p.az[0] == 0.0 && p.ay[1] == 0.0 && p.az[1] == 0.0);
  NB_CHECK_NEAR(p.pot[0], -0.75, 1e-16);
  NB_CHECK_NEAR(p.pot[1], -0.25, 1e-16);
  // Potential energy W = 0.5 sum m_i pot_i = -G m1 m2 / r = -0.75.
  NB_CHECK_NEAR(compute_diagnostics(p).potential, -0.75, 1e-15);
}

NB_TEST(cube_vertices_symmetry) {
  // 8 equal masses on the vertices of [-1,1]^3 plus a centre particle.
  // Centre: forces cancel exactly by symmetry (to rounding). Vertices: equal magnitude,
  // pointing at the centre.
  ParticleSet p;
  std::uint64_t id = 0;
  for (int sx : {-1, 1})
    for (int sy : {-1, 1})
      for (int sz : {-1, 1}) p.push_back(id++, 1.0, sx, sy, sz, 0, 0, 0);
  p.push_back(id, 1.0, 0, 0, 0, 0, 0, 0);
  PhysicsParams phys;
  phys.eps = 0.0;
  ForceStats st;
  DirectSumSolver(phys).compute_accelerations(p, st);

  const double one_vertex = 1.0 / 3.0;  // |a| from one vertex at distance sqrt(3)
  NB_CHECK_LE(norm(p.ax[8], p.ay[8], p.az[8]), 1e-15 * one_vertex);
  const double a0 = norm(p.ax[0], p.ay[0], p.az[0]);
  for (int i = 0; i < 8; ++i) {
    const double ai = norm(p.ax[i], p.ay[i], p.az[i]);
    NB_CHECK_NEAR(ai, a0, 1e-14 * a0);
    // a parallel to -r: |a x r| ~ 0 and a . r < 0
    const double cx = p.ay[i] * p.z[i] - p.az[i] * p.y[i];
    const double cy = p.az[i] * p.x[i] - p.ax[i] * p.z[i];
    const double cz = p.ax[i] * p.y[i] - p.ay[i] * p.x[i];
    NB_CHECK_LE(norm(cx, cy, cz), 1e-14 * ai * std::sqrt(3.0));
    NB_CHECK(p.ax[i] * p.x[i] + p.ay[i] * p.y[i] + p.az[i] * p.z[i] < 0.0);
  }
}

NB_TEST(total_force_vanishes_for_random_system) {
  // Pairwise antisymmetry => sum m_i a_i = 0 up to rounding. Each a_i sums ~N terms, so the
  // expected rounding is ~N * 1e-16 relative to sum m|a|; 1e-12 leaves a large margin at N = 500.
  ICParams ic;
  ic.n_total = 500;
  ic.dist = Distribution::Plummer;
  ParticleSet p;
  generate_particles(ic, 0, 500, p);
  ForceStats st;
  DirectSumSolver(PhysicsParams{}).compute_accelerations(p, st);
  double fx = 0, fy = 0, fz = 0, scale = 0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    fx += p.m[i] * p.ax[i];
    fy += p.m[i] * p.ay[i];
    fz += p.m[i] * p.az[i];
    scale += p.m[i] * norm(p.ax[i], p.ay[i], p.az[i]);
  }
  std::printf("    |sum m a| / sum m|a| = %.3e\n", norm(fx, fy, fz) / scale);
  NB_CHECK_LE(norm(fx, fy, fz), 1e-12 * scale);
}

NB_TEST(interaction_counts) {
  ICParams ic;
  ic.n_total = 100;
  ParticleSet p;
  generate_particles(ic, 0, 100, p);
  ForceStats st;
  DirectSumSolver(PhysicsParams{}).compute_accelerations(p, st);
  NB_CHECK(st.pp == 100u * 99u);
  NB_CHECK(st.pc == 0u);
  bool ok = true;
  for (double w : p.work) ok = ok && w == 99.0;
  NB_CHECK(ok);
}

NB_TEST(thread_count_does_not_change_results) {
  // Each target sums its sources in the same order regardless of the thread count,
  // so results must be bitwise identical.
  ICParams ic;
  ic.n_total = 1500;
  ic.dist = Distribution::Plummer;
  ParticleSet a, b;
  generate_particles(ic, 0, 1500, a);
  b = a;
  ForceStats st;
  DirectSumSolver(PhysicsParams{}, 1).compute_accelerations(a, st);
  DirectSumSolver(PhysicsParams{}, 4).compute_accelerations(b, st);
  NB_CHECK(a.ax == b.ax && a.ay == b.ay && a.az == b.az && a.pot == b.pot);
}

NB_TEST(single_particle_feels_no_force) {
  ParticleSet p;
  p.push_back(0, 1.0, 0.3, 0.2, 0.1, 0, 0, 0);
  ForceStats st;
  DirectSumSolver(PhysicsParams{}).compute_accelerations(p, st);
  NB_CHECK(p.ax[0] == 0.0 && p.ay[0] == 0.0 && p.az[0] == 0.0 && p.pot[0] == 0.0);
  NB_CHECK(st.pp == 0u);
}

NB_TEST_MAIN()

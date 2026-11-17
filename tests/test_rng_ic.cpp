// Random numbers and initial conditions: determinism, range independence,
// distribution properties of the uniform cube and the Plummer sphere.
#include <algorithm>
#include <cmath>
#include <vector>

#include "nbody/core/initial_conditions.hpp"
#include "nbody/core/rng.hpp"
#include "nbody/sim/diagnostics.hpp"
#include "nbody/solvers/direct_sum.hpp"
#include "test_harness.hpp"

using namespace nbody;

namespace {

bool identical(const ParticleSet& a, const ParticleSet& b) {
  return a.id == b.id && a.m == b.m && a.x == b.x && a.y == b.y && a.z == b.z && a.vx == b.vx &&
         a.vy == b.vy && a.vz == b.vz;
}

ICParams params(Distribution d, std::uint64_t n, std::uint64_t seed = 42) {
  ICParams p;
  p.n_total = n;
  p.seed = seed;
  p.dist = d;
  return p;
}

}  // namespace

NB_TEST(rng_is_deterministic_and_depends_on_seed_and_id) {
  ParticleRng a(1, 7), b(1, 7), c(2, 7), d(1, 8);
  const auto va = a.next_u64();
  NB_CHECK(va == b.next_u64());
  NB_CHECK(va != c.next_u64());
  NB_CHECK(va != d.next_u64());
}

NB_TEST(rng_uniform_range_and_mean) {
  // Mean of n U[0,1) draws has std dev sqrt(1/(12 n)) = 9.1e-4 for n = 1e5; allow 5 sigma.
  ParticleRng r(123, 0);
  const int n = 100000;
  double sum = 0.0;
  bool in_range = true;
  for (int i = 0; i < n; ++i) {
    const double u = r.uniform();
    in_range = in_range && u >= 0.0 && u < 1.0;
    sum += u;
  }
  NB_CHECK(in_range);
  NB_CHECK_NEAR(sum / n, 0.5, 5.0 * std::sqrt(1.0 / (12.0 * n)));
}

NB_TEST(ic_independent_of_id_range_split) {
  // Required so that serial and MPI runs simulate exactly the same system.
  for (Distribution dist : {Distribution::Uniform, Distribution::Plummer}) {
    const ICParams ic = params(dist, 1000);
    ParticleSet whole, split;
    generate_particles(ic, 0, 1000, whole);
    generate_particles(ic, 0, 317, split);
    generate_particles(ic, 317, 650, split);
    generate_particles(ic, 650, 1000, split);
    NB_CHECK(identical(whole, split));
  }
}

NB_TEST(ic_seed_changes_particles) {
  ParticleSet a, b;
  generate_particles(params(Distribution::Plummer, 100, 1), 0, 100, a);
  generate_particles(params(Distribution::Plummer, 100, 2), 0, 100, b);
  NB_CHECK(!identical(a, b));
}

NB_TEST(ic_uniform_in_cube_with_unit_mass_and_cold_start) {
  ParticleSet p;
  generate_particles(params(Distribution::Uniform, 5000), 0, 5000, p);
  double mass = 0.0;
  bool ok = true;
  for (std::size_t i = 0; i < p.size(); ++i) {
    mass += p.m[i];
    ok = ok && std::fabs(p.x[i]) <= 0.5 && std::fabs(p.y[i]) <= 0.5 && std::fabs(p.z[i]) <= 0.5;
    ok = ok && p.vx[i] == 0.0 && p.vy[i] == 0.0 && p.vz[i] == 0.0 && p.id[i] == i;
  }
  NB_CHECK(ok);
  NB_CHECK_NEAR(mass, 1.0, 1e-12);
}

NB_TEST(ic_plummer_half_mass_radius) {
  // Plummer half-mass radius r_h = a / sqrt(2^{2/3} - 1) = 1.30477 a. The sample median has
  // std error 1 / (2 f(r_h) sqrt(N)) with f(r_h) = 3 a^2 r_h^2 / (r_h^2 + a^2)^{5/2} = 0.4254,
  // i.e. 0.0083 for N = 20000; allow 5 standard errors.
  const std::uint64_t n = 20000;
  ParticleSet p;
  generate_particles(params(Distribution::Plummer, n), 0, n, p);
  std::vector<double> r(n);
  for (std::size_t i = 0; i < n; ++i) r[i] = std::sqrt(p.x[i] * p.x[i] + p.y[i] * p.y[i] + p.z[i] * p.z[i]);
  std::nth_element(r.begin(), r.begin() + n / 2, r.end());
  const double rh = 1.0 / std::sqrt(std::pow(2.0, 2.0 / 3.0) - 1.0);
  const double fr = 3.0 * rh * rh / std::pow(rh * rh + 1.0, 2.5);
  const double se = 1.0 / (2.0 * fr * std::sqrt(static_cast<double>(n)));
  std::printf("    median r = %.5f, expected %.5f +- %.5f\n", r[n / 2], rh, se);
  NB_CHECK_NEAR(r[n / 2], rh, 5.0 * se);
}

NB_TEST(ic_plummer_is_in_virial_equilibrium) {
  // Virial ratio Q = 2K/|W| = 1 in equilibrium. Sampling noise is O(N^{-1/2}) (~1.6% at
  // N = 4000); 0.1 allows ~6 sigma. Softening 1e-4 << a changes W by << 1%.
  const std::uint64_t n = 4000;
  ParticleSet p;
  generate_particles(params(Distribution::Plummer, n), 0, n, p);
  remove_com_velocity(p);
  PhysicsParams phys;
  phys.eps = 1e-4;
  ForceStats stats;
  DirectSumSolver(phys, 4).compute_accelerations(p, stats);
  const Diagnostics d = compute_diagnostics(p);
  const double q = 2.0 * d.kinetic / std::fabs(d.potential);
  // Analytic Plummer energy with G = M = a = 1 is W = -3 pi / 32 = -0.2945.
  std::printf("    Q = 2K/|W| = %.4f, W = %.4f (analytic -0.2945)\n", q, d.potential);
  NB_CHECK_NEAR(q, 1.0, 0.1);
  NB_CHECK_NEAR(d.potential, -3.0 * 3.14159265358979 / 32.0, 0.03);
}

NB_TEST(remove_com_velocity_zeroes_momentum) {
  ParticleSet p;
  generate_particles(params(Distribution::Plummer, 1000), 0, 1000, p);
  remove_com_velocity(p);
  const Diagnostics d = compute_diagnostics(p);
  NB_CHECK_LE(std::sqrt(d.px * d.px + d.py * d.py + d.pz * d.pz), 1e-14 * d.momentum_scale);
}

NB_TEST_MAIN()

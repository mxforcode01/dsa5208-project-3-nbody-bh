// Time stepping: exact free motion, one KDK step against a hand calculation,
// Kepler orbits (circular sanity check and second-order convergence), and
// mass/momentum conservation of the full simulation loop.
#include <algorithm>
#include <cmath>

#include "nbody/core/initial_conditions.hpp"
#include "nbody/sim/integrator.hpp"
#include "nbody/sim/simulation.hpp"
#include "nbody/solvers/direct_sum.hpp"
#include "test_harness.hpp"

using namespace nbody;

namespace {

constexpr double kPi = 3.14159265358979323846;

// Equal-mass binary (m = 0.5 each, G = M = 1) with relative semi-major axis 1 and
// eccentricity e, started at apocentre. Orbital period T = 2 pi.
ParticleSet binary(double e) {
  const double r = 1.0 + e;
  const double v = std::sqrt((1.0 - e) / (1.0 + e));  // relative apocentre speed
  ParticleSet p;
  p.push_back(0, 0.5, -0.5 * r, 0, 0, 0, -0.5 * v, 0);
  p.push_back(1, 0.5, 0.5 * r, 0, 0, 0, 0.5 * v, 0);
  return p;
}

struct OrbitResult {
  double max_energy_err = 0.0;
  double max_sep_err = 0.0;   // max |separation - initial separation| over sampled steps
  double return_err = 0.0;    // |x(T) - x(0)| of particle 1
};

OrbitResult integrate_one_period(double e, int steps) {
  ParticleSet p = binary(e);
  const double x0 = p.x[1], y0 = p.y[1], z0 = p.z[1];
  const double sep0 = 1.0 + e;
  PhysicsParams phys;
  phys.eps = 0.0;
  DirectSumSolver solver(phys);
  SimParams sp;
  sp.steps = steps;
  sp.dt = 2.0 * kPi / steps;
  sp.diag_every = 1;
  const RunResult r = Simulation(sp, solver).run(p);
  OrbitResult o;
  for (const DiagRecord& rec : r.diag)
    o.max_energy_err = std::max(o.max_energy_err, relative_energy_drift(r.diag.front().d, rec.d));
  const double sep = std::hypot(p.x[1] - p.x[0], p.y[1] - p.y[0], p.z[1] - p.z[0]);
  o.max_sep_err = std::fabs(sep - sep0);
  o.return_err = std::hypot(p.x[1] - x0, p.y[1] - y0, p.z[1] - z0);
  return o;
}

}  // namespace

NB_TEST(free_particle_moves_in_a_straight_line) {
  ParticleSet p;
  p.push_back(0, 1.0, 0.1, -0.2, 0.3, 1.5, -0.5, 0.25);
  DirectSumSolver solver(PhysicsParams{});
  SimParams sp;
  sp.steps = 100;
  sp.dt = 0.01;
  Simulation(sp, solver).run(p);
  // x = x0 + v t with t = 1; only rounding from 100 additions (~1e-15).
  NB_CHECK_NEAR(p.x[0], 0.1 + 1.5, 1e-13);
  NB_CHECK_NEAR(p.y[0], -0.2 - 0.5, 1e-13);
  NB_CHECK_NEAR(p.z[0], 0.3 + 0.25, 1e-13);
  NB_CHECK(p.vx[0] == 1.5 && p.vy[0] == -0.5 && p.vz[0] == 0.25);
}

NB_TEST(one_kdk_step_matches_hand_calculation) {
  const double G = 1.0, eps = 0.1, h = 0.01;
  const double m1 = 1.0, m2 = 2.0;
  double x1 = 0.0, x2 = 1.0, v1 = 0.3, v2 = -0.15;  // motion along x, velocities along y
  double y1 = 0.0, y2 = 0.0;
  // Acceleration of particle 1 due to 2 (and vice versa) for the given positions.
  auto acc = [&](double dx, double dy, double mj, double& axo, double& ayo) {
    const double r2 = dx * dx + dy * dy + eps * eps;
    const double f = G * mj / (r2 * std::sqrt(r2));
    axo = f * dx;
    ayo = f * dy;
  };
  double a1x, a1y, a2x, a2y;
  acc(x2 - x1, y2 - y1, m2, a1x, a1y);
  acc(x1 - x2, y1 - y2, m1, a2x, a2y);
  double u1x = a1x * h / 2, u1y = v1 + a1y * h / 2, u2x = a2x * h / 2, u2y = v2 + a2y * h / 2;
  x1 += u1x * h;
  y1 += u1y * h;
  x2 += u2x * h;
  y2 += u2y * h;
  acc(x2 - x1, y2 - y1, m2, a1x, a1y);
  acc(x1 - x2, y1 - y2, m1, a2x, a2y);
  u1x += a1x * h / 2;
  u1y += a1y * h / 2;
  u2x += a2x * h / 2;
  u2y += a2y * h / 2;

  ParticleSet p;
  p.push_back(0, m1, 0.0, 0.0, 0, 0, 0.3, 0);
  p.push_back(1, m2, 1.0, 0.0, 0, 0, -0.15, 0);
  PhysicsParams phys;
  phys.G = G;
  phys.eps = eps;
  DirectSumSolver solver(phys);
  SimParams sp;
  sp.steps = 1;
  sp.dt = h;
  Simulation(sp, solver).run(p);
  NB_CHECK_NEAR(p.x[0], x1, 1e-15);
  NB_CHECK_NEAR(p.y[0], y1, 1e-15);
  NB_CHECK_NEAR(p.x[1], x2, 1e-15);
  NB_CHECK_NEAR(p.y[1], y2, 1e-15);
  NB_CHECK_NEAR(p.vx[0], u1x, 1e-15);
  NB_CHECK_NEAR(p.vy[0], u1y, 1e-15);
  NB_CHECK_NEAR(p.vx[1], u2x, 1e-15);
  NB_CHECK_NEAR(p.vy[1], u2y, 1e-15);
}

NB_TEST(circular_orbit_one_period) {
  // Sanity bound at dt = T/1000. The O(dt^2) order itself is tested rigorously below; here
  // the measured return error is 4.1e-5 (the phase lag of leapfrog), radius error ~1e-15 and
  // energy error ~4e-10, so 1e-4 is a fixed bound with >2x margin on deterministic results.
  const OrbitResult o = integrate_one_period(0.0, 1000);
  std::printf("    return error %.3e, separation error %.3e, max |dE/E| %.3e\n", o.return_err,
              o.max_sep_err, o.max_energy_err);
  NB_CHECK_LE(o.return_err, 1e-4);
  NB_CHECK_LE(o.max_sep_err, 1e-4);
  NB_CHECK_LE(o.max_energy_err, 1e-4);
}

NB_TEST(eccentric_orbit_converges_at_second_order) {
  // KDK leapfrog is second order: halving dt must reduce the energy error and the
  // one-period return error by ~4. Accept ratios in [3.5, 4.5].
  const double e = 0.5;
  const OrbitResult a = integrate_one_period(e, 500);
  const OrbitResult b = integrate_one_period(e, 1000);
  const OrbitResult c = integrate_one_period(e, 2000);
  const double re1 = a.max_energy_err / b.max_energy_err, re2 = b.max_energy_err / c.max_energy_err;
  const double rx1 = a.return_err / b.return_err, rx2 = b.return_err / c.return_err;
  std::printf("    max |dE/E|: %.3e %.3e %.3e (ratios %.3f %.3f)\n", a.max_energy_err, b.max_energy_err,
              c.max_energy_err, re1, re2);
  std::printf("    return err: %.3e %.3e %.3e (ratios %.3f %.3f)\n", a.return_err, b.return_err, c.return_err,
              rx1, rx2);
  for (double ratio : {re1, re2, rx1, rx2}) {
    NB_CHECK(ratio >= 3.5);
    NB_CHECK(ratio <= 4.5);
  }
}

NB_TEST(simulation_conserves_mass_and_momentum) {
  // Direct sum is pairwise antisymmetric, so total momentum changes only by rounding.
  ICParams ic;
  ic.n_total = 256;
  ic.dist = Distribution::Plummer;
  ParticleSet p;
  generate_particles(ic, 0, 256, p);
  remove_com_velocity(p);
  PhysicsParams phys;
  phys.eps = 0.05;
  DirectSumSolver solver(phys);
  SimParams sp;
  sp.steps = 50;
  sp.dt = 1e-3;
  const RunResult r = Simulation(sp, solver).run(p);
  const Diagnostics& d0 = r.diag.front().d;
  const Diagnostics& d1 = r.diag.back().d;
  std::printf("    momentum drift %.3e, energy drift %.3e\n", relative_momentum_drift(d0, d1),
              relative_energy_drift(d0, d1));
  NB_CHECK(d1.mass == d0.mass);
  NB_CHECK_LE(relative_momentum_drift(d0, d1), 1e-12);
  NB_CHECK(r.diag.size() == 2u && r.diag.back().step == 50);
  NB_CHECK(r.stats.pp == 51u * 256u * 255u);  // initial evaluation + one per step
}

NB_TEST_MAIN()

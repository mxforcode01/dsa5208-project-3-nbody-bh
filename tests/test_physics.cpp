// Pair kernel: values against hand-computed Newtonian/Plummer results, antisymmetry.
#include <cmath>

#include "nbody/core/physics.hpp"
#include "test_harness.hpp"

using namespace nbody;

NB_TEST(unsoftened_kernel_matches_newton) {
  // Source m = 2 at distance 3 on +x: |a| = m / r^2 = 2/9, phi = -m / r = -2/3.
  double ax = 0, ay = 0, az = 0, phi = 0;
  accumulate_pair(3.0, 0.0, 0.0, 2.0, 0.0, ax, ay, az, phi);
  NB_CHECK_NEAR(ax, 2.0 / 9.0, 1e-16);
  NB_CHECK_NEAR(ay, 0.0, 0.0);
  NB_CHECK_NEAR(az, 0.0, 0.0);
  NB_CHECK_NEAR(phi, -2.0 / 3.0, 1e-16);
}

NB_TEST(softened_kernel_matches_plummer_formula) {
  // d = (1, 2, 2), |d| = 3, eps = 0.5: r_s^2 = 9.25; a = m d / r_s^3, phi = -m / r_s.
  const double m = 1.5, eps2 = 0.25, rs = std::sqrt(9.25);
  double ax = 0, ay = 0, az = 0, phi = 0;
  accumulate_pair(1.0, 2.0, 2.0, m, eps2, ax, ay, az, phi);
  const double c = m / (rs * rs * rs);
  NB_CHECK_NEAR(ax, c * 1.0, 1e-16);
  NB_CHECK_NEAR(ay, c * 2.0, 1e-16);
  NB_CHECK_NEAR(az, c * 2.0, 1e-16);
  NB_CHECK_NEAR(phi, -m / rs, 1e-15);
}

NB_TEST(coincident_softened_particles_give_zero_force_finite_potential) {
  double ax = 0, ay = 0, az = 0, phi = 0;
  accumulate_pair(0.0, 0.0, 0.0, 1.0, 0.01, ax, ay, az, phi);  // eps = 0.1
  NB_CHECK(ax == 0.0 && ay == 0.0 && az == 0.0);
  NB_CHECK_NEAR(phi, -10.0, 1e-13);
}

NB_TEST(kernel_is_antisymmetric) {
  // Newton's third law: m_i a_(i<-j) + m_j a_(j<-i) = 0 up to rounding (~1e-16 relative).
  const double mi = 0.7, mj = 1.9, dx = 0.31, dy = -1.27, dz = 0.58, eps2 = 1e-4;
  double aix = 0, aiy = 0, aiz = 0, pi = 0, ajx = 0, ajy = 0, ajz = 0, pj = 0;
  accumulate_pair(dx, dy, dz, mj, eps2, aix, aiy, aiz, pi);
  accumulate_pair(-dx, -dy, -dz, mi, eps2, ajx, ajy, ajz, pj);
  const double scale = mi * std::sqrt(aix * aix + aiy * aiy + aiz * aiz);
  NB_CHECK_NEAR(mi * aix + mj * ajx, 0.0, 4e-16 * scale);
  NB_CHECK_NEAR(mi * aiy + mj * ajy, 0.0, 4e-16 * scale);
  NB_CHECK_NEAR(mi * aiz + mj * ajz, 0.0, 4e-16 * scale);
  NB_CHECK_NEAR(mi * pi, mj * pj, 4e-16 * std::fabs(mi * pi));  // pair energy is symmetric
}

NB_TEST_MAIN()

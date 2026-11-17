#include "nbody/solvers/direct_sum.hpp"

namespace nbody {

void DirectSumSolver::compute_accelerations(ParticleSet& p, ForceStats& stats) {
  const long long n = static_cast<long long>(p.size());
  const double eps2 = phys_.eps * phys_.eps;
  const double G = phys_.G;
  const double* x = p.x.data();
  const double* y = p.y.data();
  const double* z = p.z.data();
  const double* m = p.m.data();
  double* ax = p.ax.data();
  double* ay = p.ay.data();
  double* az = p.az.data();
  double* pot = p.pot.data();
  double* work = p.work.data();
  const int threads = threads_;

#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
  for (long long i = 0; i < n; ++i) {
    const double xi = x[i], yi = y[i], zi = z[i];
    double sx = 0.0, sy = 0.0, sz = 0.0, phi = 0.0;
    // Two loops instead of an i == j branch; source order is still 0..n-1.
    for (long long j = 0; j < i; ++j)
      accumulate_pair(x[j] - xi, y[j] - yi, z[j] - zi, m[j], eps2, sx, sy, sz, phi);
    for (long long j = i + 1; j < n; ++j)
      accumulate_pair(x[j] - xi, y[j] - yi, z[j] - zi, m[j], eps2, sx, sy, sz, phi);
    ax[i] = G * sx;
    ay[i] = G * sy;
    az[i] = G * sz;
    pot[i] = G * phi;
    work[i] = static_cast<double>(n - 1);
  }
  if (n > 1) stats.pp += static_cast<std::uint64_t>(n) * static_cast<std::uint64_t>(n - 1);
}

}  // namespace nbody

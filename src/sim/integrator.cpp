#include "nbody/sim/integrator.hpp"

namespace nbody {

void kick(ParticleSet& p, double h, int threads) {
  const long long n = static_cast<long long>(p.size());
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
  for (long long i = 0; i < n; ++i) {
    p.vx[i] += p.ax[i] * h;
    p.vy[i] += p.ay[i] * h;
    p.vz[i] += p.az[i] * h;
  }
}

void drift(ParticleSet& p, double h, int threads) {
  const long long n = static_cast<long long>(p.size());
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
  for (long long i = 0; i < n; ++i) {
    p.x[i] += p.vx[i] * h;
    p.y[i] += p.vy[i] * h;
    p.z[i] += p.vz[i] * h;
  }
}

}  // namespace nbody

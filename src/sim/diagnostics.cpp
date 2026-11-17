#include "nbody/sim/diagnostics.hpp"

#include <algorithm>
#include <cmath>

namespace nbody {

Diagnostics compute_diagnostics(const ParticleSet& p) {
  Diagnostics d;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const double m = p.m[i];
    const double v2 = p.vx[i] * p.vx[i] + p.vy[i] * p.vy[i] + p.vz[i] * p.vz[i];
    d.mass += m;
    d.px += m * p.vx[i];
    d.py += m * p.vy[i];
    d.pz += m * p.vz[i];
    d.momentum_scale += m * std::sqrt(v2);
    d.kinetic += 0.5 * m * v2;
    d.potential += 0.5 * m * p.pot[i];
  }
  return d;
}

double relative_energy_drift(const Diagnostics& d0, const Diagnostics& d) {
  const double e0 = d0.total_energy();
  const double diff = std::fabs(d.total_energy() - e0);
  if (e0 == 0.0) return diff == 0.0 ? 0.0 : INFINITY;
  return diff / std::fabs(e0);
}

double relative_momentum_drift(const Diagnostics& d0, const Diagnostics& d) {
  const double dx = d.px - d0.px, dy = d.py - d0.py, dz = d.pz - d0.pz;
  const double diff = std::sqrt(dx * dx + dy * dy + dz * dz);
  const double scale = std::max(d0.momentum_scale, d.momentum_scale);
  if (scale == 0.0) return diff == 0.0 ? 0.0 : INFINITY;
  return diff / scale;
}

}  // namespace nbody

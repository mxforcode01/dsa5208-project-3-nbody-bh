#pragma once

#include <cmath>

#include "nbody/core/device.hpp"

namespace nbody {

struct PhysicsParams {
  double G = 1.0;     // gravitational constant (N-body units by default)
  double eps = 1e-3;  // Plummer softening length
};

// Adds the softened contribution of a source of mass mj at displacement
// d = r_j - r_i to the target's acceleration and potential, WITHOUT the
// factor G (callers multiply once at the end):
//   a   += mj * d / (|d|^2 + eps^2)^{3/2}
//   phi -= mj     / (|d|^2 + eps^2)^{1/2}
// With eps2 == 0 the caller must never pass d == 0 (self-interaction).
NBODY_DEVICE_BEGIN
inline void accumulate_pair(double dx, double dy, double dz, double mj, double eps2,
                            double& ax, double& ay, double& az, double& phi) {
  const double r2 = dx * dx + dy * dy + dz * dz + eps2;
  const double rinv = 1.0 / std::sqrt(r2);
  const double mr3 = mj * rinv * rinv * rinv;
  ax += mr3 * dx;
  ay += mr3 * dy;
  az += mr3 * dz;
  phi -= mj * rinv;
}
NBODY_DEVICE_END

}  // namespace nbody

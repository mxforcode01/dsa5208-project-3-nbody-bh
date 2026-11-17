#pragma once

#include <algorithm>
#include <cmath>

#include "nbody/core/device.hpp"
#include "nbody/tree/octree.hpp"

namespace nbody {

// Multipole acceptance criterion (offset-corrected Barnes-Hut, Barnes 1994):
// accept a cell for target r if |r - com| > 2*half/theta + delta AND r is not
// inside the cell. The second condition stops a cell from ever acting on a
// particle it contains (possible for theta > 1/sqrt(3) without it).

NBODY_DEVICE_BEGIN
inline bool cell_contains(const Node& n, double x, double y, double z) {
  return std::fabs(x - n.cx) <= n.half && std::fabs(y - n.cy) <= n.half && std::fabs(z - n.cz) <= n.half;
}

inline bool mac_accept_point(const Node& n, double x, double y, double z) {
  const double dx = x - n.mx, dy = y - n.my, dz = z - n.mz;
  return dx * dx + dy * dy + dz * dz > n.ropen2 && !cell_contains(n, x, y, z);
}
NBODY_DEVICE_END

inline bool cell_overlaps_box(const Node& n, const AABB& b) {
  return n.cx - n.half <= b.hi[0] && n.cx + n.half >= b.lo[0] && n.cy - n.half <= b.hi[1] &&
         n.cy + n.half >= b.lo[1] && n.cz - n.half <= b.hi[2] && n.cz + n.half >= b.lo[2];
}

// Squared distance from point (x, y, z) to the closest point of box b (0 if inside).
inline double distance2_to_box(const AABB& b, double x, double y, double z) {
  const double dx = std::max({b.lo[0] - x, 0.0, x - b.hi[0]});
  const double dy = std::max({b.lo[1] - y, 0.0, y - b.hi[1]});
  const double dz = std::max({b.lo[2] - z, 0.0, z - b.hi[2]});
  return dx * dx + dy * dy + dz * dz;
}

// Conservative box version used to build Locally Essential Trees: if it accepts,
// mac_accept_point accepts for EVERY point of b (every point is at least as far from
// the centre of mass, and none is inside the cell because the cell and b are disjoint).
inline bool mac_accept_box(const Node& n, const AABB& b) {
  return distance2_to_box(b, n.mx, n.my, n.mz) > n.ropen2 && !cell_overlaps_box(n, b);
}

}  // namespace nbody

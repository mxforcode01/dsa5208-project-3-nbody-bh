#pragma once

#include <string>

#include "nbody/core/particles.hpp"

namespace nbody {

// Binary snapshot in host byte order:
//   8-byte magic "NBSNAP01", uint64 n, then n records of
//   { uint64 id; double m, x, y, z, vx, vy, vz, ax, ay, az, pot; }.
// Used to store direct-sum references and compare versions by particle id.
void write_snapshot(const std::string& path, const ParticleSet& p);
ParticleSet read_snapshot(const std::string& path);

}  // namespace nbody

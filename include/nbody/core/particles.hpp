#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace nbody {

// Structure-of-arrays particle storage shared by every implementation version.
// Accelerations and potentials are outputs of a force evaluation; they are
// recomputed every step and never migrated between ranks.
struct ParticleSet {
  std::vector<std::uint64_t> id;
  std::vector<double> m;
  std::vector<double> x, y, z;
  std::vector<double> vx, vy, vz;
  std::vector<double> ax, ay, az;
  std::vector<double> pot;   // specific potential (energy per unit mass)
  std::vector<double> work;  // interactions in the last force evaluation (load-balance weight)

  std::size_t size() const { return id.size(); }
  void resize(std::size_t n);
  void reserve(std::size_t n);
  void clear();
  // Appends a particle with zero acceleration, potential and work.
  void push_back(std::uint64_t pid, double mass, double px, double py, double pz,
                 double pvx, double pvy, double pvz);
};

}  // namespace nbody

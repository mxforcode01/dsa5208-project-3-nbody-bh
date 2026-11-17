#pragma once

#include <cstdint>

namespace nbody {

inline std::uint64_t splitmix64(std::uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

// Counter-based generator: the random stream of a particle depends only on
// (seed, id), so initial conditions are identical for any MPI rank count or
// any split of the id range.
class ParticleRng {
 public:
  ParticleRng(std::uint64_t seed, std::uint64_t id)
      : state_(splitmix64(seed) ^ splitmix64(id ^ 0xD1B54A32D192ED03ULL)) {}

  std::uint64_t next_u64() {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
  }

  // Uniform double in [0, 1) with 53 random bits.
  double uniform() { return static_cast<double>(next_u64() >> 11) * 0x1.0p-53; }

 private:
  std::uint64_t state_;
};

}  // namespace nbody

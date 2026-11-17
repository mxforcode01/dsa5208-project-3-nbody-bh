#pragma once

#include <cstdint>
#include <string>

#include "nbody/core/particles.hpp"

namespace nbody {

enum class Distribution { Uniform, Plummer };

Distribution parse_distribution(const std::string& name);  // "uniform" | "plummer"
std::string to_string(Distribution d);

struct ICParams {
  std::uint64_t n_total = 1000;
  std::uint64_t seed = 42;
  Distribution dist = Distribution::Uniform;
  double scale = 1.0;       // uniform: cube side length; plummer: scale radius a
  double total_mass = 1.0;  // every particle has mass total_mass / n_total
  double G = 1.0;           // used for Plummer equilibrium velocities
};

// Appends the particles with ids in [id_begin, id_end) to p.
//  - Uniform: positions uniform in [-scale/2, scale/2)^3, zero velocities (cold start).
//  - Plummer: Aarseth-Henon-Wielen (1974) sampling of an isotropic Plummer sphere in
//    virial equilibrium; radii beyond 100*scale are resampled (mass fraction ~1.5e-6).
// Particle `id` is generated from ParticleRng(seed, id) only, so the result for a
// given id never depends on the requested range.
void generate_particles(const ICParams& params, std::uint64_t id_begin, std::uint64_t id_end,
                        ParticleSet& p);

// Subtracts the mass-weighted mean velocity (serial version; the MPI version
// performs the same operation with a global reduction).
void remove_com_velocity(ParticleSet& p);

}  // namespace nbody

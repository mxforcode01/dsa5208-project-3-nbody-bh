#include "nbody/core/initial_conditions.hpp"

#include <cmath>
#include <stdexcept>

#include "nbody/core/rng.hpp"

namespace nbody {

namespace {

constexpr double kPi = 3.14159265358979323846;

// Isotropic unit vector scaled by `len`.
void random_direction(ParticleRng& rng, double len, double& x, double& y, double& z) {
  const double cz = 2.0 * rng.uniform() - 1.0;
  const double phi = 2.0 * kPi * rng.uniform();
  const double s = std::sqrt(1.0 - cz * cz);
  x = len * s * std::cos(phi);
  y = len * s * std::sin(phi);
  z = len * cz;
}

}  // namespace

Distribution parse_distribution(const std::string& name) {
  if (name == "uniform") return Distribution::Uniform;
  if (name == "plummer") return Distribution::Plummer;
  throw std::invalid_argument("unknown distribution '" + name + "' (expected uniform|plummer)");
}

std::string to_string(Distribution d) {
  return d == Distribution::Uniform ? "uniform" : "plummer";
}

void generate_particles(const ICParams& params, std::uint64_t id_begin, std::uint64_t id_end,
                        ParticleSet& p) {
  if (id_end < id_begin || id_end > params.n_total)
    throw std::invalid_argument("generate_particles: invalid id range");
  const double mass = params.total_mass / static_cast<double>(params.n_total);
  const double a = params.scale;
  p.reserve(p.size() + (id_end - id_begin));

  for (std::uint64_t id = id_begin; id < id_end; ++id) {
    ParticleRng rng(params.seed, id);
    double x, y, z, vx = 0.0, vy = 0.0, vz = 0.0;

    if (params.dist == Distribution::Uniform) {
      x = (rng.uniform() - 0.5) * a;
      y = (rng.uniform() - 0.5) * a;
      z = (rng.uniform() - 0.5) * a;
    } else {
      // Radius from the inverted cumulative mass profile M(r)/M = r^3/(r^2+a^2)^{3/2}.
      double r;
      do {
        double u;
        do { u = rng.uniform(); } while (u == 0.0);
        r = a / std::sqrt(std::pow(u, -2.0 / 3.0) - 1.0);
      } while (!(r <= 100.0 * a));
      random_direction(rng, r, x, y, z);

      // Speed v = q * v_esc with q drawn from g(q) = q^2 (1-q^2)^{7/2} by rejection
      // (max of g is < 0.1), v_esc = sqrt(2 G M / sqrt(r^2 + a^2)).
      double q, g;
      do {
        q = rng.uniform();
        g = 0.1 * rng.uniform();
      } while (g > q * q * std::pow(1.0 - q * q, 3.5));
      const double v_esc = std::sqrt(2.0 * params.G * params.total_mass / std::sqrt(r * r + a * a));
      random_direction(rng, q * v_esc, vx, vy, vz);
    }
    p.push_back(id, mass, x, y, z, vx, vy, vz);
  }
}

void remove_com_velocity(ParticleSet& p) {
  double mt = 0.0, px = 0.0, py = 0.0, pz = 0.0;
  for (std::size_t i = 0; i < p.size(); ++i) {
    mt += p.m[i];
    px += p.m[i] * p.vx[i];
    py += p.m[i] * p.vy[i];
    pz += p.m[i] * p.vz[i];
  }
  if (mt <= 0.0) return;
  px /= mt;
  py /= mt;
  pz /= mt;
  for (std::size_t i = 0; i < p.size(); ++i) {
    p.vx[i] -= px;
    p.vy[i] -= py;
    p.vz[i] -= pz;
  }
}

}  // namespace nbody

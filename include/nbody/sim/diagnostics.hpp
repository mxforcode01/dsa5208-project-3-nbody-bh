#pragma once

#include "nbody/core/particles.hpp"

namespace nbody {

struct Diagnostics {
  double mass = 0.0;
  double px = 0.0, py = 0.0, pz = 0.0;  // total momentum
  double momentum_scale = 0.0;          // sum_i m_i |v_i|, normalises momentum drift
  double kinetic = 0.0;
  double potential = 0.0;               // 0.5 * sum_i m_i pot_i
  double total_energy() const { return kinetic + potential; }
};

// Uses p.pot from the most recent force evaluation, so call it only when
// positions and potentials are in sync (i.e. right after a force evaluation).
Diagnostics compute_diagnostics(const ParticleSet& p);

// |E - E0| / |E0| (0 if both are 0).
double relative_energy_drift(const Diagnostics& d0, const Diagnostics& d);
// |P - P0| / max(sum m|v| at t0, sum m|v| at t) (0 if both are 0).
double relative_momentum_drift(const Diagnostics& d0, const Diagnostics& d);

}  // namespace nbody

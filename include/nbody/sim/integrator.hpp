#pragma once

#include "nbody/core/particles.hpp"

namespace nbody {

// Building blocks of kick-drift-kick leapfrog (velocity Verlet):
//   v += a * h/2;  x += v * h;  a = F(x);  v += a * h/2
void kick(ParticleSet& p, double h, int threads = 1);   // v += a * h
void drift(ParticleSet& p, double h, int threads = 1);  // x += v * h

}  // namespace nbody

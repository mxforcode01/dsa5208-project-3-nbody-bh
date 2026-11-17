#include "nbody/core/particles.hpp"

namespace nbody {

void ParticleSet::resize(std::size_t n) {
  id.resize(n);
  for (auto* v : {&m, &x, &y, &z, &vx, &vy, &vz, &ax, &ay, &az, &pot, &work}) v->resize(n, 0.0);
}

void ParticleSet::reserve(std::size_t n) {
  id.reserve(n);
  for (auto* v : {&m, &x, &y, &z, &vx, &vy, &vz, &ax, &ay, &az, &pot, &work}) v->reserve(n);
}

void ParticleSet::clear() { resize(0); }

void ParticleSet::push_back(std::uint64_t pid, double mass, double px, double py, double pz,
                            double pvx, double pvy, double pvz) {
  id.push_back(pid);
  m.push_back(mass);
  x.push_back(px);
  y.push_back(py);
  z.push_back(pz);
  vx.push_back(pvx);
  vy.push_back(pvy);
  vz.push_back(pvz);
  for (auto* v : {&ax, &ay, &az, &pot, &work}) v->push_back(0.0);
}

}  // namespace nbody

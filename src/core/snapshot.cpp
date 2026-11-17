#include "nbody/core/snapshot.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace nbody {

namespace {
constexpr char kMagic[8] = {'N', 'B', 'S', 'N', 'A', 'P', '0', '1'};
}

void write_snapshot(const std::string& path, const ParticleSet& p) {
  std::ofstream out(path, std::ios::binary | std::ios::trunc);
  if (!out) throw std::runtime_error("cannot open snapshot '" + path + "' for writing");
  out.write(kMagic, sizeof(kMagic));
  const std::uint64_t n = p.size();
  out.write(reinterpret_cast<const char*>(&n), sizeof(n));
  for (std::size_t i = 0; i < p.size(); ++i) {
    const double rec[11] = {p.m[i], p.x[i], p.y[i], p.z[i], p.vx[i], p.vy[i],
                            p.vz[i], p.ax[i], p.ay[i], p.az[i], p.pot[i]};
    out.write(reinterpret_cast<const char*>(&p.id[i]), sizeof(std::uint64_t));
    out.write(reinterpret_cast<const char*>(rec), sizeof(rec));
  }
  if (!out) throw std::runtime_error("error while writing snapshot '" + path + "'");
}

ParticleSet read_snapshot(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open snapshot '" + path + "'");
  char magic[8];
  std::uint64_t n = 0;
  in.read(magic, sizeof(magic));
  in.read(reinterpret_cast<char*>(&n), sizeof(n));
  if (!in || std::memcmp(magic, kMagic, sizeof(kMagic)) != 0)
    throw std::runtime_error("'" + path + "' is not an NBSNAP01 snapshot");
  ParticleSet p;
  p.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    double rec[11];
    in.read(reinterpret_cast<char*>(&p.id[i]), sizeof(std::uint64_t));
    in.read(reinterpret_cast<char*>(rec), sizeof(rec));
    if (!in) throw std::runtime_error("snapshot '" + path + "' is truncated");
    p.m[i] = rec[0];
    p.x[i] = rec[1];
    p.y[i] = rec[2];
    p.z[i] = rec[3];
    p.vx[i] = rec[4];
    p.vy[i] = rec[5];
    p.vz[i] = rec[6];
    p.ax[i] = rec[7];
    p.ay[i] = rec[8];
    p.az[i] = rec[9];
    p.pot[i] = rec[10];
  }
  return p;
}

}  // namespace nbody

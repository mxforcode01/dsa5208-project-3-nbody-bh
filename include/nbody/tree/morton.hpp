#pragma once

#include <cstdint>

#include "nbody/tree/octree.hpp"

namespace nbody {

constexpr int kMortonBits = 21;  // bits per dimension -> 63-bit keys

// Spreads the low 21 bits of v so that bit b moves to bit 3b.
inline std::uint64_t morton_expand_bits(std::uint64_t v) {
  v &= 0x1FFFFFULL;
  v = (v | v << 32) & 0x1F00000000FFFFULL;
  v = (v | v << 16) & 0x1F0000FF0000FFULL;
  v = (v | v << 8) & 0x100F00F00F00F00FULL;
  v = (v | v << 4) & 0x10C30C30C30C30C3ULL;
  v = (v | v << 2) & 0x1249249249249249ULL;
  return v;
}

// Quantises v in [lo, lo + width) to 21 bits (values outside are clamped).
inline std::uint64_t morton_quantise(double v, double lo, double inv_width) {
  double t = (v - lo) * inv_width * 2097152.0;  // 2^21
  if (!(t >= 0.0)) t = 0.0;
  if (t > 2097151.0) t = 2097151.0;
  return static_cast<std::uint64_t>(t);
}

// 63-bit Morton (Z-order) key of a point relative to cube c. Per level, x is the most
// significant of the three bits, matching the octree's octant numbering (bit 2 = x).
inline std::uint64_t morton_key(double x, double y, double z, const Cube& c) {
  const double inv = 1.0 / (2.0 * c.half);
  const std::uint64_t qx = morton_quantise(x, c.cx - c.half, inv);
  const std::uint64_t qy = morton_quantise(y, c.cy - c.half, inv);
  const std::uint64_t qz = morton_quantise(z, c.cz - c.half, inv);
  return (morton_expand_bits(qx) << 2) | (morton_expand_bits(qy) << 1) | morton_expand_bits(qz);
}

}  // namespace nbody

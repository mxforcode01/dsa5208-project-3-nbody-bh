#pragma once

#include <mpi.h>

#include <cstdint>
#include <vector>

#include "nbody/core/particles.hpp"
#include "nbody/tree/octree.hpp"

namespace nbody {

// Padded cube containing every particle on every rank (one MPI_Allreduce).
Cube global_cube(MPI_Comm comm, const ParticleSet& p);

// Work-weighted Morton space-filling-curve decomposition with exact splitters.
// Particle weights are the interactions of the last force evaluation (1 before the first).
// Splitter r (r = 1..P-1) is the smallest 63-bit Morton key s_r with
//   W(key < s_r) >= r * W_total / P,
// found for all r at once by bisection on the key: each of the 64 rounds evaluates
// W(key < candidate) from the locally sorted keys and their prefix-summed weights and
// combines the P-1 values with one MPI_Allreduce. Rank r owns keys in [s_r, s_{r+1}).
// Only particles with identical keys are indivisible, so a rank can own nothing only
// when one key carries more than a 1/P share of the weight.
//
// (A first version split at the boundaries of a fixed 8^6-bin histogram; it reached a
// work imbalance of 4.0 on 8 ranks for a Plummer sphere because the dense core falls
// into a few bins of a cube that must also contain the outliers.)
class Decomposition {
 public:
  void compute(MPI_Comm comm, const ParticleSet& p, const Cube& cube);
  int owner(std::uint64_t key) const;  // key from morton_key(..., cube())
  int owner_of(const ParticleSet& p, std::size_t i) const;
  bool valid() const { return !splitters_.empty(); }
  const Cube& cube() const { return cube_; }
  const std::vector<std::uint64_t>& splitters() const { return splitters_; }  // size P+1
  double predicted_imbalance() const { return predicted_imbalance_; }       // max / mean of predicted weight

 private:
  Cube cube_;
  std::vector<std::uint64_t> splitters_;  // s_0 = 0, ..., s_P = 2^63
  double predicted_imbalance_ = 1.0;
};

}  // namespace nbody

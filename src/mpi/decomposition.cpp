#include "nbody/mpi/decomposition.hpp"

#include <algorithm>
#include <limits>
#include <utility>

#include "nbody/mpi/mpi_env.hpp"
#include "nbody/tree/morton.hpp"

namespace nbody {

Cube global_cube(MPI_Comm comm, const ParticleSet& p) {
  // One MIN reduction over (lo, -hi). Empty ranks contribute (+inf, +inf).
  const double inf = std::numeric_limits<double>::infinity();
  double v[6] = {inf, inf, inf, inf, inf, inf};
  for (std::size_t i = 0; i < p.size(); ++i) {
    v[0] = std::min(v[0], p.x[i]);
    v[1] = std::min(v[1], p.y[i]);
    v[2] = std::min(v[2], p.z[i]);
    v[3] = std::min(v[3], -p.x[i]);
    v[4] = std::min(v[4], -p.y[i]);
    v[5] = std::min(v[5], -p.z[i]);
  }
  double g[6];
  MPI_Allreduce(v, g, 6, MPI_DOUBLE, MPI_MIN, comm);
  const double lo[3] = {g[0], g[1], g[2]}, hi[3] = {-g[3], -g[4], -g[5]};
  return cube_from_bounds(lo, hi);
}

void Decomposition::compute(MPI_Comm comm, const ParticleSet& p, const Cube& cube) {
  const int P = comm_size(comm);
  cube_ = cube;
  constexpr std::uint64_t kEnd = 1ULL << 63;  // one past the largest 63-bit key

  // Local keys sorted, with prefix sums of their weights: W_local(key < k) in O(log n).
  std::vector<std::pair<std::uint64_t, double>> kw(p.size());
  for (std::size_t i = 0; i < p.size(); ++i)
    kw[i] = {morton_key(p.x[i], p.y[i], p.z[i], cube_), p.work[i] > 0.0 ? p.work[i] : 1.0};
  std::sort(kw.begin(), kw.end());
  std::vector<std::uint64_t> keys(kw.size());
  std::vector<double> prefix(kw.size() + 1, 0.0);
  for (std::size_t i = 0; i < kw.size(); ++i) {
    keys[i] = kw[i].first;
    prefix[i + 1] = prefix[i] + kw[i].second;
  }
  auto below = [&](std::uint64_t k) {
    return prefix[std::lower_bound(keys.begin(), keys.end(), k) - keys.begin()];
  };
  const double total = allreduce(comm, prefix.back(), MPI_SUM);

  // Bisection for all P-1 splitters in lockstep: smallest k with W(< k) >= r W / P.
  const int m = P - 1;
  std::vector<std::uint64_t> lo(m, 0), hi(m, kEnd), mid(m);
  std::vector<double> w(m);
  for (int round = 0; round < 64; ++round) {
    for (int r = 0; r < m; ++r) {
      mid[r] = lo[r] + (hi[r] - lo[r]) / 2;
      w[r] = below(mid[r]);
    }
    if (m > 0) MPI_Allreduce(MPI_IN_PLACE, w.data(), m, MPI_DOUBLE, MPI_SUM, comm);
    for (int r = 0; r < m; ++r) {
      if (lo[r] >= hi[r]) continue;
      if (w[r] >= total * (r + 1) / P) hi[r] = mid[r];
      else lo[r] = mid[r] + 1;
    }
  }

  splitters_.assign(P + 1, kEnd);
  splitters_[0] = 0;
  for (int r = 0; r < m; ++r) splitters_[r + 1] = lo[r];

  // Predicted weight of each rank: differences of W(< s_r).
  std::vector<double> at(P + 1, 0.0);
  for (int r = 1; r < P; ++r) at[r] = below(splitters_[r]);
  if (m > 0) MPI_Allreduce(MPI_IN_PLACE, at.data() + 1, m, MPI_DOUBLE, MPI_SUM, comm);
  at[P] = total;
  double max_w = 0.0;
  for (int r = 0; r < P; ++r) max_w = std::max(max_w, at[r + 1] - at[r]);
  predicted_imbalance_ = total > 0.0 ? max_w / (total / P) : 1.0;
}

int Decomposition::owner(std::uint64_t key) const {
  // Number of splitters s_1..s_{P-1} that are <= key.
  const auto it = std::upper_bound(splitters_.begin() + 1, splitters_.end() - 1, key);
  return static_cast<int>(it - (splitters_.begin() + 1));
}

int Decomposition::owner_of(const ParticleSet& p, std::size_t i) const {
  return owner(morton_key(p.x[i], p.y[i], p.z[i], cube_));
}

}  // namespace nbody

#pragma once

#include <mpi.h>

#include <cstdint>

namespace nbody {

// RAII wrapper: MPI_Init_thread with MPI_THREAD_FUNNELED (only the master thread
// calls MPI; all MPI calls in this code are made outside OpenMP parallel regions).
// Aborts if the library provides a lower thread level.
class MpiEnv {
 public:
  MpiEnv(int& argc, char**& argv);
  ~MpiEnv();
  MpiEnv(const MpiEnv&) = delete;
  MpiEnv& operator=(const MpiEnv&) = delete;
  int rank() const { return rank_; }
  int size() const { return size_; }
  int thread_level() const { return provided_; }

 private:
  int rank_ = 0, size_ = 1, provided_ = 0;
};

inline int comm_rank(MPI_Comm c) {
  int r = 0;
  MPI_Comm_rank(c, &r);
  return r;
}

inline int comm_size(MPI_Comm c) {
  int s = 1;
  MPI_Comm_size(c, &s);
  return s;
}

inline double allreduce(MPI_Comm c, double v, MPI_Op op) {
  double r = 0.0;
  MPI_Allreduce(&v, &r, 1, MPI_DOUBLE, op, c);
  return r;
}

inline std::uint64_t allreduce_u64(MPI_Comm c, std::uint64_t v, MPI_Op op) {
  std::uint64_t r = 0;
  MPI_Allreduce(&v, &r, 1, MPI_UINT64_T, op, c);
  return r;
}

// min / mean / max of a per-rank value, available on every rank.
struct RankSpread {
  double min = 0.0, mean = 0.0, max = 0.0;
  double imbalance() const { return mean > 0.0 ? max / mean : 1.0; }  // max / mean
};

inline RankSpread rank_spread(MPI_Comm c, double v) {
  RankSpread s;
  s.min = allreduce(c, v, MPI_MIN);
  s.max = allreduce(c, v, MPI_MAX);
  s.mean = allreduce(c, v, MPI_SUM) / comm_size(c);
  return s;
}

}  // namespace nbody

#include "nbody/mpi/mpi_env.hpp"

#include <cstdio>

namespace nbody {

MpiEnv::MpiEnv(int& argc, char**& argv) {
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided_);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &size_);
  if (provided_ < MPI_THREAD_FUNNELED) {
    if (rank_ == 0)
      std::fprintf(stderr, "error: MPI library provides thread level %d < MPI_THREAD_FUNNELED\n", provided_);
    MPI_Abort(MPI_COMM_WORLD, 3);
  }
}

MpiEnv::~MpiEnv() { MPI_Finalize(); }

}  // namespace nbody

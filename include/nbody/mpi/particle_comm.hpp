#pragma once

#include <mpi.h>

#include <cstdint>
#include <vector>

#include "nbody/core/initial_conditions.hpp"
#include "nbody/core/particles.hpp"

namespace nbody {

// Rank r generates the particles with ids [r*N/P, (r+1)*N/P). Because every particle
// depends only on (seed, id), the global system is identical to the serial one.
void generate_particles_distributed(MPI_Comm comm, const ICParams& params, ParticleSet& p);

// Global version of remove_com_velocity (one MPI_Allreduce).
void remove_com_velocity_global(MPI_Comm comm, ParticleSet& p);

// Sends particle i to rank dest[i] with MPI_Alltoall (counts) + MPI_Alltoallv (data).
// p is replaced by the received particles, ordered by source rank and then by their
// order on the source. Accelerations and potentials are reset; work is carried along.
// Returns the number of particles this rank sent to other ranks.
std::uint64_t migrate_particles(MPI_Comm comm, ParticleSet& p, const std::vector<int>& dest);

// Collects every particle (including accelerations, potentials and work) on `root`.
// Returns the full set on root and an empty set elsewhere.
ParticleSet gather_to_root(MPI_Comm comm, const ParticleSet& p, int root = 0);

}  // namespace nbody

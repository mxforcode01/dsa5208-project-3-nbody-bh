#include "nbody/mpi/particle_comm.hpp"

#include <climits>
#include <stdexcept>

#include "nbody/mpi/mpi_env.hpp"

namespace nbody {

namespace {

// Wire formats (sent as raw bytes; all ranks run the same binary on the same architecture).
struct MigrationRecord {
  std::uint64_t id;
  double m, x, y, z, vx, vy, vz, work;
};

struct FullRecord {
  std::uint64_t id;
  double m, x, y, z, vx, vy, vz, ax, ay, az, pot, work;
};

int checked_int(std::uint64_t v, const char* what) {
  if (v > static_cast<std::uint64_t>(INT_MAX)) throw std::runtime_error(std::string(what) + ": message too large");
  return static_cast<int>(v);
}

// Displacements from counts; returns the total.
std::uint64_t prefix(const std::vector<int>& counts, std::vector<int>& displs) {
  displs.assign(counts.size(), 0);
  std::uint64_t total = 0;
  for (std::size_t i = 0; i < counts.size(); ++i) {
    displs[i] = checked_int(total, "displacement");
    total += static_cast<std::uint64_t>(counts[i]);
  }
  return total;
}

}  // namespace

void generate_particles_distributed(MPI_Comm comm, const ICParams& params, ParticleSet& p) {
  const auto r = static_cast<std::uint64_t>(comm_rank(comm));
  const auto P = static_cast<std::uint64_t>(comm_size(comm));
  const std::uint64_t n = params.n_total;
  generate_particles(params, r * n / P, (r + 1) * n / P, p);
}

void remove_com_velocity_global(MPI_Comm comm, ParticleSet& p) {
  double s[4] = {0.0, 0.0, 0.0, 0.0};
  for (std::size_t i = 0; i < p.size(); ++i) {
    s[0] += p.m[i];
    s[1] += p.m[i] * p.vx[i];
    s[2] += p.m[i] * p.vy[i];
    s[3] += p.m[i] * p.vz[i];
  }
  MPI_Allreduce(MPI_IN_PLACE, s, 4, MPI_DOUBLE, MPI_SUM, comm);
  if (s[0] <= 0.0) return;
  for (std::size_t i = 0; i < p.size(); ++i) {
    p.vx[i] -= s[1] / s[0];
    p.vy[i] -= s[2] / s[0];
    p.vz[i] -= s[3] / s[0];
  }
}

std::uint64_t migrate_particles(MPI_Comm comm, ParticleSet& p, const std::vector<int>& dest) {
  const int P = comm_size(comm), me = comm_rank(comm);
  if (dest.size() != p.size()) throw std::invalid_argument("migrate_particles: dest size mismatch");

  std::vector<int> send_count(P, 0);
  for (int d : dest) {
    if (d < 0 || d >= P) throw std::invalid_argument("migrate_particles: invalid destination rank");
    ++send_count[d];
  }
  std::vector<int> send_displ;
  prefix(send_count, send_displ);
  std::vector<MigrationRecord> send(p.size());
  std::vector<int> fill = send_displ;
  for (std::size_t i = 0; i < p.size(); ++i)
    send[fill[dest[i]]++] = {p.id[i], p.m[i], p.x[i], p.y[i], p.z[i], p.vx[i], p.vy[i], p.vz[i], p.work[i]};

  std::vector<int> recv_count(P, 0);
  MPI_Alltoall(send_count.data(), 1, MPI_INT, recv_count.data(), 1, MPI_INT, comm);
  std::vector<int> recv_displ;
  const std::uint64_t n_recv = prefix(recv_count, recv_displ);
  std::vector<MigrationRecord> recv(n_recv);

  MPI_Datatype rec_type;
  MPI_Type_contiguous(static_cast<int>(sizeof(MigrationRecord)), MPI_BYTE, &rec_type);
  MPI_Type_commit(&rec_type);
  MPI_Alltoallv(send.data(), send_count.data(), send_displ.data(), rec_type, recv.data(), recv_count.data(),
                recv_displ.data(), rec_type, comm);
  MPI_Type_free(&rec_type);

  p.clear();
  p.reserve(n_recv);
  for (const MigrationRecord& r : recv) {
    p.push_back(r.id, r.m, r.x, r.y, r.z, r.vx, r.vy, r.vz);
    p.work.back() = r.work;
  }
  return static_cast<std::uint64_t>(dest.size() - static_cast<std::size_t>(send_count[me]));
}

ParticleSet gather_to_root(MPI_Comm comm, const ParticleSet& p, int root) {
  const int P = comm_size(comm), me = comm_rank(comm);
  std::vector<FullRecord> mine(p.size());
  for (std::size_t i = 0; i < p.size(); ++i)
    mine[i] = {p.id[i], p.m[i],  p.x[i],  p.y[i],  p.z[i],   p.vx[i],  p.vy[i],
               p.vz[i], p.ax[i], p.ay[i], p.az[i], p.pot[i], p.work[i]};
  const int bytes = checked_int(mine.size() * sizeof(FullRecord), "gather");
  std::vector<int> counts(P, 0), displs;
  MPI_Gather(&bytes, 1, MPI_INT, counts.data(), 1, MPI_INT, root, comm);
  std::vector<FullRecord> all;
  if (me == root) all.resize(prefix(counts, displs) / sizeof(FullRecord));
  MPI_Gatherv(mine.data(), bytes, MPI_BYTE, all.data(), counts.data(), me == root ? displs.data() : nullptr,
              MPI_BYTE, root, comm);

  ParticleSet out;
  if (me != root) return out;
  out.reserve(all.size());
  for (const FullRecord& r : all) {
    out.push_back(r.id, r.m, r.x, r.y, r.z, r.vx, r.vy, r.vz);
    out.ax.back() = r.ax;
    out.ay.back() = r.ay;
    out.az.back() = r.az;
    out.pot.back() = r.pot;
    out.work.back() = r.work;
  }
  return out;
}

}  // namespace nbody

#include "nbody/mpi/shm_trees.hpp"

#include <algorithm>
#include <cstring>

#include "nbody/core/timer.hpp"
#include "nbody/mpi/mpi_env.hpp"

namespace nbody {

namespace {

struct Header {
  std::int64_t n_nodes;
  std::int64_t n_parts;
};

std::uint64_t align8(std::uint64_t v) { return (v + 7) & ~std::uint64_t{7}; }

// Byte offsets of the arrays inside a segment.
struct Layout {
  std::uint64_t nodes, px, py, pz, pm, pidx, end;
  Layout(std::int64_t nn, std::int64_t np) {
    nodes = align8(sizeof(Header));
    px = align8(nodes + nn * sizeof(Node));
    py = px + np * sizeof(double);
    pz = py + np * sizeof(double);
    pm = pz + np * sizeof(double);
    pidx = pm + np * sizeof(double);
    end = align8(pidx + np * sizeof(std::uint32_t));
  }
};

}  // namespace

ShmTreeWindows::ShmTreeWindows(MPI_Comm comm, int group_size) {
  me_ = comm_rank(comm);
  const int P = comm_size(comm);
  MPI_Comm node;
  MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, me_, MPI_INFO_NULL, &node);
  if (group_size > 0) {
    MPI_Comm_split(node, comm_rank(node) / group_size, me_, &group_);
    MPI_Comm_free(&node);
  } else {
    group_ = node;
  }
  const int G = comm_size(group_);
  members_.resize(G);
  MPI_Allgather(&me_, 1, MPI_INT, members_.data(), 1, MPI_INT, group_);
  group_id_.resize(P);
  const int my_id = *std::min_element(members_.begin(), members_.end());
  MPI_Allgather(&my_id, 1, MPI_INT, group_id_.data(), 1, MPI_INT, comm);
  group_rank_.assign(P, -1);
  for (int g = 0; g < G; ++g) group_rank_[members_[g]] = g;
  bases_.assign(G, nullptr);
}

ShmTreeWindows::~ShmTreeWindows() {
  if (win_ != MPI_WIN_NULL) {
    MPI_Barrier(group_);  // no peer may still be reading our segment
    MPI_Win_unlock_all(win_);
    MPI_Win_free(&win_);
  }
  if (group_ != MPI_COMM_NULL) MPI_Comm_free(&group_);
}

void ShmTreeWindows::reallocate(std::uint64_t capacity) {
  if (win_ != MPI_WIN_NULL) {
    MPI_Win_unlock_all(win_);
    MPI_Win_free(&win_);
  }
  void* base = nullptr;
  MPI_Win_allocate_shared(static_cast<MPI_Aint>(capacity), 1, MPI_INFO_NULL, group_, &base, &win_);
  MPI_Win_lock_all(MPI_MODE_NOCHECK, win_);
  mine_ = static_cast<char*>(base);
  for (int g = 0; g < static_cast<int>(members_.size()); ++g) {
    MPI_Aint size = 0;
    int disp = 0;
    void* ptr = nullptr;
    MPI_Win_shared_query(win_, g, &size, &disp, &ptr);
    bases_[g] = static_cast<const char*>(ptr);
  }
  capacity_ = capacity;
  ++reallocations_;
}

void ShmTreeWindows::publish(const FlatTree& t) {
  const auto nn = static_cast<std::int64_t>(t.nodes.size());
  const auto np = static_cast<std::int64_t>(t.px.size());
  const Layout L(nn, np);

  double t0 = wall_time();
  // Also the "everyone has finished reading the previous trees" point (see header).
  std::uint64_t need = L.end, max_need = 0;
  MPI_Allreduce(&need, &max_need, 1, MPI_UINT64_T, MPI_MAX, group_);
  sync_time += wall_time() - t0;
  if (win_ == MPI_WIN_NULL || max_need > capacity_)
    reallocate(std::max<std::uint64_t>(4096, align8(max_need + max_need / 2)));

  Header h{nn, np};
  std::memcpy(mine_, &h, sizeof(h));
  if (nn) std::memcpy(mine_ + L.nodes, t.nodes.data(), nn * sizeof(Node));
  if (np) {
    std::memcpy(mine_ + L.px, t.px.data(), np * sizeof(double));
    std::memcpy(mine_ + L.py, t.py.data(), np * sizeof(double));
    std::memcpy(mine_ + L.pz, t.pz.data(), np * sizeof(double));
    std::memcpy(mine_ + L.pm, t.pm.data(), np * sizeof(double));
    std::memcpy(mine_ + L.pidx, t.pidx.data(), np * sizeof(std::uint32_t));
  }

  t0 = wall_time();
  MPI_Win_sync(win_);
  MPI_Barrier(group_);
  MPI_Win_sync(win_);
  sync_time += wall_time() - t0;
}

TreeView ShmTreeWindows::peer(int rank) const {
  const char* base = bases_[group_rank_[rank]];
  Header h;
  std::memcpy(&h, base, sizeof(h));
  const Layout L(h.n_nodes, h.n_parts);
  TreeView v;
  v.nodes = reinterpret_cast<const Node*>(base + L.nodes);
  v.n_nodes = static_cast<std::int32_t>(h.n_nodes);
  v.px = reinterpret_cast<const double*>(base + L.px);
  v.py = reinterpret_cast<const double*>(base + L.py);
  v.pz = reinterpret_cast<const double*>(base + L.pz);
  v.pm = reinterpret_cast<const double*>(base + L.pm);
  v.pidx = reinterpret_cast<const std::uint32_t*>(base + L.pidx);
  return v;
}

}  // namespace nbody

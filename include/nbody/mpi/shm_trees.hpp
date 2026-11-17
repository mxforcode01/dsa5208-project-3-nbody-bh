#pragma once

#include <mpi.h>

#include <cstdint>
#include <vector>

#include "nbody/tree/octree.hpp"
#include "nbody/tree/traversal.hpp"

namespace nbody {

// V6: rank-local trees shared through an MPI-3 shared-memory window.
//
// The ranks of `comm` that share a node (MPI_Comm_split_type MPI_COMM_TYPE_SHARED),
// optionally cut into groups of `group_size` consecutive node ranks to emulate several
// nodes on one machine, form a group. Every rank owns one segment of an
// MPI_Win_allocate_shared window and publishes its local tree there each step:
//   header {int64 n_nodes, int64 n_parts} | nodes | px | py | pz | pm | pidx
// Group peers then walk each other's full trees directly in shared memory; no LET is
// built or sent inside a group. Only offsets are stored (each process maps the window
// at its own address; MPI_Win_shared_query gives the local base of each segment).
//
// Synchronisation (unified memory model, one passive epoch MPI_Win_lock_all for the
// lifetime of the window):
//   publish(): MPI_Allreduce(MAX) of the space needed - every rank has entered publish,
//              so nobody is still reading the previous step's trees -> safe to overwrite;
//              reallocate collectively if any segment is too small (1.5x headroom);
//              write own segment; MPI_Win_sync; MPI_Barrier(group); MPI_Win_sync.
//   After publish() the peers' trees are read-only until the next publish().
// All calls are collective over the group (constructor/destructor: over `comm`/group)
// and are made outside OpenMP parallel regions.
class ShmTreeWindows {
 public:
  ShmTreeWindows(MPI_Comm comm, int group_size);  // group_size 0 = whole node
  ~ShmTreeWindows();
  ShmTreeWindows(const ShmTreeWindows&) = delete;
  ShmTreeWindows& operator=(const ShmTreeWindows&) = delete;

  void publish(const FlatTree& local);
  bool same_group(int rank) const { return group_id_[rank] == group_id_[me_]; }
  TreeView peer(int rank) const;  // rank in `comm`, same group; valid until the next publish()

  int group_size() const { return static_cast<int>(members_.size()); }
  std::uint64_t capacity_bytes() const { return capacity_; }  // per segment
  int reallocations() const { return reallocations_; }
  double sync_time = 0.0;  // Allreduce + Win_sync + Barrier inside publish()

 private:
  void reallocate(std::uint64_t capacity);

  MPI_Comm group_ = MPI_COMM_NULL;
  MPI_Win win_ = MPI_WIN_NULL;
  int me_ = 0;
  std::vector<int> group_id_;   // per rank of comm: id of its group (comm rank of its first member)
  std::vector<int> members_;    // group rank -> comm rank
  std::vector<int> group_rank_;  // comm rank -> group rank (-1 if not in my group)
  std::vector<const char*> bases_;  // group rank -> local address of that segment
  char* mine_ = nullptr;
  std::uint64_t capacity_ = 0;
  int reallocations_ = 0;
};

}  // namespace nbody

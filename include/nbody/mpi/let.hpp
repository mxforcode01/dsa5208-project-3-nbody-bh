#pragma once

#include <mpi.h>

#include <cstdint>
#include <vector>

#include "nbody/core/comm_mode.hpp"
#include "nbody/core/particles.hpp"
#include "nbody/tree/octree.hpp"

namespace nbody {

// A Locally Essential Tree fragment sent from one rank to another: a pruned copy of
// the sender's local tree in the same depth-first layout (with rewritten `next`
// indices), so the receiver walks it with the ordinary traversal.
//  - node accepted by mac_accept_box for the receiver's box -> sent as PRUNED monopole
//  - leaf not accepted                                      -> sent with its particles
//  - internal node not accepted                             -> header sent, children follow
struct LetBuffer {
  std::vector<Node> nodes;
  std::vector<double> parts;  // x, y, z, m per particle
};

// Tight axis-aligned bounds of p; returns false (and leaves box unchanged) if p is empty.
bool particle_bounds(const ParticleSet& p, AABB& box);

// Appends the LET of `local` for target region `target` to `out` (out must be empty).
void build_let(const FlatTree& local, const AABB& target, LetBuffer& out);

// Same, for a receiver whose particles lie in the union of `targets` (tight boxes). A cell is
// sent as one pruned monopole only if it is accepted for every box, so the receiver sees the
// same interactions as with the single bounding box, but the LET is much smaller when that box
// is mostly empty (e.g. a few halo particles stretch it across the dense core of a Plummer sphere).
void build_let(const FlatTree& local, const std::vector<AABB>& targets, LetBuffer& out);

// Tight boxes covering all particles of `local`: a cut through its octree that repeatedly
// expands the most populated cell, giving at most `max_boxes` boxes. Returns false if empty.
bool receiver_boxes(const FlatTree& local, int max_boxes, std::vector<AABB>& boxes);

// Rebuilds a walkable FlatTree from received LET data (pidx entries are 0, so walk it
// with self = kNoSelf).
FlatTree let_to_tree(const Node* nodes, std::size_t n_nodes, const double* parts, std::size_t n_parts,
                     const Cube& root, double theta);

// LET exchange in three stages so that work can be placed between them (all MPI calls are
// made by the calling thread, outside OpenMP parallel regions -> MPI_THREAD_FUNNELED):
//   start():    MPI_Allgather of particle bounds, LET construction for every other
//               non-empty rank (OpenMP over destinations), MPI_Alltoall of the sizes, then
//                 Blocking: two MPI_Alltoallv (nodes as bytes, particles as doubles) - done;
//                 P2P/Overlap: MPI_Irecv for every non-empty incoming message, then MPI_Isend.
//   progress(): MPI_Testsome on the outstanding requests (drives progress during compute).
//   finish():   MPI_Waitall, then rebuild one FlatTree per source rank (in rank order).
// The send buffers are the LET buffers themselves and stay alive until finish().
class LetExchanger {
 public:
  // skip (optional, size P): ranks that get no LET from us and send none to us (their trees
  // are reachable through shared memory). Must be symmetric across ranks.
  void start(MPI_Comm comm, const FlatTree& local, const ParticleSet& p, int threads, CommMode mode,
             const std::vector<char>* skip = nullptr);
  void progress();
  void finish();

  std::vector<FlatTree> remote;  // after finish(): one per rank that sent a non-empty LET, rank order
  std::vector<int> remote_source;  // source rank of remote[i]
  std::uint64_t bytes_sent = 0;
  std::uint64_t bytes_received = 0;
  std::uint64_t nodes_sent = 0;
  int tests = 0;               // MPI_Testsome calls made by progress()
  int max_boxes = 64;          // receiver region description: at most this many boxes per rank
  double build_time = 0.0;     // LET construction + unpacking
  double exchange_time = 0.0;  // collectives and posting (blocking: whole exchange)
  double wait_time = 0.0;      // in MPI_Waitall / MPI_Testsome (blocking: in MPI_Alltoallv)

 private:
  void unpack(int source, std::vector<Node>&& nodes, const std::vector<double>& parts);

  MPI_Comm comm_ = MPI_COMM_NULL;
  CommMode mode_ = CommMode::Blocking;
  Cube root_;
  double theta_ = 0.5;
  bool done_ = true;
  std::vector<LetBuffer> out_;
  std::vector<std::vector<Node>> in_nodes_;
  std::vector<std::vector<double>> in_parts_;
  std::vector<MPI_Request> requests_;
  std::vector<int> completed_;
};

}  // namespace nbody

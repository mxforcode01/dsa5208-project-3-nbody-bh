#pragma once

#include <stdexcept>
#include <string>

namespace nbody {

// LET exchange strategy of the distributed solver (V5).
//  Blocking: MPI_Alltoall of sizes + two MPI_Alltoallv, then compute (baseline).
//  P2P:      sizes via MPI_Alltoall, then MPI_Irecv/MPI_Isend per peer and an immediate
//            MPI_Waitall: nonblocking calls, no overlap.
//  Overlap:  as P2P, but local-tree forces are computed while the messages are in flight;
//            the master thread calls MPI_Testsome between chunks of local work.
enum class CommMode { Blocking, P2P, Overlap };

inline CommMode parse_comm_mode(const std::string& s) {
  if (s == "blocking") return CommMode::Blocking;
  if (s == "p2p") return CommMode::P2P;
  if (s == "overlap") return CommMode::Overlap;
  throw std::invalid_argument("unknown comm mode '" + s + "' (expected blocking|p2p|overlap)");
}

inline std::string to_string(CommMode m) {
  return m == CommMode::Blocking ? "blocking" : (m == CommMode::P2P ? "p2p" : "overlap");
}

}  // namespace nbody

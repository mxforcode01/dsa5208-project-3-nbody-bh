#include "nbody/mpi/let.hpp"

#include <algorithm>
#include <climits>
#include <stdexcept>

#include "nbody/core/timer.hpp"
#include "nbody/mpi/mpi_env.hpp"
#include "nbody/tree/mac.hpp"

namespace nbody {

namespace {

constexpr int kTagNodes = 101;
constexpr int kTagParts = 102;

bool accept_all(const Node& nd, const std::vector<AABB>& boxes) {
  for (const AABB& b : boxes)
    if (!mac_accept_box(nd, b)) return false;
  return true;
}

void emit(const FlatTree& t, std::int32_t i, const std::vector<AABB>& box, LetBuffer& out) {
  const Node& nd = t.nodes[i];
  const std::size_t idx = out.nodes.size();
  out.nodes.push_back(nd);
  Node copy = nd;
  copy.pbegin = static_cast<std::int32_t>(out.parts.size() / 4);
  if (accept_all(nd, box)) {
    copy.flags = node_flags::kPruned;  // every receiver particle accepts it; children not needed
    copy.pcount = 0;
  } else if (nd.is_leaf()) {
    for (std::int32_t k = nd.pbegin; k < nd.pbegin + nd.pcount; ++k) {
      out.parts.push_back(t.px[k]);
      out.parts.push_back(t.py[k]);
      out.parts.push_back(t.pz[k]);
      out.parts.push_back(t.pm[k]);
    }
  } else {
    for (std::int32_t c = i + 1; c < nd.next; c = t.nodes[c].next) emit(t, c, box, out);
    copy.pcount = static_cast<std::int32_t>(out.parts.size() / 4) - copy.pbegin;
  }
  copy.next = static_cast<std::int32_t>(out.nodes.size());
  out.nodes[idx] = copy;
}

int checked_int(std::uint64_t v) {
  if (v > static_cast<std::uint64_t>(INT_MAX)) throw std::runtime_error("LET exchange: message too large for MPI int counts");
  return static_cast<int>(v);
}

}  // namespace

bool particle_bounds(const ParticleSet& p, AABB& box) {
  if (p.size() == 0) return false;
  box.lo[0] = box.hi[0] = p.x[0];
  box.lo[1] = box.hi[1] = p.y[0];
  box.lo[2] = box.hi[2] = p.z[0];
  for (std::size_t i = 1; i < p.size(); ++i) {
    box.lo[0] = std::min(box.lo[0], p.x[i]);
    box.lo[1] = std::min(box.lo[1], p.y[i]);
    box.lo[2] = std::min(box.lo[2], p.z[i]);
    box.hi[0] = std::max(box.hi[0], p.x[i]);
    box.hi[1] = std::max(box.hi[1], p.y[i]);
    box.hi[2] = std::max(box.hi[2], p.z[i]);
  }
  return true;
}

void build_let(const FlatTree& local, const std::vector<AABB>& targets, LetBuffer& out) {
  if (!out.nodes.empty() || !out.parts.empty()) throw std::invalid_argument("build_let: output must be empty");
  if (!local.nodes.empty() && !targets.empty()) emit(local, 0, targets, out);
}

void build_let(const FlatTree& local, const AABB& target, LetBuffer& out) {
  build_let(local, std::vector<AABB>{target}, out);
}

bool receiver_boxes(const FlatTree& local, int max_boxes, std::vector<AABB>& boxes) {
  boxes.clear();
  if (local.nodes.empty() || local.px.empty()) return false;
  // Cut through the tree: expand the most populated internal cell until max_boxes cells exist.
  std::vector<std::int32_t> cut{0};
  while (static_cast<int>(cut.size()) < max_boxes) {
    int best = -1;
    std::int32_t best_count = 0;
    for (int k = 0; k < static_cast<int>(cut.size()); ++k) {
      const Node& nd = local.nodes[cut[k]];
      if (!nd.is_leaf() && nd.pcount > best_count) {
        best = k;
        best_count = nd.pcount;
      }
    }
    if (best < 0) break;
    const std::int32_t i = cut[best];
    std::vector<std::int32_t> kids;
    for (std::int32_t c = i + 1; c < local.nodes[i].next; c = local.nodes[c].next) kids.push_back(c);
    if (static_cast<int>(cut.size() - 1 + kids.size()) > max_boxes) break;
    cut.erase(cut.begin() + best);
    cut.insert(cut.end(), kids.begin(), kids.end());
  }
  for (std::int32_t i : cut) {
    const Node& nd = local.nodes[i];
    if (nd.pcount <= 0) continue;
    AABB b;
    b.lo[0] = b.hi[0] = local.px[nd.pbegin];
    b.lo[1] = b.hi[1] = local.py[nd.pbegin];
    b.lo[2] = b.hi[2] = local.pz[nd.pbegin];
    for (std::int32_t k = nd.pbegin + 1; k < nd.pbegin + nd.pcount; ++k) {
      b.lo[0] = std::min(b.lo[0], local.px[k]);
      b.lo[1] = std::min(b.lo[1], local.py[k]);
      b.lo[2] = std::min(b.lo[2], local.pz[k]);
      b.hi[0] = std::max(b.hi[0], local.px[k]);
      b.hi[1] = std::max(b.hi[1], local.py[k]);
      b.hi[2] = std::max(b.hi[2], local.pz[k]);
    }
    boxes.push_back(b);
  }
  return !boxes.empty();
}

FlatTree let_to_tree(const Node* nodes, std::size_t n_nodes, const double* parts, std::size_t n_parts,
                     const Cube& root, double theta) {
  FlatTree t;
  t.root = root;
  t.theta = theta;
  t.nodes.assign(nodes, nodes + n_nodes);
  t.px.resize(n_parts);
  t.py.resize(n_parts);
  t.pz.resize(n_parts);
  t.pm.resize(n_parts);
  t.pidx.assign(n_parts, 0u);
  for (std::size_t k = 0; k < n_parts; ++k) {
    t.px[k] = parts[4 * k];
    t.py[k] = parts[4 * k + 1];
    t.pz[k] = parts[4 * k + 2];
    t.pm[k] = parts[4 * k + 3];
  }
  for (const Node& nd : t.nodes) t.depth = std::max<int>(t.depth, nd.level);
  return t;
}

void LetExchanger::start(MPI_Comm comm, const FlatTree& local, const ParticleSet& /*p*/, int threads, CommMode mode,
                         const std::vector<char>* skip) {
  const int P = comm_size(comm), me = comm_rank(comm);
  comm_ = comm;
  mode_ = mode;
  root_ = local.root;
  theta_ = local.theta;
  remote.clear();
  remote_source.clear();
  requests_.clear();
  done_ = false;

  // 1. Receiver description of every rank: {number of boxes, then lo[3], hi[3] per box}, padded to
  //    max_boxes boxes (the pad is never read). One box reproduces the plain bounding box.
  double t0 = wall_time();
  const int K = std::max(1, max_boxes);
  const std::size_t stride = 1 + 6 * static_cast<std::size_t>(K);
  std::vector<AABB> mine;
  std::vector<double> box_msg(stride, 0.0);
  if (receiver_boxes(local, K, mine)) {
    box_msg[0] = static_cast<double>(mine.size());
    for (std::size_t b = 0; b < mine.size(); ++b) {
      std::copy(mine[b].lo, mine[b].lo + 3, &box_msg[1 + 6 * b]);
      std::copy(mine[b].hi, mine[b].hi + 3, &box_msg[1 + 6 * b + 3]);
    }
  }
  std::vector<double> boxes(stride * static_cast<std::size_t>(P));
  MPI_Allgather(box_msg.data(), static_cast<int>(stride), MPI_DOUBLE, boxes.data(), static_cast<int>(stride),
                MPI_DOUBLE, comm);
  exchange_time += wall_time() - t0;

  // 2. One LET per other non-empty rank (independent -> OpenMP over destinations).
  t0 = wall_time();
  out_.assign(P, LetBuffer{});
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if (threads > 1)
  for (int d = 0; d < P; ++d) {
    const double* bd = &boxes[stride * d];
    const int nb = static_cast<int>(bd[0]);
    if (d == me || nb == 0 || (skip && (*skip)[d])) continue;
    std::vector<AABB> target(nb);
    for (int b = 0; b < nb; ++b) {
      std::copy(bd + 1 + 6 * b, bd + 1 + 6 * b + 3, target[b].lo);
      std::copy(bd + 1 + 6 * b + 3, bd + 1 + 6 * b + 6, target[b].hi);
    }
    build_let(local, target, out_[d]);
  }
  build_time += wall_time() - t0;

  // 3. Sizes (node count, double count) to every rank.
  t0 = wall_time();
  std::vector<int> send_counts(2 * static_cast<std::size_t>(P), 0), recv_counts(2 * static_cast<std::size_t>(P), 0);
  for (int d = 0; d < P; ++d) {
    send_counts[2 * d] = checked_int(out_[d].nodes.size());
    send_counts[2 * d + 1] = checked_int(out_[d].parts.size());
    bytes_sent += out_[d].nodes.size() * sizeof(Node) + out_[d].parts.size() * sizeof(double);
    nodes_sent += out_[d].nodes.size();
  }
  MPI_Alltoall(send_counts.data(), 2, MPI_INT, recv_counts.data(), 2, MPI_INT, comm);
  in_nodes_.assign(P, {});
  in_parts_.assign(P, {});
  for (int s = 0; s < P; ++s) {
    in_nodes_[s].resize(static_cast<std::size_t>(recv_counts[2 * s]));
    in_parts_[s].resize(static_cast<std::size_t>(recv_counts[2 * s + 1]));
    bytes_received += in_nodes_[s].size() * sizeof(Node) + in_parts_[s].size() * sizeof(double);
  }

  if (mode == CommMode::Blocking) {
    // 4a. Packed MPI_Alltoallv (nodes as bytes, particles as doubles).
    std::vector<int> sn(P), sd(P), rn(P), rd(P), sp(P), spd(P), rp(P), rpd(P);
    std::uint64_t sn_tot = 0, rn_tot = 0, sp_tot = 0, rp_tot = 0;
    for (int d = 0; d < P; ++d) {
      sn[d] = checked_int(out_[d].nodes.size() * sizeof(Node));
      sp[d] = send_counts[2 * d + 1];
      rn[d] = checked_int(in_nodes_[d].size() * sizeof(Node));
      rp[d] = recv_counts[2 * d + 1];
      sd[d] = checked_int(sn_tot);
      spd[d] = checked_int(sp_tot);
      rd[d] = checked_int(rn_tot);
      rpd[d] = checked_int(rp_tot);
      sn_tot += sn[d];
      sp_tot += sp[d];
      rn_tot += rn[d];
      rp_tot += rp[d];
    }
    std::vector<char> send_nodes(sn_tot), recv_nodes(rn_tot);
    std::vector<double> send_parts(sp_tot), recv_parts(rp_tot);
    for (int d = 0; d < P; ++d) {
      std::copy(reinterpret_cast<const char*>(out_[d].nodes.data()),
                reinterpret_cast<const char*>(out_[d].nodes.data()) + sn[d], send_nodes.data() + sd[d]);
      std::copy(out_[d].parts.begin(), out_[d].parts.end(), send_parts.begin() + spd[d]);
    }
    exchange_time += wall_time() - t0;
    const double tw = wall_time();
    MPI_Alltoallv(send_nodes.data(), sn.data(), sd.data(), MPI_BYTE, recv_nodes.data(), rn.data(), rd.data(),
                  MPI_BYTE, comm);
    MPI_Alltoallv(send_parts.data(), sp.data(), spd.data(), MPI_DOUBLE, recv_parts.data(), rp.data(), rpd.data(),
                  MPI_DOUBLE, comm);
    wait_time += wall_time() - tw;
    for (int s = 0; s < P; ++s) {
      std::copy(recv_nodes.data() + rd[s], recv_nodes.data() + rd[s] + rn[s],
                reinterpret_cast<char*>(in_nodes_[s].data()));
      std::copy(recv_parts.begin() + rpd[s], recv_parts.begin() + rpd[s] + rp[s], in_parts_[s].begin());
    }
    done_ = true;
    out_.clear();
    return;
  }

  // 4b. Point-to-point: all receives first (no unexpected-message buffering), then sends.
  //     Empty messages are skipped on both sides (both know the sizes).
  for (int s = 0; s < P; ++s) {
    if (s == me) continue;
    if (!in_nodes_[s].empty()) {
      requests_.emplace_back();
      MPI_Irecv(in_nodes_[s].data(), checked_int(in_nodes_[s].size() * sizeof(Node)), MPI_BYTE, s, kTagNodes, comm,
                &requests_.back());
    }
    if (!in_parts_[s].empty()) {
      requests_.emplace_back();
      MPI_Irecv(in_parts_[s].data(), checked_int(in_parts_[s].size()), MPI_DOUBLE, s, kTagParts, comm,
                &requests_.back());
    }
  }
  for (int d = 0; d < P; ++d) {
    if (d == me) continue;
    if (!out_[d].nodes.empty()) {
      requests_.emplace_back();
      MPI_Isend(out_[d].nodes.data(), checked_int(out_[d].nodes.size() * sizeof(Node)), MPI_BYTE, d, kTagNodes, comm,
                &requests_.back());
    }
    if (!out_[d].parts.empty()) {
      requests_.emplace_back();
      MPI_Isend(out_[d].parts.data(), checked_int(out_[d].parts.size()), MPI_DOUBLE, d, kTagParts, comm,
                &requests_.back());
    }
  }
  completed_.resize(requests_.size());
  exchange_time += wall_time() - t0;
}

void LetExchanger::progress() {
  if (done_ || requests_.empty()) return;
  const double t0 = wall_time();
  int outcount = 0;
  MPI_Testsome(static_cast<int>(requests_.size()), requests_.data(), &outcount, completed_.data(),
               MPI_STATUSES_IGNORE);
  ++tests;
  wait_time += wall_time() - t0;
}

void LetExchanger::finish() {
  if (!done_) {
    const double t0 = wall_time();
    if (!requests_.empty())
      MPI_Waitall(static_cast<int>(requests_.size()), requests_.data(), MPI_STATUSES_IGNORE);
    wait_time += wall_time() - t0;
    done_ = true;
    out_.clear();  // send buffers may be released only now
  }
  const double t0 = wall_time();
  for (int s = 0; s < static_cast<int>(in_nodes_.size()); ++s)
    if (!in_nodes_[s].empty()) unpack(s, std::move(in_nodes_[s]), in_parts_[s]);
  in_nodes_.clear();
  in_parts_.clear();
  build_time += wall_time() - t0;
}

void LetExchanger::unpack(int source, std::vector<Node>&& nodes, const std::vector<double>& parts) {
  FlatTree t = let_to_tree(nullptr, 0, parts.data(), parts.size() / 4, root_, theta_);
  t.nodes = std::move(nodes);
  for (const Node& nd : t.nodes) t.depth = std::max<int>(t.depth, nd.level);
  remote.push_back(std::move(t));
  remote_source.push_back(source);
}

}  // namespace nbody

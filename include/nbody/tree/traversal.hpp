#pragma once

#include <cstdint>

#include "nbody/core/physics.hpp"
#include "nbody/tree/mac.hpp"
#include "nbody/tree/octree.hpp"

namespace nbody {

constexpr std::uint32_t kNoSelf = 0xFFFFFFFFu;

// Accumulated over one target (without the factor G).
struct WalkResult {
  double ax = 0.0, ay = 0.0, az = 0.0, phi = 0.0;
  std::uint64_t pp = 0;              // particle-particle interactions
  std::uint64_t pc = 0;              // particle-cell interactions
  std::uint64_t pruned_violations = 0;  // tried to open a PRUNED node (must stay 0)
  double mass_seen = 0.0;            // total source mass that acted on the target
};

// Non-owning view of a flat tree: a FlatTree, or a tree published by another rank in an
// MPI shared-memory window. Same layout either way.
struct TreeView {
  const Node* nodes = nullptr;
  std::int32_t n_nodes = 0;
  const double *px = nullptr, *py = nullptr, *pz = nullptr, *pm = nullptr;
  const std::uint32_t* pidx = nullptr;
};

inline TreeView view_of(const FlatTree& t) {
  return {t.nodes.data(), static_cast<std::int32_t>(t.nodes.size()), t.px.data(), t.py.data(),
          t.pz.data(),    t.pm.data(),                                 t.pidx.data()};
}

// Stackless walk of a flat tree for one target position. Leaf particles whose
// pidx equals `self` are skipped (pass kNoSelf for a tree that cannot contain the target).
// No recursion and no stack, so the same function runs inside OpenMP target regions.
NBODY_DEVICE_BEGIN
inline void walk_tree(const TreeView& t, double x, double y, double z, std::uint32_t self, double eps2,
                      WalkResult& r) {
  const Node* nodes = t.nodes;
  const std::int32_t n = t.n_nodes;
  const double* px = t.px;
  const double* py = t.py;
  const double* pz = t.pz;
  const double* pm = t.pm;
  const std::uint32_t* pidx = t.pidx;
  std::int32_t i = 0;
  while (i < n) {
    const Node& nd = nodes[i];
    if (mac_accept_point(nd, x, y, z)) {
      accumulate_pair(nd.mx - x, nd.my - y, nd.mz - z, nd.mass, eps2, r.ax, r.ay, r.az, r.phi);
      ++r.pc;
      r.mass_seen += nd.mass;
      i = nd.next;
    } else if (nd.is_leaf()) {
      const std::int32_t end = nd.pbegin + nd.pcount;
      for (std::int32_t k = nd.pbegin; k < end; ++k) {
        if (pidx[k] == self) continue;
        accumulate_pair(px[k] - x, py[k] - y, pz[k] - z, pm[k], eps2, r.ax, r.ay, r.az, r.phi);
        ++r.pp;
        r.mass_seen += pm[k];
      }
      i = nd.next;
    } else if (nd.is_pruned()) {
      // Remote cell whose children were not sent: use it anyway and record the violation.
      accumulate_pair(nd.mx - x, nd.my - y, nd.mz - z, nd.mass, eps2, r.ax, r.ay, r.az, r.phi);
      ++r.pc;
      ++r.pruned_violations;
      r.mass_seen += nd.mass;
      i = nd.next;
    } else {
      ++i;  // open: first child follows in depth-first order
    }
  }
}
NBODY_DEVICE_END

inline void walk_tree(const FlatTree& t, double x, double y, double z, std::uint32_t self, double eps2,
                      WalkResult& r) {
  walk_tree(view_of(t), x, y, z, self, eps2, r);
}

}  // namespace nbody

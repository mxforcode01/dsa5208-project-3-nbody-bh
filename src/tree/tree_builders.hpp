#pragma once
// Internal: alternative tree builders behind nbody::build_tree, and the node
// finalisation shared by all builders (identical arithmetic => identical trees).

#include <cmath>

#include "nbody/tree/octree.hpp"

namespace nbody::detail {

// Sets centre of mass, delta, ropen2 from accumulated mass and mass-weighted position sums.
inline void finalize_moments(Node& nd, double mass, double sx, double sy, double sz, double theta) {
  nd.mass = mass;
  if (mass > 0.0) {
    nd.mx = sx / mass;
    nd.my = sy / mass;
    nd.mz = sz / mass;
  } else {  // massless cell: use the geometric centre
    nd.mx = nd.cx;
    nd.my = nd.cy;
    nd.mz = nd.cz;
  }
  nd.delta = std::sqrt((nd.mx - nd.cx) * (nd.mx - nd.cx) + (nd.my - nd.cy) * (nd.my - nd.cy) +
                       (nd.mz - nd.cz) * (nd.mz - nd.cz));
  nd.ropen2 = opening_radius2(nd.half, nd.delta, theta);
}

FlatTree build_tree_morton(const ParticleSet& p, const Cube& root, const TreeParams& tp, int threads,
                           BuildTimings& timings);

}  // namespace nbody::detail

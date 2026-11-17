// Octree structure: depth-first layout, skip pointers, particle ranges, moments,
// leaf capacity / max depth, coincident particles; and the opening criterion,
// including the conservative box version needed for the LET.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "nbody/core/initial_conditions.hpp"
#include "nbody/core/rng.hpp"
#include "nbody/tree/mac.hpp"
#include "nbody/tree/morton.hpp"
#include "nbody/tree/octree.hpp"
#include "test_harness.hpp"

using namespace nbody;

namespace {

ParticleSet make(Distribution d, std::uint64_t n, std::uint64_t seed = 42) {
  ICParams ic;
  ic.n_total = n;
  ic.seed = seed;
  ic.dist = d;
  ParticleSet p;
  generate_particles(ic, 0, n, p);
  return p;
}

TreeParams params(double theta, int cap, int depth) {
  TreeParams tp;
  tp.theta = theta;
  tp.leaf_capacity = cap;
  tp.max_depth = depth;
  return tp;
}

// Checks every structural invariant of a flat tree built from p. Mass sums use a
// relative tolerance of 1e-13: each node sums at most 8 children or `cap` particles.
void check_invariants(const FlatTree& t, const ParticleSet& p, const TreeParams& tp) {
  const auto n = static_cast<std::int32_t>(p.size());
  const auto nn = static_cast<std::int32_t>(t.nodes.size());
  NB_CHECK(nn >= 1);
  NB_CHECK(static_cast<std::int32_t>(t.pidx.size()) == n);
  NB_CHECK(t.nodes[0].pbegin == 0 && t.nodes[0].pcount == n && t.nodes[0].level == 0);
  NB_CHECK(t.nodes[0].next == nn);

  std::vector<int> seen(n, 0);
  for (std::uint32_t i : t.pidx) {
    NB_CHECK(i < static_cast<std::uint32_t>(n));
    if (i < static_cast<std::uint32_t>(n)) ++seen[i];
  }
  bool permutation = true;
  for (int s : seen) permutation = permutation && s == 1;
  NB_CHECK(permutation);

  int max_level = 0;
  bool ok_next = true, ok_children = true, ok_leaf = true, ok_mass = true, ok_com = true, ok_mac = true;
  for (std::int32_t i = 0; i < nn; ++i) {
    const Node& nd = t.nodes[i];
    max_level = std::max<int>(max_level, nd.level);
    ok_next = ok_next && nd.next > i && nd.next <= nn;
    double mass = 0.0;
    if (nd.is_leaf()) {
      ok_leaf = ok_leaf && nd.next == i + 1 && nd.pcount >= 1;
      ok_leaf = ok_leaf && (nd.pcount <= tp.leaf_capacity || nd.level == tp.max_depth);
      for (std::int32_t k = nd.pbegin; k < nd.pbegin + nd.pcount; ++k) {
        const std::uint32_t j = t.pidx[k];
        ok_leaf = ok_leaf && t.px[k] == p.x[j] && t.py[k] == p.y[j] && t.pz[k] == p.z[j] && t.pm[k] == p.m[j];
        ok_leaf = ok_leaf && std::fabs(t.px[k] - nd.cx) <= nd.half && std::fabs(t.py[k] - nd.cy) <= nd.half &&
                  std::fabs(t.pz[k] - nd.cz) <= nd.half;
        mass += t.pm[k];
      }
    } else {
      ok_children = ok_children && nd.next > i + 1;
      std::int32_t c = i + 1, expect_begin = nd.pbegin, count = 0;
      while (c < nd.next) {
        const Node& ch = t.nodes[c];
        ok_children = ok_children && ch.level == nd.level + 1 && ch.half == 0.5 * nd.half;
        // Child centre = parent centre +- half/2, rounded relative to |centre|, hence the tolerance.
        auto centre_ok = [&](double c, double pc) {
          return std::fabs(std::fabs(c - pc) - 0.5 * nd.half) <= 4e-16 * (std::fabs(pc) + nd.half);
        };
        ok_children = ok_children && centre_ok(ch.cx, nd.cx) && centre_ok(ch.cy, nd.cy) && centre_ok(ch.cz, nd.cz);
        ok_children = ok_children && ch.pbegin == expect_begin;
        expect_begin += ch.pcount;
        mass += ch.mass;
        ++count;
        c = ch.next;
      }
      ok_children = ok_children && c == nd.next && count >= 1 && count <= 8;
      ok_children = ok_children && expect_begin == nd.pbegin + nd.pcount;
    }
    ok_mass = ok_mass && std::fabs(mass - nd.mass) <= 1e-13 * nd.mass;
    const double tol = 1e-12 * nd.half;
    ok_com = ok_com && std::fabs(nd.mx - nd.cx) <= nd.half + tol && std::fabs(nd.my - nd.cy) <= nd.half + tol &&
             std::fabs(nd.mz - nd.cz) <= nd.half + tol;
    const double delta = std::sqrt((nd.mx - nd.cx) * (nd.mx - nd.cx) + (nd.my - nd.cy) * (nd.my - nd.cy) +
                                   (nd.mz - nd.cz) * (nd.mz - nd.cz));
    ok_mac = ok_mac && nd.delta == delta && nd.ropen2 == opening_radius2(nd.half, nd.delta, tp.theta);
  }
  NB_CHECK(ok_next);
  NB_CHECK(ok_children);
  NB_CHECK(ok_leaf);
  NB_CHECK(ok_mass);
  NB_CHECK(ok_com);
  NB_CHECK(ok_mac);
  NB_CHECK(t.depth == max_level);
  // Root mass (hierarchical sum) vs a left-to-right sum: each rounds by at most N u.
  double total = 0.0;
  for (double m : p.m) total += m;
  NB_CHECK_NEAR(t.nodes[0].mass, total, 2.0 * n * (std::numeric_limits<double>::epsilon() / 2.0) * total);
}

}  // namespace

NB_TEST(bounding_cube_contains_all_particles) {
  for (Distribution d : {Distribution::Uniform, Distribution::Plummer}) {
    const ParticleSet p = make(d, 10000);
    const Cube c = bounding_cube(p);
    bool ok = true;
    for (std::size_t i = 0; i < p.size(); ++i)
      ok = ok && std::fabs(p.x[i] - c.cx) <= c.half && std::fabs(p.y[i] - c.cy) <= c.half &&
           std::fabs(p.z[i] - c.cz) <= c.half;
    NB_CHECK(ok);
  }
  ParticleSet same;
  same.push_back(0, 1, 0.5, 0.5, 0.5, 0, 0, 0);
  same.push_back(1, 1, 0.5, 0.5, 0.5, 0, 0, 0);
  NB_CHECK(bounding_cube(same).half > 0.0);
}

NB_TEST(invariants_hold_for_uniform_and_plummer) {
  for (Distribution d : {Distribution::Uniform, Distribution::Plummer})
    for (int cap : {1, 8, 32}) {
      const ParticleSet p = make(d, 5000);
      const TreeParams tp = params(0.5, cap, 21);
      const FlatTree t = build_tree(p, bounding_cube(p), tp);
      std::printf("    %s cap=%d: %zu nodes, depth %d\n", to_string(d).c_str(), cap, t.nodes.size(), t.depth);
      check_invariants(t, p, tp);
    }
}

NB_TEST(small_trees) {
  for (std::uint64_t n : {1u, 2u, 8u, 9u}) {
    const ParticleSet p = make(Distribution::Uniform, n);
    const TreeParams tp = params(0.5, 8, 21);
    const FlatTree t = build_tree(p, bounding_cube(p), tp);
    check_invariants(t, p, tp);
    NB_CHECK(t.nodes[0].is_leaf() == (n <= 8));
  }
  ParticleSet empty;
  NB_CHECK(build_tree(empty, Cube{}, params(0.5, 8, 21)).nodes.empty());
}

NB_TEST(coincident_particles_stop_at_max_depth) {
  // 50 particles at exactly the same point cannot be separated; the builder must stop at
  // max_depth and keep them together in one (over-full) leaf.
  ParticleSet p = make(Distribution::Uniform, 200);
  for (std::size_t i = 0; i < 50; ++i) {
    p.x[i] = 0.1;
    p.y[i] = 0.1;
    p.z[i] = 0.1;
  }
  const TreeParams tp = params(0.5, 8, 21);
  const FlatTree t = build_tree(p, bounding_cube(p), tp);
  check_invariants(t, p, tp);
  NB_CHECK(t.depth == 21);
  bool found = false;
  for (const Node& nd : t.nodes) found = found || (nd.is_leaf() && nd.level == 21 && nd.pcount >= 50);
  NB_CHECK(found);
}

NB_TEST(shallow_max_depth_keeps_full_leaves) {
  const ParticleSet p = make(Distribution::Plummer, 2000);
  const TreeParams tp = params(0.5, 8, 2);
  const FlatTree t = build_tree(p, bounding_cube(p), tp);
  check_invariants(t, p, tp);
  NB_CHECK(t.depth <= 2);
  const FlatTree root_only = build_tree(p, bounding_cube(p), params(0.5, 8, 0));
  NB_CHECK(root_only.nodes.size() == 1u && root_only.nodes[0].is_leaf());
}

NB_TEST(build_is_deterministic) {
  const ParticleSet p = make(Distribution::Plummer, 3000);
  const TreeParams tp = params(0.5, 8, 21);
  const FlatTree a = build_tree(p, bounding_cube(p), tp), b = build_tree(p, bounding_cube(p), tp);
  bool same = a.nodes.size() == b.nodes.size() && a.pidx == b.pidx;
  for (std::size_t i = 0; same && i < a.nodes.size(); ++i)
    same = a.nodes[i].mass == b.nodes[i].mass && a.nodes[i].mx == b.nodes[i].mx && a.nodes[i].next == b.nodes[i].next;
  NB_CHECK(same);
}

NB_TEST(mac_never_accepts_a_cell_containing_the_target) {
  // Even with a huge opening angle, a cell must not act on a particle inside it.
  const ParticleSet p = make(Distribution::Uniform, 2000);
  for (double theta : {0.5, 1.0, 10.0}) {
    const FlatTree t = build_tree(p, bounding_cube(p), params(theta, 8, 21));
    bool ok = true;
    for (const Node& nd : t.nodes)
      for (std::int32_t k = nd.pbegin; k < nd.pbegin + nd.pcount; ++k)
        ok = ok && !mac_accept_point(nd, t.px[k], t.py[k], t.pz[k]);
    NB_CHECK(ok);
  }
  const FlatTree exact = build_tree(p, bounding_cube(p), params(0.0, 8, 21));
  bool none = true;
  for (const Node& nd : exact.nodes) none = none && !mac_accept_point(nd, 100.0, 100.0, 100.0);
  NB_CHECK(none);  // theta = 0 never accepts, however far away
}

NB_TEST(box_mac_is_conservative) {
  // If mac_accept_box(node, box) holds, mac_accept_point must hold for every point of the
  // box. Tested on all nodes of a real tree against random boxes, sampling the 8 corners
  // and 100 random interior points per box.
  const ParticleSet p = make(Distribution::Plummer, 3000);
  const FlatTree t = build_tree(p, bounding_cube(p), params(0.6, 8, 21));
  ParticleRng rng(7, 0);
  std::uint64_t accepted = 0, rejected = 0;
  bool ok = true;
  for (int b = 0; b < 200; ++b) {
    AABB box;
    for (int d = 0; d < 3; ++d) {
      const double c = (rng.uniform() - 0.5) * 8.0, w = rng.uniform() * 1.5;
      box.lo[d] = c - w;
      box.hi[d] = c + w;
    }
    for (const Node& nd : t.nodes) {
      if (!mac_accept_box(nd, box)) {
        ++rejected;
        continue;
      }
      ++accepted;
      for (int corner = 0; corner < 8; ++corner)
        ok = ok && mac_accept_point(nd, (corner & 4) ? box.hi[0] : box.lo[0], (corner & 2) ? box.hi[1] : box.lo[1],
                                    (corner & 1) ? box.hi[2] : box.lo[2]);
      for (int s = 0; s < 100; ++s)
        ok = ok && mac_accept_point(nd, box.lo[0] + rng.uniform() * (box.hi[0] - box.lo[0]),
                                    box.lo[1] + rng.uniform() * (box.hi[1] - box.lo[1]),
                                    box.lo[2] + rng.uniform() * (box.hi[2] - box.lo[2]));
    }
  }
  std::printf("    box MAC accepted %llu, rejected %llu node/box pairs\n", static_cast<unsigned long long>(accepted),
              static_cast<unsigned long long>(rejected));
  NB_CHECK(accepted > 1000u && rejected > 1000u);  // both branches exercised
  NB_CHECK(ok);
}

NB_TEST(box_mac_rejects_overlapping_cells) {
  const ParticleSet p = make(Distribution::Uniform, 500);
  const FlatTree t = build_tree(p, bounding_cube(p), params(10.0, 8, 21));
  bool ok = true;
  for (const Node& nd : t.nodes) {
    AABB box;  // a tiny box at the cell corner overlaps the cell
    box.lo[0] = box.hi[0] = nd.cx + nd.half;
    box.lo[1] = box.hi[1] = nd.cy + nd.half;
    box.lo[2] = box.hi[2] = nd.cz + nd.half;
    ok = ok && !mac_accept_box(nd, box);
  }
  NB_CHECK(ok);
}

namespace {

// Field-by-field exact comparison (padding bytes excluded).
bool same_tree(const FlatTree& a, const FlatTree& b) {
  if (a.nodes.size() != b.nodes.size() || a.pidx != b.pidx || a.px != b.px || a.py != b.py || a.pz != b.pz ||
      a.pm != b.pm || a.depth != b.depth)
    return false;
  for (std::size_t i = 0; i < a.nodes.size(); ++i) {
    const Node &x = a.nodes[i], &y = b.nodes[i];
    if (x.mx != y.mx || x.my != y.my || x.mz != y.mz || x.mass != y.mass || x.ropen2 != y.ropen2 || x.cx != y.cx ||
        x.cy != y.cy || x.cz != y.cz || x.half != y.half || x.delta != y.delta || x.next != y.next ||
        x.pbegin != y.pbegin || x.pcount != y.pcount || x.flags != y.flags || x.level != y.level)
      return false;
  }
  return true;
}

TreeParams with_build(TreeParams tp, TreeBuild b) {
  tp.build = b;
  return tp;
}

}  // namespace

NB_TEST(morton_build_is_identical_to_insertion_build) {
  // Same topology, same leaf particle order, same moments bit for bit, for several leaf
  // capacities / depths, serially and with OpenMP tasks + parallel sort (N >= 8192).
  for (Distribution d : {Distribution::Uniform, Distribution::Plummer})
    for (int cap : {1, 8, 32})
      for (int depth : {21, 4})
        for (int threads : {1, 4}) {
          const ParticleSet p = make(d, 20000);
          const TreeParams tp = params(0.5, cap, depth);
          const Cube c = bounding_cube(p);
          const FlatTree a = build_tree(p, c, with_build(tp, TreeBuild::Insertion));
          const FlatTree b = build_tree(p, c, with_build(tp, TreeBuild::MortonSorted), threads);
          const bool same = same_tree(a, b);
          if (!same)
            std::printf("    mismatch: %s cap=%d depth=%d threads=%d\n", to_string(d).c_str(), cap, depth, threads);
          NB_CHECK(same);
          check_invariants(b, p, tp);
        }
}

NB_TEST(morton_build_large_parallel_and_edge_cases) {
  // Deep task nesting (N = 200k, 8 threads), coincident particles, tiny and empty sets.
  {
    const ParticleSet p = make(Distribution::Plummer, 200000);
    const TreeParams tp = params(0.5, 8, 21);
    const Cube c = bounding_cube(p);
    BuildTimings bt;
    const FlatTree b = build_tree(p, c, with_build(tp, TreeBuild::MortonSorted), 8, &bt);
    NB_CHECK(same_tree(build_tree(p, c, tp), b));
    NB_CHECK(bt.total >= bt.keys + bt.sort && bt.keys > 0.0 && bt.sort > 0.0);
  }
  ParticleSet q = make(Distribution::Uniform, 200);
  for (std::size_t i = 0; i < 50; ++i) q.x[i] = q.y[i] = q.z[i] = 0.1;
  for (int threads : {1, 4}) {
    const TreeParams tp = params(0.5, 8, 21);
    NB_CHECK(same_tree(build_tree(q, bounding_cube(q), tp),
                       build_tree(q, bounding_cube(q), with_build(tp, TreeBuild::MortonSorted), threads)));
  }
  for (std::uint64_t n : {1u, 2u, 9u}) {
    const ParticleSet p = make(Distribution::Uniform, n);
    const TreeParams tp = params(0.5, 8, 21);
    NB_CHECK(same_tree(build_tree(p, bounding_cube(p), tp),
                       build_tree(p, bounding_cube(p), with_build(tp, TreeBuild::MortonSorted), 4)));
  }
  ParticleSet empty;
  NB_CHECK(build_tree(empty, Cube{}, with_build(params(0.5, 8, 21), TreeBuild::MortonSorted), 4).nodes.empty());
  NB_CHECK_THROWS(build_tree(q, bounding_cube(q), with_build(params(0.5, 8, 22), TreeBuild::MortonSorted)));
}

NB_TEST(morton_bit_interleaving) {
  NB_CHECK(morton_expand_bits(0) == 0u);
  NB_CHECK(morton_expand_bits(1) == 1u);
  NB_CHECK(morton_expand_bits(2) == 8u);
  NB_CHECK(morton_expand_bits(3) == 9u);
  NB_CHECK(morton_expand_bits(0x1FFFFF) == 0x1249249249249249ULL);
  Cube c;  // [-1, 1]^3
  NB_CHECK(morton_key(-1.0, -1.0, -1.0, c) == 0u);
  NB_CHECK(morton_key(0.999999999, 0.999999999, 0.999999999, c) == (1ULL << 63) - 1);
  NB_CHECK(morton_key(5.0, 5.0, 5.0, c) == (1ULL << 63) - 1);  // clamped
}

NB_TEST(morton_top_bits_match_octree_octants) {
  // The leading 3 bits of a key are the root octant (bit 2 = x), and keys of the particles
  // in each root child form one contiguous run of that octant's keys.
  const ParticleSet p = make(Distribution::Plummer, 5000);
  const Cube c = bounding_cube(p);
  bool ok = true;
  for (std::size_t i = 0; i < p.size(); ++i) {
    const int oct = (p.x[i] >= c.cx ? 4 : 0) | (p.y[i] >= c.cy ? 2 : 0) | (p.z[i] >= c.cz ? 1 : 0);
    ok = ok && static_cast<int>(morton_key(p.x[i], p.y[i], p.z[i], c) >> 60) == oct;
  }
  NB_CHECK(ok);
}

NB_TEST_MAIN()

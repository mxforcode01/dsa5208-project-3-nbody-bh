#include "nbody/tree/octree.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "nbody/core/timer.hpp"
#include "tree_builders.hpp"

namespace nbody {

TreeBuild parse_tree_build(const std::string& name) {
  if (name == "insertion") return TreeBuild::Insertion;
  if (name == "morton") return TreeBuild::MortonSorted;
  throw std::invalid_argument("unknown tree build '" + name + "' (expected insertion|morton)");
}

std::string to_string(TreeBuild b) { return b == TreeBuild::Insertion ? "insertion" : "morton"; }

Cube bounding_cube(const ParticleSet& p) {
  if (p.size() == 0) return Cube{};
  double lo[3] = {p.x[0], p.y[0], p.z[0]}, hi[3] = {p.x[0], p.y[0], p.z[0]};
  for (std::size_t i = 1; i < p.size(); ++i) {
    lo[0] = std::min(lo[0], p.x[i]);
    lo[1] = std::min(lo[1], p.y[i]);
    lo[2] = std::min(lo[2], p.z[i]);
    hi[0] = std::max(hi[0], p.x[i]);
    hi[1] = std::max(hi[1], p.y[i]);
    hi[2] = std::max(hi[2], p.z[i]);
  }
  return cube_from_bounds(lo, hi);
}

Cube cube_from_bounds(const double lo[3], const double hi[3]) {
  Cube c;
  if (lo[0] > hi[0] || lo[1] > hi[1] || lo[2] > hi[2]) return c;
  c.cx = 0.5 * (lo[0] + hi[0]);
  c.cy = 0.5 * (lo[1] + hi[1]);
  c.cz = 0.5 * (lo[2] + hi[2]);
  const double half = 0.5 * std::max({hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]});
  // Relative padding so rounding in the centre never leaves a particle outside.
  c.half = half > 0.0 ? half * (1.0 + 1e-9) : 1.0;
  return c;
}

double opening_radius2(double half, double delta, double theta) {
  if (theta <= 0.0) return std::numeric_limits<double>::infinity();
  const double r = 2.0 * half / theta + delta;
  return r * r;
}

namespace {

// Baseline builder: inserts particles one at a time into a pointer-style octree,
// splitting leaves that exceed the capacity, then flattens it to depth-first order.
class InsertionBuilder {
 public:
  InsertionBuilder(const ParticleSet& p, const Cube& root, const TreeParams& tp)
      : p_(p), tp_(tp), link_(p.size(), -1) {
    nodes_.reserve(p.size() / 2 + 8);
    make_node(root.cx, root.cy, root.cz, root.half, 0);
  }

  FlatTree build(const Cube& root) {
    const auto n = static_cast<std::int32_t>(p_.size());
    for (std::int32_t i = 0; i < n; ++i) insert(0, i);
    FlatTree t;
    t.root = root;
    t.theta = tp_.theta;
    if (n == 0) return t;
    t.nodes.reserve(nodes_.size());
    for (auto* v : {&t.px, &t.py, &t.pz, &t.pm}) v->reserve(p_.size());
    t.pidx.reserve(p_.size());
    emit(0, t);
    return t;
  }

 private:
  struct BuildNode {
    double cx, cy, cz, half;
    std::int32_t child[8];
    std::int32_t head, tail, count;  // particle list (leaves), in insertion order
    std::uint8_t level;
    bool leaf;
  };

  std::int32_t make_node(double cx, double cy, double cz, double half, int level) {
    BuildNode b;
    b.cx = cx;
    b.cy = cy;
    b.cz = cz;
    b.half = half;
    std::fill(b.child, b.child + 8, -1);
    b.head = b.tail = -1;
    b.count = 0;
    b.level = static_cast<std::uint8_t>(level);
    b.leaf = true;
    nodes_.push_back(b);
    return static_cast<std::int32_t>(nodes_.size() - 1);
  }

  // Octant numbering x-major (bit 2 = x, bit 1 = y, bit 0 = z), matching Morton order.
  int octant(std::int32_t node, std::int32_t i) const {
    const BuildNode& b = nodes_[node];
    return (p_.x[i] >= b.cx ? 4 : 0) | (p_.y[i] >= b.cy ? 2 : 0) | (p_.z[i] >= b.cz ? 1 : 0);
  }

  std::int32_t child_for(std::int32_t node, int oct) {
    if (nodes_[node].child[oct] < 0) {
      const BuildNode b = nodes_[node];  // copy: make_node may reallocate nodes_
      const double h = 0.5 * b.half;
      const std::int32_t c = make_node(b.cx + ((oct & 4) ? h : -h), b.cy + ((oct & 2) ? h : -h),
                                       b.cz + ((oct & 1) ? h : -h), h, b.level + 1);
      nodes_[node].child[oct] = c;
    }
    return nodes_[node].child[oct];
  }

  void append(std::int32_t node, std::int32_t i) {
    BuildNode& b = nodes_[node];
    link_[i] = -1;
    if (b.tail < 0) b.head = i;
    else link_[b.tail] = i;
    b.tail = i;
    ++b.count;
  }

  void insert(std::int32_t node, std::int32_t i) {
    while (!nodes_[node].leaf) node = child_for(node, octant(node, i));
    append(node, i);
    if (nodes_[node].count > tp_.leaf_capacity && nodes_[node].level < tp_.max_depth) split(node);
  }

  void split(std::int32_t node) {
    std::int32_t j = nodes_[node].head;
    nodes_[node].leaf = false;
    nodes_[node].head = nodes_[node].tail = -1;
    nodes_[node].count = 0;
    while (j >= 0) {
      const std::int32_t following = link_[j];
      insert(node, j);  // descends into the (new) children; may split them recursively
      j = following;
    }
  }

  void emit(std::int32_t b, FlatTree& t) {
    const auto idx = static_cast<std::int32_t>(t.nodes.size());
    t.nodes.emplace_back();
    const BuildNode& bn = nodes_[b];
    Node nd{};
    nd.cx = bn.cx;
    nd.cy = bn.cy;
    nd.cz = bn.cz;
    nd.half = bn.half;
    nd.level = bn.level;
    nd.pbegin = static_cast<std::int32_t>(t.px.size());
    double mass = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
    if (bn.leaf) {
      nd.flags = node_flags::kLeaf;
      for (std::int32_t j = bn.head; j >= 0; j = link_[j]) {
        t.px.push_back(p_.x[j]);
        t.py.push_back(p_.y[j]);
        t.pz.push_back(p_.z[j]);
        t.pm.push_back(p_.m[j]);
        t.pidx.push_back(static_cast<std::uint32_t>(j));
        mass += p_.m[j];
        sx += p_.m[j] * p_.x[j];
        sy += p_.m[j] * p_.y[j];
        sz += p_.m[j] * p_.z[j];
      }
    } else {
      nd.flags = 0;
      for (int oct = 0; oct < 8; ++oct) {
        if (bn.child[oct] < 0) continue;
        const auto ci = static_cast<std::int32_t>(t.nodes.size());
        emit(bn.child[oct], t);
        const Node& cn = t.nodes[ci];
        mass += cn.mass;
        sx += cn.mass * cn.mx;
        sy += cn.mass * cn.my;
        sz += cn.mass * cn.mz;
      }
    }
    nd.pcount = static_cast<std::int32_t>(t.px.size()) - nd.pbegin;
    detail::finalize_moments(nd, mass, sx, sy, sz, tp_.theta);
    nd.next = static_cast<std::int32_t>(t.nodes.size());
    t.nodes[idx] = nd;
    t.depth = std::max<int>(t.depth, nd.level);
  }

  const ParticleSet& p_;
  TreeParams tp_;
  std::vector<BuildNode> nodes_;
  std::vector<std::int32_t> link_;  // next particle in the same leaf, -1 = end
};

}  // namespace

FlatTree build_tree(const ParticleSet& p, const Cube& root, const TreeParams& tp, int threads,
                    BuildTimings* timings) {
  if (tp.leaf_capacity < 1) throw std::invalid_argument("leaf_capacity must be >= 1");
  if (tp.max_depth < 0 || tp.max_depth > 30) throw std::invalid_argument("max_depth must be in [0, 30]");
  if (p.size() > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
    throw std::invalid_argument("too many particles for 32-bit tree indices");
  BuildTimings local;
  BuildTimings& bt = timings ? *timings : local;
  bt = BuildTimings{};
  const double t0 = wall_time();
  FlatTree t = tp.build == TreeBuild::Insertion ? InsertionBuilder(p, root, tp).build(root)
                                                : detail::build_tree_morton(p, root, tp, threads, bt);
  bt.total = wall_time() - t0;
  return t;
}

}  // namespace nbody

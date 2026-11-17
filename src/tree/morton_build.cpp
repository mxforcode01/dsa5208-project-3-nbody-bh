// Morton-sorted octree construction (V3).
//
// 1. keys:  63-bit Morton key of every particle in the root cube        (parallel for)
// 2. sort:  (key, index) pairs; chunks sorted per thread, then merged    (parallel)
// 3. build: a node covers a contiguous range of sorted keys; its children are the
//           sub-ranges with equal 3-bit digit at the next level (binary search).
//    Parallel version: the top of the tree is expanded serially until every open range
//    holds at most `grain` particles (dense regions are split further, so the work per
//    range is balanced). These "frontier" subtrees are independent and are built with
//    an OpenMP dynamic loop into separate arrays; one prefix pass over the small top
//    skeleton gives each subtree its final offset, and the subtrees are copied into
//    place in parallel. Every node is written exactly once.
//    (A first version spawned an OpenMP task per large node and concatenated the child
//    arrays after each taskwait; nodes were copied once per task level, serially, and
//    the build got slower with more threads on Plummer data.)
//
// Leaf ranges are re-sorted by particle index, and moments are accumulated in the same
// order as the insertion builder (particles in index order, children in octant order),
// so the result is bit-for-bit the same tree. The only possible difference is a particle
// lying within rounding distance (~1e-16 relative) of a cell boundary, which the key
// quantisation and the insertion builder's coordinate comparison may place on different
// sides; the tree is then still a valid octree.
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "nbody/core/timer.hpp"
#include "nbody/tree/morton.hpp"
#include "tree_builders.hpp"

namespace nbody::detail {

namespace {

constexpr std::int64_t kMinParallel = 8192;  // below this, sort and build serially
constexpr std::int64_t kMinGrain = 1024;     // smallest frontier subtree (particles)
using KeyIndex = std::pair<std::uint64_t, std::uint32_t>;

void parallel_sort(std::vector<KeyIndex>& a, int threads) {
  const std::int64_t n = static_cast<std::int64_t>(a.size());
  if (threads <= 1 || n < kMinParallel) {
    std::sort(a.begin(), a.end());
    return;
  }
  const int chunks = threads;
  std::vector<std::int64_t> bound(chunks + 1);
  for (int c = 0; c <= chunks; ++c) bound[c] = n * c / chunks;
#pragma omp parallel for schedule(static) num_threads(threads)
  for (int c = 0; c < chunks; ++c) std::sort(a.begin() + bound[c], a.begin() + bound[c + 1]);
  for (int width = 1; width < chunks; width *= 2) {
#pragma omp parallel for schedule(static) num_threads(threads)
    for (int c = 0; c < chunks; c += 2 * width) {
      const int mid = std::min(c + width, chunks), end = std::min(c + 2 * width, chunks);
      std::inplace_merge(a.begin() + bound[c], a.begin() + bound[mid], a.begin() + bound[end]);
    }
  }
}

struct Cell {
  double cx, cy, cz, half;
};

class MortonBuilder {
 public:
  MortonBuilder(const ParticleSet& p, const TreeParams& tp, const std::vector<std::uint64_t>& keys,
                std::vector<std::uint32_t>& perm)
      : p_(p), tp_(tp), keys_(keys), perm_(perm) {}

  bool is_leaf(std::int64_t b, std::int64_t e, int level) const {
    return e - b <= tp_.leaf_capacity || level >= tp_.max_depth;
  }

  // start[0..8]: sub-range of each child octant (the next 3-bit digit is non-decreasing).
  void split(std::int64_t b, std::int64_t e, int level, std::int64_t start[9]) const {
    const int shift = 3 * (kMortonBits - 1 - level);
    start[0] = b;
    for (int oct = 1; oct < 8; ++oct)
      start[oct] = std::partition_point(keys_.begin() + start[oct - 1], keys_.begin() + e,
                                        [&](std::uint64_t k) { return static_cast<int>((k >> shift) & 7) < oct; }) -
                   keys_.begin();
    start[8] = e;
  }

  static Cell child(const Cell& c, int oct) {
    const double h = 0.5 * c.half;
    return {c.cx + ((oct & 4) ? h : -h), c.cy + ((oct & 2) ? h : -h), c.cz + ((oct & 1) ? h : -h), h};
  }

  Node make_node(std::int64_t b, std::int64_t e, int level, const Cell& c) const {
    Node nd{};
    nd.cx = c.cx;
    nd.cy = c.cy;
    nd.cz = c.cz;
    nd.half = c.half;
    nd.level = static_cast<std::uint8_t>(level);
    nd.pbegin = static_cast<std::int32_t>(b);
    nd.pcount = static_cast<std::int32_t>(e - b);
    return nd;
  }

  // Appends the subtree of sorted range [b, e) to `out` in depth-first order (serial).
  void build(std::int64_t b, std::int64_t e, int level, const Cell& c, std::vector<Node>& out) {
    const std::size_t idx = out.size();
    out.emplace_back();
    Node nd = make_node(b, e, level, c);
    double mass = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
    if (is_leaf(b, e, level)) {
      nd.flags = node_flags::kLeaf;
      std::sort(perm_.begin() + b, perm_.begin() + e);  // index order, as in the insertion builder
      for (std::int64_t k = b; k < e; ++k) {
        const std::uint32_t j = perm_[k];
        mass += p_.m[j];
        sx += p_.m[j] * p_.x[j];
        sy += p_.m[j] * p_.y[j];
        sz += p_.m[j] * p_.z[j];
      }
    } else {
      nd.flags = 0;
      std::int64_t start[9];
      split(b, e, level, start);
      for (int oct = 0; oct < 8; ++oct) {
        if (start[oct] == start[oct + 1]) continue;
        const std::size_t ci = out.size();
        build(start[oct], start[oct + 1], level + 1, child(c, oct), out);
        const Node& cn = out[ci];
        mass += cn.mass;
        sx += cn.mass * cn.mx;
        sy += cn.mass * cn.my;
        sz += cn.mass * cn.mz;
      }
    }
    finalize_moments(nd, mass, sx, sy, sz, tp_.theta);
    nd.next = static_cast<std::int32_t>(out.size());
    out[idx] = nd;
  }

 private:
  const ParticleSet& p_;
  const TreeParams& tp_;
  const std::vector<std::uint64_t>& keys_;
  std::vector<std::uint32_t>& perm_;
};

// Top-of-tree skeleton for the parallel build. Items are stored in depth-first order.
struct TopItem {
  std::int64_t b, e;
  int level;
  Cell cell;
  int frontier = -1;          // >= 0: index of the independently built subtree
  std::vector<int> children;  // top-item indices in octant order (internal items only)
  std::size_t size = 0;       // nodes in the subtree
  std::size_t offset = 0;     // position of the subtree root in the final array
  Node node{};                // internal items: the node itself
};

void expand(MortonBuilder& mb, std::int64_t b, std::int64_t e, int level, const Cell& c, std::int64_t grain,
            std::vector<TopItem>& items, std::vector<int>& frontier_items) {
  const int me = static_cast<int>(items.size());
  TopItem item;
  item.b = b;
  item.e = e;
  item.level = level;
  item.cell = c;
  items.push_back(std::move(item));
  if (e - b <= grain || mb.is_leaf(b, e, level)) {
    items[me].frontier = static_cast<int>(frontier_items.size());
    frontier_items.push_back(me);
    return;
  }
  std::int64_t start[9];
  mb.split(b, e, level, start);
  for (int oct = 0; oct < 8; ++oct) {
    if (start[oct] == start[oct + 1]) continue;
    const int ci = static_cast<int>(items.size());
    items[me].children.push_back(ci);
    expand(mb, start[oct], start[oct + 1], level + 1, MortonBuilder::child(c, oct), grain, items, frontier_items);
  }
}

}  // namespace

FlatTree build_tree_morton(const ParticleSet& p, const Cube& root, const TreeParams& tp, int threads,
                           BuildTimings& timings) {
  if (tp.max_depth > kMortonBits)
    throw std::invalid_argument("the morton tree build supports max_depth <= 21");
  const std::int64_t n = static_cast<std::int64_t>(p.size());
  FlatTree t;
  t.root = root;
  t.theta = tp.theta;
  if (n == 0) return t;
  const bool parallel = threads > 1 && n >= kMinParallel;

  double t0 = wall_time();
  std::vector<KeyIndex> ki(n);
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
  for (std::int64_t i = 0; i < n; ++i)
    ki[i] = {morton_key(p.x[i], p.y[i], p.z[i], root), static_cast<std::uint32_t>(i)};
  timings.keys = wall_time() - t0;

  t0 = wall_time();
  parallel_sort(ki, threads);
  timings.sort = wall_time() - t0;

  std::vector<std::uint64_t> keys(n);
  std::vector<std::uint32_t> perm(n);
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
  for (std::int64_t i = 0; i < n; ++i) {
    keys[i] = ki[i].first;
    perm[i] = ki[i].second;
  }

  MortonBuilder mb(p, tp, keys, perm);
  const Cell rc{root.cx, root.cy, root.cz, root.half};
  if (!parallel) {
    t.nodes.reserve(static_cast<std::size_t>(n / 2 + 8));
    mb.build(0, n, 0, rc, t.nodes);
  } else {
    // a. Serial expansion of the top of the tree into independent frontier ranges.
    const std::int64_t grain = std::max<std::int64_t>(kMinGrain, n / (8 * threads));
    std::vector<TopItem> items;
    std::vector<int> frontier_items;
    expand(mb, 0, n, 0, rc, grain, items, frontier_items);

    // b. Frontier subtrees in parallel (dynamic: their sizes differ).
    const int nf = static_cast<int>(frontier_items.size());
    std::vector<std::vector<Node>> sub(nf);
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int f = 0; f < nf; ++f) {
      const TopItem& it = items[frontier_items[f]];
      sub[f].reserve(static_cast<std::size_t>((it.e - it.b) / 2 + 8));
      mb.build(it.b, it.e, it.level, it.cell, sub[f]);
    }

    // c. Sizes and moments bottom-up (items are in depth-first order, so reverse order
    //    visits children before parents), then offsets top-down.
    for (int i = static_cast<int>(items.size()) - 1; i >= 0; --i) {
      TopItem& it = items[i];
      if (it.frontier >= 0) {
        it.size = sub[it.frontier].size();
        continue;
      }
      it.node = mb.make_node(it.b, it.e, it.level, it.cell);
      it.node.flags = 0;
      double mass = 0.0, sx = 0.0, sy = 0.0, sz = 0.0;
      it.size = 1;
      for (int c : it.children) {
        const TopItem& ch = items[c];
        const Node& cn = ch.frontier >= 0 ? sub[ch.frontier][0] : ch.node;
        mass += cn.mass;
        sx += cn.mass * cn.mx;
        sy += cn.mass * cn.my;
        sz += cn.mass * cn.mz;
        it.size += ch.size;
      }
      finalize_moments(it.node, mass, sx, sy, sz, tp.theta);
    }
    for (TopItem& it : items) {
      std::size_t off = it.offset + 1;
      for (int c : it.children) {
        items[c].offset = off;
        off += items[c].size;
      }
    }

    // d. Place every node exactly once.
    t.nodes.resize(items[0].size);
    for (const TopItem& it : items)
      if (it.frontier < 0) {
        t.nodes[it.offset] = it.node;
        t.nodes[it.offset].next = static_cast<std::int32_t>(it.offset + it.size);
      }
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads)
    for (int f = 0; f < nf; ++f) {
      const auto off = static_cast<std::int32_t>(items[frontier_items[f]].offset);
      Node* dst = t.nodes.data() + off;
      for (const Node& c : sub[f]) {
        *dst = c;
        dst->next += off;
        ++dst;
      }
    }
  }

  t.px.resize(n);
  t.py.resize(n);
  t.pz.resize(n);
  t.pm.resize(n);
  t.pidx = std::move(perm);
  int depth = 0;
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
  for (std::int64_t k = 0; k < n; ++k) {
    const std::uint32_t j = t.pidx[k];
    t.px[k] = p.x[j];
    t.py[k] = p.y[j];
    t.pz[k] = p.z[j];
    t.pm[k] = p.m[j];
  }
  const auto nn = static_cast<std::int64_t>(t.nodes.size());
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1) reduction(max : depth)
  for (std::int64_t i = 0; i < nn; ++i) depth = std::max<int>(depth, t.nodes[i].level);
  t.depth = depth;
  return t;
}

}  // namespace nbody::detail

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nbody/core/particles.hpp"

namespace nbody {

// Cubic cell: centre and half side length.
struct Cube {
  double cx = 0.0, cy = 0.0, cz = 0.0, half = 1.0;
};

// Axis-aligned box (closed), used as a LET target region.
struct AABB {
  double lo[3] = {0.0, 0.0, 0.0};
  double hi[3] = {0.0, 0.0, 0.0};
};

// Insertion: baseline, particles inserted one at a time into a pointer octree (serial).
// MortonSorted: 63-bit Morton keys -> parallel sort -> recursive split of sorted key
//   ranges, with OpenMP tasks for large independent subtrees. Produces exactly the same
//   tree (topology, leaf particle order, moments) as Insertion, so forces are identical.
enum class TreeBuild { Insertion, MortonSorted };

TreeBuild parse_tree_build(const std::string& name);  // "insertion" | "morton"
std::string to_string(TreeBuild b);

// Breakdown of one build (seconds). keys and sort are 0 for Insertion.
struct BuildTimings {
  double keys = 0.0;   // Morton key computation (preprocessing)
  double sort = 0.0;   // key sort (preprocessing)
  double total = 0.0;  // whole build including preprocessing
};

struct TreeParams {
  double theta = 0.5;      // opening angle; 0 opens every cell (exact direct sum)
  int leaf_capacity = 8;   // a leaf is split when it holds more particles than this...
  int max_depth = 21;      // ...unless it is at this level (coincident particles stop here);
                           // MortonSorted supports at most 21 (bits per dimension of the key)
  TreeBuild build = TreeBuild::Insertion;
};

namespace node_flags {
constexpr std::uint8_t kLeaf = 1;
constexpr std::uint8_t kPruned = 2;  // remote LET node whose subtree was not sent
}  // namespace node_flags

// One octree cell in a flat depth-first (pre-order) array. The first child of an
// internal node i is node i+1; `next` is the index of the first node after the
// subtree of i (== nodes.size() for the last subtree). This allows a stackless
// traversal: accept or leaf -> i = next; open -> i = i + 1.
struct Node {
  double mx, my, mz, mass;  // centre of mass and total mass (monopole)
  double ropen2;            // squared opening radius (2*half/theta + delta)^2; +inf if theta == 0
  double cx, cy, cz, half;  // geometric cell
  double delta;             // |centre of mass - cell centre|
  std::int32_t next;
  std::int32_t pbegin, pcount;  // particle range in FlatTree arrays (whole subtree for internal nodes)
  std::uint8_t flags;
  std::uint8_t level;
  bool is_leaf() const { return (flags & node_flags::kLeaf) != 0; }
  bool is_pruned() const { return (flags & node_flags::kPruned) != 0; }
};

// Flat octree plus its own copy of the source particles in leaf order. The
// ParticleSet the tree was built from is not reordered.
struct FlatTree {
  Cube root;
  double theta = 0.5;
  std::vector<Node> nodes;
  std::vector<double> px, py, pz, pm;  // source particles in leaf order
  std::vector<std::uint32_t> pidx;     // leaf order -> index in the source ParticleSet
  int depth = 0;                       // deepest level present
};

// Smallest padded cube that contains every particle (centre = middle of the
// bounding box). Degenerate inputs (0 or coincident particles) give half = 1.
Cube bounding_cube(const ParticleSet& p);
// Same construction from explicit bounds (used with globally reduced bounds in MPI runs).
// lo > hi in any dimension means "no particles".
Cube cube_from_bounds(const double lo[3], const double hi[3]);

// Opening radius squared for a cell of half side `half` and offset `delta`.
double opening_radius2(double half, double delta, double theta);

// Builds the octree of all particles of p inside `root` with the requested strategy.
// `threads` is used by MortonSorted only (Insertion is inherently serial).
FlatTree build_tree(const ParticleSet& p, const Cube& root, const TreeParams& tp, int threads = 1,
                    BuildTimings* timings = nullptr);

}  // namespace nbody

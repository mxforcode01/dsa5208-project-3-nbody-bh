#pragma once

#include <cstddef>

#include "nbody/core/particles.hpp"

namespace nbody {

// Error statistics of a per-particle quantity q (vector norm or scalar) against
// a reference. The relative error of particle i is
//   e_i = |q_i - q_ref,i| / max(|q_ref,i|, floor_fraction * rms(|q_ref|)),
// so particles whose reference value is close to zero do not dominate.
struct ErrorStats {
  std::size_t n = 0;
  double max_abs = 0.0;
  double rms_abs = 0.0;
  double rms_rel = 0.0;  // rms_abs / rms(|q_ref|)
  double median_rel = 0.0;
  double p99_rel = 0.0;
  double max_rel = 0.0;
};

struct ComparisonReport {
  ErrorStats accel;     // |a - a_ref|
  ErrorStats potential; // |pot - pot_ref|
  ErrorStats position;  // |x - x_ref| (use the absolute values; |x_ref| depends on the origin)
  ErrorStats velocity;  // |v - v_ref|
};

constexpr double kDefaultErrorFloor = 1e-3;

// Joins `test` and `ref` on particle id (any order). Throws std::invalid_argument
// if the id sets differ or contain duplicates.
ComparisonReport compare_by_id(const ParticleSet& test, const ParticleSet& ref,
                               double floor_fraction = kDefaultErrorFloor);

}  // namespace nbody

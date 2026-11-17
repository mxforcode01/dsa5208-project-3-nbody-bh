#include "nbody/sim/accuracy.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace nbody {

namespace {

ErrorStats make_stats(const std::vector<double>& diff, const std::vector<double>& refnorm,
                      double floor_fraction) {
  ErrorStats s;
  s.n = diff.size();
  if (s.n == 0) return s;
  double sum_d2 = 0.0, sum_r2 = 0.0;
  for (std::size_t i = 0; i < s.n; ++i) {
    sum_d2 += diff[i] * diff[i];
    sum_r2 += refnorm[i] * refnorm[i];
    s.max_abs = std::max(s.max_abs, diff[i]);
  }
  s.rms_abs = std::sqrt(sum_d2 / s.n);
  const double rms_ref = std::sqrt(sum_r2 / s.n);
  s.rms_rel = rms_ref > 0.0 ? s.rms_abs / rms_ref : (s.rms_abs == 0.0 ? 0.0 : INFINITY);

  const double floor = floor_fraction * rms_ref;
  std::vector<double> rel(s.n);
  for (std::size_t i = 0; i < s.n; ++i) {
    const double denom = std::max(refnorm[i], floor);
    rel[i] = denom > 0.0 ? diff[i] / denom : (diff[i] == 0.0 ? 0.0 : INFINITY);
  }
  std::sort(rel.begin(), rel.end());
  auto rank = [&](double q) {  // nearest-rank percentile
    std::size_t k = static_cast<std::size_t>(std::ceil(q * s.n));
    return rel[k == 0 ? 0 : k - 1];
  };
  s.median_rel = rank(0.5);
  s.p99_rel = rank(0.99);
  s.max_rel = rel.back();
  return s;
}

double norm3(double a, double b, double c) { return std::sqrt(a * a + b * b + c * c); }

}  // namespace

ComparisonReport compare_by_id(const ParticleSet& test, const ParticleSet& ref, double floor_fraction) {
  if (test.size() != ref.size())
    throw std::invalid_argument("compare_by_id: particle counts differ");
  std::unordered_map<std::uint64_t, std::size_t> index;
  index.reserve(ref.size());
  for (std::size_t j = 0; j < ref.size(); ++j)
    if (!index.emplace(ref.id[j], j).second)
      throw std::invalid_argument("compare_by_id: duplicate id in reference");

  const std::size_t n = test.size();
  std::vector<double> da(n), ra(n), dp(n), rp(n), dx(n), rx(n), dv(n), rv(n);
  std::vector<char> seen(ref.size(), 0);
  for (std::size_t i = 0; i < n; ++i) {
    const auto it = index.find(test.id[i]);
    if (it == index.end()) throw std::invalid_argument("compare_by_id: id missing from reference");
    const std::size_t j = it->second;
    if (seen[j]++) throw std::invalid_argument("compare_by_id: duplicate id in test set");
    da[i] = norm3(test.ax[i] - ref.ax[j], test.ay[i] - ref.ay[j], test.az[i] - ref.az[j]);
    ra[i] = norm3(ref.ax[j], ref.ay[j], ref.az[j]);
    dp[i] = std::fabs(test.pot[i] - ref.pot[j]);
    rp[i] = std::fabs(ref.pot[j]);
    dx[i] = norm3(test.x[i] - ref.x[j], test.y[i] - ref.y[j], test.z[i] - ref.z[j]);
    rx[i] = norm3(ref.x[j], ref.y[j], ref.z[j]);
    dv[i] = norm3(test.vx[i] - ref.vx[j], test.vy[i] - ref.vy[j], test.vz[i] - ref.vz[j]);
    rv[i] = norm3(ref.vx[j], ref.vy[j], ref.vz[j]);
  }
  ComparisonReport r;
  r.accel = make_stats(da, ra, floor_fraction);
  r.potential = make_stats(dp, rp, floor_fraction);
  r.position = make_stats(dx, rx, floor_fraction);
  r.velocity = make_stats(dv, rv, floor_fraction);
  return r;
}

}  // namespace nbody

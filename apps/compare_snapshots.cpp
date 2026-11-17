// Compares a test snapshot against a reference snapshot, joined on particle id.
// Exit code: 0 = compared (and within --max-accel-rel if given), 1 = tolerance exceeded, 2 = error.
#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "nbody/core/csv_logger.hpp"
#include "nbody/core/snapshot.hpp"
#include "nbody/sim/accuracy.hpp"

using namespace nbody;

namespace {

void print_row(const char* name, const ErrorStats& s) {
  std::printf("%-10s %12.4e %12.4e %12.4e %12.4e %12.4e %12.4e\n", name, s.max_abs, s.rms_abs, s.rms_rel,
              s.median_rel, s.p99_rel, s.max_rel);
}

void add_stats(CsvRow& row, const std::string& q, const ErrorStats& s) {
  row.add(q + "_max_abs", s.max_abs);
  row.add(q + "_rms_abs", s.rms_abs);
  row.add(q + "_rms_rel", s.rms_rel);
  row.add(q + "_median_rel", s.median_rel);
  row.add(q + "_p99_rel", s.p99_rel);
  row.add(q + "_max_rel", s.max_rel);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::printf("Usage: %s TEST.bin REF.bin [--floor F] [--max-accel-rel TOL] [--csv FILE] [--label KEY=VALUE]...\n"
                  "  --floor F           relative-error floor as a fraction of rms(|ref|) (default 1e-3)\n"
                  "  --max-accel-rel TOL fail (exit 1) if the max relative acceleration error exceeds TOL\n"
                  "  --csv FILE          append the statistics as one row to FILE\n"
                  "  --label KEY=VALUE   extra leading column(s) in the CSV row (e.g. theta=0.5)\n",
                  argv[0]);
      return 2;
    }
    double floor = kDefaultErrorFloor;
    double tol = -1.0;
    std::string csv;
    std::vector<std::pair<std::string, std::string>> labels;
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
      if (a == "--floor") floor = std::stod(argv[++i]);
      else if (a == "--max-accel-rel") tol = std::stod(argv[++i]);
      else if (a == "--csv") csv = argv[++i];
      else if (a == "--label") {
        const std::string kv = argv[++i];
        const auto eq = kv.find('=');
        if (eq == std::string::npos || eq == 0) throw std::invalid_argument("--label expects KEY=VALUE, got " + kv);
        labels.emplace_back(kv.substr(0, eq), kv.substr(eq + 1));
      } else throw std::invalid_argument("unknown option " + a);
    }
    const ParticleSet test = read_snapshot(argv[1]);
    const ParticleSet ref = read_snapshot(argv[2]);
    const ComparisonReport r = compare_by_id(test, ref, floor);

    std::printf("n = %zu, relative-error floor = %g * rms(|ref|)\n", r.accel.n, floor);
    std::printf("%-10s %12s %12s %12s %12s %12s %12s\n", "quantity", "max_abs", "rms_abs", "rms_rel",
                "median_rel", "p99_rel", "max_rel");
    print_row("accel", r.accel);
    print_row("potential", r.potential);
    print_row("position", r.position);
    print_row("velocity", r.velocity);
    if (!csv.empty()) {
      CsvRow row;
      for (const auto& kv : labels) row.add(kv.first, kv.second);
      row.add("n", static_cast<std::uint64_t>(r.accel.n));
      row.add("floor", floor);
      add_stats(row, "accel", r.accel);
      add_stats(row, "potential", r.potential);
      append_csv_row(csv, row);
    }
    if (tol >= 0.0) {
      const bool ok = r.accel.max_rel <= tol;
      std::printf("%s: max relative acceleration error %.4e %s %.4e\n", ok ? "PASS" : "FAIL", r.accel.max_rel,
                  ok ? "<=" : ">", tol);
      return ok ? 0 : 1;
    }
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}

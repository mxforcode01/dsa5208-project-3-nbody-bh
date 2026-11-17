// Configuration parsing, CSV schema protection, snapshots and accuracy statistics.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "nbody/core/config.hpp"
#include "nbody/core/csv_logger.hpp"
#include "nbody/core/initial_conditions.hpp"
#include "nbody/core/snapshot.hpp"
#include "nbody/sim/accuracy.hpp"
#include "nbody/solvers/direct_sum.hpp"
#include "test_harness.hpp"

using namespace nbody;

namespace {

RunConfig parse(std::vector<std::string> args) {
  args.insert(args.begin(), "prog");
  std::vector<const char*> argv;
  for (const auto& a : args) argv.push_back(a.c_str());
  return parse_args(static_cast<int>(argv.size()), argv.data());
}

ParticleSet small_system() {
  ICParams ic;
  ic.n_total = 200;
  ic.dist = Distribution::Plummer;
  ParticleSet p;
  generate_particles(ic, 0, 200, p);
  ForceStats st;
  DirectSumSolver(PhysicsParams{}).compute_accelerations(p, st);
  return p;
}

}  // namespace

NB_TEST(parse_valid_options) {
  const RunConfig c = parse({"--n", "5000", "--dist", "plummer", "--steps", "20", "--dt", "5e-4", "--eps",
                             "0.01", "--G", "2", "--seed", "7", "--threads", "3", "--csv", "out.csv"});
  NB_CHECK(c.n == 5000u && c.dist == Distribution::Plummer && c.steps == 20 && c.seed == 7u);
  NB_CHECK(c.dt == 5e-4 && c.phys.eps == 0.01 && c.phys.G == 2.0 && c.threads == 3);
  NB_CHECK(c.csv_path == "out.csv" && !c.show_help);
  NB_CHECK(parse({"--help"}).show_help);
  const RunConfig b = parse({"--solver", "bh", "--theta", "0.7", "--leaf-capacity", "16", "--max-depth", "12",
                             "--tree-build", "insertion"});
  NB_CHECK(b.solver == "bh" && b.tree.theta == 0.7 && b.tree.leaf_capacity == 16 && b.tree.max_depth == 12);
  NB_CHECK(b.tree.build == TreeBuild::Insertion);
}

NB_TEST(parse_rejects_bad_input) {
  NB_CHECK_THROWS(parse({"--bogus", "1"}));
  NB_CHECK_THROWS(parse({"--n"}));
  NB_CHECK_THROWS(parse({"--n", "-5"}));
  NB_CHECK_THROWS(parse({"--n", "12abc"}));
  NB_CHECK_THROWS(parse({"--dt", "0"}));
  NB_CHECK_THROWS(parse({"--eps", "-1"}));
  NB_CHECK_THROWS(parse({"--dist", "gaussian"}));
  NB_CHECK_THROWS(parse({"--threads", "0"}));
  NB_CHECK_THROWS(parse({"--solver", "fmm"}));
  NB_CHECK_THROWS(parse({"--theta", "-0.1"}));
  NB_CHECK_THROWS(parse({"--leaf-capacity", "0"}));
  NB_CHECK_THROWS(parse({"--max-depth", "31"}));
  NB_CHECK_THROWS(parse({"--device", "tpu"}));
  NB_CHECK_THROWS(parse({"--device", "gpu"}));  // direct sum has no GPU path
  NB_CHECK(parse({"--device", "gpu", "--solver", "bh", "--allow-host-fallback", "1"}).allow_host_fallback);
  NB_CHECK_THROWS(parse({"--shm", "yes"}));
  NB_CHECK_THROWS(parse({"--shm-group-size", "-1"}));
  NB_CHECK(parse({"--shm", "1", "--shm-group-size", "2"}).shm);
  NB_CHECK_THROWS(parse({"--comm", "rma"}));
  NB_CHECK_THROWS(parse({"--overlap-chunks", "0"}));
  NB_CHECK(parse({"--comm", "overlap", "--overlap-chunks", "8"}).comm == CommMode::Overlap);
  NB_CHECK(parse({"--comm", "p2p"}).comm == CommMode::P2P);
  NB_CHECK_THROWS(parse({"--sched", "auto"}));
  NB_CHECK_THROWS(parse({"--chunk", "-1"}));
  NB_CHECK(parse({"--sched", "guided", "--chunk", "16"}).sched.kind == ScheduleKind::Guided);
  NB_CHECK(parse({"--sched", "dynamic", "--chunk", "16"}).sched.chunk == 16);
  NB_CHECK_THROWS(parse({"--tree-build", "quadtree"}));
  NB_CHECK_THROWS(parse({"--tree-build", "morton", "--max-depth", "22"}));  // keys have 21 levels
  NB_CHECK(parse({"--tree-build", "morton"}).tree.build == TreeBuild::MortonSorted);
  NB_CHECK_THROWS(parse({"positional"}));
}

NB_TEST(csv_header_written_once_and_schema_mismatch_rejected) {
  const std::string path = "test_io_tmp.csv";
  std::remove(path.c_str());
  CsvRow a;
  a.add("name", "run,1");  // comma must be quoted
  a.add("value", 0.1);
  a.add("count", static_cast<std::uint64_t>(3));
  append_csv_row(path, a);
  append_csv_row(path, a);
  std::ifstream in(path);
  std::string l1, l2, l3, l4;
  std::getline(in, l1);
  std::getline(in, l2);
  std::getline(in, l3);
  NB_CHECK(l1 == "name,value,count");
  NB_CHECK(l2 == "\"run,1\",0.10000000000000001,3");
  NB_CHECK(l3 == l2);
  NB_CHECK(!std::getline(in, l4));
  CsvRow b;
  b.add("other", 1);
  NB_CHECK_THROWS(append_csv_row(path, b));
  std::remove(path.c_str());
}

NB_TEST(snapshot_round_trip_is_exact) {
  const std::string path = "test_io_tmp.bin";
  const ParticleSet p = small_system();
  write_snapshot(path, p);
  const ParticleSet q = read_snapshot(path);
  NB_CHECK(p.id == q.id && p.m == q.m && p.x == q.x && p.y == q.y && p.z == q.z);
  NB_CHECK(p.vx == q.vx && p.vy == q.vy && p.vz == q.vz);
  NB_CHECK(p.ax == q.ax && p.ay == q.ay && p.az == q.az && p.pot == q.pot);
  std::remove(path.c_str());
  NB_CHECK_THROWS(read_snapshot("does_not_exist.bin"));
}

NB_TEST(compare_by_id_joins_on_id_and_measures_errors) {
  const ParticleSet ref = small_system();

  // Same particles in reversed order: zero error.
  ParticleSet rev;
  for (std::size_t k = ref.size(); k-- > 0;) {
    rev.push_back(ref.id[k], ref.m[k], ref.x[k], ref.y[k], ref.z[k], ref.vx[k], ref.vy[k], ref.vz[k]);
    rev.ax.back() = ref.ax[k];
    rev.ay.back() = ref.ay[k];
    rev.az.back() = ref.az[k];
    rev.pot.back() = ref.pot[k];
  }
  const ComparisonReport r0 = compare_by_id(rev, ref);
  NB_CHECK(r0.accel.max_rel == 0.0 && r0.position.max_abs == 0.0 && r0.potential.max_rel == 0.0);

  // Perturb one acceleration by 1e-3 of its magnitude: max relative error = 1e-3,
  // median stays 0 (only 1 of 200 particles changed).
  ParticleSet t = ref;
  const double a5 = std::sqrt(ref.ax[5] * ref.ax[5] + ref.ay[5] * ref.ay[5] + ref.az[5] * ref.az[5]);
  t.ax[5] += 1e-3 * a5;
  const ComparisonReport r1 = compare_by_id(t, ref);
  NB_CHECK_NEAR(r1.accel.max_rel, 1e-3, 1e-12);
  NB_CHECK_NEAR(r1.accel.max_abs, 1e-3 * a5, 1e-15);
  NB_CHECK(r1.accel.median_rel == 0.0);

  // Mismatched ids are rejected.
  ParticleSet bad = ref;
  bad.id[0] = 999999;
  NB_CHECK_THROWS(compare_by_id(bad, ref));
}

NB_TEST(relative_error_floor_handles_zero_reference) {
  // Reference |a| = 0 for particle 0 (and 1 elsewhere): the denominator becomes
  // floor * rms(|a_ref|), so a small absolute error does not become infinite.
  ParticleSet ref;
  ref.push_back(0, 1, 0, 0, 0, 0, 0, 0);
  ref.push_back(1, 1, 1, 0, 0, 0, 0, 0);
  ref.ax[1] = 1.0;
  ParticleSet t = ref;
  t.ax[0] = 1e-6;
  const ComparisonReport r = compare_by_id(t, ref, 1e-3);
  const double rms_ref = std::sqrt(0.5);
  NB_CHECK_NEAR(r.accel.max_rel, 1e-6 / (1e-3 * rms_ref), 1e-12);
}

NB_TEST_MAIN()

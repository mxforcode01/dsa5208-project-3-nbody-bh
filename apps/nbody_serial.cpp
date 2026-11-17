// Serial N-body driver: V0 (direct sum) and V1 (Barnes-Hut).
#include <cstdio>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

#include "nbody/core/config.hpp"
#include "nbody/core/csv_logger.hpp"
#include "nbody/core/initial_conditions.hpp"
#include "nbody/core/run_metadata.hpp"
#include "nbody/core/snapshot.hpp"
#include "nbody/sim/simulation.hpp"
#include "nbody/solvers/barnes_hut.hpp"
#include "nbody/solvers/direct_sum.hpp"

using namespace nbody;

namespace {

std::string version_label(const RunConfig& c) { return c.solver == "direct" ? "V0" : "V1"; }

std::unique_ptr<ForceSolver> make_solver(const RunConfig& c) {
  if (c.solver == "bh")
    return std::make_unique<BarnesHutSolver>(c.phys, c.tree, c.threads, c.sched, c.device, c.allow_host_fallback);
  return std::make_unique<DirectSumSolver>(c.phys, c.threads);
}

void write_summary_csv(const RunConfig& c, const RunResult& r) {
  const Diagnostics& d0 = r.diag.front().d;
  const Diagnostics& d1 = r.diag.back().d;
  CsvRow row;
  row.add("version", version_label(c));
  row.add("solver", c.solver);
  row.add("n", c.n);
  row.add("dist", to_string(c.dist));
  row.add("seed", c.seed);
  row.add("scale", c.scale);
  row.add("steps", c.steps);
  row.add("dt", c.dt);
  row.add("G", c.phys.G);
  row.add("eps", c.phys.eps);
  row.add("theta", c.solver == "bh" ? c.tree.theta : 0.0);
  row.add("leaf_capacity", c.tree.leaf_capacity);
  row.add("max_depth", c.tree.max_depth);
  row.add("tree_build", c.solver == "bh" ? to_string(c.tree.build) : std::string("none"));
  row.add("mpi_ranks", 1);
  row.add("omp_threads", c.threads);
  row.add("sched", c.solver == "bh" ? to_string(c.sched.kind) : std::string("static"));
  row.add("chunk", c.sched.chunk);
  row.add("t_total", r.total_time);
  row.add("t_force", r.timings.force);
  row.add("t_tree_build", r.stats.tree_build_time);
  row.add("t_tree_keys", r.stats.tree_keys_time);
  row.add("t_tree_sort", r.stats.tree_sort_time);
  row.add("t_traversal", r.stats.traversal_time);
  row.add("t_integrate", r.timings.integrate);
  row.add("t_diagnostics", r.timings.diagnostics);
  row.add("interactions_pp", r.stats.pp);
  row.add("interactions_pc", r.stats.pc);
  row.add("tree_nodes", static_cast<std::uint64_t>(r.stats.tree_nodes));
  row.add("tree_depth", r.stats.tree_depth);
  row.add("thread_imbalance", r.stats.thread_imbalance());
  row.add("thread_work_imbalance", r.stats.thread_work_imbalance());
  row.add("device", to_string(c.device));
  row.add("device_executed", to_string(static_cast<Executed>(r.stats.executed)));
  row.add("num_devices", r.stats.num_devices);
  row.add("t_gpu_h2d", r.stats.gpu_h2d_time);
  row.add("t_gpu_kernel", r.stats.gpu_kernel_time);
  row.add("t_gpu_d2h", r.stats.gpu_d2h_time);
  row.add("gpu_h2d_bytes", r.stats.gpu_h2d_bytes);
  row.add("energy_initial", d0.total_energy());
  row.add("energy_final", d1.total_energy());
  row.add("rel_energy_drift", relative_energy_drift(d0, d1));
  row.add("rel_momentum_drift", relative_momentum_drift(d0, d1));
  row.add("mass_final", d1.mass);
  add_metadata(collect_run_metadata(), row);
  append_csv_row(c.csv_path, row);
}

void write_diag_csv(const RunConfig& c, const RunResult& r) {
  const Diagnostics& d0 = r.diag.front().d;
  for (const DiagRecord& rec : r.diag) {
    CsvRow row;
    row.add("version", version_label(c));
    row.add("n", c.n);
    row.add("dist", to_string(c.dist));
    row.add("seed", c.seed);
    row.add("dt", c.dt);
    row.add("step", rec.step);
    row.add("time", rec.time);
    row.add("mass", rec.d.mass);
    row.add("px", rec.d.px);
    row.add("py", rec.d.py);
    row.add("pz", rec.d.pz);
    row.add("kinetic", rec.d.kinetic);
    row.add("potential", rec.d.potential);
    row.add("energy", rec.d.total_energy());
    row.add("rel_energy_drift", relative_energy_drift(d0, rec.d));
    row.add("rel_momentum_drift", relative_momentum_drift(d0, rec.d));
    append_csv_row(c.diag_csv_path, row);
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const RunConfig c = parse_args(argc, argv);
    if (c.show_help) {
      std::printf("%s", usage(argv[0]).c_str());
      return 0;
    }
    if (c.solver == "dist-bh") throw std::invalid_argument("--solver dist-bh needs nbody_mpi");

    ICParams ic;
    ic.n_total = c.n;
    ic.seed = c.seed;
    ic.dist = c.dist;
    ic.scale = c.scale;
    ic.G = c.phys.G;
    ParticleSet p;
    generate_particles(ic, 0, c.n, p);
    remove_com_velocity(p);

    const std::unique_ptr<ForceSolver> solver = make_solver(c);
    SimParams sp;
    sp.steps = c.steps;
    sp.dt = c.dt;
    sp.diag_every = c.diag_every;
    sp.threads = c.threads;
    const RunResult r = Simulation(sp, *solver).run(p);

    std::printf("%s solver=%s n=%llu dist=%s seed=%llu steps=%d dt=%g eps=%g threads=%d\n",
                version_label(c).c_str(), c.solver.c_str(), static_cast<unsigned long long>(c.n),
                to_string(c.dist).c_str(), static_cast<unsigned long long>(c.seed), c.steps, c.dt,
                c.phys.eps, c.threads);
    if (c.solver == "bh")
      std::printf("tree: build=%s theta=%g leaf_capacity=%d max_depth=%d | last tree %zu nodes, depth %d\n",
                  to_string(c.tree.build).c_str(), c.tree.theta, c.tree.leaf_capacity, c.tree.max_depth,
                  r.stats.tree_nodes, r.stats.tree_depth);
    std::printf("%8s %12s %22s %22s %12s %12s\n", "step", "time", "energy", "kinetic", "dE/|E0|", "dP/sum(m|v|)");
    const Diagnostics& d0 = r.diag.front().d;
    for (const DiagRecord& rec : r.diag)
      std::printf("%8d %12.6g %22.15e %22.15e %12.3e %12.3e\n", rec.step, rec.time, rec.d.total_energy(),
                  rec.d.kinetic, relative_energy_drift(d0, rec.d), relative_momentum_drift(d0, rec.d));
    std::printf("time: total %.6f s | force %.6f s (tree build %.6f [keys %.6f, sort %.6f], traversal %.6f) | "
                "integrate %.6f s | diagnostics %.6f s\n",
                r.total_time, r.timings.force, r.stats.tree_build_time, r.stats.tree_keys_time,
                r.stats.tree_sort_time, r.stats.traversal_time, r.timings.integrate, r.timings.diagnostics);
    std::printf("interactions: pp %llu pc %llu\n", static_cast<unsigned long long>(r.stats.pp),
                static_cast<unsigned long long>(r.stats.pc));
    if (c.device == Device::GPU)
      std::printf("device: requested gpu, executed on %s (omp_get_num_devices = %d) | h2d %.4f s (%.1f MB), "
                  "kernel %.4f s, d2h %.4f s\n",
                  to_string(static_cast<Executed>(r.stats.executed)).c_str(), r.stats.num_devices,
                  r.stats.gpu_h2d_time, r.stats.gpu_h2d_bytes / 1048576.0, r.stats.gpu_kernel_time,
                  r.stats.gpu_d2h_time);
    else if (c.solver == "bh")
      std::printf("openmp: sched=%s chunk=%d | thread imbalance (busy max/mean) %.3f | work imbalance %.3f\n",
                  to_string(c.sched.kind).c_str(), c.sched.chunk, r.stats.thread_imbalance(),
                  r.stats.thread_work_imbalance());

    if (!c.snapshot_path.empty()) write_snapshot(c.snapshot_path, p);
    if (!c.csv_path.empty()) write_summary_csv(c, r);
    if (!c.diag_csv_path.empty()) write_diag_csv(c, r);
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 2;
  }
}

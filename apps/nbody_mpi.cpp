// Distributed N-body driver: V2 (MPI + OpenMP Barnes-Hut with Locally Essential Trees).
// Run with e.g.  OMP_NUM_THREADS=2 mpirun -np 4 ./nbody_mpi --n 100000 --threads 2
#include <sys/resource.h>

#include <cstdio>
#include <exception>
#include <stdexcept>
#include <string>

#include "nbody/core/config.hpp"
#include "nbody/core/csv_logger.hpp"
#include "nbody/core/run_metadata.hpp"
#include "nbody/core/snapshot.hpp"
#include "nbody/mpi/dist_bh.hpp"
#include "nbody/mpi/mpi_env.hpp"
#include "nbody/mpi/particle_comm.hpp"
#include "nbody/sim/simulation.hpp"

using namespace nbody;

namespace {

void reduce_diagnostics(MPI_Comm comm, Diagnostics& d) {
  double v[7] = {d.mass, d.px, d.py, d.pz, d.momentum_scale, d.kinetic, d.potential};
  MPI_Allreduce(MPI_IN_PLACE, v, 7, MPI_DOUBLE, MPI_SUM, comm);
  d.mass = v[0];
  d.px = v[1];
  d.py = v[2];
  d.pz = v[3];
  d.momentum_scale = v[4];
  d.kinetic = v[5];
  d.potential = v[6];
}

// Peak resident set size of this process in kB (Linux). Pages of a shared-memory window
// are counted in every process that touches them, so RSS overstates shm memory use;
// the byte counters (LET received, segment size) are the exact measure.
double peak_rss_kb() {
  rusage u{};
  getrusage(RUSAGE_SELF, &u);
  return static_cast<double>(u.ru_maxrss);
}

// Adds <name>_min, <name>_mean, <name>_max over ranks (collective: call on every rank).
void add_spread(MPI_Comm comm, CsvRow& row, const std::string& name, double v) {
  const RankSpread s = rank_spread(comm, v);
  row.add(name + "_min", s.min);
  row.add(name + "_mean", s.mean);
  row.add(name + "_max", s.max);
}

}  // namespace

int main(int argc, char** argv) {
  MpiEnv env(argc, argv);
  const MPI_Comm comm = MPI_COMM_WORLD;
  const bool root = env.rank() == 0;
  try {
    RunConfig defaults;
    defaults.solver = "dist-bh";
    const RunConfig c = parse_args(argc, argv, defaults);
    if (c.show_help) {
      if (root) std::printf("%s", usage(argv[0]).c_str());
      return 0;
    }
    if (c.solver != "dist-bh") throw std::invalid_argument("nbody_mpi runs --solver dist-bh only");

    ICParams ic;
    ic.n_total = c.n;
    ic.seed = c.seed;
    ic.dist = c.dist;
    ic.scale = c.scale;
    ic.G = c.phys.G;
    ParticleSet p;
    generate_particles_distributed(comm, ic, p);
    remove_com_velocity_global(comm, p);

    DistParams dp;
    dp.rebalance_every = c.rebalance_every;
    dp.comm = c.comm;
    dp.overlap_chunks = c.overlap_chunks;
    dp.let_boxes = c.let_boxes;
    dp.shm = c.shm;
    dp.shm_group_size = c.shm_group_size;
    dp.device = c.device;
    dp.allow_host_fallback = c.allow_host_fallback;
    DistributedBHSolver solver(comm, c.phys, c.tree, dp, c.threads, c.sched);
    SimParams sp;
    sp.steps = c.steps;
    sp.dt = c.dt;
    sp.diag_every = c.diag_every;
    sp.threads = c.threads;
    sp.reduce_diagnostics = [comm](Diagnostics& d) { reduce_diagnostics(comm, d); };

    MPI_Barrier(comm);
    const RunResult r = Simulation(sp, solver).run(p);

    // Global statistics (collective; every rank participates).
    const ForceStats& s = r.stats;
    const RankSpread total = rank_spread(comm, r.total_time);
    const std::uint64_t n_global = allreduce_u64(comm, p.size(), MPI_SUM);
    const std::uint64_t pp = allreduce_u64(comm, s.pp, MPI_SUM), pc = allreduce_u64(comm, s.pc, MPI_SUM);
    const std::uint64_t violations = allreduce_u64(comm, s.pruned_violations, MPI_SUM);
    const std::uint64_t let_bytes = allreduce_u64(comm, s.let_bytes_sent, MPI_SUM);
    const std::uint64_t migrated = allreduce_u64(comm, s.particles_migrated, MPI_SUM);
    const double coverage = allreduce(comm, s.mass_coverage_err, MPI_MAX);
    double last_work = 0.0;
    for (double w : p.work) last_work += w;
    const RankSpread work = rank_spread(comm, last_work);
    const RankSpread count = rank_spread(comm, static_cast<double>(p.size()));

    CsvRow row;
    row.add("version", "V2");
    row.add("solver", c.solver);
    row.add("n", c.n);
    row.add("dist", to_string(c.dist));
    row.add("seed", c.seed);
    row.add("scale", c.scale);
    row.add("steps", c.steps);
    row.add("dt", c.dt);
    row.add("G", c.phys.G);
    row.add("eps", c.phys.eps);
    row.add("theta", c.tree.theta);
    row.add("leaf_capacity", c.tree.leaf_capacity);
    row.add("max_depth", c.tree.max_depth);
    row.add("tree_build", to_string(c.tree.build));
    row.add("comm", to_string(c.comm));
    row.add("overlap_chunks", c.comm == CommMode::Overlap ? c.overlap_chunks : 0);
    row.add("let_boxes", c.let_boxes);
    row.add("shm", c.shm ? 1 : 0);
    row.add("shm_group_size", c.shm_group_size);
    row.add("rebalance_every", c.rebalance_every);
    row.add("decomposition", "morton-bisection");
    row.add("mpi_ranks", env.size());
    row.add("omp_threads", c.threads);
    row.add("sched", to_string(c.sched.kind));
    row.add("chunk", c.sched.chunk);
    row.add("mpi_thread_level", env.thread_level());
    row.add("t_total_max", total.max);
    add_spread(comm, row, "t_force", r.timings.force);
    add_spread(comm, row, "t_decomposition", s.decomposition_time);
    add_spread(comm, row, "t_migration", s.migration_time);
    add_spread(comm, row, "t_tree_build", s.tree_build_time);
    add_spread(comm, row, "t_tree_keys", s.tree_keys_time);
    add_spread(comm, row, "t_tree_sort", s.tree_sort_time);
    add_spread(comm, row, "t_let_build", s.let_build_time);
    add_spread(comm, row, "t_let_exchange", s.let_exchange_time);
    add_spread(comm, row, "t_let_wait", s.let_wait_time);
    add_spread(comm, row, "t_force_local", s.force_local_time);
    add_spread(comm, row, "t_force_remote", s.force_remote_time);
    add_spread(comm, row, "t_integrate", r.timings.integrate);
    add_spread(comm, row, "t_diagnostics", r.timings.diagnostics);
    row.add("interactions_pp", pp);
    row.add("interactions_pc", pc);
    row.add("work_imbalance_last_step", work.imbalance());
    add_spread(comm, row, "thread_imbalance", s.thread_imbalance());
    add_spread(comm, row, "thread_work_imbalance", s.thread_work_imbalance());
    row.add("particles_min", count.min);
    row.add("particles_max", count.max);
    row.add("let_bytes_sent_total", let_bytes);
    const RankSpread let_recv = rank_spread(comm, static_cast<double>(s.let_bytes_received));
    const RankSpread publish = rank_spread(comm, s.shm_publish_time);
    const RankSpread sync = rank_spread(comm, s.shm_sync_time);
    const RankSpread rss = rank_spread(comm, peak_rss_kb());
    add_spread(comm, row, "let_bytes_received", static_cast<double>(s.let_bytes_received));
    add_spread(comm, row, "shm_segment_bytes", static_cast<double>(s.shm_segment_bytes));
    add_spread(comm, row, "t_shm_publish", s.shm_publish_time);
    add_spread(comm, row, "t_shm_sync", s.shm_sync_time);
    row.add("shm_reallocations_max", allreduce(comm, static_cast<double>(s.shm_reallocations), MPI_MAX));
    add_spread(comm, row, "maxrss_kb", peak_rss_kb());
    // Device actually used: the "worst" over ranks (host-fallback > gpu > cpu), never assumed.
    const int executed = static_cast<int>(allreduce(comm, static_cast<double>(s.executed), MPI_MAX));
    row.add("device", to_string(c.device));
    row.add("device_executed", to_string(static_cast<Executed>(executed)));
    row.add("num_devices_min", allreduce(comm, static_cast<double>(s.num_devices), MPI_MIN));
    row.add("device_id_min", allreduce(comm, static_cast<double>(s.device_id), MPI_MIN));
    row.add("device_id_max", allreduce(comm, static_cast<double>(s.device_id), MPI_MAX));
    const RankSpread h2d = rank_spread(comm, s.gpu_h2d_time), kern = rank_spread(comm, s.gpu_kernel_time),
                     d2h = rank_spread(comm, s.gpu_d2h_time);
    add_spread(comm, row, "t_gpu_h2d", s.gpu_h2d_time);
    add_spread(comm, row, "t_gpu_kernel", s.gpu_kernel_time);
    add_spread(comm, row, "t_gpu_d2h", s.gpu_d2h_time);
    row.add("particles_migrated_total", migrated);
    row.add("pruned_violations", violations);
    row.add("mass_coverage_err", coverage);
    const Diagnostics& d0 = r.diag.front().d;
    const Diagnostics& d1 = r.diag.back().d;
    row.add("energy_initial", d0.total_energy());
    row.add("energy_final", d1.total_energy());
    row.add("rel_energy_drift", relative_energy_drift(d0, d1));
    row.add("rel_momentum_drift", relative_momentum_drift(d0, d1));
    row.add("mass_final", d1.mass);
    add_metadata(collect_run_metadata(), row);

    const ParticleSet all = c.snapshot_path.empty() ? ParticleSet{} : gather_to_root(comm, p);

    if (root) {
      std::printf("V2 dist-bh n=%llu dist=%s seed=%llu steps=%d dt=%g eps=%g theta=%g ranks=%d threads=%d\n",
                  static_cast<unsigned long long>(c.n), to_string(c.dist).c_str(),
                  static_cast<unsigned long long>(c.seed), c.steps, c.dt, c.phys.eps, c.tree.theta, env.size(),
                  c.threads);
      std::printf("%8s %12s %22s %12s %12s\n", "step", "time", "energy", "dE/|E0|", "dP/sum(m|v|)");
      for (const DiagRecord& rec : r.diag)
        std::printf("%8d %12.6g %22.15e %12.3e %12.3e\n", rec.step, rec.time, rec.d.total_energy(),
                    relative_energy_drift(d0, rec.d), relative_momentum_drift(d0, rec.d));
      for (const auto& f : row.fields())
        if (f.first.rfind("t_", 0) == 0 && f.first.size() > 4 && f.first.compare(f.first.size() - 4, 4, "_max") == 0)
          std::printf("  %-22s %.6f s\n", f.first.c_str(), std::stod(f.second));
      std::printf("particles %llu (per rank %g..%g) | work imbalance %.3f | LET %.3f MB | migrated %llu\n",
                  static_cast<unsigned long long>(n_global), count.min, count.max, work.imbalance(),
                  let_bytes / 1048576.0, static_cast<unsigned long long>(migrated));
      std::printf("memory: LET received %.3f MB/rank max | peak RSS %.1f MB/rank max\n", let_recv.max / 1048576.0,
                  rss.max / 1024.0);
      if (c.device == Device::GPU)
        std::printf("device: requested gpu, executed on %s | h2d %.4f s, kernel %.4f s, d2h %.4f s (max over ranks)\n",
                    to_string(static_cast<Executed>(executed)).c_str(), h2d.max, kern.max, d2h.max);
      if (c.shm)
        std::printf("shm: group size %s | segment %.3f MB/rank | publish %.4f s (sync %.4f s), max over ranks\n",
                    c.shm_group_size > 0 ? std::to_string(c.shm_group_size).c_str() : "node",
                    s.shm_segment_bytes / 1048576.0, publish.max, sync.max);
      std::printf("openmp: sched=%s chunk=%d | comm=%s\n", to_string(c.sched.kind).c_str(), c.sched.chunk,
                  to_string(c.comm).c_str());
      std::printf("checks: pruned violations %llu, mass coverage error %.3e\n",
                  static_cast<unsigned long long>(violations), coverage);
      if (!c.snapshot_path.empty()) write_snapshot(c.snapshot_path, all);
      if (!c.csv_path.empty()) append_csv_row(c.csv_path, row);
    }
    if (n_global != c.n) throw std::runtime_error("particle count changed: " + std::to_string(n_global));
    return violations == 0 ? 0 : 4;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "rank %d error: %s\n", env.rank(), e.what());
    MPI_Abort(MPI_COMM_WORLD, 2);
    return 2;
  }
}

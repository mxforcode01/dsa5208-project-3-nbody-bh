#include "nbody/core/config.hpp"

#include <cmath>
#include <stdexcept>

namespace nbody {

namespace {

std::uint64_t to_u64(const std::string& key, const std::string& v) {
  std::size_t pos = 0;
  unsigned long long r = 0;
  try {
    if (!v.empty() && v[0] == '-') throw std::invalid_argument("negative");
    r = std::stoull(v, &pos);
  } catch (const std::exception&) {
    pos = std::string::npos;
  }
  if (pos != v.size()) throw std::invalid_argument("--" + key + ": expected a non-negative integer, got '" + v + "'");
  return static_cast<std::uint64_t>(r);
}

int to_int(const std::string& key, const std::string& v) {
  std::size_t pos = 0;
  int r = 0;
  try {
    r = std::stoi(v, &pos);
  } catch (const std::exception&) {
    pos = std::string::npos;
  }
  if (pos != v.size()) throw std::invalid_argument("--" + key + ": expected an integer, got '" + v + "'");
  return r;
}

double to_double(const std::string& key, const std::string& v) {
  std::size_t pos = 0;
  double r = 0.0;
  try {
    r = std::stod(v, &pos);
  } catch (const std::exception&) {
    pos = std::string::npos;
  }
  if (pos != v.size() || !std::isfinite(r))
    throw std::invalid_argument("--" + key + ": expected a finite number, got '" + v + "'");
  return r;
}

void require(bool ok, const std::string& msg) {
  if (!ok) throw std::invalid_argument(msg);
}

}  // namespace

RunConfig parse_args(int argc, const char* const* argv, const RunConfig& defaults) {
  RunConfig c = defaults;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      c.show_help = true;
      continue;
    }
    require(arg.size() > 2 && arg.compare(0, 2, "--") == 0, "unexpected argument '" + arg + "'");
    const std::string key = arg.substr(2);
    require(i + 1 < argc, "missing value for " + arg);
    const std::string v = argv[++i];

    if (key == "n") c.n = to_u64(key, v);
    else if (key == "seed") c.seed = to_u64(key, v);
    else if (key == "dist") c.dist = parse_distribution(v);
    else if (key == "scale") c.scale = to_double(key, v);
    else if (key == "steps") c.steps = to_int(key, v);
    else if (key == "dt") c.dt = to_double(key, v);
    else if (key == "G") c.phys.G = to_double(key, v);
    else if (key == "eps") c.phys.eps = to_double(key, v);
    else if (key == "solver") c.solver = v;
    else if (key == "theta") c.tree.theta = to_double(key, v);
    else if (key == "leaf-capacity") c.tree.leaf_capacity = to_int(key, v);
    else if (key == "max-depth") c.tree.max_depth = to_int(key, v);
    else if (key == "tree-build") c.tree.build = parse_tree_build(v);
    else if (key == "rebalance-every") c.rebalance_every = to_int(key, v);
    else if (key == "comm") c.comm = parse_comm_mode(v);
    else if (key == "overlap-chunks") c.overlap_chunks = to_int(key, v);
    else if (key == "let-boxes") c.let_boxes = to_int(key, v);
    else if (key == "shm") {
      require(v == "0" || v == "1", "--shm: expected 0 or 1");
      c.shm = v == "1";
    } else if (key == "shm-group-size") c.shm_group_size = to_int(key, v);
    else if (key == "device") c.device = parse_device(v);
    else if (key == "allow-host-fallback") {
      require(v == "0" || v == "1", "--allow-host-fallback: expected 0 or 1");
      c.allow_host_fallback = v == "1";
    }
    else if (key == "threads") c.threads = to_int(key, v);
    else if (key == "sched") c.sched.kind = parse_schedule(v);
    else if (key == "chunk") c.sched.chunk = to_int(key, v);
    else if (key == "diag-every") c.diag_every = to_int(key, v);
    else if (key == "csv") c.csv_path = v;
    else if (key == "diag-csv") c.diag_csv_path = v;
    else if (key == "snapshot") c.snapshot_path = v;
    else throw std::invalid_argument("unknown option " + arg + " (see --help)");
  }
  require(c.n >= 1, "--n must be >= 1");
  require(c.scale > 0.0, "--scale must be > 0");
  require(c.steps >= 0, "--steps must be >= 0");
  require(c.dt > 0.0, "--dt must be > 0");
  require(c.phys.G > 0.0, "--G must be > 0");
  require(c.phys.eps >= 0.0, "--eps must be >= 0");
  require(c.threads >= 1, "--threads must be >= 1");
  require(c.sched.chunk >= 0, "--chunk must be >= 0 (0 = OpenMP default)");
  require(c.diag_every >= 0, "--diag-every must be >= 0");
  require(c.solver == "direct" || c.solver == "bh" || c.solver == "dist-bh",
          "--solver must be 'direct', 'bh' or 'dist-bh'");
  require(c.rebalance_every >= 1, "--rebalance-every must be >= 1");
  require(c.overlap_chunks >= 1, "--overlap-chunks must be >= 1");
  require(c.let_boxes >= 1, "--let-boxes must be >= 1");
  require(c.shm_group_size >= 0, "--shm-group-size must be >= 0");
  require(c.device == Device::CPU || c.solver != "direct", "--device gpu needs --solver bh or dist-bh");
  require(c.tree.theta >= 0.0, "--theta must be >= 0");
  require(c.tree.leaf_capacity >= 1, "--leaf-capacity must be >= 1");
  require(c.tree.max_depth >= 0 && c.tree.max_depth <= 30, "--max-depth must be in [0, 30]");
  require(c.tree.build != TreeBuild::MortonSorted || c.tree.max_depth <= 21,
          "--tree-build morton supports --max-depth <= 21");
  return c;
}

std::string usage(const std::string& program) {
  return "Usage: " + program + " [options]\n"
         "  --n N              number of particles (default 1000)\n"
         "  --seed S           random seed (default 42)\n"
         "  --dist D           uniform | plummer (default uniform)\n"
         "  --scale L          uniform cube side / Plummer scale radius (default 1)\n"
         "  --steps K          number of time steps (default 10)\n"
         "  --dt H             time-step size (default 1e-3)\n"
         "  --G G              gravitational constant (default 1)\n"
         "  --eps E            Plummer softening length (default 1e-3)\n"
         "  --solver S         direct (V0) | bh (V1 Barnes-Hut) | dist-bh (V2, nbody_mpi)\n"
         "  --theta T          Barnes-Hut opening angle (default 0.5; 0 = exact)\n"
         "  --leaf-capacity C  max particles per leaf before splitting (default 8)\n"
         "  --max-depth D      max tree level, 0..30 (default 21)\n"
         "  --tree-build B     insertion (default, serial) | morton (parallel, uses --threads)\n"
         "  --rebalance-every K  MPI: new decomposition every K force evaluations (default 1)\n"
         "  --comm M           MPI LET exchange: blocking (default) | p2p | overlap\n"
         "  --overlap-chunks K MPI overlap: local force work split into K chunks (default 32)\n"
         "  --let-boxes K      MPI: describe each rank's particles to senders by up to K tight boxes (default 64; 1 = one box)\n"
         "  --shm 0|1          MPI: share local trees inside a node via shared-memory windows (default 0)\n"
         "  --shm-group-size K MPI shm: groups of K ranks per node (0 = whole node; >0 emulates nodes)\n"
         "  --device D         Barnes-Hut tree walks: cpu (default) | gpu (OpenMP target offload)\n"
         "  --allow-host-fallback 0|1  gpu: accept target regions that ran on the host (default 0)\n"
         "  --threads T        OpenMP threads for the force loop (default 1 = serial V0)\n"
         "  --sched S          Barnes-Hut force-loop schedule: static (default) | dynamic | guided\n"
         "  --chunk K          schedule chunk size (default 0 = OpenMP default)\n"
         "  --diag-every K     diagnostics every K steps (0 = first/last only)\n"
         "  --csv FILE         append a summary row to FILE\n"
         "  --diag-csv FILE    append per-diagnostic-step rows to FILE\n"
         "  --snapshot FILE    write the final state to FILE (binary)\n";
}

}  // namespace nbody

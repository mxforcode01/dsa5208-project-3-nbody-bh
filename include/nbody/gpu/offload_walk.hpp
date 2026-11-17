#pragma once

#include <vector>

#include "nbody/core/device.hpp"
#include "nbody/tree/octree.hpp"
#include "nbody/tree/traversal.hpp"

namespace nbody {

struct OffloadReport {
  int num_devices = 0;       // omp_get_num_devices() at the time of the call
  Executed executed = Executed::HostFallback;
  double h2d = 0.0;          // target enter data (targets, results, every tree), seconds
  double kernel = 0.0;       // target teams distribute parallel for regions
  double d2h = 0.0;          // target exit data (results back, device copies freed)
  double h2d_bytes = 0.0;
};

// V7: Barnes-Hut walks with OpenMP target offload.
// Targets are the particles of `local` in leaf order (k = 0..n-1, particle pidx[k]).
// `local` is walked first (skipping each target itself), then every tree in `remote` in
// the given order - the same order as the CPU path. acc[k] receives the sums of target k.
// One device thread per target (teams distribute parallel for); the stackless walk needs
// no device stack. Execution on the device is verified inside the first kernel with
// omp_is_initial_device(); a host fallback is reported, never counted as GPU execution.
// Throws std::runtime_error if the walks did not run on a device and !allow_host_fallback.
void offload_walks(const FlatTree& local, const std::vector<TreeView>& remote, double eps2,
                   std::vector<WalkResult>& acc, bool allow_host_fallback, OffloadReport& report);

}  // namespace nbody

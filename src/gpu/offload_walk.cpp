#include "nbody/gpu/offload_walk.hpp"

#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nbody/core/timer.hpp"

namespace nbody {

namespace {

// One tree on the device: map its arrays, run one kernel over all targets, free it.
void walk_on_device(const TreeView& t, bool skip_self, long long n, const double* tx, const double* ty,
                    const double* tz, const std::uint32_t* tself, double eps2, WalkResult* acc, int& on_device,
                    bool probe, OffloadReport& rep) {
  if (t.n_nodes == 0 || n == 0) return;
  const Node* nodes = t.nodes;
  const std::int32_t nn = t.n_nodes;
  const double *px = t.px, *py = t.py, *pz = t.pz, *pm = t.pm;
  const std::uint32_t* pidx = t.pidx;
  const std::int32_t np = nn > 0 ? nodes[0].pbegin + nodes[0].pcount : 0;  // root covers all particles

  double t0 = wall_time();
#pragma omp target enter data map(to : nodes[0:nn], px[0:np], py[0:np], pz[0:np], pm[0:np], pidx[0:np])
  rep.h2d += wall_time() - t0;
  rep.h2d_bytes += static_cast<double>(nn) * sizeof(Node) + static_cast<double>(np) * (4 * sizeof(double) + 4);

  t0 = wall_time();
  int flag = 0;
  // Every array is already present (target enter data), so these maps only take references
  // and transfer nothing; only `flag` is copied.
#pragma omp target teams distribute parallel for map(tofrom : flag) \
    map(to : nodes[0:nn], px[0:np], py[0:np], pz[0:np], pm[0:np], pidx[0:np]) \
    map(to : tx[0:n], ty[0:n], tz[0:n], tself[0:n]) map(tofrom : acc[0:n])
  for (long long k = 0; k < n; ++k) {
#ifdef _OPENMP
    if (probe && k == 0) flag = omp_is_initial_device() ? 0 : 1;
#endif
    const TreeView v{nodes, nn, px, py, pz, pm, pidx};
    walk_tree(v, tx[k], ty[k], tz[k], skip_self ? tself[k] : kNoSelf, eps2, acc[k]);
  }
  rep.kernel += wall_time() - t0;
  if (probe) on_device = flag;

  t0 = wall_time();
#pragma omp target exit data map(release : nodes[0:nn], px[0:np], py[0:np], pz[0:np], pm[0:np], pidx[0:np])
  rep.d2h += wall_time() - t0;
}

}  // namespace

void offload_walks(const FlatTree& local, const std::vector<TreeView>& remote, double eps2,
                   std::vector<WalkResult>& acc, bool allow_host_fallback, OffloadReport& rep) {
  const long long n = static_cast<long long>(local.px.size());
  acc.assign(static_cast<std::size_t>(n), WalkResult{});
#ifdef _OPENMP
  rep.num_devices = omp_get_num_devices();
#endif
  if (n == 0) {
    rep.executed = rep.num_devices > 0 ? Executed::GPU : Executed::HostFallback;
    return;
  }
  const double* tx = local.px.data();
  const double* ty = local.py.data();
  const double* tz = local.pz.data();
  const std::uint32_t* tself = local.pidx.data();
  WalkResult* out = acc.data();

  double t0 = wall_time();
#pragma omp target enter data map(to : tx[0:n], ty[0:n], tz[0:n], tself[0:n], out[0:n])
  rep.h2d += wall_time() - t0;
  rep.h2d_bytes += static_cast<double>(n) * (3 * sizeof(double) + 4 + sizeof(WalkResult));

  int on_device = 0;
  walk_on_device(view_of(local), true, n, tx, ty, tz, tself, eps2, out, on_device, true, rep);
  for (const TreeView& t : remote) walk_on_device(t, false, n, tx, ty, tz, tself, eps2, out, on_device, false, rep);

  t0 = wall_time();
#pragma omp target exit data map(from : out[0:n]) map(release : tx[0:n], ty[0:n], tz[0:n], tself[0:n])
  rep.d2h += wall_time() - t0;

  rep.executed = on_device ? Executed::GPU : Executed::HostFallback;
  if (rep.executed != Executed::GPU && !allow_host_fallback)
    throw std::runtime_error("--device gpu: the target regions ran on the host (omp_get_num_devices() = " +
                             std::to_string(rep.num_devices) +
                             "); build with offload support or pass --allow-host-fallback 1 to test the code path");
}

}  // namespace nbody

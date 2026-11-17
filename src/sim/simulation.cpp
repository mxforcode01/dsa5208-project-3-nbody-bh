#include "nbody/sim/simulation.hpp"

#include "nbody/core/timer.hpp"
#include "nbody/sim/integrator.hpp"

namespace nbody {

RunResult Simulation::run(ParticleSet& p) const {
  RunResult r;
  const double h = params_.dt;
  const int T = params_.threads;
  const double t_start = wall_time();

  auto record = [&](int step) {
    ScopedTimer t(r.timings.diagnostics);
    Diagnostics d = compute_diagnostics(p);
    if (params_.reduce_diagnostics) params_.reduce_diagnostics(d);
    r.diag.push_back({step, step * h, d});
  };

  {
    ScopedTimer t(r.timings.force);
    solver_.compute_accelerations(p, r.stats);
  }
  record(0);

  for (int s = 1; s <= params_.steps; ++s) {
    {
      ScopedTimer t(r.timings.integrate);
      kick(p, 0.5 * h, T);
      drift(p, h, T);
    }
    {
      ScopedTimer t(r.timings.force);
      solver_.compute_accelerations(p, r.stats);
    }
    {
      ScopedTimer t(r.timings.integrate);
      kick(p, 0.5 * h, T);
    }
    if ((params_.diag_every > 0 && s % params_.diag_every == 0) || s == params_.steps) record(s);
  }
  r.total_time = wall_time() - t_start;
  return r;
}

}  // namespace nbody

#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "nbody/core/timer.hpp"

namespace nbody {

// OpenMP schedule of the force-evaluation loops (V4). chunk <= 0 means the
// OpenMP default for that kind (static: one near-equal block per thread;
// dynamic/guided: 1).
enum class ScheduleKind { Static, Dynamic, Guided };

struct LoopSchedule {
  ScheduleKind kind = ScheduleKind::Static;
  int chunk = 0;
};

inline ScheduleKind parse_schedule(const std::string& s) {
  if (s == "static") return ScheduleKind::Static;
  if (s == "dynamic") return ScheduleKind::Dynamic;
  if (s == "guided") return ScheduleKind::Guided;
  throw std::invalid_argument("unknown schedule '" + s + "' (expected static|dynamic|guided)");
}

inline std::string to_string(ScheduleKind k) {
  return k == ScheduleKind::Static ? "static" : (k == ScheduleKind::Dynamic ? "dynamic" : "guided");
}

// Per-thread balance of one parallel loop. busy = time from the start of the loop until a
// thread finished its own iterations (the loop has no barrier before that point);
// work = interactions done by the thread.
struct LoopBalance {
  int threads = 1;
  double busy_max = 0.0, busy_mean = 0.0;
  double work_max = 0.0, work_mean = 0.0, work_total = 0.0;
};

// Runs body(k) for k in [0, n) with `threads` OpenMP threads and the given schedule.
// body returns the number of interactions of iteration k. Iterations must be independent.
template <class Body>
LoopBalance parallel_force_loop(long long n, int threads, const LoopSchedule& sched, Body&& body) {
  LoopBalance lb;
#ifdef _OPENMP
  if (threads > 1) {
    const omp_sched_t kind = sched.kind == ScheduleKind::Static
                                 ? omp_sched_static
                                 : (sched.kind == ScheduleKind::Dynamic ? omp_sched_dynamic : omp_sched_guided);
    omp_set_schedule(kind, sched.chunk);
    std::vector<double> busy(threads, 0.0), work(threads, 0.0);
    int team = 1;
    const double t0 = wall_time();
#pragma omp parallel num_threads(threads)
    {
      const int tid = omp_get_thread_num();
#pragma omp single nowait
      team = omp_get_num_threads();
      double w = 0.0;
#pragma omp for schedule(runtime) nowait
      for (long long k = 0; k < n; ++k) w += body(k);
      busy[tid] = wall_time() - t0;
      work[tid] = w;
    }
    lb.threads = team;
    for (int t = 0; t < team; ++t) {
      lb.busy_max = std::max(lb.busy_max, busy[t]);
      lb.busy_mean += busy[t] / team;
      lb.work_max = std::max(lb.work_max, work[t]);
      lb.work_total += work[t];
    }
    lb.work_mean = lb.work_total / team;
    return lb;
  }
#else
  (void)threads;
  (void)sched;
#endif
  const double t0 = wall_time();
  double w = 0.0;
  for (long long k = 0; k < n; ++k) w += body(k);
  lb.busy_max = lb.busy_mean = wall_time() - t0;
  lb.work_max = lb.work_mean = lb.work_total = w;
  return lb;
}

}  // namespace nbody

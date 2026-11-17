#pragma once

#include <chrono>

namespace nbody {

// Monotonic wall-clock time in seconds.
inline double wall_time() {
  using clock = std::chrono::steady_clock;
  return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

// Adds the lifetime of the object to `accumulator` (seconds).
class ScopedTimer {
 public:
  explicit ScopedTimer(double& accumulator) : acc_(accumulator), t0_(wall_time()) {}
  ~ScopedTimer() { acc_ += wall_time() - t0_; }
  ScopedTimer(const ScopedTimer&) = delete;
  ScopedTimer& operator=(const ScopedTimer&) = delete;

 private:
  double& acc_;
  double t0_;
};

}  // namespace nbody

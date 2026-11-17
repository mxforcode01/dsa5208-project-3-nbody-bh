#pragma once
// Minimal self-contained test harness (no downloads needed on HPC nodes).
// Each test executable prints [ PASS ]/[ FAIL ] per test and exits non-zero on failure.

#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace nbtest {

struct TestCase {
  const char* name;
  std::function<void()> fn;
};

inline std::vector<TestCase>& registry() {
  static std::vector<TestCase> r;
  return r;
}

inline int& failure_count() {
  static int f = 0;
  return f;
}

struct Registrar {
  Registrar(const char* name, std::function<void()> fn) { registry().push_back({name, std::move(fn)}); }
};

inline void report_failure(const char* file, int line, const std::string& msg) {
  std::printf("    FAIL %s:%d: %s\n", file, line, msg.c_str());
  ++failure_count();
}

// quiet = true suppresses RUN/PASS/summary lines (used on non-root MPI ranks);
// failures are always printed.
inline int run_all(bool quiet = false) {
  int failed = 0;
  for (const TestCase& t : registry()) {
    const int before = failure_count();
    if (!quiet) std::printf("[ RUN  ] %s\n", t.name);
    try {
      t.fn();
    } catch (const std::exception& e) {
      report_failure(__FILE__, __LINE__, std::string("unexpected exception: ") + e.what());
    }
    const bool ok = failure_count() == before;
    if (!quiet || !ok) std::printf("[ %s ] %s\n", ok ? "PASS" : "FAIL", t.name);
    if (!ok) ++failed;
    std::fflush(stdout);
  }
  if (!quiet) std::printf("%zu tests, %d failed\n", registry().size(), failed);
  return failed == 0 ? 0 : 1;
}

}  // namespace nbtest

#define NB_CONCAT_INNER(a, b) a##b
#define NB_CONCAT(a, b) NB_CONCAT_INNER(a, b)

#define NB_TEST(name)                                                   \
  static void name();                                                   \
  static nbtest::Registrar NB_CONCAT(nb_registrar_, name)(#name, name); \
  static void name()

#define NB_CHECK(cond)                                                             \
  do {                                                                             \
    if (!(cond)) nbtest::report_failure(__FILE__, __LINE__, "check failed: " #cond); \
  } while (0)

#define NB_CHECK_NEAR(a, b, tol)                                                                  \
  do {                                                                                            \
    const double nb_a_ = (a), nb_b_ = (b), nb_t_ = (tol);                                         \
    if (!(std::fabs(nb_a_ - nb_b_) <= nb_t_)) {                                                   \
      std::ostringstream nb_os_;                                                                  \
      nb_os_.precision(17);                                                                       \
      nb_os_ << #a " = " << nb_a_ << ", " #b " = " << nb_b_ << ", |diff| = "                      \
             << std::fabs(nb_a_ - nb_b_) << " > tol " << nb_t_;                                   \
      nbtest::report_failure(__FILE__, __LINE__, nb_os_.str());                                   \
    }                                                                                             \
  } while (0)

#define NB_CHECK_LE(a, b)                                                                         \
  do {                                                                                            \
    const double nb_a_ = (a), nb_b_ = (b);                                                        \
    if (!(nb_a_ <= nb_b_)) {                                                                      \
      std::ostringstream nb_os_;                                                                  \
      nb_os_.precision(17);                                                                       \
      nb_os_ << #a " = " << nb_a_ << " > " #b " = " << nb_b_;                                     \
      nbtest::report_failure(__FILE__, __LINE__, nb_os_.str());                                   \
    }                                                                                             \
  } while (0)

#define NB_CHECK_THROWS(expr)                                                           \
  do {                                                                                  \
    bool nb_thrown_ = false;                                                            \
    try {                                                                               \
      expr;                                                                             \
    } catch (...) {                                                                     \
      nb_thrown_ = true;                                                                \
    }                                                                                   \
    if (!nb_thrown_) nbtest::report_failure(__FILE__, __LINE__, "expected exception: " #expr); \
  } while (0)

#define NB_TEST_MAIN() \
  int main() { return nbtest::run_all(); }

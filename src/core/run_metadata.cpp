#include "nbody/core/run_metadata.hpp"

#include <ctime>
#include <unistd.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef NBODY_GIT_HASH
#define NBODY_GIT_HASH "unknown"
#endif
#ifndef NBODY_BUILD_TYPE
#define NBODY_BUILD_TYPE "unknown"
#endif

namespace nbody {

RunMetadata collect_run_metadata() {
  RunMetadata md;

  char buf[64];
  const std::time_t now = std::time(nullptr);
  std::tm tm_utc{};
  gmtime_r(&now, &tm_utc);
  std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%SZ", &tm_utc);
  md.timestamp_utc = buf;

  char host[256] = {0};
  md.hostname = gethostname(host, sizeof(host) - 1) == 0 ? host : "unknown";

#if defined(__VERSION__)
  md.compiler = __VERSION__;
#else
  md.compiler = "unknown";
#endif
  md.build_type = NBODY_BUILD_TYPE;
  md.git_hash = NBODY_GIT_HASH;
#ifdef _OPENMP
  md.omp_max_threads = omp_get_max_threads();
#endif
  return md;
}

void add_metadata(const RunMetadata& md, CsvRow& row) {
  row.add("timestamp_utc", md.timestamp_utc);
  row.add("hostname", md.hostname);
  row.add("compiler", md.compiler);
  row.add("build_type", md.build_type);
  row.add("git_hash", md.git_hash);
  row.add("omp_max_threads", md.omp_max_threads);
}

}  // namespace nbody

#pragma once

#include <string>

#include "nbody/core/csv_logger.hpp"

namespace nbody {

struct RunMetadata {
  std::string timestamp_utc;
  std::string hostname;
  std::string compiler;
  std::string build_type;
  std::string git_hash;
  int omp_max_threads = 1;
};

RunMetadata collect_run_metadata();
void add_metadata(const RunMetadata& md, CsvRow& row);

}  // namespace nbody

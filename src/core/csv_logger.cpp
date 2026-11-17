#include "nbody/core/csv_logger.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace nbody {

namespace {

std::string escape(const std::string& s) {
  if (s.find_first_of(",\"\n") == std::string::npos) return s;
  std::string out = "\"";
  for (char c : s) {
    if (c == '"') out += '"';
    out += c;
  }
  return out + "\"";
}

std::string join(const std::vector<std::pair<std::string, std::string>>& f, bool keys) {
  std::string line;
  for (std::size_t i = 0; i < f.size(); ++i) {
    if (i) line += ',';
    line += escape(keys ? f[i].first : f[i].second);
  }
  return line;
}

}  // namespace

void CsvRow::add(const std::string& key, const std::string& value) { fields_.emplace_back(key, value); }

std::string CsvRow::header_line() const { return join(fields_, true); }
std::string CsvRow::value_line() const { return join(fields_, false); }

std::string CsvRow::format_double(double v) {
  char buf[40];
  std::snprintf(buf, sizeof(buf), "%.17g", v);
  return buf;
}

void append_csv_row(const std::string& path, const CsvRow& row) {
  std::string existing_header;
  {
    std::ifstream in(path);
    if (in) std::getline(in, existing_header);
  }
  const std::string header = row.header_line();
  if (!existing_header.empty() && existing_header != header)
    throw std::runtime_error("CSV schema mismatch in '" + path +
                             "': existing header differs; write to a new file");
  std::ofstream out(path, std::ios::app);
  if (!out) throw std::runtime_error("cannot open CSV file '" + path + "' for writing");
  if (existing_header.empty()) out << header << '\n';
  out << row.value_line() << '\n';
}

}  // namespace nbody

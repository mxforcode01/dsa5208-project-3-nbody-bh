#pragma once

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nbody {

// Ordered key/value row. Column order is the order of add() calls.
class CsvRow {
 public:
  void add(const std::string& key, const std::string& value);
  void add(const std::string& key, const char* value) { add(key, std::string(value)); }
  template <class T>
  void add(const std::string& key, T value) {
    static_assert(std::is_arithmetic<T>::value, "CsvRow::add expects a number or string");
    if constexpr (std::is_floating_point<T>::value) add(key, format_double(static_cast<double>(value)));
    else add(key, std::to_string(value));
  }

  std::string header_line() const;
  std::string value_line() const;
  const std::vector<std::pair<std::string, std::string>>& fields() const { return fields_; }

  static std::string format_double(double v);  // 17 significant digits (round-trip exact)

 private:
  std::vector<std::pair<std::string, std::string>> fields_;
};

// Appends `row` to `path`. A new or empty file gets the header first. An existing
// file whose header differs throws std::runtime_error, so files never mix schemas.
void append_csv_row(const std::string& path, const CsvRow& row);

}  // namespace nbody

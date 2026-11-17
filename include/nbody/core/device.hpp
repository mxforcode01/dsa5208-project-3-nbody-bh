#pragma once

#include <stdexcept>
#include <string>

// Functions between NBODY_DEVICE_BEGIN / NBODY_DEVICE_END are also compiled for the
// OpenMP offload device (V7). Without OpenMP they are ordinary host functions.
#if defined(_OPENMP)
#define NBODY_DEVICE_BEGIN _Pragma("omp declare target")
#define NBODY_DEVICE_END _Pragma("omp end declare target")
#else
#define NBODY_DEVICE_BEGIN
#define NBODY_DEVICE_END
#endif

namespace nbody {

// Where the Barnes-Hut tree walks run. Trees are always built on the CPU.
enum class Device { CPU, GPU };

inline Device parse_device(const std::string& s) {
  if (s == "cpu") return Device::CPU;
  if (s == "gpu") return Device::GPU;
  throw std::invalid_argument("unknown device '" + s + "' (expected cpu|gpu)");
}

inline std::string to_string(Device d) { return d == Device::CPU ? "cpu" : "gpu"; }

// What actually executed the walks (reported, never assumed).
enum class Executed { CPU = 0, GPU = 1, HostFallback = 2 };

inline std::string to_string(Executed e) {
  return e == Executed::CPU ? "cpu" : (e == Executed::GPU ? "gpu" : "host-fallback");
}

}  // namespace nbody

cmake_minimum_required(VERSION 3.16)
project(nbody_bh LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Fail early, with a readable message, on compilers that cannot build this C++17 code
# (cluster default compilers are often old: load a newer GCC module).
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7.0)
  message(FATAL_ERROR "GCC ${CMAKE_CXX_COMPILER_VERSION} is too old: need GCC >= 7 for C++17 "
                      "(e.g. `module load GCC/11` or a newer toolchain, then delete the build directory).")
endif()

# Clusters load a newer GCC through modules, but `mpirun` may start the processes through ssh,
# where those modules are not loaded: the programs would then pick up the system's older
# libstdc++/libgomp ("CXXABI_1.3.15 not found", seen on Vanda). Recording the compiler's own
# library directory in the executables (rpath) makes them find the right runtime libraries.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  execute_process(COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
                  OUTPUT_VARIABLE _nb_libstdcxx OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(IS_ABSOLUTE "${_nb_libstdcxx}")
    get_filename_component(_nb_libstdcxx "${_nb_libstdcxx}" REALPATH)
    get_filename_component(_nb_gcc_libdir "${_nb_libstdcxx}" DIRECTORY)
    if(NOT _nb_gcc_libdir MATCHES "^/(usr/)?lib")  # system compiler: nothing to do
      message(STATUS "Using rpath for the compiler runtime: ${_nb_gcc_libdir}")
      add_link_options("-Wl,-rpath,${_nb_gcc_libdir}")
    endif()
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(NBODY_ENABLE_OPENMP "Build with OpenMP threading" ON)
option(NBODY_ENABLE_MPI "Build the distributed (MPI) version" ON)
option(NBODY_BUILD_TESTS "Build the automated correctness tests" ON)
set(NBODY_MPIEXEC_FLAGS "" CACHE STRING "Extra mpiexec flags for MPI tests (e.g. --oversubscribe)")
# GPU offload (V7). Without it, `--device gpu` target regions still compile and run on the
# host; the program then reports "host-fallback" and refuses unless --allow-host-fallback 1.
option(NBODY_ENABLE_GPU_OFFLOAD "Compile OpenMP target regions for a GPU" OFF)
set(NBODY_OFFLOAD_FLAGS "-foffload=nvptx-none;-fcf-protection=none;-fno-stack-protector"
    CACHE STRING "Compile+link flags for OpenMP offload (GCC/nvptx default; nvc++: -mp=gpu;-gpu=cc80)")
if(NBODY_ENABLE_GPU_OFFLOAD)
  add_compile_options(${NBODY_OFFLOAD_FLAGS})
  add_link_options(${NBODY_OFFLOAD_FLAGS})
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # CPU-only build: a GCC configured with offload targets (e.g. Vanda's foss/2025b GCC 14.3)
  # otherwise emits offload tables for the `omp target` regions and the link fails with
  # "undefined reference to __offload_func_table". Target regions then run on the host.
  add_compile_options(-foffload=disable)
  add_link_options(-foffload=disable)
endif()

# Record the git revision at configure time (re-run cmake to refresh it).
set(NBODY_GIT_HASH "unknown")
find_package(Git QUIET)
if(GIT_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE _git_hash
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _git_result)
  if(_git_result EQUAL 0)
    set(NBODY_GIT_HASH ${_git_hash})
  endif()
endif()
# A bundle from scripts/hpc/make_bundle.sh has no .git folder; it records the commit in this file.
if(NBODY_GIT_HASH STREQUAL "unknown" AND EXISTS "${CMAKE_SOURCE_DIR}/SOURCE_VERSION")
  file(READ "${CMAKE_SOURCE_DIR}/SOURCE_VERSION" NBODY_GIT_HASH)
  string(STRIP "${NBODY_GIT_HASH}" NBODY_GIT_HASH)
endif()

add_library(nbody_core
  src/core/particles.cpp
  src/core/initial_conditions.cpp
  src/core/config.cpp
  src/core/csv_logger.cpp
  src/core/run_metadata.cpp
  src/core/snapshot.cpp
  src/solvers/direct_sum.cpp
  src/solvers/barnes_hut.cpp
  src/tree/octree.cpp
  src/tree/morton_build.cpp
  src/gpu/offload_walk.cpp
  src/sim/integrator.cpp
  src/sim/diagnostics.cpp
  src/sim/simulation.cpp
  src/sim/accuracy.cpp)
target_include_directories(nbody_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(nbody_core PUBLIC -Wall -Wextra -Wpedantic)
set_source_files_properties(src/core/run_metadata.cpp PROPERTIES COMPILE_DEFINITIONS
  "NBODY_GIT_HASH=\"${NBODY_GIT_HASH}\";NBODY_BUILD_TYPE=\"${CMAKE_BUILD_TYPE}\"")

if(NBODY_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
  target_link_libraries(nbody_core PUBLIC OpenMP::OpenMP_CXX)
else()
  target_compile_options(nbody_core PUBLIC -Wno-unknown-pragmas)
endif()

add_executable(nbody_serial apps/nbody_serial.cpp)
target_link_libraries(nbody_serial PRIVATE nbody_core)

add_executable(compare_snapshots apps/compare_snapshots.cpp)
target_link_libraries(compare_snapshots PRIVATE nbody_core)

if(NBODY_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  if(MPI_CXX_VERSION AND MPI_CXX_VERSION VERSION_LESS 3.0)
    message(FATAL_ERROR "MPI ${MPI_CXX_VERSION} found; shared-memory windows need MPI >= 3.0 "
                        "(load a newer OpenMPI / Intel MPI / MPICH module).")
  endif()
  add_library(nbody_mpi_lib
    src/mpi/mpi_env.cpp
    src/mpi/decomposition.cpp
    src/mpi/particle_comm.cpp
    src/mpi/let.cpp
    src/mpi/dist_bh.cpp
    src/mpi/shm_trees.cpp)
  target_link_libraries(nbody_mpi_lib PUBLIC nbody_core MPI::MPI_CXX)
  add_executable(nbody_mpi apps/nbody_mpi.cpp)
  target_link_libraries(nbody_mpi PRIVATE nbody_mpi_lib)
endif()

if(NBODY_BUILD_TESTS)
  enable_testing()
  foreach(t physics rng_ic direct_sum integrator io_config octree bh_accuracy gpu_offload)
    add_executable(test_${t} tests/test_${t}.cpp)
    target_link_libraries(test_${t} PRIVATE nbody_core)
    target_include_directories(test_${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    add_test(NAME ${t} COMMAND test_${t})
  endforeach()
  add_test(NAME cli_smoke COMMAND nbody_serial --n 64 --steps 2 --dist plummer)
  add_test(NAME cli_smoke_bh COMMAND nbody_serial --n 500 --steps 2 --dist plummer --solver bh --theta 0.5)

  if(NBODY_ENABLE_MPI)
    separate_arguments(_mpiexec_flags UNIX_COMMAND "${NBODY_MPIEXEC_FLAGS}")
    add_executable(test_mpi_dist tests/test_mpi_dist.cpp)
    target_link_libraries(test_mpi_dist PRIVATE nbody_mpi_lib)
    target_include_directories(test_mpi_dist PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    foreach(np 1 2 3 4)
      add_test(NAME mpi_dist_np${np}
               COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} ${np} ${_mpiexec_flags}
                       ${MPIEXEC_PREFLAGS} $<TARGET_FILE:test_mpi_dist> ${MPIEXEC_POSTFLAGS})
      set_tests_properties(mpi_dist_np${np} PROPERTIES ENVIRONMENT "OMP_NUM_THREADS=2" PROCESSORS ${np})
    endforeach()
    add_test(NAME cli_smoke_mpi
             COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${_mpiexec_flags} $<TARGET_FILE:nbody_mpi>
                     --n 2000 --steps 3 --dist plummer --threads 2)
  endif()
endif()

#!/bin/bash -l
# GPU experiment (E6): CPU against GPU tree walks at several N, on 1 rank + 1 GPU and 2 ranks + 2 GPUs.
#
#   NB_GPU_MODULES="<compiler + MPI + CMake modules>" ./scripts/hpc/run_gpu.sh     submit from the login node
#   ./scripts/hpc/run_gpu.sh --local                                               run here, small sizes
#
# NB_GPU_MODULES       modules with an OpenMP offload compiler, its MPI, and CMake (required for a job)
# NB_GPU_CMAKE_ARGS    default: -DCMAKE_CXX_COMPILER=nvc++ -DNBODY_OFFLOAD_FLAGS=-mp=gpu
# GPU_ALLOW_FALLBACK=1 accept the host (only to test this script on a machine without a GPU)
# GPU_SIZES, GPU_THREADS, GPU_REPEATS, GPU_WALL, QUEUE, PBS_EXTRA can also be set.
# The job builds in build-gpu/, runs test_gpu_offload, and writes results/hpc/<date>-<host>-gpu/gpu.csv.
set -u

if [ -z "${NB_GPU_JOB:-}" ]; then
  NB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
  : "${NB_GPU_MODULES:=}" "${NB_GPU_CMAKE_ARGS:=-DCMAKE_CXX_COMPILER=nvc++ -DNBODY_OFFLOAD_FLAGS=-mp=gpu}"
  : "${GPU_ALLOW_FALLBACK:=0}" "${GPU_WALL:=02:00:00}" "${QUEUE:=}" "${PBS_EXTRA:=}"
  RUN_DIR="$NB_ROOT/results/hpc/$(date '+%Y%m%d-%H%M%S')-$(hostname -s 2>/dev/null || echo host)-gpu"
  if [ "${1:-}" = --local ]; then
    : "${GPU_SIZES:=20000 40000}" "${GPU_THREADS:=2}" "${GPU_REPEATS:=1}"
    NB_LOCAL=1
  else
    : "${GPU_SIZES:=500000 1000000 2000000 4000000 8000000}" "${GPU_THREADS:=36}" "${GPU_REPEATS:=3}"
    NB_LOCAL=0
    [ -n "$NB_GPU_MODULES" ] || { echo "ERROR: set NB_GPU_MODULES (compiler with GPU offload, its MPI, CMake)" >&2; exit 1; }
  fi
  mkdir -p "$RUN_DIR/pbs" "$RUN_DIR/logs"
  for v in NB_ROOT RUN_DIR NB_GPU_MODULES NB_GPU_CMAKE_ARGS GPU_ALLOW_FALLBACK GPU_SIZES GPU_THREADS GPU_REPEATS NB_LOCAL; do
    printf 'export %s=%q\n' "$v" "${!v}"
  done >"$RUN_DIR/gpu_env.sh"
  if [ "$NB_LOCAL" = 0 ]; then
    cmd=(qsub -N nb_gpu -o "$RUN_DIR/pbs/" -j oe -l "select=2:ngpus=1:ncpus=${GPU_THREADS}:mpiprocs=${GPU_THREADS}:mem=250gb"
         -l place=pack -l "walltime=$GPU_WALL" -v "NB_GPU_JOB=$RUN_DIR/gpu_env.sh")
    [ -n "$QUEUE" ] && cmd+=(-q "$QUEUE")
    # shellcheck disable=SC2206
    [ -n "$PBS_EXTRA" ] && cmd+=($PBS_EXTRA)
    "${cmd[@]}" "$NB_ROOT/scripts/hpc/run_gpu.sh" || exit 1
    echo "results will be in $RUN_DIR"
    exit 0
  fi
  NB_GPU_JOB="$RUN_DIR/gpu_env.sh"
fi

# --- from here on: inside the job (or local) ---------------------------------------------
# shellcheck disable=SC1090
. "$NB_GPU_JOB"
# shellcheck disable=SC1091
. "$NB_ROOT/scripts/hpc/lib.sh"
# shellcheck disable=SC1091
. "$NB_ROOT/scripts/hpc/experiments.sh"
NB_MODULES="$NB_GPU_MODULES"
nb_load_modules
cd "$NB_ROOT" || exit 1
BUILD_DIR="$NB_ROOT/build-gpu"
{
  nb_print_toolchain
  nvidia-smi -L 2>/dev/null || echo "nvidia-smi: not available"
  echo "hosts: $( [ -n "${PBS_NODEFILE:-}" ] && sort -u "$PBS_NODEFILE" | tr '\n' ' ' || hostname)"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<not set>}"
} >"$RUN_DIR/env_gpu.txt" 2>&1

# shellcheck disable=SC2086
cmake -S "$NB_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DNBODY_ENABLE_GPU_OFFLOAD=ON $NB_GPU_CMAKE_ARGS >"$RUN_DIR/build.log" 2>&1 &&
  cmake --build "$BUILD_DIR" -j 8 >>"$RUN_DIR/build.log" 2>&1 ||
  { tail -n 30 "$RUN_DIR/build.log" >&2; nb_die "build failed: $RUN_DIR/build.log"; }

fallback=()
if [ "$GPU_ALLOW_FALLBACK" = 1 ]; then fallback=(--allow-host-fallback 1); else export OMP_TARGET_OFFLOAD=MANDATORY; fi
"$BUILD_DIR/test_gpu_offload" >"$RUN_DIR/logs/test_gpu_offload.log" 2>&1 ||
  nb_die "test_gpu_offload failed: $RUN_DIR/logs/test_gpu_offload.log"

# launch RANKS command...: the ranks share the node's sockets, each with its own GPU.
launch() {
  local ranks=$1 host=() map=(); shift
  [ -n "${PBS_NODEFILE:-}" ] && host=(--hostfile "$PBS_NODEFILE")
  map=(--map-by "ppr:${ranks}:node:PE=${GPU_THREADS}" --bind-to core)
  [ "$NB_LOCAL" = 1 ] && map=(--bind-to none --oversubscribe)
  "$(command -v mpirun)" -np "$ranks" "${host[@]}" "${map[@]}" -x OMP_NUM_THREADS -x LD_LIBRARY_PATH "$@"
}
export OMP_NUM_THREADS=$GPU_THREADS
args=(--dist plummer --theta 0.5 --eps 0.01 --dt 0.001 --steps 2 --tree-build morton --sched dynamic --chunk 1024 --threads "$GPU_THREADS")
for (( r = 1; r <= GPU_REPEATS; r++ )); do
  for n in $GPU_SIZES; do
    for ranks in 1 2; do
      for dev in cpu gpu; do
        extra=()
        [ "$dev" = gpu ] && extra=("${fallback[@]}")
        nb_run "e6_${dev}_N${n}_p${ranks}_r$r" launch "$ranks" "$BUILD_DIR/nbody_mpi" --n "$n" "${args[@]}" \
          --device "$dev" "${extra[@]}" --csv "$RUN_DIR/gpu.csv"
      done
    done
  done
done

# The same forces on both devices (compared on one rank, smallest size).
n=${GPU_SIZES%% *}
single=(--solver bh --n "$n" --steps 0 --dist plummer --theta 0.5 --eps 0.01 --tree-build morton --sched dynamic --chunk 1024 --threads "$GPU_THREADS")
nb_run e6_snapshot_cpu "$BUILD_DIR/nbody_serial" "${single[@]}" --device cpu --snapshot "$RUN_DIR/cpu.bin"
nb_run e6_snapshot_gpu "$BUILD_DIR/nbody_serial" "${single[@]}" --device gpu "${fallback[@]}" --snapshot "$RUN_DIR/gpu.bin"
nb_run e6_compare "$BUILD_DIR/compare_snapshots" "$RUN_DIR/gpu.bin" "$RUN_DIR/cpu.bin" --csv "$RUN_DIR/gpu_accuracy.csv" --label "size=$n"
rm -f "$RUN_DIR/cpu.bin" "$RUN_DIR/gpu.bin"
nb_log "done. results: $RUN_DIR"
if [ -s "$RUN_DIR/failures.txt" ]; then
  nb_log "some runs failed:"
  cat "$RUN_DIR/failures.txt" >&2
  exit 1
fi

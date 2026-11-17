#!/bin/bash
# One command for everything: build, test, and run the experiments.
#
#   ./scripts/hpc/run_all.sh              on a PBS cluster login node (e.g. Vanda): build here,
#                                         submit the test job, then the experiment jobs, which
#                                         start only if the tests pass
#   ./scripts/hpc/run_all.sh --dry-run    build and print the qsub commands without submitting
#   ./scripts/hpc/run_all.sh --local      no PBS: build, test and run every experiment at small
#                                         sizes on this machine (checks the pipeline end to end)
# Options:
#   --only e1,e7b      run only these experiments (default: all in config.sh)
#   --tests-only       build and run the tests, no experiments
#   --no-build         reuse the existing build-hpc/ directory
# Settings: scripts/hpc/config.sh (any value can be overridden from the environment).
# Results:  results/hpc/<date>-<host>/  (CSV files, logs, failures.txt, jobs.txt)
set -u

NB_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODE=pbs; DRY=0; ONLY=""; TESTS_ONLY=0; BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --local) MODE=local ;;
    --dry-run) DRY=1 ;;
    --only) ONLY=${2:?--only needs a list}; shift ;;
    --tests-only) TESTS_ONLY=1 ;;
    --no-build) BUILD=0 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "unknown option $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ "$MODE" = local ]; then
  # Small sizes so the whole pipeline finishes in minutes on a laptop. Values set in the
  # environment still win.
  : "${E1_DIRECT_N:=500 1000 2000}" "${E1_BH_N:=1000 4000 16000}" "${E2_N:=5000}" "${E2_THETAS:=0.3 0.5 0.8}"
  : "${E3_N:=50000}" "${E4_N:=50000}" "${N_STRONG:=20000 40000}" "${N_LARGE:=40000}" "${N_PER_CORE:=5000}" "${E5B_N:=20000}"
  : "${E5A_CORES:=2 4}" "${E5B_CORES:=2 4}" "${REPEATS:=1}" "${BASE_N:=5000 40000}" "${STEPS:=2}" "${THREADS_PER_RANK:=2}"
  : "${E8_N:=2000 5000}" "${E9_N:=5000}" "${E9_T:=0.05}" "${E10_N:=40000}" "${E11_N:=40000}" "${E11_CORES:=4 8}"
  export NB_LOCAL=1
fi

# shellcheck disable=SC1091
. "$NB_ROOT/scripts/hpc/config.sh"
# shellcheck disable=SC1091
. "$NB_ROOT/scripts/hpc/lib.sh"
# shellcheck disable=SC1091
. "$NB_ROOT/scripts/hpc/experiments.sh"
[ -n "$ONLY" ] && EXPERIMENTS=$(echo "$ONLY" | tr ',' ' ')

case "$NB_ROOT" in *" "*) [ "$MODE" = pbs ] && nb_die "the repository path contains a space ('$NB_ROOT'); PBS cannot pass it — clone into a path without spaces" ;; esac

# --- toolchain ---------------------------------------------------------------------------
nb_load_modules
nb_log "toolchain:"; nb_print_toolchain >&2

# --- build (login node / this machine) ---------------------------------------------------
BUILD_DIR="$NB_ROOT/build-hpc"
if [ "$BUILD" = 1 ]; then
  # A build directory configured elsewhere (another machine or path) cannot be reused.
  if [ -f "$BUILD_DIR/CMakeCache.txt" ] && ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$NB_ROOT\$" "$BUILD_DIR/CMakeCache.txt"; then
    nb_log "removing build-hpc/ (configured for a different source path)"; rm -rf "$BUILD_DIR"
  fi
  jobs=$(nproc 2>/dev/null || echo 4); [ "$jobs" -gt 8 ] && jobs=8
  nb_log "configuring and building in build-hpc/ (-j $jobs)"
  cmake -S "$NB_ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release >"$NB_ROOT/build-hpc.log" 2>&1 &&
    cmake --build "$BUILD_DIR" -j "$jobs" >>"$NB_ROOT/build-hpc.log" 2>&1 ||
    { tail -n 30 "$NB_ROOT/build-hpc.log" >&2; nb_die "build failed — full log: build-hpc.log"; }
fi
[ -x "$BUILD_DIR/nbody_mpi" ] || nb_die "build-hpc/nbody_mpi is missing (run without --no-build)"

# --- run directory -------------------------------------------------------------------------
RUN_DIR="$NB_ROOT/results/hpc/$(date '+%Y%m%d-%H%M%S')-$(hostname -s 2>/dev/null || echo host)"
[ "$MODE" = local ] && RUN_DIR="$RUN_DIR-local"
mkdir -p "$RUN_DIR/pbs" "$RUN_DIR/logs"

if [ "$MODE" = local ]; then
  : "${CORES_PER_NODE:=$(n=$(nproc 2>/dev/null || echo 4); [ "$n" -gt 8 ] && n=8; echo "$n")}"
  MAX_CORES=$CORES_PER_NODE
elif [ -z "$CORES_PER_NODE" ]; then
  CORES_PER_NODE=$(pbsnodes -a 2>/dev/null | awk -F' = ' '/resources_available.ncpus/{print $2}' |
                   sort | uniq -c | sort -k1,1nr | awk 'NR==1{print $2}')
  [ -n "$CORES_PER_NODE" ] || nb_die "could not read cores per node from pbsnodes; set CORES_PER_NODE=<n> and rerun"
fi
nb_log "cores per node: $CORES_PER_NODE | max cores: $MAX_CORES | threads per rank: $THREADS_PER_RANK | run dir: $RUN_DIR"

# Everything a job needs, frozen at submission time (modules included).
{
  echo "# written by run_all.sh $(date '+%F %T')"
  for v in NB_ROOT BUILD_DIR RUN_DIR CORES_PER_NODE NB_MPIRUN_EXTRA MAX_CORES THREADS_PER_RANK REPEATS STEPS DIST \
           THETA EPS DT SCHED CHUNK TREE_BUILD E1_DIRECT_N E1_BH_N E2_N E2_THETAS E3_N E4_N E4_SCHEDS N_STRONG \
           N_LARGE SINGLE_RUN_N N_PER_CORE E5A_CORES E5B_CORES E5B_N REPEATS_LARGE BASE_N E7A_BIND \
           E8_N E8_CORES E9_N E9_T E9_DTS E9_DIAG E9_CORES E10_N E10_STEPS E10_CORES E11_N E11_CORES E11_BOXES; do
    printf 'export %s=%q\n' "$v" "${!v}"
  done
  printf 'export NB_MODULES=%q\n' "${NB_LOADED_MODULES:-${NB_MODULES:-}}"
  printf 'export NB_LOCAL=%q\n' "${NB_LOCAL:-0}"
} >"$RUN_DIR/run_env.sh"
export NB_ROOT BUILD_DIR RUN_DIR CORES_PER_NODE

# --- local mode: run everything here --------------------------------------------------------
if [ "$MODE" = local ]; then
  nb_log "running tests"
  if ! nb_run_experiment tests 1 >/dev/null; then nb_die "tests failed — see $RUN_DIR/tests.log"; fi
  nb_log "tests passed"
  [ "$TESTS_ONLY" = 1 ] && exit 0
  for e in $EXPERIMENTS; do
    for n in $("nodes_$e"); do
      [ "$n" -eq 1 ] || { nb_log "skip $e on $n nodes (local mode has one node)"; continue; }
      nb_run_experiment "$e" 1
    done
  done
  nb_log "done. results: $RUN_DIR"
  [ -s "$RUN_DIR/failures.txt" ] && { nb_log "some runs failed:"; cat "$RUN_DIR/failures.txt" >&2; exit 1; }
  exit 0
fi

# --- PBS mode: submit the test job, then the experiments (afterok) -------------------------
command -v qsub >/dev/null 2>&1 || nb_die "qsub not found: run on a PBS login node, or use --local"
wall_for() { case "$1" in e1) echo "$WALL_E1" ;; e2) echo "$WALL_E2" ;; e3) echo "$WALL_E3" ;; e4) echo "$WALL_E4" ;;
                          e5a|e5b) echo "$WALL_E5" ;;
                          e7b) if [ "$2" -eq 1 ]; then echo "$WALL_E7B_1NODE"; else echo "$WALL_E7"; fi ;;
                          e7bm) if [ "$2" -eq 1 ]; then echo "$WALL_E7BM_1NODE"; else echo "$WALL_E7"; fi ;;
                          ebase) echo "$WALL_EBASE" ;;
                          *) echo "$WALL_E7" ;; esac; }
submit() {  # submit NAME NODES NCPUS WALLTIME DEPENDENCY EXP  -> prints the job id
  local name=$1 nodes=$2 ncpus=$3 wall=$4 dep=$5 exp=$6
  local mem=""  # full-node jobs only; asking for every core of a node also keeps other jobs off it
  [ -n "$MEM_PER_NODE" ] && [ "$ncpus" -eq "$CORES_PER_NODE" ] && mem=":mem=$MEM_PER_NODE"
  local cmd=(qsub -N "$name" -o "$RUN_DIR/pbs/" -j oe -l "select=${nodes}:ncpus=${ncpus}:mpiprocs=${ncpus}${mem}"
             -l "walltime=$wall" -v "NB_RUN_DIR=$RUN_DIR,NB_ROOT=$NB_ROOT,EXP=$exp,NODES=$nodes")
  [ -n "$QUEUE" ] && cmd+=(-q "$QUEUE")
  [ -n "$dep" ] && cmd+=(-W "depend=afterok:$dep")
  # shellcheck disable=SC2206
  [ -n "$PBS_EXTRA" ] && cmd+=($PBS_EXTRA)
  cmd+=("$NB_ROOT/scripts/hpc/job.pbs")
  if [ "$DRY" = 1 ]; then echo "DRY:${name}"; printf '  %q' "${cmd[@]}" >&2; echo >&2; return 0; fi
  local id; id=$("${cmd[@]}") || nb_die "qsub failed for $name"
  echo "$id"
}

tcpus=8; [ "$CORES_PER_NODE" -lt 8 ] && tcpus=$CORES_PER_NODE
TEST_JOB=$(submit nb_tests 1 "$tcpus" "$WALL_TESTS" "" tests)
echo "tests $TEST_JOB" >"$RUN_DIR/jobs.txt"
nb_log "submitted tests: $TEST_JOB"
if [ "$TESTS_ONLY" = 0 ]; then
  for e in $EXPERIMENTS; do
    for n in $("nodes_$e"); do
      id=$(submit "nb_${e}_n$n" "$n" "$CORES_PER_NODE" "$(wall_for "$e" "$n")" "$([ "$DRY" = 1 ] || echo "$TEST_JOB")" "$e")
      echo "$e n=$n $id" >>"$RUN_DIR/jobs.txt"
      nb_log "submitted $e on $n node(s): $id"
    done
  done
fi
nb_log "all jobs submitted (list: $RUN_DIR/jobs.txt). Watch with: qstat -u \$USER"
nb_log "when they have finished: ./scripts/hpc/collect.sh $RUN_DIR"

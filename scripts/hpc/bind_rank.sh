#!/bin/bash
# Pins one MPI rank and its OpenMP threads to its own block of physical cores, then runs the
# program:   mpirun --bind-to none ... bind_rank.sh ./nbody_mpi ...
#
# Used where Open MPI's own binding refuses to start (exit 254): Open MPI 5 will not bind a
# rank to cores in two sockets, so 1x72, 3x24 and 9x8 on a 2 x 36-core node cannot run with
# --map-by ppr:R:node:PE=T --bind-to core. This script makes the same placement Open MPI makes
# for the layouts it accepts -- local rank k gets physical cores k*T .. k*T+T-1, counted
# socket by socket -- but lets a block cross the socket boundary.
#
# Threads: OMP_PLACES lists the T cores (one place per core) and OMP_PROC_BIND=close puts
# thread i on core i. The whole process is also restricted to those cores with taskset.
# OMP_DISPLAY_AFFINITY prints every thread's binding once, so the log shows where it ran.
#
# Testing: NB_TOPO_FILE=<file with `lscpu -p=CPU,CORE,SOCKET` output> and
# NB_ALLOWED_CPUS=<list> replace the real topology; NB_BIND_PRINT_ONLY=1 prints the plan only.
set -u

T=${OMP_NUM_THREADS:-1}
LR=${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-${PMI_LOCAL_RANK:-0}}}

topo() {
  if [ -n "${NB_TOPO_FILE:-}" ]; then cat "$NB_TOPO_FILE"; else lscpu -p=CPU,CORE,SOCKET; fi
}
allowed=${NB_ALLOWED_CPUS:-$(awk '/^Cpus_allowed_list/{print $2}' /proc/self/status)}

# First allowed logical CPU of every physical core, ordered by socket, then core.
cores=$(topo | grep -v '^#' | awk -F, -v allowed="$allowed" '
  BEGIN {
    n = split(allowed, parts, ",")
    for (i = 1; i <= n; i++) {
      if (split(parts[i], r, "-") == 2) { for (c = r[1]; c <= r[2]; c++) ok[c] = 1 }
      else ok[parts[i]] = 1
    }
  }
  ok[$1] { print $3 "," $2 "," $1 }' | sort -t, -k1,1n -k2,2n -k3,3n | awk -F, '!seen[$1 "," $2]++ { print $3 }')

mapfile -t list <<<"$cores"
have=${#list[@]}
first=$(( LR * T ))
if [ "$have" -lt $(( first + T )) ]; then
  echo "bind_rank.sh: local rank $LR needs cores $first..$(( first + T - 1 )) but only $have physical cores are" \
       "available to this job (allowed CPUs: $allowed)" >&2
  exit 3
fi
mine=("${list[@]:$first:$T}")
cpus=$(IFS=,; echo "${mine[*]}")
places=$(printf '{%s},' "${mine[@]}"); places=${places%,}

export OMP_PLACES="$places" OMP_PROC_BIND=close
export OMP_DISPLAY_AFFINITY=TRUE OMP_AFFINITY_FORMAT="affinity host=%H rank-pid=%P thread=%n cpus=%A"
echo "bind_rank.sh: host $(hostname -s 2>/dev/null) local rank $LR threads $T -> logical CPUs $cpus" >&2
[ "${NB_BIND_PRINT_ONLY:-0}" = 1 ] && exit 0
if command -v taskset >/dev/null 2>&1; then exec taskset -c "$cpus" "$@"; fi
exec "$@"

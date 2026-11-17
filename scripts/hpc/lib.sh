# Shared helpers for the HPC scripts (sourced, not executed). Written for bash >= 4.2
# (CentOS 7 era) with no machine-specific paths: everything is detected or configured.

nb_log() { printf '[%s] %s\n' "$(date '+%H:%M:%S')" "$*" >&2; }
nb_die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# True if version $1 >= version $2.
nb_version_ge() { [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]; }

# Make the `module` command available in non-interactive shells (PBS jobs, scripts).
nb_init_modules() {
  type module >/dev/null 2>&1 && return 0
  local f
  for f in "${MODULESHOME:-}/init/bash" /etc/profile.d/lmod.sh /etc/profile.d/modules.sh \
           /usr/share/lmod/lmod/init/bash /usr/share/Modules/init/bash; do
    # shellcheck disable=SC1090
    [ -n "$f" ] && [ -f "$f" ] && . "$f" && type module >/dev/null 2>&1 && return 0
  done
  return 1
}

# Toolchain good enough to build and run? (GCC >= 7 or a non-GNU C++17 compiler,
# CMake >= 3.16, an MPI C++ wrapper and launcher.)
nb_toolchain_ok() {
  command -v mpicxx >/dev/null 2>&1 || return 1
  command -v cmake >/dev/null 2>&1 || return 1
  command -v mpirun >/dev/null 2>&1 || command -v mpiexec >/dev/null 2>&1 || return 1
  local cm; cm=$(cmake --version 2>/dev/null | head -n1 | awk '{print $3}')
  nb_version_ge "${cm:-0}" 3.16 || return 1
  if command -v g++ >/dev/null 2>&1; then
    local gv; gv=$(g++ -dumpfullversion -dumpversion 2>/dev/null | head -n1)
    nb_version_ge "${gv:-0}" 7 || return 1
  fi
  return 0
}

# Newest available module whose name matches the extended regex $1 (case-insensitive).
nb_newest_module() {
  module -t avail 2>&1 | sed -e 's/(default)//' -e 's/[[:space:]]*$//' | grep -iE "$1" | grep -v ':$' |
    sort -V | tail -n1
}

# Load compilers/MPI/CMake. Sets NB_LOADED_MODULES (what was loaded, for the jobs).
nb_load_modules() {
  # Some NUS clusters need their EasyBuild environment before `module avail` shows anything.
  if [ -f /app1/ebenv ]; then
    # shellcheck disable=SC1091
    . /app1/ebenv >/dev/null 2>&1 || true
  fi
  NB_LOADED_MODULES=""
  if [ -n "${NB_MODULES:-}" ]; then
    nb_init_modules || nb_die "NB_MODULES is set but the 'module' command is not available"
    # shellcheck disable=SC2086
    module load $NB_MODULES || nb_die "module load $NB_MODULES failed"
    NB_LOADED_MODULES="$NB_MODULES"
  elif ! nb_toolchain_ok; then
    nb_init_modules || nb_die "no usable g++/mpicxx/cmake on PATH and no 'module' command; load them yourself"
    local m
    m=$(nb_newest_module '^foss/')
    if [ -z "$m" ]; then m=$(nb_newest_module '^(openmpi|open-mpi|mpich|impi|intel-mpi)/'); fi
    [ -n "$m" ] && { module load "$m" && NB_LOADED_MODULES="$m"; }
    local cm; cm=$(cmake --version 2>/dev/null | head -n1 | awk '{print $3}')
    if ! nb_version_ge "${cm:-0}" 3.16; then
      m=$(nb_newest_module '^cmake/')
      [ -n "$m" ] && { module load "$m" && NB_LOADED_MODULES="$NB_LOADED_MODULES $m"; }
    fi
  fi
  if ! nb_toolchain_ok; then
    {
      echo "Could not find a usable toolchain (need GCC >= 7, CMake >= 3.16, an MPI-3 library)."
      echo "Loaded so far: '${NB_LOADED_MODULES}'. Candidates on this system:"
      module -t avail 2>&1 | grep -iE '^(foss|gcc|openmpi|mpich|impi|intel|cmake)/' | sort -V | sed 's/^/    /' | tail -n 40
      echo "Pick a matching set and rerun with, e.g.:  NB_MODULES=\"GCC/11.3.0 OpenMPI/4.1.4-GCC-11.3.0 CMake/3.24.3\""
    } >&2
    exit 1
  fi
  NB_MPI_VENDOR=$(nb_detect_mpi)
  [ "$NB_MPI_VENDOR" = unknown ] &&
    nb_log "WARNING: could not identify the MPI library; ranks will NOT be bound to cores (set NB_MPIRUN_EXTRA)"
  export NB_LOADED_MODULES NB_MPI_VENDOR
}

# Which MPI is on PATH. `mpirun --version` alone is unreliable (some Open MPI packages print
# a "help file missing" message instead), so tool names and the compiler wrapper decide first.
nb_detect_mpi() {
  if command -v ompi_info >/dev/null 2>&1 || mpicxx --showme:version >/dev/null 2>&1; then echo openmpi; return; fi
  local v; v=$( { mpirun --version; mpiexec --version; mpicxx -v; } 2>&1)
  case "$v" in
    *"Intel(R) MPI"*|*"Intel MPI"*) echo intel ;;
    *"Open MPI"*|*"OpenRTE"*|*"open-mpi"*) echo openmpi ;;
    *"HYDRA"*|*"MPICH"*|*"mpich"*) echo mpich ;;
    *) echo unknown ;;
  esac
}

nb_print_toolchain() {
  echo "modules loaded : ${NB_LOADED_MODULES:-<none, already on PATH>}"
  echo "c++            : $(command -v g++ 2>/dev/null) $(g++ -dumpfullversion -dumpversion 2>/dev/null)"
  echo "mpicxx         : $(command -v mpicxx)"
  local mv; mv=$( (ompi_info --version 2>/dev/null || mpirun --version 2>/dev/null) | grep -iE 'mpi|version' | head -n1)
  echo "mpi            : ${NB_MPI_VENDOR} | ${mv:-<version not printed>}"
  echo "cmake          : $(cmake --version | head -n1)"
}

# Process layout for p cores with up to t threads per rank, on nodes of c cores:
# prints "ranks threads nodes ranks_per_node". Ranks are spread evenly over the nodes.
nb_layout() {
  local p=$1 t=$2 c=$3 r n
  if [ "$p" -lt "$t" ]; then t=$p; fi
  r=$(( p / t ))
  n=$(( (r * t + c - 1) / c ))
  [ "$n" -lt 1 ] && n=1
  while [ $(( r % n )) -ne 0 ]; do n=$(( n + 1 )); done
  echo "$r $t $n $(( r / n ))"
}

# Run an MPI program: nb_mpirun RANKS THREADS NODES -- command args...
# Binds each rank to THREADS cores (the default Open MPI binding pins all threads of a rank
# to one core: 1.7x slower in our tests), spread evenly over NODES nodes.
# NB_BIND=wrap (Open MPI only): Open MPI places ranks but does not bind them; bind_rank.sh
# pins each rank to its own block of THREADS physical cores, which may cross sockets
# (Open MPI refuses that: 1x72, 3x24, 9x8 on 2 x 36 cores). Default NB_BIND=mpi.
nb_mpirun() {
  local ranks=$1 threads=$2 nodes=$3; shift 4
  local rpn=$(( (ranks + nodes - 1) / nodes ))
  export OMP_NUM_THREADS=$threads
  local extra=${NB_MPIRUN_EXTRA:-}
  case "${NB_MPI_VENDOR:-unknown}" in
    openmpi)
      local host=()
      [ -n "${PBS_NODEFILE:-}" ] && host=(--hostfile "$PBS_NODEFILE")
      if [ "${NB_BIND:-mpi}" = wrap ]; then
        extra="--map-by ppr:${rpn}:node --bind-to none"
        [ "${NB_LOCAL:-0}" = 1 ] && extra="--bind-to none --oversubscribe"
        set -- "$NB_ROOT/scripts/hpc/bind_rank.sh" "$@"
      else
        [ -z "$extra" ] && extra="--map-by ppr:${rpn}:node:PE=${threads} --bind-to core"
        [ "${NB_LOCAL:-0}" = 1 ] && extra="--bind-to none --oversubscribe"
      fi
      # Full path on purpose: Open MPI then finds its own remote daemon on nodes it reaches
      # through ssh, where the modules are not loaded (a bare `mpirun` fails with exit 127).
      # shellcheck disable=SC2086
      "$(command -v mpirun)" -np "$ranks" "${host[@]}" $extra -x OMP_NUM_THREADS "$@" ;;
    intel)
      # shellcheck disable=SC2086
      mpirun -n "$ranks" -ppn "$rpn" -genv OMP_NUM_THREADS "$threads" -genv I_MPI_PIN_DOMAIN omp $extra "$@" ;;
    mpich)
      [ -z "$extra" ] && extra="-bind-to core:${threads}"
      # shellcheck disable=SC2086
      mpiexec -n "$ranks" -ppn "$rpn" -genv OMP_NUM_THREADS "$threads" $extra "$@" ;;
    *)
      # shellcheck disable=SC2086
      mpirun -np "$ranks" $extra "$@" ;;
  esac
}

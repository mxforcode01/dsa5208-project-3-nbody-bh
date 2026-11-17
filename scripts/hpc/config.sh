# Settings for scripts/hpc/run_all.sh. Every value can be overridden from the environment,
# e.g.   QUEUE=parallel24 MAX_CORES=128 ./scripts/hpc/run_all.sh
# Only edit this file if a default does not fit your cluster.

# --- Software ------------------------------------------------------------------------
# Modules to load, space separated (e.g. "foss/2022b CMake/3.24.3-GCCcore-12.2.0").
# Empty = auto-detect: an EasyBuild "foss" toolchain, else the newest OpenMPI (+ its GCC),
# plus a CMake module if the system cmake is older than 3.16.
: "${NB_MODULES:=}"
# Mapping/binding flags added to every mpirun (empty = chosen for the detected MPI).
: "${NB_MPIRUN_EXTRA:=}"

# --- PBS -----------------------------------------------------------------------------
: "${QUEUE:=}"            # empty = the cluster's default queue
: "${PBS_EXTRA:=}"        # extra qsub arguments, e.g. "-P <project>"
: "${CORES_PER_NODE:=}"   # empty = most common resources_available.ncpus from pbsnodes

# Memory per full node, added to full-node jobs (needed on Vanda: 500gb). Empty = not requested.
: "${MEM_PER_NODE:=}"

# --- Experiment sizes ----------------------------------------------------------------
: "${MAX_CORES:=256}"          # largest core count in the scaling runs (rubric: >= 256)
: "${THREADS_PER_RANK:=4}"     # OpenMP threads per MPI rank in the scaling runs
: "${REPEATS:=3}"              # repetitions of every timed configuration (median reported)
: "${REPEATS_LARGE:=$REPEATS}" # ebase/e7bm/e7cm: repetitions at N >= SINGLE_RUN_N (all core counts)
: "${STEPS:=2}"                # time steps per timed run (plus one initial force evaluation)
: "${DIST:=plummer}"           # main test system
: "${THETA:=0.5}"
: "${EPS:=0.01}"
: "${DT:=0.001}"
: "${SCHED:=dynamic}"          # schedule/chunk used outside E4
: "${CHUNK:=1024}"
: "${TREE_BUILD:=morton}"      # tree builder used outside E3

: "${E1_DIRECT_N:=1000 2000 4000 8000 16000 32000}"           # serial direct sum (O(N^2))
: "${E1_BH_N:=1000 4000 16000 64000 256000 1024000}"          # serial Barnes-Hut
: "${E2_N:=100000}"                                            # theta sweep (direct-sum reference)
: "${E2_THETAS:=0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 1.0}"
: "${E3_N:=1000000}"                                           # tree build
: "${E4_N:=500000}"                                            # OpenMP schedules
: "${E4_SCHEDS:=static:0 static:64 dynamic:1 dynamic:64 dynamic:1024 guided:0}"
: "${N_STRONG:=2000000 16000000}"                              # E7b strong scaling, two sizes
: "${N_LARGE:=16000000}"                                       # E5a, E7a, E7d
: "${SINGLE_RUN_N:=8000000}"                                   # at this N or more, 1-core runs are not repeated
: "${N_PER_CORE:=62500}"                                       # E7c weak scaling
: "${BASE_N:=62500 2000000 16000000}"                          # ebase: serial V1 baselines (weak, strong)
: "${E5A_CORES:=64 128 256}"                                   # communication modes (multi-node)
: "${E5B_CORES:=32 64 128 256}"                                # shared windows, 1 thread per rank
: "${E5B_N:=1000000}"
: "${E8_N:=100000 1000000}"                                     # E8 accuracy: forces vs direct sum (O(N^2) reference)
: "${E8_CORES:=1 4 16 64 256}"                                  # E8 pure-MPI rank counts (+ MAX_CORES at THREADS_PER_RANK)
: "${E9_N:=1000000}"                                            # E9 long run
: "${E9_T:=1}"                                                  # E9 simulated time (steps = E9_T / dt)
: "${E9_DTS:=0.001 0.0005}"                                     # E9 time steps (halving dt: 2nd-order integrator check)
: "${E9_DIAG:=10}"                                              # E9 energy/momentum every K steps
: "${E9_CORES:=}"                                               # E9 cores (empty = MAX_CORES), THREADS_PER_RANK per rank
: "${E10_N:=16000000}"                                          # E10 load balancing
: "${E10_STEPS:=4}"                                             # E10 steps (weighted split used from the 2nd evaluation)
: "${E10_CORES:=}"                                              # E10 cores (empty = MAX_CORES), 1 and THREADS_PER_RANK per rank
: "${E11_N:=1000000}"                                           # E11 LET receiver boxes
: "${E11_CORES:=64 256}"                                        # E11 rank counts (1 thread per rank)
: "${E11_BOXES:=1 4 16 64}"                                     # E11 boxes per rank K
: "${E7A_BIND:=wrap}"          # E7a binding: wrap = bind_rank.sh (all splits start), mpi = Open MPI binding

# --- Wall-clock limits per job (hh:mm:ss) ----------------------------------------------
: "${WALL_TESTS:=00:30:00}"
: "${WALL_E1:=02:00:00}"
: "${WALL_E2:=01:00:00}"
: "${WALL_E3:=01:00:00}"
: "${WALL_E4:=01:00:00}"
: "${WALL_E5:=01:00:00}"
: "${WALL_E7:=03:00:00}"
: "${WALL_E7B_1NODE:=05:30:00}"   # E7b up to one node: holds the 1-core runs at 16M
: "${WALL_E7BM_1NODE:=05:00:00}"  # E7bm up to one node: 1..64 ranks at 2M and 16M, 1-rank runs repeated
: "${WALL_EBASE:=03:00:00}"       # serial baselines (16M repeated)

# Experiments run by default (subset with --only). e8 = accuracy of the parallel code,
# e9 = long run, e10 = load balancing, e11 = LET boxes per rank (not in the default list: run them with --only).
: "${EXPERIMENTS:=e1 e2 e3 e4 e5a e5b e7a e7b e7c e7d}"  # also: ebase e7bm e7cm e8 e9 e10 e11

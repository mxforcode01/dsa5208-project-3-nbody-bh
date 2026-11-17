# Experiment definitions (sourced). Each experiment is a bash function run_<exp> that runs
# the configurations belonging to the current job, and nodes_<exp> prints the node counts
# it needs (run_all.sh submits one PBS job per node count). Identical in local and PBS mode.
#
# Needs: NB_ROOT, BUILD_DIR, RUN_DIR, CORES_PER_NODE (C), NODES, the config.sh variables,
# and lib.sh. Every timed run appends one CSV row; each job writes its own CSV files.

SERIAL() { "$BUILD_DIR/nbody_serial" "$@"; }
MPIBIN() { echo "$BUILD_DIR/nbody_mpi"; }

nb_common_args() {
  echo "--dist $DIST --theta $THETA --eps $EPS --dt $DT --tree-build $TREE_BUILD --sched $SCHED --chunk $CHUNK"
}

# Runs a command, logging its output to $RUN_DIR/logs/<name>.log. A failure is recorded in
# $RUN_DIR/failures.txt and the job carries on with the next configuration.
nb_run() {
  local name=$1; shift
  mkdir -p "$RUN_DIR/logs"
  local log="$RUN_DIR/logs/$name.log"
  nb_log "run $name"
  "$@" >>"$log" 2>&1
  local rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "$(date '+%F %T') $name exit=$rc : $*" >>"$RUN_DIR/failures.txt"
    nb_log "FAILED $name (exit $rc, see logs/$name.log)"
  fi
  return 0
}

# Powers of two from 1 to $1.
nb_pow2_upto() { local p=1; while [ "$p" -le "$1" ]; do printf '%s ' "$p"; p=$(( p * 2 )); done; echo; }
# Divisors of $1, ascending.
nb_divisors() { local d; for (( d = 1; d <= $1; d++ )); do [ $(( $1 % d )) -eq 0 ] && printf '%s ' "$d"; done; echo; }
# Core counts from list $1 that do not exceed MAX_CORES.
nb_capped() { local p; for p in $1; do [ "$p" -le "$MAX_CORES" ] && printf '%s ' "$p"; done; echo; }

# Distinct node counts for core list $1 at up to $2 threads per rank.
nb_nodes_for() {
  local p; for p in $1; do set -- $(nb_layout "$p" "$2" "$CORES_PER_NODE"); echo "$3"; done | sort -n | uniq | tr '\n' ' '
}

# --- tests ---------------------------------------------------------------------------
nb_run_tests() {
  ( cd "$BUILD_DIR" && OMP_NUM_THREADS=2 ctest --output-on-failure --timeout 600 ) 2>&1 | tee "$RUN_DIR/tests.log"
  return "${PIPESTATUS[0]}"
}

# --- E1: serial run time against N (direct sum and Barnes-Hut) --------------------------
nodes_e1() { echo 1; }
run_e1() {
  local r n
  for (( r = 1; r <= REPEATS; r++ )); do
    for n in $E1_DIRECT_N; do
      nb_run "e1_direct_n${n}_r$r" SERIAL --solver direct --n "$n" --steps 1 --threads 1 --dist "$DIST" \
        --eps "$EPS" --dt "$DT" --csv "$RUN_DIR/e1_direct.csv"
    done
    for n in $E1_BH_N; do
      # shellcheck disable=SC2046
      nb_run "e1_bh_n${n}_r$r" SERIAL --solver bh --n "$n" --steps 1 --threads 1 $(nb_common_args) \
        --csv "$RUN_DIR/e1_bh.csv"
    done
  done
}

# --- E2: accuracy and cost against theta (forces at t = 0 vs direct sum) ----------------
nodes_e2() { echo 1; }
run_e2() {
  local tmp="$RUN_DIR/tmp_e2" th r
  mkdir -p "$tmp"
  nb_run e2_reference SERIAL --solver direct --n "$E2_N" --steps 0 --threads "$CORES_PER_NODE" --dist "$DIST" \
    --eps "$EPS" --snapshot "$tmp/ref.bin"
  for th in $E2_THETAS; do
    for (( r = 1; r <= REPEATS; r++ )); do
      nb_run "e2_theta${th}_r$r" SERIAL --solver bh --n "$E2_N" --steps 0 --threads 1 --dist "$DIST" --eps "$EPS" \
        --theta "$th" --tree-build "$TREE_BUILD" --csv "$RUN_DIR/e2_time.csv" --snapshot "$tmp/bh.bin"
    done
    nb_run "e2_compare_theta$th" "$BUILD_DIR/compare_snapshots" "$tmp/bh.bin" "$tmp/ref.bin" \
      --csv "$RUN_DIR/e2_accuracy.csv" --label "theta=$th" --label "dist=$DIST" --label "n_total=$E2_N"
  done
  rm -rf "$tmp"
}

# --- E3: tree construction, insertion vs Morton, against threads ------------------------
nodes_e3() { echo 1; }
run_e3() {
  local d b t r threads
  threads="$(nb_pow2_upto "$CORES_PER_NODE")"
  case " $threads " in *" $CORES_PER_NODE "*) ;; *) threads="$threads $CORES_PER_NODE" ;; esac
  for (( r = 1; r <= REPEATS; r++ )); do
    for d in plummer uniform; do
      for b in insertion morton; do
        for t in $threads; do
          nb_run "e3_${d}_${b}_t${t}_r$r" SERIAL --solver bh --n "$E3_N" --steps 0 --threads "$t" --dist "$d" \
            --theta "$THETA" --eps "$EPS" --tree-build "$b" --sched "$SCHED" --chunk "$CHUNK" --csv "$RUN_DIR/e3.csv"
        done
      done
    done
  done
}

# --- E4: OpenMP schedules, uniform vs clustered, one full node --------------------------
nodes_e4() { echo 1; }
run_e4() {
  local d sc kind chunk r
  for (( r = 1; r <= REPEATS; r++ )); do
    for d in uniform plummer; do
      for sc in $E4_SCHEDS; do
        kind=${sc%%:*}; chunk=${sc##*:}
        nb_run "e4_${d}_${kind}${chunk}_r$r" SERIAL --solver bh --n "$E4_N" --steps 2 --threads "$CORES_PER_NODE" \
          --dist "$d" --theta "$THETA" --eps "$EPS" --tree-build "$TREE_BUILD" --sched "$kind" --chunk "$chunk" \
          --csv "$RUN_DIR/e4.csv"
      done
    done
  done
}

# --- E7a: ranks x threads on one full node ------------------------------------------------
# Every split is bound by bind_rank.sh (E7A_BIND=wrap), so all rows use the same placement and
# the splits whose ranks cross a socket (1x72, 3x24, 9x8) can start at all.
nodes_e7a() { echo 1; }
run_e7a() {
  local rk t r
  for (( r = 1; r <= REPEATS; r++ )); do
    for rk in $(nb_divisors "$CORES_PER_NODE"); do
      t=$(( CORES_PER_NODE / rk ))
      # shellcheck disable=SC2046
      NB_BIND=$E7A_BIND nb_run "e7a_${rk}x${t}_r$r" nb_mpirun "$rk" "$t" 1 -- "$(MPIBIN)" --n "$N_LARGE" \
        --steps "$STEPS" --threads "$t" $(nb_common_args) --csv "$RUN_DIR/e7a.csv"
    done
  done
}

# Repetitions for problem size $1: REPEATS_LARGE at N >= SINGLE_RUN_N, else REPEATS.
nb_repeats_for() { if [ "$1" -ge "$SINGLE_RUN_N" ]; then echo "$REPEATS_LARGE"; else echo "$REPEATS"; fi; }

# Shared body of the multi-node scaling experiments: runs core counts $1 (list) whose layout
# at $2 threads per rank needs exactly $NODES nodes. $3 = tag, $4 = N or "weak", $5... = extra args.
nb_scaling_runs() {
  local cores=$1 tpr=$2 tag=$3 nmode=$4; shift 4
  local p r rk t nn rpn n
  for (( r = 1; r <= REPEATS; r++ )); do
    for p in $cores; do
      read -r rk t nn rpn <<<"$(nb_layout "$p" "$tpr" "$CORES_PER_NODE")"
      [ "$nn" -eq "$NODES" ] || continue
      if [ "$nmode" = weak ]; then n=$(( N_PER_CORE * p )); else n=$nmode; fi
      if [ "$p" -eq 1 ] && [ "$r" -gt 1 ] && [ "$n" -ge "$SINGLE_RUN_N" ]; then continue; fi
      # shellcheck disable=SC2046
      nb_run "${tag}_N${n}_p${p}_${rk}x${t}_r$r" nb_mpirun "$rk" "$t" "$nn" -- "$(MPIBIN)" --n "$n" --steps "$STEPS" \
        --threads "$t" $(nb_common_args) "$@" --csv "$RUN_DIR/${tag}_n${NODES}.csv"
    done
  done
}

# --- E7b: strong scaling (fixed N), with the serial V1 baseline -------------------------
nodes_e7b() { nb_nodes_for "$(nb_pow2_upto "$MAX_CORES")" "$THREADS_PER_RANK"; }
run_e7b() {
  local r n
  for n in $N_STRONG; do
    if [ "$NODES" -eq 1 ]; then
      for (( r = 1; r <= REPEATS; r++ )); do
        if [ "$r" -gt 1 ] && [ "$n" -ge "$SINGLE_RUN_N" ]; then continue; fi
        # shellcheck disable=SC2046
        nb_run "e7b_v1_N${n}_r$r" SERIAL --solver bh --n "$n" --steps "$STEPS" --threads 1 $(nb_common_args) \
          --csv "$RUN_DIR/e7b_v1.csv"
      done
    fi
    nb_scaling_runs "$(nb_pow2_upto "$MAX_CORES")" "$THREADS_PER_RANK" e7b "$n"
  done
}

# --- E7c: weak scaling (N_PER_CORE particles per core), with V1 at N_PER_CORE --------------
nodes_e7c() { nodes_e7b; }
run_e7c() {
  local r
  if [ "$NODES" -eq 1 ]; then
    for (( r = 1; r <= REPEATS; r++ )); do
      # shellcheck disable=SC2046
      nb_run "e7c_v1_r$r" SERIAL --solver bh --n "$N_PER_CORE" --steps "$STEPS" --threads 1 $(nb_common_args) \
        --csv "$RUN_DIR/e7c_v1.csv"
    done
  fi
  nb_scaling_runs "$(nb_pow2_upto "$MAX_CORES")" "$THREADS_PER_RANK" e7c weak
}

# --- E7d: ranks x threads across nodes, at the full core count -----------------------------
nodes_e7d() { nb_nodes_for "$MAX_CORES" 1; }
run_e7d() {
  local t
  for t in 1 2 4 8 16; do
    [ "$t" -le "$MAX_CORES" ] && nb_scaling_runs "$MAX_CORES" "$t" e7d "$N_LARGE"
  done
}

# --- Pure MPI (1 thread per rank) and its serial baseline ----------------------------------
# Same code and settings as E7b/E7c, but THREADS = 1, so P cores = P MPI ranks. Repetitions
# follow nb_repeats_for (all sizes repeated, including the 1-core runs at 16M).
nb_mpi_runs() {  # cores, tag, N or "weak"
  local cores=$1 tag=$2 nmode=$3
  local p r rk t nn rpn n reps
  for (( r = 1; r <= $(nb_repeats_max); r++ )); do
    for p in $cores; do
      read -r rk t nn rpn <<<"$(nb_layout "$p" 1 "$CORES_PER_NODE")"
      [ "$nn" -eq "$NODES" ] || continue
      if [ "$nmode" = weak ]; then n=$(( N_PER_CORE * p )); else n=$nmode; fi
      reps=$(nb_repeats_for "$n"); [ "$r" -le "$reps" ] || continue
      # shellcheck disable=SC2046
      nb_run "${tag}_N${n}_p${p}_${rk}x1_r$r" nb_mpirun "$rk" 1 "$nn" -- "$(MPIBIN)" --n "$n" --steps "$STEPS" \
        --threads 1 $(nb_common_args) --csv "$RUN_DIR/${tag}_n${NODES}.csv"
    done
  done
}
nb_repeats_max() { echo $(( REPEATS > REPEATS_LARGE ? REPEATS : REPEATS_LARGE )); }

# EBASE: serial tree code (V1) at every baseline size, repeated (one CSV for all).
nodes_ebase() { echo 1; }
run_ebase() {
  local r n
  for (( r = 1; r <= $(nb_repeats_max); r++ )); do
    for n in $BASE_N; do
      [ "$r" -le "$(nb_repeats_for "$n")" ] || continue
      # shellcheck disable=SC2046
      nb_run "ebase_v1_N${n}_r$r" SERIAL --solver bh --n "$n" --steps "$STEPS" --threads 1 $(nb_common_args) \
        --csv "$RUN_DIR/ebase_v1.csv"
    done
  done
}

# E7bm: pure-MPI strong scaling, 1 .. MAX_CORES ranks, at every N in N_STRONG.
nodes_e7bm() { nb_nodes_for "$(nb_pow2_upto "$MAX_CORES")" 1; }
run_e7bm() { local n; for n in $N_STRONG; do nb_mpi_runs "$(nb_pow2_upto "$MAX_CORES")" e7bm "$n"; done; }

# E7cm: pure-MPI weak scaling, N_PER_CORE particles per rank.
nodes_e7cm() { nodes_e7bm; }
run_e7cm() { nb_mpi_runs "$(nb_pow2_upto "$MAX_CORES")" e7cm weak; }

# --- E5a: blocking / point-to-point / overlapped LET exchange across nodes ---------------
nodes_e5a() { nb_nodes_for "$(nb_capped "$E5A_CORES")" "$THREADS_PER_RANK"; }
run_e5a() {
  local m
  for m in blocking p2p overlap; do
    nb_scaling_runs "$(nb_capped "$E5A_CORES")" "$THREADS_PER_RANK" "e5a_$m" "$N_LARGE" --comm "$m"
  done
}

# --- E5b: shared-memory windows on/off, one thread per rank (many ranks per node) --------
nodes_e5b() { nb_nodes_for "$(nb_capped "$E5B_CORES")" 1; }
run_e5b() {
  local s
  for s in 0 1; do
    nb_scaling_runs "$(nb_capped "$E5B_CORES")" 1 "e5b_shm$s" "$E5B_N" --shm "$s"
  done
}

# --- E8: force accuracy of the parallel code (serial vs MPI vs direct sum) ----------------
# Forces at t = 0 (--steps 0). Each job makes its own direct-sum reference and serial
# Barnes-Hut snapshot, then compares every MPI run against both: error vs direct sum shows the
# parallel code is as accurate as the serial tree code; difference vs serial BH shows how much
# the distributed tree + LET change the approximation.
nb_e8_layouts() {  # prints "cores threads_per_rank" pairs
  local p; for p in $(nb_capped "$E8_CORES"); do echo "$p 1"; done
  [ "$THREADS_PER_RANK" -gt 1 ] && [ "$MAX_CORES" -ge "$THREADS_PER_RANK" ] && echo "$MAX_CORES $THREADS_PER_RANK"
}
nodes_e8() {
  local p t
  while read -r p t; do set -- $(nb_layout "$p" "$t" "$CORES_PER_NODE"); echo "$3"; done <<<"$(nb_e8_layouts)" |
    sort -n | uniq | tr '\n' ' '
}
run_e8() {
  # </dev/null on every run inside the loop: mpirun would otherwise read the list of layouts.
  local tmp="$RUN_DIR/tmp_e8_n$NODES" n p tpr rk t nn rpn tag
  mkdir -p "$tmp"
  for n in $E8_N; do
    nb_run "e8_N${n}_direct_ref" SERIAL --solver direct --n "$n" --steps 0 --threads "$CORES_PER_NODE" \
      --dist "$DIST" --eps "$EPS" --snapshot "$tmp/ref.bin"
    # shellcheck disable=SC2046
    nb_run "e8_N${n}_serial_bh" SERIAL --solver bh --n "$n" --steps 0 --threads "$CORES_PER_NODE" \
      $(nb_common_args) --snapshot "$tmp/bh.bin"
    if [ "$NODES" -eq 1 ]; then
      nb_run "e8_N${n}_cmp_serial_direct" "$BUILD_DIR/compare_snapshots" "$tmp/bh.bin" "$tmp/ref.bin" \
        --csv "$RUN_DIR/e8_accuracy_n$NODES.csv" --label "n_total=$n" --label "code=serial BH" --label "ranks=1" \
        --label "threads=$CORES_PER_NODE" --label "reference=direct sum"
    fi
    while read -r p tpr; do
      read -r rk t nn rpn <<<"$(nb_layout "$p" "$tpr" "$CORES_PER_NODE")"
      [ "$nn" -eq "$NODES" ] || continue
      tag="e8_N${n}_p${p}_${rk}x${t}"
      # shellcheck disable=SC2046
      nb_run "$tag" nb_mpirun "$rk" "$t" "$nn" -- "$(MPIBIN)" --n "$n" --steps 0 --threads "$t" $(nb_common_args) \
        --snapshot "$tmp/mpi.bin" --csv "$RUN_DIR/e8_time_n$NODES.csv" </dev/null
      nb_run "${tag}_cmp_direct" "$BUILD_DIR/compare_snapshots" "$tmp/mpi.bin" "$tmp/ref.bin" \
        --csv "$RUN_DIR/e8_accuracy_n$NODES.csv" --label "n_total=$n" --label "code=parallel (MPI)" \
        --label "ranks=$rk" --label "threads=$t" --label "reference=direct sum" </dev/null
      nb_run "${tag}_cmp_serial" "$BUILD_DIR/compare_snapshots" "$tmp/mpi.bin" "$tmp/bh.bin" \
        --csv "$RUN_DIR/e8_accuracy_n$NODES.csv" --label "n_total=$n" --label "code=parallel (MPI)" \
        --label "ranks=$rk" --label "threads=$t" --label "reference=serial BH" </dev/null
      rm -f "$tmp/mpi.bin"
    done <<<"$(nb_e8_layouts)"
  done
  rm -rf "$tmp"
}

# --- E9: one long run (energy and momentum over many steps) ------------------------------
# E9_T time units at each dt in E9_DTS, diagnostics every E9_DIAG steps. The per-step energy
# table the program prints is turned into e9_diag_n<nodes>.csv (columns below).
nodes_e9() { nb_nodes_for "$(nb_capped "${E9_CORES:-$MAX_CORES}")" "$THREADS_PER_RANK"; }
run_e9() {
  local p rk t nn rpn dt steps name log
  p=$(nb_capped "${E9_CORES:-$MAX_CORES}"); p=${p%% *}
  read -r rk t nn rpn <<<"$(nb_layout "$p" "$THREADS_PER_RANK" "$CORES_PER_NODE")"
  [ "$nn" -eq "$NODES" ] || return 0
  for dt in $E9_DTS; do
    steps=$(awk -v T="$E9_T" -v h="$dt" 'BEGIN { printf "%d", T / h + 0.5 }')
    name="e9_N${E9_N}_p${p}_${rk}x${t}_dt${dt}"
    log="$RUN_DIR/logs/$name.log"
    # shellcheck disable=SC2046
    nb_run "$name" nb_mpirun "$rk" "$t" "$nn" -- "$(MPIBIN)" --n "$E9_N" --steps "$steps" --threads "$t" \
      $(nb_common_args) --dt "$dt" --diag-every "$E9_DIAG" --csv "$RUN_DIR/e9_n$NODES.csv"
    local out="$RUN_DIR/e9_diag_n$NODES.csv"
    [ -s "$out" ] || echo "n,ranks,threads,dt,step,time,energy,rel_energy_drift,rel_momentum_drift" >"$out"
    awk -v n="$E9_N" -v rk="$rk" -v t="$t" -v dt="$dt" '
      /^ *step +time +energy/ { on = 1; next }
      on && NF == 5 && $1 ~ /^[0-9]+$/ { printf "%s,%s,%s,%s,%s,%s,%s,%s,%s\n", n, rk, t, dt, $1, $2, $3, $4, $5; next }
      on { on = 0 }' "$log" >>"$out"
  done
}

# --- E10: load balancing, equal particle count vs work-weighted split ---------------------
# --rebalance-every 1: a new split every force evaluation, weighted by each particle's
# interaction count from the previous one. --rebalance-every > steps: the split made at the
# first evaluation (no work known yet, so equal counts along the Morton curve) is kept.
nodes_e10() {
  local p; p=$(nb_capped "${E10_CORES:-$MAX_CORES}"); p=${p%% *}
  { nb_nodes_for "$p" 1; nb_nodes_for "$p" "$THREADS_PER_RANK"; } | tr ' ' '\n' | grep . | sort -n | uniq | tr '\n' ' '
}
run_e10() {
  local p tpr d bal every r rk t nn rpn
  p=$(nb_capped "${E10_CORES:-$MAX_CORES}"); p=${p%% *}
  for (( r = 1; r <= REPEATS; r++ )); do
    for tpr in 1 "$THREADS_PER_RANK"; do
      read -r rk t nn rpn <<<"$(nb_layout "$p" "$tpr" "$CORES_PER_NODE")"
      [ "$nn" -eq "$NODES" ] || continue
      for d in plummer uniform; do
        for bal in work count; do
          if [ "$bal" = work ]; then every=1; else every=1000000; fi
          # shellcheck disable=SC2046
          nb_run "e10_${d}_${bal}_p${p}_${rk}x${t}_r$r" nb_mpirun "$rk" "$t" "$nn" -- "$(MPIBIN)" --n "$E10_N" \
            --steps "$E10_STEPS" --threads "$t" $(nb_common_args) --dist "$d" --rebalance-every "$every" \
            --csv "$RUN_DIR/e10_n$NODES.csv"
        done
      done
      if [ "$THREADS_PER_RANK" -eq 1 ]; then break; fi  # (an `[ ] && break` here made the job exit 1)
    done
  done
  return 0
}

# --- E11: LET receiver boxes per rank (K), pure MPI ---------------------------------------
# Each rank describes its particles to the senders by up to K tight boxes; the sender prunes
# its LET against them. K = 1 is one bounding box per rank; the other experiments use 64.
nodes_e11() { nb_nodes_for "$(nb_capped "$E11_CORES")" 1; }
run_e11() {
  local p k r rk t nn rpn
  for (( r = 1; r <= REPEATS; r++ )); do
    for p in $(nb_capped "$E11_CORES"); do
      read -r rk t nn rpn <<<"$(nb_layout "$p" 1 "$CORES_PER_NODE")"
      [ "$nn" -eq "$NODES" ] || continue
      for k in $E11_BOXES; do
        # shellcheck disable=SC2046
        nb_run "e11_K${k}_p${p}_${rk}x1_r$r" nb_mpirun "$rk" 1 "$nn" -- "$(MPIBIN)" --n "$E11_N" --steps "$STEPS" \
          --threads 1 $(nb_common_args) --let-boxes "$k" --csv "$RUN_DIR/e11_n$NODES.csv"
      done
    done
  done
}

# Entry point used by job.pbs and by run_all.sh --local.
nb_run_experiment() {
  local exp=$1
  NODES=$2
  export NODES
  {
    echo "experiment $exp on $NODES node(s), $CORES_PER_NODE cores per node, $(date '+%F %T')"
    echo "hosts: $( [ -n "${PBS_NODEFILE:-}" ] && sort -u "$PBS_NODEFILE" | tr '\n' ' ' || hostname)"
    nb_print_toolchain
    lscpu 2>/dev/null | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\))' || true
  } >"$RUN_DIR/env_${exp}_n${NODES}.txt"
  if [ "$exp" = tests ]; then nb_run_tests; return $?; fi
  type "run_$exp" >/dev/null 2>&1 || nb_die "unknown experiment '$exp'"
  "run_$exp"
}

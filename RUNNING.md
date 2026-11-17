# Running the experiments on Vanda

All results in the report were produced on the NUS Vanda cluster (PBS). `README.md` shows how to
copy the code to Vanda, build it, test it and run a small example. This file describes the
experiment pipeline in more detail.

One script builds the code, runs the tests and submits every experiment:
`scripts/hpc/run_all.sh`. The experiment jobs start only if the tests pass. Compilers, MPI, CMake
and cores per node are detected, or can be set by hand (Section 5).

## 1. Build, test and submit

Vanda needs two settings on every submission (`CORES_PER_NODE` and `MEM_PER_NODE`), and its login
node should not compile. With the code unpacked in `~/dsa5208-project-3-nbody-bh` (a path **without
spaces**: PBS cannot pass such paths):

```bash
# Build and test on a compute node
qsub -I -l select=1:ncpus=8:mpiprocs=8 -l walltime=01:00:00
cd ~/dsa5208-project-3-nbody-bh
export NB_MODULES="foss/2025b CMake/4.0.3-GCCcore-14.3.0"
./scripts/hpc/run_all.sh --local --tests-only     # expect: 100% tests passed, 0 tests failed out of 15
exit

# Submit from the login node
cd ~/dsa5208-project-3-nbody-bh
export CORES_PER_NODE=72 MEM_PER_NODE=500gb
export NB_MODULES="foss/2025b CMake/4.0.3-GCCcore-14.3.0"
./scripts/hpc/run_all.sh --no-build --dry-run       # prints the jobs it would submit, submits nothing
./scripts/hpc/run_all.sh --no-build --only e7b      # add REPEATS=1 in front for a pilot
qstat -u $USER                                      # watch the jobs
```

Vanda runs a limited number of jobs per user at once and holds the rest in the queue.

Results go to `results/hpc/<date>-<host>/`: one CSV file per experiment and job, a log for
every single run under `logs/`, the PBS output under `pbs/`, the list of submitted jobs in
`jobs.txt`, and `failures.txt` if any run failed.

### Main experiments

```bash
./scripts/hpc/run_all.sh --no-build                 # e1, e2, e3, e4, e5a, e5b, e7a, e7b, e7c, e7d
```

### Pure-MPI scaling, repeated baselines and the full ranks x threads sweep

```bash
REPEATS=5 REPEATS_LARGE=3 ./scripts/hpc/run_all.sh --no-build --only e7a,ebase,e7bm,e7cm
```

- `e7bm` / `e7cm`: strong (N = 2M, 16M) and weak (62,500 per rank) scaling with **one thread
  per rank**, 1 to 256 MPI ranks (64 ranks per node from 128 up).
- `ebase`: the serial tree code at 62,500, 2M and 16M, repeated (REPEATS_LARGE times at 16M).
- `e7a`: all 12 ranks x threads splits of one node. Every split is pinned by
  `scripts/hpc/bind_rank.sh` (`E7A_BIND=wrap`): Open MPI 5 refuses to bind a rank to cores in
  both sockets, which is why 1x72, 3x24 and 9x8 failed with exit 254 in the first run. Each log
  shows the cores of every rank (`bind_rank.sh: ...`) and of every thread (`affinity ...`).

`--no-build` reuses the binaries already in `build-hpc/`. The CSV column `git_hash` names the
commit the binaries were built from.

### Accuracy of the parallel code, a long run, load balancing, LET boxes

```bash
./scripts/hpc/run_all.sh --no-build --only e8,e9,e10,e11
```

- `e8`: forces at t = 0 from the serial tree code and from the MPI code on 1, 4, 16, 64, 256 ranks
  (and 64 x 4), each compared with a direct-sum reference and with the serial forces, N = 100k
  and 1M. Jobs on 1 and 4 nodes.
- `e9`: N = 1M on 256 cores (64 x 4) for one time unit, dt = 0.001 and 0.0005 (1000 and 2000
  steps), energy and momentum every 10 steps (`e9_diag_n4.csv`). One 4-node job.
- `e10`: N = 16M on 256 cores (256 x 1 and 64 x 4), Plummer and uniform: the work-weighted split
  recomputed every evaluation (`--rebalance-every 1`) against the first, equal-count split kept
  for the whole run. One 4-node job.
- `e11`: N = 1M, 64 and 256 ranks x 1 thread, K = 1, 4, 16, 64 LET receiver boxes per rank
  (`--let-boxes`; every other experiment uses 64). Jobs on 1 and 4 nodes.

Tables 17-20 and `fig_long_run`, `fig_load_balance`, `fig_letboxes` come from these.

## 2. GPU runs

The GPU experiment has its own script, build folder (`build-gpu/`) and compiler: the NVIDIA HPC
SDK (`nvc++`, OpenMP target offload) with the Open MPI that ships with it. On Vanda:

```bash
# Small check on an interactive GPU node (builds, runs test_gpu_offload, two small sizes)
qsub -I -l select=1:ngpus=1:ncpus=8:mpiprocs=8 -l walltime=01:00:00
cd ~/dsa5208-project-3-nbody-bh
export NB_GPU_MODULES="nvhpc/26.1 CMake/4.0.3-GCCcore-14.3.0"
./scripts/hpc/run_gpu.sh --local
exit

# Full experiment, from the login node: one job with 2 GPUs (NVIDIA A40 in our runs)
cd ~/dsa5208-project-3-nbody-bh
export NB_GPU_MODULES="nvhpc/26.1 CMake/4.0.3-GCCcore-14.3.0"
./scripts/hpc/run_gpu.sh
```

The full job compares CPU and GPU tree walks at N = 0.5M to 8M, on 1 rank + 1 GPU and on
2 ranks + 2 GPUs, with 36 CPU threads per rank and 3 repeats. Results go to
`results/hpc/<date>-<host>-gpu/gpu.csv`. The column `device_executed` must say `gpu`: the program
stops instead of silently falling back to the CPU.

## 3. Tables and figures

On any machine with Python 3, pandas, matplotlib and openpyxl, after copying the results back:

```bash
python scripts/hpc/make_tables.py results/hpc/<run> [more run folders ...] --out tables
python scripts/hpc/plot_results.py results/hpc/<run> [more run folders ...] --out figures
```

## 4. Send the results back

```bash
./scripts/hpc/collect.sh results/hpc/<run>   # summary + results/hpc/<run>.tar.gz
```

Copy the `.tar.gz` to your laptop (`scp`, the command is printed) and share it. Do not edit
the CSV files by hand.

## 5. What each experiment runs

| Exp | Runs | Jobs |
| --- | --- | --- |
| tests | all 15 test programs (`ctest`), MPI tests on 1–4 ranks | 1 small job |
| e1 | serial direct sum (N = 1k…32k) and serial Barnes–Hut (N = 1k…1M) | 1 node |
| e2 | θ = 0.1…1.0 at N = 100k: time, and error against a direct-sum reference | 1 node |
| e3 | tree build, insertion vs Morton, 1…all threads, N = 1M, uniform and Plummer | 1 node |
| e4 | OpenMP schedules (static/dynamic/guided, several chunks), N = 500k, all threads | 1 node |
| e7a | every ranks × threads split of one full node, N = 16M | 1 node |
| e7b | strong scaling: N = 2M and 16M on 1, 2, 4 … 256 cores (+ serial V1 baseline; 1-core runs at 16M run once) | one job per node count |
| e7c | weak scaling: 62,500 particles per core on 1 … 256 cores (+ V1 baseline) | one job per node count |
| e7d | ranks × threads across nodes: 256 cores, N = 16M, 1 to 16 threads per rank | 1 job |
| e5a | blocking / point-to-point / overlap exchange on 64, 128, 256 cores, N = 16M | one job per node count |
| e5b | shared-memory windows off / on, one thread per rank, 32 … 256 cores | one job per node count |

Every configuration is repeated 3 times. Sizes, repeats, wall-time limits and the
experiment list are in `scripts/hpc/config.sh`.

### Changing settings

Any setting in `scripts/hpc/config.sh` can be given on the command line, no editing needed:

```bash
QUEUE=<queue> ./scripts/hpc/run_all.sh                     # a specific queue
CORES_PER_NODE=72 MAX_CORES=128 ./scripts/hpc/run_all.sh   # fewer nodes
REPEATS=1 ./scripts/hpc/run_all.sh --only e7b              # quick first pass
NB_MODULES="foss/2025b CMake/4.0.3-GCCcore-14.3.0" ./scripts/hpc/run_all.sh
```

## 6. If something goes wrong

| Message | What to do |
| --- | --- |
| `Could not find a usable toolchain` | It lists the modules it found. Pick a GCC (≥ 7) + MPI + CMake (≥ 3.16) set and pass it as `NB_MODULES="…"`. |
| `GCC … is too old` / `MPI … found; shared-memory windows need MPI >= 3.0` | Same: load newer modules via `NB_MODULES`, then rerun (the old build is removed automatically). |
| `could not read cores per node from pbsnodes` | Set `CORES_PER_NODE=<n>` (see `lscpu` on a compute node or ask the HPC guide). |
| `qsub: … exceeds queue limit` | Lower `MAX_CORES`, raise/lower the `WALL_*` limits, or pick another `QUEUE`. |
| `WARNING: could not identify the MPI library` | Ranks will not be bound to cores (runs slower). Set `NB_MPIRUN_EXTRA` to your MPI's binding flags. |
| entries in `failures.txt` | The named run's log is in `logs/<name>.log`; the other runs carried on. |
| experiment jobs stay in state H (held) | The test job failed, so they will not start: read `tests.log`, fix, rerun. Delete the held jobs with `qdel`. |

## 7. Check the pipeline without a cluster

On any Linux machine with MPI (e.g. WSL):

```bash
./scripts/hpc/run_all.sh --local     # builds, tests, runs every experiment at small sizes (about 1 minute)
```

# Distributed 3D Barnes–Hut N-body (DSA5208 Project 3)

How to build and run the code on the NUS Vanda cluster.

## 1. Get the code and zip it

On your own computer:

```bash
git clone https://github.com/mxforcode01/dsa5208-project-3-nbody-bh.git
```

This creates the folder `dsa5208-project-3-nbody-bh`. Zip it as `dsa5208-project-3-nbody-bh.zip`:

- Windows: right-click the folder → *Send to* → *Compressed (zipped) folder*.
- macOS / Linux: `zip -r dsa5208-project-3-nbody-bh.zip dsa5208-project-3-nbody-bh`

## 2. Copy the zip to Vanda and log in

From your own computer (on the NUS network or VPN), replace `<nusnet-id>` with your NUSNET ID:

```bash
scp dsa5208-project-3-nbody-bh.zip <nusnet-id>@vanda.nus.edu.sg:~
ssh <nusnet-id>@vanda.nus.edu.sg
```

On Vanda:

```bash
cd ~
unzip dsa5208-project-3-nbody-bh.zip
cd dsa5208-project-3-nbody-bh
```

Keep the folder in a path **without spaces** (your home directory is fine).

## 3. Build and test on a compute node

Vanda's login node should not compile, so open an interactive session on a compute node first:

```bash
qsub -I -l select=1:ncpus=8:mpiprocs=8 -l walltime=01:00:00
```

When the session starts:

```bash
cd ~/dsa5208-project-3-nbody-bh
export NB_MODULES="foss/2025b CMake/4.0.3-GCCcore-14.3.0"
./scripts/hpc/run_all.sh --local --tests-only
```

This loads the compilers and MPI, builds everything into `build-hpc/` and runs the 15 test programs.
Expected last lines: `100% tests passed, 0 tests failed out of 15`.

## 4. Run a small example (same interactive session)

```bash
module load foss/2025b CMake/4.0.3-GCCcore-14.3.0

# Serial Barnes-Hut, 100,000 particles, 5 steps, 8 threads
./build-hpc/nbody_serial --solver bh --n 100000 --dist plummer --steps 5 --threads 8

# Distributed Barnes-Hut: 4 MPI ranks x 2 OpenMP threads, 200,000 particles
OMP_NUM_THREADS=2 mpirun -np 4 --bind-to none ./build-hpc/nbody_mpi --n 200000 --dist plummer --steps 5 --threads 2
```

Each run prints the energy and momentum drift, the time of every phase and the interaction counts.
`./build-hpc/nbody_mpi --help` lists all options. Leave the session with `exit`.

## 5. Run the full experiments (batch jobs)

From the **login node**, in `~/dsa5208-project-3-nbody-bh`:

```bash
export CORES_PER_NODE=72 MEM_PER_NODE=500gb
export NB_MODULES="foss/2025b CMake/4.0.3-GCCcore-14.3.0"
./scripts/hpc/run_all.sh --no-build
./scripts/hpc/run_all.sh --no-build --only ebase,e7bm,e7cm,e8,e9,e10,e11
```

The first command submits the main experiments (serial scaling, accuracy against θ, tree build,
OpenMP schedules, communication modes, shared-memory windows, ranks × threads, hybrid strong and weak
scaling up to 256 cores). The second submits the serial baselines, pure-MPI strong and weak scaling,
force accuracy of the parallel code, the long run, load balancing and the LET-box study. A test job
runs first; the experiment jobs start only if it passes.

Watch the jobs with `qstat -u $USER`. Results go to `results/hpc/<date>-<host>/`: one CSV file per
experiment, and `failures.txt` if any run failed.

## 6. GPU runs

The GPU version (OpenMP target offload of the tree walks) is built with the NVIDIA HPC SDK into
its own folder, `build-gpu/`. On an interactive GPU node:

```bash
qsub -I -l select=1:ngpus=1:ncpus=8:mpiprocs=8 -l walltime=01:00:00
cd ~/dsa5208-project-3-nbody-bh
export NB_GPU_MODULES="nvhpc/26.1 CMake/4.0.3-GCCcore-14.3.0"
./scripts/hpc/run_gpu.sh --local
exit
```

This builds the code, runs the GPU test (`test_gpu_offload`) and compares CPU and GPU runs at two
small sizes. To submit the full CPU-against-GPU experiment of the report (1 and 2 NVIDIA A40 GPUs,
N = 0.5M to 8M) from the login node:

```bash
export NB_GPU_MODULES="nvhpc/26.1 CMake/4.0.3-GCCcore-14.3.0"
./scripts/hpc/run_gpu.sh
```

To run one GPU case by hand inside the interactive session, add `--device gpu`:

```bash
module load nvhpc/26.1
OMP_NUM_THREADS=8 mpirun -np 1 --bind-to none ./build-gpu/nbody_mpi --n 500000 --dist plummer --steps 2 --threads 8 --tree-build morton --device gpu
```

The program stops with an error if the tree walks did not run on a GPU.

## 7. Tables and figures

On any machine with Python 3, pandas, matplotlib and openpyxl, from the `dsa5208-project-3-nbody`
folder:

```bash
python scripts/hpc/make_tables.py results/hpc/<run> [more run folders ...] --out tables
python scripts/hpc/plot_results.py results/hpc/<run> [more run folders ...] --out figures
```

The results used in the report, with their tables and figures, are already in `results/final/`.
Their CSV column `git_hash` (`ac8e98a`) names the team's internal development commit that built the
programs; the C++ sources in this repository are the same code (later changes were to scripts,
documentation and source comments only).

## 8. More detail

- `RUNNING.md`: every experiment, the GPU job, settings and common errors.
- `./build-hpc/nbody_mpi --help`: all program options.

To build on any Linux machine with a C++17 compiler, CMake (3.16 or newer), MPI and OpenMP:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

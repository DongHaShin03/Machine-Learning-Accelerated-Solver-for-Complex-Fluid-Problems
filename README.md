# Machine-Learning-Accelerated-Solver-for-Complex-Fluid-Problems
Multidisciplinary Project

## Introduction

This project builds on the paper by P.F. Antonietti, M. Caldana and L. Dede', *Accelerating Algebraic Multigrid Methods via Artificial Neural Networks*, Vietnam J. Math. 51 (2023), available in `reference/2_MD.pdf`. In the paper, a neural network chooses the AMG *strong threshold* parameter θ.

For now, the code solves the steady Stokes problem of Sec. 6.5 of the paper **without a neural network**, in order to study how AMG performance depends on θ:

- **domain**: channel (0, 2.2) × (0, 0.41) with a cylinder of radius 0.05 centered at (0.2, 0.2);
- **boundary conditions**: parabolic inflow velocity profile, zero velocity on the walls and on the cylinder, natural ("do-nothing") condition at the outflow;
- **discretization**: Taylor-Hood Q2–Q1 finite elements with [deal.II](https://www.dealii.org);
- **solver**: MINRES with a block-diagonal preconditioner, i.e. one BoomerAMG V-cycle (hypre, via PETSc) on the velocity block and Jacobi on the pressure mass matrix scaled by 1/ν;
- **analysis**: for each mesh refinement level, viscosity ν, maximum velocity U and value of θ, the program records the number of MINRES iterations, the approximate convergence factor ρ and the timings.

Two Python scripts use these results:

- `scripts/make_table2.py` builds tables in the format of Table 2 of the paper, i.e. `ρ(iterations)` for each θ;
- `scripts/make_figure4.py` draws the plots of Figure 4 of the paper, i.e. mean and standard deviation of the solver time as a function of θ, with one plot per mesh size h and one curve per (ν, U) pair in place of ε.

## Commands

### Environment (container)

In every new shell inside the container:

```bash
source /u/sw/etc/bash.bashrc
module load gcc-glibc
module load dealii
```

The deal.II module must be built with PETSc and hypre. To check:

```bash
grep -E "DEAL_II_WITH_PETSC|DEAL_II_PETSC_WITH_HYPRE" $mkDealiiPrefix/include/deal.II/base/config.h
```

### Build

From the repository root:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Run

From the `build/` folder:

```bash
./stokes [max_refinement] [n_repetitions]
```

- `max_refinement`: maximum refinement level; the program uses i = 0, ..., `max_refinement` (default 5, as in the paper);
- `n_repetitions`: how many times each solve is repeated to compute the mean and standard deviation of the timings (default 1). With 1 repetition the standard deviation is 0 and Figure 4 has no error bars. The iteration count and ρ do not depend on the repetitions.

Examples:

```bash
./stokes 2      # i = 0, 1, 2, one repetition (quick test)
./stokes 2 5    # i = 0, 1, 2, five repetitions per solve
./stokes        # i = 0, ..., 5, as in the paper
make run        # same as ./stokes (no arguments), with NPROCS MPI processes
```

To use more MPI processes with `make run`: `cmake -DNPROCS=4 ..`, then `make run`. To also pass arguments, call `mpirun` directly, e.g. `mpirun -n 4 ./stokes 2 5`.

The program writes to `build/`:

- `results.csv`: one row per solve, with columns `i, h, n_dofs, nu, U, theta, converged, iterations, rho, t_amg_setup, t_solve, t_solve_std`;
- `output-stokes_nu<ν>_U<U>_i<i>_*.vtu` / `.pvtu`: the solution for θ = 0.25, to be opened with ParaView.

### Python dependencies

Required for PNG images, i.e. for `make_figure4.py` and for the `--png` / `--png-dir` options of `make_table2.py`:

```bash
python3 -m pip install --user --force-reinstall -r requirements.txt
```

### Tables (Table 2 of the paper)

From the repository root:

```bash
# Markdown tables for all θ
python3 scripts/make_table2.py build/results.csv

# only the θ values shown in the paper
python3 scripts/make_table2.py build/results.csv --theta 0.24 0.48 0.72

# a single PNG image with tables colored by ρ, as in the paper
python3 scripts/make_table2.py build/results.csv --theta 0.24 0.48 0.72 --png table2.png

# one PNG image per θ, in the tables/ folder
python3 scripts/make_table2.py build/results.csv --png-dir tables

# LaTeX version, with colored cells
python3 scripts/make_table2.py build/results.csv --theta 0.24 0.48 0.72 --latex table2.tex
```

Other options: `--cmap` selects the colormap for the images (default `rainbow`). The full list is available via `python3 scripts/make_table2.py --help`.

### Timing plots (Figure 4 of the paper)

From the repository root:

```bash
# a single PNG image with one plot per mesh size h
python3 scripts/make_figure4.py build/results.csv --png figure4.png
```

Each plot shows the mean solver time (`t_solve`) as a function of θ, with the standard deviation (`t_solve_std`) as error bar. To get error bars, run `./stokes` with `n_repetitions` > 1. Without `--png` the image is saved to `figure4.png`. `--ncols` sets the number of plots per row (default 2, as in the paper).

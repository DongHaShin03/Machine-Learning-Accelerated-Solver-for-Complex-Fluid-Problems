# Machine-Learning-Accelerated-Solver-for-Complex-Fluid-Problems
Multidisciplinary Project

## Introduction

This project builds on the paper by P.F. Antonietti, M. Caldana and L. Dede', *Accelerating Algebraic Multigrid Methods via Artificial Neural Networks*, Vietnam J. Math. 51 (2023), available in `reference/2_MD.pdf`. In the paper, a neural network chooses the AMG *strong threshold* parameter θ.

For now, the code generates the two datasets of the paper **without a neural network**, in order to study how AMG performance depends on θ. Both programs run in serial, as in the paper.

The steady Stokes problem of Sec. 6.5 (`stokes`):

- **domain**: channel (0, 2.2) × (0, 0.41) with a cylinder of radius 0.05 centered at (0.2, 0.2);
- **boundary conditions**: parabolic inflow velocity profile, zero velocity on the walls and on the cylinder, natural ("do-nothing") condition at the outflow;
- **discretization**: Taylor-Hood Q2–Q1 finite elements with [deal.II](https://www.dealii.org);
- **solver**: MINRES with the block-diagonal preconditioner diag(A, M_h) of the paper, i.e. one BoomerAMG V-cycle (hypre, via PETSc) on the velocity block and the pressure mass matrix M_h inverted exactly (inner CG); `pressure_mass=scaled` uses M_p/ν instead of M_h;
- **analysis**: for each mesh refinement level (h = 0.14/2^i), viscosity ν, maximum velocity U and each of the 50 values of θ (3600 samples, as in the paper), the program records the number of MINRES iterations, the approximate convergence factor ρ and the timings.

The diffusion problem with a tile-wise coefficient of Sec. 6.4 (`diffusion`): Q1 elements on (−1, 1)² refined globally (h = 2/N_el = 0.125/2^i, i = 0, ..., 7), μ = 10^ε on size^mode tiles with ε uniform in (−20, 20) (Algorithm 5), CG preconditioned by one BoomerAMG V-cycle, 25 values of θ. The solves that do not converge in 100 iterations are kept with `converged = 0` (with ε up to 20 this happens for every θ on some misaligned checkerboards).

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
./stokes [key=value ...]
```

- `min_level`, `max_level`: refinement levels i = `min_level`, ..., `max_level` (default 0 and 5, as in the paper);
- `n_repetitions`: how many times each solve is repeated to compute the mean and standard deviation of the timings (default 1). With 1 repetition the standard deviation is 0 and Figure 4 has no error bars. The iteration count and ρ do not depend on the repetitions;
- `pressure_mass`: `paper` (M_h, default) or `scaled` (M_p/ν);
- `output`: name of the csv file (default `results.csv`).

Examples:

```bash
./stokes max_level=2                    # i = 0, 1, 2, one repetition (quick test)
./stokes max_level=2 n_repetitions=5    # i = 0, 1, 2, five repetitions per solve
./stokes                                # i = 0, ..., 5, as in the paper
make run                                # same as ./stokes (no arguments)
```

The programs stop if started with more than one MPI process: in parallel, the coarsening of BoomerAMG and the numbering of the DoFs would depend on the partition.

The program writes to the current folder:

- `results.csv`: one row per solve, with columns `i, h, n_dofs, nu, U, theta, converged, iterations, rho, t_amg_setup, t_solve, t_solve_std` (`h` = 0.14/2^i, as in the paper);
- `output-stokes_nu<ν>_U<U>_i<i>_*.vtu` / `.pvtu`: the solution for θ = 0.25, to be opened with ParaView.

The diffusion dataset:

```bash
./diffusion                      # 72 coefficients x 8 levels x 25 theta = 14400 samples (Sec. 6.4.2)
./diffusion seed=1 prefix=s1     # 72 other coefficients; seeds 0, ..., 13 give the ~200000 samples of Sec. 6.4.3
```

The options (`min_level`, `max_level`, `n_patterns`, `eps`, `max_iterations`, `n_repetitions`, `seed`, `prefix`) are described at the top of `src/main_diffusion.cpp`. The program writes `<prefix>_results.csv` (one row per solve, `h` = 2/N_el as in the paper) and `<prefix>_patterns.csv` (the exponents ε of each coefficient). Level 7 (4.2 million DoFs) needs about 3.5 GB of memory.

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

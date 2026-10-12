# Machine-Learning-Accelerated-Solver-for-Complex-Fluid-Problems
Multidisciplinary Project

## Introduzione

Questo progetto parte dal paper di P.F. Antonietti, M. Caldana e L. Dede', *Accelerating Algebraic Multigrid Methods via Artificial Neural Networks*, Vietnam J. Math. 51 (2023), in `reference/2_MD.pdf`. Nel paper una rete neurale sceglie il parametro di *strong threshold* θ dell'AMG.

Per ora il codice risolve il problema di Stokes stazionario della Sez. 6.5 del paper **senza rete neurale**, per studiare come le prestazioni di AMG dipendono da θ:

- **dominio**: canale (0, 2.2) × (0, 0.41) con un cilindro di raggio 0.05 centrato in (0.2, 0.2);
- **condizioni al bordo**: profilo parabolico di velocità in ingresso, velocità nulla su pareti e cilindro, condizione naturale ("do-nothing") in uscita;
- **discretizzazione**: elementi finiti Taylor-Hood Q2–Q1 con [deal.II](https://www.dealii.org);
- **solutore**: MINRES con precondizionatore diagonale a blocchi, cioè un V-ciclo di BoomerAMG (hypre, tramite PETSc) sul blocco della velocità e Jacobi sulla massa di pressione scalata per 1/ν;
- **analisi**: per ogni livello di raffinamento della mesh, viscosità ν, velocità massima U e valore di θ, il programma registra il numero di iterazioni di MINRES, il fattore di convergenza approssimato ρ e i tempi.

Lo script `scripts/make_table2.py` usa questi risultati per costruire le tabelle nel formato della Tabella 2 del paper, cioè `ρ(iterazioni)` per ogni θ.

## Comandi

### Ambiente (container)

A ogni nuova shell nel container:

```bash
source /u/sw/etc/bash.bashrc
module load gcc-glibc
module load dealii
```

Il modulo deal.II deve essere compilato con PETSc e hypre. Per controllarlo:

```bash
grep -E "DEAL_II_WITH_PETSC|DEAL_II_PETSC_WITH_HYPRE" $mkDealiiPrefix/include/deal.II/base/config.h
```

### Compilazione

Dalla cartella principale della repo:

```bash
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

### Esecuzione

Dalla cartella `build/`:

```bash
./stokes 2      # livelli di raffinamento i = 0, 1, 2 (prova rapida)
./stokes        # i = 0, ..., 5, come nel paper
make run        # come ./stokes, con NPROCS processi MPI
```

Per usare più processi MPI con `make run`: `cmake -DNPROCS=4 ..`, poi `make run`. In alternativa, `mpirun -n 4 ./stokes 2`.

Il programma scrive in `build/`:

- `results.csv`: una riga per ogni soluzione, con le colonne `i, h, n_dofs, nu, U, theta, converged, iterations, rho, t_amg_setup, t_solve, t_solve_std`;
- `output-stokes_nu<ν>_U<U>_i<i>_*.vtu` / `.pvtu`: la soluzione per θ = 0.25, da aprire con ParaView.

### Tabelle (Tabella 2 del paper)

Dipendenze Python, solo per le immagini PNG:

```bash
python3 -m pip install --user --force-reinstall -r requirements.txt
```

Dalla cartella principale della repo:

```bash
# tabelle in formato Markdown per tutti i θ
python3 scripts/make_table2.py build/results.csv

# solo i θ mostrati nel paper
python3 scripts/make_table2.py build/results.csv --theta 0.24 0.48 0.72

# un'unica immagine PNG con le tabelle colorate in base a ρ, come nel paper
python3 scripts/make_table2.py build/results.csv --theta 0.24 0.48 0.72 --png table2.png

# un'immagine PNG per ogni θ, nella cartella tables/
python3 scripts/make_table2.py build/results.csv --png-dir tables

# versione LaTeX, con le celle colorate
python3 scripts/make_table2.py build/results.csv --theta 0.24 0.48 0.72 --latex table2.tex
```

Altre opzioni: `--cmap` sceglie la colormap delle immagini (default `rainbow`). L'elenco completo si ottiene con `python3 scripts/make_table2.py --help`.

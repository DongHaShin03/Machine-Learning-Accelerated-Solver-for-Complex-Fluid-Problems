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

Due script Python usano questi risultati:

- `scripts/make_table2.py` costruisce le tabelle nel formato della Tabella 2 del paper, cioè `ρ(iterazioni)` per ogni θ;
- `scripts/make_figure4.py` disegna i grafici della Figura 4 del paper, cioè media e deviazione standard del tempo del solutore in funzione di θ, con un grafico per ogni dimensione della mesh h e una curva per ogni coppia (ν, U) al posto di ε.

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
./stokes [max_refinement] [n_repetitions]
```

- `max_refinement`: livello di raffinamento massimo, il programma usa i = 0, ..., `max_refinement` (default 5, come nel paper);
- `n_repetitions`: quante volte ripetere ogni soluzione per calcolare media e deviazione standard dei tempi (default 1). Con 1 ripetizione la deviazione standard è 0 e la Figura 4 non ha barre d'errore. Il numero di iterazioni e ρ non dipendono dalle ripetizioni.

Esempi:

```bash
./stokes 2      # i = 0, 1, 2, una ripetizione (prova rapida)
./stokes 2 5    # i = 0, 1, 2, cinque ripetizioni per ogni soluzione
./stokes        # i = 0, ..., 5, come nel paper
make run        # come ./stokes (senza argomenti), con NPROCS processi MPI
```

Per usare più processi MPI con `make run`: `cmake -DNPROCS=4 ..`, poi `make run`. Per passare anche gli argomenti, usa direttamente `mpirun`, ad esempio `mpirun -n 4 ./stokes 2 5`.

Il programma scrive in `build/`:

- `results.csv`: una riga per ogni soluzione, con le colonne `i, h, n_dofs, nu, U, theta, converged, iterations, rho, t_amg_setup, t_solve, t_solve_std`;
- `output-stokes_nu<ν>_U<U>_i<i>_*.vtu` / `.pvtu`: la soluzione per θ = 0.25, da aprire con ParaView.

### Dipendenze Python

Servono per le immagini PNG, cioè per `make_figure4.py` e per l'opzione `--png` / `--png-dir` di `make_table2.py`:

```bash
python3 -m pip install --user --force-reinstall -r requirements.txt
```

### Tabelle (Tabella 2 del paper)

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

### Grafici dei tempi (Figura 4 del paper)

Dalla cartella principale della repo:

```bash
# un'unica immagine PNG con un grafico per ogni dimensione della mesh h
python3 scripts/make_figure4.py build/results.csv --png figure4.png
```

Ogni grafico mostra il tempo medio del solutore (`t_solve`) in funzione di θ, con la deviazione standard (`t_solve_std`) come barra d'errore. Per avere le barre d'errore bisogna eseguire `./stokes` con `n_repetitions` > 1. Senza `--png` l'immagine viene salvata in `figure4.png`. Con `--ncols` si sceglie il numero di grafici per riga (default 2, come nel paper).

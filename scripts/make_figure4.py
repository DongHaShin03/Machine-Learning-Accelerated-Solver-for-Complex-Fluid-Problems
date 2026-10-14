#!/usr/bin/env python3
"""Figure 4 of Antonietti, Caldana, Dede' (Vietnam J. Math. 2023), for the Stokes problem.

One subplot per mesh size h, all in a single PNG image. Each subplot shows the
elapsed solver time (mean over the repetitions, with the standard deviation as
error bar) w.r.t. the strong threshold theta, with one curve per pair of
problem parameters (nu, U), in place of epsilon of the paper.

Input: the results.csv written by the stokes executable (columns t_solve and
t_solve_std, i.e. mean and standard deviation over n_repetitions solves).

Usage:
    python3 scripts/make_figure4.py build/results.csv
    python3 scripts/make_figure4.py build/results.csv --png build/figure4.png
"""

import argparse
import csv
import math
import sys

REQUIRED_COLUMNS = ("i", "h", "nu", "U", "theta", "t_solve", "t_solve_std")
THETA_DECIMALS = 6


def read_results(path):
    """Read results.csv; returns a list of dicts with typed values."""
    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
            if missing:
                sys.exit(f"{path}: missing columns {', '.join(missing)}")

            rows = []
            for line in reader:
                rows.append({
                    "i": int(line["i"]),
                    "h": float(line["h"]),
                    "nu": float(line["nu"]),
                    "U": float(line["U"]),
                    "theta": round(float(line["theta"]), THETA_DECIMALS),
                    "t_solve": float(line["t_solve"]),
                    "t_solve_std": float(line["t_solve_std"]),
                })
    except FileNotFoundError:
        sys.exit(f"{path}: file not found")

    if not rows:
        sys.exit(f"{path}: no data")
    return rows


def latex_number(x):
    """1e-05 -> 10^{-5}, 0.001 -> 0.001, 10 -> 10."""
    if x != 0 and not 1e-3 <= abs(x) < 1e4:
        mantissa, exponent = f"{x:e}".split("e")
        mantissa = float(mantissa)
        power = f"10^{{{int(exponent)}}}"
        return power if mantissa == 1 else rf"{mantissa:g} \cdot {power}"
    return f"{x:g}"


def write_png(path, rows, ncols):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        sys.exit(f"this script needs matplotlib ({error}).\n"
                 f"Install it for this interpreter ({sys.executable}) with:\n"
                 f"    {sys.executable} -m pip install --user -r requirements.txt")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    levels = sorted({r["i"] for r in rows})
    h_of_level = {}
    for r in rows:
        h_of_level.setdefault(r["i"], r["h"])
    params = sorted({(r["nu"], r["U"]) for r in rows})
    U_values = sorted({U for _, U in params})

    # As in the paper, one jet color per curve; a different marker for each U,
    # so that the curves with the same nu (often overlapping) can be told apart.
    colors = plt.get_cmap("jet")(np.linspace(0.0, 1.0, len(params)))
    markers = dict(zip(U_values, "osD^v<>ph"))

    nrows = math.ceil(len(levels) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.4 * nrows), squeeze=False)

    for k, i in enumerate(levels):
        ax = axes[k // ncols][k % ncols]
        for color, (nu, U) in zip(colors, params):
            curve = sorted((r for r in rows if r["i"] == i and r["nu"] == nu and r["U"] == U),
                           key=lambda r: r["theta"])
            if not curve:
                continue
            ax.errorbar([r["theta"] for r in curve],
                        [r["t_solve"] for r in curve],
                        yerr=[r["t_solve_std"] for r in curve],
                        color=color, marker=markers[U], markersize=4, linewidth=1.2,
                        elinewidth=1.0, label=rf"$\nu = {latex_number(nu)},\ U = {latex_number(U)}$")
        ax.set_ylim(bottom=0.0)
        ax.set_xlabel(r"$\theta$")
        ax.set_ylabel("solver time [s]")
        ax.text(0.5, 0.95, f"h={h_of_level[i]:.2e}", transform=ax.transAxes, ha="center", va="top")

    for k in range(len(levels), nrows * ncols):
        axes[k // ncols][k % ncols].axis("off")
    fig.tight_layout()

    # One legend for all the subplots (12 curves do not fit inside a subplot), one column
    # per nu: in the empty subplot if there is one, otherwise below the figure.
    handles, labels = axes[0][0].get_legend_handles_labels()
    n_nu = len({nu for nu, _ in params})
    if len(levels) < nrows * ncols:
        axes[-1][-1].legend(handles, labels, loc="center", ncol=n_nu, fontsize="small")
    else:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=n_nu)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG written to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Figure 4 of the paper (solver time w.r.t. theta, one subplot per h) from results.csv.")
    parser.add_argument("csv", nargs="?", default="results.csv", help="results.csv written by stokes (default: results.csv)")
    parser.add_argument("--png", metavar="FILE", default="figure4.png", help="output image (default: figure4.png)")
    parser.add_argument("--ncols", type=int, default=2, help="subplots per row (default: 2, as in the paper)")
    args = parser.parse_args()

    rows = read_results(args.csv)
    if all(r["t_solve_std"] == 0.0 for r in rows):
        print("warning: all standard deviations are 0 (n_repetitions = 1 in main.cpp?): "
              "no error bars", file=sys.stderr)
    write_png(args.png, rows, args.ncols)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Table 2 of Antonietti, Caldana, Dede' (Vietnam J. Math. 2023), for the Stokes problem.

For each strong threshold theta, a table whose rows are the problem parameters
(nu, U), whose columns are the mesh sizes h, and whose entries are

    rho(N_it)

i.e. the approximate convergence factor and the number of MINRES iterations.
Entries marked with * did not converge.

Input: the results.csv written by the stokes executable.

Usage:
    python3 scripts/make_table2.py build/results.csv
    python3 scripts/make_table2.py build/results.csv --theta 0.24 0.48 0.72
    python3 scripts/make_table2.py build/results.csv --latex table2.tex
"""

import argparse
import csv
import sys

REQUIRED_COLUMNS = ("i", "h", "nu", "U", "theta", "iterations", "rho")
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
                    "iterations": int(line["iterations"]),
                    "rho": float(line["rho"]),
                    "converged": line.get("converged", "1").strip() not in ("0", "false", "False"),
                })
    except FileNotFoundError:
        sys.exit(f"{path}: file not found")

    if not rows:
        sys.exit(f"{path}: no data")
    return rows


def select_thetas(rows, requested):
    """All theta values in the file, or the requested ones that are present."""
    available = sorted({r["theta"] for r in rows})
    if not requested:
        return available

    selected = []
    for theta in requested:
        theta = round(theta, THETA_DECIMALS)
        if theta in available:
            selected.append(theta)
        else:
            print(f"warning: theta = {theta:g} not in the data "
                  f"(available: {', '.join(f'{t:g}' for t in available)})", file=sys.stderr)
    if not selected:
        sys.exit("none of the requested theta values is in the data")
    return selected


def build_tables(rows, thetas):
    """Returns (levels, h of each level, parameter pairs, cells[(theta, nu, U, i)] = row)."""
    levels = sorted({r["i"] for r in rows})
    h_of_level = {}
    for r in rows:
        h_of_level.setdefault(r["i"], r["h"])
    params = sorted({(r["nu"], r["U"]) for r in rows})

    cells = {}
    for r in rows:
        if r["theta"] in thetas:
            cells[(r["theta"], r["nu"], r["U"], r["i"])] = r
    return levels, h_of_level, params, cells


def cell_text(row):
    """rho(N_it), with * if MINRES did not converge."""
    if row is None:
        return "-"
    return f"{row['rho']:.3f}({row['iterations']})" + ("" if row["converged"] else "*")


def print_markdown(thetas, levels, h_of_level, params, cells):
    for theta in thetas:
        print(f"\n### theta = {theta:g}\n")
        header = ["nu", "U"] + [f"h = {h_of_level[i]:.2e}" for i in levels]
        print("| " + " | ".join(header) + " |")
        print("|" + "|".join("---" for _ in header) + "|")
        for nu, U in params:
            entries = [cell_text(cells.get((theta, nu, U, i))) for i in levels]
            print("| " + " | ".join([f"{nu:g}", f"{U:g}"] + entries) + " |")
    print("\nEntries: rho(N_it) = approximate convergence factor (number of MINRES iterations); "
          "* = not converged.")


def jet_color(x):
    """Jet colormap (blue -> cyan -> green -> yellow -> red), lightened, for x in [0, 1]."""
    def clip(v):
        return max(0.0, min(1.0, v))

    rgb = (clip(1.5 - abs(4 * x - 3)), clip(1.5 - abs(4 * x - 2)), clip(1.5 - abs(4 * x - 1)))
    return tuple(0.5 + 0.5 * c for c in rgb)  # mix with white so the text stays readable


def write_latex(path, thetas, levels, h_of_level, params, cells):
    # As in the paper, the color scale goes from the minimum to the maximum rho among all tables.
    rhos = [row["rho"] for row in cells.values()]
    rho_min, rho_max = min(rhos), max(rhos)

    def color(rho):
        x = 0.0 if rho_max == rho_min else (rho - rho_min) / (rho_max - rho_min)
        return "{:.3f},{:.3f},{:.3f}".format(*jet_color(x))

    lines = [
        r"\documentclass{article}",
        r"\usepackage[table]{xcolor}",
        r"\usepackage{booktabs}",
        r"\usepackage[margin=1.5cm]{geometry}",
        r"\begin{document}",
        "",
    ]
    for theta in thetas:
        lines += [
            r"\begin{table}[ht]",
            r"\centering",
            r"\footnotesize",
            r"\caption{Approximate convergence factor $\rho$ and number of MINRES iterations "
            r"(in parentheses) w.r.t.\ the parameters $(\nu, U)$ on rows and the mesh size $h$ "
            rf"on columns, strong threshold $\theta = {theta:g}$. "
            rf"The background color depends on $\rho$, scaled between {rho_min:.3f} and {rho_max:.3f}"
            r" (all tables). $^*$ = not converged.}",
            r"\begin{tabular}{rr" + "c" * len(levels) + "}",
            r"\toprule",
            r"$\nu$ & $U \backslash h$ & " + " & ".join(f"{h_of_level[i]:.2e}" for i in levels) + r" \\",
            r"\midrule",
        ]
        for nu, U in params:
            entries = []
            for i in levels:
                row = cells.get((theta, nu, U, i))
                if row is None:
                    entries.append("--")
                else:
                    text = f"{row['rho']:.3f}({row['iterations']})" + ("" if row["converged"] else "$^*$")
                    entries.append(rf"\cellcolor[rgb]{{{color(row['rho'])}}}{text}")
            lines.append(f"{nu:g} & {U:g} & " + " & ".join(entries) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    lines.append(r"\end{document}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nLaTeX tables written to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Table 2 of the paper (rho(N_it) for each theta) from results.csv.")
    parser.add_argument("csv", nargs="?", default="results.csv", help="results.csv written by stokes (default: results.csv)")
    parser.add_argument("--theta", type=float, nargs="+", help="theta values to show (default: all), e.g. 0.24 0.48 0.72")
    parser.add_argument("--latex", metavar="FILE", help="also write the tables to a LaTeX file, colored as in the paper")
    args = parser.parse_args()

    rows = read_results(args.csv)
    thetas = select_thetas(rows, args.theta)
    levels, h_of_level, params, cells = build_tables(rows, set(thetas))

    print_markdown(thetas, levels, h_of_level, params, cells)
    if args.latex:
        write_latex(args.latex, thetas, levels, h_of_level, params, cells)


if __name__ == "__main__":
    main()

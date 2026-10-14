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
    python3 scripts/make_table2.py build/results.csv --theta 0.24 0.48 0.72 --png build/table2.png
    python3 scripts/make_table2.py build/results.csv --png-dir tables

The PNG images (--png, --png-dir) need matplotlib.
"""

import argparse
import csv
import os
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


def rho_range(cells):
    """As in the paper, the color scale goes from the minimum to the maximum rho among all tables."""
    rhos = [row["rho"] for row in cells.values()]
    return min(rhos), max(rhos)


def write_latex(path, thetas, levels, h_of_level, params, cells):
    rho_min, rho_max = rho_range(cells)

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


def write_png(path, thetas, levels, h_of_level, params, cells, rho_min, rho_max, cmap_name):
    """Draw the tables of the given thetas one below the other, as Table 2 of the paper,
    with the cell background colored by rho and a colorbar at the bottom."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colorbar import ColorbarBase
        from matplotlib.colors import ListedColormap, Normalize
        from matplotlib.patches import Rectangle
    except ImportError as error:
        sys.exit(f"--png / --png-dir need matplotlib ({error}).\n"
                 f"Install it for this interpreter ({sys.executable}) with:\n"
                 f"    {sys.executable} -m pip install --user -r requirements.txt")

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # Lighter version of the colormap (mixed with white), as the backgrounds in the paper.
    base = plt.get_cmap(cmap_name)
    cmap = ListedColormap(0.55 + 0.45 * base(np.linspace(0.0, 1.0, 256))[:, :3])
    norm = Normalize(vmin=rho_min, vmax=rho_max if rho_max > rho_min else rho_min + 1e-12)

    # Layout in inches; y grows downwards.
    row_h, gap, margin = 0.26, 0.35, 0.15
    col_theta, col_nu, col_U, col_h = 0.95, 0.65, 0.75, 1.15
    cbar_h, cbar_label_h = 0.16, 0.35
    font = 8.5

    table_h = (len(params) + 1) * row_h
    x_table = margin + col_theta
    x_end = x_table + col_nu + col_U + len(levels) * col_h
    width = x_end + margin
    height = margin + len(thetas) * table_h + (len(thetas) - 1) * gap + gap + cbar_h + cbar_label_h + margin

    fig = plt.figure(figsize=(width, height), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")

    y0 = margin
    for theta in thetas:
        # theta label on the left, as the pattern icons in the paper
        ax.text(margin + col_theta / 2, y0 + table_h / 2, rf"$\theta = {theta:g}$",
                ha="center", va="center", fontsize=font + 1)

        # header row
        y_mid = y0 + row_h / 2
        ax.text(x_table + col_nu / 2, y_mid, r"$\nu$", ha="center", va="center", fontsize=font)
        ax.text(x_table + col_nu + col_U / 2, y_mid, r"$U \backslash h$", ha="center", va="center", fontsize=font)
        for k, i in enumerate(levels):
            ax.text(x_table + col_nu + col_U + (k + 0.5) * col_h, y_mid, f"{h_of_level[i]:.2e}",
                    ha="center", va="center", fontsize=font)

        # body
        for r, (nu, U) in enumerate(params):
            y = y0 + (r + 1) * row_h
            ax.text(x_table + col_nu / 2, y + row_h / 2, f"{nu:g}", ha="center", va="center", fontsize=font)
            ax.text(x_table + col_nu + col_U / 2, y + row_h / 2, f"{U:g}", ha="center", va="center", fontsize=font)
            for k, i in enumerate(levels):
                x = x_table + col_nu + col_U + k * col_h
                row = cells.get((theta, nu, U, i))
                if row is not None:
                    ax.add_patch(Rectangle((x, y), col_h, row_h, facecolor=cmap(norm(row["rho"])),
                                           edgecolor="white", linewidth=0.6))
                ax.text(x + col_h / 2, y + row_h / 2, cell_text(row), ha="center", va="center", fontsize=font)

        # rules (booktabs style)
        ax.plot([x_table, x_end], [y0, y0], color="black", linewidth=1.0)
        ax.plot([x_table, x_end], [y0 + row_h, y0 + row_h], color="black", linewidth=0.6)
        ax.plot([x_table, x_end], [y0 + table_h, y0 + table_h], color="black", linewidth=1.0)

        y0 += table_h + gap

    # colorbar, below the tables and aligned with the rho columns
    x_cb = x_table + col_nu + col_U
    cax = fig.add_axes([x_cb / width, (height - y0 - cbar_h) / height, (x_end - x_cb) / width, cbar_h / height])
    cbar = ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")
    cbar.set_label(r"$\rho$", fontsize=font + 1)
    cbar.ax.tick_params(labelsize=font - 1)

    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"PNG written to {path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Table 2 of the paper (rho(N_it) for each theta) from results.csv.")
    parser.add_argument("csv", nargs="?", default="results.csv", help="results.csv written by stokes (default: results.csv)")
    parser.add_argument("--theta", type=float, nargs="+", help="theta values to show (default: all), e.g. 0.24 0.48 0.72")
    parser.add_argument("--latex", metavar="FILE", help="also write the tables to a LaTeX file, colored as in the paper")
    parser.add_argument("--png", metavar="FILE", help="also draw the tables of the selected theta values, one below the other, in a PNG image")
    parser.add_argument("--png-dir", metavar="DIR", help="also draw one PNG image per theta in DIR (table2_theta<theta>.png)")
    parser.add_argument("--cmap", default="rainbow", help="matplotlib colormap for the PNG images (default: rainbow)")
    args = parser.parse_args()

    rows = read_results(args.csv)
    thetas = select_thetas(rows, args.theta)
    levels, h_of_level, params, cells = build_tables(rows, set(thetas))

    print_markdown(thetas, levels, h_of_level, params, cells)
    if args.latex:
        write_latex(args.latex, thetas, levels, h_of_level, params, cells)

    # Same color scale for all the images: min/max rho among the selected tables.
    rho_min, rho_max = rho_range(cells)
    if args.png:
        if len(thetas) > 6:
            print(f"warning: {len(thetas)} tables in one image; use --theta to select fewer "
                  "(e.g. --theta 0.24 0.48 0.72 as in the paper)", file=sys.stderr)
        write_png(args.png, thetas, levels, h_of_level, params, cells, rho_min, rho_max, args.cmap)
    if args.png_dir:
        os.makedirs(args.png_dir, exist_ok=True)
        for theta in thetas:
            write_png(os.path.join(args.png_dir, f"table2_theta{theta:g}.png"),
                      [theta], levels, h_of_level, params, cells, rho_min, rho_max, args.cmap)


if __name__ == "__main__":
    main()

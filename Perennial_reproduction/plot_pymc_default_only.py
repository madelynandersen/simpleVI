#!/usr/bin/env python3
"""Default-only trajectory plots from saved PyMC ADVI trajectories (no refitting).

Reads extra_figs/pymc_trajectory_vi_default<suffix>.csv (written by pymc_reproducing.py) and draws the same two 2 x 4
figures as pymc_reproducing.py, but for the default ADVI run only:
  extra_figs/pymc_trajectories_means<suffix>_default_only.png   mean of each coefficient vs iteration
  extra_figs/pymc_trajectories_sd<suffix>_default_only.png      posterior SD of each coefficient vs iteration
Reference lines: R NUTS (TableS1_reproduced.csv) dashed black, best VB approximation (TableS1_best_vi.csv) solid black;
a dashed vertical line at 10,000 iterations is labelled "default number of iterations".

Run from the Perennial_reproduction folder:
  python plot_pymc_default_only.py                    # the 150k run
  python plot_pymc_default_only.py --suffix _75k      # another iteration count
  python plot_pymc_default_only.py --suffix ""        # the 10k run
Requires numpy, pandas, matplotlib.
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TERM_ORDER = ["muPrB_Intercept", "muPrB_MAT", "muPrB_polyMAP21", "muPrB_polyMAP22",
              "muDeB_Intercept", "muDeB_MAT", "muDeB_polyMAP21", "muDeB_polyMAP22"]
TERM_TITLES = {"muPrB_Intercept": "Precipitation: Intercept", "muPrB_MAT": "Precipitation: MAT",
               "muPrB_polyMAP21": "Precipitation: MAP", "muPrB_polyMAP22": "Precipitation: MAP²",
               "muDeB_Intercept": "Moisture deficit: Intercept", "muDeB_MAT": "Moisture deficit: MAT",
               "muDeB_polyMAP21": "Moisture deficit: MAP", "muDeB_polyMAP22": "Moisture deficit: MAP²"}
DEFAULT_ITERS = 10000
RED = "#E41A1C"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suffix", default="_150k", help='file suffix of the run, e.g. "_150k" (default), "_75k" or ""')
    ap.add_argument("--outdir", default="extra_figs")
    args = ap.parse_args()
    sfx = args.suffix
    traj_file = Path(args.outdir) / f"pymc_trajectory_vi_default{sfx}.csv"
    if not traj_file.exists():
        sys.exit(f"{traj_file} not found - run this from the Perennial_reproduction folder")

    df = pd.read_csv(traj_file)
    n_iter = int(df["iteration"].max())
    seed_file = Path(f"TableS1_py_vi_default{sfx}.csv")
    seed = int(pd.read_csv(seed_file)["seed"].iloc[0]) if seed_file.exists() else None
    nuts = pd.read_csv("TableS1_reproduced.csv").set_index("term") if Path("TableS1_reproduced.csv").exists() else None
    best = pd.read_csv("TableS1_best_vi.csv").set_index("term") if Path("TableS1_best_vi.csv").exists() else None
    if nuts is None:
        print("Note: TableS1_reproduced.csv not found - no R NUTS reference line")
    if best is None:
        print("Note: TableS1_best_vi.csv not found - no best-VB reference line")
    plt.rcParams.update({"font.size": 13})
    run_label = "PyMC ADVI default" + (f" (seed {seed})" if seed is not None else "")

    def grid(suffix_col, ref_col, ylabel, title, fname):
        fig, axes = plt.subplots(2, 4, figsize=(20, 8.5))
        for ax, term in zip(axes.ravel(), TERM_ORDER):
            ax.plot(df["iteration"], df[f"{term}_{suffix_col}"], color=RED, lw=1.5, label=run_label)
            if nuts is not None and term in nuts.index:
                ax.axhline(nuts.loc[term, ref_col], color="black", ls=(0, (3, 3)), lw=2, label="R NUTS (reference)")
            if best is not None and term in best.index:
                ax.axhline(best.loc[term, ref_col], color="black", ls="-", lw=2, label="best VB approx")
            if n_iter > DEFAULT_ITERS:
                ax.axvline(DEFAULT_ITERS, color="grey", ls="--", lw=1.5)
                ax.text(DEFAULT_ITERS, 0.97, " default number\n of iterations", transform=ax.get_xaxis_transform(),
                        va="top", ha="left", fontsize=10, color="dimgrey")
            ax.set_title(TERM_TITLES[term], fontsize=13, fontweight="bold")
            ax.set_xlabel("ADVI iteration")
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x / 1000:g}k"))
        axes[0, 0].set_ylabel(ylabel)
        axes[1, 0].set_ylabel(ylabel)
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=12)
        fig.suptitle(title, fontsize=15)
        fig.tight_layout(rect=(0, 0.05, 1, 0.97))
        out = Path(args.outdir) / fname
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print("Wrote", out)

    grid("mean", "Estimate", "Mean (logit scale)",
         f"PyMC default ADVI: trajectories of the Table S1 coefficient means ({n_iter:,} iterations)",
         f"pymc_trajectories_means{sfx}_default_only.png")
    grid("sd", "Est.Error", "Posterior SD (logit scale)",
         f"PyMC default ADVI: trajectories of the Table S1 coefficient SDs ({n_iter:,} iterations)",
         f"pymc_trajectories_sd{sfx}_default_only.png")


if __name__ == "__main__":
    main()

"""Seed spread of default PyMC ADVI: refit with 50 seeds (default settings, no fallback to 'first success'),
save each fit's summary, then plot the spread of each coefficient's mean against R NUTS and the best VB approx.
Run from the Perennial_reproduction folder:  python seed_spread.py            (fit + plot)
                                             python seed_spread.py --plot     (plot only, from saved csv)
Writes seed_spread.csv and extra_D_seed_spread.png.  UNTESTED where it was written (no PyMC there): run a quick
check with  --n 3  first.  A seed-11 run is highlighted: it is the one reported in TableS1_py_vi_default.csv."""
import argparse
import numpy as np, pandas as pd
import pymc_reproducing as pr

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=50)
ap.add_argument("--plot", action="store_true")
ap.add_argument("--highlight", type=int, default=11)
a = ap.parse_args()

if not a.plot:
    import pymc as pm
    model = pr.build_model(pr.load_data())
    rows = []
    for s in range(1, a.n + 1):
        try:
            with model:
                approx = pm.fit(method="advi", random_seed=s, progressbar=False)     # PyMC defaults
            post = approx.sample(draws=10000, random_seed=s).posterior
            t = pr.summary_table(pr.term_draws(post), s, "vi_default_seed")
            rows.append(t); print("seed", s, "done")
        except Exception as e:                                                          # noqa: BLE001
            print("seed", s, "failed:", e)
    pd.concat(rows).to_csv("seed_spread.csv", index=False)

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sp = pd.read_csv("seed_spread.csv")
ref = pd.read_csv("TableS1_reproduced.csv").set_index("term")
best = pd.read_csv("TableS1_best_vi.csv").set_index("term") if __import__("os").path.exists("TableS1_best_vi.csv") else None
terms = list(ref.index)
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, t in zip(axes.ravel(), terms):
    off = (sp[sp.term == t].set_index("seed").Estimate - ref.loc[t, "Estimate"]) / ref.loc[t, "Est.Error"]
    ax.axhspan(-1, 1, color="grey", alpha=0.2); ax.axhline(0, color="black", lw=1.5)
    if best is not None:
        ax.axhline((best.loc[t, "Estimate"] - ref.loc[t, "Estimate"]) / ref.loc[t, "Est.Error"],
                   color="#762A83", lw=2, ls="--", label="best VB approx")
    ax.boxplot(off.values, widths=0.5, showfliers=False)
    ax.scatter(np.random.default_rng(0).normal(1, 0.06, len(off)), off.values, s=18, color="#9ECAE1", zorder=3)
    if a.highlight in off.index:
        ax.scatter([1.25], [off[a.highlight]], s=110, color="#E41A1C", zorder=4,
                   label=f"seed {a.highlight} (reported)")
    ax.set_title(t, fontsize=13); ax.set_xticks([])
axes[0, 0].legend(fontsize=10)
fig.supylabel("Mean offset from R NUTS (NUTS SDs)", fontsize=14)
fig.suptitle("Default PyMC ADVI across seeds; grey band = within 1 NUTS SD", fontsize=16)
fig.tight_layout(); fig.savefig("extra_D_seed_spread.png", dpi=300)
print("wrote extra_D_seed_spread.png")

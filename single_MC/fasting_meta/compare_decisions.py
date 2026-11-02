"""
Compare the authors' decision rule (95% HDI excludes 0, as reported in the
paper; set LO, HI to "q2.5", "q97.5" for equal-tailed intervals) between
brms/NUTS, PyMC default ADVI (many seeds), and the best PyMC ADVI fit (100 MC draws, Adam, 100k iters; highest-ELBO restart).

Writes results/decisions.csv and results/sd_comparison.csv.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from modulars.fasting_meta import load_fasting_brms_draws, summarize_fasting_draws  # noqa: E402


LO, HI = "hdi_lo", "hdi_hi"


def decide(lo, hi):
    return "+" if lo > 0 else ("-" if hi < 0 else "0")


def load_vi(results_dir):
    runs = []
    for p in sorted((results_dir / "vi").glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        for row in r["summary"]:
            runs.append({"model": r["model"], "config": r["config"], "seed": r["seed"],
                         "elbo": r["elbo"], **row})
    return pd.DataFrame(runs)


def main(results_dir=HERE / "results"):
    vi = load_vi(results_dir)
    dec_rows, sd_rows = [], []
    for model in vi["model"].unique():
        mcmc = summarize_fasting_draws(load_fasting_brms_draws(model)).set_index("variable")
        long_path = results_dir / "brms_long" / f"draws_{model}.csv"
        mcmc_long = (summarize_fasting_draws(load_fasting_brms_draws(model, subdir="brms_long")).set_index("variable")
                     if long_path.exists() else None)
        v = vi[vi["model"] == model]
        best = v[v["config"] == "best"]
        best_seed = best.groupby("seed")["elbo"].first().idxmax() if len(best) else None
        default = v[v["config"] == "default"]
        for var in mcmc.index:
            m = mcmc.loc[var]
            d = default[default["variable"] == var]
            b = best[(best["variable"] == var) & (best["seed"] == best_seed)]
            row = {"model": model, "variable": var,
                   "mcmc_mean": m["mean"], "mcmc_lo": m[LO], "mcmc_hi": m[HI]}
            if var.startswith("sd_"):
                row.update({"default_mean_med": d["mean"].median(), "best_mean": b["mean"].iloc[0] if len(b) else np.nan,
                            "mcmc_sd": m["sd"], "default_sd_med": d["sd"].median(),
                            "best_sd": b["sd"].iloc[0] if len(b) else np.nan})
                sd_rows.append(row)
                continue
            mcmc_dec = decide(m[LO], m[HI])
            default_decs = [decide(lo, hi) for lo, hi in zip(d[LO], d[HI])]
            row.update({
                "mcmc_dec": mcmc_dec,
                "mcmc_long_dec": decide(mcmc_long.loc[var, LO], mcmc_long.loc[var, HI]) if mcmc_long is not None else None,
                "n_default": len(default_decs),
                "default_flip_rate": np.mean([x != mcmc_dec for x in default_decs]) if default_decs else np.nan,
                "default_decs": "".join(default_decs),
                "default_lo_med": d[LO].median(), "default_hi_med": d[HI].median(),
                "best_seed": best_seed,
                "best_dec": decide(b[LO].iloc[0], b[HI].iloc[0]) if len(b) else None,
                "best_lo": b[LO].iloc[0] if len(b) else np.nan,
                "best_hi": b[HI].iloc[0] if len(b) else np.nan,
                "best_decs_all_restarts": "".join(decide(lo, hi) for lo, hi in
                                                  zip(best[best["variable"] == var][LO],
                                                      best[best["variable"] == var][HI])),
            })
            row["best_flip"] = (row["best_dec"] != mcmc_dec) if row["best_dec"] else np.nan
            dec_rows.append(row)

    dec = pd.DataFrame(dec_rows)
    sds = pd.DataFrame(sd_rows)
    dec.to_csv(results_dir / "decisions.csv", index=False)
    sds.to_csv(results_dir / "sd_comparison.csv", index=False)
    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_columns", 30)
    print(dec[["model", "variable", "mcmc_lo", "mcmc_hi", "mcmc_dec", "mcmc_long_dec", "default_flip_rate",
               "default_decs", "best_lo", "best_hi", "best_dec", "best_decs_all_restarts"]].round(4))
    print(sds[["model", "variable", "mcmc_mean", "default_mean_med", "best_mean",
               "mcmc_sd", "default_sd_med", "best_sd"]].round(3))
    return dec, sds


if __name__ == "__main__":
    main()

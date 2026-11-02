"""
Fit the fasting meta-analysis models with PyMC mean-field ADVI.

  default: pm.fit() defaults (10k iters, 1 MC draw, adagrad_window lr=1e-3)
  best:    100k iters, 100 MC draws, Adam (lr=1e-3)

Each run writes results/vi/<model>__<config>__seed<seed>.json with brms-named
summaries, a fixed-draw ELBO estimate, and a thinned loss trace.

Usage: python run_vi.py --models main fdur --configs default best --seeds 0 1 2 --procs 12
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

ALL_MODELS = [
    "main", "fdur", "fdur_r", "fdur_quad", "ToD", "ToD_fdur_inter", "cog", "food",
    "nofood_and_neutral", "foodmixed_and_foodonly", "study_type", "study_type_rm_removed",
    "risk", "age", "children", "adult", "BMI", "female", "isRT", "year",
    "multi", "multi_fdur_food_stim", "multi_cog_food_stim",
]

CONFIGS = {
    "default": {},
    "best": {"n": 100_000, "obj_n_mc": 100, "adam": True},
}


def run_one(job):
    model_name, config, seed = job
    out = HERE / "results" / "vi" / f"{model_name}__{config}__seed{seed}.json"
    if out.exists():
        return str(out)

    import numpy as np
    import modulars.pymc_rr_test  # noqa: F401  (sets the pytensor compiler on macOS)
    import pymc as pm
    from modulars.fasting_meta import (
        build_pymc_fasting_model, load_fasting_standata, meanfield_elbo,
        pymc_approx_named_draws, summarize_fasting_draws,
    )

    data = load_fasting_standata(model_name)
    model = build_pymc_fasting_model(data)
    cfg = dict(CONFIGS[config])
    fit_kwargs = {}
    if "n" in cfg:
        fit_kwargs["n"] = cfg["n"]
    if "obj_n_mc" in cfg:
        fit_kwargs["obj_n_mc"] = cfg["obj_n_mc"]
    if cfg.get("adam"):
        fit_kwargs["obj_optimizer"] = pm.adam()

    t0 = time.time()
    approx = pm.fit(method="advi", model=model, random_seed=seed, progressbar=False, **fit_kwargs)
    elapsed = time.time() - t0
    hist = np.asarray(approx.hist)
    elbo, elbo_se = meanfield_elbo(model, approx, n_draws=2000, seed=10_000 + seed)
    draws = pymc_approx_named_draws(approx, data, n_draws=8000, seed=20_000 + seed)
    summary = summarize_fasting_draws(draws)

    stride = max(1, len(hist) // 1000)
    result = {
        "model": model_name, "config": config, "seed": seed, "fit_kwargs": {k: str(v) for k, v in fit_kwargs.items()},
        "elapsed_s": elapsed, "elbo": elbo, "elbo_se": elbo_se,
        "final_avg_loss": float(np.nanmean(hist[-max(1, len(hist) // 10):])),
        "loss_trace_stride": stride, "loss_trace": hist[::stride].tolist(),
        "summary": summary.to_dict(orient="records"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f)
    return str(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(20)))
    parser.add_argument("--procs", type=int, default=12)
    args = parser.parse_args()

    # longest jobs first so the pool stays busy
    jobs = [(m, c, s) for c in sorted(args.configs, key=lambda c: c != "best")
            for m in args.models for s in args.seeds]
    with get_context("spawn").Pool(args.procs) as pool:
        for i, path in enumerate(pool.imap_unordered(run_one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {path}", flush=True)

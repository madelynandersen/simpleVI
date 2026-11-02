"""Shared pieces of the --restarts mode of pymc_reproducing.py, numpyro_tfp_reproducing.py and stan_reproducing.py.

--restarts runs random restarts of every ADVI setup of one package on the Table S1 model and saves the
last `window` iterations of the mean and SD of the 8 Table S1 coefficients, in the layout read by
paper_vignettes/overall_bias_comparisons.ipynb (model key "journe", through
paper_vignettes/overall_bias_runners/bias_io.py):

    <results_dir>/journe/<package>_tracking.csv.gz, <package>_reference.csv, <package>_run_info.json

Finished seeds are cached under <results_dir>/_seed_cache/journe/<package>/, so a rerun resumes.

--restarts --full_defaults runs only each package's full-default fit (setup "full_default", the package's default
number of iterations, no changed settings) into overall_bias_results_full_default/ for
paper_vignettes/overall_default_bias_comparisons.ipynb, with every seed's fit time (seed_seconds in run_info.json). The reference
for every package is the best variational approximation in TableS1_best_vi.csv (best of the Stan + PyMC
restarts ranked by NumPyro ELBO, see best_vi.py).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "paper_vignettes" / "overall_bias_runners"))
import bias_io  # noqa: E402

from pymc_reproducing import TERMS  # noqa: E402  (importing it does not import pymc)

MODEL = "journe"
BEST_FILE = HERE / "TableS1_best_vi.csv"
COORDINATES_NOTE = ("coordinates are the 8 Table S1 fixed effects (intercept, MAT, MAP, MAP^2 for muPrB and muDeB); "
                    "the group-level sd / z and phi are not tracked")


def add_restart_args(ap, default_seeds, seeds_help=None):
    g = ap.add_argument_group("random restarts for paper_vignettes/overall_bias_comparisons.ipynb")
    g.add_argument("--restarts", action="store_true",
                   help="run --n-seeds random restarts of every setup and save the last --window iterations of the "
                        "8 Table S1 coefficients' mean and SD (instead of the Table S1 fits)")
    g.add_argument("--n-seeds", type=int, default=default_seeds, help=f"restarts per setup (default {seeds_help or default_seeds})")
    g.add_argument("--window", type=int, default=500, help="final iterations tracked at every iteration (default 500)")
    g.add_argument("--overwrite", type=int, choices=(0, 1), default=0,
                   help="1 reruns and overwrites saved restarts (and ignores the seed cache)")
    g.add_argument("--results-dir", default=None,
                   help=f"default {bias_io.DEFAULT_RESULTS_DIR} ({bias_io.FULL_DEFAULT_RESULTS_DIR} with --full_defaults)")
    g.add_argument("--track-every", type=int, default=1,
                   help="keep every n-th iteration of the window (default 1; stan_reproducing.py: 10, i.e. 50 reruns)")
    g.add_argument("--max-workers", type=int, default=None, help="(stan only) parallel CmdStan workers; default cores - 1")
    g.add_argument("--single-setup", default=None,
                   help="(tfp, stan) run only the 1 MC 'default' setup and save it as this setup name (e.g. default_10k) "
                        f"to {bias_io.SINGLE_SETUP_RESULTS_DIR.name}/ unless --results-dir is given")


def restart_args(args, max_iters):
    """The namespace the bias_io helpers expect (model, max_iters, n_seeds, window, overwrite, results_dir)."""
    full_default = bool(getattr(args, "full_defaults", False))
    results_dir = args.results_dir or (bias_io.FULL_DEFAULT_RESULTS_DIR if full_default
                                       else bias_io.SINGLE_SETUP_RESULTS_DIR if args.single_setup
                                       else bias_io.DEFAULT_RESULTS_DIR)
    out = argparse.Namespace(model=MODEL, max_iters=int(max_iters), n_seeds=int(args.n_seeds), window=int(args.window),
                             overwrite=bool(args.overwrite), results_dir=Path(results_dir).resolve(),
                             max_workers=args.max_workers, track_every=int(args.track_every), full_default=full_default,
                             single_setup=args.single_setup)
    bias_io.window_iterations(out.max_iters, out.window, out.track_every)  # validates the window
    return out


def term_matrix(traj):
    """{Table S1 term: trajectory (T,)} -> (T, 8) array in TERMS order."""
    return np.stack([np.asarray(traj[t], dtype=float) for t in TERMS], axis=-1)


def best_reference():
    best = pd.read_csv(BEST_FILE).set_index("term").loc[TERMS]
    source = (f"Perennial_reproduction/{BEST_FILE.name} ({best['method'].iloc[0]}; best of the Stan + PyMC restarts "
              f"ranked by NumPyro ELBO), shared by all four packages")
    return best["Estimate"].to_numpy(float), best["Est.Error"].to_numpy(float), source


def run_and_save(rargs, package, setup_names, run_one_seed, extra_info=None):
    """run_one_seed(seed) -> [mean, std] per setup in setup_names order, each (window, 8); NaN for a failed fit."""
    if bias_io.already_saved(rargs, package):
        return
    seeds = list(range(rargs.n_seeds))
    per_seed = bias_io.run_seeds_cached(rargs, package, seeds, run_one_seed)
    setups = {name: (np.stack([arrays[2 * k] for arrays in per_seed]), np.stack([arrays[2 * k + 1] for arrays in per_seed]))
              for k, name in enumerate(setup_names)}
    save(rargs, package, setups, seeds, extra_info)


def save(rargs, package, setups, seeds, extra_info=None):
    best_mean, best_std, source = best_reference()
    n_failed = {name: int((~np.isfinite(means).all(axis=(1, 2))).sum()) for name, (means, _) in setups.items()}
    info = bias_io.base_run_info(rargs, package, seeds, {"coordinates_note": COORDINATES_NOTE,
                                                         "seeds_with_nonfinite_values": n_failed, **(extra_info or {})})
    iterations = bias_io.window_iterations(rargs.max_iters, rargs.window, rargs.track_every)
    bias_io.save_model_outputs(rargs, package, MODEL, setups, iterations,
                               TERMS, best_mean, best_std, source, info)


def nan_window(rargs):
    return np.full((len(bias_io.window_iterations(rargs.max_iters, rargs.window, rargs.track_every)), len(TERMS)), np.nan)

"""Shared helpers for the overall bias comparison runners and notebook.

Only numpy / pandas are imported here so this module works in every package's
conda environment (the runners) and in whatever environment the
overall_bias_comparisons notebook is run in (reading + plotting).

Saved layout, one folder per model under the results directory:

    <results_dir>/<model>/<package>_tracking.csv.gz
        wide csv, one row per (setup, statistic, seed, coordinate), one column
        per tracked iteration (the last `window` iterations of the run)
    <results_dir>/<model>/<package>_reference.csv
        best mean / best std per coordinate, exactly as that package's
        single_MC random-restart notebook computes them
    <results_dir>/<model>/<package>_run_info.json
        settings, seeds, timing, package versions and any failures
"""

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS_DIR = REPO_ROOT / "paper_vignettes" / "overall_bias_results"
DEFAULT_MAX_ITERS = 50_000

# --full-default: each package's out-of-the-box fit (1 MC sample, no changed settings), saved as setup
# "full_default" in its own results folder for overall_default_bias_comparisons.ipynb
FULL_DEFAULT_SETUP = "full_default"
FULL_DEFAULT_RESULTS_DIR = REPO_ROOT / "paper_vignettes" / "overall_bias_results_full_default"
# each package's default number of iterations: pm.fit's n and CmdStan's iter (10,000); NumPyro and TFP have no
# default, so the 30,000 and 10,000 we use as their defaults
FULL_DEFAULT_ITERS = {"pymc": 10_000, "numpyro": 30_000, "tfp": 10_000, "stan": 10_000}
# --single-setup NAME: only a runner's 1 MC "default" setup (every setting as in the main runs except --max-iters),
# saved as setup NAME in its own results folder, e.g. TFP for 10,000 or Stan for 50,000 iterations
SINGLE_SETUP_RESULTS_DIR = REPO_ROOT / "paper_vignettes" / "overall_bias_results_1mc"

# model keys used for folder names; display labels live in the notebook config
MODEL_KEYS = (
    "1d_gaussian",
    "40d_gaussian",
    "beta_binomial",
    "gamma_exponential",
    "gamma_poisson",
    "inverse_gamma",
    "dirichlet_multinomial",
    "logistic_regression",
    "radon",
)
# radon is run once and saved on two scales
OUTPUT_MODEL_KEYS = {"radon": ("radon_constrained", "radon_log")}
# models saved in the same layout by scripts outside these runners (not run by run_all_runners.sh):
#   journe  Perennial_reproduction/{pymc,numpyro_tfp,stan}_reproducing.py --restarts
EXTERNAL_MODEL_KEYS = ("journe",)
# every model the notebook reads results for
RESULT_MODEL_KEYS = MODEL_KEYS + EXTERNAL_MODEL_KEYS

# setup keys saved by each package's runner, in column order
PACKAGE_SETUPS = {
    "pymc": ("default", "100mc", "adam", "adam_100mc"),
    "tfp": ("default", "100mc"),
    "numpyro": ("default", "100mc"),
    "stan": ("default", "100mc"),
}

ID_COLUMNS = ["setup", "statistic", "seed", "coordinate"]


def output_models(model):
    return OUTPUT_MODEL_KEYS.get(model, (model,))


def tracking_path(results_dir, model, package):
    return Path(results_dir) / model / f"{package}_tracking.csv.gz"


def reference_path(results_dir, model, package):
    return Path(results_dir) / model / f"{package}_reference.csv"


def run_info_path(results_dir, model, package):
    return Path(results_dir) / model / f"{package}_run_info.json"


def seed_cache_dir(results_dir, model, package):
    return Path(results_dir) / "_seed_cache" / model / package


def window_iterations(max_iters, window, track_every=1):
    """1-indexed iteration numbers of the tracked tail window.

    track_every > 1 keeps every track_every-th iteration of the window, ending at max_iters
    (the iterations modulars.stan_rr_test.tracked_iterations gives with min_iter = the window start).
    """
    max_iters, window, track_every = int(max_iters), int(window), int(track_every)
    if not 1 <= window <= max_iters:
        raise ValueError(f"window must be between 1 and max_iters ({max_iters}), got {window}")
    if not 1 <= track_every <= window or window % track_every:
        raise ValueError(f"track_every must divide the window ({window}), got {track_every}")
    return np.arange(max_iters - window + track_every, max_iters + 1, track_every)


def track_start(max_iters, window):
    return int(window_iterations(max_iters, window)[0])


"""
Command line handling shared by every runner
"""
def parse_runner_args(package, argv=None):
    parser = argparse.ArgumentParser(description=f"Run the {package} random restarts for the overall bias comparison.")
    parser.add_argument("--model", required=True, choices=MODEL_KEYS)
    parser.add_argument("--max-iters", type=int, default=None,
                        help=f"default {DEFAULT_MAX_ITERS:,}; with --full-default 1 the package's default "
                             f"({FULL_DEFAULT_ITERS[package]:,})")
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--window", type=int, default=500,
                        help="number of final iterations tracked at every iteration")
    parser.add_argument("--overwrite", type=int, choices=(0, 1), default=0,
                        help="1 reruns and overwrites saved results, 0 skips models already saved")
    parser.add_argument("--results-dir", default=None,
                        help=f"default {DEFAULT_RESULTS_DIR} ({FULL_DEFAULT_RESULTS_DIR} with --full-default 1)")
    parser.add_argument("--full-default", type=int, choices=(0, 1), default=0,
                        help="1 runs only the package's full-default fit (setup 'full_default'), timing every seed")
    parser.add_argument("--single-setup", default=None,
                        help="(tfp, stan) run only the 1 MC 'default' setup and save it as this setup name "
                             f"(e.g. default_10k) to {SINGLE_SETUP_RESULTS_DIR.name}/ unless --results-dir is given")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="(stan only) parallel CmdStan workers; default is cores - 1")
    args = parser.parse_args(argv)
    args.overwrite = bool(args.overwrite)
    args.full_default = bool(args.full_default)
    if args.max_iters is None:
        args.max_iters = FULL_DEFAULT_ITERS[package] if args.full_default else DEFAULT_MAX_ITERS
    if args.full_default and args.single_setup:
        parser.error("--full-default and --single-setup are separate runs; give one of them")
    if args.results_dir is None:
        args.results_dir = (FULL_DEFAULT_RESULTS_DIR if args.full_default
                            else SINGLE_SETUP_RESULTS_DIR if args.single_setup else DEFAULT_RESULTS_DIR)
    args.results_dir = Path(args.results_dir).resolve()
    window_iterations(args.max_iters, args.window)  # validates the window
    return args


def already_saved(args, package):
    paths = [tracking_path(args.results_dir, m, package) for m in output_models(args.model)]
    if all(p.exists() for p in paths) and not args.overwrite:
        print(f"[{package}/{args.model}] results already saved ({', '.join(map(str, paths))}); "
              f"skipping. Pass --overwrite 1 to rerun.")
        return True
    return False


"""
Per-seed caching so a crash part way through a long run only loses one seed
"""
def _cache_settings(args):
    settings = {"max_iters": int(args.max_iters), "window": int(args.window)}
    if int(getattr(args, "track_every", 1)) != 1:   # only when thinned, so caches of every-iteration runs stay valid
        settings["track_every"] = int(args.track_every)
    return settings


def load_cached_seed(args, package, seed):
    path = seed_cache_dir(args.results_dir, args.model, package) / f"seed_{seed:03d}.npz"
    if args.overwrite or not path.exists():
        return None
    with np.load(path, allow_pickle=False) as cached:
        settings = json.loads(str(cached["settings"]))
        if settings != _cache_settings(args):
            return None
        n_arrays = int(cached["n_arrays"])
        if "seconds" in cached.files:
            record_seed_seconds(args, seed, float(cached["seconds"]))
        return [cached[f"arr_{i}"] for i in range(n_arrays)]


def record_seed_seconds(args, seed, seconds):
    """Keep the wall-clock time of a seed's run on args (saved as seed_seconds in run_info.json)."""
    if not hasattr(args, "seed_seconds"):
        args.seed_seconds = {}
    args.seed_seconds[int(seed)] = float(seconds)


def save_cached_seed(args, package, seed, arrays, seconds=None):
    directory = seed_cache_dir(args.results_dir, args.model, package)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"seed_{seed:03d}.npz"
    payload = {f"arr_{i}": np.asarray(a) for i, a in enumerate(arrays)}
    payload["n_arrays"] = np.asarray(len(arrays))
    payload["settings"] = np.asarray(json.dumps(_cache_settings(args)))
    if seconds is not None:
        payload["seconds"] = np.asarray(float(seconds))
        record_seed_seconds(args, seed, seconds)
    tmp_path = path.with_name(path.stem + "_tmp.npz")
    np.savez(tmp_path, **payload)
    os.replace(tmp_path, path)


def run_seeds_cached(args, package, seeds, run_one_seed):
    """Run `run_one_seed(seed) -> list of arrays` for every seed, reusing cached seeds.

    Each seed's wall-clock time is cached with it and saved to run_info.json as seed_seconds (with --full-default
    a seed is one fit, so these are the fit times).
    """
    out = []
    for i, seed in enumerate(seeds):
        cached = load_cached_seed(args, package, seed)
        if cached is not None:
            print(f"[{package}/{args.model}] seed {seed}: loaded from cache", flush=True)
            out.append(cached)
            continue
        start = time.time()
        arrays = [np.asarray(a) for a in run_one_seed(seed)]
        seconds = time.time() - start
        save_cached_seed(args, package, seed, arrays, seconds=seconds)
        print(f"[{package}/{args.model}] seed {seed} ({i + 1}/{len(seeds)}) done in {seconds:.1f}s", flush=True)
        out.append(arrays)
    return out


"""
Saving / loading the windowed trajectories and reference values
"""
def _as_seed_iter_coord(arr, n_iters):
    """Return arr as (n_seeds, n_iters, n_coords)."""
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim != 3 or arr.shape[1] != n_iters:
        raise ValueError(f"expected (n_seeds, {n_iters}, n_coords) trajectories, got {arr.shape}")
    return arr


def save_tracking(path, setups, iterations, coordinates, float_format="%.9g"):
    """Save {setup: (means, stds)} trajectories, each (n_seeds, n_iters[, n_coords])."""
    iterations = np.asarray(iterations, dtype=int)
    coordinates = [str(c) for c in coordinates]
    frames = []
    for setup, (means, stds) in setups.items():
        for statistic, values in (("mean", means), ("std", stds)):
            values = _as_seed_iter_coord(values, len(iterations))
            if values.shape[2] != len(coordinates):
                raise ValueError(f"{setup}/{statistic}: {values.shape[2]} coordinates but {len(coordinates)} names")
            n_seeds, _, n_coords = values.shape
            block = values.transpose(0, 2, 1).reshape(n_seeds * n_coords, len(iterations))
            ids = pd.DataFrame({
                "setup": setup,
                "statistic": statistic,
                "seed": np.repeat(np.arange(n_seeds), n_coords),
                "coordinate": np.tile(coordinates, n_seeds),
            })
            frames.append(pd.concat([ids, pd.DataFrame(block, columns=[str(i) for i in iterations])], axis=1))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    pd.concat(frames, ignore_index=True).to_csv(tmp_path, index=False, float_format=float_format, compression="gzip")
    os.replace(tmp_path, path)


def load_tracking(path):
    """Load a tracking csv into {setup: {"mean": arr, "std": arr}}, iterations, coordinates.

    arrays are (n_seeds, n_iters, n_coords) with coordinates in saved order.
    """
    frame = pd.read_csv(path, compression="gzip")
    iteration_cols = [c for c in frame.columns if c not in ID_COLUMNS]
    iterations = np.asarray(iteration_cols, dtype=int)
    coordinates = list(dict.fromkeys(frame["coordinate"].astype(str)))
    out = {}
    for (setup, statistic), block in frame.groupby(["setup", "statistic"], sort=False):
        n_seeds = block["seed"].nunique()
        # sort rows by (seed, coordinate in saved order) without adding a column to the wide frame
        coord_index = pd.Categorical(block["coordinate"].astype(str), categories=coordinates).codes
        order = np.lexsort((coord_index, block["seed"].to_numpy()))
        values = block[iteration_cols].to_numpy(dtype=float)[order]
        values = values.reshape(n_seeds, len(coordinates), len(iterations)).transpose(0, 2, 1)
        out.setdefault(setup, {})[statistic] = values
    return out, iterations, coordinates


def save_reference(path, coordinates, best_mean, best_std, source):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "coordinate": [str(c) for c in coordinates],
        "best_mean": np.atleast_1d(np.asarray(best_mean, dtype=float)),
        "best_std": np.atleast_1d(np.asarray(best_std, dtype=float)),
        "source": source,
    }).to_csv(path, index=False, float_format="%.17g")


def load_reference(path):
    return pd.read_csv(path)


def save_run_info(path, info):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2, default=str)


def load_run_info(path):
    with open(path) as f:
        return json.load(f)


def base_run_info(args, package, seeds, extra=None):
    info = {
        "package": package,
        "model": args.model,
        "max_iters": int(args.max_iters),
        "window": int(args.window),
        "tracked_iterations": [int(args.max_iters) - int(args.window) + 1, int(args.max_iters)],
        "track_every": int(getattr(args, "track_every", 1)),
        "n_seeds": int(args.n_seeds),
        "seeds": [int(s) for s in seeds],
        "full_default": bool(getattr(args, "full_default", False)),
        "python": sys.executable,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    seed_seconds = getattr(args, "seed_seconds", {})
    if all(int(s) in seed_seconds for s in seeds):
        info["seed_seconds"] = [seed_seconds[int(s)] for s in seeds]
    if extra:
        info.update(extra)
    return info


def save_model_outputs(args, package, model, setups, iterations, coordinates,
                       best_mean, best_std, reference_source, info):
    save_tracking(tracking_path(args.results_dir, model, package), setups, iterations, coordinates)
    save_reference(reference_path(args.results_dir, model, package), coordinates, best_mean, best_std, reference_source)
    save_run_info(run_info_path(args.results_dir, model, package), info)
    print(f"[{package}/{model}] saved {tracking_path(args.results_dir, model, package)}", flush=True)


def radon_log_to_constrained(means, stds, coordinates, log_coordinates, transform_fn):
    """Push the log-scale radon coordinates through `transform_fn` (exp_lognormal_moments).

    means / stds are (..., n_coords); returns constrained copies plus new coordinate names.
    """
    means = np.array(means, dtype=float, copy=True)
    stds = np.array(stds, dtype=float, copy=True)
    names = list(coordinates)
    for name in log_coordinates:
        j = names.index(name)
        flat_mean, flat_std = transform_fn(means[..., j].ravel(), stds[..., j].ravel())
        means[..., j] = np.asarray(flat_mean).reshape(means[..., j].shape)
        stds[..., j] = np.asarray(flat_std).reshape(stds[..., j].shape)
        names[j] = name.replace("_log__", "")
    return means, stds, names


def radon_outputs(setups, param_names, best_mean, best_std, transform_fn):
    """Save radon on the log scale (as the radon notebooks) and on the constrained scale."""
    log_coordinates = ["sigma_a_log__", "sd_y_log__"]
    source = "single_MC/radon_example/pymc_tutorial/best_reference_values.csv (load_best_references)"
    constrained_setups = {}
    for setup, (means, stds) in setups.items():
        c_means, c_stds, c_names = radon_log_to_constrained(means, stds, param_names, log_coordinates, transform_fn)
        constrained_setups[setup] = (c_means, c_stds)
    c_best_mean, c_best_std, c_names = radon_log_to_constrained(best_mean, best_std, param_names, log_coordinates, transform_fn)
    return {
        "radon_log": dict(setups=setups, coordinates=list(param_names), best_mean=best_mean, best_std=best_std,
                          reference_source=source),
        "radon_constrained": dict(setups=constrained_setups, coordinates=c_names, best_mean=c_best_mean, best_std=c_best_std,
                                  reference_source=source + "; sigma_a / sd_y pushed through exp_lognormal_moments"),
    }


"""
Metrics behind the bias / noise heatmaps (overall_default_bias_comparisons.ipynb)
"""
def per_coordinate_metrics(track, best_mean, best_std, window, ddof=1, zero_tol=1e-12, noise_scale="relative_sd"):
    """All seven metrics for every coordinate of one (model, package, setup).

    track: {"mean": (seeds, iterations, coords), "std": ...}; uses the last `window` iterations.
    Returns (metrics, standardized_errors, zero_mask), each metric an array over coordinates.
    """
    n_tracked = track["mean"].shape[1]
    if window > n_tracked:
        raise ValueError(f"window {window} is longer than the {n_tracked} saved iterations")
    m = track["mean"][:, -window:, :]
    s = track["std"][:, -window:, :]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # a seed whose Stan fits all failed is all NaN
        m_bar = np.nanmean(m, axis=1)    # (seeds, coords)
        s_bar = np.nanmean(s, axis=1)
    zero = np.abs(best_mean) <= zero_tol

    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN columns (e.g. failed Stan checkpoints) -> NaN
        rel_mean = np.where(zero, np.nan, np.abs(m_bar - best_mean) / np.abs(best_mean))
        metrics = {
            "mean_bias": np.nanmean(rel_mean, axis=0) if not zero.all() else np.full(zero.shape, np.nan),
            "std_bias": np.nanmean(np.abs(s_bar - best_std) / best_std, axis=0),
            "mean_std_units": np.nanmean(np.abs(m_bar - best_mean) / best_std, axis=0),
        }
        if noise_scale not in ("relative_sd", "relative_var", "raw"):
            raise ValueError(f"unknown noise_scale {noise_scale!r}; use 'relative_sd', 'relative_var' or 'raw'")
        # relative scales divide each trajectory by its best value *before* taking the variance / SD,
        # e.g. SD across seeds of (<mu> / |mu*|); "raw" keeps the trajectories as they are
        if noise_scale == "raw":
            m_rel, s_rel = m, s
        else:
            m_rel = np.where(zero, np.nan, m / np.where(zero, 1.0, np.abs(best_mean)))   # mu / |mu*|
            s_rel = s / best_std                                                        # sigma / sigma*
        m_rel_bar = np.nanmean(m_rel, axis=1)   # <mu / |mu*|> per seed
        s_rel_bar = np.nanmean(s_rel, axis=1)
        variances = {
            "mean_between": np.nanvar(m_rel_bar, axis=0, ddof=ddof),               # across seeds
            "std_between": np.nanvar(s_rel_bar, axis=0, ddof=ddof),
            "mean_within": np.nanmean(np.nanvar(m_rel, axis=1, ddof=ddof), axis=0),  # within window, mean over seeds
            "std_within": np.nanmean(np.nanvar(s_rel, axis=1, ddof=ddof), axis=0),
        }
        for key, variance in variances.items():
            metrics[key] = np.sqrt(variance) if noise_scale == "relative_sd" else variance
        standardized_errors = {
            "mean_bias": (m_bar - best_mean) / best_std,
            "std_bias": (s_bar - best_std) / best_std,
            "mean_std_units": (m_bar - best_mean) / best_std,
        }
        # noise in units of the best variance, used by rms_standardized whatever the normalization
        standardized_noise = {
            key: value / best_std**2 for key, value in {
                "mean_between": np.nanvar(m_bar, axis=0, ddof=ddof),
                "std_between": np.nanvar(s_bar, axis=0, ddof=ddof),
                "mean_within": np.nanmean(np.nanvar(m, axis=1, ddof=ddof), axis=0),
                "std_within": np.nanmean(np.nanvar(s, axis=1, ddof=ddof), axis=0),
            }.items()
        }
    return metrics, {**standardized_errors, **standardized_noise}, zero

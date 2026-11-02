#!/usr/bin/env python3
"""CmdStanPy version of the R Stan (brms -> cmdstanr / rstan) mean-field ADVI fits of the Journe et al. 2025 Table S1
model. The Stan program and data are the ones brms generates (export them once with `Rscript export_stan_model.R`,
which writes stan_model/journe_brms.stan and stan_model/journe_standata.json), so the model is exactly the brms model.

Setups (the two Stan setups of paper_vignettes/overall_bias_comparisons.ipynb):
  default    every CmdStan ADVI default except iter, tol_rel_obj=1e-6, output_samples=10000
  mc100      as default, plus grad_samples=100 (100 Monte-Carlo samples per gradient step, "adjusted")
Both run --iters iterations (default 25,000).

Modes (run from the Perennial_reproduction folder, in the stan3 conda env):
  python stan_reproducing.py                Table S1 fits: seeds 11..20 tried in order and the first that does not fail
                                            is reported, as in pymc_reproducing.py. Writes TableS1_stan_<setting>.csv
                                            (and TableS1_stan_<setting>_vs_NUTS.csv if TableS1_reproduced.csv exists).
  python stan_reproducing.py --restarts     random restarts for overall_bias_comparisons.ipynb (seeds 0..4 by default):
                                            CmdStan only reports the final fit, so each tracked iteration k is a
                                            separate rerun of that seed with iter=k: every --track-every (10) of the
                                            last --window (500) iterations, i.e. 50 reruns (simpleVI's
                                            modulars.run_stan_random_restarts, exactly as
                                            paper_vignettes/overall_bias_runners/stan_runner.py). Saves the mean and SD of
                                            the 10,000 output draws of the 8 Table S1 coefficients at every tracked
                                            iteration to paper_vignettes/overall_bias_results/journe/stan_*.
                                            2 setups x 5 seeds x 50 reruns of up to 25,000 iterations.
  python stan_reproducing.py --restarts --full_defaults
                                            the same for one setup with every CmdStan ADVI default (tol_rel_obj=0.01,
                                            output_samples=1000, ...), 10,000 iterations, 10 seeds, every one of the last
                                            500 iterations, saved to paper_vignettes/overall_bias_results_full_default/
                                            with each seed's fit time (its iter=10,000 run) for
                                            overall_default_bias_comparisons.ipynb. Fits that meet tol_rel_obj stop
                                            early, so the window's reruns are then all the same fit.
  python stan_reproducing.py --timing_analysis
                                            time every Stan ADVI setup in CmdStanPy over seeds 11-20 (--timing_seeds): full
                                            defaults (every CmdStan default; only the seed is set), default (50,000 max
                                            iterations, tol_rel_obj 1e-5, 10,000 draws; the same as run_repro_vi.R, not the 25,000 /
                                            1e-6 above) and adjusted (that plus grad_samples=100); --only restricts them. One
                                            attempt per seed; total wall time of variational() + reading the draws + summarising;
                                            the model is compiled once (forced, timed on its own "stan_compile" row). default and
                                            adjusted use require_converged=False, as rstan's vb() does not fail on a fit that did not
                                            converge. Writes timing_stan_cmdstanpy.csv / timing_stan_cmdstanpy_summary.csv only.
Other options: --only default | mc100, --iters N, and the --restarts options (--n-seeds, --window, --overwrite,
--results-dir, --max-workers, --track-every).
Requires numpy, pandas, cmdstanpy (+ CmdStan), and simpleVI's modulars for --restarts.
"""
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import pymc_reproducing as P          # table helpers; importing it does not import pymc

STAN_DIR = Path("stan_model")
STAN_FILE = STAN_DIR / "journe_brms.stan"
DATA_JSON = STAN_DIR / "journe_standata.json"
DEFAULT_ITERS = 25000
FULL_DEFAULT_ITERS = 10000      # CmdStan's default iter, for --restarts --full_defaults
SEEDS = range(11, 21)
TOL_REL_OBJ = 1e-6
OUTPUT_SAMPLES = 10_000
SETTINGS = {  # name -> extra CmdStan arguments, as stan_runner.py's STAN_SCENARIOS
    "default": {},
    "mc100": {"grad_samples": 100},
}
RESTART_SETUPS = {"default": "default", "100mc": "mc100"}   # overall_bias_comparisons setup -> SETTINGS name
LABELS = {"default": "Stan ADVI default (CmdStanPy)", "mc100": "Stan ADVI adjusted (100 MC, CmdStanPy)"}
# brms Stan names of the Table S1 terms (b_<resp>_Intercept is brms' generated quantity, so it is the intercept on the
# uncentred predictors whatever the centring; b_<resp>[k] are MAT, poly(MAP, 2)1, poly(MAP, 2)2)
STAN_COLUMNS = {"muPrB_Intercept": "b_muPrB_Intercept", "muDeB_Intercept": "b_muDeB_Intercept",
                "muPrB_MAT": "b_muPrB[1]", "muPrB_polyMAP21": "b_muPrB[2]", "muPrB_polyMAP22": "b_muPrB[3]",
                "muDeB_MAT": "b_muDeB[1]", "muDeB_polyMAP21": "b_muDeB[2]", "muDeB_polyMAP22": "b_muDeB[3]"}


def set_cmdstan():
    """SIMPLEVI_CMDSTAN_PATH, else the CmdStan bundled in the active conda env, else cmdstanpy's default
    (the same choice as overall_bias_runners/stan_runner.py)."""
    from cmdstanpy import cmdstan_path, set_cmdstan_path
    env_cmdstan = Path(sys.prefix).resolve() / "bin" / "cmdstan"
    path = Path(os.environ.get("SIMPLEVI_CMDSTAN_PATH") or (env_cmdstan if env_cmdstan.exists() else cmdstan_path()))
    os.environ["CMDSTAN"] = str(path)
    set_cmdstan_path(str(path))
    return path


def check_files():
    missing = [str(f) for f in (STAN_FILE, DATA_JSON) if not f.exists()]
    if missing:
        sys.exit(f"{', '.join(missing)} not found in {Path.cwd()} - run `Rscript export_stan_model.R` first "
                 "(from the Perennial_reproduction folder)")


# ----------------------------------------------------------------------------- Table S1 fits
def run_fit(model, setting, iters):
    last_err = None
    for seed in SEEDS:
        print(f"  attempt with seed {seed}")
        try:
            fit = model.variational(data=str(DATA_JSON), seed=seed, algorithm="meanfield", iter=iters,
                                    tol_rel_obj=TOL_REL_OBJ, output_samples=OUTPUT_SAMPLES, refresh=0,
                                    show_console=False, **SETTINGS[setting])
            draws = fit.variational_sample_pd
            draws = {term: draws[col].to_numpy() for term, col in STAN_COLUMNS.items()}
            if not all(np.all(np.isfinite(x)) for x in draws.values()):
                raise FloatingPointError("non-finite draws")
            return draws, seed
        except Exception as e:                                    # noqa: BLE001
            last_err = e
            print(f"    failed: {e}")
    raise RuntimeError(f"Stan ADVI ({setting}) failed for all seeds; last error: {last_err}")


def run_tables(args):
    from cmdstanpy import CmdStanModel
    model = CmdStanModel(stan_file=str(STAN_FILE))
    nuts = pd.read_csv(P.NUTS_FILE) if Path(P.NUTS_FILE).exists() else None
    suffix = "" if args.iters == DEFAULT_ITERS else f"_{args.iters // 1000}k"
    chosen = [s.strip() for s in args.only.split(",") if s.strip()]
    for setting in chosen:
        print(f"\n=== {LABELS[setting]} ({args.iters} iterations) ===")
        draws, seed = run_fit(model, setting, args.iters)
        tab = P.summary_table(draws, seed=seed, method=LABELS[setting])
        tab.to_csv(f"TableS1_stan_{setting}{suffix}.csv", index=False)
        print(tab.drop(columns=["method"]).round(4).to_string(index=False))
        if nuts is not None:
            P.compare_to_nuts(tab, nuts, LABELS[setting]).to_csv(f"TableS1_stan_{setting}{suffix}_vs_NUTS.csv", index=False)


# ----------------------------------------------------------------------------- --timing_analysis
TIMING_SETTINGS = {  # name -> (label, max iterations, MC samples, CmdStanPy arguments besides data / seed / output_dir)
    "full_defaults": ("Stan full defaults (CmdStanPy)", 10000, 1, {}),
    "default": ("Stan ADVI default (CmdStanPy)", 50000, 1,
                dict(algorithm="meanfield", iter=50000, tol_rel_obj=1e-5, output_samples=10000, require_converged=False)),
    "adjusted": ("Stan ADVI adjusted, 100 MC (CmdStanPy)", 50000, 100,
                 dict(algorithm="meanfield", grad_samples=100, iter=50000, tol_rel_obj=1e-5, output_samples=10000,
                      require_converged=False)),
}


def run_timing_analysis(args):
    import re
    import tempfile
    from cmdstanpy import CmdStanModel
    seeds = P.parse_seeds(args.timing_seeds)
    names = list(TIMING_SETTINGS) if args.only == ",".join(SETTINGS) else [s.strip() for s in args.only.split(",") if s.strip()]
    bad = [n for n in names if n not in TIMING_SETTINGS]
    if bad:
        sys.exit(f"Unknown setting(s): {bad}. Choose from {list(TIMING_SETTINGS)}")
    print(f"CmdStanPy timing analysis: settings {names} | seeds {seeds}")
    import subprocess
    exe = STAN_FILE.with_suffix("")                      # stan_model/journe_brms
    if exe.exists():
        exe.unlink()      # recompile into a fresh file (an executable overwritten in place can be killed by macOS code signing)
    t0 = time.perf_counter()
    try:                                                 # recompiled so the compile time is real
        model = CmdStanModel(stan_file=str(STAN_FILE), force_compile=True)
    except TypeError:                                    # older CmdStanPy
        model = CmdStanModel(stan_file=str(STAN_FILE), compile="force")
    compile_s = time.perf_counter() - t0
    print(f"  compile / setup: {compile_s:.1f} s")

    def smoke():                                         # a short fit, to find out BEFORE the timed fits whether the executable runs
        with tempfile.TemporaryDirectory() as out:
            model.variational(data=str(DATA_JSON), seed=1, algorithm="meanfield", iter=200, output_dir=out,
                              require_converged=False)         # only checks that the executable runs
    try:
        smoke()
    except Exception as e:                               # noqa: BLE001
        if sys.platform != "darwin":
            raise
        print(f"  test fit failed ({' '.join(str(e).split())[:200]}); re-signing {exe} (macOS kills executables with an invalid "
              "code signature, 'code -9') and trying again", flush=True)
        subprocess.run(["codesign", "--force", "-s", "-", str(exe)], check=False)
        smoke()                                          # if this fails too, the error stops the run with its real message
    rows = [dict(package="stan_cmdstanpy", setting="stan_compile", label="Stan model compile / setup (once, CmdStanPy)",
                 seed=np.nan, iterations=np.nan, mc_samples=np.nan, seconds=round(compile_s, 3), status="ok", error="",
                 iterations_run=np.nan, run_order=0, timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))]
    # rows of CmdStan's ELBO table: iter, ELBO, delta_mean, delta_med (an optional "Chain 1:" prefix is allowed)
    elbo_row = re.compile(r"^\s*(?:Chain\s+\d+:\s*)?(\d+)\s+-?[\d.]+(?:e[+-]?\d+)?\s+[\d.]+", re.M)
    for name in names:
        label, max_it, mc, kw = TIMING_SETTINGS[name]

        def fit(seed, kw=kw, label=label):
            with tempfile.TemporaryDirectory() as out:
                vb = model.variational(data=str(DATA_JSON), seed=seed, output_dir=out, **kw)
                draws = vb.variational_sample_pd
                P.summary_table({term: draws[col].to_numpy() for term, col in STAN_COLUMNS.items()}, seed=seed, method=label)
                text = "\n".join(Path(f).read_text(errors="ignore") for f in Path(out).glob("*.txt"))
                its = [int(m) for m in elbo_row.findall(text)]
                if not its:                       # keep one sample of what was captured so the parsing can be fixed
                    Path("timing_logs").mkdir(exist_ok=True)
                    Path(f"timing_logs/cmdstanpy_vb_output_{name}.txt").write_text(
                        f"files in the output dir: {sorted(p.name for p in Path(out).iterdir())}\n" + text[:5000])
                return float(max(its)) if its else float("nan")         # iterations actually run (NaN if not readable)

        print(f"\n=== {label}: {len(seeds)} seeds, up to {max_it} iterations ===")
        P.time_seeds("stan_cmdstanpy", name, label, seeds, fit, max_it, mc, rows,
                     save=lambda r: P.write_timing("stan_cmdstanpy", r, quiet=True))
        P.write_timing("stan_cmdstanpy", rows)


# ----------------------------------------------------------------------------- --restarts
def run_restarts(args):
    import journe_restarts as J
    sys.path.insert(0, str(J.bias_io.REPO_ROOT))
    import cmdstanpy
    from modulars.stan_rr_test import run_stan_random_restarts

    rargs = J.restart_args(args, args.iters)
    if J.bias_io.already_saved(rargs, "stan"):
        return
    start_time = time.time()
    iterations = J.bias_io.window_iterations(rargs.max_iters, rargs.window, rargs.track_every)
    if rargs.full_default:    # every CmdStan ADVI default except iter
        scenarios = ({"name": J.bias_io.FULL_DEFAULT_SETUP, "mc_label": "1", "cmd_args": ()},)
    else:
        # --single-setup NAME: only the 1 MC "default" setting, saved as setup NAME
        restart_setups = {rargs.single_setup: RESTART_SETUPS["default"]} if rargs.single_setup else RESTART_SETUPS
        scenarios = tuple({"name": name, "mc_label": str(SETTINGS[s].get("grad_samples", 1)),
                           "cmd_args": tuple(f"{k}={v}" for k, v in SETTINGS[s].items()) + (f"output_samples={OUTPUT_SAMPLES}",),
                           "tol_rel_obj": TOL_REL_OBJ} for name, s in restart_setups.items())
    setup_names = [scenario["name"] for scenario in scenarios]
    seeds = list(range(rargs.n_seeds))
    max_workers = rargs.max_workers or max(1, (os.cpu_count() or 2) - 1)
    run_dir = rargs.results_dir / "_stan_runs" / J.MODEL
    with open(DATA_JSON) as f:
        stan_data = json.load(f)
    param_columns = [STAN_COLUMNS[t] for t in P.TERMS]

    # seeds in chunks of max_workers (each seed's two setups are separate parallel tasks), cached per seed
    per_seed, failures = {}, []
    for seed in seeds:
        cached = J.bias_io.load_cached_seed(rargs, "stan", seed)
        if cached is not None:
            per_seed[seed] = cached
    todo = [s for s in seeds if s not in per_seed]
    chunks = [todo[i:i + max_workers] for i in range(0, len(todo), max_workers)]
    print(f"Stan restarts: {rargs.n_seeds} seeds x {setup_names}, {rargs.max_iters:,} iterations, last "
          f"{rargs.window} tracked every {rargs.track_every} ({len(iterations)} CmdStan reruns), {max_workers} workers; {len(seeds) - len(todo)} seeds cached")
    for chunk in chunks:
        assert chunk == list(range(chunk[0], chunk[-1] + 1)), "run_stan_random_restarts needs consecutive seeds"
        t0 = time.time()
        chunk_dir = run_dir / f"seeds_{chunk[0]:03d}_{chunk[-1]:03d}"
        result = run_stan_random_restarts(
            stan_file=STAN_FILE.resolve(), data=stan_data, param_columns=param_columns, log_columns=(),
            output_dir=chunk_dir / "cmdstan_runs", max_iters=rargs.max_iters, n_restarts=len(chunk), track_every=rargs.track_every,
            min_iter=iterations[0], scenarios=scenarios, seed_offset=chunk[0], parallel=True, max_workers=max_workers,
            keep_outputs=False, refresh=0, dtype=np.float64, show_progress=True)
        with open(chunk_dir / "stan_result.pkl", "wb") as f:
            pickle.dump(result, f)
        for i, seed in enumerate(chunk):
            arrays = []
            for scenario in scenarios:
                assert np.array_equal(result["iterations_by_scenario"][scenario["name"]], iterations)
                arrays += [result["means_by_scenario"][scenario["name"]][i], result["stds_by_scenario"][scenario["name"]][i]]
            # a full-default seed's fit time is its CmdStan run with iter=max_iters (the last rerun)
            seconds = result["seconds_by_scenario"][setup_names[0]][i][-1] if rargs.full_default else None
            J.bias_io.save_cached_seed(rargs, "stan", seed, arrays, seconds=seconds)
            per_seed[seed] = arrays
        failures.extend(result["failures"])
        print(f"[stan/{J.MODEL}] seeds {chunk[0]}-{chunk[-1]} done in {time.time() - t0:.1f}s", flush=True)

    setups = {name: (np.stack([per_seed[s][2 * k] for s in seeds]), np.stack([per_seed[s][2 * k + 1] for s in seeds]))
              for k, name in enumerate(setup_names)}
    if failures:
        print(f"[stan/{J.MODEL}] WARNING: {len(failures)} failed checkpoints (saved as NaN); see run_info.json")
    J.save(rargs, "stan", setups, seeds, {
        "setups": {s["name"]: s for s in scenarios}, "stan_file": str(STAN_FILE), "stan_columns": param_columns,
        "n_failed_checkpoints": len(failures), "failures": failures[:200], "runtime_seconds": time.time() - start_time,
        "max_workers": max_workers, "cmdstan_path": os.environ["CMDSTAN"], "cmdstanpy_version": cmdstanpy.__version__})


# ----------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(SETTINGS), help="comma-separated subset of: " + ", ".join(SETTINGS))
    ap.add_argument("--iters", type=int, default=None,
                    help=f"ADVI iterations (default {DEFAULT_ITERS}; CmdStan's default {FULL_DEFAULT_ITERS} with --full_defaults)")
    ap.add_argument("--full_defaults", action="store_true",
                    help="with --restarts: one setup with every CmdStan ADVI default (see above)")
    ap.add_argument("--timing_analysis", action="store_true",
                    help="time every Stan ADVI setup in CmdStanPy over seeds 11-20 (see above); writes timing_stan_cmdstanpy*.csv only")
    ap.add_argument("--timing_seeds", default="11-20", help='seeds for --timing_analysis, e.g. "11-20" (default) or "11,13,15"')
    import journe_restarts
    journe_restarts.add_restart_args(ap, default_seeds=None, seeds_help="5; 10 with --full_defaults")
    ap.set_defaults(track_every=None)
    args = ap.parse_args(argv)
    # full defaults: CmdStan's iter, 10 seeds and every iteration of the window; otherwise 5 seeds, every 10th
    args.iters = args.iters or (FULL_DEFAULT_ITERS if args.full_defaults else DEFAULT_ITERS)
    args.n_seeds = args.n_seeds or (10 if args.full_defaults else 5)
    args.track_every = args.track_every or (1 if args.full_defaults else 10)
    bad = [s for s in args.only.split(",") if s.strip() and s.strip() not in SETTINGS]
    if bad and not args.timing_analysis:
        sys.exit(f"Unknown setting(s): {bad}. Choose from {list(SETTINGS)}")
    check_files()
    print("CmdStan:", set_cmdstan())
    if args.timing_analysis:
        return run_timing_analysis(args)
    if args.restarts:
        return run_restarts(args)
    run_tables(args)


if __name__ == "__main__":
    main()

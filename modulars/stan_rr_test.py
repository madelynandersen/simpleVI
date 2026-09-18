"""CmdStan random-restart helpers for the single_MC comparisons.

Stan's variational method only writes final approximate-posterior draws.  To
build trajectories comparable to the PyMC, TFP, and NumPyro restart notebooks,
we rerun the same seed at a grid of prefix iteration counts and summarize the
final draws at each prefix.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_STAN_SCENARIOS = (
    {
        "name": "default",
        "mc_label": "1",
        "cmd_args": (),
    },
    {
        "name": "grad100_elbo100_adapt_off",
        "mc_label": "100",
        "cmd_args": ("grad_samples=100", "elbo_samples=100", "adapt", "engaged=0"),
    },
)

# Four-way comparison: default everything; grad_samples=100 with all other
# settings left at their defaults (including adaptive eta); grad_samples=100
# with adaptation disabled (eta left at its default); and grad_samples=100
# with a much tighter ELBO convergence tolerance.
#
# The first three scenarios converge almost immediately (well under
# max_iters/10 iterations) with Stan's default tol_rel_obj, so they only run
# for max_iters/10 at the caller's own track_every -- there's nothing to see
# past that. The tol_rel_obj scenario needs a much looser stopping bar to
# actually keep optimizing, so it runs for max_iters*2 at a 10x coarser
# track_every (still the same number of tracked checkpoints as the other
# three, just spread over a longer run).
FOUR_WAY_STAN_SCENARIOS = (
    {
        "name": "default",
        "mc_label": "1",
        "cmd_args": (),
        "max_iters_scale": 0.1,
    },
    {
        "name": "grad100_default_rest",
        "mc_label": "100",
        "cmd_args": ("grad_samples=100",),
        "max_iters_scale": 0.1,
    },
    {
        "name": "grad100_adapt_off",
        "mc_label": "100_adapt_off",
        "cmd_args": ("grad_samples=100", "adapt", "engaged=0"),
        "max_iters_scale": 0.1,
    },
    {
        "name": "grad100_tol_rel_obj1e-6",
        "mc_label": "100_tol1e-6",
        "cmd_args": ("grad_samples=100",),
        "tol_rel_obj": 1e-6,
        "max_iters_scale": 2.0,
        "track_every_scale": 10.0,
    },
)


# Five-way comparison: the four FOUR_WAY_STAN_SCENARIOS scenarios, plus a
# second tight-tolerance scenario that fixes eta=0.01 (adaptation must be
# disabled for an explicit eta to actually take effect -- Stan ignores a
# user-supplied eta and runs its own eta search whenever adapt is engaged),
# to see whether the tol_rel_obj1e-6 scenario's late-iteration behavior is
# eta-search-driven or persists under a fixed learning rate. Both
# tol_rel_obj1e-6 scenarios run 4x longer than the base max_iters (200k at
# the 1dgaussian notebook's max_iters=50_000) at a 10x coarser tracking
# resolution, matching each other's grid for direct comparison.
FIVE_WAY_STAN_SCENARIOS = (
    {
        "name": "default",
        "mc_label": "1",
        "cmd_args": (),
        "max_iters_scale": 0.1,
    },
    {
        "name": "grad100_default_rest",
        "mc_label": "100",
        "cmd_args": ("grad_samples=100",),
        "max_iters_scale": 0.1,
    },
    {
        "name": "grad100_adapt_off",
        "mc_label": "100_adapt_off",
        "cmd_args": ("grad_samples=100", "adapt", "engaged=0"),
        "max_iters_scale": 0.1,
    },
    {
        "name": "grad100_tol_rel_obj1e-6",
        "mc_label": "100_tol1e-6",
        "cmd_args": ("grad_samples=100",),
        "tol_rel_obj": 1e-6,
        "max_iters_scale": 4.0,
        "track_every_scale": 10.0,
    },
    {
        "name": "grad100_tol_rel_obj1e-6_eta0.01",
        "mc_label": "100_tol1e-6_eta0.01",
        "cmd_args": ("grad_samples=100", "adapt", "engaged=0", "eta=0.01"),
        "tol_rel_obj": 1e-6,
        "max_iters_scale": 4.0,
        "track_every_scale": 10.0,
    },
)


def stan_vector_columns(name, dim):
    """Return CmdStan CSV column names for a 1-indexed Stan vector/simplex."""
    return [f"{name}[{i}]" for i in range(1, int(dim) + 1)]


def tracked_iterations(max_iters, track_every=10, include_first=True, min_iter=None):
    """Return the prefix iteration counts used for Stan trajectory tracking.

    ``min_iter`` restricts tracking to a tail window (e.g. the last 10k of a
    50k-iteration run) so we don't pay for a rerun at every small prefix when
    only the late-iteration convergence behavior is of interest.
    """
    max_iters = int(max_iters)
    track_every = max(1, int(track_every))
    values = []
    if include_first:
        values.append(1)
    values.extend(range(track_every, max_iters + 1, track_every))
    if values[-1] != max_iters:
        values.append(max_iters)
    if min_iter is not None:
        min_iter = int(min_iter)
        values = [v for v in values if v >= min_iter] or [max_iters]
    return np.asarray(sorted(set(values)), dtype=int)


def compile_stan_model(stan_file, force_compile=False):
    """Compile a Stan model with CmdStanPy and return the executable path."""
    from cmdstanpy import CmdStanModel, set_cmdstan_path

    cmdstan_override = os.environ.get("SIMPLEVI_CMDSTAN_PATH")
    conda_cmdstan = Path(sys.prefix) / "bin" / "cmdstan"
    if cmdstan_override:
        set_cmdstan_path(cmdstan_override)
    elif conda_cmdstan.exists():
        os.environ["CMDSTAN"] = str(conda_cmdstan)
        set_cmdstan_path(str(conda_cmdstan))

    model = CmdStanModel(stan_file=str(stan_file), force_compile=force_compile)
    return str(model.exe_file)


def write_stan_data_json(path, data):
    """Write a CmdStan JSON data file, converting NumPy objects as needed."""
    from cmdstanpy import write_stan_json

    write_stan_json(str(path), _jsonable(data))
    return Path(path)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(val) for val in value]
    return value


def _column_lookup(header, name):
    if name in header:
        return header.index(name)

    alternatives = []
    if "[" in name and name.endswith("]"):
        alternatives.append(name.replace("[", ".").replace("]", ""))
    if "." in name:
        base, idx = name.rsplit(".", 1)
        if idx.isdigit():
            alternatives.append(f"{base}[{idx}]")

    for alt_name in alternatives:
        if alt_name in header:
            return header.index(alt_name)

    raise ValueError(f"Could not find Stan CSV column {name!r}. Available columns: {header}")


def read_variational_draws(csv_path, param_columns, log_columns=(), drop_mean_row=True):
    """Read approximate posterior draws from a CmdStan variational CSV file."""
    header = None
    rows = []
    with open(csv_path, "r", newline="") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = next(csv.reader([line]))
            if header is None:
                header = [part.strip() for part in parts]
            else:
                rows.append([float(part) for part in parts])

    if header is None or len(rows) == 0:
        raise ValueError(f"No variational draws found in {csv_path}")

    indices = [_column_lookup(header, col) for col in param_columns]
    values = np.asarray(rows, dtype=float)[:, indices]
    if drop_mean_row and values.shape[0] > 1:
        values = values[1:, :]

    log_columns = set(log_columns or ())
    for col_idx, col_name in enumerate(param_columns):
        if col_name in log_columns:
            values[:, col_idx] = np.log(np.clip(values[:, col_idx], 1e-300, None))
    return values


def read_variational_moments(csv_path, param_columns, log_columns=()):
    """Return empirical marginal means/stds from CmdStan variational draws."""
    draws = read_variational_draws(csv_path, param_columns, log_columns=log_columns)
    if draws.shape[0] == 1:
        std = np.zeros(draws.shape[1], dtype=float)
    else:
        std = draws.std(axis=0, ddof=1)
    return draws.mean(axis=0), std


def _run_prefix_variational(
        model_exe,
        data_json,
        run_dir,
        seed,
        n_iters,
        param_columns,
        log_columns,
        cmd_args,
        keep_outputs,
        refresh,
        max_retries=2):
    """Run one Stan variational prefix and return its parsed moments.

    On very long runs (thousands of sequential Stan calls per restart, each
    briefly creating and deleting a couple of files), run_dir has occasionally
    been observed to vanish out from under an in-flight call -- most likely a
    side effect of this living in a heavily-synced (iCloud) folder under
    sustained file churn -- leaving the Stan CSV output, and even run_dir
    itself, missing right after the subprocess exits. That is distinct from a
    genuine ADVI failure (where run_dir survives but the CSV is merely
    empty/malformed), so only a vanished run_dir triggers a retry of the whole
    checkpoint; a real algorithmic failure is still recorded as one, as before.
    """
    run_dir = Path(run_dir)
    stem = f"iter_{int(n_iters):06d}"
    csv_path = run_dir / f"{stem}.csv"
    diagnostic_path = run_dir / f"{stem}_diagnostic.csv"
    stdout_path = run_dir / f"{stem}_stdout.txt"

    cmd = [
        str(model_exe),
        "random", f"seed={int(seed)}",
        "data", f"file={data_json}",
        "output", f"file={csv_path}", f"diagnostic_file={diagnostic_path}", f"refresh={int(refresh)}",
        "method=variational", "algorithm=meanfield", f"iter={int(n_iters)}",
    ]
    cmd.extend(cmd_args)

    for attempt in range(int(max_retries) + 1):
        run_dir.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(cmd, capture_output=True, text=True)

        try:
            if completed.returncode != 0 or keep_outputs:
                stdout_path.write_text(completed.stdout + completed.stderr)

            mean, std = read_variational_moments(
                csv_path,
                param_columns=param_columns,
                log_columns=log_columns,
            )
            status = "ok" if completed.returncode == 0 else "parsed_after_nonzero_return"
            error = ""
            break
        except FileNotFoundError as err:
            if not run_dir.is_dir() and attempt < max_retries:
                time.sleep(1.0)
                continue
            n_params = len(param_columns)
            mean = np.full(n_params, np.nan)
            std = np.full(n_params, np.nan)
            status = "failed"
            error = repr(err)
            try:
                if not stdout_path.exists():
                    stdout_path.write_text(completed.stdout + completed.stderr)
            except FileNotFoundError:
                pass
            break
        except Exception as err:
            n_params = len(param_columns)
            mean = np.full(n_params, np.nan)
            std = np.full(n_params, np.nan)
            status = "failed"
            error = repr(err)
            try:
                if not stdout_path.exists():
                    stdout_path.write_text(completed.stdout + completed.stderr)
            except FileNotFoundError:
                pass
            break

    if not keep_outputs:
        for path in (csv_path, diagnostic_path):
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    return {
        "iter": int(n_iters),
        "returncode": int(completed.returncode),
        "status": status,
        "error": error,
        "mean": mean,
        "std": std,
    }


def _run_restart_worker(args):
    (
        scenario_idx,
        scenario_name,
        restart_idx,
        seed,
        model_exe,
        data_json,
        output_dir,
        iterations,
        param_columns,
        log_columns,
        cmd_args,
        keep_outputs,
        refresh,
        dtype_name,
    ) = args

    dtype = np.dtype(dtype_name)
    n_track = len(iterations)
    n_params = len(param_columns)
    means = np.empty((n_track, n_params), dtype=dtype)
    stds = np.empty((n_track, n_params), dtype=dtype)
    failures = []
    run_dir = Path(output_dir) / scenario_name / f"restart_{restart_idx:03d}"

    for track_idx, n_iters in enumerate(iterations):
        result = _run_prefix_variational(
            model_exe=model_exe,
            data_json=data_json,
            run_dir=run_dir,
            seed=seed,
            n_iters=int(n_iters),
            param_columns=param_columns,
            log_columns=log_columns,
            cmd_args=cmd_args,
            keep_outputs=keep_outputs,
            refresh=refresh,
        )
        means[track_idx] = result["mean"]
        stds[track_idx] = result["std"]
        if result["status"] != "ok":
            failures.append(
                {
                    "scenario": scenario_name,
                    "restart_idx": restart_idx,
                    "seed": seed,
                    "iter": int(n_iters),
                    "returncode": result["returncode"],
                    "status": result["status"],
                    "error": result["error"],
                }
            )

    return scenario_idx, restart_idx, means, stds, failures


def run_stan_random_restarts(
        stan_file,
        data,
        param_columns,
        output_dir,
        max_iters=300_000,
        n_restarts=50,
        track_every=10,
        min_iter=None,
        scenarios=DEFAULT_STAN_SCENARIOS,
        log_columns=(),
        tol_rel_obj=None,
        seed_offset=0,
        parallel=False,
        max_workers=None,
        keep_outputs=False,
        force_compile=False,
        refresh=0,
        dtype=np.float32,
        show_progress=True):
    """Run Stan variational prefix trajectories for random restarts.

    Each scenario dict may include a "tol_rel_obj" key to override the ADVI
    relative-ELBO convergence tolerance for that scenario specifically (e.g.
    the "grad100_tol_rel_obj1e-6" entry in FOUR_WAY_STAN_SCENARIOS). The
    ``tol_rel_obj`` argument here sets a default applied to any scenario that
    does not specify its own.

    Each scenario dict may also include "max_iters_scale" and/or
    "track_every_scale" (both default to 1.0) to run that scenario for a
    different length/tracking resolution than the ``max_iters``/``track_every``
    arguments given here -- e.g. a scenario that needs many more iterations to
    show its effect (a tight tol_rel_obj) can use a larger max_iters_scale and
    a coarser track_every_scale than the other scenarios in the same call.
    Because scenarios can therefore end up with differently-shaped tracked
    iteration grids, results are returned per scenario (see
    "iterations_by_scenario"/"means_by_scenario"/"stds_by_scenario") rather
    than as one shared array.

    Returns a dictionary containing arrays in the same semantic order used by
    the other restart notebooks: default Stan output as "single" and the
    100-gradient/100-ELBO/fixed-eta output as "multi".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_json = output_dir / "stan_data.json"
    write_stan_data_json(data_json, data)

    model_exe = compile_stan_model(stan_file, force_compile=force_compile)
    param_columns = tuple(param_columns)
    log_columns = tuple(log_columns or ())
    dtype = np.dtype(dtype)

    scenario_list = [dict(scenario) for scenario in scenarios]
    n_params = len(param_columns)

    iterations_by_scenario = {}
    means_by_scenario = {}
    stds_by_scenario = {}

    tasks = []
    for scenario_idx, scenario in enumerate(scenario_list):
        scenario_max_iters = int(round(max_iters * scenario.get("max_iters_scale", 1.0)))
        scenario_track_every = max(1, int(round(track_every * scenario.get("track_every_scale", 1.0))))
        scenario_min_iter = scenario.get("min_iter", min_iter)
        scenario_iterations = tracked_iterations(
            max_iters=scenario_max_iters, track_every=scenario_track_every, min_iter=scenario_min_iter
        )
        iterations_by_scenario[scenario["name"]] = scenario_iterations
        n_track = len(scenario_iterations)
        means_by_scenario[scenario["name"]] = np.empty((n_restarts, n_track, n_params), dtype=dtype)
        stds_by_scenario[scenario["name"]] = np.empty((n_restarts, n_track, n_params), dtype=dtype)

        scenario_tol_rel_obj = scenario.get("tol_rel_obj", tol_rel_obj)
        cmd_args = tuple(scenario.get("cmd_args", ()))
        if scenario_tol_rel_obj is not None:
            cmd_args = (f"tol_rel_obj={scenario_tol_rel_obj}",) + cmd_args
        for restart_idx in range(int(n_restarts)):
            tasks.append(
                (
                    scenario_idx,
                    scenario["name"],
                    restart_idx,
                    int(seed_offset) + restart_idx,
                    model_exe,
                    str(data_json),
                    str(output_dir),
                    scenario_iterations,
                    param_columns,
                    log_columns,
                    cmd_args,
                    bool(keep_outputs),
                    int(refresh),
                    dtype.name,
                )
            )

    iterator = tasks
    if show_progress:
        from tqdm import tqdm
        iterator = tqdm(tasks, desc="Stan restart/scenario tasks")

    failures = []
    if parallel:
        if max_workers is None or int(max_workers) <= 0:
            available_cores = os.cpu_count() or 1
            max_workers = min(len(tasks), max(1, available_cores - 1))
        with ProcessPoolExecutor(max_workers=int(max_workers)) as executor:
            future_to_task = {executor.submit(_run_restart_worker, task): task for task in tasks}
            future_iter = as_completed(future_to_task)
            if show_progress:
                from tqdm import tqdm
                future_iter = tqdm(future_iter, total=len(tasks), desc="Stan restart/scenario tasks")
            for future in future_iter:
                scenario_idx, restart_idx, mean_traj, std_traj, task_failures = future.result()
                name = scenario_list[scenario_idx]["name"]
                means_by_scenario[name][restart_idx] = mean_traj
                stds_by_scenario[name][restart_idx] = std_traj
                failures.extend(task_failures)
    else:
        for task in iterator:
            scenario_idx, restart_idx, mean_traj, std_traj, task_failures = _run_restart_worker(task)
            name = scenario_list[scenario_idx]["name"]
            means_by_scenario[name][restart_idx] = mean_traj
            stds_by_scenario[name][restart_idx] = std_traj
            failures.extend(task_failures)

    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "stan_run_failures.csv", index=False)

    result = {
        "scenario_names": [scenario["name"] for scenario in scenario_list],
        "scenario_mc_labels": [str(scenario.get("mc_label", idx)) for idx, scenario in enumerate(scenario_list)],
        "iterations_by_scenario": iterations_by_scenario,
        "means_by_scenario": means_by_scenario,
        "stds_by_scenario": stds_by_scenario,
        "failures": failures,
        "param_columns": list(param_columns),
        "stan_file": str(stan_file),
        "data_json": str(data_json),
    }
    if len(scenario_list) >= 2:
        result.update(stan_result_to_tracking_arrays(result))
    return result


def stan_result_to_tracking_arrays(result):
    """Convert a Stan result dict to the plotting tuple used by other methods.

    Uses the first two scenarios (by declaration order) as "single"/"multi".
    Both must share the same tracked-iteration grid; scenarios with a
    different max_iters_scale/track_every_scale (e.g. a tol_rel_obj ablation
    run much longer) can't be combined this way -- use stan_result_by_scenario
    for those instead.
    """
    scenario_names = result["scenario_names"]
    if len(scenario_names) < 2:
        raise ValueError("Expected at least two Stan scenarios: default and 100-MC.")

    name0, name1 = scenario_names[0], scenario_names[1]
    means0 = np.asarray(result["means_by_scenario"][name0])
    stds0 = np.asarray(result["stds_by_scenario"][name0])
    means1 = np.asarray(result["means_by_scenario"][name1])
    stds1 = np.asarray(result["stds_by_scenario"][name1])

    if means0.shape != means1.shape:
        raise ValueError(
            f"Cannot build the legacy single/multi tuple: scenarios {name0!r} and "
            f"{name1!r} have different tracked shapes ({means0.shape} vs {means1.shape}). "
            "Use stan_result_by_scenario instead."
        )

    single_means = means0
    single_stds = stds0
    multi_means = means1
    multi_stds = stds1

    if single_means.shape[-1] == 1:
        single_means = single_means[..., 0]
        single_stds = single_stds[..., 0]
        multi_means = multi_means[..., 0]
        multi_stds = multi_stds[..., 0]

    return {
        "single_means": single_means,
        "single_stds": single_stds,
        "multi_means": multi_means,
        "multi_stds": multi_stds,
        "iterations": np.asarray(result["iterations_by_scenario"][name0]),
    }


def stan_result_tuple(result):
    """Return (single_means, single_stds, multi_means, multi_stds)."""
    arrays = stan_result_to_tracking_arrays(result)
    return (
        arrays["single_means"],
        arrays["single_stds"],
        arrays["multi_means"],
        arrays["multi_stds"],
    )


def stan_result_by_scenario(result):
    """Return {scenario_name: (means, stds, iterations)} for any number of scenarios.

    Unlike stan_result_tuple (which only exposes the first two scenarios as
    "single"/"multi" and requires them to share one iteration grid), this
    works for the 3-way (or N-way) comparisons and returns each scenario's own
    tracked iteration grid, since scenarios may have been run for different
    lengths/tracking resolutions (see max_iters_scale/track_every_scale).
    """
    by_scenario = {}
    for name in result["scenario_names"]:
        means = np.asarray(result["means_by_scenario"][name])
        stds = np.asarray(result["stds_by_scenario"][name])
        iterations = np.asarray(result["iterations_by_scenario"][name])
        if means.shape[-1] == 1:
            means = means[..., 0]
            stds = stds[..., 0]
        by_scenario[name] = (means, stds, iterations)
    return by_scenario


def save_stan_by_scenario(path, by_scenario):
    """Save a stan_result_by_scenario() dict to a single .npz file.

    Scenarios can have different tracked-iteration grid lengths (e.g. a
    tol_rel_obj ablation run much longer than the others), so each array is
    stored under its own scenario-prefixed key rather than stacked into one
    shared array. Load back with load_stan_by_scenario to skip rerunning
    Stan and go straight to plotting.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {"__scenario_names__": np.asarray(list(by_scenario.keys()))}
    for name, (means, stds, iterations) in by_scenario.items():
        arrays[f"{name}__means"] = np.asarray(means)
        arrays[f"{name}__stds"] = np.asarray(stds)
        arrays[f"{name}__iterations"] = np.asarray(iterations)
    np.savez(path, **arrays)
    return path


def load_stan_by_scenario(path):
    """Inverse of save_stan_by_scenario: returns {name: (means, stds, iterations)}."""
    with np.load(path, allow_pickle=False) as data:
        scenario_names = [str(name) for name in data["__scenario_names__"]]
        return {
            name: (
                data[f"{name}__means"],
                data[f"{name}__stds"],
                data[f"{name}__iterations"],
            )
            for name in scenario_names
        }

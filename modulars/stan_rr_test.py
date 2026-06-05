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


def stan_vector_columns(name, dim):
    """Return CmdStan CSV column names for a 1-indexed Stan vector/simplex."""
    return [f"{name}[{i}]" for i in range(1, int(dim) + 1)]


def tracked_iterations(max_iters, track_every=10, include_first=True):
    """Return the prefix iteration counts used for Stan trajectory tracking."""
    max_iters = int(max_iters)
    track_every = max(1, int(track_every))
    values = []
    if include_first:
        values.append(1)
    values.extend(range(track_every, max_iters + 1, track_every))
    if values[-1] != max_iters:
        values.append(max_iters)
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
        refresh):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
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

    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0 or keep_outputs:
        stdout_path.write_text(completed.stdout + completed.stderr)

    try:
        mean, std = read_variational_moments(
            csv_path,
            param_columns=param_columns,
            log_columns=log_columns,
        )
        status = "ok" if completed.returncode == 0 else "parsed_after_nonzero_return"
        error = ""
    except Exception as err:
        n_params = len(param_columns)
        mean = np.full(n_params, np.nan)
        std = np.full(n_params, np.nan)
        status = "failed"
        error = repr(err)
        if not stdout_path.exists():
            stdout_path.write_text(completed.stdout + completed.stderr)

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
        scenarios=DEFAULT_STAN_SCENARIOS,
        log_columns=(),
        seed_offset=0,
        parallel=False,
        max_workers=None,
        keep_outputs=False,
        force_compile=False,
        refresh=0,
        dtype=np.float32,
        show_progress=True):
    """Run Stan variational prefix trajectories for random restarts.

    Returns a dictionary containing arrays in the same semantic order used by
    the other restart notebooks: default Stan output as "single" and the
    100-gradient/100-ELBO/fixed-eta output as "multi".
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_json = output_dir / "stan_data.json"
    write_stan_data_json(data_json, data)

    model_exe = compile_stan_model(stan_file, force_compile=force_compile)
    iterations = tracked_iterations(max_iters=max_iters, track_every=track_every)
    param_columns = tuple(param_columns)
    log_columns = tuple(log_columns or ())
    dtype = np.dtype(dtype)

    scenario_list = [dict(scenario) for scenario in scenarios]
    n_scenarios = len(scenario_list)
    n_track = len(iterations)
    n_params = len(param_columns)

    means = np.empty((n_scenarios, n_restarts, n_track, n_params), dtype=dtype)
    stds = np.empty_like(means)

    tasks = []
    for scenario_idx, scenario in enumerate(scenario_list):
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
                    iterations,
                    param_columns,
                    log_columns,
                    tuple(scenario.get("cmd_args", ())),
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
                means[scenario_idx, restart_idx] = mean_traj
                stds[scenario_idx, restart_idx] = std_traj
                failures.extend(task_failures)
    else:
        for task in iterator:
            scenario_idx, restart_idx, mean_traj, std_traj, task_failures = _run_restart_worker(task)
            means[scenario_idx, restart_idx] = mean_traj
            stds[scenario_idx, restart_idx] = std_traj
            failures.extend(task_failures)

    if failures:
        pd.DataFrame(failures).to_csv(output_dir / "stan_run_failures.csv", index=False)

    result = {
        "iterations": iterations,
        "scenario_names": [scenario["name"] for scenario in scenario_list],
        "scenario_mc_labels": [str(scenario.get("mc_label", idx)) for idx, scenario in enumerate(scenario_list)],
        "means": means,
        "stds": stds,
        "failures": failures,
        "param_columns": list(param_columns),
        "stan_file": str(stan_file),
        "data_json": str(data_json),
    }
    result.update(stan_result_to_tracking_arrays(result))
    return result


def stan_result_to_tracking_arrays(result):
    """Convert a Stan result dict to the plotting tuple used by other methods."""
    means = np.asarray(result["means"])
    stds = np.asarray(result["stds"])
    if means.shape[0] < 2:
        raise ValueError("Expected at least two Stan scenarios: default and 100-MC.")

    single_means = means[0]
    single_stds = stds[0]
    multi_means = means[1]
    multi_stds = stds[1]

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

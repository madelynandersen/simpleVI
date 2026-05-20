import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from modulars.radon import (
    build_numpyro_price_radon_model,
    load_radon_data,
    make_price_radon_param_names,
    price_theta_summary_from_unconstrained,
)
from modulars.utils import find_best_params


INPUT_SPECS = [
    {
        "package": "numpyro",
        "setting": "adam_100_mc",
        "path": "results/numpyro_processed_restarts.csv",
        "summary_path": "results/numpyro_final_restarts.csv",
    },
    {
        "package": "tfp",
        "setting": "adam_100_mc",
        "path": "results/tfp_processed_restarts.csv",
        "summary_path": "results/tfp_final_restarts.csv",
    },
    {
        "package": "pymc",
        "setting": "default_100_mc",
        "path": "results/pymc_default_processed_restarts.csv",
        "summary_path": "results/pymc_default_final_restarts.csv",
    },
    {
        "package": "pymc",
        "setting": "adam_100_mc",
        "path": "results/pymc_adam_processed_restarts.csv",
        "summary_path": "results/pymc_adam_final_restarts.csv",
    },
]

FOCAL_PARAMETERS = ["mu", "kappa_sq_log__", "sigma_sq_log__"]
MAX_PROCESSED_LOAD_BYTES = 1_000_000_000


def _resolve_output(path, base_dir):
    path = Path(path)
    if not path.is_absolute():
        path = base_dir / path
    return path


def _method_label(package, setting):
    return f"{package}::{setting}"


def _split_method_label(method):
    package, setting = str(method).split("::", maxsplit=1)
    return package, setting


def _prepare_result_specs(base_dir, allow_missing, excluded_packages=None):
    excluded_packages = set(excluded_packages or [])
    active_specs = [
        spec for spec in INPUT_SPECS
        if spec["package"] not in excluded_packages
    ]
    result_specs = []
    missing = []
    skipped = []

    for spec in active_specs:
        summary_path = base_dir / spec["summary_path"]
        processed_path = base_dir / spec["path"]
        label = _method_label(spec["package"], spec["setting"])

        if summary_path.exists():
            result_specs.append((label, str(summary_path)))
            continue

        if processed_path.exists():
            if processed_path.stat().st_size > MAX_PROCESSED_LOAD_BYTES:
                reason = (
                    f"Processed restart file is too large to load safely "
                    f"({processed_path.stat().st_size:,} bytes): {processed_path}. "
                    "Rerun the notebook with the current track_every=10 settings "
                    "to create the small final summary file."
                )
                skipped.append({**spec, "path": str(processed_path), "reason": reason})
                if allow_missing:
                    continue
                raise ValueError(reason)
            result_specs.append((label, str(processed_path)))
            continue

        missing.append(
            {
                **spec,
                "path": str(processed_path),
                "summary_path": str(summary_path),
            }
        )

    if missing and not allow_missing:
        missing_paths = "\n".join(
            f"- {item['summary_path']} or {item['path']}" for item in missing
        )
        raise FileNotFoundError(
            "Missing saved restart files. Run the corresponding notebooks first, "
            "or pass --allow-missing for a smoke check with available files only:\n"
            f"{missing_paths}"
        )

    if not result_specs:
        details = ""
        if skipped:
            details = "\nSkipped inputs:\n" + "\n".join(
                f"- {item['path']}: {item['reason']}" for item in skipped
            )
        raise FileNotFoundError("No saved restart summaries were available." + details)

    return result_specs, missing, skipped


def _make_fixed_diagonal_guide(mean, sd, site_specs):
    """
    Build a fixed mean-field guide from a flattened unconstrained mean/std vector.

    Positive model sites are represented as exp(Z), where Z is Normal with the
    saved unconstrained location and scale. This matches the log-scale
    parameterization used by the package notebooks.
    """
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist

    mean = jnp.asarray(mean)
    sd = jnp.maximum(jnp.asarray(sd), 1e-8)

    def guide(*args, **kwargs):
        del args, kwargs
        for spec in site_specs:
            site_slice = spec["slice"]
            base = dist.Normal(mean[site_slice], sd[site_slice])
            if spec.get("positive", False):
                distribution = dist.TransformedDistribution(
                    base, dist.transforms.ExpTransform()
                )
            else:
                distribution = base
            if spec.get("event_dim", 0):
                distribution = distribution.to_event(spec["event_dim"])
            numpyro.sample(spec["name"], distribution)

    return guide


def _make_price_radon_fixed_guide(mean, sd, num_counties):
    theta_start = 3
    theta_stop = theta_start + num_counties
    return _make_fixed_diagonal_guide(
        mean,
        sd,
        [
            {"name": "mu", "slice": 0},
            {"name": "kappa_sq", "slice": 1, "positive": True},
            {"name": "sigma_sq", "slice": 2, "positive": True},
            {"name": "theta", "slice": slice(theta_start, theta_stop), "event_dim": 1},
        ],
    )


def _numpyro_model_args(data):
    import jax.numpy as jnp

    return (
        jnp.asarray(data["log_radon_bq_adjusted"]),
        jnp.asarray(data["floor"]),
        jnp.asarray(data["log_uranium"]),
        jnp.asarray(data["county"]),
        jnp.asarray(data["floor_by_county"]),
    )


def _price_numpyro_score(mean, std, data, grad_samps, seed, **_kwargs):
    import jax
    from numpyro.infer import TraceMeanField_ELBO

    guide = _make_price_radon_fixed_guide(mean, std, data["num_counties"])
    elbo = TraceMeanField_ELBO(num_particles=int(grad_samps))
    loss = elbo.loss(
        jax.random.PRNGKey(int(seed)),
        {},
        data["model"],
        guide,
        *data["model_args"],
    )
    return float(-loss)


def _validate_scored_runs(best_info, param_names):
    for row in best_info["all_scored_runs"]:
        if np.asarray(row["mean"]).shape[-1] != len(param_names):
            raise ValueError(
                f"Parameter dimension mismatch in {row['source_file']}: "
                f"{np.asarray(row['mean']).shape[-1]} vs {len(param_names)}"
            )


def _build_reference_frame(best_info, param_names):
    package, setting = _split_method_label(best_info["method"])
    source = (
        "Saved VI NumPyro-ELBO best "
        f"({package} {setting} restart {best_info['restart_idx']})"
    )
    rows = []
    for idx, parameter in enumerate(param_names):
        rows.append(
            {
                "parameter": parameter,
                "mean": float(best_info["best_mean"][idx]),
                "sd": float(best_info["best_std"][idx]),
                "source": source,
                "model": "price_nero_gelman_1996",
                "scale": "unconstrained_for_positive_variances",
                "best_package": package,
                "best_setting": setting,
                "best_restart_idx": int(best_info["restart_idx"]),
                "selection_score": float(best_info["best_elbo_np"]),
            }
        )
    return pd.DataFrame(rows)


def _build_diagnostics_frame(best_info, param_names):
    focal_indices = [param_names.index(name) for name in FOCAL_PARAMETERS if name in param_names]
    focal_names = [param_names[idx] for idx in focal_indices]
    rows = []
    for candidate_idx, row_info in enumerate(best_info["all_scored_runs"]):
        package, setting = _split_method_label(row_info["method"])
        row = {
            "candidate_idx": candidate_idx,
            "package": package,
            "setting": setting,
            "restart_idx": int(row_info["restart_idx"]),
            "input_file": row_info["source_file"],
            "n_records": row_info["n_records"],
            "selection_score": float(row_info["best_elbo_np"]),
            "numpyro_elbo": float(row_info["best_elbo_np"]),
            "selected_best": candidate_idx == best_info["best_candidate_idx"],
        }
        for name, idx in zip(focal_names, focal_indices):
            row[f"{name}_mean"] = float(row_info["mean"][idx])
            row[f"{name}_sd"] = float(row_info["std"][idx])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute a shared variational reference for the Price/Nero/Gelman radon model "
            "from saved processed NumPyro, TFP, and PyMC restart outputs."
        )
    )
    parser.add_argument("--output", default="best_reference_values.csv")
    parser.add_argument("--diagnostics-output", default="best_reference_restarts.csv")
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Use available saved outputs only. By default all expected package outputs are required.",
    )
    parser.add_argument(
        "--exclude-package",
        action="append",
        default=[],
        help="Exclude one package from reference selection. May be passed multiple times.",
    )
    parser.add_argument(
        "--elbo-particles",
        type=int,
        default=4096,
        help="Number of NumPyro ELBO particles used to score each fixed guide.",
    )
    parser.add_argument(
        "--elbo-seed",
        type=int,
        default=0,
        help="Base random seed for NumPyro ELBO scoring.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data = load_radon_data()
    param_names = make_price_radon_param_names(data["county_names"], unconstrained=True)
    result_specs, missing, skipped = _prepare_result_specs(
        base_dir,
        allow_missing=args.allow_missing,
        excluded_packages=args.exclude_package,
    )
    score_data = {
        "model": build_numpyro_price_radon_model(),
        "model_args": _numpyro_model_args(data),
        "num_counties": len(data["county_names"]),
    }

    best_info = find_best_params(
        file_name_list=[path for _, path in result_specs],
        alpha_prior=None,
        obs_counts=score_data,
        grad_samps=args.elbo_particles,
        seed=args.elbo_seed,
        score_fn=_price_numpyro_score,
        result_specs=result_specs,
        processed_mc_settings=("100_mc",),
        score_seed_by_restart=False,
    )
    _validate_scored_runs(best_info, param_names)
    reference = _build_reference_frame(best_info, param_names)
    diagnostics = _build_diagnostics_frame(best_info, param_names)

    output = _resolve_output(args.output, base_dir)
    diagnostics_output = _resolve_output(args.diagnostics_output, base_dir)
    reference.to_csv(output, index=False)
    diagnostics.to_csv(diagnostics_output, index=False)

    eta_mean, eta_sd = price_theta_summary_from_unconstrained(
        best_info["best_mean"],
        best_info["best_std"],
    )
    eta_summary = {
        "mu": {"mean": float(eta_mean[0]), "sd": float(eta_sd[0])},
        "kappa_sq": {"mean": float(eta_mean[1]), "sd": float(eta_sd[1])},
        "sigma_sq": {"mean": float(eta_mean[2]), "sd": float(eta_sd[2])},
    }
    best_package, best_setting = _split_method_label(best_info["method"])
    metadata = {
        "method": "find_best_params_saved_outputs_numpyro_elbo",
        "model": "price_nero_gelman_1996",
        "included_settings": [
            {
                "package": _split_method_label(row["method"])[0],
                "setting": _split_method_label(row["method"])[1],
                "restart_idx": int(row["restart_idx"]),
                "input_file": row["source_file"],
            }
            for row in best_info["all_scored_runs"]
        ],
        "missing_inputs": missing,
        "skipped_inputs": skipped,
        "excluded_packages": sorted(set(args.exclude_package)),
        "selection_metric": "maximum NumPyro TraceMeanField_ELBO for fixed saved diagonal guide",
        "elbo_particles": int(args.elbo_particles),
        "elbo_seed": int(args.elbo_seed),
        "best_package": best_package,
        "best_setting": best_setting,
        "best_restart_idx": int(best_info["restart_idx"]),
        "selection_score": float(best_info["best_elbo_np"]),
        "best_numpyro_elbo": float(best_info["best_elbo_np"]),
        "eta_summary_constrained": eta_summary,
        "output": str(output),
        "diagnostics_output": str(diagnostics_output),
    }
    output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))

    selected = reference.set_index("parameter")
    print(f"Wrote {output}")
    print(f"Wrote {diagnostics_output}")
    print(diagnostics)
    print(selected.loc[[name for name in FOCAL_PARAMETERS if name in selected.index], ["mean", "sd"]])
    print("Constrained eta summary:", eta_summary)


if __name__ == "__main__":
    main()

from pathlib import Path
import argparse
import sys

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import TraceMeanField_ELBO


sys.path.append(str(Path(__file__).resolve().parents[2]))

from modulars.logistic_regression import load_logistic_regression_data
from modulars.utils import find_best_params


RESULT_SPECS = [
    ("numpyro", "results/numpyro_processed_restarts.csv"),
    ("tfp", "results/tfp_processed_restarts.csv"),
    ("pymc_default", "results/pymc_default_processed_restarts.csv"),
    ("pymc_adam", "results/pymc_adam_processed_restarts.csv"),
]


def numpyro_logistic_regression_model(data):
    x, y = data
    n_coeffs = x.shape[1]
    b = numpyro.sample(
        "b",
        dist.Normal(jnp.zeros(n_coeffs), jnp.ones(n_coeffs)).to_event(1),
    )
    logits = jnp.matmul(x, b)
    numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=y)


def make_numpyro_fixed_diag_guide(loc, scale):
    loc = jnp.asarray(loc, dtype=jnp.float32)
    scale = jnp.clip(jnp.asarray(scale, dtype=jnp.float32), 1e-8)

    def guide(data):
        numpyro.sample("b", dist.Normal(loc, scale).to_event(1))

    return guide


def compute_numpyro_elbo(loc, scale, data, grad_samps=50_000, seed=20240512):
    guide = make_numpyro_fixed_diag_guide(loc, scale)
    elbo = TraceMeanField_ELBO(num_particles=int(grad_samps))
    rng_key = jax.random.PRNGKey(seed)
    loss_value = elbo.loss(
        rng_key,
        {},
        numpyro_logistic_regression_model,
        guide,
        data,
    )
    return float(-loss_value)


def _logistic_numpyro_score(mean, std, data, grad_samps, seed, **_kwargs):
    return compute_numpyro_elbo(
        mean,
        std,
        data,
        grad_samps=grad_samps,
        seed=seed,
    )


def main(ignore_pymc=False, grad_samps=50_000, seed=20240512):
    base_dir = Path(__file__).resolve().parent
    data = load_logistic_regression_data()
    numpyro_data = (
        jnp.asarray(data["x_train"], dtype=jnp.float32),
        jnp.asarray(data["y_train"], dtype=jnp.float32),
    )

    active_specs = []
    for method, rel_path in RESULT_SPECS:
        if ignore_pymc and method.startswith("pymc"):
            print(f"skipping {method} because --ignore-pymc was set")
            continue
        path = base_dir / rel_path
        if not path.exists():
            print(f"skipping missing result file: {path}")
            continue
        active_specs.append((method, str(path)))

    if len(active_specs) == 0:
        raise FileNotFoundError(
            "we did not find any processed restart files. Run the method notebooks first."
        )

    best_info = find_best_params(
        file_name_list=[path for _, path in active_specs],
        alpha_prior=None,
        obs_counts=numpyro_data,
        grad_samps=grad_samps,
        seed=seed,
        score_fn=_logistic_numpyro_score,
        result_specs=active_specs,
    )
    all_rows = best_info["all_scored_runs"]
    scored = pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in ("mean", "cov", "std", "raw")
            }
            for row in all_rows
        ]
    )

    reference = pd.DataFrame(
        {
            "parameter": data["labels"],
            "mean": np.asarray(best_info["best_mean"], dtype=float),
            "sd": np.asarray(best_info["best_std"], dtype=float),
            "method": best_info["method"],
            "source_file": best_info["source_file"],
            "restart_idx": best_info["restart_idx"],
            "mc_setting": best_info["mc_setting"],
            "best_elbo_np": best_info["best_elbo_np"],
        }
    )

    scored.to_csv(base_dir / "best_variational_reference_scored_restarts.csv", index=False)
    reference.to_csv(base_dir / "best_variational_reference.csv", index=False)
    print(reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ignore-pymc",
        action="store_true",
        help="skip pymc_default and pymc_adam processed restart files",
    )
    parser.add_argument(
        "--grad-samps",
        type=int,
        default=50_000,
        help="number of NumPyro TraceMeanField_ELBO particles for scoring each candidate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20240512,
        help="base random seed for NumPyro ELBO scoring",
    )
    args = parser.parse_args()
    main(ignore_pymc=args.ignore_pymc, grad_samps=args.grad_samps, seed=args.seed)

"""Numpyro-style ELBO check for Stan's grad100_tol_rel_obj1e-6 scenario.

Scores Stan's tightest-tolerance scenario's final variational approximation
with the same fixed-guide numpyro ELBO metric (see compute_best_reference.py:
_make_radon_fixed_guide / _radon_numpyro_score, TraceMeanField_ELBO with
elbo_particles=4096, elbo_seed=0) used to pick the "best_reference_values.csv"
reference recorded in best_reference_values.metadata.json, so the two numbers
are directly comparable.

Stan's own mean-field ADVI output is already in the same unconstrained
parameterization (see LOG_COLUMNS=["sigma_a", "sd_y"] applied in
rr_stan_radon.ipynb / modulars.stan_rr_test.stan_result_by_scenario), so no
rerun is needed -- this just rescoring already-saved trajectories.

This needs jax/numpyro, which are not installed in the "stan3" conda
environment this notebook otherwise runs in -- rr_stan_radon.ipynb invokes
this script as a subprocess under a separate interpreter that has them (see
the ELBO-check cell), the same way it shells out to the `stan` CmdStan binary
rather than importing it.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path("/Users/madelynandersen/Documents/VB Bakeoff/simpleVI")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NB_DIR = REPO_ROOT / "single_MC/radon_example/pymc_tutorial"
sys.path.insert(0, str(NB_DIR))

from modulars.radon import build_numpyro_radon_model, load_radon_data, make_radon_unconstrained_param_names
from modulars.stan_rr_test import stan_result_by_scenario
from compute_best_reference import _make_radon_fixed_guide, _numpyro_model_args

import jax
from numpyro.infer import TraceMeanField_ELBO

ELBO_PARTICLES = 4096  # matches best_reference_values.metadata.json's elbo_particles
ELBO_SEED = 0  # matches best_reference_values.metadata.json's elbo_seed
N_RESTARTS_CHECK = 3  # how many Stan restarts of the scenario to score

data = load_radon_data()
county_names = data["county_names"]
num_counties = len(county_names)
param_names = make_radon_unconstrained_param_names(county_names)
model = build_numpyro_radon_model()
model_args = _numpyro_model_args(data)

with open(NB_DIR / "best_reference_values.metadata.json") as f:
    ref_metadata = json.load(f)
reference_elbo = ref_metadata["best_numpyro_elbo"]

with open(NB_DIR / "results/stan_result.pkl", "rb") as f:
    stan_result = pickle.load(f)
by_scenario = stan_result_by_scenario(stan_result)


def score(mean, std, seed):
    guide = _make_radon_fixed_guide(mean, std, num_counties)
    elbo = TraceMeanField_ELBO(num_particles=ELBO_PARTICLES)
    loss = elbo.loss(jax.random.PRNGKey(int(seed)), {}, model, guide, *model_args)
    return float(-loss)


print("===== rr_stan_radon: numpyro-style ELBO check vs best_reference_values.csv =====")
print(f"reference best_numpyro_elbo (from {ref_metadata['best_package']}::{ref_metadata['best_setting']}, "
      f"restart {ref_metadata['best_restart_idx']}): {reference_elbo:.4f}")
print(f"  (elbo_particles={ref_metadata['elbo_particles']}, elbo_seed={ref_metadata['elbo_seed']})")
print()

for scenario_name in ("grad100_tol_rel_obj1e-6", "grad100_default_rest"):
    means, stds, iterations = by_scenario[scenario_name]
    n_restarts = means.shape[0]
    print(f"--- Stan {scenario_name} (final iter={int(iterations[-1])}, {n_restarts} restarts) ---")
    for restart_idx in range(min(N_RESTARTS_CHECK, n_restarts)):
        mean = means[restart_idx, -1, :].astype(float)
        std = stds[restart_idx, -1, :].astype(float)
        elbo_val = score(mean, std, seed=ELBO_SEED)
        print(f"  restart {restart_idx}: numpyro-style ELBO = {elbo_val:10.4f}   "
              f"(vs best {reference_elbo:.4f}, diff={elbo_val - reference_elbo:+.4f})")
    print()

with open(NB_DIR / "tol_1e-6_default_rest_result.pkl", "rb") as f:
    td_result = pickle.load(f)
td_by_scenario = stan_result_by_scenario(td_result)
td_means, td_stds, td_iterations = td_by_scenario["tol_1e-6_default_rest"]
n_td_restarts = td_means.shape[0]
print(f"--- Stan tol_1e-6_default_rest (final iter={int(td_iterations[-1])}, {n_td_restarts} restarts) ---")
for restart_idx in range(min(N_RESTARTS_CHECK, n_td_restarts)):
    mean = td_means[restart_idx, -1, :].astype(float)
    std = td_stds[restart_idx, -1, :].astype(float)
    elbo_val = score(mean, std, seed=ELBO_SEED)
    print(f"  restart {restart_idx}: numpyro-style ELBO = {elbo_val:10.4f}   "
          f"(vs best {reference_elbo:.4f}, diff={elbo_val - reference_elbo:+.4f})")

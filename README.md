# simpleVI

Code and results for "The Curious Case of the Default Settings: Evaluating Default Performance of Variational
Inference Software" (citation incoming).

To reproduce a figure or table from the manuscript, see [Reproducing the manuscript](#reproducing-the-manuscript).

simpleVI is a modular framework for running and comparing variational inference (mean-field ADVI) across
PyMC, NumPyro, TensorFlow Probability (TFP) and Stan (via CmdStanPy in Python; rstan and cmdstanr in R for the
Journé et al. reanalysis). Experiments not described in the manuscript are not guaranteed to be stable.

## Installation

The Python experiments use four conda environments, one per package. Each is recorded in `envs/`
(exported on macOS, Apple silicon, with `conda env export --no-builds`):

| Environment file | Used for |
|---|---|
| `envs/pymc_env_pm.yml` | PyMC 5.25.1 (PyTensor 2.31.7) |
| `envs/num_pyro.yml` | NumPyro 0.18.0 (JAX 0.4.35) |
| `envs/tfp_2.yml` | TensorFlow Probability 0.25.0 (TensorFlow nightly 2.20.0.dev20250616), and the plotting notebooks |
| `envs/stan3.yml` | CmdStanPy 1.3.0 with CmdStan 2.39.0 |

Create each environment and install this repository into it, from the repository root:

```bash
conda env create -f envs/pymc_env_pm.yml   # likewise for num_pyro, tfp_2 and stan3
conda activate pymc_env_pm
pip install -e .
```

The R parts of `Perennial_reproduction/` use R 4.4.1 with brms 2.23.0, rstan 2.32.7 and cmdstanr 0.9.0 (CmdStan
2.40.0); every R package version is listed in `Perennial_reproduction/R_package_versions.csv`.

Notes:

- `--no-builds` leaves out platform-specific build strings, so the files are more likely to resolve on other
  platforms, but some exact versions (e.g. `jaxlib`) may not be available everywhere.
- `envs/tfp_2.yml` contains a dated TensorFlow nightly build (the version our TFP results were run with), which
  may eventually be removed from PyPI.
- On macOS, PyTensor (PyMC's backend) is set to use `/usr/bin/clang++` when `modulars` is imported in an
  environment with PyMC. To use a different compiler, set the environment variable `SIMPLEVI_PYTENSOR_CXX` (or
  `cxx=` in `PYTENSOR_FLAGS`).
- The Stan runners use the CmdStan installed in the active conda environment. To use a different CmdStan, set
  `SIMPLEVI_CMDSTAN_PATH`.

## Repository layout

```
simpleVI/
├── README.md, LICENSE, CITATION.cff
├── pyproject.toml, setup.py      # installs modulars/ (pip install -e .)
├── print_versions.py             # prints the installed package versions (Table 5)
├── envs/                         # the four conda environments
├── modulars/                     # shared code imported by every notebook and runner
├── paper_vignettes/
│   ├── overall_bias_runners/     # test-suite runners and heatmap code (Fig. 2, Figs. 14–17)
│   ├── overall_default_bias_comparisons.ipynb
│   ├── 3_3_tfp_transforms_matter.ipynb
│   ├── changing_default_transforms/
│   └── (other examples, not in the manuscript)
├── single_MC/                    # one folder per test model (App. E), plus a Stan demo
├── Perennial_reproduction/       # Journé et al. (2025) reanalysis (Fig. 1, Figs. 10–11)
└── tests/                        # unit tests
```

### `modulars/`

| File | Contents |
|---|---|
| `pymc_rr_test.py`, `numpyro_rr_test.py`, `tfp_rr_test.py`, `stan_rr_test.py` | Run random restarts of mean-field ADVI in each package and return the parameter trajectories. Stan only reports its final fit, so `stan_rr_test.py` reruns CmdStan at each tracked iteration count |
| `distributions.py` | Data generation and closed-form posteriors for the conjugate models |
| `elbo_computations.py` | NumPyro ELBO used to score restarts and pick the best variational approximation |
| `sampled_kl.py` | Monte Carlo KL(q ‖ posterior), comparable across packages with different transforms |
| `dirichlet.py`, `logistic_regression.py`, `radon.py` | Model-specific data loading, references and plots |
| `plot_rr.py` | Random-restart plots (two single restarts, then the 1 MC and 100 MC averages with ±1 SD bands) |
| `utils.py` | Configuration loading, saving and reading results, scoring candidate references |

### `paper_vignettes/overall_bias_runners/`

| File | Contents |
|---|---|
| `run_all_runners.sh` | Runs every package × model in its own conda environment; settings are in its header |
| `pymc_runner.py`, `numpyro_runner.py`, `tfp_runner.py`, `stan_runner.py` | One runner per package; each defines the test-suite models in that package |
| `bias_io.py` | Reads the saved runs and references and computes the bias and noise metrics |
| `bias_heatmaps.py` | Builds the tables and draws the heatmaps for `overall_default_bias_comparisons.ipynb` |
| `recompute_dirichlet_reference.py` | Recomputes the Multinomial–Dirichlet reference from the runner output (follows `rr_dirich_find_best.ipynb`) |

The runners write to `paper_vignettes/overall_bias_results/`, `overall_bias_results_full_default/` and
`overall_bias_results_1mc/`. These are not tracked in git, so on a fresh clone the runners have to be run
before `overall_default_bias_comparisons.ipynb` can plot anything.

### `single_MC/`

Each test-model folder holds the model's configuration (data seed, priors and, where there is no closed form,
the reference solution), its Stan program, single-run notebooks per package, and random-restart notebooks
(`rr_<package>_*.ipynb`, in `random_restarts/` where that subfolder exists).

| Folder | Model (App. E) | Reference |
|---|---|---|
| `1dgaussian_knownvar/` | E.1 Gaussian–Gaussian; also Fig. 4 | closed-form posterior (100 observations) |
| `1dbeta_binomial/` | E.2 Beta–Bernoulli | best variational approximation in `beta_config.json` (`compute_best_reference.py`; `random_restarts/elbo_space_stan.ipynb` walks Stan's path on the exact ELBO) |
| `gamma_poisson/` | E.3 Gamma–Poisson | closed-form log-normal VI optimum |
| `gamma_exponential/` | E.4 Gamma–Exponential | closed-form log-normal VI optimum |
| `invgamma_gaussianknownmean/` | E.5 Inverse-Gamma–Normal | best variational approximation in `invgamma_config.json` |
| `multinomial_dirichlet/` | E.6 Multinomial–Dirichlet; also Figs. 5–7 | best variational approximation in `dirichlet_config.json` (`rr_dirich_find_best.ipynb`) |
| `multidim_gaussian_knowncovar/` | E.7 High-dimensional (40-d) Gaussian–Gaussian | closed form (analytic posterior) |
| `logistic_regression/` | E.8 Bayesian logistic regression | `best_variational_reference.csv` (`compute_best_reference.py`) |
| `radon_example/` | E.9 Hierarchical radon model | `pymc_tutorial/best_reference_values.csv` (`compute_best_reference.py`); data in `data/` |
| `stan_conj1dgaussian_demo/` | Not in the test suite: a conjugate 1D Gaussian in Stan | — |

The models as fitted for Fig. 2 are defined in the runners (`paper_vignettes/overall_bias_runners/<package>_runner.py`),
which read the same configuration files.

### `Perennial_reproduction/`

Scripts in R and Python that refit the Table S1 model of Journé et al. (2025) with NUTS and with ADVI in every
package. Their README gives the run order and lists every file. The NUTS fit (`TableS1_reproduced.csv`) and the
global VI optimum (`TableS1_best_vi.csv`) are included, so the ADVI runs and figures can be made without
rerunning those two steps.

### `tests/`

Unit tests for transforms, log densities, references, KL estimates, saved-result round trips, defaults and
reproducibility. Packages that are not installed are skipped, so run the suite in each environment:
`python -m unittest discover -s tests -t .`

## Reproducing the manuscript

Paths are relative to this repository. The GitHub usage analysis (Tables 3 and 9, Fig. 9) is produced by the
companion repository [github-advi-usage](https://github.com/madelynandersen/github-advi-usage).

| Manuscript item | Code | Environment |
|---|---|---|
| Table 1 (default hyperparameters) | Package documentation and source; no code | — |
| Tables 2 and 4 (download counts) | Retrieved from PyPI download statistics and CRAN logs (`cranlogs`); no code in this repository | — |
| Table 3 (§4) and Table 9 (App. D): default usage on GitHub, by repository and by call site | `github_advi_usage/component_defaults.py` in [github-advi-usage](https://github.com/madelynandersen/github-advi-usage) | see that repository |
| Fig. 9 (App. D, TFP optimizer choices on GitHub) | `github_advi_usage/plot_tfp_adam.py` in [github-advi-usage](https://github.com/madelynandersen/github-advi-usage) | see that repository |
| Fig. 1 (§5), Figs. 10–11 (App. F) | `Perennial_reproduction/` (run order in its README). After the ADVI runs, `extra_figs.R` draws the figures from their output tables: `extra_figs/fig1_map2_map_1x2_full_defaults.png` (Fig. 1), `extra_figs/fig1_grid_2x4_full_defaults.png` (Fig. 10) and `extra_figs/fig1_grid_2x4_long_defaults.png` (Fig. 11) | R; `pymc_env_pm`, `num_pyro`, `tfp_2` |
| Fig. 2 (§6), Figs. 14–17 (App. H) | 1. `paper_vignettes/overall_bias_runners/run_all_runners.sh` runs the adjusted settings for every package × model and saves to `paper_vignettes/overall_bias_results/`; 2. `FULL_DEFAULT=1 ./run_all_runners.sh` runs the full-default fits into `paper_vignettes/overall_bias_results_full_default/`; 3. `Perennial_reproduction/run_journe_restarts.sh`, once as is and once with `FULL_DEFAULTS=1`, adds the Journé et al. row; 4. `paper_vignettes/overall_default_bias_comparisons.ipynb` draws the heatmaps from the saved results (`PLOT_ONLY = True` skips the runs). The TFP 1 MC, 10,000-iteration runs in the right-hand panel are read from `paper_vignettes/overall_bias_results_1mc/` | runners: one per package (set in the scripts); notebook: `tfp_2` |
| Fig. 3 (§7), Figs. 12–13 (App. G) | `paper_vignettes/3_3_tfp_transforms_matter.ipynb` | `tfp_2` |
| Fig. 4a (§8–9) | `single_MC/1dgaussian_knownvar/3_1_pymc_100obs.ipynb` (the manuscript uses 3 million iterations: set `SIMPLEVI_PYMC_DEMO_ITERS=3000000`; the default is 300,000) | `pymc_env_pm` |
| Fig. 4b (§8–9), Fig. 8 (App. C.3.1) | `single_MC/1dgaussian_knownvar/random_restarts/rr_stan_100obs.ipynb` (20 restarts; restart 9 is the one shown in Fig. 8 and left out of Fig. 4b) | `stan3` |
| Table 5 (versions) | `envs/*.yml`; `Perennial_reproduction/R_package_versions.csv` for R | — |
| Tables 6–8 (default transforms, App. C.1) | Package documentation and source; no code | — |
| Fig. 5 (App. C.2) | `single_MC/multinomial_dirichlet/rr_dirich_2d_transform_compare.ipynb` | `pymc_env_pm` |
| Figs. 6–7 and the Dirichlet ELBO comparison (App. C.2) | `single_MC/multinomial_dirichlet/simplex_transform_differences.ipynb` | `pymc_env_pm` |
| Stick-breaking transform for the Multinomial–Dirichlet in PyMC and Stan (Fig. 2, App. C.2) | `paper_vignettes/changing_default_transforms/change_simplex_transform_{pymc,numpyro,stan}.ipynb` | `pymc_env_pm`, `num_pyro`, `stan3` |
| Test-suite models (App. E) | `single_MC/<model>/` (see the [`single_MC/` table](#single_mc) for which folder is which model); the model code used for Fig. 2 is in `paper_vignettes/overall_bias_runners/<package>_runner.py` | — |

The runners take hours to days in total (most of it Stan, where every tracked iteration is a separate CmdStan
run). Each finished seed is cached, and rerunning `run_all_runners.sh` resumes where it stopped. Settings can
be changed with environment variables or flags; see the header of `run_all_runners.sh`.

## Other examples

These are not part of the manuscript.

- [Tracking parameters in constrained vs. unconstrained space](paper_vignettes/example_analysis.ipynb)
- Stan's ADVI defaults: [step-size adaptation, MC samples and tolerance](paper_vignettes/stan_default_demos.ipynb)
  and [convergence tolerance](paper_vignettes/stan_tolerance_demo.ipynb)
- [PyMC trajectories on the exact objective for a 1D Gaussian with no observations](paper_vignettes/1d_gaussian_no_obs_advi_optimizer_space.ipynb)
- Single runs of each package on the 1D Gaussian with known variance and no observations:
  `single_MC/1dgaussian_knownvar/3_1_{pymc,numpyro,tfp,stan}.ipynb`
- Random-restart analyses for each package on the 1D Gaussian with known variance:
  [PyMC](single_MC/1dgaussian_knownvar/random_restarts/rr_pymc.ipynb),
  [NumPyro](single_MC/1dgaussian_knownvar/random_restarts/rr_numpyro.ipynb),
  [TFP](single_MC/1dgaussian_knownvar/random_restarts/rr_tfp.ipynb),
  [Stan](single_MC/1dgaussian_knownvar/random_restarts/rr_stan.ipynb)
- Random-restart analyses for the other test models are in their [`single_MC`](single_MC) folders.
- [A conjugate 1D Gaussian in Stan](single_MC/stan_conj1dgaussian_demo/stan_conj1dgaussian_demo.ipynb)

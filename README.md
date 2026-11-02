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

| Path | Contents |
|---|---|
| `modulars/` | Reusable code: running random restarts in each package, transforms, references, plotting |
| `paper_vignettes/overall_bias_runners/` | One runner per package for the test-suite comparison, and `run_all_runners.sh`, which runs every package in its own conda environment |
| `paper_vignettes/overall_default_bias_comparisons.ipynb` | Reads the saved runner results and draws the full-default vs. adjusted-settings heatmaps (Fig. 2, Figs. 14–17) |
| `paper_vignettes/overall_bias_comparisons.ipynb` | Reads the saved adjusted-setting runs only and draws the same heatmaps without the full-default panel |
| `paper_vignettes/` (other notebooks) | Stand-alone examples; the ones used in the manuscript are listed below |
| `paper_vignettes/changing_default_transforms/` | Minimal examples of changing the simplex transform in PyMC, NumPyro and Stan |
| `single_MC/<model>/` | One folder per test model: its configuration, Stan model files, single-run notebooks, and random-restart notebooks (in `random_restarts/` where present) |
| `Perennial_reproduction/` | Reanalysis of Journé et al. (2025) with NUTS and ADVI (Fig. 1, Figs. 10–11); run order in its README |
| `envs/` | The four conda environments |
| `tests/` | Unit tests (`python -m unittest discover -s tests -t .`, run in each package's environment) |

The runner results (`paper_vignettes/overall_bias_results*/`) are not tracked in git, so on a fresh clone the
runners have to be run before the heatmap notebooks can plot anything.

## Reproducing the manuscript

Paths are relative to this repository. The GitHub usage analysis (Tables 3 and 9, Fig. 9) is produced by the
companion repository [github-advi-usage](https://github.com/madelynandersen/github-advi-usage).

| Manuscript item | Code | Environment |
|---|---|---|
| Table 1 (default hyperparameters) | Package documentation and source; no code | — |
| Tables 2 and 4 (download counts) | Retrieved from PyPI download statistics and CRAN logs (`cranlogs`); no code in this repository | — |
| Table 3 (§4) and Table 9 (App. D): default usage on GitHub, by repository and by call site | `github_advi_usage/component_defaults.py` in [github-advi-usage](https://github.com/madelynandersen/github-advi-usage) | see that repository |
| Fig. 9 (App. D, TFP optimizer choices on GitHub) | `github_advi_usage/plot_tfp_adam.py` in [github-advi-usage](https://github.com/madelynandersen/github-advi-usage) | see that repository |
| Fig. 1 (§5), Figs. 10–11 (App. F) | `Perennial_reproduction/` (run order in its README). `extra_figs.R` draws the figures from the saved tables: `extra_figs/fig1_map2_map_1x2_full_defaults.png` (Fig. 1), `extra_figs/fig1_grid_2x4_full_defaults.png` (Fig. 10) and `extra_figs/fig1_grid_2x4_long_defaults.png` (Fig. 11) | R; `pymc_env_pm`, `num_pyro`, `tfp_2` |
| Fig. 2 (§6), Figs. 14–17 (App. H) | 1. `paper_vignettes/overall_bias_runners/run_all_runners.sh` runs the adjusted settings for every package × model and saves to `paper_vignettes/overall_bias_results/`; 2. `FULL_DEFAULT=1 ./run_all_runners.sh` runs the full-default fits into `paper_vignettes/overall_bias_results_full_default/`; 3. `Perennial_reproduction/run_journe_restarts.sh`, once as is and once with `FULL_DEFAULTS=1`, adds the Journé et al. row; 4. `paper_vignettes/overall_default_bias_comparisons.ipynb` draws the heatmaps from the saved results (`PLOT_ONLY = True` skips the runs). The TFP 1 MC, 10,000-iteration runs in the right-hand panel are read from `paper_vignettes/overall_bias_results_1mc/` | runners: one per package (set in the scripts); notebook: `tfp_2` |
| Fig. 3 (§7), Figs. 12–13 (App. G) | `paper_vignettes/3_3_tfp_transforms_matter.ipynb` | `tfp_2` |
| Fig. 4a (§8–9) | `single_MC/1dgaussian_knownvar/3_1_pymc_100obs.ipynb` (the manuscript uses 3 million iterations: set `SIMPLEVI_PYMC_DEMO_ITERS=3000000`; the default is 300,000) | `pymc_env_pm` |
| Fig. 4b (§8–9), Fig. 8 (App. C.3.1) | `single_MC/1dgaussian_knownvar/random_restarts/rr_stan_100obs.ipynb` (20 restarts; restart 9 is the one shown in Fig. 8 and left out of Fig. 4b) | `stan3` |
| Table 5 (versions) | `envs/*.yml`; `Perennial_reproduction/R_package_versions.csv` for R | — |
| Tables 6–8 (default transforms, App. C.1) | Package documentation and source; no code | — |
| Fig. 5 (App. C.2) | `single_MC/multinomial_dirichlet/rr_dirich_2d_transform_compare.ipynb` | `pymc_env_pm` |
| Figs. 6–7 and the Dirichlet ELBO comparison (App. C.2) | `single_MC/multinomial_dirichlet/simplex_transform_differences.ipynb` | `pymc_env_pm` |
| Stick-breaking transform for the Multinomial–Dirichlet in PyMC and Stan (Fig. 2, App. C.2) | `paper_vignettes/changing_default_transforms/change_simplex_transform_{pymc,numpyro,stan}.ipynb` | `pymc_env_pm`, `num_pyro`, `stan3` |
| Test-suite models (App. E) | `single_MC/<model>/` configurations; the model code used for Fig. 2 is in `paper_vignettes/overall_bias_runners/<package>_runner.py` | — |

The runners take hours to days in total (most of it Stan, where every tracked iteration is a separate CmdStan
run). Each finished seed is cached, and rerunning `run_all_runners.sh` resumes where it stopped. Settings can
be changed with environment variables or flags; see the header of `run_all_runners.sh`.

## Other examples

These are not part of the manuscript.

- [Tracking parameters in constrained vs. unconstrained space](paper_vignettes/example_analysis.ipynb)
- Default initialization in each package: `paper_vignettes/3_2_{pymc,numpyro,tfp}_initialization.ipynb` and
  `paper_vignettes/3_2_pymc_support_points.ipynb`
- [Initialization sensitivity on a bimodal beta in NumPyro](paper_vignettes/3_5_numpyro_betamix_initialization_challenges.ipynb)
- Stan's ADVI defaults: [step-size adaptation, MC samples and tolerance](paper_vignettes/stan_default_demos.ipynb)
  and [convergence tolerance](paper_vignettes/stan_tolerance_demo.ipynb)
- [PyMC trajectories on the exact objective for a 1D Gaussian with no observations](paper_vignettes/1d_gaussian_no_obs_advi_optimizer_space.ipynb)
- Single runs of each package on the 1D Gaussian with known variance and no observations:
  `single_MC/1dgaussian_knownvar/3_1_{pymc,numpyro,tfp,stan}.ipynb`
- Random-restart analyses for each package on the 1D Gaussian with known variance:
  [PyMC](single_MC/1dgaussian_knownvar/random_restarts/rr_pymc.ipynb),
  [NumPyro](single_MC/1dgaussian_knownvar/random_restarts/rr_numpyro.ipynb),
  [TFP](single_MC/1dgaussian_knownvar/random_restarts/rr_tfp.ipynb),
  [Stan](single_MC/1dgaussian_knownvar/random_restarts/rr_stan.ipynb),
  [Pyro](single_MC/1dgaussian_knownvar/random_restarts/rr_pyro.ipynb) (Pyro is not part of the manuscript)
- Random-restart analyses for the other test models are in their [`single_MC`](single_MC) folders.

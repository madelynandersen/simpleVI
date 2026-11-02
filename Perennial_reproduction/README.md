# Reproducing Table S1 of Journé et al. 2025 (Nat Commun 16:9226) with NUTS and variational inference

The model is the Dirichlet regression behind Table S1: `Y | weights(1/RMSE) ~ MAT + poly(MAP, 2) + (1 | Genus)`,
fitted with brms. This folder runs the authors' own NUTS code, then fits the same model with Stan in R and PyMC in Python, and compares everything with the R NUTS fit written by Journé et al.


## Files

| File | Role |
|---|---|
| `Journe_global_cues.R` | The original script from Journé et al. (from OSF vaucp). |
| `species.ecoregion.data.csv` | Input data for the model (from OSF vaucp). |
| `run_repro.R` | Runs lines 20-30 and 58-65 of the authors' script (the Table S1 model, NUTS).* |
| `run_repro_vi.R` | R ADVI (rstan), default and adjusted settings, compared with the NUTS fit* |
| `pymc_reproducing.py` | PyMC copy of the model: NUTS and three ADVI setups* |
| `best_vi_stan.R`, `best_vi.py` | Best-variational-approximation search (50 Stan + 50 PyMC restarts, ranked by NumPyro ELBO)* |
| `run_best_vi_parallel.sh` | Runs steps 4–6 in parallel jobs, then step 7* |
| `compare_plots.R` | Figures and table comparing all runs with R NUTS* |
| `extra_figs.R` | Optional alternative layouts of Figure 1, from the saved tables (writes `extra_figs/`)* |
| `stan_reproducing.py`, `export_stan_model.R` | CmdStanPy version of the R Stan ADVI fits (default and 100 MC, 25,000 iterations), run on the brms-generated Stan program and data that `export_stan_model.R` writes to `stan_model/`* |
| `journe_restarts.py`, `run_journe_restarts.sh` | `--restarts` mode of the PyMC, NumPyro/TFP and Stan scripts: random restarts for the "Journé et al. example" row of `paper_vignettes/overall_bias_comparisons.ipynb`* |
| `seed_spread.py` | Optional: spread of default PyMC ADVI over 50 seeds (not used in the manuscript)* |
| `R_package_versions.csv` | R version, CmdStan version and every R package (with dependencies) used by these scripts* |
| `model.coefficients.best.csv`, `output.climwin.csv`, `relative.contribution.sitelevel.csv`, `renv.lock` | Other files from the OSF archive; not used by these scripts |

*Files authored by Madelyn Andersen
## Setup (once)

- **R packages:** `tidyverse`, `brms`, `rstan`; for the best-VI search also `cmdstanr` plus CmdStan.
  Install cmdstanr (Xcode command-line tools first: `xcode-select --install`):
  ```r
  install.packages("cmdstanr", repos = c("https://stan-dev.r-universe.dev", getOption("repos")))
  library(cmdstanr); check_cmdstan_toolchain(fix = TRUE); install_cmdstan(cores = 4); cmdstan_version()
  ```
  `run_repro.R` uses cmdstanr automatically when it is installed (the authors' own backend); otherwise it falls back to rstan and says so.
- **Python:** `pymc` (5 or later), `numpy`, `pandas`; for stage 6 also `numpyro` (`pip install numpyro`).
  If PyMC's C compiler fails on your Mac, `pymc_reproducing.py` sets `PYTENSOR_FLAGS` to Apple's compiler at the top of the file.

## Run order

Steps 2-5 need `TableS1_reproduced.csv` from step 1 for their comparison tables. Step 6 needs the files from steps 4 and/or 5 (it scores whichever exist). Step 7 needs step 1; every other table is optional.

1. **`run_repro.R`** (R, Source). The authors' NUTS fit.
   Writes `TableS1_reproduced.csv` (the reference for everything else) and `brms_model_spec.txt`.
   The original Journé et al. `brm()` call has no seed, so `set.seed(1)` is called first and recorded in the table.
2. **`run_repro_vi.R`** (R, Source). rstan ADVI, default and adjusted (100 gradient samples).
   Writes `TableS1_VI_default.csv`, `TableS1_VI_adjusted.csv` and `TableS1_VI_{default,adjusted}_vs_NUTS.csv`.
   Output is silent while it runs (`refresh = 0`); the adjusted fit may take longer than the default fit.
3. **`python pymc_reproducing.py`** (Terminal). PyMC NUTS, default ADVI, ADVI with 100 Monte-Carlo samples, ADVI with Adam.
   Writes `TableS1_py_{mcmc,vi_default,vi_mc100,vi_adam}.csv` and matching `_vs_NUTS.csv` files.
   Run only some fits with `--only vi_default,vi_adam` command line flags. First check the priors in the script header against `brms_model_spec.txt`.
4. **`Rscript best_vi_stan.R`** (or Source in RStudio). 50 mean-field ADVI restarts of the brms Stan model through cmdstanr
   (100 gradient samples). Writes `best_vi/stan_restarts_moments.csv`, or, when given a seed range
   (`Rscript best_vi_stan.R 1 5`), one part file per range (`best_vi/stan_restarts_moments_part_001-005.csv`).
5. **`python best_vi.py pymc`**. 50 PyMC ADVI restarts (100 Monte-Carlo samples, 50,000 iterations; see `--help`).
   Writes `best_vi/pymc_restarts_moments.csv`, or, with `--first`/`--last`, one part file per seed range.
6. **`python best_vi.py score`**. Scores every Stan and PyMC restart (main and part files) with NumPyro's
   `TraceMeanField_ELBO` (50,000 particles, seed 20240512, same scoring idea as the simpleVI code) and keeps the highest ELBO.
   Writes `best_vi/restart_elbos.csv` and `TableS1_best_vi.csv` (winner's package and seed are in its `method` and `seed` columns).
   For the manuscript, the best of the 100 restarts was PyMC seed 27; the best Stan restart (seed 7) was 0.017 nats lower.

Steps 4–7 can also be run together, in parallel, with
`nohup caffeinate -i bash run_best_vi_parallel.sh > best_vi/logs/run.log 2>&1 &`.
7. **`compare_plots.R`** (R, Source). Reads every table above (any missing optional file is skipped with a message).
   Writes `fig1_coefficients.png` (forest plot plus table of offsets and SD errors), `fig2_effect_curves.png`
   and `comparison_table_all_vs_NUTS.csv`.

Minimal run (no best-VI search): steps 1, 2, 3, 7.

## Random restarts for the overall bias comparison (`--restarts`)

The "Journé et al. example" row of `paper_vignettes/overall_bias_comparisons.ipynb` uses random restarts of every
ADVI setup, saving only the mean and SD of the 8 Table S1 coefficients over the last 500 iterations of each run
(to `paper_vignettes/overall_bias_results/journe/`). All four packages are compared with `TableS1_best_vi.csv` (step 6).

| Package | Setups | Iterations | Seeds |
|---|---|---|---|
| PyMC (`python pymc_reproducing.py --restarts`) | default, 100 MC, Adam, Adam + 100 MC | 75,000 | 0..9 |
| NumPyro (`python numpyro_tfp_reproducing.py --backend numpyro --restarts`) | Adam(5e-4), 1 and 100 MC | 75,000 | 0..9 |
| TFP (`python numpyro_tfp_reproducing.py --backend tfp --restarts`) | with transforms, 1 and 100 MC (XLA-compiled) | 75,000 | 0..9 |
| Stan (`Rscript export_stan_model.R`, then `python stan_reproducing.py --restarts`) | default and 100 MC, `tol_rel_obj=1e-6`, 10,000 output draws | 25,000 | 0..4 |

Stan only reports its final fit, so each tracked Stan iteration is a separate CmdStan rerun; to keep that affordable
Stan tracks every 10th of the last 500 iterations (24,510, 24,520, ..., 25,000: 50 reruns, `--track-every`), and the
notebook averages over those 50.
`run_journe_restarts.sh` runs PyMC, NumPyro and TFP in parallel and then Stan, each in its conda environment:
`nohup caffeinate -i bash run_journe_restarts.sh > journe_restarts.log 2>&1 &`. Finished seeds are cached, so a
rerun resumes.

**Full defaults** (`--restarts --full_defaults`, or `FULL_DEFAULTS=1 bash run_journe_restarts.sh`): the left-hand
panel of `paper_vignettes/overall_default_bias_comparisons.ipynb`. Only each package's out-of-the-box fit, 10 seeds
(0..9), the last 500 iterations, every fit timed (`seed_seconds` in `run_info.json`), saved as setup `full_default` to
`paper_vignettes/overall_bias_results_full_default/journe/`. The launcher runs the packages one after the other so
the times are comparable.

| Package | Full-default fit | Iterations |
|---|---|---|
| PyMC | 1 MC sample, `adagrad_window` (PyMC's defaults) | 10,000 |
| NumPyro | AutoNormal, 1 MC sample, Adam(5e-4) | 30,000 |
| TFP | 1 MC sample, `tf.optimizers.Adam()`, **no bijectors** (as `--full_defaults`) | 10,000 |
| Stan | every CmdStan ADVI default (`tol_rel_obj=0.01`, 1,000 output draws), every one of the last 500 iterations | 10,000 |

## Seeds

| Run | Seed |
|---|---|
| R NUTS | `set.seed(1)` before the authors' `brm()` call (R-level seed) |
| R ADVI | first seed in 1..10 that does not fail, recorded per table |
| PyMC NUTS | 1 (`--mcmc-seed`) |
| PyMC ADVI | first seed in 11..20 that does not fail (11 for all three runs), recorded per table; seeds start at 11 to keep the ADVI seeds separate from the NUTS seed (1). |
| Best VI | restart seeds 1..50 in each package; the winner (PyMC seed 27) is recorded in `TableS1_best_vi.csv` |

ADVI results depend on the seed, and the first successful seed is not checked for convergence.

## Licenses

- The scripts marked * above are by Madelyn Andersen and are released under the MIT License (see `LICENSE` at the root of this repository).
- `Journe_global_cues.R`, `species.ecoregion.data.csv`, `model.coefficients.best.csv`, `output.climwin.csv`,
  `relative.contribution.sitelevel.csv` and `renv.lock` are from Journé et al. (2025), *Weather drivers of reproductive
  variability in perennial plants and their implications for climate change risks*, Nature Communications 16:9226,
  OSF project https://osf.io/vaucp/, licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
  They are included unmodified.

## Timing analysis (`--timing_analysis`)

Times every variational setting of each package over seeds 11-20 (`--timing_seeds "11-20"` or `"11,13"` to change), plus Stan MCMC (NUTS) as a reference. One attempt per seed, no retries (failures are recorded); the time is the **total wall time of a seed** (fit + drawing the approximation + summarising), with no trajectory recording. No Table S1 files are written. Compile / JIT time sits in the first fits of the Python packages (see `run_order`); Stan's model is compiled once and logged on its own `stan_compile` row.

| Package | Command | Settings timed |
|---|---|---|
| PyMC | `python pymc_reproducing.py --timing_analysis` | `full_defaults` (10,000 iterations); `vi_default`, `vi_mc100`, `vi_adam`, `vi_adam_mc100` at `--iters` (default **150,000**) |
| NumPyro | `python numpyro_tfp_reproducing.py --backend numpyro --timing_analysis` | `default`, `mc100` (`--iters`, default 75,000) and `full_defaults` (30,000) |
| TFP | `python numpyro_tfp_reproducing.py --backend tfp --timing_analysis` | `default`, `mc100`, `bare`, `bare_mc100` (`--iters`, default 75,000) and `full_defaults` (10,000) |
| Stan, rstan (R) | `Rscript run_repro_vi.R --timing_analysis` | VI: `default`, `adjusted` (50,000 max iterations), `full_defaults` (10,000 max); MCMC: `mcmc` (NUTS, 4 chains x 10,000 iterations) |
| Stan, CmdStanPy | `python stan_reproducing.py --timing_analysis` (env `stan3`) | VI: `default`, `adjusted` (same settings as the rstan runs, `require_converged=False`) and `full_defaults` |

`--only a,b` restricts the settings. Outputs: `timing_<package>.csv` (one row per seed; `iterations_run` is the number of iterations actually run, read from the printed ELBO table for Stan VI), `timing_<package>_summary.csv`. Re-running a setting with the same iterations replaces its rows; other settings are kept.

`bash run_timing_analysis.sh` runs everything one package at a time and finishes with `Rscript make_timing_figs.R`, which writes `extra_figs/timing_analysis.png` (wall time per seed), `extra_figs/timing_per_iteration.png` (milliseconds per iteration) and `extra_figs/timing_summary_all.csv`. Layout: full defaults | defaults run for longer | adjusted settings | Stan MCMC; the legend separates Stan MCMC, Stan VI (rstan) and Stan VI (CmdStanPy).

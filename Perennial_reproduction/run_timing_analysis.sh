#!/usr/bin/env bash
# Timing analysis (--timing_analysis) of the variational fits for the Journe et al. Table S1 model: PyMC, NumPyro, TFP, Stan in R
# (rstan: VI and MCMC) and Stan in Python (CmdStanPy: VI), each in its own environment, ONE PACKAGE AT A TIME (so the packages do
# not compete for the CPU and the times are comparable). Every setting of each package is fitted for seeds 11-20; one attempt per
# seed; total wall time per seed. Settings: full defaults, defaults run for longer, adjusted settings (100 MC, Adam, ...) and Stan
# MCMC (NUTS, rstan); PyMC runs its ADVI setups at 150,000 iterations (plus its full defaults at 10,000).
# Writes timing_<package>.csv and timing_<package>_summary.csv in this folder (timing_pymc, timing_numpyro, timing_tfp,
# timing_stan, timing_stan_cmdstanpy) and a log per package in timing_logs/. Last step: make_timing_figs.R draws
# extra_figs/timing_analysis.png and extra_figs/timing_per_iteration.png and writes extra_figs/timing_summary_all.csv from those
# files. No Table S1 files are written.
#
# Usage (from anywhere):  nohup caffeinate -i bash run_timing_analysis.sh > timing_analysis.log 2>&1 &
# Settings (environment variables, all optional):
#   PACKAGES        which packages, in this order (default "pymc numpyro tfp stan cmdstanpy"; stan = rstan VI + MCMC)
#   TIMING_SEEDS    seeds, e.g. "11-20" (default) or "11,13,15"
#   PYMC_ITERS      PyMC ADVI iterations (default: the script's 150,000; PyMC full defaults always use PyMC's 10,000)
#   NP_TFP_ITERS    NumPyro / TFP steps for the non-full-default settings (default: the script's 75,000)
#   PYMC_ONLY / STAN_ONLY / CMDSTANPY_ONLY   re-run only some settings, e.g. PYMC_ONLY=vi_adam_mc100  STAN_ONLY=default,full_defaults
#                   (rows of the settings that are not re-run stay in the timing files)
#   CONDA_ENVS_DIR  (default ~/miniconda3/envs)   PYMC_ENV / NUMPYRO_ENV / TFP_ENV / STAN_ENV (pymc_env_pm / num_pyro / tfp_2 /
#                   stan3; STAN_ENV is the CmdStanPy environment)
# Examples:  PACKAGES="pymc stan" bash run_timing_analysis.sh
#            PYMC_ITERS=10000 NP_TFP_ITERS=10000 TIMING_SEEDS=11-15 bash run_timing_analysis.sh     (a much shorter run)
# TFP at 75,000 steps x 5 settings x 10 seeds, PyMC with 100 MC at 150,000 iterations and 10 Stan NUTS fits take a very long time;
# lower PYMC_ITERS / NP_TFP_ITERS or the seeds to cut it down.
# Stan in R runs with the Rscript on your PATH (as in run_best_vi_parallel.sh).

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
exec < /dev/null     # no job may read or configure your terminal (a background job that does gets "suspended (tty output)")

PACKAGES="${PACKAGES:-pymc numpyro tfp stan cmdstanpy}"
TIMING_SEEDS="${TIMING_SEEDS:-11-20}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$HOME/miniconda3/envs}"
mkdir -p timing_logs

run() {   # run <package> <command...>
    local package="$1"; shift
    local log="timing_logs/${package}.log"
    echo "[$(date '+%F %T')] $package -> $log"
    if "$@" > "$log" 2>&1; then
        echo "[$(date '+%F %T')] $package finished"
    else
        echo "[$(date '+%F %T')] FAILED $package (see $log; results of settings that finished are kept)"
    fi
}

for package in $PACKAGES; do
    case "$package" in
        pymc)
            ITERS=(); [[ -n "${PYMC_ITERS:-}" ]] && ITERS=(--iters "$PYMC_ITERS")
            ONLY=(); [[ -n "${PYMC_ONLY:-}" ]] && ONLY=(--only "$PYMC_ONLY")
            run pymc "$CONDA_ENVS_DIR/${PYMC_ENV:-pymc_env_pm}/bin/python" -u pymc_reproducing.py --timing_analysis \
                --timing_seeds "$TIMING_SEEDS" ${ITERS[@]+"${ITERS[@]}"} ${ONLY[@]+"${ONLY[@]}"} ;;
        numpyro|tfp)
            ENV_NAME="${NUMPYRO_ENV:-num_pyro}"; [[ "$package" == "tfp" ]] && ENV_NAME="${TFP_ENV:-tfp_2}"
            ITERS=(); [[ -n "${NP_TFP_ITERS:-}" ]] && ITERS=(--iters "$NP_TFP_ITERS")
            run "$package" "$CONDA_ENVS_DIR/$ENV_NAME/bin/python" -u numpyro_tfp_reproducing.py --backend "$package" \
                --timing_analysis --timing_seeds "$TIMING_SEEDS" ${ITERS[@]+"${ITERS[@]}"} ;;
        stan)
            ONLY=(); [[ -n "${STAN_ONLY:-}" ]] && ONLY=("--only=$STAN_ONLY")
            run stan Rscript run_repro_vi.R --timing_analysis "--timing_seeds=$TIMING_SEEDS" ${ONLY[@]+"${ONLY[@]}"} ;;
        cmdstanpy)
            [[ -f stan_model/journe_brms.stan ]] || Rscript export_stan_model.R
            ONLY=(); [[ -n "${CMDSTANPY_ONLY:-}" ]] && ONLY=(--only "$CMDSTANPY_ONLY")
            run stan_cmdstanpy "$CONDA_ENVS_DIR/${STAN_ENV:-stan3}/bin/python" -u stan_reproducing.py --timing_analysis \
                --timing_seeds "$TIMING_SEEDS" ${ONLY[@]+"${ONLY[@]}"} ;;
        *) echo "Unknown package '$package' (use: pymc numpyro tfp stan cmdstanpy)" ;;
    esac
done
# last step: figures and table of all the timing results (whatever timing_<package>.csv files exist)
run figures Rscript make_timing_figs.R
echo "[$(date '+%F %T')] done: extra_figs/timing_analysis.png, extra_figs/timing_per_iteration.png, extra_figs/timing_summary_all.csv"

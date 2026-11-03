#!/usr/bin/env bash
# Random restarts of the Table S1 model for paper_vignettes/overall_default_bias_comparisons.ipynb ("Journé et al. example"
# row): PyMC, NumPyro and TFP in parallel (75,000 iterations, 10 seeds), then Stan (25,000 iterations, 5 seeds,
# one CmdStan rerun per tracked iteration). Each step skips results that are already saved and resumes from the
# per-seed cache, so rerunning this script picks up where it stopped.
#
# Usage (from anywhere):  nohup caffeinate -i bash run_journe_restarts.sh > journe_restarts.log 2>&1 &
# FULL_DEFAULTS=1 instead runs only each package's full-default fit (--full_defaults: PyMC 10,000, NumPyro 30,000,
# TFP 10,000 iterations with no bijectors, Stan every CmdStan default; 10 seeds each, every fit timed) into
# ../paper_vignettes/overall_bias_results_full_default, one package after the other so the fit times are comparable.
# Settings (environment variables): PACKAGES (default "pymc numpyro tfp stan"), OVERWRITE (0), FULL_DEFAULTS (0), RESULTS_DIR,
#   CONDA_ENVS_DIR (~/miniconda3/envs), PYMC_ENV / NUMPYRO_ENV / TFP_ENV / STAN_ENV (pymc_env_pm / num_pyro / tfp_2 / stan3)
# Logs: $RESULTS_DIR/logs/<package>_journe.log

set -u
cd "$(dirname "${BASH_SOURCE[0]}")"

PACKAGES="${PACKAGES:-pymc numpyro tfp stan}"
OVERWRITE="${OVERWRITE:-0}"
FULL_DEFAULTS="${FULL_DEFAULTS:-0}"
if [[ "$FULL_DEFAULTS" == "1" ]]; then
    RESULTS_DIR="${RESULTS_DIR:-../paper_vignettes/overall_bias_results_full_default}"
    MODE=(--full_defaults)
else
    RESULTS_DIR="${RESULTS_DIR:-../paper_vignettes/overall_bias_results}"
    MODE=()
fi
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$HOME/miniconda3/envs}"
mkdir -p "$RESULTS_DIR/logs"

run() {   # run <package> <env> <script args...>
    local package="$1" env="$2"; shift 2
    local log="$RESULTS_DIR/logs/${package}_journe.log"
    echo "[$(date '+%F %T')] $package -> $log"
    if "$CONDA_ENVS_DIR/$env/bin/python" -u "$@" --restarts "${MODE[@]+"${MODE[@]}"}" --overwrite "$OVERWRITE" --results-dir "$RESULTS_DIR" > "$log" 2>&1; then
        echo "[$(date '+%F %T')] $package finished"
    else
        echo "[$(date '+%F %T')] FAILED $package (see $log)"
    fi
}

for package in $PACKAGES; do
    case "$package" in
        pymc) run pymc "${PYMC_ENV:-pymc_env_pm}" pymc_reproducing.py & ;;
        numpyro) run numpyro "${NUMPYRO_ENV:-num_pyro}" numpyro_tfp_reproducing.py --backend numpyro & ;;
        tfp) run tfp "${TFP_ENV:-tfp_2}" numpyro_tfp_reproducing.py --backend tfp & ;;
    esac
    [[ "$FULL_DEFAULTS" == "1" ]] && wait      # timed fits: one package at a time
done
wait

if [[ " $PACKAGES " == *" stan "* ]]; then
    [[ -f stan_model/journe_brms.stan ]] || Rscript export_stan_model.R
    run stan "${STAN_ENV:-stan3}" stan_reproducing.py
fi
echo "[$(date '+%F %T')] done"

#!/usr/bin/env bash
# Run every package x model random-restart runner for overall_default_bias_comparisons.ipynb.
#
# Each (package, model) pair runs as its own process in that package's conda
# environment and writes to $RESULTS_DIR/<model>/<package>_*. Existing results
# are skipped unless OVERWRITE=1, and finished seeds are cached, so rerunning
# this script resumes where it stopped.
#
# Every setting is an overridable default, either as an environment variable
#   MAX_ITERS=500000 N_SEEDS=10 ./run_all_runners.sh
# or as a flag
#   ./run_all_runners.sh --max-iters 500000 --n-seeds 10 --packages "pymc numpyro" --models "gamma_poisson radon"
#
#   MAX_ITERS          total VI iterations per run                          (default 50000)
#   N_SEEDS            random restarts per setup                            (default 10)
#   WINDOW             final iterations tracked at every iteration          (default 500)
#   STAN_MAX_ITERS     Stan's total iterations                              (default 25000)
#   STAN_N_SEEDS       Stan's random restarts per setup                     (default 5)
#   STAN_WINDOW        Stan's window; each tracked Stan iteration is a full
#                      CmdStan rerun, so Stan runs fewer / shorter            (default 500)
#   OVERWRITE          1 reruns and overwrites saved results                 (default 0)
#   PACKAGES           space-separated subset of: pymc tfp numpyro stan      (default all)
#   MODELS             space-separated subset of the model keys below        (default all)
#   PARALLEL_PACKAGES  1 runs the packages at the same time                  (default 0)
#   STAN_MAX_WORKERS   parallel CmdStan processes                            (default cores - 1)
#   RESULTS_DIR        output folder                     (default ../overall_bias_results)
#   FULL_DEFAULT       1 runs each package's full-default fit only (setup "full_default", each package's default
#                      number of iterations, N_SEEDS seeds for every package including Stan, fit times recorded)
#                      into ../overall_bias_results_full_default; MAX_ITERS / STAN_* are then ignored and the
#                      packages run one after the other so the fit times aren't slowed by each other  (default 0)
#   CONDA_ENVS_DIR     where the conda envs live           (default ~/miniconda3/envs)
#   PYMC_ENV / TFP_ENV / NUMPYRO_ENV / STAN_ENV   conda env names
#                      (defaults pymc_env_pm / tfp_2 / num_pyro / stan3)

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAX_ITERS="${MAX_ITERS:-50000}"
N_SEEDS="${N_SEEDS:-10}"
WINDOW="${WINDOW:-500}"
OVERWRITE="${OVERWRITE:-0}"
PACKAGES="${PACKAGES:-pymc tfp numpyro stan}"
MODELS="${MODELS:-1d_gaussian 40d_gaussian beta_binomial gamma_exponential gamma_poisson inverse_gamma dirichlet_multinomial logistic_regression radon}"
PARALLEL_PACKAGES="${PARALLEL_PACKAGES:-0}"
STAN_MAX_WORKERS="${STAN_MAX_WORKERS:-}"
FULL_DEFAULT="${FULL_DEFAULT:-0}"
RESULTS_DIR="${RESULTS_DIR:-}"
CONDA_ENVS_DIR="${CONDA_ENVS_DIR:-$HOME/miniconda3/envs}"
PYMC_ENV="${PYMC_ENV:-pymc_env_pm}"
TFP_ENV="${TFP_ENV:-tfp_2}"
NUMPYRO_ENV="${NUMPYRO_ENV:-num_pyro}"
STAN_ENV="${STAN_ENV:-stan3}"
STAN_MAX_ITERS="${STAN_MAX_ITERS:-25000}"
STAN_N_SEEDS="${STAN_N_SEEDS:-5}"
STAN_WINDOW="${STAN_WINDOW:-500}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-iters) MAX_ITERS="$2"; shift 2 ;;
        --n-seeds) N_SEEDS="$2"; shift 2 ;;
        --window) WINDOW="$2"; shift 2 ;;
        --stan-max-iters) STAN_MAX_ITERS="$2"; shift 2 ;;
        --stan-n-seeds) STAN_N_SEEDS="$2"; shift 2 ;;
        --stan-window) STAN_WINDOW="$2"; shift 2 ;;
        --overwrite) OVERWRITE="$2"; shift 2 ;;
        --packages) PACKAGES="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --parallel-packages) PARALLEL_PACKAGES="$2"; shift 2 ;;
        --stan-max-workers) STAN_MAX_WORKERS="$2"; shift 2 ;;
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --full-default) FULL_DEFAULT="$2"; shift 2 ;;
        -h|--help) sed -n '2,38p' "$0"; exit 0 ;;
        *) echo "unknown option: $1 (see --help)"; exit 2 ;;
    esac
done

if [[ -z "$RESULTS_DIR" ]]; then
    if [[ "$FULL_DEFAULT" == "1" ]]; then RESULTS_DIR="$SCRIPT_DIR/../overall_bias_results_full_default"
    else RESULTS_DIR="$SCRIPT_DIR/../overall_bias_results"; fi
fi
if [[ "$FULL_DEFAULT" == "1" && "$PARALLEL_PACKAGES" == "1" ]]; then
    echo "FULL_DEFAULT=1 times every fit, so the packages run one after the other (ignoring PARALLEL_PACKAGES=1)"
    PARALLEL_PACKAGES=0
fi
mkdir -p "$RESULTS_DIR/logs"
RESULTS_DIR="$(cd "$RESULTS_DIR" && pwd)"
FAILED_FILE="$RESULTS_DIR/logs/failed_runs.txt"
: > "$FAILED_FILE"

env_python() {
    case "$1" in
        pymc) echo "$CONDA_ENVS_DIR/$PYMC_ENV/bin/python" ;;
        tfp) echo "$CONDA_ENVS_DIR/$TFP_ENV/bin/python" ;;
        numpyro) echo "$CONDA_ENVS_DIR/$NUMPYRO_ENV/bin/python" ;;
        stan) echo "$CONDA_ENVS_DIR/$STAN_ENV/bin/python" ;;
        *) echo "unknown package $1" >&2; return 1 ;;
    esac
}

run_package() {
    local package="$1"
    local python
    python="$(env_python "$package")" || return 1
    if [[ ! -x "$python" ]]; then
        echo "[$package] python not found at $python (set CONDA_ENVS_DIR or the *_ENV names)" | tee -a "$FAILED_FILE"
        return 1
    fi
    local n_seeds="$N_SEEDS" window="$WINDOW"
    local extra=()
    if [[ "$FULL_DEFAULT" == "1" ]]; then
        extra+=(--full-default 1)          # each runner's own default number of iterations
    elif [[ "$package" == "stan" ]]; then
        n_seeds="$STAN_N_SEEDS" window="$STAN_WINDOW"
        extra+=(--max-iters "$STAN_MAX_ITERS")
    else
        extra+=(--max-iters "$MAX_ITERS")
    fi
    if [[ "$package" == "stan" && -n "$STAN_MAX_WORKERS" ]]; then
        extra+=(--max-workers "$STAN_MAX_WORKERS")
    fi
    for model in $MODELS; do
        local log="$RESULTS_DIR/logs/${package}_${model}.log"
        echo "[$(date '+%F %T')] $package / $model -> $log"
        if ! "$python" -u "$SCRIPT_DIR/${package}_runner.py" \
                --model "$model" \
                --n-seeds "$n_seeds" \
                --window "$window" \
                --overwrite "$OVERWRITE" \
                --results-dir "$RESULTS_DIR" \
                "${extra[@]+"${extra[@]}"}" > "$log" 2>&1; then
            echo "[$(date '+%F %T')] FAILED $package / $model (see $log)" | tee -a "$FAILED_FILE"
        fi
    done
}

echo "overall bias runners: MAX_ITERS=$MAX_ITERS N_SEEDS=$N_SEEDS WINDOW=$WINDOW OVERWRITE=$OVERWRITE FULL_DEFAULT=$FULL_DEFAULT"
echo "stan:                 STAN_MAX_ITERS=$STAN_MAX_ITERS STAN_N_SEEDS=$STAN_N_SEEDS STAN_WINDOW=$STAN_WINDOW"
echo "packages: $PACKAGES"
echo "models:   $MODELS"
echo "results:  $RESULTS_DIR"

if [[ "$PARALLEL_PACKAGES" == "1" ]]; then
    for package in $PACKAGES; do
        run_package "$package" &
    done
    wait
else
    for package in $PACKAGES; do
        run_package "$package"
    done
fi

if [[ -s "$FAILED_FILE" ]]; then
    echo "finished with failures:"
    cat "$FAILED_FILE"
    exit 1
fi
echo "[$(date '+%F %T')] all runs finished"

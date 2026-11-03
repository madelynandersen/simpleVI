#!/usr/bin/env bash
# Right-hand ("adjusted settings") runs for overall_default_bias_comparisons.ipynb that run_all_runners.sh does not make:
# TFP 1 MC with the notebooks' bijectors for 10,000 iterations (10 seeds), for the test suite and the Journé et al.
# example. Writes to ../overall_bias_results_1mc/. (PyMC 50k and Stan 25k are the runs in overall_bias_results/.)
#   bash run_tfp_1mc_10k.sh
# CONDA_ENVS_DIR is the folder holding the conda environments (default ~/miniconda3/envs).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
E="${CONDA_ENVS_DIR:-$HOME/miniconda3/envs}"
mkdir -p ../overall_bias_results_1mc/logs
R="$(cd ../overall_bias_results_1mc && pwd)"
MODELS="1d_gaussian 40d_gaussian beta_binomial gamma_exponential gamma_poisson inverse_gamma dirichlet_multinomial logistic_regression radon"
run() { local name="$1"; shift; echo "[$(date '+%F %T')] $name"; "$@" > "$R/logs/$name.log" 2>&1 || echo "[$(date '+%F %T')] FAILED $name"; }
for m in $MODELS; do
  run "tfp_$m" "$E/tfp_2/bin/python" -u tfp_runner.py --model "$m" --single-setup default_10k --max-iters 10000 --n-seeds 10
done
cd ../../Perennial_reproduction
run tfp_journe "$E/tfp_2/bin/python" -u numpyro_tfp_reproducing.py --backend tfp --restarts --single-setup default_10k \
  --iters 10000 --n-seeds 10
echo "[$(date '+%F %T')] done"

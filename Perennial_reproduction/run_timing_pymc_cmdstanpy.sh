#!/usr/bin/env bash
# Finishes the timing analysis: CmdStanPy (all three Stan VI settings: full defaults, default, adjusted; seeds 11-20) and then the
# one PyMC setting that was lost when a run was killed (Adam + 100 MC at 150,000 iterations), then draws the timing figures and
# table (extra_figs/timing_analysis.png, timing_per_iteration.png, timing_summary_all.csv) from all the timing_*.csv files.
# CmdStanPy goes first so a problem with it shows up in the first minute. Nothing else is re-run; the rows of every other setting
# stay as they are. This is run_timing_analysis.sh with fixed settings.
#
# Usage (from anywhere):  nohup caffeinate -i bash run_timing_pymc_cmdstanpy.sh >> timing_analysis.log 2>&1 < /dev/null &
# Optional: PYMC_ONLY=vi_default,vi_adam_mc100 (a different PyMC setting list), TIMING_SEEDS=11-20, CONDA_ENVS_DIR, STAN_ENV, PYMC_ENV.
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
export PACKAGES="cmdstanpy pymc"
export PYMC_ONLY="${PYMC_ONLY:-vi_adam_mc100}"
exec bash run_timing_analysis.sh

#!/usr/bin/env bash
# Run the best-VI restarts in parallel (50 Stan restarts as 10 jobs, PyMC seeds 38-50 as 3 jobs),
# then score all restarts and rebuild the comparison figures. Seeds already saved are skipped, so
# rerunning resumes. Run detached so closing the terminal does not stop it:
#   nohup caffeinate -i bash run_best_vi_parallel.sh > best_vi/logs/run.log 2>&1 &
set -u
cd "$(dirname "$0")"
mkdir -p best_vi/logs
PY_PYMC=${PY_PYMC:-$HOME/miniconda3/envs/pymc_env_pm/bin/python}
PY_SCORE=${PY_SCORE:-$HOME/miniconda3/envs/num_pyro/bin/python}

echo "started: $(date)"
for a in 1 6 11 16 21 26 31 36 41 46; do
  b=$((a + 4))
  Rscript best_vi_stan.R "$a" "$b" > "best_vi/logs/stan_${a}-${b}.log" 2>&1 &
done
for r in "38 41" "42 45" "46 50"; do
  set -- $r
  "$PY_PYMC" best_vi.py pymc --first "$1" --last "$2" > "best_vi/logs/pymc_$1-$2.log" 2>&1 &
done
wait
echo "all restarts finished: $(date)"

"$PY_SCORE" best_vi.py score
Rscript extra_figs.R
echo "plots finished: $(date)"

#!/bin/bash
# Rebuild the simpleVI conda envs on Engaging from envs/linux/*.yml.
#
# Runs ON ENGAGING as a batch job (compute nodes have internet access):
#   sbatch cluster/remote/build_envs.sh                       # all four envs
#   sbatch cluster/remote/build_envs.sh tfp_2 stan3           # just some
#
# For each env: removes the existing env of that name (if any), creates it from
# envs/linux/<env>.yml, applies the extra pip steps below, installs this repo
# (simplevi / modulars) in editable mode, then prints the versions it imports.
#
#SBATCH -J build_envs
#SBATCH -p mit_normal
#SBATCH -t 03:00:00
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -o build_envs_%j.log

set -euo pipefail

REPO="${REPO:-$HOME/simpleVI}"
CONDA_ROOT="${CONDA_ROOT:-$HOME/miniconda3}"
ENVS=("$@")
[[ ${#ENVS[@]} -eq 0 ]] && ENVS=(num_pyro pymc_env_pm stan3 tfp_2)

source "$CONDA_ROOT/etc/profile.d/conda.sh"
echo "[$(date '+%F %T')] host=$(hostname) conda=$(conda --version) repo=$REPO envs=${ENVS[*]}"

for env in "${ENVS[@]}"; do
  spec="$REPO/envs/linux/$env.yml"
  [[ -f "$spec" ]] || { echo "missing $spec" >&2; exit 1; }
  echo
  echo "================ $env ================"

  if conda env list | awk '{print $1}' | grep -qx "$env"; then
    echo "[$(date '+%T')] removing existing env $env"
    conda env remove -n "$env" -y
  fi

  echo "[$(date '+%T')] creating $env from $spec"
  conda env create -f "$spec" -y

  py="$CONDA_ROOT/envs/$env/bin/python"
  case "$env" in
    tfp_2)
      # Mac has tf_keras 2.18.0 alongside TF 2.20; its metadata pins tensorflow<2.19,
      # so install it without letting pip downgrade TensorFlow.
      "$py" -m pip install --no-deps tf-keras==2.18.0
      ;;
  esac

  # this repo, editable (matches the Mac: modulars imports from the repo folder)
  "$py" -m pip install --no-deps -e "$REPO"

  "$py" -m pip check || echo "(pip check reported conflicts above; see notes in envs/linux/$env.yml)"
  (cd /tmp && "$py" -I "$REPO/cluster/check_envs.py")
done

conda clean -y --tarballs >/dev/null 2>&1 || true
echo
echo "[$(date '+%F %T')] done"

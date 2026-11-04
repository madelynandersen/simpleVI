#!/usr/bin/env bash
# Get an interactive compute node on Engaging and open VS Code on it.
#
#   1. ssh orcd-login and request an allocation with `salloc --no-shell`
#      (waits until the job starts, then returns; the job keeps running
#      on its own until its time limit or until you cancel it)
#   2. point `orcd-compute` in ~/.ssh/config at the allocated node
#   3. open VS Code on that node
#
# If a running job with the same name already exists, it is reused instead of
# requesting a new one.
#
# Usage (from the repo root on your Mac):
#   cluster/start_job.sh
#   cluster/start_job.sh -t 4:00:00 -c 2 --mem 8G
#   cluster/start_job.sh -p mit_preemptable
#   cluster/start_job.sh --no-open          # just allocate + update ssh config
#   cluster/start_job.sh --cancel           # end the job (scancel) when you're done
#
# Defaults can also be set via environment variables of the same names below.

set -euo pipefail
trap 'echo "start_job.sh: failed at line $LINENO (exit $?): $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGIN_HOST="${LOGIN_HOST:-orcd-login}"
PARTITION="${PARTITION:-mit_normal}"
TIME="${TIME:-12:00:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-16G}"
JOB_NAME="${JOB_NAME:-vscode}"
REMOTE_DIR="${REMOTE_DIR:-/home/madelyna/simpleVI}"
open=1
cancel=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--partition) PARTITION="$2"; shift 2 ;;
    -t|--time)      TIME="$2"; shift 2 ;;
    -c|--cpus)      CPUS="$2"; shift 2 ;;
    --mem)          MEM="$2"; shift 2 ;;
    -J|--job-name)  JOB_NAME="$2"; shift 2 ;;
    --dir)          REMOTE_DIR="$2"; shift 2 ;;
    --no-open)      open=0; shift ;;
    --cancel)       cancel=1; shift ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
done

running_job() {
  # prints "jobid|node|time_left" of a running job named $JOB_NAME, if any
  ssh "$LOGIN_HOST" "squeue --me -h -t RUNNING -n '$JOB_NAME' -o '%i|%N|%L'" | head -1
}

if [[ "$cancel" -eq 1 ]]; then
  echo "Cancelling your '$JOB_NAME' jobs on $LOGIN_HOST..."
  ssh "$LOGIN_HOST" "scancel --me -n '$JOB_NAME' && squeue --me"
  exit 0
fi

existing=$(running_job)
if [[ -z "$existing" ]]; then
  # e.g. a preempted job that Slurm put back in the queue: don't submit a duplicate
  pending=$(ssh "$LOGIN_HOST" "squeue --me -h -t PENDING -n '$JOB_NAME' -o '%i|%P|%r'")
  if [[ -n "$pending" ]]; then
    echo "A '$JOB_NAME' job is already waiting in the queue (jobid|partition|reason):" >&2
    echo "$pending" >&2
    echo "Rerun this script once it starts, or cancel it with: $0 --cancel -J $JOB_NAME" >&2
    exit 1
  fi
fi
if [[ -n "$existing" ]]; then
  echo "Reusing running job: $existing  (jobid|node|time left)"
else
  echo "Requesting: partition=$PARTITION time=$TIME cpus=$CPUS mem=$MEM name=$JOB_NAME"
  echo "Waiting for the allocation (this can take a while if the queue is busy)..."
  ssh "$LOGIN_HOST" "salloc --no-shell -J '$JOB_NAME' -p '$PARTITION' -t '$TIME' -c '$CPUS' --mem='$MEM'"
  existing=$(running_job)
  if [[ -z "$existing" ]]; then
    echo "salloc returned but no running '$JOB_NAME' job was found; check 'squeue --me'." >&2
    exit 1
  fi
  echo "Allocated: $existing  (jobid|node|time left)"
fi

node=$(cut -d'|' -f2 <<<"$existing")
"$SCRIPT_DIR/set_compute_node.sh" "$node"

# accept the new node's host key now so VS Code doesn't stall on the prompt
ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=30 orcd-compute true \
  || { echo "Could not ssh to $node via orcd-compute." >&2; exit 1; }

if [[ "$open" -eq 1 ]]; then
  "$SCRIPT_DIR/set_compute_node.sh" "$node" --open "$REMOTE_DIR" | tail -1
fi

echo
echo "Done. The job keeps running after you close VS Code; end it with:"
echo "  cluster/start_job.sh --cancel${JOB_NAME:+ -J $JOB_NAME}"

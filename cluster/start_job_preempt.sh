#!/usr/bin/env bash
# Same as start_job.sh, but on the mit_preemptable partition (up to 48 h, many
# more nodes, but the job can be killed at any time to make room for others).
#
# Use this for an interactive / VS Code node when mit_normal is busy. Do NOT
# run long experiments from a terminal on this node: if the node is preempted,
# whatever is running in it dies and does not restart. Long runs go through
# sbatch with --requeue instead.
#
# Usage (from the repo root on your Mac):
#   cluster/start_job_preempt.sh                 # 48 h, 4 CPUs, 16G
#   cluster/start_job_preempt.sh -t 8:00:00      # any start_job.sh option works
#   cluster/start_job_preempt.sh --cancel

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/start_job.sh" -p mit_preemptable -t 2-00:00:00 -J vscode_preempt "$@"

#!/usr/bin/env bash
# Point the `orcd-compute` entry in ~/.ssh/config at a given Engaging compute node.
#
# Usage:
#   cluster/set_compute_node.sh node2704        # set explicitly
#   cluster/set_compute_node.sh 2704            # "node" prefix optional
#   cluster/set_compute_node.sh                 # auto: ask Slurm for your running job's node
#   cluster/set_compute_node.sh --job vscode    # auto, but only consider jobs with this name
#   cluster/set_compute_node.sh node2704 --open /home/madelyna/simpleVI
#                                               # also open VS Code on that folder
#
# Only the HostName line inside the `Host orcd-compute` block is changed.
# A backup of the previous config is written to ~/.ssh/config.bak.

set -euo pipefail

SSH_CONFIG="${SSH_CONFIG:-$HOME/.ssh/config}"
COMPUTE_HOST="${COMPUTE_HOST:-orcd-compute}"
LOGIN_HOST="${LOGIN_HOST:-orcd-login}"

node=""
job_name=""
open_path=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --job)  job_name="$2"; shift 2 ;;
    --open) open_path="$2"; shift 2 ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    -*) echo "Unknown option: $1" >&2; exit 2 ;;
    *)  node="$1"; shift ;;
  esac
done

# --- Auto-detect node from Slurm if none given --------------------------------
if [[ -z "$node" ]]; then
  echo "No node given; asking Slurm on $LOGIN_HOST for your running jobs..."
  # BatchMode: fail instead of prompting (avoids surprise Duo pushes).
  fmt='%i|%j|%P|%N|%L'   # jobid|name|partition|nodelist|time left
  jobs=$(ssh -o BatchMode=yes -o ConnectTimeout=20 "$LOGIN_HOST" \
           "squeue --me -h -t RUNNING -o '$fmt'") || {
    echo "Could not reach $LOGIN_HOST non-interactively. Run 'ssh $LOGIN_HOST' once, then retry." >&2
    exit 1
  }
  if [[ -n "$job_name" ]]; then
    jobs=$(awk -F'|' -v n="$job_name" '$2==n' <<<"$jobs")
  fi
  count=$(grep -c . <<<"$jobs" || true)
  if [[ "$count" -eq 0 ]]; then
    echo "No running jobs found${job_name:+ named '$job_name'}." >&2
    exit 1
  elif [[ "$count" -gt 1 ]]; then
    echo "Multiple running jobs; pass the node explicitly or use --job NAME:" >&2
    echo "JOBID|NAME|PARTITION|NODE|TIME_LEFT" >&2
    echo "$jobs" >&2
    exit 1
  fi
  node=$(cut -d'|' -f4 <<<"$jobs")
  echo "Found job: $jobs"
fi

# --- Normalize / validate ------------------------------------------------------
[[ "$node" =~ ^[0-9]+$ ]] && node="node$node"
if [[ ! "$node" =~ ^node[0-9]+$ ]]; then
  echo "Node name '$node' doesn't look like nodeNNNN (multi-node lists aren't supported)." >&2
  exit 1
fi

# --- Edit the config -----------------------------------------------------------
if ! grep -qE "^[[:space:]]*Host[[:space:]]+$COMPUTE_HOST[[:space:]]*$" "$SSH_CONFIG"; then
  echo "No 'Host $COMPUTE_HOST' entry in $SSH_CONFIG." >&2
  exit 1
fi

cp "$SSH_CONFIG" "$SSH_CONFIG.bak"
tmp=$(mktemp)
awk -v host="$COMPUTE_HOST" -v node="$node" '
  /^[[:space:]]*Host[[:space:]]/ {
    in_block = ($2 == host && NF == 2)
  }
  in_block && /^[[:space:]]*HostName[[:space:]]/ {
    match($0, /^[[:space:]]*/)
    print substr($0, 1, RLENGTH) "HostName " node
    done = 1
    next
  }
  { print }
  END { if (!done) exit 3 }
' "$SSH_CONFIG" > "$tmp" || {
  rm -f "$tmp"
  echo "Found 'Host $COMPUTE_HOST' but no HostName line in it; config unchanged." >&2
  exit 1
}
# Write in place (keeps the file's permissions/attributes) rather than mv.
cat "$tmp" > "$SSH_CONFIG"
rm -f "$tmp"

echo "Updated $COMPUTE_HOST -> $node in $SSH_CONFIG (backup: $SSH_CONFIG.bak)"
awk -v host="$COMPUTE_HOST" '
  /^[[:space:]]*Host[[:space:]]/ { show = ($2 == host) }
  show
' "$SSH_CONFIG"

# --- Optionally open VS Code ---------------------------------------------------
if [[ -n "$open_path" ]]; then
  code_cli=$(command -v code || true)
  mac_cli="/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code"
  [[ -z "$code_cli" && -x "$mac_cli" ]] && code_cli="$mac_cli"
  if [[ -n "$code_cli" ]]; then
    echo "Opening VS Code on $COMPUTE_HOST:$open_path ..."
    "$code_cli" --remote "ssh-remote+$COMPUTE_HOST" "$open_path"
  else
    echo "'code' CLI not on PATH; connect manually via >< -> Connect to Host -> $COMPUTE_HOST" >&2
  fi
fi

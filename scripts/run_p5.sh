#!/usr/bin/env bash
# Run every P5 measurement (scripts/run_p5.py) with the paths that scripts/setup_remote.sh wrote to p5.env.
# Resumable: just run it again after an interruption. Log: results/p5.log. Extra arguments go to run_p5.py,
# e.g. --dry-run (list what is left), --sections sharegpt, --table-only.
#
# On the rented machine run it inside tmux so that a dropped SSH connection doesn't stop it:
#   tmux new -s p5 'bash scripts/run_p5.sh; bash'      (detach: Ctrl+b then d; back: tmux attach -t p5)
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
[ -f p5.env ] || { echo "p5.env not found: run scripts/setup_remote.sh first"; exit 1; }
source p5.env
mkdir -p results
echo "=== $(date '+%F %T') run_p5 $*" >> results/p5.log
"$NANO_PY" -u scripts/run_p5.py "$@" 2>&1 | tee -a results/p5.log

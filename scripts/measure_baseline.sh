#!/usr/bin/env bash
# One-time baseline run.
#
# Runs the unmodified train_gpt.py on the 8xH100 box, parses the resulting log,
# and writes the measured train_time_ms / val_loss to .autoresearch/state/best.json.
# The daemon then attacks this measured baseline (not the README's record-#80
# number, which depends on hardware specifics).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Running unmodified train_gpt.py once (this includes ~7min torch.compile)"
./run.sh

# Find the newest log produced by the run.
LOG=$(ls -t logs/*.txt | head -n1)
echo "==> Parsing $LOG"

python -m autoresearch baseline --log "$LOG"
python -m autoresearch status

#!/usr/bin/env bash
# Run the unmodified speedrun once on this hardware, then:
#   1. Calibrate: compare measured time vs the most recent records/track_1_short log.
#      The daemon will refuse to start if |deviation| > 25%.
#   2. Record baseline: write .autoresearch/state/best.json with our measured numbers
#      so the daemon attacks the actual hardware time, not the README's official number.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Running unmodified train_gpt.py on this hardware"
echo "    (~7 min torch.compile + ~1.5-2 min training; total ~9-10 min on 8xH100)"
./run.sh

LOG=$(ls -t logs/*.txt | head -n1)
if [[ -z "${LOG}" || ! -f "${LOG}" ]]; then
  echo "ERROR: no log produced under logs/" >&2
  exit 1
fi
echo
echo "==> Log: $LOG"

echo
echo "==> Calibration (measured vs upstream record reference)"
python -m autoresearch calibrate --log "$LOG"

echo
echo "==> Recording our baseline as state/best.json"
python -m autoresearch baseline --log "$LOG"

echo
python -m autoresearch status
echo
echo "Baseline measurement complete."
echo "If calibration showed |deviation| > 5%, expect timing differences on this box."
echo "If |deviation| > 25%, the daemon will refuse to start until you re-measure or"
echo "set AUTORESEARCH_SKIP_CALIBRATION=1 (not recommended)."

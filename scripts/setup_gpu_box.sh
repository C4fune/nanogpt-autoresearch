#!/usr/bin/env bash
# One-shot setup on any 8xH100 GPU box (Lambda, Hyperbolic, Prime Intellect, etc.).
#
# Usage (run from the cloned repo root on the GPU box):
#   export ANTHROPIC_API_KEY=sk-...
#   bash scripts/setup_gpu_box.sh
#
# What it does:
#   1. Installs nanogpt + autoresearch deps.
#   2. Downloads ~900M FineWeb tokens (the standard speedrun shard count).
#   3. Bootstraps the autoresearch state (code map + record index).
#
# After this finishes, run:
#   bash scripts/measure_baseline.sh   # one-time baseline measurement
#   tmux new -s ar
#   python -m autoresearch run         # the daemon (Ctrl+B then D to detach)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is not set. export it before running this script." >&2
  exit 1
fi

echo "==> Installing system Python deps (nanogpt requirements)"
pip install -r requirements.txt

echo "==> Installing autoresearch deps"
pip install -r autoresearch/requirements.txt

echo "==> Verifying GPUs"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')
if [[ "${GPU_COUNT}" != "8" ]]; then
  echo
  echo "WARNING: detected ${GPU_COUNT} GPUs. The canonical speedrun (run.sh) uses 8."
  echo "Training will still run on fewer GPUs but timings will not be comparable to leaderboard records."
  echo
fi

echo "==> Downloading FineWeb shards (idempotent — skips already-downloaded files)"
python data/cached_fineweb10B.py 9

echo "==> Bootstrapping autoresearch state"
python -m autoresearch bootstrap

echo
echo "Setup complete."
echo "Next: run 'bash scripts/measure_baseline.sh' to take a one-time baseline measurement."

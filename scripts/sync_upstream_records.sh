#!/usr/bin/env bash
# Refresh records/ from KellerJordan/modded-nanogpt without touching anything else.
#
# Why: the planner's record-card context is derived from records/track_1_short/*.
# If the local checkout lags upstream (which it will, the moment new records land),
# the agent can't cite recent records when proposing patches. Run this periodically
# (or from cron) to keep records/ current.
#
# How it works: configures `upstream` as a remote if missing, fetches it, and
# checks out ONLY records/ from upstream/master onto your current branch. We do
# NOT merge train_gpt.py / triton_kernels.py — those are owned by the wins chain
# the agent is building, and a blind merge would scramble that.
#
# Safety:
#   - Refuses to run with uncommitted changes in records/.
#   - Leaves your branch state otherwise untouched.

set -euo pipefail

UPSTREAM_URL="${AUTORESEARCH_UPSTREAM_URL:-https://github.com/KellerJordan/modded-nanogpt.git}"
UPSTREAM_BRANCH="${AUTORESEARCH_UPSTREAM_BRANCH:-master}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if ! git remote get-url upstream >/dev/null 2>&1; then
  echo "==> adding 'upstream' remote -> $UPSTREAM_URL"
  git remote add upstream "$UPSTREAM_URL"
fi

if ! git diff --quiet -- records/ || ! git diff --cached --quiet -- records/; then
  echo "ERROR: records/ has uncommitted changes. Commit or stash before syncing." >&2
  exit 1
fi

echo "==> fetching upstream/$UPSTREAM_BRANCH"
git fetch upstream "$UPSTREAM_BRANCH"

UPSTREAM_REF="upstream/$UPSTREAM_BRANCH"
BEFORE=$(find records/track_1_short -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')

echo "==> checking out records/ from $UPSTREAM_REF"
# Pathspec scoped to records/; touches nothing else in the worktree.
git checkout "$UPSTREAM_REF" -- records/

AFTER=$(find records/track_1_short -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d ' ')
echo "==> records/track_1_short folders: $BEFORE -> $AFTER"

if ! git diff --cached --quiet -- records/; then
  echo "==> staged record changes:"
  git diff --cached --stat -- records/ | tail -n +1
  echo
  echo "Next steps:"
  echo "  git commit -m 'Sync records/ from upstream/$UPSTREAM_BRANCH'"
  echo "  python -m autoresearch bootstrap   # rebuild record_index from the new folders"
else
  echo "==> records/ already current with $UPSTREAM_REF; nothing to commit."
fi

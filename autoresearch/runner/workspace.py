"""Per-attempt git worktree.

Each attempt gets its own worktree under .autoresearch/_worktrees/<run_id>/.
That gives us an isolated working copy of the repo to patch + run, and lets us
restore the original files via git semantics if anything goes wrong.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from autoresearch.config import Config


@dataclass(frozen=True)
class Worktree:
    run_id: str
    path: Path
    branch: str


def create(config: Config, run_id: str, *, from_ref: str = "HEAD") -> Worktree:
    """Create a per-attempt worktree branched from `from_ref`.

    `from_ref` defaults to HEAD but the daemon passes `state.best.baseline_commit_sha`
    so that subsequent attempts build on top of the cumulative wins chain.
    """
    path = config.paths.worktree_for(run_id)
    branch = f"autoresearch/{run_id}"
    if path.exists():
        remove(config, run_id)

    proc = subprocess.run(
        ["git", "worktree", "add", "-B", branch, str(path), from_ref],
        cwd=config.repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"git worktree add failed: {proc.stderr.strip()}")
    return Worktree(run_id=run_id, path=path, branch=branch)


def current_baseline_sha(config: Config) -> str:
    """SHA of the current parent for new attempts. Reads state.best.baseline_commit_sha
    if present, else falls back to HEAD of the main repo.
    """
    from autoresearch.state import load_best

    best = load_best(config.paths.best_json)
    if best.baseline_commit_sha:
        return best.baseline_commit_sha
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=config.repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def remove(config: Config, run_id: str) -> None:
    path = config.paths.worktree_for(run_id)
    if path.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=config.repo_root,
            capture_output=True,
            check=False,
        )
    branch = f"autoresearch/{run_id}"
    subprocess.run(
        ["git", "branch", "-D", branch],
        cwd=config.repo_root,
        capture_output=True,
        check=False,
    )


def snapshot_editable_files(wt: Worktree, config: Config) -> dict[str, str]:
    return {
        rel: (wt.path / rel).read_text()
        for rel in config.editable_files
        if (wt.path / rel).exists()
    }

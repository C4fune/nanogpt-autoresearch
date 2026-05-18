"""Run training in a worktree, capture log + stderr tail, gzip the log.

Returns a small RunResult; the full log lives on disk only.
"""

from __future__ import annotations

import gzip
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from autoresearch.config import Config
from autoresearch.parser.log_parser import LogMetrics, compact_log_summary, parse_log


@dataclass
class RunResult:
    success: bool
    metrics: LogMetrics | None
    summary: str
    error: str | None
    stderr_tail: str
    duration_s: float
    log_path_gz: Path | None  # forensics only


def run_training(wt_path: Path, config: Config, run_dir: Path) -> RunResult:
    """Run `./run.sh` in worktree; copy the produced log into run_dir/log.txt.gz.

    DATA_PATH points to the main repo (which holds data/fineweb10B/*.bin shards),
    NOT the worktree. Worktrees share .git but the binary data shards are gitignored
    and only exist in the main checkout.
    """
    logs_before = _list_logs(wt_path)
    t0 = time.monotonic()
    env = os.environ.copy()
    env["DATA_PATH"] = str(config.repo_root)

    try:
        proc = subprocess.run(
            list(config.run_command),
            cwd=wt_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=config.run_timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        stderr_tail = _tail((e.stderr.decode() if isinstance(e.stderr, bytes) else (e.stderr or "")), 80)
        return RunResult(
            success=False,
            metrics=None,
            summary="",
            error=f"timeout after {config.run_timeout_s}s",
            stderr_tail=stderr_tail,
            duration_s=time.monotonic() - t0,
            log_path_gz=None,
        )

    duration = time.monotonic() - t0
    stderr_tail = _tail(proc.stderr or "", 80)

    if proc.returncode != 0:
        return RunResult(
            success=False,
            metrics=None,
            summary="",
            error=f"exit {proc.returncode}",
            stderr_tail=stderr_tail,
            duration_s=duration,
            log_path_gz=None,
        )

    log_path = _newest_log(wt_path, logs_before)
    if not log_path:
        return RunResult(
            success=False,
            metrics=None,
            summary="",
            error="no log produced in logs/",
            stderr_tail=stderr_tail,
            duration_s=duration,
            log_path_gz=None,
        )

    metrics = parse_log(log_path)
    log_dest_gz = run_dir / "log.txt.gz"
    _gzip_copy(log_path, log_dest_gz)

    return RunResult(
        success=True,
        metrics=metrics,
        summary=compact_log_summary(metrics),
        error=None,
        stderr_tail=stderr_tail,
        duration_s=duration,
        log_path_gz=log_dest_gz,
    )


def _list_logs(root: Path) -> set[Path]:
    d = root / "logs"
    return {p.resolve() for p in d.glob("*.txt")} if d.exists() else set()


def _newest_log(root: Path, before: set[Path]) -> Path | None:
    d = root / "logs"
    if not d.exists():
        return None
    new = [p.resolve() for p in d.glob("*.txt") if p.resolve() not in before]
    cands = new or [p.resolve() for p in d.glob("*.txt")]
    return max(cands, key=lambda p: p.stat().st_mtime) if cands else None


def _gzip_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, gzip.open(dst, "wb", compresslevel=6) as fout:
        shutil.copyfileobj(fin, fout)


def _tail(s: str, n_lines: int) -> str:
    if not s:
        return ""
    lines = s.splitlines()
    return "\n".join(lines[-n_lines:])


def prune_old_logs(config: Config) -> int:
    """Keep only the most recent N gzipped logs to bound disk usage."""
    runs_dir = config.paths.runs_dir
    if not runs_dir.exists():
        return 0
    keep = config.keep_last_n_logs
    runs = sorted(runs_dir.iterdir(), reverse=True)
    pruned = 0
    for d in runs[keep:]:
        log = d / "log.txt.gz"
        if log.exists():
            log.unlink()
            pruned += 1
    return pruned

"""Atomic JSON state files. Each one is small, single-purpose, replaceable."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return default


# ----- best.json --------------------------------------------------------------


@dataclass
class Best:
    train_time_ms: int | None = None
    val_loss: float | None = None
    val_loss_std: float | None = None
    run_id: str | None = None
    patch_branch: str | None = None       # "wins/<id>" if pinned
    baseline_commit_sha: str | None = None  # ref new worktrees branch from
    n_seeds_confirmed: int = 0
    n_wins_chain: int = 0                 # how many cumulative wins so far
    confirmed_at: str | None = None
    notes: str = ""

    def beats(self, train_time_ms: int, val_loss: float, val_loss_max: float) -> bool:
        if val_loss > val_loss_max:
            return False
        if self.train_time_ms is None:
            return True
        return train_time_ms < self.train_time_ms


def load_best(path: Path) -> Best:
    raw = _read_json(path, {})
    return Best(**{k: v for k, v in raw.items() if k in Best.__annotations__})


def save_best(path: Path, best: Best) -> None:
    _atomic_write_json(path, asdict(best))


# ----- cursor.json ------------------------------------------------------------


@dataclass
class Cursor:
    """Marks the run currently in flight; lets us recover after a crash."""
    run_id: str | None = None
    started_at: str | None = None
    phase: str | None = None  # "patching" | "prechecks" | "training" | "distilling"

    def is_in_flight(self) -> bool:
        return self.run_id is not None


def load_cursor(path: Path) -> Cursor:
    raw = _read_json(path, {})
    return Cursor(**{k: v for k, v in raw.items() if k in Cursor.__annotations__})


def save_cursor(path: Path, cursor: Cursor) -> None:
    _atomic_write_json(path, asdict(cursor))


def clear_cursor(path: Path) -> None:
    save_cursor(path, Cursor())


# ----- budget.json ------------------------------------------------------------


@dataclass
class Budget:
    """Counter for runs and GPU-time. Informational; the daemon doesn't stop on it."""
    gpu_hours_used: float = 0.0
    runs_completed: int = 0
    started_at: str = field(default_factory=_utc_now)


def load_budget(path: Path) -> Budget:
    raw = _read_json(path, {})
    return Budget(**{k: v for k, v in raw.items() if k in Budget.__annotations__})


def save_budget(path: Path, budget: Budget) -> None:
    _atomic_write_json(path, asdict(budget))

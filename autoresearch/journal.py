"""Append-only event log. Every decision and outcome lands here.

Reading the journal end-to-end is a forensics tool, never planner input.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def emit(journal_path: Path, event: str, **fields: Any) -> None:
    """Atomic append of a single JSON line."""
    journal_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **fields,
    }
    line = json.dumps(record, default=str) + "\n"
    # O_APPEND on POSIX guarantees atomic writes <= PIPE_BUF; our lines are < 4KB.
    fd = os.open(journal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def tail(journal_path: Path, n: int = 50) -> list[dict[str, Any]]:
    """Forensics helper. Not for the LLM."""
    if not journal_path.exists():
        return []
    lines = journal_path.read_text(errors="replace").splitlines()[-n:]
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out

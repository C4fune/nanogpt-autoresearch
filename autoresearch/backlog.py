"""Idea backlog as JSONL. Append-only file; consumption tracked by `consumed_at`.

Each idea row carries everything needed to attempt it:
  {
    "id": "...",
    "created_at": "...",
    "consumed_at": null,
    "priority": 0.7,
    "hypothesis": "...",
    "category": "optimizer|schedule|kernel|architecture|systems|mixed",
    "rationale": "...",
    "edits": [{"file": "...", "old": "...", "new": "..."}],
    "tags": ["muon", "schedule"]
  }
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append(path: Path, idea: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    idea = dict(idea)
    idea.setdefault("id", uuid.uuid4().hex[:8])
    idea.setdefault("created_at", _utc_now())
    idea.setdefault("consumed_at", None)
    idea.setdefault("priority", 0.5)
    line = json.dumps(idea, default=str) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)
    return idea["id"]


def _load_all(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def remaining(path: Path) -> list[dict[str, Any]]:
    return [i for i in _load_all(path) if i.get("consumed_at") is None]


def remaining_count(path: Path) -> int:
    return len(remaining(path))


def consumed(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    items = [i for i in _load_all(path) if i.get("consumed_at") is not None]
    items.sort(key=lambda x: x.get("consumed_at") or "", reverse=True)
    return items[:limit] if limit else items


def pop_next(path: Path) -> dict[str, Any] | None:
    """Mark and return the highest-priority unconsumed idea."""
    items = _load_all(path)
    open_indices = [i for i, x in enumerate(items) if x.get("consumed_at") is None]
    if not open_indices:
        return None
    chosen = max(open_indices, key=lambda i: items[i].get("priority", 0))
    items[chosen]["consumed_at"] = _utc_now()
    _rewrite(path, items)
    return items[chosen]


def mark_priority(path: Path, idea_id: str, priority: float) -> None:
    items = _load_all(path)
    for it in items:
        if it.get("id") == idea_id and it.get("consumed_at") is None:
            it["priority"] = priority
    _rewrite(path, items)


def _rewrite(path: Path, items: list[dict[str, Any]]) -> None:
    """Atomic rewrite — only safe because we're single-process."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(x, default=str) + "\n" for x in items))
    os.replace(tmp, path)


def stats(path: Path) -> dict[str, int]:
    items = _load_all(path)
    return {
        "total": len(items),
        "remaining": sum(1 for i in items if i.get("consumed_at") is None),
        "consumed": sum(1 for i in items if i.get("consumed_at") is not None),
    }

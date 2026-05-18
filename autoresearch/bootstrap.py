"""One-shot bootstrap: build code_map.md and record_index.jsonl.

Run this once when setting up the agent on a new machine, then again after big
upstream merges. The daemon does NOT do this every iteration — it's a slow
walk over records/ and not needed at runtime.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from autoresearch.agent.code_index import find_sections, render_code_map
from autoresearch.config import Config
from autoresearch.parser.log_parser import compact_log_summary, parse_log


def bootstrap(config: Config) -> dict[str, int]:
    config.paths.ensure()
    n_anchors = _build_code_map(config)
    n_records = _build_record_index(config)
    return {"anchors": n_anchors, "records": n_records}


def _build_code_map(config: Config) -> int:
    sections = find_sections(config.repo_root, "train_gpt.py")
    sections += find_sections(config.repo_root, "triton_kernels.py")
    text = render_code_map(sections)
    config.paths.code_map_md.write_text(text)
    return len(sections)


def _build_record_index(config: Config) -> int:
    """Walk records/track_1_short/* and emit one JSON line per folder.

    Each card is small (folder name, summary excerpt, best metrics). Designed to
    be filtered/queried by the planner; never the full source/log.
    """
    root = config.repo_root / "records" / "track_1_short"
    if not root.exists():
        return 0
    out = config.paths.record_index_jsonl
    tmp = out.with_suffix(out.suffix + ".tmp")
    count = 0
    now = datetime.now(timezone.utc).isoformat()
    with tmp.open("w") as f:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            card = _record_card(d, now)
            if card:
                f.write(json.dumps(card, default=str) + "\n")
                count += 1
    os.replace(tmp, out)
    return count


def _record_card(record_dir: Path, indexed_at: str) -> dict | None:
    folder = record_dir.name
    readme = record_dir / "README.md"
    summary = ""
    if readme.exists():
        text = readme.read_text(errors="replace")
        summary = _truncate(text, 600)

    metrics = _best_log_metrics(record_dir)
    val_loss = train_ms = steps = None
    if metrics and metrics.final:
        val_loss = metrics.final.val_loss
        train_ms = metrics.final.train_time_ms
        steps = metrics.final.step
        if not summary:
            summary = compact_log_summary(metrics)

    if not summary and not metrics:
        return None

    date = _extract_date_from_folder(folder)
    return {
        "id": f"records/track_1_short/{folder}",
        "folder": folder,
        "date": date,
        "summary": summary,
        "val_loss": val_loss,
        "train_time_ms": train_ms,
        "steps": steps,
        "tags": _tags_from_folder(folder),
        "indexed_at": indexed_at,
    }


def _best_log_metrics(record_dir: Path):
    best = None
    best_t = 10**12
    candidates = list(record_dir.rglob("*.txt")) + list(record_dir.rglob("*.log"))
    for p in candidates:
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        if sz > 8_000_000:
            continue
        try:
            m = parse_log(p)
        except OSError:
            continue
        if m.final and m.final.train_time_ms is not None and m.final.train_time_ms < best_t:
            best, best_t = m, m.final.train_time_ms
    return best


def _tags_from_folder(folder: str) -> list[str]:
    parts = re.split(r"[_\-]", folder)
    return [p.lower() for p in parts if len(p) > 2 and not p.isdigit()][:8]


def _extract_date_from_folder(folder: str) -> str | None:
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", folder)
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else None


def _truncate(s: str, n: int) -> str:
    s = re.sub(r"\n{3,}", "\n\n", s.strip())
    return s if len(s) <= n else s[: n - 3] + "..."

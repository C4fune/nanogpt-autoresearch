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

    Each card carries the metric tuple AND — when extractable from the upstream
    README table — the human-readable description of WHAT changed in that record
    (e.g. "Reuse and tune backward transpose kernel"). Without those descriptions,
    the planner only sees numbers and can't reason about the SOTA arc.
    """
    root = config.repo_root / "records" / "track_1_short"
    if not root.exists():
        return 0
    out = config.paths.record_index_jsonl
    tmp = out.with_suffix(out.suffix + ".tmp")
    count = 0
    now = datetime.now(timezone.utc).isoformat()

    descriptions = _parse_readme_records_table(config.repo_root / "README.md")

    with tmp.open("w") as f:
        for d in sorted(root.iterdir()):
            if not d.is_dir():
                continue
            card = _record_card(d, now)
            if card:
                meta = descriptions.get(d.name)
                if meta:
                    card["record_number"] = meta.get("record_number")
                    card["record_time"] = meta.get("record_time")
                    card["description"] = meta.get("description")
                    card["contributors"] = meta.get("contributors")
                f.write(json.dumps(card, default=str) + "\n")
                count += 1
    os.replace(tmp, out)
    return count


_README_TABLE_ROW = re.compile(
    r"^(\d+)\s*\|\s*([^|]+?)\s*\|\s*(.+?)\s*\|\s*([\d/]+)\s*\|\s*(.+?)\s*\|\s*(.+?)\s*$"
)
_README_LOG_HREF = re.compile(r"records/track_1_short/([^/)\s]+)")
_README_LINK_TEXT = re.compile(r"\[([^\]]+)\]\([^)]+\)")


def _parse_readme_records_table(readme_path: Path) -> dict[str, dict]:
    """Extract one row per record from the upstream README track-1 table.

    Returns: dict mapping records/track_1_short/<folder_name> -> {
        record_number, record_time, description (plain text), contributors
    }.

    The README markdown is loose (pipes inside link text, multi-line cells in some
    rows). We're strict-but-forgiving: skip rows we can't parse rather than break.
    """
    if not readme_path.exists():
        return {}
    out: dict[str, dict] = {}
    text = readme_path.read_text(errors="replace")
    in_table = False
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("| # |") and "Record time" in s and "Description" in s:
            in_table = True
            continue
        if not in_table:
            continue
        # Table ends on first blank line or non-pipe line that isn't a row.
        if not s or not (s[0].isdigit() or s.startswith("|")):
            in_table = False
            continue
        # Normalize leading pipe + spaces.
        row = s.lstrip("| ").rstrip("|").rstrip()
        m = _README_TABLE_ROW.match(row)
        if not m:
            continue
        record_n, time_cell, desc_cell, date_cell, log_cell, contrib_cell = m.groups()
        # Find the folder this row references via its log link.
        folder = None
        log_match = _README_LOG_HREF.search(log_cell) or _README_LOG_HREF.search(desc_cell)
        if log_match:
            folder = log_match.group(1)
        if not folder:
            continue
        # Strip markdown link wrappers in the description for clean storage.
        desc_text = _README_LINK_TEXT.sub(r"\1", desc_cell).strip()
        time_text = _README_LINK_TEXT.sub(r"\1", time_cell).strip()
        contrib_text = _README_LINK_TEXT.sub(r"\1", contrib_cell).strip()
        out[folder] = {
            "record_number": int(record_n),
            "record_time": time_text,
            "description": desc_text,
            "contributors": contrib_text,
        }
    return out


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

"""Apply edits to a worktree. Rule-gated; never accepts whole-file rewrites.

Two edit kinds:
  - search_replace (default): exact substring replacement, must match exactly once.
  - insert_after:              insert `new` immediately after `old` (used to add new code blocks).

The rules module gates BOTH the edit list (before any IO) and the resulting file
(after writes), so a malformed broad patch can't sneak past us.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from autoresearch.config import Config
from autoresearch.rules import RuleViolation, check_patch, check_resulting_file


class MutatorError(Exception):
    """Patch was rejected. Message goes into the run summary."""


@dataclass
class Edit:
    file: str
    old: str
    new: str
    kind: str = "search_replace"  # or "insert_after"

    def to_dict(self) -> dict:
        return {"file": self.file, "old": self.old, "new": self.new, "kind": self.kind}


@dataclass
class PatchPlan:
    hypothesis: str
    category: str
    rationale: str
    edits: list[Edit]
    expected_speedup_ms: int | None = None
    expected_loss_delta: float | None = None

    def diff_size(self) -> tuple[int, int]:
        """(lines_added, lines_removed) — naive but useful for the readability discretion."""
        added = removed = 0
        for e in self.edits:
            old_lines = e.old.count("\n") + (1 if e.old else 0)
            new_lines = e.new.count("\n") + (1 if e.new else 0)
            if e.kind == "search_replace":
                if new_lines > old_lines:
                    added += new_lines - old_lines
                else:
                    removed += old_lines - new_lines
            elif e.kind == "insert_after":
                added += new_lines
        return added, removed


def parse_plan_json(raw: str) -> PatchPlan:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    data = json.loads(raw)
    edits = [
        Edit(
            file=e["file"],
            old=e["old"],
            new=e["new"],
            kind=e.get("kind", "search_replace"),
        )
        for e in data.get("edits", [])
    ]
    return PatchPlan(
        hypothesis=data["hypothesis"],
        category=data.get("category", "mixed"),
        rationale=data.get("rationale", ""),
        edits=edits,
        expected_speedup_ms=data.get("expected_speedup_ms"),
        expected_loss_delta=data.get("expected_loss_delta"),
    )


def apply_plan(plan: PatchPlan, worktree: Path, config: Config) -> str:
    """Apply plan inside a worktree, return a unified-diff-ish summary string.

    Raises MutatorError if any rule check or apply fails.
    """
    if not plan.edits:
        raise MutatorError("plan contains no edits")

    edits_dicts = [e.to_dict() for e in plan.edits]
    violation = check_patch(edits_dicts, repo_root=worktree, editable_files=config.editable_files)
    if violation:
        raise MutatorError(f"rule_violation:{violation.code}: {violation.message}")

    diff_chunks: list[str] = []
    for i, edit in enumerate(plan.edits):
        path = worktree / edit.file
        if not path.exists():
            raise MutatorError(f"edit[{i}]: missing file {edit.file}")
        text = path.read_text()
        new_text = _apply_edit(text, edit, idx=i)
        path.write_text(new_text)
        diff_chunks.append(_short_diff(edit.file, edit, kind=edit.kind))

    # Post-write file-level rule check (catches surface anchors that weren't in old/new).
    for f in config.editable_files:
        viol = check_resulting_file(worktree / f)
        if viol:
            raise MutatorError(f"post_write_violation:{viol.code}: {viol.message}")

    return "\n\n".join(diff_chunks)


def _apply_edit(source: str, edit: Edit, *, idx: int) -> str:
    count = source.count(edit.old)
    if count == 0:
        raise MutatorError(f"edit[{idx}] {edit.file}: old_string not found")
    if count > 1:
        raise MutatorError(f"edit[{idx}] {edit.file}: old_string matched {count} times (must be 1)")

    if edit.kind == "search_replace":
        return source.replace(edit.old, edit.new, 1)
    if edit.kind == "insert_after":
        i = source.index(edit.old) + len(edit.old)
        return source[:i] + edit.new + source[i:]
    raise MutatorError(f"edit[{idx}] unknown kind: {edit.kind}")


def _short_diff(file: str, edit: Edit, *, kind: str) -> str:
    cap = 400
    old_preview = edit.old if len(edit.old) <= cap else edit.old[:cap] + "..."
    new_preview = edit.new if len(edit.new) <= cap else edit.new[:cap] + "..."
    head = f"{kind} {file}"
    return f"--- {head}\n[old]\n{old_preview}\n[new]\n{new_preview}"


def revert_files(worktree: Path, snapshots: dict[str, str]) -> None:
    """Used when a run errors out mid-way; restores the snapshotted files."""
    for rel, content in snapshots.items():
        (worktree / rel).write_text(content)

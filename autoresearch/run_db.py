"""Append-only run database. The agent's long-horizon memory.

Why this exists: after a week of 24/7 operation (~700-2800 iterations), the planner
cannot remember every prior attempt from natural-language summaries alone. This file
stores a fixed-schema row per iteration so we can cheaply ask:

  - "have we tried THIS patch before?" (edits-hash lookup → dedup)
  - "how have OPTIMIZER ideas done lately?" (per-category roll-up)
  - "what are the last 20 wins, in chain order?" (planner context)
  - "which crashes are dominating?" (failure-signature clustering)

Storage is JSONL so a human can `tail -f` it; on daemon start we replay the file
into in-memory indices and append from there. Single-process; no locking needed.

Schema is intentionally narrow — anything large (full log, full patch text) stays
in .autoresearch/runs/<id>/ and is referenced by run_id only.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def edits_hash(edits: Iterable[dict]) -> str:
    """Canonical sha1 of a list of edits. Whitespace-insensitive, kind-aware.

    Two planner ideas that propose the same logical change always produce the
    same hash; cosmetic differences (trailing newline, kind default) collapse.
    """
    canonical = sorted(
        (
            e.get("file", ""),
            e.get("kind", "search_replace"),
            (e.get("old", "") or "").strip(),
            (e.get("new", "") or "").strip(),
        )
        for e in (edits or [])
    )
    blob = json.dumps(canonical, sort_keys=True)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:12]


def hypothesis_hash(hypothesis: str) -> str:
    """Loose hash for prose dedup. Lowercased + collapsed whitespace."""
    norm = " ".join((hypothesis or "").lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12]


@dataclass
class RunRecord:
    """One row per iteration. Keep this narrow — large blobs live elsewhere."""
    run_id: str
    ts: str
    parent_sha: str | None
    category: str
    hypothesis: str
    edits_hash: str
    hypothesis_hash: str
    verdict: str                   # win | loss | crash | precheck_failed | patch_rejected | invalid_loss | dry_run
    val_loss: float | None = None
    train_time_ms: int | None = None
    duration_s: float = 0.0
    is_replication: bool = False
    batch_id: str | None = None
    replication_of: str | None = None
    error: str | None = None
    diff_lines_added: int | None = None
    diff_lines_removed: int | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        # Tolerate extra/missing keys so format can evolve without breaking history.
        valid = {k: d.get(k) for k in cls.__annotations__ if k in d}
        valid.setdefault("run_id", "")
        valid.setdefault("ts", "")
        valid.setdefault("parent_sha", None)
        valid.setdefault("category", "mixed")
        valid.setdefault("hypothesis", "")
        valid.setdefault("edits_hash", "")
        valid.setdefault("hypothesis_hash", "")
        valid.setdefault("verdict", "crash")
        return cls(**valid)


# Verdicts that mean "we have a real answer for this patch; don't redo it."
TERMINAL_VERDICTS = frozenset({
    "win", "loss", "invalid_loss", "precheck_failed", "patch_rejected",
})

# Verdicts where a re-attempt might be worthwhile (flaky GPU, transient crash, etc.).
RETRYABLE_VERDICTS = frozenset({"crash"})


@dataclass
class RunDB:
    path: Path
    records: list[RunRecord] = field(default_factory=list)
    by_edits_hash: dict[str, list[int]] = field(default_factory=dict)  # hash → indices
    by_category: dict[str, list[int]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "RunDB":
        db = cls(path=path)
        if not path.exists():
            return db
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            rec = RunRecord.from_dict(d)
            db._index(rec)
            db.records.append(rec)
        return db

    def _index(self, rec: RunRecord) -> None:
        idx = len(self.records)
        if rec.edits_hash:
            self.by_edits_hash.setdefault(rec.edits_hash, []).append(idx)
        self.by_category.setdefault(rec.category, []).append(idx)

    def append(self, rec: RunRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(asdict(rec), default=str) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            os.write(fd, line.encode("utf-8"))
        finally:
            os.close(fd)
        self._index(rec)
        self.records.append(rec)

    # ----- queries used by planner / daemon -----------------------------------

    def attempts_with_edits(self, h: str) -> list[RunRecord]:
        return [self.records[i] for i in self.by_edits_hash.get(h, [])]

    def is_redundant(self, h: str) -> RunRecord | None:
        """Return the most recent terminal attempt at this edits-hash, or None.

        Caller uses this to decide whether to skip an LLM-proposed edit.
        Replications don't count — they're by design re-attempts.
        """
        for rec in reversed(self.attempts_with_edits(h)):
            if rec.is_replication:
                continue
            if rec.verdict in TERMINAL_VERDICTS:
                return rec
        return None

    def category_stats(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for rec in self.records:
            if rec.is_replication:
                continue
            d = out.setdefault(rec.category, {})
            d[rec.verdict] = d.get(rec.verdict, 0) + 1
            d["_total"] = d.get("_total", 0) + 1
        return out

    def recent(self, n: int) -> list[RunRecord]:
        return list(reversed(self.records[-n:]))

    def recent_dedup_hints(self, n: int = 12) -> list[tuple[str, str, str]]:
        """For planner context: (hash, verdict, hypothesis-snippet) of last N
        terminal attempts. Shows the LLM *what we already tried* so it diversifies.
        """
        out: list[tuple[str, str, str]] = []
        for rec in reversed(self.records):
            if rec.is_replication:
                continue
            if rec.verdict not in TERMINAL_VERDICTS:
                continue
            out.append((rec.edits_hash, rec.verdict, (rec.hypothesis or "")[:120]))
            if len(out) >= n:
                break
        return out

    def failure_signatures(self, n: int = 8) -> dict[str, int]:
        """Cheap frequency count of error-message heads over last 50 crashes.

        No clustering — just first-80-chars of `error`. Surfaces "OOM in attention"
        vs "compile timeout" so the planner can avoid the dominant failure mode.
        """
        counts: dict[str, int] = {}
        crashes = [r for r in reversed(self.records) if r.verdict == "crash" and r.error]
        for rec in crashes[:50]:
            sig = (rec.error or "")[:80].strip()
            counts[sig] = counts.get(sig, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1])[:n])

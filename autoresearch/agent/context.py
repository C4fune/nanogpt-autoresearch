"""Build the planner's hot context. Constant-size regardless of run count.

Key invariant: this function never reads a full training log or the full source
of train_gpt.py. It assembles short, pre-distilled artifacts.

After a week of 24/7 operation the lessons.md and recent-summaries alone are
not enough — the planner needs to see (a) the full wins chain, (b) per-category
attempt rollup, (c) which patches we've ALREADY tried (dedup hints), (d) the
dominant crash signatures. All four come from run_db.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from autoresearch.config import Config
from autoresearch.paths import Paths
from autoresearch.run_db import RunDB

RULES_HEADER = """\
# Speedrun rules (Track 1) — hard constraints

1. Do NOT modify train/val data pipelines, val_tokens, val_files, train_files, or any
   data shard / BOS-alignment logic. Hard-rejected by the mutator.
2. Final mean val_loss must be <= 3.28 (with p<0.01 across replication runs for ML
   changes). Pure-systems changes (kernels, comm, dtype) waive the p-value rule.
3. No new `torch._inductor.config.*` enables and no `coordinate_descent_tuning`.
4. Final timed train_time_ms must be lower than the current best to count as a win.

## Editable
Only `train_gpt.py` and `triton_kernels.py`. Edits are search-replace; the `old`
substring must match exactly once. `kind: insert_after` is also available for
adding new code blocks immediately after a known anchor.

## Self-regulation
- Stagnant best for many runs -> bias broad / new methods.
- Recent runs all crashed at compile -> bias narrow / refine.
- Best is far above prior record -> exploitation, smaller diffs.
- Readability counts: 200 lines for 300ms is fine, 500 lines for 50ms is not.
"""


@dataclass(frozen=True)
class HotContext:
    text: str
    char_count: int
    sections: tuple[str, ...]


def build(config: Config, *, run_db: RunDB | None = None) -> HotContext:
    paths = config.paths
    budget = config.llm
    parts: list[str] = []
    used = 0
    section_titles: list[str] = []

    def add(title: str, body: str, cap: int) -> None:
        nonlocal used
        body = _truncate(body, cap)
        block = f"## {title}\n{body}\n"
        if used + len(block) > budget.total_chars:
            block = _truncate(block, max(0, budget.total_chars - used))
        if not block.strip():
            return
        parts.append(block)
        used += len(block)
        section_titles.append(title)

    add("Rules", RULES_HEADER, budget.rules_chars)
    add("Current state", _state_block(config), budget.state_chars)
    add("Wins chain (advanced, oldest→newest)", _wins_chain(config), budget.wins_chain_chars)
    add("Category stats (non-replication attempts)",
        _category_stats(run_db), budget.category_stats_chars)
    add("Already-attempted patches — do NOT repeat",
        _dedup_hints(run_db), budget.dedup_hints_chars)
    add("Top crash signatures", _failure_signatures(run_db), budget.failure_sig_chars)
    add("Lessons", _safe_read(paths.lessons_md, "(none yet)"), budget.lessons_chars)
    add("Code map", _safe_read(paths.code_map_md, "(no code map; run bootstrap)"), budget.code_map_chars)
    add("Recent run summaries (last 10)", _recent_summaries(paths, n=10), budget.summaries_chars)
    add("Top record cards", _record_cards(paths, k=5), budget.record_index_chars)
    add("Backlog (pending)", _backlog_preview(paths, k=8), 1500)

    text = "\n".join(parts)
    return HotContext(text=text, char_count=len(text), sections=tuple(section_titles))


def _state_block(config: Config) -> str:
    p = config.paths
    best = _safe_read(p.best_json, "{}")
    budget = _safe_read(p.budget_json, "{}")
    return f"### best.json\n```json\n{best}\n```\n### budget.json\n```json\n{budget}\n```"


def _recent_summaries(paths: Paths, n: int) -> str:
    if not paths.runs_dir.exists():
        return "(no runs yet)"
    run_dirs = sorted(paths.runs_dir.iterdir(), reverse=True)[:n]
    if not run_dirs:
        return "(no runs yet)"
    out: list[str] = []
    for d in run_dirs:
        s = d / "summary.md"
        if not s.exists():
            continue
        text = s.read_text(errors="replace").strip()
        out.append(f"### {d.name}\n{text}\n")
    return "\n".join(out) or "(no summaries yet)"


def _record_cards(paths: Paths, k: int) -> str:
    """Top-k record cards by tags/recency. Cheap heuristic: most recent wins."""
    if not paths.record_index_jsonl.exists():
        return "(record index not built; run bootstrap)"
    cards = []
    for line in paths.record_index_jsonl.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            cards.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    cards.sort(key=lambda c: c.get("date", ""), reverse=True)
    out = []
    for c in cards[:k]:
        out.append(
            f"- {c.get('folder')}: {c.get('summary', '')[:300]}"
            f" (val={c.get('val_loss')}, time={c.get('train_time_ms')}ms)"
        )
    return "\n".join(out) or "(no records indexed)"


def _backlog_preview(paths: Paths, k: int) -> str:
    if not paths.backlog_jsonl.exists():
        return "(empty)"
    lines = [l for l in paths.backlog_jsonl.read_text().splitlines() if l.strip()]
    items = []
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if obj.get("consumed_at") is None:
            items.append(obj)
    items.sort(key=lambda x: x.get("priority", 0), reverse=True)
    items = items[:k]
    if not items:
        return "(empty)"
    return "\n".join(
        f"- [{i.get('priority', 0):.2f}] {i.get('hypothesis', '')[:150]}"
        for i in items
    )


def _wins_chain(config: Config) -> str:
    """Render every ADVANCED win (from pending_wins.json) in chain order.

    This is the agent's cumulative PR history — what we've already shipped.
    Distinct from `lessons.md` (compressed prose) and `Recent run summaries`
    (last 10 attempts of any kind).
    """
    p = config.paths.state_dir / "pending_wins.json"
    if not p.exists():
        return "(no wins yet — agent is still pre-first-win)"
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return "(pending_wins.json unreadable)"
    advanced = [
        b for b in data.get("batches", {}).values()
        if b.get("status") == "advanced"
    ]
    advanced.sort(key=lambda b: b.get("created_at", ""))
    if not advanced:
        return "(no advanced wins yet)"
    rows = []
    for i, b in enumerate(advanced, 1):
        m = b.get("candidate_metrics", {}) or {}
        rows.append(
            f"{i:>2}. [{b.get('classification', '?'):8s}] "
            f"t={m.get('train_time_ms', '?')}ms "
            f"val={m.get('val_loss', '?')} — {(b.get('hypothesis') or '')[:120]}"
        )
    return "\n".join(rows)


def _category_stats(run_db: RunDB | None) -> str:
    if run_db is None or not run_db.records:
        return "(no run history yet)"
    stats = run_db.category_stats()
    if not stats:
        return "(no non-replication attempts yet)"
    rows = []
    for cat in sorted(stats):
        s = stats[cat]
        total = s.get("_total", 0)
        wins = s.get("win", 0)
        losses = s.get("loss", 0)
        crashes = s.get("crash", 0)
        rejected = s.get("patch_rejected", 0) + s.get("precheck_failed", 0)
        invalid = s.get("invalid_loss", 0)
        win_rate = (wins / total * 100) if total else 0.0
        rows.append(
            f"- {cat:12s} n={total:>3}  wins={wins:>2} "
            f"loss={losses:>2} crash={crashes:>2} rejected={rejected:>2} "
            f"invalid_loss={invalid:>2}  win_rate={win_rate:.0f}%"
        )
    return "\n".join(rows)


def _dedup_hints(run_db: RunDB | None) -> str:
    if run_db is None:
        return "(no run_db)"
    hints = run_db.recent_dedup_hints(n=15)
    if not hints:
        return "(no prior attempts to dedup against)"
    rows = [f"- [{h} {verdict:14s}] {snippet}" for h, verdict, snippet in hints]
    rows.insert(0, "Identical patches will be auto-rejected at intake; cite a NEW angle:")
    return "\n".join(rows)


def _failure_signatures(run_db: RunDB | None) -> str:
    if run_db is None:
        return "(no run_db)"
    sigs = run_db.failure_signatures(n=6)
    if not sigs:
        return "(no recent crashes — proceed normally)"
    return "\n".join(f"- ×{count:<3} {sig}" for sig, count in sigs.items())


def _safe_read(path: Path, fallback: str = "") -> str:
    try:
        return path.read_text(errors="replace")
    except (OSError, FileNotFoundError):
        return fallback


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."

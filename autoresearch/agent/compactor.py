"""Periodic compactor: rewrites knowledge/lessons.md from recent run summaries.

Runs every N runs. Keeps lessons.md small (~2KB) so the planner prompt size stays
constant whether we've done 50 or 5000 attempts. Old summaries stay on disk.
"""

from __future__ import annotations

from pathlib import Path

from autoresearch.agent.llm import LLMClient
from autoresearch.config import Config


COMPACTOR_SYSTEM = """\
You maintain a tiny, dense rulebook for an autonomous ML researcher (the file
knowledge/lessons.md). It must stay under ~2 KB. Bullet-point only. No prose.
Each lesson is a falsifiable rule learned from runs, not a wish list.
"""


COMPACTOR_USER_TEMPLATE = """\
Rewrite lessons.md from scratch using the existing lessons plus the recent run summaries.

## Existing lessons.md
{lessons}

## Recent run summaries (most recent first)
{summaries}

Rules:
- Output ONLY the new lessons.md content (no fences, no prose).
- <= 30 bullets total. Drop redundancies.
- Bias toward things that have replicated or have clear mechanism.
- Each bullet under 25 words.
- Group with `## Optimizer / Schedule / Kernel / Architecture / Systems / Process` headings only when there's >= 2 bullets in that group.
"""


def compact_lessons(
    *,
    config: Config,
    llm: LLMClient,
    n_summaries: int = 30,
) -> int:
    """Returns number of summaries fed in. 0 means no-op."""
    summaries = _recent_summaries_text(config, n_summaries)
    if not summaries:
        return 0

    existing = ""
    if config.paths.lessons_md.exists():
        existing = config.paths.lessons_md.read_text(errors="replace")

    user = COMPACTOR_USER_TEMPLATE.format(
        lessons=existing or "(empty)",
        summaries=summaries,
    )
    rewrite = llm.complete(COMPACTOR_SYSTEM, user).strip()
    if not rewrite.startswith("#"):
        rewrite = "# Lessons\n\n" + rewrite
    config.paths.lessons_md.write_text(rewrite + "\n")
    return n_summaries


def _recent_summaries_text(config: Config, n: int) -> str:
    runs = sorted(config.paths.runs_dir.iterdir(), reverse=True) if config.paths.runs_dir.exists() else []
    parts: list[str] = []
    used = 0
    for d in runs:
        if len(parts) >= n:
            break
        s = d / "summary.md"
        if not s.exists():
            continue
        text = s.read_text(errors="replace").strip()
        chunk = f"### {d.name}\n{text}\n"
        if used + len(chunk) > 60_000:
            break
        parts.append(chunk)
        used += len(chunk)
    return "\n".join(parts)

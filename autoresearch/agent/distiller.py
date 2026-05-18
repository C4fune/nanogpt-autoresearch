"""Post-run summarizer. Runs once per attempt, emits ~150 words to summary.md.

After this runs the full log.txt.gz becomes forensics-only — never re-read by the
planner. This is the single most important compression step in the system.
"""

from __future__ import annotations

import json
from pathlib import Path

from autoresearch.agent.llm import LLMClient
from autoresearch.config import Config


DISTILL_SYSTEM = """\
You write extremely concise post-run digests for an autonomous ML researcher.
Each digest is read by future planners; it's the only thing they see about this run.
Be specific: name parameters, cite numbers, say what (if anything) was learned.
Never speculate about runs you didn't see.
"""


DISTILL_USER_TEMPLATE = """\
Distill this run into <= 180 words of markdown.

## Hypothesis
{hypothesis}

## Category
{category}

## Rationale (planner's own)
{rationale}

## Patch summary
{patch_preview}

## Verdict
{verdict}

## Metrics
{metrics}

## Stderr tail (if any)
{stderr_tail}

Format:
- One-line headline.
- 3-5 bullets covering what changed, what happened, what we learned (if anything).
- One bullet "next angle:" suggesting how a future planner might iterate.
- No filler, no apologies.
"""


def distill_run(
    *,
    config: Config,
    llm: LLMClient,
    run_dir: Path,
    hypothesis: str,
    category: str,
    rationale: str,
    verdict: str,
    metrics: dict | None,
    patch_preview: str,
    stderr_tail: str = "",
) -> str:
    user = DISTILL_USER_TEMPLATE.format(
        hypothesis=hypothesis,
        category=category,
        rationale=rationale,
        patch_preview=_clip(patch_preview, 1200),
        verdict=verdict,
        metrics=json.dumps(metrics or {}, indent=2),
        stderr_tail=_clip(stderr_tail, 800) or "(none)",
    )
    text = llm.complete(DISTILL_SYSTEM, user).strip()
    summary_path = run_dir / "summary.md"
    summary_path.write_text(text + "\n")
    return text


def _clip(s: str, n: int) -> str:
    s = s.replace("\r", "")
    return s if len(s) <= n else s[: n - 3] + "..."

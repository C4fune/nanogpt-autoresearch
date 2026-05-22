"""Refill the backlog when it runs low.

The planner is invoked rarely (every ~5 runs) and outputs a batch of ideas.
Decoupling proposal from execution means the daemon can keep iterating even when
the LLM API blips for a while.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from autoresearch import backlog, journal
from autoresearch.agent.context import build as build_context
from autoresearch.agent.llm import LLMClient
from autoresearch.config import Config
from autoresearch.run_db import RunDB, edits_hash

log = logging.getLogger(__name__)


PLANNER_USER_TEMPLATE = """\
{context}

# Task: refill the backlog

Read the current state above. Propose {n} new candidate experiments as a single JSON object.

Schema:
{{
  "ideas": [
    {{
      "hypothesis": "one sentence describing what you're trying",
      "category": "optimizer|schedule|kernel|architecture|systems|mixed",
      "rationale": "2-4 sentences citing record cards or run summaries",
      "priority": 0.0-1.0,
      "tags": ["short", "tags"],
      "edits": [
        {{"file": "train_gpt.py", "old": "...", "new": "...", "kind": "search_replace"}}
      ]
    }}
  ]
}}

Constraints:
- Each `old` MUST appear exactly once in the listed file. If you're not sure, request a code excerpt instead and propose fewer ideas this round.
- Do not edit the data pipeline (`distributed_data_generator`, val_tokens, etc.). The mutator will reject those.
- Diversify: mix narrow refinements with at least one broader / structural attempt unless the recent summaries show structural changes are crashing.
- Prioritize ideas backed by record-card evidence over pure speculation.

Output ONLY the JSON object, no prose, no fences.
"""


def refill_backlog_if_needed(
    config: Config,
    llm: LLMClient,
    *,
    target_size: int = 10,
    batch_size: int = 5,
    run_db: RunDB | None = None,
) -> int:
    """Top up the backlog. Returns number of ideas added (0 if no refill was needed).

    If `run_db` is provided, edits that match a prior terminal attempt are dropped
    before being appended. Keeps the agent from grinding the same patch forever.
    """
    remaining = backlog.remaining_count(config.paths.backlog_jsonl)
    if remaining >= config.llm.backlog_low_threshold:
        return 0

    system = _system_prompt()
    ctx = build_context(config, run_db=run_db, include_source=True)
    user = PLANNER_USER_TEMPLATE.format(context=ctx.text, n=batch_size)

    raw = llm.complete(system, user)
    ideas = _parse_ideas(raw)
    if not ideas:
        # Make the failure visible in the journal — otherwise the loop just
        # quietly emits `backlog_empty` forever and you can't tell why.
        log.warning("Planner returned no parseable ideas")
        journal.emit(
            config.paths.journal_jsonl, "planner_no_ideas",
            raw_head=raw[:400], raw_len=len(raw),
        )
        return 0

    added = 0
    rejected_dup = 0
    for idea in ideas:
        try:
            if run_db is not None:
                h = edits_hash(idea.get("edits", []))
                prior = run_db.is_redundant(h)
                if prior is not None:
                    rejected_dup += 1
                    journal.emit(
                        config.paths.journal_jsonl, "idea_dedup_rejected",
                        edits_hash=h, prior_run_id=prior.run_id,
                        prior_verdict=prior.verdict,
                        hypothesis_head=(idea.get("hypothesis") or "")[:120],
                    )
                    continue
            backlog.append(config.paths.backlog_jsonl, idea)
            added += 1
        except (KeyError, ValueError) as e:
            log.warning("Skipping malformed idea: %s", e)
            journal.emit(
                config.paths.journal_jsonl, "planner_malformed_idea",
                error=str(e), idea_keys=sorted(idea.keys()) if isinstance(idea, dict) else None,
            )
    if rejected_dup:
        journal.emit(
            config.paths.journal_jsonl, "planner_dedup_summary",
            rejected=rejected_dup, accepted=added,
        )
    return added


def _parse_ideas(raw: str) -> list[dict[str, Any]]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    items = data.get("ideas", [])
    out: list[dict[str, Any]] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        if "hypothesis" not in it or "edits" not in it:
            continue
        if not isinstance(it["edits"], list) or not it["edits"]:
            continue
        # Normalize.
        it.setdefault("category", "mixed")
        it.setdefault("rationale", "")
        it.setdefault("priority", 0.5)
        it.setdefault("tags", [])
        out.append(it)
    return out


def _system_prompt() -> str:
    from pathlib import Path
    return (Path(__file__).parent.parent / "prompts" / "planner_system.txt").read_text()

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

from autoresearch import backlog
from autoresearch.agent.context import build as build_context
from autoresearch.agent.llm import LLMClient
from autoresearch.config import Config

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
) -> int:
    """Top up the backlog. Returns number of ideas added (0 if no refill was needed)."""
    remaining = backlog.remaining_count(config.paths.backlog_jsonl)
    if remaining >= config.llm.backlog_low_threshold:
        return 0

    system = _system_prompt()
    ctx = build_context(config)
    user = PLANNER_USER_TEMPLATE.format(context=ctx.text, n=batch_size)

    raw = llm.complete(system, user)
    ideas = _parse_ideas(raw)
    if not ideas:
        log.warning("Planner returned no parseable ideas")
        return 0

    added = 0
    for idea in ideas:
        try:
            backlog.append(config.paths.backlog_jsonl, idea)
            added += 1
        except (KeyError, ValueError) as e:
            log.warning("Skipping malformed idea: %s", e)
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

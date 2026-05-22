"""LLM client. Default backend is Anthropic Claude; OpenAI kept as fallback.

Selection order:
  1. AUTORESEARCH_BACKEND env (one of: anthropic, openai)
  2. ANTHROPIC_API_KEY -> anthropic
  3. OPENAI_API_KEY    -> openai
  4. error

Tracing: wrap any client with LLMTracer to record every (system, user, response)
triple to disk. The daemon uses this so every iteration's prompts and responses
are inspectable via `python -m autoresearch traces` — no run is a black box.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

_log = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def default_client(temperature: float = 0.4) -> "LLMClient":
    backend = os.environ.get("AUTORESEARCH_BACKEND", "").lower()
    if backend == "anthropic" or (not backend and os.environ.get("ANTHROPIC_API_KEY")):
        return AnthropicClient(temperature=temperature)
    if backend == "openai" or os.environ.get("OPENAI_API_KEY"):
        return OpenAICompatibleClient(temperature=temperature)
    raise RuntimeError(
        "No LLM backend configured. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
    )


class AnthropicClient:
    """Anthropic Messages API. Default model: claude-opus-4-7.

    Extended thinking is enabled on claude-opus-4-7+ — the model's reasoning
    is returned as `thinking` content blocks and exposed on `self.last_thinking`.
    The tracer copies it into each traces.jsonl row so the user can see the
    full thought process behind every decision.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.4,
        max_tokens: int | None = None,
        thinking_budget_tokens: int = 8000,
    ) -> None:
        # Default to the most capable model; user can override via AUTORESEARCH_MODEL.
        self.model = model or os.environ.get("AUTORESEARCH_MODEL", "claude-opus-4-7")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        # max_tokens has to cover BOTH thinking AND the JSON response.
        # With effort=high adaptive thinking, opus-4-7 can burn 30-60k tokens
        # of reasoning. Default 64k so the actual response isn't truncated.
        # Override via AUTORESEARCH_MAX_TOKENS.
        if max_tokens is None:
            max_tokens = int(os.environ.get("AUTORESEARCH_MAX_TOKENS", "64000"))
        self.max_tokens = max_tokens
        self.thinking_budget_tokens = thinking_budget_tokens
        # Populated after each complete() call so LLMTracer can record it.
        self.last_thinking: str = ""
        self.last_usage: dict | None = None
        self.last_stop_reason: str | None = None

    def _supports_thinking(self) -> bool:
        return self.model.startswith(("claude-opus-4-7", "claude-sonnet-4-7", "claude-haiku-4-7"))

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic") from e

        client = anthropic.Anthropic(api_key=self.api_key)
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if self._supports_thinking():
            # Extended thinking on opus-4-7+. The API surface here uses
            # adaptive thinking + output_config.effort; the old
            # `thinking.type=enabled, budget_tokens=...` form is rejected.
            # Temperature is also rejected for these models, so don't set it.
            effort = os.environ.get("AUTORESEARCH_THINKING_EFFORT", "high")
            kwargs["thinking"] = {"type": "adaptive"}
            kwargs["output_config"] = {"effort": effort}
        else:
            kwargs["temperature"] = self.temperature

        # Anthropic SDK refuses .create() when max_tokens implies a possibly
        # >10-min request. With 64K max_tokens + adaptive thinking, opus-4-7
        # can legitimately reason for that long. Use streaming and pull the
        # final aggregated Message; same response shape, no 10-min wall.
        with client.messages.stream(**kwargs) as stream:
            resp = stream.get_final_message()

        # Split content into thinking + text. Thinking is the model's reasoning;
        # text is the actual response payload (JSON ideas, distillation, etc.).
        text_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in resp.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                t = getattr(block, "thinking", None)
                if t:
                    thinking_parts.append(t)
            elif btype == "text" or getattr(block, "text", None) is not None:
                t = getattr(block, "text", None)
                if t:
                    text_parts.append(t)

        self.last_thinking = "".join(thinking_parts)
        usage = getattr(resp, "usage", None)
        self.last_usage = {
            "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
            "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
        }
        self.last_stop_reason = getattr(resp, "stop_reason", None)
        return "".join(text_parts)


class OpenAICompatibleClient:
    """Fallback for OpenAI / OpenRouter / local OpenAI-style endpoints."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.4,
    ) -> None:
        self.model = model or os.environ.get("AUTORESEARCH_MODEL", "gpt-4o")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self.temperature = temperature

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError("pip install openai") from e

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""


class DryRunClient:
    """For pipeline tests with no API. Returns canned responses keyed by user prompt."""

    model = "dryrun"

    def complete(self, system: str, user: str) -> str:
        if "refill the backlog" in user.lower():
            return """{
  "ideas": [
    {
      "hypothesis": "Probe loss-margin: add 5 extension iterations",
      "category": "schedule",
      "rationale": "Cheap probe to verify pipeline; not a real research idea.",
      "priority": 0.4,
      "tags": ["dryrun", "schedule"],
      "edits": [{
        "file": "train_gpt.py",
        "old": "    num_extension_iterations: int = 40",
        "new": "    num_extension_iterations: int = 45",
        "kind": "search_replace"
      }]
    }
  ]
}"""
        if "distill this run" in user.lower():
            return "Mock distillation: pipeline test run, no real signal."
        if "rewrite lessons" in user.lower():
            return "# Lessons\n\n- Mock lessons; replace once live LLM is wired.\n"
        return "{}"


# ----- Tracing ----------------------------------------------------------------

# Per-field caps. Disk is cheap, but a runaway prompt shouldn't write multi-GB
# JSONL rows. 64 KB per field is enough to capture the full planner context.
_TRACE_FIELD_CAP = 64 * 1024


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(s: str, cap: int = _TRACE_FIELD_CAP) -> str:
    if not s:
        return s
    return s if len(s) <= cap else s[:cap] + f"\n...[truncated; full={len(s)} chars]"


def _append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, default=str) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


class LLMTracer:
    """Wraps another LLMClient; records each call to disk before returning.

    Why this exists: every win produced by the agent is the result of a chain
    of LLM decisions (planner → patch → distill → compact). For a 24/7 agent
    you need to be able to ask "what did the model actually say on the iteration
    that produced this win?" Without a trace, every run is a black box.

    Writes go to two places:
      - .autoresearch/traces.jsonl       (global, append-only)
      - .autoresearch/runs/<id>/llm_calls.jsonl  (only when tagged with run_id)
    """

    # Mark the wrapper as an LLMClient so isinstance/duck checks succeed.

    def __init__(self, inner: LLMClient, *, traces_path: Path, runs_dir: Path):
        self.inner = inner
        self.traces_path = traces_path
        self.runs_dir = runs_dir
        self._tag: dict = {}
        self._call_seq = 0

    # ----- tagging -----

    def set_tag(self, **kwargs) -> None:
        """Replace tag context for subsequent calls. Pass purpose= and optionally run_id=.
        Use clear_tag() between unrelated call sites to avoid sticky run_ids leaking
        into planner/compactor traces.
        """
        self._tag = {k: v for k, v in kwargs.items() if v is not None}

    def clear_tag(self) -> None:
        self._tag = {}

    @property
    def model(self) -> str:
        return getattr(self.inner, "model", "unknown")

    # ----- LLMClient surface -----

    def complete(self, system: str, user: str) -> str:
        self._call_seq += 1
        t0 = time.monotonic()
        rec = {
            "ts": _utc_now(),
            "seq": self._call_seq,
            "model": self.model,
            **self._tag,
            "system_chars": len(system),
            "user_chars": len(user),
            "system": _clip(system),
            "user": _clip(user),
        }
        try:
            response = self.inner.complete(system, user)
            rec["response"] = _clip(response)
            rec["response_chars"] = len(response)
            # Capture extended thinking + token usage if the wrapped client exposes them.
            thinking = getattr(self.inner, "last_thinking", "") or ""
            if thinking:
                rec["thinking"] = _clip(thinking)
                rec["thinking_chars"] = len(thinking)
            usage = getattr(self.inner, "last_usage", None)
            if usage:
                rec["usage"] = usage
            stop = getattr(self.inner, "last_stop_reason", None)
            if stop:
                rec["stop_reason"] = stop
            rec["ok"] = True
            return response
        except Exception as e:
            rec["error"] = str(e)
            rec["ok"] = False
            raise
        finally:
            rec["duration_s"] = round(time.monotonic() - t0, 3)
            try:
                _append_jsonl(self.traces_path, rec)
                run_id = self._tag.get("run_id")
                if run_id:
                    _append_jsonl(self.runs_dir / run_id / "llm_calls.jsonl", rec)
            except Exception as e:
                # Tracing must NEVER take down the daemon — but DO log so the
                # user can see when trace writes fail (silent swallow burned
                # us before: empty traces.jsonl with no explanation).
                _log.warning("LLMTracer write failed: %s", e, exc_info=True)

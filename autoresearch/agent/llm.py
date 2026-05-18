"""LLM client. Default backend is Anthropic Claude; OpenAI kept as fallback.

Selection order:
  1. AUTORESEARCH_BACKEND env (one of: anthropic, openai)
  2. ANTHROPIC_API_KEY -> anthropic
  3. OPENAI_API_KEY    -> openai
  4. error
"""

from __future__ import annotations

import os
from typing import Protocol


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
    """Anthropic Messages API. Default model: claude-sonnet-4-5."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.4,
        max_tokens: int = 4096,
    ) -> None:
        # Default to the most capable model; user can override via AUTORESEARCH_MODEL.
        self.model = model or os.environ.get("AUTORESEARCH_MODEL", "claude-opus-4-5")
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError("pip install anthropic") from e

        client = anthropic.Anthropic(api_key=self.api_key)
        resp = client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        # Concatenate text blocks (resp.content is a list of content blocks).
        parts: list[str] = []
        for block in resp.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)


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

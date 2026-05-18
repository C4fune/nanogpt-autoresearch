"""Single on-disk tree for the running researcher.

Everything the agent needs to survive crashes / restarts lives under .autoresearch/.
Logs are gzip'd; nothing in this tree should ever be re-read whole into the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    repo_root: Path

    @property
    def root(self) -> Path:
        return self.repo_root / ".autoresearch"

    # state/
    @property
    def state_dir(self) -> Path:
        return self.root / "state"

    @property
    def best_json(self) -> Path:
        return self.state_dir / "best.json"

    @property
    def cursor_json(self) -> Path:
        return self.state_dir / "cursor.json"

    @property
    def budget_json(self) -> Path:
        return self.state_dir / "budget.json"

    # knowledge/
    @property
    def knowledge_dir(self) -> Path:
        return self.root / "knowledge"

    @property
    def lessons_md(self) -> Path:
        return self.knowledge_dir / "lessons.md"

    @property
    def code_map_md(self) -> Path:
        return self.knowledge_dir / "code_map.md"

    @property
    def record_index_jsonl(self) -> Path:
        return self.knowledge_dir / "record_index.jsonl"

    # ideas/
    @property
    def ideas_dir(self) -> Path:
        return self.root / "ideas"

    @property
    def backlog_jsonl(self) -> Path:
        return self.ideas_dir / "backlog.jsonl"

    # runs/
    @property
    def runs_dir(self) -> Path:
        return self.root / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id

    # event log
    @property
    def journal_jsonl(self) -> Path:
        return self.root / "journal.jsonl"

    # ephemeral worktrees (one per attempt)
    @property
    def worktrees_dir(self) -> Path:
        return self.root / "_worktrees"

    def worktree_for(self, run_id: str) -> Path:
        return self.worktrees_dir / run_id

    def ensure(self) -> None:
        for p in (
            self.state_dir,
            self.knowledge_dir,
            self.ideas_dir,
            self.runs_dir,
            self.worktrees_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

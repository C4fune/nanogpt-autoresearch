"""Runtime config — small, immutable, env-overrideable."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from autoresearch.paths import Paths


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class Targets:
    val_loss_max: float = 3.28
    # Record #80 (~1.406 min). Override via AUTORESEARCH_BASELINE_MS once we have local timing.
    baseline_train_time_ms: int = 84_360
    baseline_val_loss: float = 3.2796
    baseline_steps: int = 1480
    # Replication count for promoting a "win" to baseline (ML changes only).
    replication_n: int = 4
    # One-sided p-value threshold for advancing baseline.
    p_threshold: float = 0.01


@dataclass
class LLMBudget:
    """Token budgets keep the prompt size constant regardless of run count."""
    total_chars: int = 80_000
    rules_chars: int = 1_500
    state_chars: int = 600
    lessons_chars: int = 3_000
    code_map_chars: int = 2_000
    summaries_chars: int = 8_000      # last ~10 run summaries
    record_index_chars: int = 4_000   # ~5 most relevant record cards
    code_excerpt_chars: int = 8_000   # on-demand source slice
    # Long-horizon memory blocks (sourced from run_db / pending_wins):
    wins_chain_chars: int = 2_500     # advanced wins, in chain order
    category_stats_chars: int = 800   # per-category attempt/win rollup
    dedup_hints_chars: int = 2_000    # "we already tried these patches" head list
    failure_sig_chars: int = 1_000    # frequent crash signatures
    backlog_low_threshold: int = 5    # refill when fewer than this remain


@dataclass
class Config:
    repo_root: Path = field(default_factory=_repo_root)
    paths: Paths = field(init=False)
    targets: Targets = field(default_factory=Targets)
    llm: LLMBudget = field(default_factory=LLMBudget)

    editable_files: tuple[str, ...] = ("train_gpt.py", "triton_kernels.py")
    run_command: tuple[str, ...] = ("./run.sh",)
    # README warns torch.compile alone is ~7 min on a cold inductor cache.
    # Pad generously: a first-ever run on a fresh box is ~10 min compile +
    # ~1.5 min train + warmup; pathological compile blowups need headroom.
    # Override via AUTORESEARCH_RUN_TIMEOUT_S.
    run_timeout_s: int = 3000             # 50 min
    keep_last_n_logs: int = 200           # gzip kept; older deleted
    compact_every_n_runs: int = 25
    distill_word_target: int = 150

    def __post_init__(self) -> None:
        self.paths = Paths(repo_root=self.repo_root)
        self.paths.ensure()
        env_ms = os.environ.get("AUTORESEARCH_BASELINE_MS")
        if env_ms:
            self.targets.baseline_train_time_ms = int(env_ms)
        env_timeout = os.environ.get("AUTORESEARCH_RUN_TIMEOUT_S")
        if env_timeout:
            self.run_timeout_s = int(env_timeout)


def load_config() -> Config:
    return Config()

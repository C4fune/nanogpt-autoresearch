"""Parse training logs — metrics only, never return embedded source."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ValCheckpoint:
    step: int
    total_steps: int
    val_loss: float
    train_time_ms: int
    step_avg_ms: float


@dataclass(frozen=True)
class LogMetrics:
    log_path: str
    val_checkpoints: tuple[ValCheckpoint, ...]
    final: ValCheckpoint | None
    peak_memory_mib: int | None
    compile_note: str | None

    @property
    def beats_target_loss(self) -> bool:
        return self.final is not None and self.final.val_loss <= 3.28


_VAL_RE = re.compile(
    r"step:(\d+)/(\d+)\s+val_loss:([\d.]+)\s+train_time:(\d+)ms\s+step_avg:([\d.]+)ms"
)
_TRAIN_RE = re.compile(
    r"step:(\d+)/(\d+)\s+train_time:(\d+)ms\s+step_avg:([\d.]+)ms"
)
_PEAK_RE = re.compile(r"peak memory allocated:\s*(\d+)\s*MiB")


def _find_metrics_start(lines: list[str]) -> int:
    """Skip embedded train_gpt.py source at top of log."""
    for i, line in enumerate(lines):
        if line.startswith("Running Python "):
            return i
    for i, line in enumerate(lines):
        if line.startswith("step:") and "val_loss:" in line:
            return i
    return 0


def parse_log(path: Path | str) -> LogMetrics:
    path = Path(path)
    text = path.read_text(errors="replace")
    lines = text.splitlines()
    start = _find_metrics_start(lines)
    body = lines[start:]

    val_checkpoints: list[ValCheckpoint] = []
    last_train: ValCheckpoint | None = None
    peak: int | None = None
    compile_note: str | None = None

    for line in body:
        if "Compiling model and warming up" in line:
            compile_note = line.strip()
        m = _VAL_RE.search(line)
        if m:
            cp = ValCheckpoint(
                step=int(m.group(1)),
                total_steps=int(m.group(2)),
                val_loss=float(m.group(3)),
                train_time_ms=int(m.group(4)),
                step_avg_ms=float(m.group(5)),
            )
            val_checkpoints.append(cp)
            last_train = cp
            continue
        m = _TRAIN_RE.search(line)
        if m:
            last_train = ValCheckpoint(
                step=int(m.group(1)),
                total_steps=int(m.group(2)),
                val_loss=float("nan"),
                train_time_ms=int(m.group(3)),
                step_avg_ms=float(m.group(4)),
            )
        pm = _PEAK_RE.search(line)
        if pm:
            peak = int(pm.group(1))

    final = val_checkpoints[-1] if val_checkpoints else last_train
    return LogMetrics(
        log_path=str(path),
        val_checkpoints=tuple(val_checkpoints),
        final=final,
        peak_memory_mib=peak,
        compile_note=compile_note,
    )


def compact_log_summary(metrics: LogMetrics, *, max_val_points: int = 4) -> str:
    """One short paragraph for memory DB / LLM context."""
    if not metrics.final:
        return f"No metrics parsed from {metrics.log_path}"
    f = metrics.final
    parts = [
        f"final step {f.step}/{f.total_steps}",
        f"val_loss={f.val_loss:.4f}",
        f"train_time={f.train_time_ms}ms",
        f"step_avg={f.step_avg_ms:.2f}ms",
    ]
    if metrics.peak_memory_mib:
        parts.append(f"peak_mem={metrics.peak_memory_mib}MiB")
    if metrics.val_checkpoints and len(metrics.val_checkpoints) > 1:
        early = metrics.val_checkpoints[:max_val_points]
        curve = ", ".join(f"s{cp.step}={cp.val_loss:.3f}" for cp in early)
        parts.append(f"val_curve[{curve}]")
    return "; ".join(parts)

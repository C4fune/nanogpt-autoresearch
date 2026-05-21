"""Anchor-based line ranges into train_gpt.py / triton_kernels.py.

Used to:
  1. Build knowledge/code_map.md once at bootstrap (and refresh when train_gpt.py mtime changes).
  2. Fetch a small code excerpt on demand for the planner prompt — we never load the
     full 2000-line file into LLM context.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# (label, anchor_substring, lines_below) — first match wins.
# Anchors are scoped per file: a triton anchor will only be searched in
# triton_kernels.py and vice-versa. Previously every anchor was tried against
# every file, which left the triton code map empty.
TRAIN_GPT_ANCHORS: tuple[tuple[str, str, int], ...] = (
    ("hyperparameters", "class Hyperparameters:", 30),
    ("training_stages", "TRAINING_STAGES = [", 40),
    ("training_schedule", "training_schedule = TrainingSchedule", 5),
    ("training_manager", "class TrainingManager", 120),
    ("param_table", "self.param_table = {", 80),
    ("nor_muon", "class NorMuonAndAdam", 200),
    ("muon_helper", "def polar_express", 60),
    ("gpt_init", "class GPT(nn.Module):", 200),
    ("attention", "class CausalSelfAttention", 80),
    ("mlp", "ReLUSqrdMLP = ", 4),
    ("loss_fn_callsite", "FusedSoftcappedCrossEntropy.apply", 4),
    ("data_loader_NOEDIT", "def distributed_data_generator", 100),
    ("main_train_loop", "#        Training and validation", 80),
)

TRITON_ANCHORS: tuple[tuple[str, str, int], ...] = (
    ("xxt_kernel", "def XXT_kernel", 60),
    ("xxt_wrapper", "def XXT(A:", 40),
    ("xtx_kernel", "def XTX_kernel", 60),
    ("xtx_wrapper", "def XTX(A:", 40),
    ("ba_plus_caa_kernel", "def ba_plus_cAA_kernel", 60),
    ("ba_plus_caa_wrapper", "def ba_plus_cAA(A:", 30),
    ("linear_relu_sq_kernel", "def linear_relu_square_kernel", 60),
    ("linear_relu_sq_wrapper", "def linear_relu_square", 40),
    ("fused_linear_relu_sq", "class FusedLinearReLUSquareFunction", 30),
    ("transpose_copy_kernel", "def _transpose_copy_kernel", 30),
    ("transpose_copy_wrapper", "def transpose_copy(", 30),
    ("transpose_add_kernel", "def _transpose_add_kernel", 30),
    ("transpose_add_wrapper", "def transpose_add(", 30),
    ("ce_fwd_bwd", "def ce_fwd_bwd", 30),
    ("fused_softcap_ce", "class FusedSoftcappedCrossEntropy", 30),
)

ANCHORS_BY_FILE: dict[str, tuple[tuple[str, str, int], ...]] = {
    "train_gpt.py": TRAIN_GPT_ANCHORS,
    "triton_kernels.py": TRITON_ANCHORS,
}

CATEGORY_TO_LABELS: dict[str, tuple[str, ...]] = {
    "optimizer": ("param_table", "nor_muon", "muon_helper", "training_manager"),
    "schedule": ("training_stages", "training_schedule", "hyperparameters"),
    "kernel": (
        "linear_relu_sq_kernel", "linear_relu_sq_wrapper", "fused_linear_relu_sq",
        "ce_fwd_bwd", "fused_softcap_ce",
        "xxt_kernel", "xtx_kernel", "ba_plus_caa_kernel",
        "transpose_copy_kernel", "transpose_add_kernel",
    ),
    "architecture": ("gpt_init", "attention", "mlp"),
    "systems": ("main_train_loop", "training_manager"),
    "mixed": ("hyperparameters", "param_table", "training_stages"),
}


@dataclass(frozen=True)
class Section:
    label: str
    file: str
    start: int  # 1-indexed inclusive
    end: int    # 1-indexed inclusive


def find_sections(repo_root: Path, file: str = "train_gpt.py") -> list[Section]:
    path = repo_root / file
    if not path.exists():
        return []
    anchors = ANCHORS_BY_FILE.get(file, ())
    if not anchors:
        return []
    lines = path.read_text().splitlines()
    out: list[Section] = []
    for label, anchor, span in anchors:
        for i, line in enumerate(lines):
            if anchor in line:
                start = max(1, i + 1 - 4)
                end = min(len(lines), i + 1 + span)
                out.append(Section(label=label, file=file, start=start, end=end))
                break
    return out


def excerpt(repo_root: Path, file: str, start: int, end: int, *, max_chars: int = 8000) -> str:
    path = repo_root / file
    if not path.exists():
        return ""
    lines = path.read_text().splitlines()
    start = max(1, start)
    end = min(len(lines), end)
    chunk = "\n".join(f"{i:5d}| {line}" for i, line in enumerate(lines[start - 1 : end], start=start))
    return chunk[:max_chars]


def render_code_map(sections: list[Section]) -> str:
    """Compact map for knowledge/code_map.md. Stays in the LLM prompt."""
    head = "# Code map (line anchors)\n\n"
    head += "Use these to request a specific excerpt rather than the full file.\n\n"
    rows = [f"- `{s.label}`: {s.file}:{s.start}-{s.end}" for s in sections]
    return head + "\n".join(rows) + "\n"

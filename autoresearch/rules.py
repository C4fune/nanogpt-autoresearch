"""Hard rule gates. Patches that violate these are rejected before any GPU touches them.

Source: README "Rules" section + post-record-21 amendments.
We don't trust the planner LLM; we verify every patch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Substrings whose introduction (or modification touching them) is forbidden.
BANNED_INTRODUCED_SUBSTRINGS: tuple[str, ...] = (
    "torch._inductor.config",
    "coordinate_descent_tuning",
    "dynamic=False",  # don't let it switch to dynamic compile or vice versa subtly
)

# Anchors that mark protected regions in train_gpt.py.
# Any edit whose `old_string` contains one of these is rejected.
#
# What we protect = the underlying token stream and val constants:
#   - the data generator function and its callers
#   - the shard reader and shard class
#   - val_tokens count and the .bin file paths
#
# What we DELIBERATELY allow (so planner can do real research):
#   - get_bigram_hash function body (auxiliary signal computed from tokens; record history
#     shows it has been changed, e.g. 2026-01-31-BigramHashH2D, 2026-02-06_SparseBigramGradient).
#     Its CALLSITE inside distributed_data_generator is still protected via that anchor.
#   - TRAINING_STAGES, batch sizes, sequence lengths, attention windows, max_doc_len.
PROTECTED_ANCHORS: tuple[str, ...] = (
    "distributed_data_generator",
    "_load_data_shard",
    "_BIN_MAGIC",
    "class Shard",
    "val_tokens: int = 10485760",
    "fineweb_train_*.bin",
    "fineweb_val_*.bin",
    "val_files",
    "train_files",
)

# Pinned hyperparameter literal that MUST appear unchanged in the resulting file.
PROTECTED_LITERALS_REQUIRED: tuple[str, ...] = (
    "val_tokens: int = 10485760",
)

# torch.compile kwargs that exist in the upstream baseline.
# Anything else introduced into a torch.compile(...) call counts as an "extra flag" (rule 3).
ALLOWED_TORCH_COMPILE_KWARGS: frozenset[str] = frozenset({"dynamic", "fullgraph"})


@dataclass(frozen=True)
class RuleViolation:
    code: str
    message: str


def check_patch(
    edits: list[dict],
    *,
    repo_root: Path,
    editable_files: tuple[str, ...],
) -> RuleViolation | None:
    """Return None if patch passes, else a RuleViolation."""
    for i, edit in enumerate(edits):
        f = edit.get("file", "")
        if f not in editable_files:
            return RuleViolation("file_not_editable", f"edit[{i}] touches non-editable file: {f}")

        old = edit.get("old", "")
        new = edit.get("new", "")

        if not old.strip():
            return RuleViolation("empty_old_string", f"edit[{i}]: old must be non-empty")

        for anchor in PROTECTED_ANCHORS:
            if anchor in old:
                return RuleViolation(
                    "protected_region",
                    f"edit[{i}] {f}: old contains protected anchor '{anchor}' (data pipeline / val constants)",
                )
            if anchor in new and anchor not in old:
                return RuleViolation(
                    "protected_region_introduced",
                    f"edit[{i}] {f}: new introduces protected anchor '{anchor}'",
                )

        for banned in BANNED_INTRODUCED_SUBSTRINGS:
            # Reject any net introduction of the banned substring within this edit.
            if new.count(banned) > old.count(banned):
                return RuleViolation(
                    "banned_substring",
                    f"edit[{i}] {f}: introduces banned substring '{banned}'",
                )
            # Also reject the comment-uncomment trick: if `new` contains the substring on
            # a non-comment line and `old` only had it commented, the post-write check
            # would catch it, but doing it here saves a worktree round-trip.
            new_live = _live_count(new, banned)
            old_live = _live_count(old, banned)
            if new_live > old_live:
                return RuleViolation(
                    "banned_substring_uncommented",
                    f"edit[{i}] {f}: enables previously-commented '{banned}'",
                )

    return None


_TORCH_COMPILE_CALL = re.compile(r"torch\.compile\s*\(", re.MULTILINE)
_KWARG = re.compile(r"(\b\w+)\s*=")


def _live_count(blob: str, needle: str) -> int:
    """Count occurrences of `needle` on lines that aren't pure comments."""
    n = 0
    for line in blob.splitlines():
        if line.lstrip().startswith("#"):
            continue
        n += line.count(needle)
    return n


def check_resulting_file(path: Path) -> RuleViolation | None:
    """After patch application, sanity-check the patched file globally.

    Enforces:
      - No uncommented torch._inductor.config.* assignment.
      - No uncommented coordinate_descent_tuning enable.
      - Every torch.compile(...) call uses only ALLOWED_TORCH_COMPILE_KWARGS.
      - PROTECTED_LITERALS_REQUIRED must still appear verbatim (val_tokens, etc.).
    """
    try:
        text = path.read_text()
    except OSError as e:
        return RuleViolation("unreadable", f"{path}: {e}")

    # Required literals must still appear (only checked for train_gpt.py since that's
    # where they live; harmless to scan triton_kernels.py too).
    for required in PROTECTED_LITERALS_REQUIRED:
        if path.name == "train_gpt.py" and required not in text:
            return RuleViolation(
                "required_literal_missing",
                f"{path}: protected literal removed: {required!r}",
            )

    for line_no, line in enumerate(text.splitlines(), 1):
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "torch._inductor.config" in line and "=" in line:
            return RuleViolation(
                "inductor_flag_enabled",
                f"{path}:{line_no} sets a torch._inductor.config flag: {line.strip()}",
            )
        if "coordinate_descent_tuning" in line:
            return RuleViolation(
                "coord_descent_enabled",
                f"{path}:{line_no} mentions coordinate_descent_tuning: {line.strip()}",
            )

    # torch.compile(...) signature check.
    for match in _TORCH_COMPILE_CALL.finditer(text):
        call_args = _extract_call_args(text, match.end() - 1)
        if call_args is None:
            continue
        for kwarg in _KWARG.findall(call_args):
            if kwarg not in ALLOWED_TORCH_COMPILE_KWARGS:
                return RuleViolation(
                    "torch_compile_extra_kwarg",
                    f"{path}: torch.compile(...) uses disallowed kwarg '{kwarg}'. "
                    f"Allowed: {sorted(ALLOWED_TORCH_COMPILE_KWARGS)}",
                )

    return None


def _extract_call_args(text: str, open_paren_idx: int) -> str | None:
    """Given index of '(' in text, return everything between it and the matching ')'."""
    depth = 0
    i = open_paren_idx
    n = len(text)
    while i < n:
        ch = text[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[open_paren_idx + 1 : i]
        i += 1
    return None


def classify_patch(edits: list[dict], repo_root: Path) -> str:
    """Heuristic: 'systems' or 'ml'. Drives whether replication is required.

    A 'systems' patch only touches kernels, communication, dtype/layout for speed.
    An 'ml' patch touches the optimization or model semantics (loss, schedule, init,
    optimizer state, attention structure, etc.).

    Default to 'ml' when uncertain — replication is the safer default.
    """
    ml_signals = (
        "param_table",
        "TRAINING_STAGES",
        "num_scheduled_iterations",
        "num_extension_iterations",
        "cooldown_frac",
        "NorMuon",
        "Muon",
        "Adam",
        "softcap",
        "lr_mul",
        "wd_mul",
        "betas",
        "ReLUSqrdMLP",
        "attn_gate",
        "ve_gate",
        "skip_gate",
        "MTP",
        "value_emb",
        "bigram_vocab_size",
        "window_sizes",
        "head_dim",
        "num_heads",
        "num_layers",
    )
    pure_systems_signals = (
        "reduce_scatter",
        "all_gather",
        "all_reduce",
        "nccl",
        "process_group",
        "dist.barrier",
        "scatter_order",
        "work_order",
        "torch.compile",
    )

    text_blob = " ".join((e.get("old", "") + "\n" + e.get("new", "")) for e in edits)

    if any(sig in text_blob for sig in ml_signals):
        return "ml"
    if any(sig in text_blob for sig in pure_systems_signals):
        # Only pure-systems if NO ml signals appeared.
        return "systems"
    # Triton kernels, low-level ops without semantic ML changes.
    if all(e.get("file") == "triton_kernels.py" for e in edits):
        return "systems"
    return "ml"

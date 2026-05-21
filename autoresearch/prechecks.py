"""Cheap pre-checks that run BEFORE we burn 9 minutes of GPU time.

- py_compile: catches syntax errors instantly.
- import smoke test: catches dataclass / global / top-level errors in seconds.

We do NOT run torch.compile here; that's too expensive and only the real run
exercises it meaningfully.
"""

from __future__ import annotations

import os
import py_compile
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PrecheckResult:
    ok: bool
    stage: str       # "py_compile" | "import_smoke" | "passed"
    message: str = ""


def py_compile_check(path: Path) -> PrecheckResult:
    try:
        py_compile.compile(str(path), doraise=True)
        return PrecheckResult(True, "py_compile")
    except py_compile.PyCompileError as e:
        return PrecheckResult(False, "py_compile", str(e)[-2000:])


def import_smoke_check(repo_root: Path, file_rel: str = "train_gpt.py", timeout_s: int = 30) -> PrecheckResult:
    """Parse + AST-check via `python -c "ast.parse(...)"`. Avoids running torch."""
    cmd = [
        sys.executable,
        "-c",
        f"import ast,sys;src=open({str((repo_root / file_rel))!r}).read();ast.parse(src,filename={file_rel!r})",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return PrecheckResult(False, "import_smoke", f"timeout after {timeout_s}s: {file_rel}")
    if proc.returncode != 0:
        return PrecheckResult(False, "import_smoke", f"{file_rel}:\n{(proc.stderr or proc.stdout)[-2000:]}")
    return PrecheckResult(True, "import_smoke")


def run_all(repo_root: Path, files: tuple[str, ...]) -> PrecheckResult:
    for f in files:
        path = repo_root / f
        r = py_compile_check(path)
        if not r.ok:
            return r
    # AST-parse every editable file, not just train_gpt.py. A syntax error in
    # triton_kernels.py would otherwise slip through and burn ~10 min of GPU.
    for f in files:
        r = import_smoke_check(repo_root, f)
        if not r.ok:
            return r
    return PrecheckResult(True, "passed")

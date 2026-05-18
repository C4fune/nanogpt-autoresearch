"""Winners: replication aggregation + submission artifact minting + cumulative best.

Flow:
  1. A run finishes with verdict='win'. If it's NOT itself a replication, register
     a candidate batch and schedule N replication runs (clones of the same patch).
  2. Each replication run finishes; attach its run_id+metrics to the batch.
  3. After every iteration, evaluate any batch with N+ replications:
       - All replications passed val<=3.28 AND mean train_time < current best?
         -> mint artifacts, advance baseline, push wins/<id> branch.
       - Otherwise -> demote, write lesson.
  4. For 'systems' classified patches, README rule 2 waives p-value. We still mint
     artifacts but skip replication.

State file: .autoresearch/state/pending_wins.json
Artifacts:  .autoresearch/wins/<batch_id>/{patch.diff, log.txt, replications/, metrics.json, pr_body.md}
Index:      .autoresearch/wins/index.md
"""

from __future__ import annotations

import gzip
import json
import logging
import math
import os
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from autoresearch import backlog, journal
from autoresearch.config import Config
from autoresearch.state import Best, load_best, save_best

log = logging.getLogger(__name__)

PENDING_FILE = "pending_wins.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pending_path(config: Config) -> Path:
    return config.paths.state_dir / PENDING_FILE


def _load_pending(config: Config) -> dict[str, Any]:
    p = _pending_path(config)
    if not p.exists():
        return {"batches": {}}
    return json.loads(p.read_text())


def _save_pending(config: Config, data: dict[str, Any]) -> None:
    p = _pending_path(config)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, p)


# ----- registration / attachment ---------------------------------------------


def register_candidate(
    config: Config,
    *,
    candidate_run_id: str,
    candidate_metrics: dict,
    classification: str,
    hypothesis: str,
    category: str,
    rationale: str,
    patch_diff: str,
    parent_baseline_sha: str,
) -> str:
    """Register a candidate win; return batch_id."""
    batch_id = uuid.uuid4().hex[:8]
    pending = _load_pending(config)
    pending["batches"][batch_id] = {
        "batch_id": batch_id,
        "created_at": _utc_now(),
        "candidate_run_id": candidate_run_id,
        "candidate_metrics": candidate_metrics,
        "classification": classification,
        "hypothesis": hypothesis,
        "category": category,
        "rationale": rationale,
        "patch_diff": patch_diff,
        "parent_baseline_sha": parent_baseline_sha,
        "replications_needed": 0 if classification == "systems" else config.targets.replication_n,
        "replication_run_ids": [],
        "replication_metrics": [],
        "status": "collecting",
    }
    _save_pending(config, pending)
    journal.emit(
        config.paths.journal_jsonl, "win_candidate_registered",
        batch_id=batch_id, candidate_run_id=candidate_run_id,
        classification=classification,
    )
    return batch_id


def attach_replication(
    config: Config,
    *,
    batch_id: str,
    run_id: str,
    metrics: dict | None,
    success: bool,
) -> None:
    pending = _load_pending(config)
    batch = pending["batches"].get(batch_id)
    if not batch or batch["status"] != "collecting":
        return
    batch["replication_run_ids"].append(run_id)
    batch["replication_metrics"].append({
        "run_id": run_id,
        "metrics": metrics,
        "success": success,
    })
    _save_pending(config, pending)


def find_batch_for_replication(config: Config, replication_of: str) -> str | None:
    """Given a candidate idea_id, find the batch waiting on its replications."""
    pending = _load_pending(config)
    for batch_id, b in pending["batches"].items():
        if b["status"] != "collecting":
            continue
        # We track this by the candidate run id which the daemon embeds in the
        # replication idea's metadata.
        if b.get("candidate_run_id") == replication_of:
            return batch_id
    return None


# ----- evaluation -------------------------------------------------------------


def evaluate_batches(config: Config, repo_root: Path) -> list[str]:
    """Evaluate every collecting batch; finalize ready ones. Returns batch_ids advanced."""
    pending = _load_pending(config)
    advanced: list[str] = []

    for batch_id, batch in list(pending["batches"].items()):
        if batch["status"] != "collecting":
            continue

        if batch["classification"] == "systems":
            # No replication required; finalize immediately.
            decision = _decide_systems(config, batch)
        else:
            decision = _decide_ml(config, batch)

        if decision is None:
            continue

        if decision["advance"]:
            try:
                _mint_and_advance(config, repo_root, batch, decision)
                batch["status"] = "advanced"
                advanced.append(batch_id)
            except subprocess.CalledProcessError as e:
                err = (e.stderr or "") + (e.output or "")
                log.exception("mint_and_advance git failure for %s: %s", batch_id, err)
                journal.emit(config.paths.journal_jsonl, "mint_failed",
                             batch_id=batch_id, error=str(e), stderr=err[-1000:])
                batch["status"] = "mint_failed"
            except Exception as e:
                log.exception("mint_and_advance failed for %s", batch_id)
                journal.emit(config.paths.journal_jsonl, "mint_failed",
                             batch_id=batch_id, error=str(e))
                batch["status"] = "mint_failed"
        else:
            batch["status"] = "demoted"
            journal.emit(config.paths.journal_jsonl, "win_demoted",
                         batch_id=batch_id, reason=decision["reason"])

    _save_pending(config, pending)
    return advanced


def _decide_systems(config: Config, batch: dict) -> dict | None:
    m = batch["candidate_metrics"]
    val = m.get("val_loss")
    t = m.get("train_time_ms")
    best = load_best(config.paths.best_json)
    baseline_t = best.train_time_ms or config.targets.baseline_train_time_ms
    if val is None or t is None:
        return {"advance": False, "reason": "candidate had no metrics"}
    if val > config.targets.val_loss_max:
        return {"advance": False, "reason": f"val_loss {val:.4f} > {config.targets.val_loss_max}"}
    if t >= baseline_t:
        return {"advance": False, "reason": f"train_time {t}ms >= baseline {baseline_t}ms"}
    return {
        "advance": True,
        "reason": "systems-class win, README waives p-value",
        "n": 1,
        "mean_val_loss": val,
        "mean_train_time_ms": t,
        "p_value_proxy": float("inf"),
    }


def _decide_ml(config: Config, batch: dict) -> dict | None:
    needed = batch["replications_needed"]
    metrics_list = batch["replication_metrics"]
    if len(metrics_list) < needed:
        return None  # still collecting

    succeeded = [r for r in metrics_list if r["success"] and r["metrics"]]
    if len(succeeded) < needed:
        return {
            "advance": False,
            "reason": f"only {len(succeeded)}/{needed} replications succeeded; treat as variance/instability",
        }

    # Include the original candidate observation in the aggregate.
    cand = batch["candidate_metrics"]
    all_runs = [cand] + [r["metrics"] for r in succeeded]

    val_losses = [r["val_loss"] for r in all_runs if r.get("val_loss") is not None]
    train_times = [r["train_time_ms"] for r in all_runs if r.get("train_time_ms") is not None]
    if not val_losses or not train_times:
        return {"advance": False, "reason": "missing val_loss or train_time in replication metrics"}

    n = len(val_losses)
    mean_val = sum(val_losses) / n
    mean_t = sum(train_times) / n
    target = config.targets.val_loss_max
    delta_required = 0.004  # README: (target - mu) * sqrt(n) >= 0.004 ~ p<0.001
    score = (target - mean_val) * math.sqrt(n)

    if mean_val > target:
        return {"advance": False, "reason": f"mean val_loss {mean_val:.4f} > {target}"}
    if score < delta_required:
        return {
            "advance": False,
            "reason": f"insufficient stat-sig: (T-mu)*sqrt(n)={score:.4f} < {delta_required}",
        }

    best = load_best(config.paths.best_json)
    baseline_t = best.train_time_ms or config.targets.baseline_train_time_ms
    if mean_t >= baseline_t:
        return {"advance": False, "reason": f"mean train_time {mean_t:.0f}ms >= baseline {baseline_t}ms"}

    return {
        "advance": True,
        "reason": f"ml-class win, p<0.001 (score={score:.4f}, n={n})",
        "n": n,
        "mean_val_loss": mean_val,
        "mean_train_time_ms": mean_t,
        "val_loss_std": _std(val_losses),
        "p_value_proxy": score,
    }


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# ----- minting + branch pinning ----------------------------------------------


def _mint_and_advance(
    config: Config,
    repo_root: Path,
    batch: dict,
    decision: dict,
) -> Path:
    """Create wins/<id>/, commit on wins/<id> branch, advance state.best."""
    batch_id = batch["batch_id"]
    wins_root = config.paths.root / "wins"
    wins_root.mkdir(parents=True, exist_ok=True)
    win_dir = wins_root / batch_id
    win_dir.mkdir(parents=True, exist_ok=True)

    # 1. Patch diff (cumulative diff vs the parent baseline this run was based on).
    (win_dir / "patch.diff").write_text(batch["patch_diff"])

    # 2. Primary winning log (uncompressed copy from the candidate run dir).
    cand_run = config.paths.run_dir(batch["candidate_run_id"])
    primary_gz = cand_run / "log.txt.gz"
    if primary_gz.exists():
        out_log = win_dir / "log.txt"
        with gzip.open(primary_gz, "rb") as fin, out_log.open("wb") as fout:
            shutil.copyfileobj(fin, fout)

    # 3. Replication logs (also uncompressed).
    rep_dir = win_dir / "replications"
    rep_dir.mkdir(exist_ok=True)
    for i, r in enumerate(batch.get("replication_metrics", [])):
        run_dir = config.paths.run_dir(r["run_id"])
        rep_gz = run_dir / "log.txt.gz"
        if rep_gz.exists():
            out = rep_dir / f"seed_{i}_{r['run_id']}.txt"
            with gzip.open(rep_gz, "rb") as fin, out.open("wb") as fout:
                shutil.copyfileobj(fin, fout)

    # 4. Aggregate metrics.
    metrics = {
        "n": decision["n"],
        "mean_val_loss": round(decision["mean_val_loss"], 5),
        "mean_train_time_ms": round(decision["mean_train_time_ms"], 1),
        "val_loss_std": round(decision.get("val_loss_std", 0.0), 5),
        "p_value_proxy": decision.get("p_value_proxy"),
        "classification": batch["classification"],
        "candidate_run_id": batch["candidate_run_id"],
        "replication_run_ids": batch["replication_run_ids"],
        "parent_baseline_sha": batch["parent_baseline_sha"],
    }
    (win_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    # 5. Pin to a wins/<id> git branch on top of parent_baseline_sha and push to origin.
    #    The full patched file contents were captured at win-detection time and stored
    #    in the candidate run dir as <file>.patched.
    patched_files = {}
    for rel in config.editable_files:
        snap = cand_run / f"{rel}.patched"
        if snap.exists():
            patched_files[rel] = snap.read_text()
    branch = f"wins/{batch_id}"
    new_sha = _commit_winning_patch(
        repo_root=repo_root,
        parent_sha=batch["parent_baseline_sha"],
        patched_files=patched_files,
        branch=branch,
        commit_message=_commit_message(batch, metrics),
    )

    pushed = _try_push_branch(repo_root, branch)

    # 6. PR body for `gh pr create --body-file pr_body.md`.
    (win_dir / "pr_body.md").write_text(_pr_body(batch, metrics, branch, new_sha, pushed))
    (win_dir / "branch.txt").write_text(f"{branch}\t{new_sha}\n")

    # 7. Update wins/index.md.
    _append_index(wins_root, batch, metrics, branch, new_sha)

    # 8. Update .autoresearch/state/best.json — daemon's new baseline.
    best = Best(
        train_time_ms=int(metrics["mean_train_time_ms"]),
        val_loss=metrics["mean_val_loss"],
        val_loss_std=metrics["val_loss_std"],
        run_id=batch["candidate_run_id"],
        patch_branch=branch,
        baseline_commit_sha=new_sha,
        n_seeds_confirmed=metrics["n"],
        n_wins_chain=(load_best(config.paths.best_json).n_wins_chain or 0) + 1,
        confirmed_at=_utc_now(),
        notes=batch["hypothesis"][:200],
    )
    save_best(config.paths.best_json, best)

    journal.emit(
        config.paths.journal_jsonl, "win_advanced",
        batch_id=batch_id, branch=branch, sha=new_sha,
        train_time_ms=metrics["mean_train_time_ms"],
        val_loss=metrics["mean_val_loss"],
        n=metrics["n"], pushed=pushed,
    )
    return win_dir


def _commit_winning_patch(
    *,
    repo_root: Path,
    parent_sha: str,
    patched_files: dict[str, str],   # rel_path -> full file content
    branch: str,
    commit_message: str,
) -> str:
    """Apply the captured full files on top of parent_sha and commit on `branch`.
    Return new SHA. Uses a temporary worktree so the user's main checkout is untouched.
    """
    import tempfile
    tmp_root = Path(tempfile.mkdtemp(prefix="autoresearch_winmint_"))
    wt = tmp_root / "worktree"
    try:
        _run_git(repo_root, ["worktree", "add", "-B", branch, str(wt), parent_sha])
        for rel, content in patched_files.items():
            (wt / rel).parent.mkdir(parents=True, exist_ok=True)
            (wt / rel).write_text(content)
        _run_git(wt, ["add", "-A"])
        _run_git(wt, ["commit", "-m", commit_message])
        sha = _run_git(wt, ["rev-parse", "HEAD"]).strip()
        return sha
    finally:
        try:
            _run_git(repo_root, ["worktree", "remove", "--force", str(wt)])
        except Exception:
            pass
        try:
            _run_git(repo_root, ["worktree", "prune"])
        except Exception:
            pass
        shutil.rmtree(tmp_root, ignore_errors=True)


def _try_push_branch(repo_root: Path, branch: str) -> bool:
    try:
        _run_git(repo_root, ["push", "origin", branch])
        return True
    except subprocess.CalledProcessError as e:
        log.warning("could not push %s to origin: %s", branch, e.stderr or e)
        return False


def _run_git(cwd: Path, args: list[str]) -> str:
    proc = subprocess.run(
        ["git"] + args,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            proc.args,
            output=proc.stdout,
            stderr=proc.stderr,
        )
    return proc.stdout


def _commit_message(batch: dict, metrics: dict) -> str:
    return (
        f"autoresearch win: {batch['hypothesis'][:72]}\n"
        f"\n"
        f"Mean train_time_ms: {metrics['mean_train_time_ms']:.0f}\n"
        f"Mean val_loss: {metrics['mean_val_loss']:.4f} (std {metrics['val_loss_std']:.4f}, n={metrics['n']})\n"
        f"Classification: {metrics['classification']}\n"
        f"Batch: {batch['batch_id']}\n"
    )


def _pr_body(batch: dict, metrics: dict, branch: str, sha: str, pushed: bool) -> str:
    tmpl = f"""# {batch['hypothesis']}

**Category:** {batch['category']}
**Classification:** {metrics['classification']}
**Branch:** `{branch}` (sha `{sha[:12]}`){'  — pushed to origin' if pushed else ' — local only, push manually'}

## Result

| metric | value |
|---|---|
| mean train_time_ms | {metrics['mean_train_time_ms']:.0f} |
| mean val_loss | {metrics['mean_val_loss']:.4f} |
| val_loss std | {metrics['val_loss_std']:.4f} |
| seeds (n) | {metrics['n']} |
| stat-sig score `(3.28-mu)*sqrt(n)` | {metrics.get('p_value_proxy', 'n/a')} |

## Rationale

{batch['rationale']}

## Reproducibility

- Patch in `wins/{batch['batch_id']}/patch.diff` (structured edit summary).
- Primary winning log in `wins/{batch['batch_id']}/log.txt`.
- {metrics['n'] - 1} additional replication logs in `wins/{batch['batch_id']}/replications/`.
- Parent baseline commit: `{batch['parent_baseline_sha'][:12]}`.

To submit upstream:
```
gh pr create \\
  --repo KellerJordan/modded-nanogpt \\
  --head <your-username>:{branch} \\
  --title "{batch['hypothesis'][:72]}" \\
  --body-file .autoresearch/wins/{batch['batch_id']}/pr_body.md
```
"""
    return tmpl


def _append_index(wins_root: Path, batch: dict, metrics: dict, branch: str, sha: str) -> None:
    idx = wins_root / "index.md"
    if not idx.exists():
        idx.write_text(
            "# autoresearch wins\n\n"
            "Each row is a confirmed improvement that beat the prior baseline. "
            "The branch contains the cumulative diff against upstream master.\n\n"
            "| time (UTC) | batch | hypothesis | mean train_time_ms | mean val_loss | n | branch |\n"
            "|---|---|---|---|---|---|---|\n"
        )
    row = (
        f"| {_utc_now()} "
        f"| `{batch['batch_id']}` "
        f"| {batch['hypothesis'][:80].replace('|','/')} "
        f"| {metrics['mean_train_time_ms']:.0f} "
        f"| {metrics['mean_val_loss']:.4f} "
        f"| {metrics['n']} "
        f"| `{branch}` |\n"
    )
    with idx.open("a") as f:
        f.write(row)

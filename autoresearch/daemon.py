"""Long-running loop. Built to run unattended for days.

Each iteration:
  1. Recover any in-flight crash from cursor.json.
  2. Refill backlog when low (planner LLM).
  3. Pop highest-priority idea.
  4. Create worktree FROM state.best.baseline_commit_sha (so wins compound).
  5. Apply patch (rule-gated). Snapshot the patched files into the run dir
     (used later if this attempt becomes a confirmed win).
  6. Pre-checks (py_compile + AST parse).
  7. Run training; capture metrics + gzip log.
  8. Distill 150-word summary.
  9. If candidate win and not itself a replication: register batch, schedule
     N replications. If this run was a replication: attach to batch.
 10. Evaluate any batches that have collected enough replications. Wins are
     minted into .autoresearch/wins/<id>/, committed on a wins/<id> branch
     (built on the parent baseline SHA), and best.json is advanced.
 11. Periodically compact lessons; prune old logs.

Everything is restart-safe: cursor.json + journal.jsonl + atomic state writes
mean a SIGKILL loses at most the in-flight attempt.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

from autoresearch import backlog, journal, winners
from autoresearch.agent import compactor, distiller, planner
from autoresearch.agent.llm import DryRunClient, LLMClient, default_client
from autoresearch.agent.mutator import MutatorError, PatchPlan, apply_plan, parse_plan_json
from autoresearch.calibration import assert_calibrated
from autoresearch.config import Config
from autoresearch.parser.log_parser import LogMetrics, compact_log_summary
from autoresearch.prechecks import run_all as run_prechecks
from autoresearch.rules import classify_patch
from autoresearch.runner import experiment, workspace
from autoresearch.state import (
    Cursor,
    clear_cursor,
    load_best,
    load_budget,
    load_cursor,
    save_budget,
    save_cursor,
)

log = logging.getLogger(__name__)


def run_daemon(
    config: Config,
    *,
    max_iterations: int | None = None,
    dry_run: bool = False,
) -> None:
    paths = config.paths
    paths.ensure()

    if not dry_run:
        # Refuse to start on uncalibrated hardware (or |deviation|>25%).
        assert_calibrated(config)

    llm: LLMClient = DryRunClient() if dry_run else default_client()

    _install_signal_handlers()
    _recover_in_flight(config)

    journal.emit(paths.journal_jsonl, "daemon_started", dry_run=dry_run)

    iteration = 0
    while True:
        if max_iterations is not None and iteration >= max_iterations:
            journal.emit(paths.journal_jsonl, "daemon_stopped", reason="max_iterations")
            return

        try:
            _maybe_refill_backlog(config, llm)
        except Exception as e:
            journal.emit(paths.journal_jsonl, "planner_error", error=str(e))
            log.exception("planner refill failed; sleeping before retry")
            time.sleep(15)
            continue

        idea = backlog.pop_next(paths.backlog_jsonl)
        if idea is None:
            journal.emit(paths.journal_jsonl, "backlog_empty")
            log.warning("backlog empty after refill attempt; sleeping 30s")
            time.sleep(30)
            continue

        _execute_iteration(config, llm, idea, dry_run=dry_run)

        # Aggregate any batches whose replications are now complete.
        try:
            advanced = winners.evaluate_batches(config, config.repo_root)
            if advanced:
                journal.emit(paths.journal_jsonl, "batches_advanced", batch_ids=advanced)
        except Exception as e:
            log.exception("winners.evaluate_batches failed")
            journal.emit(paths.journal_jsonl, "winners_error", error=str(e))

        iteration += 1

        budget = load_budget(paths.budget_json)
        if budget.runs_completed % config.compact_every_n_runs == 0 and budget.runs_completed > 0:
            try:
                compactor.compact_lessons(config=config, llm=llm)
                journal.emit(paths.journal_jsonl, "lessons_compacted")
            except Exception as e:
                journal.emit(paths.journal_jsonl, "compactor_error", error=str(e))

        experiment.prune_old_logs(config)


# ----- single iteration -------------------------------------------------------


def _execute_iteration(config: Config, llm: LLMClient, idea: dict, *, dry_run: bool) -> None:
    paths = config.paths
    run_id = _new_run_id()
    save_cursor(paths.cursor_json, Cursor(run_id=run_id, started_at=_utc_now(), phase="patching"))
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    journal.emit(paths.journal_jsonl, "iteration_start", run_id=run_id, idea_id=idea.get("id"),
                 hypothesis=idea.get("hypothesis"))

    plan = _idea_to_plan(idea)
    (run_dir / "idea.json").write_text(json.dumps(idea, default=str, indent=2))

    # Each new attempt branches from the cumulative-best baseline SHA, not HEAD.
    parent_sha = workspace.current_baseline_sha(config)
    (run_dir / "parent_baseline_sha.txt").write_text(parent_sha + "\n")

    wt = None
    patch_preview = ""
    metrics: LogMetrics | None = None
    metrics_dict: dict | None = None
    stderr_tail = ""
    verdict = "loss"
    error: str | None = None
    duration_s = 0.0

    try:
        wt = workspace.create(config, run_id, from_ref=parent_sha)
        patch_preview = apply_plan(plan, wt.path, config)
        (run_dir / "patch.diff").write_text(patch_preview)

        # Snapshot patched files NOW so we can use them later if this run
        # becomes a confirmed win (worktree is removed at end of iteration).
        for rel in config.editable_files:
            src = wt.path / rel
            if src.exists():
                (run_dir / f"{rel}.patched").write_text(src.read_text())

        save_cursor(paths.cursor_json, Cursor(run_id=run_id, started_at=_utc_now(), phase="prechecks"))
        pre = run_prechecks(wt.path, config.editable_files)
        if not pre.ok:
            verdict = "precheck_failed"
            error = f"{pre.stage}: {pre.message[-400:]}"
            stderr_tail = pre.message
            (run_dir / "stderr.tail.txt").write_text(stderr_tail or "")
        else:
            if dry_run:
                verdict = "dry_run"
            else:
                save_cursor(paths.cursor_json, Cursor(run_id=run_id, started_at=_utc_now(), phase="training"))
                t0 = time.monotonic()
                result = experiment.run_training(wt.path, config, run_dir)
                duration_s = result.duration_s or (time.monotonic() - t0)
                stderr_tail = result.stderr_tail
                if not result.success or not result.metrics:
                    verdict = "crash"
                    error = result.error
                    if stderr_tail:
                        (run_dir / "stderr.tail.txt").write_text(stderr_tail)
                else:
                    metrics = result.metrics
                    metrics_dict = _metrics_dict(metrics)
                    added, removed = plan.diff_size()
                    metrics_dict["diff_lines_added"] = added
                    metrics_dict["diff_lines_removed"] = removed
                    best_now = load_best(paths.best_json)
                    baseline_loss = best_now.val_loss or config.targets.baseline_val_loss
                    if metrics.final and baseline_loss is not None:
                        metrics_dict["loss_buffer_delta"] = round(
                            baseline_loss - metrics.final.val_loss, 5
                        )
                    (run_dir / "metrics.json").write_text(json.dumps(metrics_dict, indent=2))
                    verdict = _verdict_from_metrics(metrics, config)

    except MutatorError as e:
        verdict = "patch_rejected"
        error = str(e)
    except Exception as e:
        verdict = "crash"
        error = f"daemon exception: {e}"
        log.exception("iteration crashed")
        traceback.print_exc(file=sys.stderr)

    finally:
        (run_dir / "verdict").write_text(verdict + "\n")
        if wt is not None:
            try:
                workspace.remove(config, run_id)
            except Exception:
                pass

        try:
            distiller.distill_run(
                config=config,
                llm=llm,
                run_dir=run_dir,
                hypothesis=idea.get("hypothesis", ""),
                category=idea.get("category", "mixed"),
                rationale=idea.get("rationale", ""),
                verdict=verdict,
                metrics=metrics_dict,
                patch_preview=patch_preview,
                stderr_tail=stderr_tail,
            )
        except Exception as e:
            (run_dir / "summary.md").write_text(
                f"verdict={verdict}\nerror={error or ''}\nfallback (distiller failed: {e})\n"
            )

        # Tick budget counter (informational only).
        budget = load_budget(paths.budget_json)
        budget.runs_completed += 1
        if not dry_run and verdict not in ("patch_rejected", "precheck_failed"):
            budget.gpu_hours_used += duration_s / 3600.0
        save_budget(paths.budget_json, budget)

        # Wire into the winners pipeline.
        is_replication = "replication_of_run_id" in (idea.get("metadata") or {})
        if is_replication:
            batch_id = winners.find_batch_for_replication(
                config,
                replication_of=idea["metadata"]["replication_of_run_id"],
            )
            if batch_id:
                winners.attach_replication(
                    config,
                    batch_id=batch_id,
                    run_id=run_id,
                    metrics=metrics_dict,
                    success=(verdict in ("win", "loss") and metrics_dict is not None
                             and metrics_dict.get("val_loss") is not None
                             and metrics_dict.get("val_loss") <= config.targets.val_loss_max),
                )
        elif verdict == "win" and metrics_dict is not None:
            klass = classify_patch([e.to_dict() for e in plan.edits], config.repo_root)
            batch_id = winners.register_candidate(
                config,
                candidate_run_id=run_id,
                candidate_metrics=metrics_dict,
                classification=klass,
                hypothesis=idea.get("hypothesis", ""),
                category=idea.get("category", "mixed"),
                rationale=idea.get("rationale", ""),
                patch_diff=patch_preview,
                parent_baseline_sha=parent_sha,
            )
            if klass != "systems":
                # Schedule N replication runs that re-use the same edits.
                for i in range(config.targets.replication_n):
                    rep_idea = {
                        "id": f"replicate_{run_id}_{i}",
                        "hypothesis": f"REPLICATE seed#{i}: {idea.get('hypothesis', '')}",
                        "category": idea.get("category", "mixed"),
                        "rationale": "Replication for statistical confirmation per README rule 2.",
                        "priority": 0.99 - i * 0.01,
                        "tags": ["replication"] + list(idea.get("tags", [])),
                        "edits": idea.get("edits", []),
                        "metadata": {
                            "replication_of_run_id": run_id,
                            "replication_of_batch": batch_id,
                            "seed": i,
                        },
                    }
                    backlog.append(paths.backlog_jsonl, rep_idea)
                journal.emit(paths.journal_jsonl, "replication_scheduled",
                             of_run_id=run_id, batch_id=batch_id,
                             n=config.targets.replication_n)

        clear_cursor(paths.cursor_json)
        journal.emit(
            paths.journal_jsonl,
            "iteration_end",
            run_id=run_id,
            verdict=verdict,
            error=error,
            train_time_ms=metrics.final.train_time_ms if metrics and metrics.final else None,
            val_loss=metrics.final.val_loss if metrics and metrics.final else None,
            duration_s=duration_s,
            is_replication=is_replication,
        )


# ----- helpers ----------------------------------------------------------------


def _maybe_refill_backlog(config: Config, llm: LLMClient) -> None:
    added = planner.refill_backlog_if_needed(config, llm)
    if added:
        journal.emit(config.paths.journal_jsonl, "backlog_refilled", added=added)


def _idea_to_plan(idea: dict) -> PatchPlan:
    raw = json.dumps(idea, default=str)
    return parse_plan_json(raw)


def _verdict_from_metrics(metrics: LogMetrics, config: Config) -> str:
    if not metrics.final:
        return "crash"
    f = metrics.final
    if f.val_loss > config.targets.val_loss_max:
        return "invalid_loss"
    best = load_best(config.paths.best_json)
    baseline_ms = best.train_time_ms or config.targets.baseline_train_time_ms
    if f.train_time_ms < baseline_ms:
        return "win"
    return "loss"


def _metrics_dict(metrics: LogMetrics) -> dict:
    if not metrics or not metrics.final:
        return {}
    f = metrics.final
    return {
        "step": f.step,
        "total_steps": f.total_steps,
        "val_loss": f.val_loss,
        "train_time_ms": f.train_time_ms,
        "step_avg_ms": f.step_avg_ms,
        "peak_memory_mib": metrics.peak_memory_mib,
        "summary": compact_log_summary(metrics),
    }


def _new_run_id() -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _recover_in_flight(config: Config) -> None:
    cursor = load_cursor(config.paths.cursor_json)
    if not cursor.is_in_flight():
        return
    run_dir = config.paths.run_dir(cursor.run_id)
    if run_dir.exists():
        verdict_path = run_dir / "verdict"
        if not verdict_path.exists():
            verdict_path.write_text("crash\n")
        (run_dir / "stderr.tail.txt").write_text(
            f"recovered as crash: was in phase {cursor.phase} when daemon died\n"
        )
    try:
        workspace.remove(config, cursor.run_id)
    except Exception:
        pass
    journal.emit(config.paths.journal_jsonl, "crash_recovered", run_id=cursor.run_id, phase=cursor.phase)
    clear_cursor(config.paths.cursor_json)


def _install_signal_handlers() -> None:
    def graceful(signum, frame):
        log.info("Received signal %s; will exit after current iteration.", signum)
        sys.exit(0)
    signal.signal(signal.SIGTERM, graceful)
    signal.signal(signal.SIGINT, graceful)

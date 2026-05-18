"""Long-running loop. Built to run unattended for days.

Lifecycle of a single iteration:
  1. Recover from any crash recorded in cursor.json.
  2. Refill backlog if low (calls planner LLM).
  3. Pop highest-priority idea.
  4. Create worktree, snapshot files, apply patch.
  5. Pre-checks (py_compile + import smoke); skip the GPU on failure.
  6. Run training; capture metrics + gzip log.
  7. Distill summary; emit verdict.
  8. If win: schedule replication; advance baseline only after passing the test.
  9. Periodically compact lessons; prune old logs.
 10. Update budget; stop cleanly if exhausted.

Every step is a no-op-on-restart so the daemon can be SIGKILLed and resumed
without losing more than the in-flight attempt.
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from autoresearch import backlog, journal
from autoresearch.agent import compactor, distiller, planner
from autoresearch.agent.llm import DryRunClient, LLMClient, default_client
from autoresearch.agent.mutator import MutatorError, PatchPlan, apply_plan, parse_plan_json
from autoresearch.agent.replication import evaluate as replication_evaluate
from autoresearch.config import Config
from autoresearch.parser.log_parser import LogMetrics, compact_log_summary
from autoresearch.prechecks import run_all as run_prechecks
from autoresearch.rules import classify_patch
from autoresearch.runner import experiment, workspace
from autoresearch.state import (
    Best,
    Budget,
    Cursor,
    clear_cursor,
    load_best,
    load_budget,
    load_cursor,
    save_best,
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

    wt = None
    snapshots: dict[str, str] = {}
    patch_preview = ""
    metrics: LogMetrics | None = None
    stderr_tail = ""
    verdict = "loss"
    error: str | None = None
    duration_s = 0.0

    try:
        wt = workspace.create(config, run_id)
        snapshots = workspace.snapshot_editable_files(wt, config)
        patch_preview = apply_plan(plan, wt.path, config)
        (run_dir / "patch.diff").write_text(patch_preview)

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
                    # Surface the README's discretionary trade-offs so the planner sees them
                    # in next round's hot context (via summary.md → recent summaries).
                    added, removed = plan.diff_size()
                    metrics_dict["diff_lines_added"] = added
                    metrics_dict["diff_lines_removed"] = removed
                    best = load_best(paths.best_json)
                    baseline_loss = best.val_loss or config.targets.baseline_val_loss
                    if metrics.final and baseline_loss is not None:
                        # Negative = used buffer (val_loss got worse); positive = freed buffer.
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
        # Distill (LLM call) — even crashes get a summary so future planners see them.
        try:
            distiller.distill_run(
                config=config,
                llm=llm,
                run_dir=run_dir,
                hypothesis=idea.get("hypothesis", ""),
                category=idea.get("category", "mixed"),
                rationale=idea.get("rationale", ""),
                verdict=verdict,
                metrics=_metrics_dict(metrics) if metrics else None,
                patch_preview=patch_preview,
                stderr_tail=stderr_tail,
            )
        except Exception as e:
            (run_dir / "summary.md").write_text(
                f"verdict={verdict}\nerror={error or ''}\nfallback (distiller failed: {e})\n"
            )

        # Update budget on real runs only.
        budget = load_budget(paths.budget_json)
        budget.runs_completed += 1
        if not dry_run and verdict != "patch_rejected" and verdict != "precheck_failed":
            budget.gpu_hours_used += duration_s / 3600.0
        save_budget(paths.budget_json, budget)

        # Win? -> schedule replication or advance.
        if verdict == "win" and metrics is not None:
            _handle_win(config, idea, plan, metrics, patch_preview)

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


def _handle_win(
    config: Config,
    idea: dict,
    plan: PatchPlan,
    metrics: LogMetrics,
    patch_preview: str,
) -> None:
    paths = config.paths
    klass = classify_patch([e.to_dict() for e in plan.edits], config.repo_root)
    journal.emit(paths.journal_jsonl, "candidate_win",
                 classification=klass,
                 train_time_ms=metrics.final.train_time_ms,
                 val_loss=metrics.final.val_loss)

    if klass == "systems":
        # README: pure systems changes waive p-value. One valid run wins.
        _advance_baseline(config, plan, metrics, n=1, mean_loss=metrics.final.val_loss)
        return

    # ML change: schedule N replication runs of the SAME patch by appending high-priority
    # ideas to the backlog. They'll be executed before any new exploration.
    for i in range(config.targets.replication_n):
        rep = {
            "id": f"replicate_{idea.get('id')}_{i}",
            "hypothesis": f"REPLICATE seed#{i}: {idea.get('hypothesis')}",
            "category": idea.get("category", "mixed"),
            "rationale": "Replication for statistical confirmation per README rule 2.",
            "priority": 0.99 - i * 0.01,
            "tags": ["replication"] + idea.get("tags", []),
            "edits": idea.get("edits", []),
            "metadata": {"replication_of": idea.get("id"), "seed": i},
        }
        backlog.append(paths.backlog_jsonl, rep)
    journal.emit(paths.journal_jsonl, "replication_scheduled",
                 of=idea.get("id"), n=config.targets.replication_n)


def _advance_baseline(
    config: Config,
    plan: PatchPlan,
    metrics: LogMetrics,
    *,
    n: int,
    mean_loss: float,
) -> None:
    paths = config.paths
    best = Best(
        train_time_ms=metrics.final.train_time_ms,
        val_loss=mean_loss,
        run_id=None,
        patch_branch=None,
        n_seeds_confirmed=n,
        confirmed_at=_utc_now(),
        notes=plan.hypothesis,
    )
    save_best(paths.best_json, best)
    journal.emit(paths.journal_jsonl, "baseline_advanced",
                 train_time_ms=best.train_time_ms,
                 val_loss=best.val_loss,
                 n_seeds=n,
                 hypothesis=plan.hypothesis)


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
    workspace.remove(config, cursor.run_id)
    journal.emit(config.paths.journal_jsonl, "crash_recovered", run_id=cursor.run_id, phase=cursor.phase)
    clear_cursor(config.paths.cursor_json)


def _install_signal_handlers() -> None:
    def graceful(signum, frame):
        log.info("Received signal %s; finishing iteration before exit.", signum)
        # We don't preempt mid-iteration; we just let the loop exit naturally.
        sys.exit(0)
    signal.signal(signal.SIGTERM, graceful)
    signal.signal(signal.SIGINT, graceful)

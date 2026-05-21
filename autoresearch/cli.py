"""CLI: bootstrap, run, status, parse-log, propose-once.

Long-running command is `run` (the daemon). Everything else is a one-shot.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from autoresearch import backlog, journal
from autoresearch.bootstrap import bootstrap as do_bootstrap
from autoresearch.config import load_config
from autoresearch.daemon import run_daemon
from autoresearch.parser.log_parser import compact_log_summary, parse_log
from autoresearch.state import (
    load_best,
    load_budget,
)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(description="modded-nanogpt autoresearcher")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("bootstrap", help="Build code_map + record_index (run once at setup)")

    p_run = sub.add_parser("run", help="Start the long-running daemon loop")
    p_run.add_argument("--max-iterations", type=int, default=None)
    p_run.add_argument("--dry-run", action="store_true",
                       help="Skip GPU training; useful for shaking down the loop without spending GPU time")

    p_status = sub.add_parser("status", help="Print state summary")

    p_parse = sub.add_parser("parse-log", help="Parse one training log to compact summary")
    p_parse.add_argument("path")

    p_journal = sub.add_parser("journal", help="Tail recent journal events")
    p_journal.add_argument("-n", type=int, default=30)

    p_base = sub.add_parser("baseline", help="Record measured baseline from a training log")
    p_base.add_argument("--log", required=True, help="Path to logs/<uuid>.txt from a clean run")
    p_base.add_argument("--note", default="measured baseline (no-op patch)")

    p_cal = sub.add_parser("calibrate", help="Compare a clean training log to upstream record reference")
    p_cal.add_argument("--log", required=True, help="Path to logs/<uuid>.txt from an unmodified run")

    p_mon = sub.add_parser("monitor", help="Live-tail journal events (pretty-printed)")
    p_mon.add_argument("--since", type=int, default=20, help="Show last N events on start")
    p_mon.add_argument("--follow", action="store_true", help="Keep streaming new events (Ctrl+C to stop)")

    sub.add_parser("history", help="Summarize the run database (wins, categories, dedup hits)")

    p_tr = sub.add_parser("traces", help="Inspect LLM call traces (prompts + responses)")
    p_tr.add_argument("--last", type=int, default=10, help="Show last N traces (default 10)")
    p_tr.add_argument("--run-id", default=None, help="Show traces for this run only")
    p_tr.add_argument("--purpose", default=None,
                      help="Filter by purpose: planner | distill | compactor")
    p_tr.add_argument("--full", action="store_true",
                      help="Print full system + user + response bodies (default: heads only)")

    args = parser.parse_args(argv)
    config = load_config()

    if args.cmd == "bootstrap":
        stats = do_bootstrap(config)
        print(f"bootstrap complete: {stats}")
        print(f"  code_map:     {config.paths.code_map_md}")
        print(f"  record_index: {config.paths.record_index_jsonl}")
        return 0

    if args.cmd == "parse-log":
        m = parse_log(args.path)
        print(compact_log_summary(m))
        return 0

    if args.cmd == "status":
        return _cmd_status(config)

    if args.cmd == "journal":
        for ev in journal.tail(config.paths.journal_jsonl, n=args.n):
            print(json.dumps(ev))
        return 0

    if args.cmd == "baseline":
        from autoresearch.state import Best, save_best
        from autoresearch.runner.workspace import current_baseline_sha

        m = parse_log(args.log)
        if not m.final or m.final.val_loss != m.final.val_loss:
            print(f"ERROR: log {args.log} has no final val_loss; refusing to set baseline.")
            return 2

        # Record the upstream HEAD sha so subsequent worktrees branch from it.
        try:
            head_sha = current_baseline_sha(config)
        except Exception:
            head_sha = None

        best = Best(
            train_time_ms=m.final.train_time_ms,
            val_loss=m.final.val_loss,
            run_id="manual_baseline",
            patch_branch=None,
            baseline_commit_sha=head_sha,
            n_seeds_confirmed=1,
            n_wins_chain=0,
            confirmed_at=None,
            notes=args.note,
        )
        save_best(config.paths.best_json, best)
        print(f"baseline recorded: train_time={m.final.train_time_ms}ms val_loss={m.final.val_loss:.4f} parent_sha={head_sha[:12] if head_sha else 'n/a'}")
        return 0

    if args.cmd == "calibrate":
        from autoresearch.calibration import format_report, record_calibration
        cal = record_calibration(config, Path(args.log))
        print(format_report(cal))
        return 0

    if args.cmd == "monitor":
        return _cmd_monitor(config, since=args.since, follow=args.follow)

    if args.cmd == "history":
        return _cmd_history(config)

    if args.cmd == "traces":
        return _cmd_traces(
            config,
            last=args.last,
            run_id=args.run_id,
            purpose=args.purpose,
            full=args.full,
        )

    if args.cmd == "run":
        run_daemon(config, max_iterations=args.max_iterations, dry_run=args.dry_run)
        return 0

    return 1


def _cmd_status(config) -> int:
    paths = config.paths
    best = load_best(paths.best_json)
    budget = load_budget(paths.budget_json)
    bstats = backlog.stats(paths.backlog_jsonl) if paths.backlog_jsonl.exists() else {"total": 0, "remaining": 0, "consumed": 0}

    notes = (best.notes or "")[:60]
    print(f"repo:         {config.repo_root}")
    print(f"state root:   {paths.root}")
    print(f"baseline:     train_time_ms<={config.targets.baseline_train_time_ms}, val_loss<={config.targets.val_loss_max}")
    print(f"best:         time={best.train_time_ms}ms val={best.val_loss} n_seeds={best.n_seeds_confirmed} ({notes})")
    print(f"runs:         completed={budget.runs_completed} gpu_hours={budget.gpu_hours_used:.2f}")
    print(f"backlog:      total={bstats['total']} remaining={bstats['remaining']} consumed={bstats['consumed']}")

    runs = sorted(paths.runs_dir.iterdir(), reverse=True)[:10] if paths.runs_dir.exists() else []
    if runs:
        print("recent runs:")
        for d in runs:
            v = (d / "verdict").read_text().strip() if (d / "verdict").exists() else "?"
            print(f"  {d.name}  verdict={v}")
    return 0


# Glyphs are intentional here — `monitor` is a TTY tool and the symbols make
# event types instantly recognizable when scanning hundreds of lines.
_EVENT_GLYPH = {
    "iteration_start":         "▶ ",
    "iteration_end":           "■ ",
    "iteration_crash_caught":  "✖ ",
    "backlog_refilled":        "+ ",
    "backlog_empty":           "· ",
    "planner_no_ideas":        "? ",
    "planner_malformed_idea":  "? ",
    "planner_error":           "✖ ",
    "planner_dedup_summary":   "= ",
    "idea_dedup_rejected":     "= ",
    "win_candidate_registered":"★ ",
    "replication_scheduled":   "↻ ",
    "win_advanced":            "✔ ",
    "win_demoted":             "↓ ",
    "mint_failed":             "✖ ",
    "crash_recovered":         "↺ ",
    "lessons_compacted":       "✎ ",
    "health_alert":            "‼ ",
    "daemon_started":          "▷ ",
    "daemon_stopped":          "■ ",
    "run_db_error":            "✖ ",
}


def _cmd_monitor(config, *, since: int, follow: bool) -> int:
    """Live-tail the journal with one-line, scannable formatting."""
    import time as _time
    path = config.paths.journal_jsonl
    if not path.exists():
        print(f"(no journal at {path} yet — daemon hasn't started)")
        return 0
    lines = path.read_text(errors="replace").splitlines()
    for line in lines[-since:]:
        _print_event_line(line)
    if not follow:
        return 0
    # tail -f
    f = open(path, "r")
    f.seek(0, 2)
    try:
        while True:
            line = f.readline()
            if not line:
                _time.sleep(0.5)
                continue
            _print_event_line(line.rstrip("\n"))
    except KeyboardInterrupt:
        return 0


def _print_event_line(raw: str) -> None:
    if not raw.strip():
        return
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        print(raw)
        return
    ts = (ev.get("ts") or "")[:19].replace("T", " ")
    name = ev.get("event", "?")
    glyph = _EVENT_GLYPH.get(name, "· ")
    extras = {k: v for k, v in ev.items() if k not in ("ts", "event") and v not in (None, "", [])}
    if name == "iteration_end":
        verdict = extras.pop("verdict", "?")
        tt = extras.pop("train_time_ms", None)
        vl = extras.pop("val_loss", None)
        dur = extras.pop("duration_s", None)
        rid = extras.pop("run_id", "")
        head = f"{verdict:<14s}"
        if vl is not None and tt is not None:
            head += f" val={vl:.4f} t={tt}ms"
        if dur:
            head += f" dur={dur:.0f}s"
        if rid:
            head += f"  {rid}"
        print(f"{ts}  {glyph}{name:<26s} {head}")
        return
    if name == "iteration_start":
        rid = extras.pop("run_id", "")
        hyp = (extras.pop("hypothesis", "") or "")[:80]
        print(f"{ts}  {glyph}{name:<26s} {rid}  {hyp}")
        return
    inline = " ".join(f"{k}={v}" for k, v in list(extras.items())[:5])
    print(f"{ts}  {glyph}{name:<26s} {inline}")


def _cmd_history(config) -> int:
    """One-screen view of the long-horizon run database."""
    from autoresearch.run_db import RunDB
    db = RunDB.load(config.paths.run_db_jsonl)
    print(f"runs.jsonl:   {config.paths.run_db_jsonl}")
    print(f"total rows:   {len(db.records)}")
    if not db.records:
        print("(empty — no iterations recorded yet)")
        return 0
    wins = [r for r in db.records if r.verdict == "win" and not r.is_replication]
    print(f"candidate wins: {len(wins)}")
    print()
    print("per-category (non-replication attempts):")
    for cat, s in sorted(db.category_stats().items()):
        total = s.get("_total", 0)
        wins_c = s.get("win", 0)
        crashes = s.get("crash", 0)
        rejected = s.get("patch_rejected", 0) + s.get("precheck_failed", 0)
        print(f"  {cat:12s} n={total:>3} wins={wins_c:>2} crash={crashes:>2} rejected={rejected:>2}")
    sigs = db.failure_signatures(n=5)
    if sigs:
        print()
        print("top crash signatures:")
        for sig, c in sigs.items():
            print(f"  ×{c:<3} {sig}")
    print()
    print("last 10 attempts (newest first):")
    for r in db.recent(10):
        rid = r.run_id[:14]
        cat = (r.category or "?")[:10]
        ver = (r.verdict or "?")[:14]
        val = f"{r.val_loss:.4f}" if r.val_loss is not None else "    ?"
        tt = f"{r.train_time_ms}" if r.train_time_ms is not None else "    ?"
        marker = "↻" if r.is_replication else " "
        print(f"  {rid} {marker} {cat:<10} {ver:<14} val={val} t={tt}ms  {(r.hypothesis or '')[:60]}")
    return 0


def _cmd_traces(config, *, last: int, run_id: str | None, purpose: str | None, full: bool) -> int:
    """Print the most recent LLM traces. With --full, dump the whole prompt and
    response — otherwise heads only. With --run-id, narrow to that run's mirror.
    """
    if run_id:
        path = config.paths.runs_dir / run_id / "llm_calls.jsonl"
        if not path.exists():
            print(f"(no llm_calls.jsonl for run {run_id} — agent didn't make any LLM calls for it)")
            return 0
    else:
        path = config.paths.traces_jsonl
        if not path.exists():
            print(f"(no traces yet at {path} — daemon hasn't made any LLM calls)")
            return 0

    rows: list[dict] = []
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    if purpose:
        rows = [r for r in rows if r.get("purpose") == purpose]
    rows = rows[-last:]
    if not rows:
        print("(no traces matched the filter)")
        return 0

    for r in rows:
        ts = (r.get("ts") or "")[:19].replace("T", " ")
        head = (
            f"== {ts}  purpose={r.get('purpose','?')}  model={r.get('model','?')}  "
            f"dur={r.get('duration_s','?')}s  "
            f"chars(s/u/r)={r.get('system_chars','?')}/"
            f"{r.get('user_chars','?')}/{r.get('response_chars','?')}"
        )
        if r.get("run_id"):
            head += f"  run_id={r['run_id']}"
        if not r.get("ok", True):
            head += f"  ERROR={r.get('error','?')[:120]}"
        print(head)
        thinking = r.get("thinking", "")
        usage = r.get("usage", {}) or {}
        if usage:
            print(
                f"  tokens:       in={usage.get('input_tokens','?')} "
                f"out={usage.get('output_tokens','?')} "
                f"thinking_chars={r.get('thinking_chars',0)}"
            )
        if full:
            print("--- SYSTEM ---")
            print(r.get("system", ""))
            print("--- USER ---")
            print(r.get("user", ""))
            if thinking:
                print("--- THINKING ---")
                print(thinking)
            print("--- RESPONSE ---")
            print(r.get("response", ""))
        else:
            if thinking:
                think_lines = thinking.strip().splitlines()
                head = " ".join(think_lines[:3])[:280]
                print(f"  thinking:     {head}")
            print(f"  user head:    {(r.get('user','') or '').splitlines()[0][:200] if r.get('user') else ''}")
            resp_head = (r.get("response") or "").strip().splitlines()
            print(f"  resp head:    {(resp_head[0] if resp_head else '')[:200]}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

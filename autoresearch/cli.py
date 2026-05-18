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
        m = parse_log(args.log)
        if not m.final or m.final.val_loss != m.final.val_loss:
            print(f"ERROR: log {args.log} has no final val_loss; refusing to set baseline.")
            return 2
        best = Best(
            train_time_ms=m.final.train_time_ms,
            val_loss=m.final.val_loss,
            run_id="manual_baseline",
            patch_branch=None,
            n_seeds_confirmed=1,
            confirmed_at=None,
            notes=args.note,
        )
        save_best(config.paths.best_json, best)
        # Also override the static target so daemon goal-checking uses real hardware time.
        config.targets.baseline_train_time_ms = m.final.train_time_ms
        config.targets.baseline_val_loss = m.final.val_loss
        print(f"baseline recorded: train_time={m.final.train_time_ms}ms val_loss={m.final.val_loss:.4f}")
        return 0

    if args.cmd == "run":
        run_daemon(config, max_iterations=args.max_iterations, dry_run=args.dry_run)
        return 0

    return 1


def _cmd_status(config) -> int:
    paths = config.paths
    best = load_best(paths.best_json)
    budget = load_budget(paths.budget_json)
    bstats = backlog.stats(paths.backlog_jsonl) if paths.backlog_jsonl.exists() else {"total": 0, "remaining": 0, "consumed": 0}

    print(f"repo:         {config.repo_root}")
    print(f"state root:   {paths.root}")
    print(f"baseline:     train_time_ms<={config.targets.baseline_train_time_ms}, val_loss<={config.targets.val_loss_max}")
    print(f"best:         time={best.train_time_ms}ms val={best.val_loss} n_seeds={best.n_seeds_confirmed} ({best.notes[:60]})")
    print(f"runs:         completed={budget.runs_completed} gpu_hours={budget.gpu_hours_used:.2f}")
    print(f"backlog:      total={bstats['total']} remaining={bstats['remaining']} consumed={bstats['consumed']}")

    runs = sorted(paths.runs_dir.iterdir(), reverse=True)[:10] if paths.runs_dir.exists() else []
    if runs:
        print("recent runs:")
        for d in runs:
            v = (d / "verdict").read_text().strip() if (d / "verdict").exists() else "?"
            print(f"  {d.name}  verdict={v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

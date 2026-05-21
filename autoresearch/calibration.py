"""Hardware calibration. Runs unmodified train_gpt.py once on this box and compares
to a reference log from the upstream `records/` history. Required before the daemon
will trust its own "wins" — we need to know our hardware reproduces leaderboard timings.

Writes .autoresearch/state/calibration.json with:
  - measured_train_time_ms / measured_val_loss
  - reference_train_time_ms / reference_val_loss / reference_source (path)
  - deviation_pct (negative = faster than reference, positive = slower)
  - calibrated_at, calibration_run_id

The daemon refuses to start if calibration is missing or |deviation| > 25%.
A 5-25% deviation prints a warning but still proceeds.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from autoresearch.config import Config
from autoresearch.parser.log_parser import parse_log


@dataclass
class Calibration:
    measured_train_time_ms: int
    measured_val_loss: float
    measured_step_count: int
    reference_train_time_ms: int
    reference_val_loss: float
    reference_source: str
    deviation_pct: float
    calibrated_at: str
    calibration_log_path: str


WARN_DEVIATION_PCT = 5.0
# Default block threshold. The local records/ folder can lag the upstream
# train_gpt.py by several versions, which inflates measured deviation even on
# identical hardware. Override with AUTORESEARCH_CAL_BLOCK_PCT when the fork is
# significantly ahead of the records you have on disk.
BLOCK_DEVIATION_PCT = float(os.environ.get("AUTORESEARCH_CAL_BLOCK_PCT", "25.0"))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def select_reference(config: Config) -> tuple[int, float, Path] | None:
    """Pick the best (fastest valid) log within the most recent records/track_1_short/* folder.

    The current train_gpt.py IS that record (this is a fork of upstream master), so the
    fastest log in the latest record folder = the actual winning timing on the upstream
    hardware. We use that as the reference for calibrating our own hardware.
    """
    root = config.repo_root / "records" / "track_1_short"
    if not root.exists():
        return None

    folders = sorted([d for d in root.iterdir() if d.is_dir()], reverse=True)
    for d in folders:
        best: tuple[int, float, Path] | None = None
        for log in list(d.rglob("*.txt")) + list(d.rglob("*.log")):
            try:
                if log.stat().st_size > 8_000_000:
                    continue
            except OSError:
                continue
            try:
                m = parse_log(log)
            except OSError:
                continue
            if (
                m.final
                and m.final.train_time_ms
                and m.final.val_loss == m.final.val_loss  # not NaN
                and m.final.val_loss <= 3.28              # an actual passing run
            ):
                if best is None or m.final.train_time_ms < best[0]:
                    best = (m.final.train_time_ms, m.final.val_loss, log)
        if best is not None:
            return best
    return None


def record_calibration(config: Config, log_path: Path) -> Calibration:
    metrics = parse_log(log_path)
    if not metrics.final:
        raise RuntimeError(f"calibration log {log_path} has no final metrics")

    ref = select_reference(config)
    if ref is None:
        raise RuntimeError(
            "no reference log found in records/track_1_short — cannot calibrate"
        )
    ref_ms, ref_val, ref_path = ref

    meas_ms = metrics.final.train_time_ms
    deviation = (meas_ms - ref_ms) / ref_ms * 100.0

    cal = Calibration(
        measured_train_time_ms=meas_ms,
        measured_val_loss=metrics.final.val_loss,
        measured_step_count=metrics.final.step,
        reference_train_time_ms=ref_ms,
        reference_val_loss=ref_val,
        reference_source=str(ref_path.relative_to(config.repo_root)),
        deviation_pct=round(deviation, 3),
        calibrated_at=_utc_now(),
        calibration_log_path=str(log_path),
    )
    out = config.paths.state_dir / "calibration.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(cal), indent=2) + "\n")
    return cal


def load_calibration(config: Config) -> Calibration | None:
    path = config.paths.state_dir / "calibration.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return Calibration(**raw)


def assert_calibrated(config: Config) -> None:
    """Daemon entry guard. Raises if not calibrated or deviation too large."""
    if os.environ.get("AUTORESEARCH_SKIP_CALIBRATION") == "1":
        return
    cal = load_calibration(config)
    if cal is None:
        raise RuntimeError(
            "Hardware not calibrated. Run:  bash scripts/measure_baseline.sh\n"
            "Or set AUTORESEARCH_SKIP_CALIBRATION=1 to override (not recommended)."
        )
    if abs(cal.deviation_pct) > BLOCK_DEVIATION_PCT:
        raise RuntimeError(
            f"Hardware deviates {cal.deviation_pct:+.1f}% from reference "
            f"({cal.reference_train_time_ms}ms vs measured {cal.measured_train_time_ms}ms; "
            f"reference={cal.reference_source}).\n"
            f"  - If you actually have 8x H100s, this usually means the local records/ "
            f"folder lags the current train_gpt.py. Either update records/, or raise "
            f"the block threshold with AUTORESEARCH_CAL_BLOCK_PCT (e.g. 40).\n"
            f"  - Wins on mismatched hardware will not generalize to the leaderboard.\n"
            f"  - Last-resort override: AUTORESEARCH_SKIP_CALIBRATION=1."
        )


def format_report(cal: Calibration) -> str:
    if abs(cal.deviation_pct) <= WARN_DEVIATION_PCT:
        verdict = f"OK — within {WARN_DEVIATION_PCT:.0f}% of reference"
    elif abs(cal.deviation_pct) <= BLOCK_DEVIATION_PCT:
        verdict = (
            f"WARN — within {BLOCK_DEVIATION_PCT:.0f}% but >{WARN_DEVIATION_PCT:.0f}%; "
            "wins still trustworthy directionally"
        )
    else:
        verdict = f"BLOCK — >{BLOCK_DEVIATION_PCT:.0f}% off; daemon will refuse to start"
    return (
        f"Calibration\n"
        f"  measured: {cal.measured_train_time_ms}ms (val_loss={cal.measured_val_loss:.4f})\n"
        f"  reference: {cal.reference_train_time_ms}ms (val_loss={cal.reference_val_loss:.4f}) "
        f"[{cal.reference_source}]\n"
        f"  deviation: {cal.deviation_pct:+.2f}%\n"
        f"  verdict:  {verdict}"
    )

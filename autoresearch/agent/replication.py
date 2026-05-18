"""Replication scheduling for ML wins.

When an attempt beats best AND the patch is classified 'ml', we replicate it N times
and only advance the baseline if `(3.28 - mean) * sqrt(n) >= 0.004` per the README.

This module just decides; the daemon executes the reruns the same way it executes
ordinary backlog ideas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplicationVerdict:
    advance: bool
    mean_val_loss: float
    mean_train_time_ms: float
    n: int
    p_value_proxy: float  # The README's z-test rule: (3.28-mean)*sqrt(n) >= 0.004 ⇒ p<0.001
    reason: str


def evaluate(
    *,
    val_losses: list[float],
    train_times_ms: list[int],
    target_loss: float,
    baseline_train_time_ms: int,
    sigma: float = 0.0013,
    delta_required: float = 0.004,  # README: 3.09σ → p<0.001
) -> ReplicationVerdict:
    """Decide whether to advance the baseline, given replication results.

    The README test: `(3.28 - mu) * sqrt(n) >= 0.004` (sigma~0.0013).
    """
    n = len(val_losses)
    if n == 0:
        return ReplicationVerdict(False, 0.0, 0.0, 0, 0.0, "no replications")
    mean_loss = sum(val_losses) / n
    mean_time = sum(train_times_ms) / n
    score = (target_loss - mean_loss) * math.sqrt(n)
    p_value_proxy = score
    if mean_loss > target_loss:
        return ReplicationVerdict(False, mean_loss, mean_time, n, p_value_proxy,
                                  f"mean val_loss {mean_loss:.4f} > {target_loss}")
    if score < delta_required:
        return ReplicationVerdict(False, mean_loss, mean_time, n, p_value_proxy,
                                  f"insufficient stat-sig: (T-mu)*sqrt(n)={score:.4f} < {delta_required}")
    if mean_time >= baseline_train_time_ms:
        return ReplicationVerdict(False, mean_loss, mean_time, n, p_value_proxy,
                                  f"mean train_time {mean_time:.0f}ms >= baseline {baseline_train_time_ms}ms")
    return ReplicationVerdict(True, mean_loss, mean_time, n, p_value_proxy,
                              f"advance: mean={mean_loss:.4f}, time={mean_time:.0f}ms, n={n}, score={score:.4f}")

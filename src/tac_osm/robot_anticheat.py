"""Real-robot anti-cheat adapter around the source-owned causal window API.

Operates on original RealLoggedWindow objects produced by
real_windows_from_original_episode(). It does not download data, train,
assign causal effects, or certify source authenticity.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from .real_robot_transition_contract import RealLoggedWindow, assert_no_future_inputs
from .research_anticheat import (
    AuditLedger, Finding, audit_input_boundary, probe_future_invariance,
    fingerprint,
)

ROBOT_ALLOWED_ROOTS = (
    "past_and_current_measured_states",
    "strictly_prior_executed_actions",
    "current_measured_state",
    "current_logged_action",
    "currently_human_controlled",
)


def audit_robot_windows(
    windows: Iterable[RealLoggedWindow], *, run_id: str
) -> AuditLedger:
    """Check every observed decision cut, including deliberate future poisoning.

    Source-owned episode continuity and original frame indices are checked;
    training/checkpoint-level and other-variant channel checks are separate.
    Missing windows, unexpected fields, or tampered outcome-dependent inputs
    fail rather than silently reducing the denominator.
    """
    ledger = AuditLedger(run_id, ("source_window_structure", "input_boundary",
                                  "future_invariance"))
    count = 0
    violations: list[str] = []
    frames: dict[int, int] = {}
    first_digest = ""
    for row in windows:
        count += 1
        if not isinstance(row, RealLoggedWindow):
            violations.append(f"window[{count}]: not RealLoggedWindow")
            continue
        try:
            assert_no_future_inputs(row)
            if row.current_measured_state != row.observed_history_state[-1]:
                raise ValueError("current state differs from final observed state")
            if row.episode_index in frames and row.current_frame_index <= frames[row.episode_index]:
                raise ValueError("frame order repeats/reverses within episode")
            frames[row.episode_index] = row.current_frame_index
        except (ValueError, TypeError, AssertionError, IndexError) as exc:
            violations.append(f"window[{count}]: invalid causal window: {type(exc).__name__}")
            continue
        boundary = audit_input_boundary(row.model_inputs(), ROBOT_ALLOWED_ROOTS)
        if boundary.status != "PASS":
            violations.extend(f"window[{count}] {p}" for p in boundary.evidence["paths"])
        if not first_digest:
            first_digest = fingerprint(row.model_inputs())
        # Tamper only a newly observed target, not an input or an original
        # checkpoint. The modified object is a diagnostic negative control.
        tampered = replace(row, next_measured_state=(
            row.next_measured_state[0] + 1.0, *row.next_measured_state[1:]))
        poison = probe_future_invariance(row, tampered, lambda x: x.model_inputs(),
                                         lambda x: x.factual_targets())
        if poison.status != "PASS":
            violations.append(f"window[{count}]: {poison.reason}")

    source_ok = bool(count) and not any(
        "invalid causal window" in v or "not RealLoggedWindow" in v for v in violations)
    inputs_ok = bool(count) and not any(
        ("not in declared allowlist" in v or "forbidden field" in v or
         "uninspectable object" in v) for v in violations)
    future_ok = bool(count) and not any(
        ("changed" in v or "poison" in v or "mutation" in v) for v in violations)
    # Any violation is a blocker to all three, even if its precise class is
    # unknown: avoid a green partial result if a row could not be inspected.
    if violations:
        source_ok = inputs_ok = future_ok = False
    common = {"windows_checked": count, "episodes_checked": len(frames),
              "first_input_sha256": first_digest,
              "violations": violations[:50],
              "truncated_violations": max(0, len(violations) - 50)}
    for check, ok in (("source_window_structure", source_ok),
                      ("input_boundary", inputs_ok),
                      ("future_invariance", future_ok)):
        ledger.add(Finding(check, "PASS" if ok else "FAIL",
                           "window contract verified" if ok else "one or more windows rejected",
                           common))
    return ledger
